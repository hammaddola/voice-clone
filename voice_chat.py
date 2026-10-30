"""Hands-free voice chat: mic -> speech detection -> Whisper -> Claude (streamed) -> XTTS (streamed) -> speakers.

Replies are written by Claude and spoken in a saved, consented voice. Every played chunk is watermarked.
"""
import html
import os
import queue
import re
import threading
from pathlib import Path

import numpy as np
import torch

import ui_theme as ui
from audio import watermark
from engines.xtts_engine import DEVICE, XTTSEngine

MODEL = "claude-opus-5-5"
MIC_SR, OUT_SR = 16000, 24000
FRAME = 0.03                       # mic block, seconds
END_SILENCE, MAX_UTTERANCE = 0.8, 30.0
_LOCAL_WHISPER = Path(__file__).resolve().parent / "models" / "whisper-small"
WHISPER = os.environ.get("VC_WHISPER_MODEL") or (
    str(_LOCAL_WHISPER) if (_LOCAL_WHISPER / "model.bin").exists() else "small")
_SENTENCE = re.compile(r"(.+?[.!?।۔。！？])(\s+|$)", re.S)


class ChatError(Exception):
    pass


def system_prompt(speaker, user_name, persona, lang):
    user = user_name.strip() or "the user"
    script = ("They may speak Urdu or Hindi. Reply in everyday Hindustani written in Devanagari script, because a "
              "Hindi voice reads your words aloud." if lang == "ur" else
              "Reply in the language they speak to you in.")
    return f"""You are speaking out loud, as {speaker}, in a private voice call with {user}. {speaker} agreed to their \
voice being used this way, and {user} is far away from them and misses them.

Personality, way of speaking and shared memories, written by {user}:
<persona>
{persona.strip() or "(none given: be warm, caring and natural)"}
</persona>

This is a live phone-style conversation, and everything you write is spoken by a text-to-speech voice:
- Reply in one to three short, natural sentences, the way people actually talk on a call. Ask things back sometimes.
- Plain spoken words only: no lists, markdown, emoji, stage directions or sound effects.
- {script}

You are an AI voice companion modelled on {speaker}, not {speaker} in person. Don't make promises, plans or claims about \
real events on {speaker}'s behalf, and if {user} sincerely asks whether this is really {speaker}, say gently that you \
are the AI version."""


class Chat:
    def __init__(self, voice, speaker, persona, lang, user_name):
        import anthropic
        from faster_whisper import WhisperModel

        self.client = anthropic.Anthropic()
        self.whisper = WhisperModel(WHISPER, device="cuda" if DEVICE == "cuda" else "cpu",
                                    compute_type="float16" if DEVICE == "cuda" else "int8")
        self.xtts = XTTSEngine()
        self.voice, self.lang, self.speaker, self.user = voice, lang, speaker, user_name.strip() or "You"
        self.system = system_prompt(speaker, user_name, persona, lang)
        self.history, self.transcript = [], []
        self.status, self.detail = "Starting", ""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # --- listening ---
    def _calibrate(self, sd):
        rec = sd.rec(int(MIC_SR * 1.0), samplerate=MIC_SR, channels=1, dtype="float32")
        sd.wait()
        floor = float(np.sqrt(np.mean(rec ** 2)))
        self.threshold = max(0.012, floor * 3.5)  # speech must be clearly above this room's noise

    def _listen(self, sd):
        """Block until one utterance (speech followed by END_SILENCE of quiet) is captured."""
        frames, speaking, quiet = [], False, 0.0
        with sd.InputStream(samplerate=MIC_SR, channels=1, dtype="float32", blocksize=int(MIC_SR * FRAME)) as mic:
            while self.running:
                block, _ = mic.read(int(MIC_SR * FRAME))
                loud = float(np.sqrt(np.mean(block ** 2))) > self.threshold
                if loud and not speaking:
                    speaking, self.status = True, "Listening"
                if speaking:
                    frames.append(block[:, 0].copy())
                    quiet = 0.0 if loud else quiet + FRAME
                    if quiet >= END_SILENCE or len(frames) * FRAME >= MAX_UTTERANCE:
                        break
        audio = np.concatenate(frames) if frames else np.zeros(0, np.float32)
        return audio if len(audio) > MIC_SR * 0.4 else None  # ignore clicks and coughs

    # --- speaking ---
    def _speaker_loop(self, sentences, sd):
        lang = self.xtts.lang_map.get(self.lang, self.lang)
        with sd.OutputStream(samplerate=OUT_SR, channels=1, dtype="float32") as out:
            while (s := sentences.get()) is not None:
                self.status = "Speaking" if DEVICE == "cuda" else "Preparing her reply"
                chunks = []
                with torch.inference_mode():
                    for chunk in self.xtts.model.inference_stream(
                            s, lang, self.voice["gpt"].to(DEVICE), self.voice["spk"].to(DEVICE),
                            stream_chunk_size=20, temperature=0.7, enable_text_splitting=False):
                        if not self.running:
                            return
                        wav = watermark.embed(chunk.cpu().numpy().astype(np.float32), OUT_SR)
                        if DEVICE == "cuda":
                            out.write(wav[:, None])  # GPU generates faster than real time: play as it streams
                        else:
                            chunks.append(wav)
                if chunks:  # CPU is ~15x slower than real time; streaming would stutter, so play whole sentences
                    self.status = "Speaking"
                    out.write(np.concatenate(chunks)[:, None])

    def _reply(self, user_text, sd):
        self.history.append({"role": "user", "content": user_text})
        sentences = queue.Queue()
        talker = threading.Thread(target=self._speaker_loop, args=(sentences, sd), daemon=True)
        talker.start()
        said, buf = "", ""
        self.transcript.append([self.speaker, ""])
        self.status = "Thinking"
        with self.client.beta.messages.stream(
            model=MODEL, max_tokens=1024, system=self.system, messages=self.history,
            output_config={"effort": "low"},  # conversational turns: low effort keeps replies fast
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        ) as stream:
            for text in stream.text_stream:
                buf += text
                while m := _SENTENCE.match(buf):
                    sentences.put(m.group(1).strip())
                    said += m.group(0)
                    buf = buf[m.end():]
                    self.transcript[-1][1] = said.strip()
                if not self.running:
                    sentences.put(None)
                    return
            final = stream.get_final_message()
        if buf.strip():
            sentences.put(buf.strip())
            said += buf
        sentences.put(None)
        if final.stop_reason == "refusal":
            said = said or "(no reply)"
        self.transcript[-1][1] = said.strip()
        self.history.append({"role": "assistant", "content": final.content})  # full content, thinking blocks included
        talker.join()

    def _run(self):
        import sounddevice as sd
        try:
            self.status = "Calibrating"
            self.detail = "Stay quiet for a second while I measure the room."
            self._calibrate(sd)
            self.detail = ""
            while self.running:
                self.status = "Your turn"
                audio = self._listen(sd)
                if audio is None or not self.running:
                    continue
                self.status = "Transcribing"
                hint = {"zh-cn": "zh"}.get(self.lang, self.lang)
                segs, _ = self.whisper.transcribe(audio, language=hint, beam_size=1, vad_filter=True)
                text = " ".join(s.text.strip() for s in segs).strip()
                if not text:
                    continue
                self.transcript.append([self.user, text])
                self._reply(text, sd)
        except Exception as e:  # surface any failure in the UI instead of dying silently
            self.status, self.detail = "Error", f"{type(e).__name__}: {e}"
            self.running = False
        else:
            self.status = "Stopped"


_chat: Chat | None = None


def api_key_problem():
    """The one place that decides whether live chat can reach Claude; None when it can."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ("Live chat needs an Anthropic API key (ANTHROPIC_API_KEY). Get one at console.anthropic.com, "
                "run: set ANTHROPIC_API_KEY=sk-ant-... and restart the app.")
    return None


def start(voice, speaker, persona, lang, user_name):
    global _chat
    if problem := api_key_problem():
        raise ChatError(problem)
    stop()
    _chat = Chat(voice, speaker, persona, lang, user_name)


def stop():
    if _chat and _chat.running:
        _chat.running = False
        _chat.status = "Stopping"


_TONE = {"Your turn": ui.OK, "Listening": "#22D3EE", "Speaking": "#A78BFA", "Error": ui.BAD}


def status_html():
    if _chat is None:
        if problem := api_key_problem():
            return ui.notes([problem])
        return '<p class="vc-empty">Pick a voice and press <b>Start talking</b>.</p>'
    tone = _TONE.get(_chat.status, ui.WARN)
    detail = f'<span class="vc-hint">{html.escape(_chat.detail)}</span>' if _chat.detail else ""
    return (f'<div class="vc-pill" role="status" aria-live="polite" style="color:{tone};border-color:{tone}55">'
            f'<span class="vc-dot" style="background:{tone}"></span>{html.escape(_chat.status)}</div> {detail}')


def transcript_html():
    if not _chat or not _chat.transcript:
        return ""
    rows = []
    for who, text in _chat.transcript[-30:]:
        mine = who == _chat.user
        rows.append(f'<div style="display:flex;justify-content:{"flex-end" if mine else "flex-start"};margin:6px 0">'
                    f'<div style="max-width:78%;padding:10px 14px;border-radius:14px;'
                    f'background:{"#1F2A44" if mine else "rgba(124,58,237,.16)"};border:1px solid var(--vc-line)">'
                    f'<div class="vc-hint" style="font-size:11.5px">{html.escape(who)}</div>'
                    f'{html.escape(text) or "…"}</div></div>')
    return "".join(rows)


if __name__ == "__main__":
    m = list(_SENTENCE.finditer("Hi love. How was your day? I missed you! still typing"))
    assert [x.group(1) for x in m] == ["Hi love.", "How was your day?", "I missed you!"]
    p = system_prompt("Sara", "Hammad", "", "ur")
    assert "Devanagari" in p and "AI version" in p
    print("voice_chat self-check ok")
