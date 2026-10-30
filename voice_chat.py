"""Hands-free voice chat: mic -> speech detection -> Whisper -> Claude or Gemini (streamed) -> XTTS -> speakers.

Replies are written by the chosen AI and spoken in a saved, consented voice. Every played chunk is watermarked.
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
# Flash-Lite first: Google's fastest model, and live chat is latency-bound. Flash is the fallback when it's overloaded.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_MODELS = list(dict.fromkeys([GEMINI_MODEL, "gemini-3.5-flash-lite", "gemini-3.8-flash"]))
GEMINI_FAILOVER_CODES = (429, 500, 503, 504)  # rate-limited, overloaded or timed out: try the next model
MIC_SR, OUT_SR = 16000, 24000
FRAME = 0.03                       # mic block, seconds
END_SILENCE, MAX_UTTERANCE = 0.8, 30.0
_LOCAL_WHISPER = Path(__file__).resolve().parent / "models" / "whisper-small"
WHISPER = os.environ.get("VC_WHISPER_MODEL") or (
    str(_LOCAL_WHISPER) if (_LOCAL_WHISPER / "model.bin").exists() else "small")
_SENTENCE = re.compile(r"(.+?[.!?।۔。！？])(\s+|$)", re.S)


class ChatError(Exception):
    pass


def split_stream(deltas):
    """Turn a stream of text deltas into whole sentences as soon as each one ends."""
    buf = ""
    for text in deltas:
        buf += text
        while m := _SENTENCE.match(buf):
            yield m.group(1).strip()
            buf = buf[m.end():]
    if buf.strip():
        yield buf.strip()


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


# --- reply writers: each streams text deltas and records the turn in its own history format ---

def stream_claude(client, system, history, user_text):
    history.append({"role": "user", "content": user_text})
    with client.beta.messages.stream(
        model=MODEL, max_tokens=1024, system=system, messages=history,
        output_config={"effort": "low"},  # conversational turns: low effort keeps replies fast
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
    ) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()
    history.append({"role": "assistant", "content": final.content})  # full content, thinking blocks included


def stream_gemini(client, system, history, user_text):
    import httpx
    from google.genai import errors, types

    history.append({"role": "user", "parts": [{"text": user_text}]})
    config = types.GenerateContentConfig(system_instruction=system,
                                         thinking_config=types.ThinkingConfig(thinking_level="LOW"))
    reply = ""
    for i, model in enumerate(GEMINI_MODELS):
        try:
            for chunk in client.models.generate_content_stream(model=model, contents=history, config=config):
                text = chunk.text or ""
                reply += text
                yield text
            break
        except (errors.APIError, httpx.TimeoutException) as e:
            # busy or timed out before any words came back: try the next model; mid-reply or bad requests surface
            busy = isinstance(e, httpx.TimeoutException) or getattr(e, "code", None) in GEMINI_FAILOVER_CODES
            if reply or not busy or i == len(GEMINI_MODELS) - 1:
                raise
    history.append({"role": "model", "parts": [{"text": reply}]})


def gemini_client():
    from google import genai
    from google.genai import types
    # live chat: fail over to the next model fast instead of the SDK's long backoff on an overloaded one
    return genai.Client(http_options=types.HttpOptions(
        timeout=20_000, retry_options=types.HttpRetryOptions(attempts=1)))  # ms; one try, then our failover


PROVIDERS = {  # UI label -> (env vars holding the key, model, client factory, reply writer)
    "Claude": (("ANTHROPIC_API_KEY",), MODEL,
               lambda: __import__("anthropic").Anthropic(), stream_claude),
    "Gemini": (("GEMINI_API_KEY", "GOOGLE_API_KEY"), GEMINI_MODEL, lambda: gemini_client(), stream_gemini),
}


def configured_providers():
    return [p for p, (keys, *_) in PROVIDERS.items() if any(os.environ.get(k) for k in keys)]


class Chat:
    def __init__(self, voice, speaker, persona, lang, user_name, provider="Claude"):
        from faster_whisper import WhisperModel

        _, _, make_client, self.writer = PROVIDERS[provider]
        self.client = make_client()
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
                        wav = chunk.cpu().numpy().astype(np.float32)
                        if DEVICE == "cuda":  # GPU generates faster than real time: play as it streams
                            # shortcut: per-chunk marks are weaker than one per sentence; buffer ~2 s if GPU detection is low
                            out.write(watermark.embed(wav, OUT_SR)[:, None])
                        else:
                            chunks.append(wav)
                if chunks:  # CPU is ~15x slower than real time; streaming would stutter, so play whole sentences
                    self.status = "Speaking"
                    out.write(watermark.embed(np.concatenate(chunks), OUT_SR)[:, None])  # one mark per sentence

    def _reply(self, user_text, sd):
        sentences = queue.Queue()
        talker = threading.Thread(target=self._speaker_loop, args=(sentences, sd), daemon=True)
        talker.start()
        self.transcript.append([self.speaker, ""])
        self.status = "Thinking"
        said = ""
        for sentence in split_stream(self.writer(self.client, self.system, self.history, user_text)):
            sentences.put(sentence)
            said += sentence + " "
            self.transcript[-1][1] = said.strip()
            if not self.running:
                break
        sentences.put(None)
        self.transcript[-1][1] = said.strip() or "(no reply)"
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


_KEY_HELP = {
    "Claude": "an Anthropic API key from console.anthropic.com: set ANTHROPIC_API_KEY=sk-ant-...",
    "Gemini": "a Gemini API key from aistudio.google.com: set GEMINI_API_KEY=...",
}


def api_key_problem(provider=None):
    """The one place that decides whether live chat can reach an AI; None when it can.
    provider=None asks whether any provider is usable."""
    if provider is None:
        if configured_providers():
            return None
        return ("Live chat needs an API key for the AI that writes the replies. Either get "
                f"{_KEY_HELP['Claude']}, or get {_KEY_HELP['Gemini']}. Then restart the app.")
    if provider not in configured_providers():
        return f"{provider} needs {_KEY_HELP[provider]}, then restart the app."
    return None


def start(voice, speaker, persona, lang, user_name, provider="Claude"):
    global _chat
    if problem := api_key_problem(provider):
        raise ChatError(problem)
    stop()
    _chat = Chat(voice, speaker, persona, lang, user_name, provider)


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
    assert list(split_stream(["Hi lo", "ve. How ", "are you? Fine"])) == ["Hi love.", "How are you?", "Fine"]

    from types import SimpleNamespace as NS
    from contextlib import contextmanager

    # Gemini writer against a fake client: request shape, streamed text, history format
    seen = {}

    def fake_gemini_stream(model, contents, config):
        seen.update(model=model, n=len(contents), system=config.system_instruction,
                    level=config.thinking_config.thinking_level)
        return iter([NS(text="Hi there. "), NS(text=None), NS(text="Missed you!")])

    hist = []
    out = list(stream_gemini(NS(models=NS(generate_content_stream=fake_gemini_stream)), "SYS", hist, "hello"))
    assert "".join(out) == "Hi there. Missed you!" and seen["model"] == GEMINI_MODEL and seen["system"] == "SYS"
    assert str(seen["level"]).upper().endswith("LOW"), seen
    assert hist == [{"role": "user", "parts": [{"text": "hello"}]},
                    {"role": "model", "parts": [{"text": "Hi there. Missed you!"}]}]

    # overloaded first model (503) falls back to the next one; a 400 is not retried
    from google.genai import errors
    tried = []

    def busy_then_ok(model, contents, config):
        tried.append(model)
        if len(tried) == 1:
            raise errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})
        return iter([NS(text="Still here.")])

    hist = []
    assert list(stream_gemini(NS(models=NS(generate_content_stream=busy_then_ok)), "S", hist, "hi")) == ["Still here."]
    assert tried == GEMINI_MODELS[:2] and len(hist) == 2, tried

    def bad_request(model, contents, config):
        raise errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})

    try:
        list(stream_gemini(NS(models=NS(generate_content_stream=bad_request)), "S", [], "hi"))
        raise AssertionError("400 should not fall back")
    except errors.ClientError:
        pass

    # Claude writer against a fake client
    @contextmanager
    def fake_claude_stream(**kw):
        seen.update(kw)
        yield NS(text_stream=iter(["Hello ", "love."]), get_final_message=lambda: NS(content=["BLOCKS"]))

    hist = []
    out = list(stream_claude(NS(beta=NS(messages=NS(stream=fake_claude_stream))), "SYS", hist, "hi"))
    assert "".join(out) == "Hello love." and seen["model"] == MODEL and seen["fallbacks"] == "default"
    assert hist == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ["BLOCKS"]}]

    # key detection: Gemini accepts GEMINI_API_KEY or GOOGLE_API_KEY; messages name the right variable
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(k, None)
    assert configured_providers() == [] and "ANTHROPIC_API_KEY" in api_key_problem() and "GEMINI_API_KEY" in api_key_problem()
    os.environ["GOOGLE_API_KEY"] = "x"
    assert configured_providers() == ["Gemini"] and api_key_problem() is None and api_key_problem("Gemini") is None
    assert "ANTHROPIC_API_KEY" in api_key_problem("Claude")
    os.environ.pop("GOOGLE_API_KEY")
    print("voice_chat self-check ok")
