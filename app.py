"""Gradio UI: Clone & Speak, Speech-to-Speech, Saved Voices, Watermark Check."""
import json
import re
import shutil
import time
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch

import consent
import ui_theme as ui
import voice_chat
from audio import watermark
from audio.preprocess import SR, load, normalize_loudness, preprocess
from engines import LANGUAGES, synthesize
from engines.openvoice_engine import OpenVoiceEngine
from engines.xtts_engine import DEVICE, XTTSUrduEngine, available
from eval import similarity

ROOT = Path(__file__).parent
VOICES, OUTPUTS = ROOT / "voices", ROOT / "outputs"
VOICES.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

OPENVOICE = OpenVoiceEngine()
ENGINES = {e.name: e for e in [*available(), OPENVOICE]}
_session_voices = {}  # (engine, clip hashes) -> voice, so re-generating with the same clips skips re-encoding


# ---------- reference handling ----------

def clip_paths(files, mic):
    paths = [f if isinstance(f, str) else f.name for f in (files or [])]
    if mic:
        paths.append(mic)
    if len(paths) > 3:
        raise gr.Error("Use at most 3 reference clips.")
    return paths


def clean_refs(paths, out_dir: Path):
    """Preprocess each clip into out_dir; return (processed paths, joined reference wav, warnings)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean, wavs, warns = [], [], []
    for i, p in enumerate(paths):
        wav, w = preprocess(p)
        warns += [f"Clip {i + 1}: {m}" for m in w]
        dst = out_dir / f"ref_{i}.wav"
        sf.write(dst, wav, SR)
        clean.append(dst)
        wavs.append(wav)
    return clean, np.concatenate(wavs), warns


def slug(name):
    return re.sub(r"[^\w-]+", "_", name.strip()).strip("_")[:60] or "voice"


def saved_names():
    return sorted(p.name for p in VOICES.iterdir() if (p / "meta.json").exists())


def load_profile(name, engine):
    """Return (voice, reference wav) for a saved profile, computing and caching this engine's latents on first use."""
    d = VOICES / name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if not set(meta["sha256"]) <= consent.logged_hashes():
        raise gr.Error(f"No consent record found for '{name}' in consent_log.csv; re-save the voice.")
    refs = sorted(d.glob("ref_*.wav"))
    cache = d / f"{slug(engine.name)}.pt"
    if cache.exists():
        voice = torch.load(cache, weights_only=True)
    else:
        voice = engine.make_voice(refs)
        torch.save(voice, cache)
    return voice, np.concatenate([load(r) for r in refs])


def resolve_voice(saved, files, mic, speaker, agreed, engine, work_dir):
    """Saved profile, or freshly uploaded clips behind the consent gate. Returns (voice, ref wav, warnings)."""
    if saved:
        return (*load_profile(saved, engine), [])
    paths = clip_paths(files, mic)
    try:
        hashes = consent.require(paths, speaker, agreed)
    except consent.ConsentError as e:
        raise gr.Error(str(e))
    refs, ref_wav, warns = clean_refs(paths, work_dir)
    key = (engine.name, tuple(hashes))
    if key not in _session_voices:
        _session_voices[key] = engine.make_voice(refs)
    return _session_voices[key], ref_wav, warns


def finish(wav, ref_wav, fmt, warns, engine_label):
    """Watermark, save, score."""
    wav = watermark.embed(normalize_loudness(wav), SR)
    out = OUTPUTS / f"{time.strftime('%Y%m%d-%H%M%S')}.{fmt.lower()}"
    sf.write(out, wav, SR, format=fmt.upper())
    score = similarity.score(ref_wav, wav, SR)
    return str(out), ui.result_card(score, engine_label, warns)


# ---------- tab callbacks ----------

def clone_and_speak(files, mic, speaker, agreed, saved, text, lang_label, engine_name, speed, temperature,
                    rep_pen, fmt, progress=gr.Progress()):
    if not text or not text.strip():
        raise gr.Error("Type some text to speak.")
    engine, lang = ENGINES[engine_name], LANGUAGES[lang_label]
    warns = []
    if lang == "ur" and not isinstance(engine, XTTSUrduEngine):
        if re.search(r"[؀-ۿ]", text):
            raise gr.Error("Urdu script (Nastaliq) isn't readable in Hindi mode. Type Roman Urdu or Devanagari, "
                           "or use a fine-tuned Urdu model (see FINETUNE_URDU.md).")
        warns.append("Urdu is approximate: spoken through XTTS Hindi mode.")
    progress(0, desc="Preparing voice")
    voice, ref_wav, w = resolve_voice(saved, files, mic, speaker, agreed, engine, OUTPUTS / "_refs")
    wav = synthesize(engine, text, lang, voice, speed=speed, progress=progress,
                     temperature=temperature, repetition_penalty=rep_pen)
    return finish(wav, ref_wav, fmt, warns + w, engine.name)


def speech_to_speech(source, files, mic, speaker, agreed, saved, fmt):
    if not source:
        raise gr.Error("Record or upload the line you want converted.")
    voice, ref_wav, warns = resolve_voice(saved, files, mic, speaker, agreed, OPENVOICE, OUTPUTS / "_refs")
    wav = OPENVOICE.convert(normalize_loudness(load(source)), voice)
    return finish(wav, ref_wav, fmt, warns, "OpenVoice v2")


def save_voice(name, files, mic, agreed):
    if not name or not name.strip():
        raise gr.Error("Give the voice a name (the speaker's name).")
    paths = clip_paths(files, mic)
    try:
        hashes = consent.require(paths, name, agreed)
    except consent.ConsentError as e:
        raise gr.Error(str(e))
    d = VOICES / slug(name)
    if d.exists():
        shutil.rmtree(d)
    _, _, warns = clean_refs(paths, d)
    (d / "meta.json").write_text(json.dumps({"name": name.strip(), "sha256": hashes,
                                            "created": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
    for engine in ENGINES.values():  # cache latents/embeddings now so generation starts instantly later
        load_profile(d.name, engine)
    msg = f'<p class="vc-hint" role="status">Saved <b>{d.name}</b>. It now appears under "Saved voice" in the other tabs.</p>'
    return msg + ui.notes(warns), *refresh_choices()


def delete_voice(name):
    if not name:
        raise gr.Error("Pick a saved voice to delete.")
    shutil.rmtree(VOICES / name, ignore_errors=True)
    return f'<p class="vc-hint" role="status">Deleted <b>{name}</b>.</p>', *refresh_choices()


def refresh_choices():
    names = saved_names()
    return (gr.update(choices=[""] + names, value=""), gr.update(choices=[""] + names, value=""),
            gr.update(choices=names, value=None))


def check_watermark(path):
    if not path:
        raise gr.Error("Upload an audio file.")
    return ui.watermark_card(watermark.detect(load(path, 16000), 16000), watermark.APP_ID.tolist())


# ---------- UI ----------

def reference_inputs(name_label="Speaker name"):
    files = gr.File(label="Reference clips · 1–3 clips, 6–30 s each", file_count="multiple", file_types=["audio"])
    mic = gr.Audio(label="Or record a clip", sources=["microphone"], type="filepath")
    speaker = gr.Textbox(label=name_label, placeholder="e.g. Hammad Khan")
    agreed = gr.Checkbox(label="I confirm this speaker gave permission to clone their voice.", elem_classes="vc-consent")
    return files, mic, speaker, agreed


# ---------- Live voice chat ----------

def chat_start(saved, persona, lang_label, user_name):
    if not saved:
        raise gr.Error("Pick a saved voice. Live chat uses Saved Voices, so consent is already on record.")
    voice, _ = load_profile(saved, ENGINES["XTTS-v2"])
    (VOICES / saved / "persona.txt").write_text(persona or "", encoding="utf-8")
    try:
        voice_chat.start(voice, speaker=saved, persona=persona, lang=LANGUAGES[lang_label], user_name=user_name)
    except voice_chat.ChatError as e:
        raise gr.Error(str(e))
    return chat_poll()


def chat_stop():
    voice_chat.stop()
    return chat_poll()


def chat_poll():
    return voice_chat.status_html(), voice_chat.transcript_html()


def load_persona(saved):
    p = VOICES / saved / "persona.txt" if saved else None
    return p.read_text(encoding="utf-8") if p and p.exists() else ""


with gr.Blocks(title="Voice Clone") as demo:
    gr.HTML(ui.header(DEVICE))

    with gr.Tab("Clone & Speak"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="vc-card"):
                gr.HTML(ui.step(1, "Voice"))
                saved1 = gr.Dropdown([""] + saved_names(), value="", label="Saved voice (skips upload and consent)")
                ref1 = reference_inputs()
            with gr.Column(scale=7):
                with gr.Column(elem_classes="vc-card"):
                    gr.HTML(ui.step(2, "Script"))
                    text = gr.Textbox(label="Text to speak", lines=6, max_length=5000,
                                      placeholder="Type what the voice should say. Long text is split into sentences automatically.")
                    with gr.Row():
                        lang = gr.Dropdown(list(LANGUAGES), value="English", label="Language")
                        engine = gr.Dropdown(list(ENGINES), value=next(iter(ENGINES)), label="Engine",
                                             info="XTTS-v2: best likeness · OpenVoice v2: faster voice switching")
                    with gr.Row():
                        speed = gr.Slider(0.5, 2.0, 1.0, step=0.05, label="Speed")
                        fmt1 = gr.Radio(["WAV", "MP3"], value="WAV", label="Format")
                    with gr.Accordion("Advanced (XTTS)", open=False):
                        temperature = gr.Slider(0.1, 1.0, 0.7, step=0.05, label="Temperature",
                                                info="Lower is steadier, higher is more expressive")
                        rep_pen = gr.Slider(1.0, 20.0, 10.0, step=0.5, label="Repetition penalty",
                                            info="Raise it if words repeat or babble")
                    go = gr.Button("Generate speech", variant="primary", size="lg")
                with gr.Column(elem_classes="vc-card"):
                    gr.HTML(ui.step(3, "Result"))
                    out1 = gr.Audio(label="Output", type="filepath", interactive=False)
                    info1 = gr.HTML(ui.EMPTY_RESULT)
        go.click(clone_and_speak, [*ref1, saved1, text, lang, engine, speed, temperature, rep_pen, fmt1], [out1, info1])

    with gr.Tab("Speech-to-Speech"):
        with gr.Row(equal_height=False):
            with gr.Column(elem_classes="vc-card"):
                gr.HTML(ui.step(1, "Your performance") +
                        '<p class="vc-hint">Say the line exactly how it should feel. Your timing, pauses and emotion are kept.</p>')
                source = gr.Audio(label="Your line", sources=["microphone", "upload"], type="filepath")
                gr.HTML(ui.step(2, "Target voice"))
                saved2 = gr.Dropdown([""] + saved_names(), value="", label="Saved voice")
                with gr.Accordion("Or upload target clips (consent required)", open=False):
                    ref2 = reference_inputs()
            with gr.Column(elem_classes="vc-card"):
                gr.HTML(ui.step(3, "Converted") + '<p class="vc-hint">OpenVoice v2 tone-color conversion.</p>')
                fmt2 = gr.Radio(["WAV", "MP3"], value="WAV", label="Format")
                go2 = gr.Button("Convert voice", variant="primary", size="lg")
                out2 = gr.Audio(label="Output", type="filepath", interactive=False)
                info2 = gr.HTML(ui.EMPTY_RESULT)
        go2.click(speech_to_speech, [source, *ref2, saved2, fmt2], [out2, info2])

    with gr.Tab("Live Voice Chat"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="vc-card"):
                gr.HTML(ui.step(1, "Who you're talking to"))
                saved3 = gr.Dropdown([""] + saved_names(), value="", label="Saved voice")
                persona = gr.Textbox(label="Personality and shared memories", lines=7,
                                     placeholder="How she talks, pet names, inside jokes, what's going on in her life... "
                                                 "Written by you; the AI improvises from this.")
                with gr.Row():
                    user_name = gr.Textbox(label="Your name", placeholder="e.g. Hammad")
                    lang3 = gr.Dropdown(list(LANGUAGES), value="English", label="Conversation language")
                with gr.Row():
                    start3 = gr.Button("Start talking", variant="primary", size="lg")
                    stop3 = gr.Button("Stop", variant="secondary", size="lg")
                gr.HTML('<p class="vc-hint">Hands-free: just speak, then pause. Replies are written by Claude and spoken '
                        'in the cloned voice, so it sounds like her but it is an AI, not her. Wear headphones so it doesn\'t '
                        'hear itself. Needs <code>ANTHROPIC_API_KEY</code>.</p>')
            with gr.Column(scale=7, elem_classes="vc-card"):
                gr.HTML(ui.step(2, "Conversation"))
                chat_status = gr.HTML(voice_chat.status_html())
                chat_log = gr.HTML(voice_chat.transcript_html())
        saved3.change(load_persona, [saved3], [persona])
        start3.click(chat_start, [saved3, persona, lang3, user_name], [chat_status, chat_log])
        stop3.click(chat_stop, None, [chat_status, chat_log])
        gr.Timer(1.0).tick(chat_poll, None, [chat_status, chat_log], concurrency_limit=None, show_progress="hidden")

    with gr.Tab("Saved Voices"):
        with gr.Row(equal_height=False):
            with gr.Column(elem_classes="vc-card"):
                gr.HTML(ui.step("+", "New voice profile"))
                v_files, v_mic, v_name, v_agreed = reference_inputs("Speaker name (also the profile name)")
                save_btn = gr.Button("Save voice profile", variant="primary", size="lg")
                gr.HTML('<p class="vc-hint">Saving pre-computes XTTS latents and the OpenVoice embedding, '
                        'so generation starts instantly next time.</p>')
            with gr.Column(elem_classes="vc-card"):
                gr.HTML(ui.step("≡", "Library"))
                existing = gr.Dropdown(saved_names(), label="Saved voices")
                del_btn = gr.Button("Delete selected", variant="stop")
                v_msg = gr.HTML()
        outs = [v_msg, saved1, saved2, existing]
        save_btn.click(save_voice, [v_name, v_files, v_mic, v_agreed], outs).then(
            lambda: gr.update(choices=[""] + saved_names()), None, saved3)
        del_btn.click(delete_voice, [existing], outs).then(
            lambda: gr.update(choices=[""] + saved_names(), value=""), None, saved3)

    with gr.Tab("Watermark Check"):
        with gr.Column(elem_classes="vc-card"):
            gr.HTML(ui.step(1, "Audio to check") +
                    '<p class="vc-hint">Upload any file. Re-encoded or trimmed audio works too.</p>')
            wm_in = gr.Audio(label="Audio file", sources=["upload"], type="filepath")
            wm_btn = gr.Button("Check for watermark", variant="primary", size="lg")
            wm_out = gr.HTML()
        wm_btn.click(check_watermark, [wm_in], [wm_out])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860,
                                                   theme=ui.THEME, css=ui.CSS, head=ui.HEAD)
