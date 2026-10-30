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


def finish(wav, ref_wav, fmt, warns, extra=""):
    """Watermark, save, score."""
    wav = watermark.embed(normalize_loudness(wav), SR)
    out = OUTPUTS / f"{time.strftime('%Y%m%d-%H%M%S')}.{fmt.lower()}"
    sf.write(out, wav, SR, format=fmt.upper())
    score = similarity.score(ref_wav, wav, SR)
    report = f"**Speaker similarity:** {score}%  \n**Watermark:** embedded (AudioSeal)  \n{extra}"
    if warns:
        report += "\n\n⚠️ " + "  \n⚠️ ".join(warns)
    return str(out), report


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
    return finish(wav, ref_wav, fmt, warns + w, f"**Engine:** {engine.name} on {DEVICE}")


def speech_to_speech(source, files, mic, speaker, agreed, saved, fmt):
    if not source:
        raise gr.Error("Record or upload the line you want converted.")
    voice, ref_wav, warns = resolve_voice(saved, files, mic, speaker, agreed, OPENVOICE, OUTPUTS / "_refs")
    wav = OPENVOICE.convert(normalize_loudness(load(source)), voice)
    return finish(wav, ref_wav, fmt, warns, "**Engine:** OpenVoice v2 tone-color conversion")


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
    msg = f"Saved **{d.name}**." + ("\n\n⚠️ " + "  \n⚠️ ".join(warns) if warns else "")
    return msg, *refresh_choices()


def delete_voice(name):
    if name:
        shutil.rmtree(VOICES / name, ignore_errors=True)
    return f"Deleted **{name}**." if name else "Pick a voice.", *refresh_choices()


def refresh_choices():
    names = saved_names()
    return (gr.update(choices=[""] + names, value=""), gr.update(choices=[""] + names, value=""),
            gr.update(choices=names, value=None))


def check_watermark(path):
    if not path:
        raise gr.Error("Upload an audio file.")
    r = watermark.detect(load(path, 16000), 16000)
    verdict = ("✅ **Contains this app's watermark.**" if r["is_this_app"] else
               "⚠️ **Watermarked, but not with this app's ID.**" if r["watermarked_by_someone"] else
               "❌ **No watermark detected.**")
    return f"{verdict}\n\nDetection probability: {r['probability']:.2f} · ID bit errors: {r['bit_errors']}/16"


# ---------- UI ----------

def reference_inputs():
    with gr.Group():
        files = gr.File(label="Reference clips (1-3, 6-30 s each)", file_count="multiple", file_types=["audio"])
        mic = gr.Audio(label="...or record one", sources=["microphone"], type="filepath")
        speaker = gr.Textbox(label="Speaker name")
        agreed = gr.Checkbox(label="I confirm this speaker gave permission to clone their voice.")
    return files, mic, speaker, agreed


with gr.Blocks(title="Voice Clone") as demo:
    gr.Markdown("# Voice Clone\nZero-shot multilingual voice cloning. Every output carries an AudioSeal watermark. "
                "Only clone voices you have permission to use.")
    with gr.Tab("Clone & Speak"):
        with gr.Row():
            with gr.Column():
                saved1 = gr.Dropdown([""] + saved_names(), value="", label="Saved voice (optional, skips upload)")
                ref1 = reference_inputs()
            with gr.Column():
                text = gr.Textbox(label="Text", lines=6)
                with gr.Row():
                    lang = gr.Dropdown(list(LANGUAGES), value="English", label="Language")
                    engine = gr.Dropdown(list(ENGINES), value=next(iter(ENGINES)), label="Engine")
                speed = gr.Slider(0.5, 2.0, 1.0, step=0.05, label="Speed")
                with gr.Accordion("Advanced (XTTS)", open=False):
                    temperature = gr.Slider(0.1, 1.0, 0.7, step=0.05, label="Temperature (lower = steadier)")
                    rep_pen = gr.Slider(1.0, 20.0, 10.0, step=0.5, label="Repetition penalty")
                fmt1 = gr.Radio(["WAV", "MP3"], value="WAV", label="Format")
                go = gr.Button("Generate", variant="primary")
                out1 = gr.Audio(label="Output", type="filepath")
                info1 = gr.Markdown()
        go.click(clone_and_speak, [*ref1, saved1, text, lang, engine, speed, temperature, rep_pen, fmt1], [out1, info1])

    with gr.Tab("Speech-to-Speech"):
        gr.Markdown("Say a line yourself; it is re-voiced as the target speaker, keeping your timing and emotion.")
        with gr.Row():
            with gr.Column():
                source = gr.Audio(label="Your line", sources=["microphone", "upload"], type="filepath")
                saved2 = gr.Dropdown([""] + saved_names(), value="", label="Target: saved voice")
                ref2 = reference_inputs()
            with gr.Column():
                fmt2 = gr.Radio(["WAV", "MP3"], value="WAV", label="Format")
                go2 = gr.Button("Convert", variant="primary")
                out2 = gr.Audio(label="Output", type="filepath")
                info2 = gr.Markdown()
        go2.click(speech_to_speech, [source, *ref2, saved2, fmt2], [out2, info2])

    with gr.Tab("Saved Voices"):
        with gr.Row():
            with gr.Column():
                v_files, v_mic, v_name, v_agreed = reference_inputs()
                save_btn = gr.Button("Save voice", variant="primary")
            with gr.Column():
                existing = gr.Dropdown(saved_names(), label="Saved voices")
                del_btn = gr.Button("Delete selected", variant="stop")
                v_msg = gr.Markdown()
        save_btn.click(save_voice, [v_name, v_files, v_mic, v_agreed], [v_msg, saved1, saved2, existing])
        del_btn.click(delete_voice, [existing], [v_msg, saved1, saved2, existing])

    with gr.Tab("Watermark Check"):
        wm_in = gr.Audio(label="Audio to check", sources=["upload"], type="filepath")
        wm_btn = gr.Button("Check", variant="primary")
        wm_out = gr.Markdown()
        wm_btn.click(check_watermark, [wm_in], [wm_out])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860)
