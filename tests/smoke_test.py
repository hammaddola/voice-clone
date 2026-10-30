"""End-to-end: consent gate -> clone -> generate (XTTS + OpenVoice) -> watermark detectable -> similarity.

Uses a built-in XTTS voice as the "reference speaker", so no real person's voice is needed.
Run:  .venv\\Scripts\\python tests\\smoke_test.py
"""
import os
import re
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import soundfile as sf

import app
import consent
from audio import watermark
from audio.preprocess import SR, load
from engines.xtts_engine import XTTSEngine

tmp = Path(tempfile.mkdtemp())
consent.LOG = tmp / "consent_log.csv"  # keep test entries out of the real log
no_progress = lambda *a, **k: None
t0 = time.time()

print("1) making a reference clip from built-in XTTS voice 'Ana Florence'")
xtts = XTTSEngine()
ref = xtts.tts("Hello, this is a short reference recording. I am speaking clearly at a normal pace, "
               "so the model can learn the sound of my voice.", "en", xtts.builtin_voice("Ana Florence"))
ref_path = tmp / "ref.wav"
sf.write(ref_path, ref, SR)
print(f"   {len(ref) / SR:.1f}s")

print("2) consent gate blocks generation without confirmation")
try:
    app.clone_and_speak([str(ref_path)], None, "Ana Florence", False, "", "Hi.", "English", "XTTS-v2",
                        1.0, 0.7, 10.0, "WAV", progress=no_progress)
    raise AssertionError("generation was not blocked")
except gr.Error as e:
    assert "Consent" in str(e), e

print("3) Urdu script is rejected in Hindi mode")
try:
    app.clone_and_speak([str(ref_path)], None, "Ana Florence", True, "", "آپ کیسے ہیں", list(app.LANGUAGES)[-1],
                        "XTTS-v2", 1.0, 0.7, 10.0, "WAV", progress=no_progress)
    raise AssertionError("Nastaliq not rejected")
except gr.Error as e:
    assert "Roman Urdu" in str(e), e

for engine, fmt in [("XTTS-v2", "WAV"), ("OpenVoice v2", "MP3")]:
    print(f"4) {engine} -> {fmt}")
    s = time.time()
    out, report = app.clone_and_speak(
        [str(ref_path)], None, "Ana Florence (built-in test voice)", True, "",
        "The weather is lovely today. Let's go for a walk in the park!", "English", engine,
        1.0, 0.7, 10.0, fmt, progress=no_progress)
    sim = float(re.search(r"<strong>([\d.]+)%</strong>", report).group(1))
    print(f"   {out} in {time.time() - s:.0f}s, similarity {sim}%")
    det = watermark.detect(load(out, 16000), 16000)
    print(f"   watermark: {det}")
    assert det["is_this_app"], "watermark not detected"
    assert sim > 40, "similarity implausibly low"

print("5) un-watermarked audio is not flagged")
det = watermark.detect(load(ref_path, 16000), 16000)
print(f"   {det}")
assert not det["is_this_app"]

assert consent.LOG.exists() and len(consent.LOG.read_text().splitlines()) == 2  # header + one clip
print(f"\nALL PASSED in {time.time() - t0:.0f}s")
