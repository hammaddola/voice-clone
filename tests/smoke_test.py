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
from eval import similarity

tmp = Path(tempfile.mkdtemp())
consent.LOG = tmp / "consent_log.csv"  # keep test entries out of the real log
no_progress = lambda *a, **k: None
t0 = time.time()

print("1) making a reference clip from built-in XTTS voice 'Damien Black'")
xtts = XTTSEngine()
ref = xtts.tts("Hello, this is a short reference recording. I am speaking clearly at a normal pace, "
               "so the model can learn the sound of my voice.", "en", xtts.builtin_voice("Damien Black"))
ref_path = tmp / "ref.wav"
sf.write(ref_path, ref, SR)
print(f"   {len(ref) / SR:.1f}s")

print("2) consent gate blocks generation without confirmation")
try:
    app.clone_and_speak([str(ref_path)], None, "Damien Black", False, "", "Hi.", "English", "XTTS-v2",
                        1.0, 0.7, 10.0, "WAV", progress=no_progress)
    raise AssertionError("generation was not blocked")
except gr.Error as e:
    assert "Consent" in str(e), e

print("3) Urdu script is rejected in Hindi mode")
try:
    app.clone_and_speak([str(ref_path)], None, "Damien Black", True, "", "آپ کیسے ہیں", list(app.LANGUAGES)[-1],
                        "XTTS-v2", 1.0, 0.7, 10.0, "WAV", progress=no_progress)
    raise AssertionError("Nastaliq not rejected")
except gr.Error as e:
    assert "Roman Urdu" in str(e), e

def check_output(out, report, started, min_sim=40):
    sim = float(re.search(r"<strong>([\d.]+)%</strong>", report).group(1))
    print(f"   {out} in {time.time() - started:.0f}s, similarity {sim}%")
    det = watermark.detect(load(out, 16000), 16000)
    print(f"   watermark: {det}")
    assert det["is_this_app"], "watermark not detected"
    assert sim > min_sim, "similarity implausibly low"


app.VOICES = tmp / "voices"  # keep test profiles out of the real library
app.VOICES.mkdir()
SPEAKER = "Damien Black (built-in test voice)"

print("4) XTTS-v2 -> WAV from uploaded clips, int sliders as the UI sends them, saved as a profile")
s = time.time()
out, report, picked = app.clone_and_speak([str(ref_path)], None, SPEAKER, True, "",
                                          "The weather is lovely today. Let's go for a walk in the park!", "English",
                                          "XTTS-v2", 1, 0.7, 10, "WAV", remember=True, progress=no_progress)
check_output(out, report, s)
profile = app.slug(SPEAKER)
assert app.saved_names() == [profile], app.saved_names()
assert "Saved as voice profile" in report and picked["value"] == profile
assert (app.VOICES / profile / "XTTS-v2.pt").exists(), "encoded voice not cached into the profile"

print("5) OpenVoice v2 -> MP3 from the saved profile (no upload, no consent box)")
LINE = "The weather is lovely today. Let's go for a walk in the park!"
s = time.time()
out, report, picked = app.clone_and_speak(None, None, "", False, profile, LINE, "English",
                                          "OpenVoice v2", 1.0, 0.7, 10.0, "MP3", progress=no_progress)
# OpenVoice's score varies run to run (its base voice is sampled), so test what it must do: move the base voice
# toward the target. Converted output has to be closer to the target than the unconverted base speaker is.
# The test speaker is male and the OpenVoice base speaker (Claribel Dervla) is female, so there is room to move.
check_output(out, report, s, min_sim=0)
base = xtts.tts(LINE, "en", xtts.builtin_voice())
target = load(app.VOICES / profile / "ref_0.wav")
before, after = similarity.score(target, base, SR), similarity.score(target, load(out), SR)
print(f"   base speaker vs target {before}% -> converted vs target {after}%")
assert after > before + 5, "OpenVoice conversion did not move the voice toward the target"
assert picked == {"__type__": "update"}, "using a saved voice must not touch the picker"

print("6) saving never destroys a profile")
persona = app.VOICES / profile / "persona.txt"
persona.write_text("Loves chai and old songs.", encoding="utf-8")
cache = app.VOICES / profile / "XTTS-v2.pt"
assert app.save_profile(SPEAKER, [str(ref_path)], True) == (profile, [])  # same clips again: no-op
assert persona.exists() and cache.exists(), "re-saving the same clips wiped the profile"
other = tmp / "other.wav"
sf.write(other, ref[: len(ref) * 2 // 3], SR)  # a genuinely different clip of the same speaker
try:
    app.save_profile(SPEAKER, [str(other)], True)
    raise AssertionError("different clips silently replaced an existing profile")
except app.ProfileExists:
    pass
app.save_profile(SPEAKER, [str(other)], True, replace=True)  # explicit replace (Saved Voices tab)
assert persona.read_text(encoding="utf-8") == "Loves chai and old songs.", "replace lost the persona"
print("   same clips: no-op · different clips: refused · explicit replace: persona kept")

print("7) pickers keep each tab's selection; library details tell the truth about consent")
ups = app.refresh_pickers(profile, "", "gone-voice", "gone-voice")
assert "value" not in ups[0] and "value" not in ups[1], "refresh overwrote another tab's selection"
assert ups[2]["value"] == "" and ups[3]["value"] is None, "deleted voice still selected"
audio, details = app.voice_details(profile)
assert audio and "Consent on record" in details, details
real_log = consent.LOG
consent.LOG = tmp / "empty_log.csv"
assert "No consent record" in app.voice_details(profile)[1], "claimed consent that isn't logged"
consent.LOG = real_log
print(f"   {details}")

print("8) Live chat: missing API key is reported before any saving; no voice gives guidance")
os.environ.pop("ANTHROPIC_API_KEY", None)
assert "API key" in app.voice_chat.status_html()
app.VOICES = tmp / "voices2"
app.VOICES.mkdir()
try:
    app.chat_start("", "", "English", "Tester", [str(ref_path)], None, SPEAKER, True)
    raise AssertionError("started without an API key")
except gr.Error as e:
    assert "ANTHROPIC_API_KEY" in e.message, e
assert app.saved_names() == [], "saved a profile before checking the API key"
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-used"
try:
    app.chat_start("", "", "English", "Tester", None, None, "", False)
    raise AssertionError("started with no voice")
except gr.Error as e:
    assert "saved voice" in e.message, e
os.environ.pop("ANTHROPIC_API_KEY")
print("   ok")

print("9) un-watermarked audio is not flagged")
det = watermark.detect(load(ref_path, 16000), 16000)
print(f"   {det}")
assert not det["is_this_app"]

assert consent.LOG.exists() and len(consent.LOG.read_text().splitlines()) == 3  # header + ref + other clip
print(f"\nALL PASSED in {time.time() - t0:.0f}s")
