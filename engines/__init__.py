"""Swappable TTS engines sharing one interface, plus long-text synthesis (split -> per-sentence -> join)."""
import re

import numpy as np

SR = 24000  # every engine returns mono float32 at this rate

# XTTS-v2 language codes; "ur" is routed through Hindi (approximate).
LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Polish": "pl", "Turkish": "tr", "Russian": "ru", "Dutch": "nl",
    "Czech": "cs", "Arabic": "ar", "Chinese": "zh-cn", "Japanese": "ja", "Hungarian": "hu",
    "Korean": "ko", "Hindi": "hi",
    "Urdu (approximate, via Hindi; Roman Urdu or Devanagari)": "ur",
}


class Engine:
    """Interface. `voice` is whatever make_voice() returns; it is cached in Saved Voices via torch.save."""
    name = "base"

    def make_voice(self, ref_paths: list):
        raise NotImplementedError

    def tts(self, text: str, lang: str, voice, speed: float = 1.0, **kw) -> np.ndarray:
        raise NotImplementedError

    def max_chars(self, lang: str) -> int:
        return 220


_SENT_END = re.compile(r"(?<=[.!?।۔。！？])\s+|(?<=[。！？])")


def split_sentences(text: str, max_chars: int = 220) -> list:
    """Split on sentence ends (incl. Hindi danda, Urdu full stop, CJK), then break over-long sentences at commas/spaces."""
    out = []
    for s in (p.strip() for p in _SENT_END.split(text.strip())):
        while len(s) > max_chars:
            cut = max(s.rfind(",", 0, max_chars), s.rfind(" ", 0, max_chars))
            cut = cut if cut > 0 else max_chars
            out.append(s[: cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            out.append(s)
    return out


def join(chunks: list, pause_s: float = 0.25, fade_s: float = 0.015) -> np.ndarray:
    """Fade each chunk in/out (crossfade through silence, so no clicks) and join with a short natural pause."""
    fade = int(fade_s * SR)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    parts = []
    for c in chunks:
        c = np.asarray(c, dtype=np.float32).copy()
        if len(c) > 2 * fade:
            c[:fade] *= ramp
            c[-fade:] *= ramp[::-1]
        parts.append(c)
    gap = np.zeros(int(pause_s * SR), dtype=np.float32)
    out = [parts[0]] if parts else []
    for p in parts[1:]:
        out += [gap, p]
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def synthesize(engine: Engine, text: str, lang: str, voice, speed=1.0, progress=None, **kw) -> np.ndarray:
    sentences = split_sentences(text, engine.max_chars(lang))
    if not sentences:
        raise ValueError("Text is empty.")
    chunks = []
    for i, s in enumerate(sentences):
        if progress:
            progress((i, len(sentences)), desc=f"Sentence {i + 1}/{len(sentences)}")
        chunks.append(engine.tts(s, lang, voice, speed=speed, **kw))
    return join(chunks)


if __name__ == "__main__":
    assert split_sentences("Hi there. How are you? Fine!") == ["Hi there.", "How are you?", "Fine!"]
    assert split_sentences("यह एक है। दूसरा।") == ["यह एक है।", "दूसरा।"]
    long = split_sentences("word " * 100)
    assert all(len(s) <= 220 for s in long) and " ".join(long).split() == ["word"] * 100
    j = join([np.ones(SR), np.ones(SR)])
    assert len(j) == 2 * SR + int(0.25 * SR) and j[0] == 0 and j[-1] == 0
    print("engines self-check ok")
