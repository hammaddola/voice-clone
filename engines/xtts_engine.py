"""XTTS-v2 (Coqui, maintained `coqui-tts` fork). Weights: Coqui Public Model License (non-commercial)."""
import os
from functools import cache
from pathlib import Path

import torch

from . import Engine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SPEAKER = "Claribel Dervla"  # built-in XTTS voice, used as the OpenVoice base
LOCAL_DIR = Path(__file__).resolve().parent.parent / "models" / "xtts_v2"  # optional manual download (see README)


def _local_ready():
    m = LOCAL_DIR / "model.pth"
    # shortcut: size check guards against a half-finished download; the official checkpoint is ~1.87 GB
    return m.exists() and m.stat().st_size > 1_800_000_000 and all(
        (LOCAL_DIR / f).exists() for f in ("config.json", "vocab.json", "speakers_xtts.pth"))


@cache
def load_model(model_dir: str | None = None):
    if model_dir is None and _local_ready():
        model_dir = str(LOCAL_DIR)
    if model_dir is None:
        from TTS.api import TTS
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE).synthesizer.tts_model
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    config = XttsConfig()
    config.load_json(str(Path(model_dir) / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
    return model.to(DEVICE)


class XTTSEngine(Engine):
    name = "XTTS-v2"
    lang_map = {"ur": "hi"}  # Urdu is spoken through Hindi mode (approximate)

    def __init__(self, model_dir=None):
        self.model_dir = model_dir

    @property
    def model(self):
        return load_model(self.model_dir)

    def max_chars(self, lang):
        return self.model.tokenizer.char_limits.get(self.lang_map.get(lang, lang).split("-")[0], 220) - 10

    def make_voice(self, ref_paths):
        gpt, spk = self.model.get_conditioning_latents(audio_path=[str(p) for p in ref_paths])
        return {"gpt": gpt.cpu(), "spk": spk.cpu()}

    def builtin_voice(self, name=BASE_SPEAKER):
        gpt, spk = self.model.speaker_manager.speakers[name].values()
        return {"gpt": gpt, "spk": spk}

    @torch.inference_mode()
    def tts(self, text, lang, voice, speed=1.0, temperature=0.7, repetition_penalty=10.0, **kw):
        out = self.model.inference(
            text, self.lang_map.get(lang, lang), voice["gpt"].to(DEVICE), voice["spk"].to(DEVICE),
            # UI sliders can send ints (10 not 10.0); transformers 5 rejects non-float penalties
            temperature=float(temperature), repetition_penalty=float(repetition_penalty), speed=float(speed),
            enable_text_splitting=False,
        )
        wav = out["wav"]
        return (wav.cpu().numpy() if torch.is_tensor(wav) else wav).astype("float32")  # XTTS outputs 24 kHz == SR


class XTTSUrduEngine(XTTSEngine):
    """A model fine-tuned by finetune/train_xtts_urdu.py; Urdu was trained under the [ar] language token."""
    name = "XTTS-v2 Urdu (fine-tuned)"
    lang_map = {"ur": "ar"}


def available():
    engines = [XTTSEngine()]
    urdu_dir = os.environ.get("XTTS_URDU_DIR")
    if urdu_dir and (Path(urdu_dir) / "config.json").exists():
        engines.append(XTTSUrduEngine(urdu_dir))
    return engines
