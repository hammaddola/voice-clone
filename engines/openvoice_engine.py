"""OpenVoice v2 tone-color converter (MIT), via coqui-tts.

Same pipeline as upstream OpenVoice: a base TTS speaks the text, then the converter re-colors it to the target
voice. Upstream uses MeloTTS as the base; here the base is a built-in XTTS voice (one less heavy install).
Also used directly for speech-to-speech."""
from functools import cache

import librosa
import numpy as np
import torch

from . import SR, Engine
from .xtts_engine import DEVICE, XTTSEngine


@cache
def load_vc():
    from TTS.api import TTS
    return TTS("voice_conversion_models/multilingual/multi-dataset/openvoice_v2").to(DEVICE).voice_converter.vc_model


class OpenVoiceEngine(Engine):
    name = "OpenVoice v2"

    def __init__(self):
        self.base = XTTSEngine()

    def max_chars(self, lang):
        return self.base.max_chars(lang)

    def make_voice(self, ref_paths):
        return {"se": load_vc()._extract_target_se([str(p) for p in ref_paths]).cpu()}

    @torch.inference_mode()
    def convert(self, wav: np.ndarray, voice, sr=SR) -> np.ndarray:
        """Re-voice `wav` (mono float32 at `sr`) as the target, keeping its timing, prosody and emotion."""
        vc = load_vc()
        vc_sr = vc.config.audio.output_sample_rate
        src = torch.from_numpy(librosa.resample(wav, orig_sr=sr, target_sr=vc_sr))
        src_se, src_spec = vc.extract_se(src)
        out = vc.inference(src_spec, {"g_src": src_se, "g_tgt": voice["se"].to(DEVICE)})["model_outputs"][0, 0]
        return librosa.resample(out.cpu().float().numpy(), orig_sr=vc_sr, target_sr=SR)

    def tts(self, text, lang, voice, speed=1.0, **kw):
        base = self.base.tts(text, lang, self.base.builtin_voice(), speed=speed, **kw)
        return self.convert(base, voice)
