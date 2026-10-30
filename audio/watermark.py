"""Imperceptible AudioSeal watermark (Meta, MIT) carrying a fixed 16-bit app ID."""
from functools import cache

import librosa
import numpy as np
import torch

WM_SR = 16000  # AudioSeal works at 16 kHz and (since v0.2) does not resample for us
APP_ID = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1], dtype=torch.int32)  # "this app"
MAX_BIT_ERRORS = 2


@cache
def _models():
    from audioseal import AudioSeal
    return AudioSeal.load_generator("audioseal_wm_16bits").eval(), AudioSeal.load_detector("audioseal_detector_16bits").eval()


@torch.inference_mode()
def embed(wav: np.ndarray, sr: int) -> np.ndarray:
    """Compute the watermark at 16 kHz, upsample it, and add it to the full-band original."""
    gen, _ = _models()
    x = torch.from_numpy(librosa.resample(wav, orig_sr=sr, target_sr=WM_SR)).float()[None, None]
    wm = gen.get_watermark(x, message=APP_ID[None]).squeeze().numpy()
    wm = librosa.resample(wm, orig_sr=WM_SR, target_sr=sr)
    n = min(len(wm), len(wav))
    out = wav.copy()
    out[:n] += wm[:n]
    return np.clip(out, -1, 1).astype(np.float32)


@torch.inference_mode()
def detect(wav: np.ndarray, sr: int) -> dict:
    _, det = _models()
    x = torch.from_numpy(librosa.resample(wav, orig_sr=sr, target_sr=WM_SR)).float()[None, None]
    prob, msg = det.detect_watermark(x)
    prob = float(prob[0])
    bit_errors = int((msg[0].int() != APP_ID).sum())
    return {
        "probability": prob,
        "bit_errors": bit_errors,
        "is_this_app": prob > 0.5 and bit_errors <= MAX_BIT_ERRORS,
        "watermarked_by_someone": prob > 0.5,
    }
