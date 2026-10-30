"""Reference-clip cleanup: load -> mono -> resample -> denoise -> trim -> loudness-normalize."""
import numpy as np
import librosa
import noisereduce as nr
import pyloudnorm as pyln
import soundfile as sf

SR = 24000
MIN_SECONDS, MAX_SECONDS = 6.0, 30.0
MIN_SNR_DB = 15.0
TARGET_LUFS = -20.0


def load(path, sr=SR) -> np.ndarray:
    """Any format -> mono float32 at `sr`. soundfile covers wav/flac/ogg/mp3; librosa's fallback needs ffmpeg (m4a/aac/webm)."""
    try:
        wav, file_sr = sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    except Exception:
        wav, file_sr = librosa.load(path, sr=None, mono=True)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return wav.astype(np.float32)


def estimate_snr_db(wav, sr=SR) -> float:
    """Rough SNR: loud frames (speech) vs. quietest 10% of frames (noise floor)."""
    rms = librosa.feature.rms(y=wav, frame_length=int(0.05 * sr), hop_length=int(0.025 * sr))[0]
    rms = np.sort(rms[rms > 1e-8])
    if len(rms) < 10:
        return 0.0
    noise = rms[: max(1, len(rms) // 10)].mean()
    speech = rms[-max(1, len(rms) // 2):].mean()
    return float(20 * np.log10(speech / max(noise, 1e-8)))


def normalize_loudness(wav, sr=SR, lufs=TARGET_LUFS) -> np.ndarray:
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(wav)
    if np.isfinite(loud):
        wav = pyln.normalize.loudness(wav, loud, lufs)
    peak = np.abs(wav).max()
    return (wav / peak * 0.95 if peak > 0.95 else wav).astype(np.float32)


def preprocess(path, denoise=True, sr=SR):
    """Return (clean_wav, warnings)."""
    wav = load(path, sr)
    warnings = []
    snr = estimate_snr_db(wav, sr)
    if snr < MIN_SNR_DB:
        warnings.append(f"Clip is noisy (est. SNR {snr:.0f} dB < {MIN_SNR_DB:.0f} dB); cloning quality may suffer.")
    if denoise:
        wav = nr.reduce_noise(y=wav, sr=sr, stationary=False, prop_decrease=0.8)
    wav, _ = librosa.effects.trim(wav, top_db=35)
    dur = len(wav) / sr
    if dur < MIN_SECONDS:
        warnings.append(f"Clip is only {dur:.1f}s of speech; 6-30s recommended.")
    elif dur > MAX_SECONDS:
        warnings.append(f"Clip is {dur:.0f}s; only the first {MAX_SECONDS:.0f}s are used.")
        wav = wav[: int(MAX_SECONDS * sr)]
    return normalize_loudness(wav, sr), warnings
