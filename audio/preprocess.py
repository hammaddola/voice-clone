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


def _temporal_snr_db(wav, sr):
    """Loud frames vs. the quietest 10% of frames. Needs pauses: on pause-free audio the 'quiet' frames are speech."""
    rms = librosa.feature.rms(y=wav, frame_length=int(0.05 * sr), hop_length=int(0.025 * sr))[0]
    rms = np.sort(rms[rms > 1e-8])
    if len(rms) < 10:
        return 0.0
    noise = rms[: max(1, len(rms) // 10)].mean()
    speech = rms[-max(1, len(rms) // 2):].mean()
    return float(20 * np.log10(speech / max(noise, 1e-8)))


def _spectral_snr_db(wav, sr):
    """Within loud frames: speech is peaky across frequency (harmonics, formants) while noise is flat, so the median
    bin's power approximates the noise floor. Works without pauses; misses narrow-band noise such as mains hum."""
    S = np.abs(librosa.stft(wav, n_fft=1024, hop_length=256)) ** 2
    S = S[: int(8000 / (sr / 1024))]  # speech band; above 8 kHz is mostly empty and would drag the median down
    energy = S.sum(axis=0)
    loud = energy > np.percentile(energy, 50)
    if loud.sum() < 5:
        return 0.0
    noise = np.median(S[:, loud], axis=0) * S.shape[0]
    signal = np.maximum(energy[loud] - noise, 1e-12)
    return float(10 * np.log10(signal.sum() / max(noise.sum(), 1e-12)))


def estimate_snr_db(wav, sr=SR) -> float:
    """Flag a clip as noisy only when both estimators agree, so pause-free clean speech isn't a false alarm."""
    return max(_temporal_snr_db(wav, sr), _spectral_snr_db(wav, sr))


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


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.arange(SR * 8) / SR
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.4 * t)
    voiced = sum(np.sin(k * 2 * np.pi * np.cumsum(f0) / SR) / k for k in range(1, 12))
    continuous = 0.25 * voiced / np.abs(voiced).max() * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2)
    gated = continuous * (np.sin(2 * np.pi * 1.5 * t) > -0.3)  # same "speech" with word gaps
    cases = {
        "continuous, clean": (continuous + 0.003 * rng.standard_normal(len(t)), False),
        "with pauses, clean": (gated + 0.003 * rng.standard_normal(len(t)), False),
        "continuous, noisy": (continuous + 0.08 * rng.standard_normal(len(t)), True),
        "with pauses, noisy": (gated + 0.08 * rng.standard_normal(len(t)), True),
    }
    for name, (x, expect_noisy) in cases.items():
        snr = estimate_snr_db(x.astype(np.float32))
        print(f"{name:20s} SNR {snr:5.1f} dB -> {'noisy' if snr < MIN_SNR_DB else 'ok'}")
        assert (snr < MIN_SNR_DB) == expect_noisy, name
    print("preprocess self-check ok")
