"""Speaker similarity (0-100%) from SpeechBrain ECAPA-TDNN embeddings (Apache-2.0)."""
from functools import cache
from pathlib import Path

import librosa
import numpy as np
import torch

EMB_SR = 16000


@cache
def _encoder():
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    return EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path(__file__).parent.parent / "models" / "ecapa"),
        local_strategy=LocalStrategy.COPY,  # Windows can't symlink without admin
        run_opts={"device": "cpu"},
    )


@torch.inference_mode()
def embedding(wav: np.ndarray, sr: int) -> torch.Tensor:
    x = torch.from_numpy(librosa.resample(wav, orig_sr=sr, target_sr=EMB_SR)).float()[None]
    return _encoder().encode_batch(x).squeeze()


def score(ref: np.ndarray, out: np.ndarray, sr: int) -> float:
    """Cosine similarity mapped to 0-100. Same speaker typically lands > 60-70%, different speakers < 40%."""
    cos = torch.nn.functional.cosine_similarity(embedding(ref, sr), embedding(out, sr), dim=0).item()
    return round(max(0.0, cos) * 100, 1)
