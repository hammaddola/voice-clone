"""Turn a training run into a folder the app can load (set XTTS_URDU_DIR to it).

    python finetune/export_model.py --run runs/xtts_ur/xtts_urdu-<date> --base runs/xtts_ur/base --out models/xtts_urdu
"""
import argparse
import shutil
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="trainer output folder containing best_model.pth")
    ap.add_argument("--base", required=True, type=Path, help="folder with the base config.json/vocab.json/speakers_xtts.pth")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ckpt = args.run / "best_model.pth"
    if not ckpt.exists():
        ckpt = max(args.run.glob("checkpoint_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    args.out.mkdir(parents=True, exist_ok=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)  # our own trainer checkpoint
    # weights only (drops optimizer state); Xtts.load_checkpoint strips the "xtts." prefix and DVAE keys itself
    torch.save({"model": state["model"]}, args.out / "model.pth")
    for name in ["config.json", "vocab.json", "speakers_xtts.pth"]:
        shutil.copy(args.base / name, args.out / name)
    print(f"exported {ckpt.name} -> {args.out}. Start the app with XTTS_URDU_DIR={args.out.resolve()}")


if __name__ == "__main__":
    main()
