"""Fine-tune the XTTS-v2 GPT on Urdu (adapted from coqui-tts recipes/ljspeech/xtts_v2/train_gpt_xtts.py).

    python finetune/train_xtts_urdu.py --data data/cv_ur --out runs/xtts_ur

Needs an NVIDIA GPU with >= 16 GB VRAM (CPU training is not practical). Urdu is trained under the
existing [ar] language token, so the tokenizer and model shapes need no changes; the app's
"XTTS-v2 Urdu (fine-tuned)" engine routes Urdu to "ar" for the same reason.
"""
import argparse
import os
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager

LANG_TOKEN = "ar"
BASE = "https://huggingface.co/coqui/XTTS-v2/resolve/main/"


def fetch_base(dir_: Path):
    dir_.mkdir(parents=True, exist_ok=True)
    files = {n: dir_ / n for n in ["dvae.pth", "mel_stats.pth", "vocab.json", "model.pth", "config.json",
                                   "speakers_xtts.pth"]}
    missing = [BASE + n for n, p in files.items() if not p.exists()]
    if missing:
        ModelManager._download_model_files(missing, str(dir_), progress_bar=True)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="output of prepare_common_voice.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=63, help="batch * grad_accum should be >= 252")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--test-speaker-wav", type=Path, help="reference clip for test sentences during training")
    args = ap.parse_args()

    base = fetch_base(args.out / "base")
    dataset = BaseDatasetConfig(formatter="coqui", dataset_name="cv_ur", path=str(args.data),
                                meta_file_train="metadata_train.csv", meta_file_val="metadata_eval.csv",
                                language=LANG_TOKEN)
    model_args = GPTArgs(
        max_conditioning_length=132300, min_conditioning_length=66150,  # 3-6 s at 22.05 kHz
        max_wav_length=255995, max_text_length=200,
        mel_norm_file=str(base["mel_stats.pth"]), dvae_checkpoint=str(base["dvae.pth"]),
        xtts_checkpoint=str(base["model.pth"]), tokenizer_file=str(base["vocab.json"]),
        gpt_num_audio_tokens=1026, gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True,
    )
    test_sentences = []
    if args.test_speaker_wav:
        test_sentences = [{"text": t, "speaker_wav": [str(args.test_speaker_wav)], "language": LANG_TOKEN}
                          for t in ["آپ کیسے ہیں؟ آج موسم بہت اچھا ہے۔", "میں اردو بول رہا ہوں۔"]]
    config = GPTTrainerConfig(
        output_path=str(args.out), model_args=model_args, run_name="xtts_urdu", project_name="xtts_urdu",
        audio=XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000),
        batch_size=args.batch, batch_group_size=48, eval_batch_size=args.batch, num_loader_workers=4,
        eval_split_max_size=256, print_step=50, plot_step=100, log_model_step=1000,
        save_step=2000, save_n_checkpoints=2, save_checkpoints=True, print_eval=False, epochs=args.epochs,
        optimizer="AdamW", optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=args.lr, lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=test_sentences,
    )
    model = GPTTrainer.init_from_config(config)
    train, evals = load_tts_samples([dataset], eval_split=True, eval_split_max_size=config.eval_split_max_size)
    trainer = Trainer(
        TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=True,
                    grad_accum_steps=args.grad_accum),
        config, output_path=str(args.out), model=model, train_samples=train, eval_samples=evals,
    )
    trainer.fit()
    print("done. Export with: python finetune/export_model.py --run", trainer.output_path,
          "--base", base["model.pth"].parent, "--out models/xtts_urdu")


if __name__ == "__main__":
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    main()
