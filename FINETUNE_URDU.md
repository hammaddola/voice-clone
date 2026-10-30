# Phase 2: fine-tuning XTTS-v2 on Urdu

XTTS-v2 has no Urdu. The app's built-in fallback speaks Roman Urdu or Devanagari through Hindi mode, which
is understandable but clearly accented, and it can't read Urdu script (Nastaliq). Fine-tuning the XTTS GPT
on Mozilla Common Voice Urdu teaches it real Urdu text-to-sound mapping.

## 0. What you need

- **An NVIDIA GPU with ≥16 GB VRAM** (e.g. RTX 4080/4090, A10G, L4, A100). This laptop has none, and CPU
  training would take months, so rent one: RunPod, Lambda, Vast.ai, or Colab Pro (A100/L4). Linux is easiest.
- ~30 GB disk.
- The same `coqui-tts==0.28.0` install as the app, with CUDA PyTorch:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
  ```

## 1. Get the data

1. Go to <https://commonvoice.mozilla.org/en/datasets>, pick **Urdu**, take the latest version, enter
   your email and accept the terms (CC0 licence).
2. Extract it. You want the folder `cv-corpus-<version>/ur/`, which holds `validated.tsv` and `clips/*.mp3`.

Common Voice Urdu is small (tens of hours, many speakers, phone-quality mics). That's enough to teach
pronunciation; voice quality still comes from your reference clip at inference time.

## 2. Prepare

```bash
python finetune/prepare_common_voice.py --cv /data/cv-corpus-XX/ur --out data/cv_ur \
    --vocab ~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/vocab.json
```

What it does:
- keeps clips with ≥2 up-votes and 0 down-votes, 1–11.5 s after trimming silence (XTTS's max is ~11.6 s)
- drops sentences containing digits (XTTS would expand them with *Arabic* number words). Write numbers
  out in words when you use the model too.
- normalizes Arabic look-alike code points to Urdu ones (ي→ی, ك→ک, ه→ہ)
- writes 22.05 kHz mono WAVs plus `metadata_train.csv` / `metadata_eval.csv` in Coqui format
- prints **Urdu characters missing from the XTTS vocab**. XTTS's BPE vocab already covers Arabic script
  because Arabic is supported. If Urdu-only letters (ٹ ڈ ڑ ں ے ھ ۓ) show up as missing, they become `[UNK]` and
  that sound can't be learned. Fixes, easiest first:
  1. Map each missing letter to its closest covered letter in `FOLD` in `prepare_common_voice.py`
     (e.g. ٹ→ت), and do the same at inference. Lossy, but no model changes.
  2. Extend the tokenizer: add the characters to `vocab.json`, then resize the GPT text-embedding and
     text-head layers before training. That's more involved and needs more data to learn the new rows well.

## 3. Train

```bash
python finetune/train_xtts_urdu.py --data data/cv_ur --out runs/xtts_ur \
    --epochs 10 --batch 4 --grad-accum 63 --test-speaker-wav some_urdu_speaker.wav
```

- Base files (`model.pth`, `dvae.pth`, `mel_stats.pth`, `vocab.json`, …) download from HF `coqui/XTTS-v2`
  into `runs/xtts_ur/base/`.
- **Language token:** Urdu is trained under the existing `[ar]` token, so tokenizer and model shapes stay
  the same. The app's fine-tuned engine routes Urdu to `ar` for this reason. Arabic output from the
  fine-tuned model will degrade; keep using the stock engine for Arabic.
- Key settings: lr 5e-6 (the official recipe value; higher forgets the other languages faster),
  `batch × grad_accum ≥ 252`, 3–6 s conditioning windows. With 16 GB VRAM use `--batch 2 --grad-accum 126`.
- Watch the logs: `tensorboard --logdir runs/xtts_ur`. Eval `loss_mel_ce` should fall then flatten.
  Stop when it rises for a couple of evals; that's overfitting. Expect roughly 2–8 GPU-hours for Common
  Voice Urdu on an A100.
- Listen to the test sentences in TensorBoard's Audio tab each eval.

## 4. Export and load in the app

```bash
python finetune/export_model.py --run runs/xtts_ur/xtts_urdu-<date> --base runs/xtts_ur/base --out models/xtts_urdu
```

Copy `models/xtts_urdu/` (model.pth, config.json, vocab.json, speakers_xtts.pth; ~1.8 GB) to the
laptop, then start the app with:

```bat
set XTTS_URDU_DIR=C:\path\to\voice_clone\models\xtts_urdu
run.bat
```

A new **"XTTS-v2 Urdu (fine-tuned)"** engine appears. Pick it with the Urdu language and type normal Urdu
script. The Nastaliq block and the "approximate" warning apply only to the stock engines.

## Licence

The fine-tuned model is a derivative of XTTS-v2 and stays under the **Coqui Public Model License
(non-commercial)**. Common Voice data is CC0.
