"""Mozilla Common Voice Urdu -> Coqui-format XTTS training set.

    python finetune/prepare_common_voice.py --cv path/to/cv-corpus-XX/ur --out data/cv_ur

Keeps validated clips (>=2 up-votes, 0 down-votes), 1-11.5 s long, without digits (XTTS would read them
with Arabic number words). Writes 22.05 kHz mono WAVs plus metadata_train.csv / metadata_eval.csv
(audio_file|text|speaker_name), and reports Urdu characters missing from the XTTS vocab.
"""
import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import soundfile as sf

SR = 22050
MIN_S, MAX_S = 1.0, 11.5  # XTTS GPTArgs.max_wav_length is ~11.6 s
# Arabic code points that look like Urdu letters -> the Urdu ones
FOLD = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ"})


def clean(text):
    text = unicodedata.normalize("NFC", text).translate(FOLD)
    return re.sub(r"\s+", " ", text).strip()


def convert(src, dst):
    try:
        wav, _ = librosa.load(src, sr=SR, mono=True)
    except Exception as e:
        return None, f"{src.name}: {e}"
    wav, _ = librosa.effects.trim(wav, top_db=35)
    dur = len(wav) / SR
    if not MIN_S <= dur <= MAX_S:
        return None, None
    sf.write(dst, wav, SR)
    return dur, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True, type=Path, help="Common Voice language folder (has validated.tsv, clips/)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--eval-frac", type=float, default=0.03)
    ap.add_argument("--vocab", type=Path, help="XTTS vocab.json, for the character-coverage report")
    args = ap.parse_args()

    with open(args.cv / "validated.tsv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                if int(r.get("up_votes") or 0) >= 2 and int(r.get("down_votes") or 0) == 0]
    rows = [r | {"text": clean(r["sentence"])} for r in rows]
    rows = [r for r in rows if r["text"] and not re.search(r"[0-9٠-٩۰-۹]", r["text"])]
    print(f"{len(rows)} validated clips after vote/digit filtering")

    wav_dir = args.out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.cv / "clips" / r["path"], wav_dir / (Path(r["path"]).stem + ".wav")) for r in rows]
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda j: convert(*j), jobs))
    for _, err in results:
        if err:
            print("skip", err, file=sys.stderr)
    kept = [(r, d) for r, (d, _) in zip(rows, results) if d]
    hours = sum(d for _, d in kept) / 3600
    print(f"kept {len(kept)} clips, {hours:.1f} h")

    random.seed(0)
    random.shuffle(kept)
    n_eval = max(1, int(len(kept) * args.eval_frac))
    for name, part in [("metadata_eval.csv", kept[:n_eval]), ("metadata_train.csv", kept[n_eval:])]:
        with open(args.out / name, "w", encoding="utf-8", newline="") as f:
            f.write("audio_file|text|speaker_name\n")
            for r, _ in part:
                f.write(f"wavs/{Path(r['path']).stem}.wav|{r['text'].replace('|', ' ')}|{r['client_id'][:8]}\n")
    print(f"wrote {args.out / 'metadata_train.csv'} ({len(kept) - n_eval}) and metadata_eval.csv ({n_eval})")

    if args.vocab:
        vocab = json.loads(args.vocab.read_text(encoding="utf-8"))["model"]["vocab"]
        chars = {c for r, _ in kept for c in r["text"] if not c.isspace()}
        missing = sorted(c for c in chars if c not in vocab)
        print("characters missing from XTTS vocab (become [UNK]):",
              " ".join(f"{c}(U+{ord(c):04X})" for c in missing) or "none")


if __name__ == "__main__":
    main()
