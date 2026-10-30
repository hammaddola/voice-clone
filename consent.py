"""Consent gate: nothing gets cloned unless the speaker's permission is logged against the exact audio."""
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(__file__).parent / "consent_log.csv"
FIELDS = ["timestamp_utc", "speaker", "sha256", "filename"]


class ConsentError(Exception):
    pass


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def logged_hashes() -> set:
    if not LOG.exists():
        return set()
    with open(LOG, newline="", encoding="utf-8") as f:
        return {row["sha256"] for row in csv.DictReader(f)}


def require(paths, speaker: str, confirmed: bool) -> list:
    """Return the SHA-256 of each clip. Clips not yet in the log need a name + confirmation, and get logged."""
    if not paths:
        raise ConsentError("Upload at least one reference clip.")
    hashes = [sha256(p) for p in paths]
    known = logged_hashes()
    new = [(p, h) for p, h in zip(paths, hashes) if h not in known]
    if new:
        speaker = (speaker or "").strip()
        if not speaker or not confirmed:
            raise ConsentError(
                "Consent required: enter the speaker's name and tick the box confirming "
                "they gave permission to clone their voice."
            )
        write_header = not LOG.exists()
        with open(LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, FIELDS)
            if write_header:
                w.writeheader()
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for p, h in new:
                w.writerow({"timestamp_utc": now, "speaker": speaker, "sha256": h, "filename": Path(p).name})
    return hashes
