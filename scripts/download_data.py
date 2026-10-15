"""Download the study data from Zenodo into data/ and check the files.

Usage (from the repository root):
    python scripts/download_data.py

Every file is checked against the SHA-256 list published in the same record.
Files that are already in data/ and match the list are not downloaded again.
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = "23395396"
DEFAULT_BASE = f"https://zenodo.org/records/{RECORD}/files"
FILES = ["sentences.csv", "reference_standard.csv", "participants.csv",
         "responses_control.csv", "responses_experimental.csv"]


def fetch(base, name):
    with urllib.request.urlopen(f"{base}/{name}?download=1", timeout=60) as response:
        return response.read()


def read_checksums(text):
    checksums = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            checksums[name.strip().lstrip("*")] = digest
    return checksums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=ROOT / "data")
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    checksums = read_checksums(fetch(args.base_url, "SHA256SUMS.txt").decode("utf-8"))
    for name in FILES:
        target = out / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == checksums[name]:
            print(f"ok       {name} (already there)")
            continue
        content = fetch(args.base_url, name)
        if hashlib.sha256(content).hexdigest() != checksums[name]:
            raise SystemExit(f"Checksum mismatch for {name}. Nothing was written for this file.")
        target.write_bytes(content)
        print(f"saved    {name}")
    print(f"Data in {out}")


if __name__ == "__main__":
    main()
