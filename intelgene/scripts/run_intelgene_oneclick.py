#!/usr/bin/env python3
"""One-click setup + train + chat launcher for intelgene.

This orchestrates the local DNA pipeline in one command.
It does NOT create GPT-level general intelligence from DNA.
"""
from __future__ import annotations

import argparse
import pathlib
import shlex
import subprocess
import sys


def run(cmd: str, env=None) -> None:
    print(f"\n$ {cmd}")
    rc = subprocess.run(shlex.split(cmd), env=env).returncode
    if rc != 0:
        raise SystemExit(rc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--python", default="python")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=500_000)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--chat", choices=["english", "openai", "dna"], default="english")
    p.add_argument("--openai-model", default="gpt-5")
    args = p.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    venv = repo / ".venv"

    run(f"{args.python} -m venv {venv}")
    py = venv / "bin" / "python"
    if not py.exists():
        raise SystemExit(f"Virtualenv Python not found at {py}")

    run(f"{py} -m pip install --upgrade pip")
    run(f"{py} -m pip install -r {repo / 'requirements.txt'}")

    if not args.skip_download:
        run(f"{py} {repo / 'scripts/download_hg38.py'} --out {repo / 'data/raw'}")

    run(
        f"{py} {repo / 'scripts/prepare_dataset.py'} "
        f"--input {repo / 'data/raw/hg38.fa.gz'} --out {repo / 'data/processed'}"
    )
    run(
        f"{py} {repo / 'scripts/train_model.py'} "
        f"--data {repo / 'data/processed/genome_tokens.txt'} --out {repo / 'models'} "
        f"--epochs {args.epochs} --max-tokens {args.max_tokens}"
    )
    run(
        f"{py} {repo / 'scripts/evaluate_model.py'} "
        f"--data {repo / 'data/processed/genome_tokens.txt'} --model {repo / 'models/char_lm.pt'}"
    )

    print("\nSetup complete. Launching chat...\n")
    if args.chat == "english":
        run(f"{py} {repo / 'scripts/chat_english.py'}")
    elif args.chat == "openai":
        run(f"{py} {repo / 'scripts/chat_with_openai.py'} --model {args.openai_model}")
    else:
        run(f"{py} {repo / 'scripts/talk_to_model.py'} --model {repo / 'models/char_lm.pt'}")


if __name__ == "__main__":
    sys.exit(main())
