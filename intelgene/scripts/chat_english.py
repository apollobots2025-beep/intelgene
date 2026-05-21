#!/usr/bin/env python3
"""Plain-English interface for the intelgene project.

This is an orchestration/chat wrapper, not a human-level cognition system.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

HELP_TEXT = """
You can chat in plain English. Supported intents:
- "download hg38"
- "prepare dataset"
- "train model"
- "evaluate model"
- "generate dna from ACGT"
- "status"
- "help"
- "exit"
""".strip()

DISCLAIMER = (
    "I can chat in English, but this project does NOT produce human-like intelligence. "
    "It trains a DNA next-token model and reports sequence-model metrics."
)


def run_cmd(cmd: str) -> int:
    print(f"\n$ {cmd}")
    proc = subprocess.run(shlex.split(cmd))
    return proc.returncode


def handle_intent(text: str) -> str:
    t = text.lower().strip()
    if t in {"help", "?"}:
        return HELP_TEXT
    if t == "status":
        return DISCLAIMER
    if "human" in t and "think" in t:
        return (
            "I can't honestly claim human-like thinking. "
            "I can only help run genome data prep, training, evaluation, and sampling."
        )
    if "download" in t and "hg38" in t:
        rc = run_cmd("python scripts/download_hg38.py --out data/raw")
        return "Download complete." if rc == 0 else "Download failed."
    if "prepare" in t or "token" in t:
        rc = run_cmd("python scripts/prepare_dataset.py --input data/raw/hg38.fa.gz --out data/processed")
        return "Dataset preparation complete." if rc == 0 else "Dataset preparation failed."
    if "train" in t:
        rc = run_cmd(
            "python scripts/train_model.py --data data/processed/genome_tokens.txt --out models --epochs 1 --max-tokens 500000"
        )
        return "Training complete." if rc == 0 else "Training failed."
    if "evaluate" in t:
        rc = run_cmd(
            "python scripts/evaluate_model.py --data data/processed/genome_tokens.txt --model models/char_lm.pt"
        )
        return "Evaluation complete." if rc == 0 else "Evaluation failed."
    if "generate" in t and "dna" in t:
        prompt = "ACGT"
        words = text.split()
        if "from" in [w.lower() for w in words]:
            i = [w.lower() for w in words].index("from")
            if i + 1 < len(words):
                prompt = words[i + 1].strip().upper()
        rc = run_cmd(f"python scripts/talk_to_model.py --model models/char_lm.pt --prompt {prompt} --steps 80 --temperature 0.8")
        return "Generation complete." if rc == 0 else "Generation failed."
    return "I understood plain English, but not that request yet. Type 'help'."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", help="Run one English instruction and exit.")
    args = parser.parse_args()

    print("intelgene English interface")
    print(DISCLAIMER)
    print("Type 'help' for commands.\n")

    if args.once:
        print(handle_intent(args.once))
        return

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if text.lower() in {"exit", "quit"}:
            print("bye")
            return
        print("intelgene>", handle_intent(text))


if __name__ == "__main__":
    sys.exit(main())
