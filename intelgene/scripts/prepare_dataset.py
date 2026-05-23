#!/usr/bin/env python3
"""Convert hg38 FASTA to an AI-readable token stream.

Format: space-separated integer tokens where
A=0 C=1 G=2 T=3 N=4
"""
from __future__ import annotations

import argparse
import gzip
import pathlib

VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def fasta_iter_lines(path: pathlib.Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.startswith(">"):
                yield line.strip().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()

    in_path = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "genome_tokens.txt"

    total = 0
    with open(out_file, "w", encoding="utf-8") as w:
        for seq in fasta_iter_lines(in_path):
            tokens = [str(VOCAB.get(base, 4)) for base in seq]
            w.write(" ".join(tokens))
            w.write("\n")
            total += len(tokens)

    print(f"Wrote {out_file} with {total} tokens")


if __name__ == "__main__":
    main()
