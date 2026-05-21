#!/usr/bin/env python3
"""Interact with a trained genome model by prompting DNA bases."""
from __future__ import annotations

import argparse
import pathlib

import torch

from train_model import TinyLM

VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
INV_VOCAB = {v: k for k, v in VOCAB.items()}


def encode(seq: str) -> list[int]:
    return [VOCAB.get(ch, VOCAB["N"]) for ch in seq.upper()]


def decode(ids: list[int]) -> str:
    return "".join(INV_VOCAB.get(i, "N") for i in ids)


def sample_next(logits: torch.Tensor, temperature: float) -> int:
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/char_lm.pt")
    p.add_argument("--prompt", default="ACGTACGT")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyLM().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    ids = encode(args.prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(args.steps):
            logits = model(x)
            nxt = sample_next(logits[0, -1], args.temperature)
            ids.append(nxt)
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    print("prompt:", args.prompt)
    print("generated:", decode(ids))


if __name__ == "__main__":
    main()
