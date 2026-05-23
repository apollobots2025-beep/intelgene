#!/usr/bin/env python3
"""Evaluate trained genome model and report practical progress metrics."""
from __future__ import annotations

import argparse
import pathlib

import torch
from torch import nn

from train_model import SeqDataset, TinyLM


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=100_000)
    p.add_argument("--sample-steps", type=int, default=100)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SeqDataset(pathlib.Path(args.data), block_size=args.block_size, max_tokens=args.max_tokens)
    if len(ds) == 0:
        raise SystemExit("Dataset too small for evaluation. Increase --max-tokens or lower --block-size.")

    model = TinyLM().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        for i in range(min(args.sample_steps, len(ds))):
            x, y = ds[i]
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += float(loss.item())

            preds = logits.argmax(dim=-1)
            correct += int((preds == y).sum().item())
            total_tokens += int(y.numel())

    avg_loss = total_loss / min(args.sample_steps, len(ds))
    acc = correct / max(1, total_tokens)
    random_baseline = 0.2
    normalized = max(0.0, min(1.0, (acc - random_baseline) / (1.0 - random_baseline)))

    print(f"eval_loss={avg_loss:.4f}")
    print(f"next_token_accuracy={acc:.4f}")
    print(f"genome_modeling_progress={normalized*100:.2f}%")
    print("human_intelligence_percent=not_measurable_with_this_pipeline")


if __name__ == "__main__":
    main()
