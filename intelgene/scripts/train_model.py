#!/usr/bin/env python3
"""Train a tiny character-level LM on tokenized genome data."""
from __future__ import annotations

import argparse
import json
import pathlib

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def load_first_tokens(token_file: pathlib.Path, max_tokens: int) -> list[int]:
    tokens: list[int] = []
    with token_file.open("r", encoding="utf-8") as f:
        for line in f:
            for tok in line.split():
                tokens.append(int(tok))
                if len(tokens) >= max_tokens:
                    return tokens
    return tokens


class SeqDataset(Dataset):
    def __init__(self, token_file: pathlib.Path, block_size: int = 256, max_tokens: int = 2_000_000):
        ids = load_first_tokens(token_file, max_tokens=max_tokens)
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block - 1)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block]
        y = self.data[idx + 1 : idx + self.block + 1]
        return x, y


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 5, d_model: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        z = self.embed(x)
        z, _ = self.rnn(z)
        return self.head(z)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="models")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=2_000_000)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = SeqDataset(pathlib.Path(args.data), block_size=args.block_size, max_tokens=args.max_tokens)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = TinyLM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for i, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            if (i + 1) % 100 == 0:
                print(f"epoch={epoch + 1} step={i + 1} loss={running / 100:.4f}")
                running = 0.0

    torch.save(model.state_dict(), out_dir / "char_lm.pt")
    (out_dir / "vocab.json").write_text(
        json.dumps({"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}, indent=2), encoding="utf-8"
    )
    print(f"Saved model to {out_dir}")


if __name__ == "__main__":
    main()
