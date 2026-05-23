#!/usr/bin/env python3
"""Train a tiny Transformer LM from scratch on local text data.

This script is intentionally simple and educational. It does NOT reproduce
ChatGPT-level intelligence; it provides a true from-scratch baseline.
"""
from __future__ import annotations

import argparse
import pathlib

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CharDataset(Dataset):
    def __init__(self, text_path: pathlib.Path, block_size: int = 128, max_chars: int = 2_000_000):
        raw = text_path.read_text(encoding="utf-8")[:max_chars]
        self.chars = sorted(set(raw))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.data = torch.tensor([self.stoi[c] for c in raw], dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, d_model: int = 192, n_heads: int = 6, n_layers: int = 4):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self.tok(x) + self.pos(pos)
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        h = self.tr(h, mask=mask)
        h = self.ln(h)
        return self.head(h)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True, help="Plain text file for from-scratch LM training")
    p.add_argument("--out", default="models/scratch")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--max-chars", type=int, default=2_000_000)
    args = p.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = CharDataset(pathlib.Path(args.text), block_size=args.block_size, max_chars=args.max_chars)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTransformerLM(vocab_size=len(ds.chars), block_size=args.block_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
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
                print(f"epoch={epoch+1} step={i+1} loss={running/100:.4f}")
                running = 0.0

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "chars": ds.chars,
            "block_size": args.block_size,
        },
        out / "tiny_transformer_scratch.pt",
    )
    print(f"Saved {out / 'tiny_transformer_scratch.pt'}")


if __name__ == "__main__":
    main()
