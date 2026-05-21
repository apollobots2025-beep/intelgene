# intelgene

A practical genomics AI starter project.

## What this project does

1. Downloads the UCSC `hg38.fa.gz` reference genome.
2. Converts FASTA into an AI-readable tokenized corpus.
3. Trains a small character-level language model on genome sequence.

> Note: No current AI system can literally "think like a human" from genome data alone. This project provides a realistic baseline for sequence modeling.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_hg38.py --out data/raw
python scripts/prepare_dataset.py --input data/raw/hg38.fa.gz --out data/processed
python scripts/train_model.py --data data/processed/genome_tokens.txt --out models --epochs 1 --max-tokens 500000
python scripts/evaluate_model.py --data data/processed/genome_tokens.txt --model models/char_lm.pt
```

## Data source

UCSC Genome Browser download location:
- https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/

## Outputs

- `data/raw/hg38.fa.gz`
- `data/processed/genome_tokens.txt`
- `models/char_lm.pt`
- `models/vocab.json`


## Important limitations

- This project can model DNA sequence statistics.
- It cannot be used to measure or claim a valid "percent of human intelligence."
- `scripts/evaluate_model.py` reports **genome modeling progress** (next-token performance vs random baseline), not human cognition.


## How to talk to it

This model is a **DNA sequence model**, so you "talk" to it with DNA-base prompts (`A/C/G/T/N`) and it continues the sequence.

Example:

```bash
source .venv/bin/activate
python scripts/talk_to_model.py --model models/char_lm.pt --prompt ACGTACGT --steps 80 --temperature 0.8
```

What you'll get:
- `prompt`: your input seed sequence
- `generated`: the prompt plus model-predicted continuation

It does not understand natural language chat; it predicts likely next DNA tokens.


## Plain-English chat interface

If you want to interact in normal English (instead of DNA tokens), use:

```bash
source .venv/bin/activate
python scripts/chat_english.py
```

Example prompts:
- `download hg38`
- `prepare dataset`
- `train model`
- `evaluate model`
- `generate dna from ACGT`

Important: this is an orchestration/chat wrapper around the DNA model pipeline. It is **not** human-like intelligence.
