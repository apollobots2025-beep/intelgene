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


## Reality check: architecture for your goal

Your request combines two different systems:

1. **Genome model (local)**: learns DNA sequence statistics.
2. **General reasoning model (LLM)**: handles English chat and complex reasoning.

This repo can support both, but it cannot scientifically "compress GPT-level reasoning into DNA tokens" with current methods.

## ChatGPT-like English chat

Use the OpenAI-backed chat script for natural-language interaction:

```bash
source .venv/bin/activate
export OPENAI_API_KEY=your_key_here
python scripts/chat_with_openai.py --model gpt-5
```

One-shot example:

```bash
python scripts/chat_with_openai.py --model gpt-5 --once "Summarize chr1 motifs from my run"
```

This gives you ChatGPT-style English interaction while keeping the DNA pipeline separate and honest.


## From-scratch path (realistic)

If you want a model trained from absolute scratch, use the included tiny Transformer trainer:

```bash
source .venv/bin/activate
python scripts/train_scratch_transformer.py --text data/corpus/internet_sample.txt --out models/scratch --epochs 1
```

### Important reality check

- Training a truly ChatGPT-level model from scratch requires massive data, compute, engineering, alignment, and evaluation infrastructure.
- This repository now includes a **from-scratch baseline** you can build on, but it will not reach ChatGPT-level intelligence on local hardware.
- Best practice is hybrid: train your own specialist model + use frontier LLMs where high-level reasoning is required.


## One-command setup, train, and chat

If you want a single command that installs dependencies, downloads hg38, prepares data, trains, evaluates, and then opens chat:

```bash
python scripts/run_intelgene_oneclick.py --chat english
```

Optional chat modes:
- `--chat english` (local orchestration wrapper)
- `--chat openai` (OpenAI-backed, requires `OPENAI_API_KEY`)
- `--chat dna` (direct DNA continuation)

Reality check: this one-click pipeline is practical for experimentation, but it does **not** produce GPT-5.5-level reasoning from DNA.
