#!/usr/bin/env python3
"""ChatGPT-like English chat via OpenAI API, alongside genome tooling.

This does NOT transfer human cognition into the DNA model. It provides
an external LLM chat interface for natural-language reasoning.
"""
from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are an assistant in the intelgene project. "
    "Be clear and accurate. Distinguish DNA-model results from general reasoning."
)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--once", help="One-shot question and exit.")
    args = parser.parse_args()

    require_env("OPENAI_API_KEY")
    client = OpenAI()

    print("intelgene English chat (OpenAI-backed)")
    print("Type 'exit' to quit.\n")

    def respond(user_text: str) -> str:
        completion = client.responses.create(
            model=args.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        )
        return completion.output_text

    if args.once:
        print(respond(args.once))
        return

    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if msg.lower() in {"exit", "quit"}:
            print("bye")
            return
        print("assistant>", respond(msg))


if __name__ == "__main__":
    sys.exit(main())
