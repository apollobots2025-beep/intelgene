#!/usr/bin/env python3
"""Download hg38 reference genome from UCSC."""
from __future__ import annotations

import argparse
import pathlib
import urllib.request

URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw", help="Output directory")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hg38.fa.gz"

    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"Already exists: {out_file}")
        return

    print(f"Downloading {URL} -> {out_file}")
    urllib.request.urlretrieve(URL, out_file)
    print("Done")


if __name__ == "__main__":
    main()
