"""Create a JSONL manifest from aligned mixture, target, and visual directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixtures", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--visuals", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mixture_dir = Path(args.mixtures)
    target_dir = Path(args.targets)
    visual_dir = Path(args.visuals)
    records = []
    for mixture in sorted(mixture_dir.glob("*.wav")):
        target = target_dir / mixture.name
        visual = visual_dir / f"{mixture.stem}.npy"
        if target.exists() and visual.exists():
            records.append({"id": mixture.stem, "mixture": str(mixture.resolve()), "target": str(target.resolve()), "visual": str(visual.resolve())})
    with open(args.output, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
