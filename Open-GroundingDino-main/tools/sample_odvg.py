import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Sample a subset from ODVG jsonl file.")
    parser.add_argument("--input", type=str, required=True, help="Input ODVG jsonl path.")
    parser.add_argument("--output", type=str, required=True, help="Output sampled jsonl path.")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Sampling ratio in (0, 1]. Default: 0.2",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop samples with zero detection instances before sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0 < args.ratio <= 1.0):
        raise ValueError("ratio must be in (0, 1].")

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if args.drop_empty:
        filtered = []
        for line in lines:
            item = json.loads(line)
            if len(item.get("detection", {}).get("instances", [])) > 0:
                filtered.append(line)
        lines = filtered

    total = len(lines)
    sample_count = max(1, int(total * args.ratio))
    rng = random.Random(args.seed)
    indices = list(range(total))
    rng.shuffle(indices)
    chosen = sorted(indices[:sample_count])

    with open(out_path, "w", encoding="utf-8") as f:
        for idx in chosen:
            f.write(lines[idx])

    print(
        {
            "input": str(in_path),
            "output": str(out_path),
            "total": total,
            "sampled": sample_count,
            "ratio": args.ratio,
            "seed": args.seed,
        }
    )


if __name__ == "__main__":
    main()
