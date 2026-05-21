import argparse
import json
from pathlib import Path

import torch


def list_checkpoints(output_dir):
    output_path = Path(output_dir)
    ckpts = sorted(output_path.glob("checkpoint*.pth"))
    # Filter to epoch-style checkpoints (checkpoint0004.pth), skip checkpoint.pth.
    epoch_ckpts = [p for p in ckpts if p.stem != "checkpoint" and p.stem[10:].isdigit()]
    return epoch_ckpts


def parse_log_ap(log_path):
    rows = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "test_coco_eval_bbox" in item:
                rows.append(item)
    return [r["test_coco_eval_bbox"][0] for r in rows]


def choose_checkpoints(output_dir, mode, top_k, last_k, log_path):
    ckpts = list_checkpoints(output_dir)
    if not ckpts:
        raise ValueError(f"No epoch checkpoints found under {output_dir}")

    if mode == "last":
        selected = ckpts[-last_k:]
        if len(selected) < last_k:
            raise ValueError(
                f"Not enough checkpoints for last-{last_k}. found={len(selected)}"
            )
        return selected

    if log_path is None:
        raise ValueError("--log is required for mode=top")
    ap = parse_log_ap(log_path)
    if len(ap) != len(ckpts):
        raise ValueError(
            f"Checkpoint count ({len(ckpts)}) does not match eval rows ({len(ap)})."
        )
    ranked = sorted(range(len(ap)), key=lambda i: ap[i], reverse=True)[:top_k]
    ranked = sorted(ranked)
    return [ckpts[i] for i in ranked]


def average_state_dicts(paths, key="model"):
    avg = None
    for path in paths:
        state = torch.load(path, map_location="cpu")
        state_dict = state[key]
        if avg is None:
            avg = {k: v.clone().float() for k, v in state_dict.items()}
        else:
            for k, v in state_dict.items():
                avg[k] += v.float()
    num = float(len(paths))
    for k in avg:
        avg[k] /= num
    return avg


def main():
    parser = argparse.ArgumentParser("Average model checkpoints")
    parser.add_argument("--output-dir", required=True, help="Training output directory")
    parser.add_argument(
        "--mode",
        choices=["last", "top"],
        default="last",
        help="Average last-k checkpoints or top-k AP checkpoints",
    )
    parser.add_argument("--last-k", type=int, default=3, help="k for mode=last")
    parser.add_argument("--top-k", type=int, default=3, help="k for mode=top")
    parser.add_argument("--log", default=None, help="log.txt path for mode=top")
    parser.add_argument(
        "--key",
        default="model",
        choices=["model", "model_ema"],
        help="Checkpoint key to average",
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help="Where to save averaged checkpoint (.pth)",
    )
    args = parser.parse_args()

    selected = choose_checkpoints(
        output_dir=args.output_dir,
        mode=args.mode,
        top_k=args.top_k,
        last_k=args.last_k,
        log_path=args.log,
    )
    print("Selected checkpoints:")
    for p in selected:
        print(f"  {p}")

    avg_model = average_state_dicts(selected, key=args.key)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": avg_model,
            "meta": {
                "source_checkpoints": [str(p) for p in selected],
                "mode": args.mode,
                "key": args.key,
            },
        },
        save_path,
    )
    print(f"Saved averaged checkpoint: {save_path}")


if __name__ == "__main__":
    main()
