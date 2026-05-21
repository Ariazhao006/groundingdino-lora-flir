import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path


def build_main_eval_command(args):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "main.py"),
        "--eval",
        "--output_dir",
        args.output_dir,
        "-c",
        args.config,
        "--datasets",
        args.datasets,
        "--num_workers",
        str(args.num_workers),
        "--pretrain_model_path",
        args.pretrain_model_path,
    ]
    if args.amp:
        cmd.append("--amp")
    if args.options:
        cmd.extend(["--options", *args.options])
    return cmd


def format_aligned_log(raw_test_stats):
    bbox = raw_test_stats.get("test_coco_eval_bbox", [])
    best_map = float(bbox[0]) if len(bbox) > 0 else 0.0
    best_aps = float(bbox[3]) if len(bbox) > 3 else 0.0
    return {
        "train_lr": 0.0,
        "train_loss": None,
        "test_coco_eval_bbox": bbox,
        "early_stop_best_ap": None,
        "early_stop_bad_epochs": 0,
        "early_stop_should_stop": False,
        "best_map_summary": {"best_res": best_map, "best_ep": 0},
        "best_aps_summary": {"best_res": best_aps, "best_ep": 0},
        "now_time": str(datetime.datetime.now()),
        "epoch_time": "0:00:00",
    }


def main():
    parser = argparse.ArgumentParser("Run zero-epoch eval and align outputs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--pretrain_model_path", required=True)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--options", nargs="*", default=[])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_main_eval_command(args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    log_path = output_dir / "log.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"Expected eval log not found: {log_path}")

    with log_path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Eval log is empty: {log_path}")

    raw_eval = json.loads(lines[-1])
    if "test_coco_eval_bbox" not in raw_eval:
        raise ValueError("Eval output missing test_coco_eval_bbox, cannot align.")

    # Keep the raw eval output for reference, then overwrite log.txt with aligned schema.
    with (output_dir / "log_eval_raw.txt").open("w") as f:
        for line in lines:
            f.write(line + "\n")

    aligned = format_aligned_log(raw_eval)
    with log_path.open("w") as f:
        f.write(json.dumps(aligned) + "\n")

    best_summary = {
        "best_map": {"best_res": aligned["best_map_summary"]["best_res"], "best_ep": 0},
        "best_aps": {"best_res": aligned["best_aps_summary"]["best_res"], "best_ep": 0},
    }
    with (output_dir / "best_summary.json").open("w") as f:
        json.dump(best_summary, f, indent=2)

    preferred = {
        "checkpoint": Path(args.pretrain_model_path).name,
        "metric": aligned["best_map_summary"]["best_res"],
        "selector": "best_map",
    }
    with (output_dir / "preferred_checkpoint.json").open("w") as f:
        json.dump(preferred, f, indent=2)

    print(f"Done. Aligned outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
