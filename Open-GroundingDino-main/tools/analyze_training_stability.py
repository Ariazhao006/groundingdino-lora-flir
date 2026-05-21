import argparse
import json
from pathlib import Path


def load_eval_rows(log_path):
    rows = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "test_coco_eval_bbox" not in record:
                continue
            rows.append(record)
    return rows


def summarize_log(log_path):
    rows = load_eval_rows(log_path)
    if not rows:
        raise ValueError(f"No eval rows found in {log_path}")

    ap = [r["test_coco_eval_bbox"][0] for r in rows]
    ap50 = [r["test_coco_eval_bbox"][1] for r in rows]
    ap75 = [r["test_coco_eval_bbox"][2] for r in rows]
    aps = [r["test_coco_eval_bbox"][3] for r in rows]

    best_idx = max(range(len(ap)), key=lambda i: ap[i])
    last_k = min(5, len(ap))
    last5_avg_ap = sum(ap[-last_k:]) / last_k
    rollback = ap[best_idx] - ap[-1]

    return {
        "log_path": str(log_path),
        "epochs": len(rows),
        "best_epoch": best_idx + 1,
        "best_ap": ap[best_idx],
        "best_ap50": ap50[best_idx],
        "best_ap75": ap75[best_idx],
        "best_aps": aps[best_idx],
        "last_ap": ap[-1],
        "last_ap50": ap50[-1],
        "last_ap75": ap75[-1],
        "last_aps": aps[-1],
        "last5_avg_ap": last5_avg_ap,
        "rollback": rollback,
    }


def main():
    parser = argparse.ArgumentParser("Analyze training stability from log.txt files")
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="One or more log.txt paths",
    )
    parser.add_argument(
        "--baseline-log",
        default=None,
        help="Optional baseline log path for delta_best_ap reporting",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional output json path for machine-readable summary",
    )
    args = parser.parse_args()

    summaries = []
    for log_path in args.logs:
        summaries.append(summarize_log(log_path))

    baseline_best_ap = None
    if args.baseline_log:
        baseline_best_ap = summarize_log(args.baseline_log)["best_ap"]
        for summary in summaries:
            summary["delta_best_ap_vs_baseline"] = summary["best_ap"] - baseline_best_ap

    print("=== Stability Summary ===")
    for summary in summaries:
        print(f"log: {summary['log_path']}")
        print(
            "  best_ap={:.6f} (e{}), last_ap={:.6f}, last5_avg_ap={:.6f}, rollback={:.6f}".format(
                summary["best_ap"],
                summary["best_epoch"],
                summary["last_ap"],
                summary["last5_avg_ap"],
                summary["rollback"],
            )
        )
        print(
            "  best_ap50={:.6f}, best_ap75={:.6f}, best_aps={:.6f}".format(
                summary["best_ap50"], summary["best_ap75"], summary["best_aps"]
            )
        )
        if baseline_best_ap is not None:
            print(
                "  delta_best_ap_vs_baseline={:.6f}".format(
                    summary["delta_best_ap_vs_baseline"]
                )
            )
        print("---")

    if args.save_json:
        payload = {
            "baseline_best_ap": baseline_best_ap,
            "summaries": summaries,
        }
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved json report to: {save_path}")


if __name__ == "__main__":
    main()
