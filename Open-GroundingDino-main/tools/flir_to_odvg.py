import argparse
import json
from collections import defaultdict
from pathlib import Path


def coco_to_xyxy_clipped(bbox, width, height):
    x, y, w, h = bbox
    x1 = max(0.0, min(float(width), float(x)))
    y1 = max(0.0, min(float(height), float(y)))
    x2 = max(0.0, min(float(width), float(x + w)))
    y2 = max(0.0, min(float(height), float(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def coco_to_xywh_clipped(bbox, width, height):
    clipped = coco_to_xyxy_clipped(bbox, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def category_name_map(coco_json):
    return {int(item["id"]): item["name"] for item in coco_json["categories"]}


def build_image_records(coco_json):
    return {int(img["id"]): img for img in coco_json["images"]}


def build_annotations_by_image(coco_json):
    grouped = defaultdict(list)
    for ann in coco_json["annotations"]:
        grouped[int(ann["image_id"])].append(ann)
    return grouped


def collect_used_categories(coco_json, drop_category_names):
    id_to_name = category_name_map(coco_json)
    keep = defaultdict(int)
    for ann in coco_json["annotations"]:
        cat_id = int(ann["category_id"])
        cat_name = id_to_name[cat_id]
        if cat_name in drop_category_names:
            continue
        keep[cat_id] += 1
    return keep


def build_label_mapping(category_counts, id_to_name):
    sorted_ids = sorted(category_counts.keys())
    old_to_new = {}
    new_to_name = {}
    for idx, cat_id in enumerate(sorted_ids):
        old_to_new[cat_id] = idx
        new_to_name[str(idx)] = id_to_name[cat_id]
    return old_to_new, new_to_name


def dump_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")


def build_odvg_records(coco_json, old_to_new, id_to_name):
    images = build_image_records(coco_json)
    anns_by_image = build_annotations_by_image(coco_json)
    records = []
    skipped_bad_bbox = 0

    for image_id, img in images.items():
        width = int(img["width"])
        height = int(img["height"])
        instances = []
        for ann in anns_by_image.get(image_id, []):
            old_cat = int(ann["category_id"])
            if old_cat not in old_to_new:
                continue
            bbox_xyxy = coco_to_xyxy_clipped(ann["bbox"], width, height)
            if bbox_xyxy is None:
                skipped_bad_bbox += 1
                continue
            new_label = old_to_new[old_cat]
            instances.append(
                {
                    "bbox": bbox_xyxy,
                    "label": new_label,
                    "category": id_to_name[old_cat],
                }
            )
        records.append(
            {
                "filename": img["file_name"],
                "height": height,
                "width": width,
                "detection": {"instances": instances},
            }
        )
    return records, skipped_bad_bbox


def build_remapped_coco(coco_json, old_to_new, id_to_name):
    images = build_image_records(coco_json)
    output = {
        "info": coco_json.get("info", {}),
        "licenses": coco_json.get("licenses", []),
        "images": coco_json["images"],
        "annotations": [],
        "categories": [],
    }
    for cat_id, new_id in sorted(old_to_new.items(), key=lambda x: x[1]):
        output["categories"].append(
            {"id": new_id, "name": id_to_name[cat_id], "supercategory": "flir"}
        )

    for ann in coco_json["annotations"]:
        old_cat = int(ann["category_id"])
        if old_cat not in old_to_new:
            continue
        image_id = int(ann["image_id"])
        # Need image size to clip bboxes for pycocotools stability.
        img = images[image_id]
        bbox_xywh = coco_to_xywh_clipped(ann["bbox"], img["width"], img["height"])
        if bbox_xywh is None:
            continue
        out_ann = dict(ann)
        out_ann["category_id"] = old_to_new[old_cat]
        out_ann["bbox"] = bbox_xywh
        out_ann["area"] = round(bbox_xywh[2] * bbox_xywh[3], 4)
        output["annotations"].append(out_ann)
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        "Convert FLIR ADAS 1.3 COCO JSON to Open-GroundingDINO ODVG."
    )
    parser.add_argument(
        "--flir-root",
        type=str,
        required=True,
        help="Path to FLIR_ADAS_1_3 root containing train/ and val/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for converted ODVG/coco files.",
    )
    parser.add_argument(
        "--label-source",
        type=str,
        choices=["train", "trainval"],
        default="train",
        help="Split(s) used to build class mapping.",
    )
    parser.add_argument(
        "--drop-category-names",
        type=str,
        nargs="*",
        default=["empty"],
        help="Category names to ignore (default drops 'empty').",
    )
    parser.add_argument(
        "--write-val-odvg",
        action="store_true",
        help="Write val_odvg.jsonl for debugging/inspection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    flir_root = Path(args.flir_root)
    out_dir = Path(args.output_dir)

    train_json = load_coco(flir_root / "train" / "thermal_annotations.json")
    val_json = load_coco(flir_root / "val" / "thermal_annotations.json")
    train_id_to_name = category_name_map(train_json)
    train_id_to_name.update(category_name_map(val_json))
    train_counts = collect_used_categories(train_json, set(args.drop_category_names))
    val_counts = collect_used_categories(val_json, set(args.drop_category_names))

    counts = dict(train_counts)
    if args.label_source == "trainval":
        for key, value in val_counts.items():
            counts[key] = counts.get(key, 0) + value

    old_to_new, new_to_name = build_label_mapping(counts, train_id_to_name)
    class_stats = {
        "label_source": args.label_source,
        "num_classes": len(old_to_new),
        "classes": [
            {
                "new_label": old_to_new[old_id],
                "old_category_id": old_id,
                "category_name": train_id_to_name[old_id],
                "train_count": train_counts.get(old_id, 0),
                "val_count": val_counts.get(old_id, 0),
            }
            for old_id in sorted(old_to_new.keys())
        ],
    }

    train_odvg, train_bad_bbox = build_odvg_records(train_json, old_to_new, train_id_to_name)
    val_odvg, val_bad_bbox = build_odvg_records(val_json, old_to_new, train_id_to_name)
    val_coco_remapped = build_remapped_coco(val_json, old_to_new, train_id_to_name)

    dump_json(out_dir / "label_map.json", new_to_name)
    dump_json(out_dir / "class_stats.json", class_stats)
    dump_jsonl(out_dir / "train_odvg.jsonl", train_odvg)
    dump_json(out_dir / "val_coco_remapped.json", val_coco_remapped)
    if args.write_val_odvg:
        dump_jsonl(out_dir / "val_odvg.jsonl", val_odvg)

    summary = {
        "train_images": len(train_json["images"]),
        "val_images": len(val_json["images"]),
        "train_odvg_items": len(train_odvg),
        "val_odvg_items": len(val_odvg),
        "num_classes": len(new_to_name),
        "train_skipped_bad_bbox": train_bad_bbox,
        "val_skipped_bad_bbox": val_bad_bbox,
        "dropped_categories": args.drop_category_names,
    }
    dump_json(out_dir / "conversion_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
