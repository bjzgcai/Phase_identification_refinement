#!/usr/bin/env python3
"""Analyze stage1 top-1/top-10 space-group and crystal-system accuracy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

CRYSTAL_SYSTEMS = [
    ("triclinic", "三斜晶系"),
    ("monoclinic", "单斜晶系"),
    ("orthorhombic", "正交晶系"),
    ("tetragonal", "四方晶系"),
    ("trigonal", "三方晶系"),
    ("hexagonal", "六方晶系"),
    ("cubic", "立方晶系"),
]


def crystal_system_from_number(number: int) -> str:
    if 1 <= number <= 2:
        return "triclinic"
    if 3 <= number <= 15:
        return "monoclinic"
    if 16 <= number <= 74:
        return "orthorhombic"
    if 75 <= number <= 142:
        return "tetragonal"
    if 143 <= number <= 167:
        return "trigonal"
    if 168 <= number <= 194:
        return "hexagonal"
    if 195 <= number <= 230:
        return "cubic"
    return "unknown"


def label_to_mpid(label: str | int, entries: dict) -> str | None:
    item = entries.get(str(int(label)))
    if not item:
        return None
    value = str(item.get("value", ""))
    if value.endswith(".cif"):
        value = value[:-4]
    return value


def sg_info_from_mpid(mpid: str | None, mp_spacegroups: dict) -> dict:
    info = mp_spacegroups.get(mpid) if mpid else None
    if not info:
        return {
            "mpid": mpid,
            "space_group": "P1",
            "space_group_number": 1,
            "crystal_system": "triclinic",
            "missing": True,
        }

    space_group = info.get("space_group") or {}
    number = space_group.get("number")
    if number is None:
        return {
            "mpid": mpid,
            "space_group": "P1",
            "space_group_number": 1,
            "crystal_system": "triclinic",
            "missing": True,
        }

    number = int(number)
    crystal_system = (info.get("crystal_system") or {}).get("symbol")
    return {
        "mpid": mpid,
        "space_group": space_group.get("symbol") or "",
        "space_group_number": number,
        "crystal_system": crystal_system or crystal_system_from_number(number),
        "missing": False,
    }


def init_stats() -> dict:
    return {"n": 0, "sg_top1": 0, "sg_top10": 0, "cs_top1": 0, "cs_top10": 0}


def accuracy(correct: int, total: int) -> float | None:
    return correct / total if total else None


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.4f}%"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp-rank-csv", type=Path, default=Path("temp_rank_0.csv"))
    parser.add_argument("--entries-json", type=Path, default=Path("../data/entries_dict.json"))
    parser.add_argument("--mp-spacegroup-json", type=Path, default=Path("../data/mp_spacegroup.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    parser.add_argument(
        "--write-sample-details",
        action="store_true",
        help="Write one row per sample. This file is large for the full stage1 set.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries = json.load(args.entries_json.open(encoding="utf-8"))
    mp_spacegroups = json.load(args.mp_spacegroup_json.open(encoding="utf-8"))

    overall = init_stats()
    by_system = defaultdict(init_stats)
    missing_true = Counter()
    missing_pred = Counter()
    missing_label = Counter()
    forced_pred_hits = 0
    forced_top1_hits = 0
    bad_top10_len = 0
    sample_details = []

    with args.temp_rank_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "top10_idx" not in (reader.fieldnames or []):
            raise ValueError(f"{args.temp_rank_csv} does not contain a top10_idx column")

        for row in reader:
            true_mpid = label_to_mpid(row["y"], entries)
            if true_mpid is None:
                missing_label["true_label"] += 1
            true_info = sg_info_from_mpid(true_mpid, mp_spacegroups)
            if true_info["missing"]:
                missing_true[true_info["mpid"]] += 1

            labels = [item for item in str(row["top10_idx"]).split(";") if item != ""]
            if len(labels) < 10:
                bad_top10_len += 1

            predictions = []
            for label in labels[:10]:
                pred_mpid = label_to_mpid(label, entries)
                if pred_mpid is None:
                    missing_label["pred_label"] += 1
                pred_info = sg_info_from_mpid(pred_mpid, mp_spacegroups)
                if pred_info["missing"]:
                    missing_pred[pred_info["mpid"]] += 1
                predictions.append(pred_info)

            if not predictions:
                continue

            top1 = predictions[0]
            sg_top1 = top1["missing"] or top1["space_group_number"] == true_info["space_group_number"]
            sg_top10 = any(
                pred["missing"] or pred["space_group_number"] == true_info["space_group_number"]
                for pred in predictions
            )
            cs_top1 = top1["missing"] or top1["crystal_system"] == true_info["crystal_system"]
            cs_top10 = any(
                pred["missing"] or pred["crystal_system"] == true_info["crystal_system"]
                for pred in predictions
            )

            if top1["missing"]:
                forced_top1_hits += 1
            forced_pred_hits += sum(1 for pred in predictions if pred["missing"])

            for stats in (overall, by_system[true_info["crystal_system"]]):
                stats["n"] += 1
                stats["sg_top1"] += int(sg_top1)
                stats["sg_top10"] += int(sg_top10)
                stats["cs_top1"] += int(cs_top1)
                stats["cs_top10"] += int(cs_top10)

            if args.write_sample_details:
                sample_details.append(
                    {
                        "y": row["y"],
                        "true_mpid": true_mpid,
                        "true_space_group": true_info["space_group"],
                        "true_space_group_number": true_info["space_group_number"],
                        "true_crystal_system": true_info["crystal_system"],
                        "top1_mpid": top1["mpid"],
                        "top1_space_group": top1["space_group"],
                        "top1_space_group_number": top1["space_group_number"],
                        "top1_crystal_system": top1["crystal_system"],
                        "top1_forced_p1": top1["missing"],
                        "space_group_top1_correct": sg_top1,
                        "space_group_top10_correct": sg_top10,
                        "crystal_system_top1_correct": cs_top1,
                        "crystal_system_top10_correct": cs_top10,
                    }
                )

    by_system_rows = []
    system_name = dict(CRYSTAL_SYSTEMS)
    for system, system_cn in CRYSTAL_SYSTEMS:
        stats = by_system.get(system, init_stats())
        row = {"true_crystal_system_cn": system_cn, "true_crystal_system": system, "n": stats["n"]}
        for key in ["sg_top1", "sg_top10", "cs_top1", "cs_top10"]:
            row[f"{key}_correct"] = stats[key]
            row[f"{key}_total"] = stats["n"]
            row[f"{key}_accuracy"] = accuracy(stats[key], stats["n"])
        by_system_rows.append(row)

    if any(system not in system_name for system in by_system):
        for system, stats in sorted(by_system.items()):
            if system in system_name:
                continue
            row = {"true_crystal_system_cn": system, "true_crystal_system": system, "n": stats["n"]}
            for key in ["sg_top1", "sg_top10", "cs_top1", "cs_top10"]:
                row[f"{key}_correct"] = stats[key]
                row[f"{key}_total"] = stats["n"]
                row[f"{key}_accuracy"] = accuracy(stats[key], stats["n"])
            by_system_rows.append(row)

    overall_summary = {
        "samples": overall["n"],
        "bad_top10_len": bad_top10_len,
        "missing_true_mpid_or_sg_occurrences": sum(missing_true.values()),
        "missing_true_mpid_or_sg_unique": len(missing_true),
        "missing_pred_mpid_or_sg_occurrences": sum(missing_pred.values()),
        "missing_pred_mpid_or_sg_unique": len(missing_pred),
        "forced_missing_pred_candidates_counted_correct": forced_pred_hits,
        "forced_missing_top1_counted_correct": forced_top1_hits,
        "missing_label_mapping": dict(missing_label),
    }
    for key in ["sg_top1", "sg_top10", "cs_top1", "cs_top10"]:
        overall_summary[key] = {
            "correct": overall[key],
            "total": overall["n"],
            "accuracy": accuracy(overall[key], overall["n"]),
        }

    summary = {
        "inputs": {
            "temp_rank_csv": str(args.temp_rank_csv),
            "entries_json": str(args.entries_json),
            "mp_spacegroup_json": str(args.mp_spacegroup_json),
        },
        "policy": {
            "label_mapping": "Label -> entries_dict value with .cif suffix removed -> mp-id.",
            "match": "Compare space groups by space_group.number; compare crystal systems by symbol.",
            "missing_prediction_rule": "If a predicted mp-id has no space group/crystal system, assign P1/triclinic and count that prediction as correct.",
        },
        "overall": overall_summary,
        "by_crystal_system": by_system_rows,
    }

    (args.output_dir / "stage1_spacegroup_accuracy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "stage1_spacegroup_accuracy_by_crystal_system.csv", list(by_system_rows[0].keys()), by_system_rows)
    write_csv(
        args.output_dir / "stage1_missing_true_mpid_or_spacegroup.csv",
        ["mpid", "count"],
        [{"mpid": key, "count": value} for key, value in missing_true.most_common()],
    )
    write_csv(
        args.output_dir / "stage1_missing_pred_mpid_or_spacegroup.csv",
        ["mpid", "count"],
        [{"mpid": key, "count": value} for key, value in missing_pred.most_common()],
    )
    if args.write_sample_details and sample_details:
        write_csv(args.output_dir / "stage1_spacegroup_accuracy_sample_details.csv", list(sample_details[0].keys()), sample_details)

    lines = [
        "# Stage1 Space Group Accuracy",
        "",
        "Policy: labels are mapped through data/entries_dict.json; predicted mp-ids missing from mp_spacegroup.json are assigned P1 and counted as correct.",
        "",
        "## Overall",
        "",
        "| Metric | Correct/Total | Accuracy |",
        "|---|---:|---:|",
    ]
    metric_labels = [
        ("Space group Top-1", "sg_top1"),
        ("Space group Top-10", "sg_top10"),
        ("Crystal system Top-1", "cs_top1"),
        ("Crystal system Top-10", "cs_top10"),
    ]
    for label, key in metric_labels:
        stats = overall_summary[key]
        lines.append(f"| {label} | {stats['correct']}/{stats['total']} | {pct(stats['accuracy'])} |")
    lines.extend([
        "",
        "## By Crystal System",
        "",
        "| True system | N | SG Top-1 | SG Top-10 | CS Top-1 | CS Top-10 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in by_system_rows:
        n = row["n"]
        lines.append(
            f"| {row['true_crystal_system_cn']} | {n} | "
            f"{row['sg_top1_correct']}/{n} ({pct(row['sg_top1_accuracy'])}) | "
            f"{row['sg_top10_correct']}/{n} ({pct(row['sg_top10_accuracy'])}) | "
            f"{row['cs_top1_correct']}/{n} ({pct(row['cs_top1_accuracy'])}) | "
            f"{row['cs_top10_correct']}/{n} ({pct(row['cs_top10_accuracy'])}) |"
        )
    lines.extend([
        "",
        "## Checks",
        "",
        f"- Samples: {overall_summary['samples']}",
        f"- Rows with fewer than 10 top-k labels: {overall_summary['bad_top10_len']}",
        f"- Missing predicted mp-id/space-group occurrences counted as correct: {overall_summary['forced_missing_pred_candidates_counted_correct']}",
        f"- Missing top-1 prediction occurrences counted as correct: {overall_summary['forced_missing_top1_counted_correct']}",
        f"- Missing label mappings: {overall_summary['missing_label_mapping']}",
        "",
    ])
    (args.output_dir / "stage1_spacegroup_accuracy_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote analysis outputs to {args.output_dir}")
    for label, key in metric_labels:
        stats = overall_summary[key]
        print(f"{label}: {stats['correct']}/{stats['total']} ({pct(stats['accuracy'])})")


if __name__ == "__main__":
    main()
