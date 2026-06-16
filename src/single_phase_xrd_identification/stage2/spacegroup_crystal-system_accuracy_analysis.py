#!/usr/bin/env python3
"""Evaluate space-group and crystal-system top-k accuracy for stage2 output."""

from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from pymatgen.symmetry.groups import SpaceGroup


CRYSTAL_SYSTEMS = [
    ("triclinic", "三斜晶系"),
    ("monoclinic", "单斜晶系"),
    ("orthorhombic", "正交晶系"),
    ("tetragonal", "四方晶系"),
    ("trigonal", "三方晶系"),
    ("hexagonal", "六方晶系"),
    ("cubic", "立方晶系"),
]

FORCED_P1_MPIDS = {"mp-1255268", "mp-1247833"}

SYMBOL_FIXES = {
    "Fm3m": "Fm-3m",
    "Pa3": "Pa-3",
    "P2_1/a": "P2_1/c",
    "P2_1/n": "P2_1/c",
}

MANUAL_SPACEGROUP_NUMBERS = {
    "C-1": 2,
    "Fm3m": 225,
    "Fm-3m": 225,
    "Pa3": 205,
    "Pa-3": 205,
    "P2_1/a": 14,
    "P2_1/n": 14,
    "P1": 1,
    "P 1": 1,
}


def normalize_symbol(symbol: str) -> str:
    symbol = (symbol or "").strip().replace(" ", "")
    return SYMBOL_FIXES.get(symbol, symbol)


def spacegroup_number(symbol: str) -> int:
    fixed = normalize_symbol(symbol)
    try:
        return int(SpaceGroup(fixed).int_number)
    except Exception:
        if symbol in MANUAL_SPACEGROUP_NUMBERS:
            return MANUAL_SPACEGROUP_NUMBERS[symbol]
        if fixed in MANUAL_SPACEGROUP_NUMBERS:
            return MANUAL_SPACEGROUP_NUMBERS[fixed]
        raise


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


def read_true_spacegroups(exp_data_dir: Path, rruff_ids: set[str]) -> dict[str, dict]:
    true_spacegroups = {}
    for rruff_id in sorted(rruff_ids):
        cif_path = exp_data_dir / f"{rruff_id}_CIF.txt"
        text = cif_path.read_text(errors="ignore")
        match = re.search(r"SPACE GROUP:\s*([^\n\r]+)", text)
        if not match:
            raise ValueError(f"Cannot find SPACE GROUP line in {cif_path}")
        symbol = match.group(1).strip()
        number = spacegroup_number(symbol)
        true_spacegroups[rruff_id] = {
            "symbol": symbol,
            "number": number,
            "crystal_system": crystal_system_from_number(number),
        }
    return true_spacegroups


def accuracy(records: list[dict], key: str) -> dict:
    total = len(records)
    correct = sum(1 for row in records if row[key])
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.2f}%"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top10-csv",
        default="stage2/analysis_results/top10_candidates.csv",
        type=Path,
    )
    parser.add_argument("--exp-data-dir", default="data/Exp_data", type=Path)
    parser.add_argument("--mp-spacegroup-json", default="data/mp_spacegroup.json", type=Path)
    parser.add_argument("--output-dir", default="stage2/analysis", type=Path)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.top10_csv.open(encoding="utf-8-sig") as handle:
        candidate_rows = list(csv.DictReader(handle))
    with args.mp_spacegroup_json.open(encoding="utf-8") as handle:
        mp_spacegroups = json.load(handle)

    by_rruff_id = defaultdict(list)
    for row in candidate_rows:
        by_rruff_id[row["rruff_id"]].append(row)
    for rows in by_rruff_id.values():
        rows.sort(key=lambda row: int(row["rank"]))

    true_spacegroups = read_true_spacegroups(args.exp_data_dir, set(by_rruff_id))
    records = []
    forced_p1_rows = []
    missing_mp_rows = []

    for rruff_id, candidates in sorted(by_rruff_id.items()):
        truth = true_spacegroups[rruff_id]
        predictions = []
        for candidate in candidates:
            mpid = candidate["pred_mpid"]
            mp_info = mp_spacegroups.get(mpid)
            forced_p1 = mpid in FORCED_P1_MPIDS and mp_info is None
            if forced_p1:
                pred = {
                    "rank": int(candidate["rank"]),
                    "mpid": mpid,
                    "symbol": "P1",
                    "number": 1,
                    "crystal_system": "triclinic",
                    "forced_correct": True,
                }
                forced_p1_rows.append(
                    {
                        "rruff_id": rruff_id,
                        "rank": candidate["rank"],
                        "pred_mpid": mpid,
                        "assigned_space_group": "P1",
                        "assigned_space_group_number": 1,
                    }
                )
            elif mp_info:
                pred_number = int(mp_info["space_group"]["number"])
                pred = {
                    "rank": int(candidate["rank"]),
                    "mpid": mpid,
                    "symbol": mp_info["space_group"]["symbol"],
                    "number": pred_number,
                    "crystal_system": crystal_system_from_number(pred_number),
                    "forced_correct": False,
                }
            else:
                missing_mp_rows.append(
                    {
                        "rruff_id": rruff_id,
                        "rank": candidate["rank"],
                        "pred_mpid": mpid,
                    }
                )
                continue
            predictions.append(pred)

        if not predictions:
            raise ValueError(f"No usable predictions for {rruff_id}")

        top1 = predictions[0]
        sg_top1 = top1["number"] == truth["number"] or top1["forced_correct"]
        sg_top10 = any(
            pred["number"] == truth["number"] or pred["forced_correct"]
            for pred in predictions[:10]
        )
        cs_top1 = (
            top1["crystal_system"] == truth["crystal_system"] or top1["forced_correct"]
        )
        cs_top10 = any(
            pred["crystal_system"] == truth["crystal_system"] or pred["forced_correct"]
            for pred in predictions[:10]
        )
        records.append(
            {
                "rruff_id": rruff_id,
                "true_space_group": truth["symbol"],
                "true_space_group_number": truth["number"],
                "true_crystal_system": truth["crystal_system"],
                "top1_mpid": top1["mpid"],
                "top1_space_group": top1["symbol"],
                "top1_space_group_number": top1["number"],
                "top1_crystal_system": top1["crystal_system"],
                "space_group_top1_correct": sg_top1,
                "space_group_top10_correct": sg_top10,
                "crystal_system_top1_correct": cs_top1,
                "crystal_system_top10_correct": cs_top10,
            }
        )

    overall = {
        "candidate_rows": len(candidate_rows),
        "unique_rruff_ids": len(by_rruff_id),
        "evaluated_samples": len(records),
        "forced_p1_candidates": len(forced_p1_rows),
        "unresolved_missing_mp_candidates": len(missing_mp_rows),
        "space_group_top1": accuracy(records, "space_group_top1_correct"),
        "space_group_top10": accuracy(records, "space_group_top10_correct"),
        "crystal_system_top1": accuracy(records, "crystal_system_top1_correct"),
        "crystal_system_top10": accuracy(records, "crystal_system_top10_correct"),
    }

    by_system_rows = []
    for system, system_cn in CRYSTAL_SYSTEMS:
        system_records = [row for row in records if row["true_crystal_system"] == system]
        row = {
            "true_crystal_system_cn": system_cn,
            "true_crystal_system": system,
            "n": len(system_records),
        }
        for key, record_key in [
            ("space_group_top1", "space_group_top1_correct"),
            ("space_group_top10", "space_group_top10_correct"),
            ("crystal_system_top1", "crystal_system_top1_correct"),
            ("crystal_system_top10", "crystal_system_top10_correct"),
        ]:
            stats = accuracy(system_records, record_key)
            row[f"{key}_correct"] = stats["correct"]
            row[f"{key}_total"] = stats["total"]
            row[f"{key}_accuracy"] = stats["accuracy"]
        by_system_rows.append(row)

    distribution_rows = []
    counter = Counter(
        (
            row["true_space_group"],
            row["true_space_group_number"],
            row["true_crystal_system"],
        )
        for row in records
    )
    for (symbol, number, system), count in sorted(
        counter.items(), key=lambda item: (item[0][2], item[0][1], item[0][0])
    ):
        distribution_rows.append(
            {
                "true_space_group": symbol,
                "true_space_group_number": number,
                "true_crystal_system": system,
                "n": count,
            }
        )

    summary = {
        "inputs": {
            "top10_csv": str(args.top10_csv),
            "exp_data_dir": str(args.exp_data_dir),
            "mp_spacegroup_json": str(args.mp_spacegroup_json),
        },
        "policy": {
            "space_group_match": "Compare by international space group number.",
            "forced_p1_mpids": sorted(FORCED_P1_MPIDS),
            "forced_p1_rule": "If a forced mp-id is missing from mp_spacegroup.json, assign P1 and count it as correct.",
        },
        "overall": overall,
        "by_crystal_system": by_system_rows,
    }

    (args.output_dir / "spacegroup_accuracy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(
        args.output_dir / "spacegroup_accuracy_by_crystal_system.csv",
        list(by_system_rows[0].keys()),
        by_system_rows,
    )
    write_csv(
        args.output_dir / "spacegroup_accuracy_sample_details.csv",
        list(records[0].keys()),
        records,
    )
    write_csv(
        args.output_dir / "true_spacegroup_distribution.csv",
        list(distribution_rows[0].keys()),
        distribution_rows,
    )
    write_csv(
        args.output_dir / "forced_p1_candidates.csv",
        [
            "rruff_id",
            "rank",
            "pred_mpid",
            "assigned_space_group",
            "assigned_space_group_number",
        ],
        forced_p1_rows,
    )
    if missing_mp_rows:
        write_csv(
            args.output_dir / "unresolved_missing_mp_candidates.csv",
            ["rruff_id", "rank", "pred_mpid"],
            missing_mp_rows,
        )

    report_lines = [
        "# Stage2 Space Group Accuracy",
        "",
        "Policy: compare space groups by international number. Missing forced mp-ids "
        "`mp-1247833` and `mp-1255268` are assigned to `P1` and counted as correct.",
        "",
        "## Overall",
        "",
        "| Metric | Correct/Total | Accuracy |",
        "|---|---:|---:|",
    ]
    for label, key in [
        ("Space group Top-1", "space_group_top1"),
        ("Space group Top-10", "space_group_top10"),
        ("Crystal system Top-1", "crystal_system_top1"),
        ("Crystal system Top-10", "crystal_system_top10"),
    ]:
        stats = overall[key]
        report_lines.append(
            f"| {label} | {stats['correct']}/{stats['total']} | {pct(stats['accuracy'])} |"
        )
    report_lines.extend(
        [
            "",
            "## By Crystal System",
            "",
            "| True system | N | SG Top-1 | SG Top-10 | CS Top-1 | CS Top-10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in by_system_rows:
        report_lines.append(
            "| {name} | {n} | {sg1}/{n} ({sg1a}) | {sg10}/{n} ({sg10a}) | "
            "{cs1}/{n} ({cs1a}) | {cs10}/{n} ({cs10a}) |".format(
                name=row["true_crystal_system_cn"],
                n=row["n"],
                sg1=row["space_group_top1_correct"],
                sg1a=pct(row["space_group_top1_accuracy"]),
                sg10=row["space_group_top10_correct"],
                sg10a=pct(row["space_group_top10_accuracy"]),
                cs1=row["crystal_system_top1_correct"],
                cs1a=pct(row["crystal_system_top1_accuracy"]),
                cs10=row["crystal_system_top10_correct"],
                cs10a=pct(row["crystal_system_top10_accuracy"]),
            )
        )
    report_lines.extend(
        [
            "",
            "## Forced P1 Candidates",
            "",
            "| RRUFF ID | Rank | MP ID |",
            "|---|---:|---|",
        ]
    )
    for row in forced_p1_rows:
        report_lines.append(f"| {row['rruff_id']} | {row['rank']} | {row['pred_mpid']} |")
    report_lines.append("")
    (args.output_dir / "spacegroup_accuracy_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"Wrote analysis outputs to {args.output_dir}")
    for key in [
        "space_group_top1",
        "space_group_top10",
        "crystal_system_top1",
        "crystal_system_top10",
    ]:
        stats = overall[key]
        print(f"{key}: {stats['correct']}/{stats['total']} ({pct(stats['accuracy'])})")


if __name__ == "__main__":
    main()
