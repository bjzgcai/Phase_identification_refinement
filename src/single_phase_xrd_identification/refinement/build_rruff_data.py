#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
ATOM_LINE = re.compile(
    rf"^\s*([A-Za-z][A-Za-z0-9]*)\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$"
)


@dataclass(frozen=True)
class Atom:
    element: str
    x: float
    y: float
    z: float
    occupancy: float
    iso_b: float


@dataclass(frozen=True)
class ParsedExpCif:
    rruff_id: str
    cell: tuple[float, float, float, float, float, float]
    space_group: str
    atoms: list[Atom]


def clean_element(label: str) -> str:
    match = re.match(r"[A-Za-z]+", label.strip())
    if not match:
        raise ValueError(f"Cannot parse element from atom label {label!r}")
    raw = match.group(0)
    return raw[0].upper() + raw[1:].lower()


def parse_exp_cif_txt(path: Path) -> ParsedExpCif:
    rruff_id = path.name.replace("_CIF.txt", "")
    cell: tuple[float, float, float, float, float, float] | None = None
    space_group: str | None = None
    atoms: list[Atom] = []
    in_atoms = False

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "CELL PARAMETERS:" in line:
            match = re.search(r"CELL PARAMETERS:\s*([\d\.\s]+)", line)
            if match:
                values = tuple(float(x) for x in match.group(1).split()[:6])
                if len(values) == 6:
                    cell = values

        if "SPACE GROUP:" in line:
            match = re.search(r"SPACE GROUP:\s*(.+?)\s*$", line)
            if match:
                space_group = match.group(1).strip()

        if re.search(r"\bATOM\b.*\bOCCUPANCY\b", line):
            in_atoms = True
            continue

        if not in_atoms:
            continue

        match = ATOM_LINE.match(line)
        if match:
            atoms.append(
                Atom(
                    element=clean_element(match.group(1)),
                    x=float(match.group(2)),
                    y=float(match.group(3)),
                    z=float(match.group(4)),
                    occupancy=float(match.group(5)),
                    iso_b=float(match.group(6)),
                )
            )
        elif atoms and not line.strip():
            break

    if cell is None:
        raise ValueError(f"{path}: missing CELL PARAMETERS")
    if space_group is None:
        raise ValueError(f"{path}: missing SPACE GROUP")
    if not atoms:
        raise ValueError(f"{path}: missing ATOM table")
    return ParsedExpCif(rruff_id=rruff_id, cell=cell, space_group=space_group, atoms=atoms)


def coord_key(atom: Atom, ndigits: int = 6) -> tuple[float, float, float]:
    return (round(atom.x % 1.0, ndigits), round(atom.y % 1.0, ndigits), round(atom.z % 1.0, ndigits))


def normalize_composition(composition: dict[str, float]) -> str | dict[str, float]:
    cleaned = {el: min(max(float(occ), 0.0), 1.0) for el, occ in composition.items() if occ > 1e-12}
    total_occ = sum(cleaned.values())
    if total_occ > 1.0 + 1e-8:
        cleaned = {el: occ / total_occ for el, occ in cleaned.items()}
    if len(cleaned) == 1:
        element, occupancy = next(iter(cleaned.items()))
        return element if abs(occupancy - 1.0) <= 1e-8 else {element: occupancy}
    return cleaned


def merge_structure_sites(structure: Structure, ndigits: int = 4) -> Structure:
    grouped: dict[tuple[float, float, float], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for site in structure.sites:
        xyz = tuple(round(float(v) % 1.0, ndigits) for v in site.frac_coords)
        for el, occ in site.species.items():
            grouped[xyz][str(el)] += float(occ)

    coords: list[tuple[float, float, float]] = []
    species: list[str | dict[str, float]] = []
    for xyz, composition in grouped.items():
        coords.append(xyz)
        species.append(normalize_composition(dict(composition)))
    return Structure(structure.lattice, species, coords)


def build_structure(parsed: ParsedExpCif) -> Structure:
    grouped: dict[tuple[float, float, float], list[Atom]] = defaultdict(list)
    for atom in parsed.atoms:
        grouped[coord_key(atom)].append(atom)

    species: list[str | dict[str, float]] = []
    coords: list[tuple[float, float, float]] = []
    for xyz, atoms in grouped.items():
        composition: dict[str, float] = defaultdict(float)
        for atom in atoms:
            composition[atom.element] += atom.occupancy
        species.append(normalize_composition(dict(composition)))
        coords.append(xyz)

    lattice = Lattice.from_parameters(*parsed.cell)
    try:
        expanded = Structure.from_spacegroup(parsed.space_group, lattice, species, coords)
    except Exception:
        expanded = Structure(lattice, species, coords)
    return merge_structure_sites(expanded)


def write_rruff_cif(exp_txt: Path, out_path: Path, symprec: float) -> None:
    parsed = parse_exp_cif_txt(exp_txt)
    structure = build_structure(parsed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            CifWriter(structure, symprec=symprec).write_file(out_path)
        except Exception:
            CifWriter(structure, symprec=None).write_file(out_path)


def read_rank_csv(rank_csv: Path, top_k: int) -> dict[str, list[tuple[int, str]]]:
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with rank_csv.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"rruff_id", "rank", "pred_mpid"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{rank_csv}: missing columns {sorted(missing)}")
        for row in reader:
            rruff_id = row["rruff_id"].strip()
            mpid = row["pred_mpid"].strip()
            if not rruff_id or not mpid:
                continue
            grouped[rruff_id].append((int(row["rank"]), mpid))

    for rruff_id, items in grouped.items():
        items.sort(key=lambda x: x[0])
        grouped[rruff_id] = items[:top_k]
    return dict(sorted(grouped.items()))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_rruff_data(
    rank_csv: Path,
    mp_json: Path,
    exp_dir: Path,
    out_dir: Path,
    top_k: int,
    symprec: float,
    overwrite: bool,
) -> tuple[int, int, int]:
    ranked = read_rank_csv(rank_csv, top_k)
    with mp_json.open(encoding="utf-8") as fh:
        mp_data = json.load(fh)

    made = 0
    failed = 0
    skipped = 0
    manifest_rows = [["rruff_id", "kind", "rank", "source_id", "output_cif"]]
    missing_rows = [["rruff_id", "rank", "missing_mpid"]]

    for rruff_id, mpids in ranked.items():
        target_dir = out_dir / rruff_id
        if target_dir.exists() and not overwrite:
            skipped += 1
            continue

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            if overwrite:
                for old_cif in target_dir.glob("*.cif"):
                    old_cif.unlink()
            rruff_cif = target_dir / f"{rruff_id}.cif"
            write_rruff_cif(exp_dir / f"{rruff_id}_CIF.txt", rruff_cif, symprec=symprec)
            manifest_rows.append([rruff_id, "rruff", "", rruff_id, str(rruff_cif)])

            missing_for_rruff: list[tuple[int, str]] = []
            for rank, mpid in mpids:
                entry = mp_data.get(mpid)
                if entry is None or not entry.get("cif"):
                    missing_for_rruff.append((rank, mpid))
                    missing_rows.append([rruff_id, str(rank), mpid])
                    manifest_rows.append([rruff_id, "missing_mp", str(rank), mpid, ""])
                    continue
                mp_cif = target_dir / f"rank_{rank:02d}_{mpid}.cif"
                write_text(mp_cif, entry["cif"])
                manifest_rows.append([rruff_id, "mp", str(rank), mpid, str(mp_cif)])

            cif_count = len(list(target_dir.glob("*.cif")))
            if missing_for_rruff or cif_count != top_k + 1:
                missing_msg = ", ".join(f"rank {rank}: {mpid}" for rank, mpid in missing_for_rruff)
                raise RuntimeError(f"{rruff_id}: expected {top_k + 1} CIFs, found {cif_count}; missing {missing_msg}")
            made += 1
        except Exception as exc:
            failed += 1
            print(f"[failed] {rruff_id}: {exc}")

    manifest_path = out_dir / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(manifest_rows)

    missing_path = out_dir / "missing_mpids.csv"
    with missing_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(missing_rows)

    return made, skipped, failed


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build flat RRUFF_data/RRUFF-id CIF candidate folders from temp_rank_0.csv."
    )
    parser.add_argument("--rank-csv", type=Path, default=repo_root / "stage2/analysis_results/temp_rank_0.csv")
    parser.add_argument("--mp-json", type=Path, default=repo_root / "data/mp_spacegroup.json")
    parser.add_argument("--exp-dir", type=Path, default=repo_root / "data/Exp_data")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "RRUFF_data")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate folders that already exist.")
    args = parser.parse_args()

    made, skipped, failed = build_rruff_data(
        rank_csv=args.rank_csv,
        mp_json=args.mp_json,
        exp_dir=args.exp_dir,
        out_dir=args.out_dir,
        top_k=args.top_k,
        symprec=args.symprec,
        overwrite=args.overwrite,
    )
    print(f"done: made={made}, skipped_existing={skipped}, failed={failed}, out_dir={args.out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
