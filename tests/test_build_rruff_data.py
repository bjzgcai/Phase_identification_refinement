from pathlib import Path

import torch

from single_phase_xrd_identification.common.serialization import safe_torch_load
from single_phase_xrd_identification.refinement.build_rruff_data import parse_exp_cif_txt, read_rank_csv


def test_parse_exp_cif_txt_minimal_fixture(tmp_path: Path) -> None:
    exp_txt = tmp_path / "R000001_CIF.txt"
    exp_txt.write_text(
        "\n".join(
            [
                "CELL PARAMETERS: 5.0 5.0 5.0 90 90 90",
                "SPACE GROUP: P 1",
                "ATOM X Y Z OCCUPANCY ISOTROPIC",
                "Na1 0.0 0.0 0.0 1.0 0.5",
                "Cl1 0.5 0.5 0.5 1.0 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_exp_cif_txt(exp_txt)

    assert parsed.rruff_id == "R000001"
    assert parsed.cell == (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
    assert parsed.space_group == "P 1"
    assert [atom.element for atom in parsed.atoms] == ["Na", "Cl"]


def test_read_rank_csv_sorts_and_truncates(tmp_path: Path) -> None:
    rank_csv = tmp_path / "rank.csv"
    rank_csv.write_text(
        "rruff_id,rank,pred_mpid\n"
        "R1,2,mp-2\n"
        "R1,1,mp-1\n"
        "R1,3,mp-3\n"
        "R2,1,mp-9\n",
        encoding="utf-8",
    )

    ranked = read_rank_csv(rank_csv, top_k=2)

    assert ranked == {"R1": [(1, "mp-1"), (2, "mp-2")], "R2": [(1, "mp-9")]}


def test_safe_torch_load_state_dict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pth"
    torch.save({"linear.weight": torch.ones(1, 2)}, checkpoint)

    loaded = safe_torch_load(checkpoint, map_location="cpu")

    assert torch.equal(loaded["linear.weight"], torch.ones(1, 2))
