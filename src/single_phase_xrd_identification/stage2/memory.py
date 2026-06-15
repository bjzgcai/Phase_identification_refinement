# stage3/memory.py
# 一键生成 common/memory_bank.pkl
# - embedding：对 trainD.db(默认)/trainV.db 生成每条样本 embedding
# - 聚合：按 label 聚合（同 label 多条 embedding 全保留）
# - formula：优先使用 stage3/label_formula.csv（Label,formula），避免出现 Oxx 的错误公式
#
# 用法：在 stage3 目录下：
#   python memory.py
#
# 可选参数：
#   python memory.py --db ../data/trainD.db --model model_best.pth --out ../common/memory_bank.pkl
#   python memory.py --precision -1  # 完全不截断浮点

import os
import sys
import csv
import argparse
import pickle
from collections import defaultdict
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))

from single_phase_xrd_identification.common.model import PerceiverXRD
from single_phase_xrd_identification.common.dataset import XRDDataset


def get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def resolve_db_path() -> str:
    """
    默认优先 trainD.db，若不存在则回退 trainV.db
    """
    cand = [
        os.path.join(ROOT_DIR, "data", "trainD.db"),
        os.path.join(ROOT_DIR, "data", "trainV.db"),
    ]
    for p in cand:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"未找到 trainD.db/trainV.db，候选路径：{cand}")


def normalize_out_pkl_path(out_path: str) -> str:
    """
    兼容用户仍传 .json：统一输出为 .pkl
    - 传入 xxx.pkl  -> 原样
    - 传入 xxx.json -> 变成 xxx.pkl
    - 传入 xxx      -> 变成 xxx.pkl
    """
    out_path = out_path.strip()
    if not out_path:
        out_path = os.path.join(ROOT_DIR, "common", "memory_bank.pkl")

    base, ext = os.path.splitext(out_path)
    if ext.lower() == ".pkl":
        return out_path
    if ext:  # .json / .txt / 其他
        return base + ".pkl"
    return out_path + ".pkl"


def load_label_formula_csv(path: str) -> Dict[int, str]:
    """
    读取 label_formula.csv（你 match 脚本生成的那份）
    必须包含两列：Label, formula（大小写不敏感）
    """
    if not path or (not os.path.isfile(path)):
        print(f"⚠️ label_formula.csv 不存在：{path}")
        return {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        field_map = {k.strip().lower(): k for k in fields}

        if "label" not in field_map or "formula" not in field_map:
            raise ValueError(f"{path} 列名必须包含 Label 和 formula，实际列={fields}")

        k_label = field_map["label"]
        k_formula = field_map["formula"]

        out: Dict[int, str] = {}
        for row in reader:
            try:
                lab = int(str(row.get(k_label, "")).strip())
            except Exception:
                continue
            formula = str(row.get(k_formula, "")).strip()
            if formula:
                out[lab] = formula
        return out


def pick_formula_from_db_row(row) -> str:
    """
    db 兜底补齐：优先 key_value_pairs 里的 formula 字段，最后才用 row.formula
    """
    kv = row.key_value_pairs or {}
    for k in ["formula", "Formula", "chemical_formula", "chemical_formula_sum", "_chemical_formula_sum"]:
        v = kv.get(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()

    v = getattr(row, "formula", "") or ""
    return v.strip() if isinstance(v, str) else ""


def load_model(model_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = PerceiverXRD(num_classes=num_classes).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"⚠️ missing keys: {len(missing)}")
    if unexpected:
        print(f"⚠️ unexpected keys: {len(unexpected)}")
    model.eval()
    return model


def tensor_to_float_list(x: torch.Tensor, precision: Optional[int]) -> List[float]:
    arr = x.detach().float().cpu().numpy().tolist()
    if precision is None:
        return arr
    return [round(v, precision) for v in arr]


@torch.no_grad()
def build_memory_pkl(
    db_path: str,
    model_path: str,
    out_pkl: str,
    *,
    label_formula_csv: str,
    num_classes: int = 100315,
    batch_size: int = 64,
    num_workers: int = 8,
    precision: Optional[int] = 6,
):
    device = get_device()
    print(f"ROOT_DIR = {ROOT_DIR}")
    print(f"db_path  = {db_path}")
    print(f"model    = {model_path}")
    print(f"device   = {device}")
    print(f"out_pkl  = {out_pkl}")
    print(f"label_formula_csv = {label_formula_csv}")
    print(f"batch_size={batch_size} num_workers={num_workers} precision={precision}")

    # 1) db 行数 & ids
    from ase.db import connect
    with connect(db_path) as db:
        n = db.count()
    ids = list(range(1, n + 1))
    print(f"db rows = {n}")

    # 2) Dataset（复用训练读取/peak token；augment=False）
    ds = XRDDataset(
        db_path=db_path,
        ids=ids,
        augment=False,
        return_id_str=False,
        return_elem_onehot=True,   # forward 接口保持一致
        num_classes=num_classes,
        target_dim=3500,
        max_peaks=32,
        theta_min=10.0,
        theta_max=80.0,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # 3) Model
    model = load_model(model_path, num_classes=num_classes, device=device)

    # 4) label->formula：优先用 stage3/label_formula.csv
    formula_map: Dict[int, str] = {}
    print("Loading label->formula from CSV ...")
    formula_map.update(load_label_formula_csv(label_formula_csv))
    print(f"  loaded_from_csv = {len(formula_map)} labels")

    # 5) db 补齐缺失 label（只补空缺）
    print("Filling missing label->formula from DB (key_value_pairs fallback) ...")
    from ase.db import connect
    with connect(db_path) as db:
        for rid in tqdm(range(1, n + 1), desc="label->formula(db)", unit="row"):
            row = db.get(id=rid)
            kv = row.key_value_pairs or {}
            if "Label" not in kv:
                continue
            lab = int(kv["Label"])
            if lab in formula_map and formula_map[lab]:
                continue
            f = pick_formula_from_db_row(row)
            if f:
                formula_map[lab] = f
    print(f"  final_formula_map = {len(formula_map)} labels")

    # 6) 生成 embedding 并按 label 聚合（不限制数量）
    emb_lists = defaultdict(list)  # label -> list[embedding]

    use_amp = (device.type == "cuda")
    for x, peaks, elem_onehot, label in tqdm(loader, desc="Building embeddings", unit="batch"):
        x = x.to(device, non_blocking=True)
        peaks = peaks.to(device, non_blocking=True)
        elem_onehot = elem_onehot.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                emb = model(x, peaks, elem_onehot, return_embedding=True)  # [B,D]
        else:
            emb = model(x, peaks, elem_onehot, return_embedding=True)

        labels = label.detach().cpu().numpy().tolist()
        for i, lab in enumerate(labels):
            emb_lists[int(lab)].append(tensor_to_float_list(emb[i], precision))

    # 7) 写 pkl：label -> {formula, embeddings}
    memory: Dict[str, Dict[str, Any]] = {}
    for lab, embs in emb_lists.items():
        memory[str(lab)] = {
            "formula": formula_map.get(int(lab), ""),
            "embeddings": embs,
        }

    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(memory, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("✅ memory_bank.pkl 已生成")
    print(f"   labels_with_embeddings = {len(memory)}")
    total_emb = sum(len(v["embeddings"]) for v in memory.values())
    print(f"   total_embeddings       = {total_emb}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--db", type=str, default=resolve_db_path())
    ap.add_argument("--model", type=str, default=os.path.join(THIS_DIR, "model_best.pth"))

    # 默认改成 pkl（但仍兼容你手动传 json，会自动改成 pkl）
    ap.add_argument("--out", type=str, default=os.path.join(ROOT_DIR, "common", "memory_bank.pkl"))

    # ✅ 你的正确路径：stage3/label_formula.csv
    ap.add_argument("--label_formula_csv", type=str, default=os.path.join(THIS_DIR, "label_formula.csv"))

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--num_classes", type=int, default=100315)

    # 默认 6 位小数减少体积；完全不截断用 --precision -1
    ap.add_argument("--precision", type=int, default=6)

    args = ap.parse_args()
    precision = None if (args.precision is not None and args.precision < 0) else args.precision
    out_pkl = normalize_out_pkl_path(args.out)

    build_memory_pkl(
        db_path=args.db,
        model_path=args.model,
        out_pkl=out_pkl,
        label_formula_csv=args.label_formula_csv,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        precision=precision,
    )


if __name__ == "__main__":
    main()
