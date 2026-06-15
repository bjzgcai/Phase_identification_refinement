# verify_stage1.py
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))

import json
import math
import csv
import re
from typing import Dict, Any, List, Tuple

import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from ase.db import connect

# 引入你的模型和数据集（按你工程现状保持不变）
from single_phase_xrd_identification.common.dataset import XRDDataset
from single_phase_xrd_identification.common.model import PerceiverXRD

# -----------------------------
# 配置（按需改这几行）
# -----------------------------
DB_PATH = os.path.join(ROOT_DIR, "data", "valueV.db")
MODEL_PATH = os.path.join(CURRENT_DIR, "checkpoints_stage1", "model_best.pth")
NUM_CLASSES = 100315
BATCH_SIZE = 64
NUM_WORKERS = 4


OUT_DIR = os.path.join(CURRENT_DIR, "analysis_results")
TOPK_LIST = (1, 5, 10)
CACHE_FORMULA_JSON = os.path.join(OUT_DIR, "label_to_formula_cache.json")

# 输出文件名
TEMP_CSV_PATTERN = os.path.join(OUT_DIR, "temp_rank_{rank}.csv")
FULL_REPORT_CSV = os.path.join(OUT_DIR, "full_analysis_report.csv")
TOP5_CANDIDATES_CSV = os.path.join(OUT_DIR, "top5_candidates.csv")
CALIBRATION_CSV = os.path.join(OUT_DIR, "calibration_bins.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, "summary.json")
FIG_PATH = os.path.join(OUT_DIR, "combined_analysis.png")


def setup_ddp():
    # 只要 torchrun，就会有 RANK/WORLD_SIZE/LOCAL_RANK
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank_world() -> Tuple[int, int]:
    if is_ddp():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def get_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def load_model(model_path: str, device: torch.device) -> PerceiverXRD:
    model = PerceiverXRD(num_classes=NUM_CLASSES).to(device)

    # 更稳的 map_location（尤其你在 DDP 多卡上跑）
    if device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        map_loc = {"cuda:0": f"cuda:{local_rank}"}
        checkpoint = torch.load(model_path, map_location=map_loc)
    else:
        checkpoint = torch.load(model_path, map_location=device)

    # 兼容：既可能保存的是 state_dict，也可能保存的是 dict(model=..., optimizer=...)
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and ("model" in checkpoint):
        state_dict = checkpoint["model"]

    cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state, strict=True)
    model.eval()
    return model


def build_or_load_formula_map(db_path: str, cache_json: str, is_master: bool) -> Dict[int, str]:
    """
    Label -> formula 映射（用于 polymorph 分析、复杂度分析）。
    支持缓存：第一次构建慢，之后直接 load json。
    """
    os.makedirs(os.path.dirname(cache_json), exist_ok=True)

    if os.path.exists(cache_json):
        if is_master:
            print(f"✅ 读取化学式映射缓存: {cache_json}")
        with open(cache_json, "r") as f:
            raw = json.load(f)
        # json key 是 str，需要转回 int
        return {int(k): str(v) for k, v in raw.items()}

    if is_master:
        print("📖 正在从数据库构建化学式映射表（首次会比较慢）...")

    label_to_formula: Dict[int, str] = {}
    with connect(db_path) as db:
        total = db.count()
        iterator = tqdm(db.select(), total=total, desc="Building Formula Map") if is_master else db.select()
        for row in iterator:
            kv = row.key_value_pairs or {}
            label = kv.get("Label")
            # ase.db row 常见有 formula 字段
            formula = getattr(row, "formula", "")
            if label is not None:
                label_to_formula[int(label)] = str(formula)

    if is_master:
        with open(cache_json, "w") as f:
            json.dump({str(k): v for k, v in label_to_formula.items()}, f)
        print(f"💾 已保存化学式映射缓存: {cache_json}")

    return label_to_formula


def compute_elem_count(formula: Any) -> int:
    # 提取元素符号（粗略但够用）：Li2MnO3 -> ["Li","Mn","O"]
    elems = re.findall(r"[A-Z][a-z]?", str(formula))
    return len(set(elems))


def plot_science_figures(df: pd.DataFrame, calib: pd.DataFrame, comp_acc: pd.Series, poly_rate: float, ece: float):
    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1：Top-K
    ks = ["Top-1", "Top-5", "Top-10"]
    accs = [
        (df["y"] == df["pred"]).mean() * 100,
        df["hit_top5"].mean() * 100,
        df["hit_top10"].mean() * 100,
    ]
    axes[0].bar(ks, accs, alpha=0.85, edgecolor="black")
    for i, v in enumerate(accs):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=12)
    axes[0].set_title("Top-K Accuracy", fontsize=14)
    axes[0].set_ylabel("Accuracy (%)")

    # 图2：Reliability Diagram
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    axes[1].plot(calib["conf"], calib["acc"], "-o", label="Model")
    axes[1].fill_between(calib["conf"], calib["acc"], calib["conf"], alpha=0.12)
    axes[1].set_title(f"Confidence Calibration (ECE={ece:.4f})", fontsize=14)
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    # 图3：复杂度 vs acc
    comp_acc.plot(kind="line", marker="s", ax=axes[2], linewidth=2)
    axes[2].set_title("Accuracy vs. Chemical Complexity", fontsize=14)
    axes[2].set_xlabel("Number of Elements in Formula")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    plt.close(fig)


@torch.no_grad()
def main():
    setup_ddp()
    rank, world_size = get_rank_world()
    device = get_device()
    is_master = (rank == 0)

    os.makedirs(OUT_DIR, exist_ok=True)

    if is_master:
        print(f"🔍 [Verify Stage1] DB={DB_PATH} | model={MODEL_PATH}")
        print(f"   world_size={world_size} | batch={BATCH_SIZE} | device={device}")

    # 1) 模型
    model = load_model(MODEL_PATH, device)
    if is_master:
        print("✅ 模型加载成功")

    # 2) 数据
    with connect(DB_PATH) as db:
        total_rows = db.count()

    # 你现在 XRDDataset 的调用风格就是这样（保持一致）
    ds = XRDDataset(DB_PATH, list(range(1, total_rows + 1)), augment=False, num_classes=NUM_CLASSES)

    sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if is_ddp() else None
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False if sampler is not None else False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 3) 每个 rank：边跑边写 temp csv（不堆内存）
    temp_csv = TEMP_CSV_PATTERN.format(rank=rank)
    fcsv = open(temp_csv, "w", newline="")
    writer = csv.writer(fcsv)

    # temp csv 包含：用于 master 事后分析的列
    writer.writerow([
        "y", "pred",
        "conf", "true_prob", "margin", "nll",
        "hit_top5", "hit_top10",
        "top5_idx", "top5_logit"
    ])

    # 统计（用累计正确数，避免 batch 均值偏差）
    total = 0
    correct_top1 = 0
    hit_top5_total = 0
    hit_top10_total = 0
    nll_sum = 0.0

    iterator = tqdm(loader, desc=f"Rank {rank} verifying", unit="batch") if is_master else loader

    use_amp = (device.type == "cuda")
    for x, peaks, elem, y in iterator:
        x = x.to(device, non_blocking=True)
        peaks = peaks.to(device, non_blocking=True)
        elem = elem.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        # forward
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(x, peaks, elem)
        else:
            logits = model(x, peaks, elem)

        # topk on logits（只做 top10，既能算 top5/top10，也能拿 top5 候选）
        top10_vals, top10_idx = torch.topk(logits, k=10, dim=1)  # [B,10]
        pred = top10_idx[:, 0]
        logit_top1 = top10_vals[:, 0]
        logit_top2 = top10_vals[:, 1]

        # 用 logsumexp 计算概率相关量（避免对 100k 类做 softmax）
        lse = torch.logsumexp(logits, dim=1)  # [B]
        logit_true = logits.gather(1, y.view(-1, 1)).squeeze(1)  # [B]

        conf = torch.exp(logit_top1 - lse)         # top1 prob
        true_prob = torch.exp(logit_true - lse)    # prob of true class
        prob_top2 = torch.exp(logit_top2 - lse)
        margin = conf - prob_top2                 # prob margin
        nll = (lse - logit_true)                  # cross-entropy per sample

        # hits
        hit5 = (top10_idx[:, :5] == y.unsqueeze(1)).any(dim=1).int()
        hit10 = (top10_idx[:, :10] == y.unsqueeze(1)).any(dim=1).int()

        bs = y.size(0)
        total += bs
        correct_top1 += (pred == y).sum().item()
        hit_top5_total += hit5.sum().item()
        hit_top10_total += hit10.sum().item()
        nll_sum += nll.sum().item()

        # top5 candidates 输出（idx + logit）
        top5_idx = top10_idx[:, :5]
        top5_logit = top10_vals[:, :5]

        # 写 temp csv（每个样本一行）
        for i in range(bs):
            writer.writerow([
                int(y[i].item()),
                int(pred[i].item()),
                float(conf[i].item()),
                float(true_prob[i].item()),
                float(margin[i].item()),
                float(nll[i].item()),
                int(hit5[i].item()),
                int(hit10[i].item()),
                ";".join(str(int(v)) for v in top5_idx[i].tolist()),
                ";".join(f"{float(v):.6f}" for v in top5_logit[i].tolist()),
            ])

    fcsv.close()

    # 等所有 rank 写完 temp csv
    if is_ddp():
        dist.barrier()

    # 4) Master：合并、分析、出图、输出最终 CSV/JSON
    if is_master:
        if is_ddp():
            temp_files = [TEMP_CSV_PATTERN.format(rank=i) for i in range(world_size)]
        else:
            temp_files = [temp_csv]

        dfs = [pd.read_csv(p) for p in temp_files]
        df = pd.concat(dfs, ignore_index=True)

        # ---- 基础指标（严格按样本累计）----
        top1 = (df["y"] == df["pred"]).mean() * 100
        top5 = df["hit_top5"].mean() * 100
        top10 = df["hit_top10"].mean() * 100
        avg_nll = df["nll"].mean()

        # ---- Formula map（缓存）----
        formula_map = build_or_load_formula_map(DB_PATH, CACHE_FORMULA_JSON, is_master=True)
        df["formula"] = df["y"].map(formula_map)
        df["pred_formula"] = df["pred"].map(formula_map)

        # ---- 化学复杂度：元素个数 vs acc ----
        df["elem_count"] = df["formula"].apply(compute_elem_count)
        complexity_acc = df.groupby("elem_count").apply(lambda g: (g["y"] == g["pred"]).mean() * 100)

        # ---- Polymorph error + recoverable（top5 可救率）----
        wrong = df[df["y"] != df["pred"]]
        polymorph_errors = wrong[(wrong["formula"] == wrong["pred_formula"])]
        poly_rate = (len(polymorph_errors) / max(1, len(wrong))) * 100

        # “可救”：虽然 top1 错，但 top5 里包含真结构
        poly_recoverable = polymorph_errors[polymorph_errors["hit_top5"] == 1]
        poly_recoverable_rate = (len(poly_recoverable) / max(1, len(polymorph_errors))) * 100

        # ---- Calibration bins + ECE(10 bins) ----
        # 分箱：10 个 bins
        df["bin"] = pd.cut(df["conf"], bins=np.linspace(0, 1, 11), labels=False, include_lowest=True)
        calib = df.groupby("bin").apply(lambda g: pd.Series({
            "acc": float((g["y"] == g["pred"]).mean()),
            "conf": float(g["conf"].mean()) if len(g) > 0 else 0.0,
            "count": int(len(g)),
        })).reset_index()

        N = len(df)
        ece = 0.0
        for _, row in calib.iterrows():
            nb = row["count"]
            if nb <= 0:
                continue
            ece += (nb / N) * abs(row["acc"] - row["conf"])

        # 保存校准曲线数据
        calib.to_csv(CALIBRATION_CSV, index=False)

        # ---- 输出 full report ----
        # master 报告里保留 top5 字符串，方便 Stage-2 读
        df.to_csv(FULL_REPORT_CSV, index=False)

        # ---- 输出 top5_candidates.csv（给 Stage-2 用）----
        # 你后续可以直接读这个文件来跑候选相精修（top5）
        cand = df[["y", "formula", "top5_idx", "top5_logit", "hit_top5"]].copy()
        cand.to_csv(TOP5_CANDIDATES_CSV, index=False)

        # ---- summary.json ----
        summary: Dict[str, Any] = {
            "db": DB_PATH,
            "model": MODEL_PATH,
            "num_classes": NUM_CLASSES,
            "total_samples": int(N),
            "top1_acc": float(top1),
            "top5_acc": float(top5),
            "top10_acc": float(top10),
            "avg_nll": float(avg_nll),
            "ece_10bins": float(ece),
            "polymorph_error_rate_in_wrong(%)": float(poly_rate),
            "polymorph_recoverable_rate_in_polymorph(%)": float(poly_recoverable_rate),
            "outputs": {
                "full_report_csv": FULL_REPORT_CSV,
                "top5_candidates_csv": TOP5_CANDIDATES_CSV,
                "calibration_bins_csv": CALIBRATION_CSV,
                "figure_png": FIG_PATH,
                "formula_cache_json": CACHE_FORMULA_JSON,
            }
        }
        with open(SUMMARY_JSON, "w") as f:
            json.dump(summary, f, indent=2)

        # ---- 绘图 ----
        plot_science_figures(df, calib, complexity_acc, poly_rate, ece)

        print("\n📊 [分析报告总结]")
        print(f"   Top-1 Acc: {top1:.2f}% | Top-5: {top5:.2f}% | Top-10: {top10:.2f}%")
        print(f"   Avg NLL: {avg_nll:.4f} | ECE(10 bins): {ece:.4f}")
        print(f"   Polymorph error 占比(错误样本中): {poly_rate:.2f}%")
        print(f"   Polymorph 可救率(同式错中 top5 命中): {poly_recoverable_rate:.2f}%")
        print(f"   输出目录: {OUT_DIR}")
        print(f"   - {FULL_REPORT_CSV}")
        print(f"   - {TOP5_CANDIDATES_CSV}")
        print(f"   - {CALIBRATION_CSV}")
        print(f"   - {SUMMARY_JSON}")
        print(f"   - {FIG_PATH}")

    cleanup_ddp()


if __name__ == "__main__":
    main()