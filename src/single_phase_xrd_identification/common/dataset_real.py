# common/dataset_real.py
# ✅ Stage3 用：真实谱 + peak token + elem_onehot（来自 *_CIF.txt）
# - CSV: 两列 (2theta, intensity)
# - 裁剪到 10–80°
# - 插值到 3500 点
# - peak token 与训练阶段一致
# - 新增：从 *_CIF.txt 解析元素集合 -> 118 维 one-hot

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# -----------------------------
# 常量（与训练保持一致）
# -----------------------------
THETA_MIN = 10.0
THETA_MAX = 80.0
TARGET_DIM = 3500
MAX_PEAKS = 48
ELEM_DIM = 118

# -----------------------------
# 118 元素表（必须与训练 dataset 完全一致）
# -----------------------------
_ELEMENT_SYMBOLS = [
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
]
_SYM2IDX = {s: i for i, s in enumerate(_ELEMENT_SYMBOLS)}

# -----------------------------
# CIF.txt -> 元素集合 -> one-hot
# -----------------------------
_ELEM_RE = re.compile(r"\b([A-Z][a-z]?)\b")

def parse_cif_txt_to_onehot(txt_path: str, dim: int = ELEM_DIM) -> torch.Tensor:
    v = torch.zeros(dim, dtype=torch.float32)
    if not os.path.exists(txt_path):
        return v

    elems = set()
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 优先：ATOM 表
    start = -1
    for i, line in enumerate(lines):
        if "ATOM" in line and "OCCUPANCY" in line:
            start = i + 1
            break

    if start != -1:
        for j in range(start, min(start + 500, len(lines))):
            s = lines[j].strip()
            if not s:
                break
            tok = s.split()
            sym = tok[0]
            if sym in _SYM2IDX:
                elems.add(sym)

    # 兜底：全文 token 扫描
    if not elems:
        for line in lines:
            for sym in _ELEM_RE.findall(line):
                if sym in _SYM2IDX:
                    elems.add(sym)

    for sym in elems:
        v[_SYM2IDX[sym]] = 1.0
    return v

# -----------------------------
# CSV 读取 & 插值
# -----------------------------
def read_xy_csv(csv_path: str):
    try:
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32, comments="#")
    except ValueError:
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32, comments="#", skiprows=1)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"CSV 格式错误: {csv_path}")

    return data[:, 0], data[:, 1]

def upsample_10_80(theta, intensity, n=TARGET_DIM):
    mask = (
        np.isfinite(theta)
        & np.isfinite(intensity)
        & (theta >= THETA_MIN)
        & (theta <= THETA_MAX)
    )
    theta = theta[mask]
    intensity = intensity[mask]

    if theta.size < 2:
        return np.zeros(n, dtype=np.float32)

    order = np.argsort(theta)
    theta = theta[order]
    intensity = intensity[order]

    _, idx = np.unique(theta, return_index=True)
    theta = theta[idx]
    intensity = intensity[idx]

    if theta[0] > THETA_MIN:
        theta = np.insert(theta, 0, THETA_MIN)
        intensity = np.insert(intensity, 0, intensity[0])
    if theta[-1] < THETA_MAX:
        theta = np.append(theta, THETA_MAX)
        intensity = np.append(intensity, intensity[-1])

    xnew = np.linspace(THETA_MIN, THETA_MAX, n, dtype=np.float32)
    f = interp1d(theta, intensity, kind="slinear", fill_value="extrapolate")
    return f(xnew).astype(np.float32)

# -----------------------------
# Dataset
# -----------------------------
class XRDDatasetStrict(Dataset):
    """
    返回:
      x_tensor      [3500]
      peaks_tensor  [32, 2]
      elem_onehot   [118]   ✅ 新增
      label         int
      name          str (R040009)
    """

    def __init__(self, strict_dir, rruff2mp_dict, *, num_classes=100315):
        self.strict_dir = strict_dir
        self.rruff2mp = rruff2mp_dict
        self.num_classes = num_classes
        self.target_grid = np.linspace(0.0, 1.0, TARGET_DIM, dtype=np.float32)
        self.samples = []
        self._collect()

    def _collect(self):
        for fn in os.listdir(self.strict_dir):
            if not fn.endswith(".csv"):
                continue
            stem = fn[:-4]
            cif_txt = os.path.join(self.strict_dir, f"{stem}_CIF.txt")
            if not os.path.exists(cif_txt):
                continue
            if stem not in self.rruff2mp:
                continue
            self.samples.append({
                "name": stem,
                "csv": os.path.join(self.strict_dir, fn),
                "cif": cif_txt,
                "label": int(self.rruff2mp[stem]),
            })
        self.samples.sort(key=lambda x: x["name"])

    def __len__(self):
        return len(self.samples)


    def extract_peaks(self, y):
        """

        输出 shape: [MAX_PEAKS, 5]
        每行(按角度升序):
        0: pos_norm                 (0~1)
        1: height_norm              (0~1)  直接用峰高(输入已0~1归一化)
        2: sin2_norm                (0~1)
        3: delta_sin2_norm          (0~1)
        4: delta_ratio_local_norm   (0~1)
        """
        # ✅ 选峰参数与 based 一致
        peaks, props = find_peaks(y, height=0.05, distance=10)
        if len(peaks) == 0:
            return torch.zeros((MAX_PEAKS, 5), dtype=torch.float32)

        peak_pos = self.target_grid[peaks].astype(np.float32)     # 0~1
        peak_h   = props["peak_heights"].astype(np.float32)       # y已归一化 => 0~1

        # ✅ NMS：与 based 一致（最小间隔 0.5°）
        min_sep_deg = 0.5
        angles_deg_all = THETA_MIN + peak_pos * (THETA_MAX - THETA_MIN)

        order = np.argsort(-peak_h)  # 高->低
        sel = []
        for j in order:
            a = angles_deg_all[j]
            ok = True
            for p in sel:
                if abs(a - angles_deg_all[p]) < min_sep_deg:
                    ok = False
                    break
            if ok:
                sel.append(j)
                if len(sel) >= MAX_PEAKS:
                    break

        sel = np.array(sorted(sel), dtype=int)  # 最后按角度升序喂给 transformer

        peak_pos = peak_pos[sel]
        peak_h   = peak_h[sel]

        # ---- 计算 sin^2(theta) ----
        two_theta = THETA_MIN + peak_pos * (THETA_MAX - THETA_MIN)   # degree
        theta_rad = np.deg2rad(two_theta / 2.0)
        sin2 = (np.sin(theta_rad) ** 2).astype(np.float32)

        if sin2.max() > sin2.min():
            sin2_norm = (sin2 - sin2.min()) / (sin2.max() - sin2.min())
        else:
            sin2_norm = np.zeros_like(sin2, dtype=np.float32)

        # ---- delta_sin2 ----
        delta_sin2 = np.zeros_like(sin2, dtype=np.float32)
        delta_sin2[1:] = sin2[1:] - sin2[:-1]

        if delta_sin2.max() > delta_sin2.min():
            delta_sin2_norm = (delta_sin2 - delta_sin2.min()) / (delta_sin2.max() - delta_sin2.min())
        else:
            delta_sin2_norm = np.zeros_like(delta_sin2, dtype=np.float32)

        # ---- delta ratio ----
        K = 5
        ref = delta_sin2[1:min(len(delta_sin2), K + 1)]
        ref = ref[ref > 1e-8]
        scale = float(np.median(ref)) if ref.size > 0 else 1.0
        if scale < 1e-8:
            scale = 1.0

        delta_ratio = (delta_sin2 / scale).astype(np.float32)
        if delta_ratio.max() > delta_ratio.min():
            delta_ratio_norm = (delta_ratio - delta_ratio.min()) / (delta_ratio.max() - delta_ratio.min())
        else:
            delta_ratio_norm = np.zeros_like(delta_ratio, dtype=np.float32)

        feats = np.stack([peak_pos, peak_h, sin2_norm, delta_sin2_norm, delta_ratio_norm], axis=1).astype(np.float32)

        out = torch.zeros((MAX_PEAKS, 5), dtype=torch.float32)
        out[:len(feats)] = torch.from_numpy(feats)
        return out


    def __getitem__(self, idx):
        s = self.samples[idx]

        theta, intensity = read_xy_csv(s["csv"])
        y = upsample_10_80(theta, intensity)

        # 归一化到 0–1
        y_min, y_max = float(y.min()), float(y.max())
        if y_max - y_min > 1e-12:
            y = (y - y_min) / (y_max - y_min)
        else:
            y[:] = 0.0

        x_tensor = torch.from_numpy(y).float()
        peaks_tensor = self.extract_peaks(y)

        # ✅ 元素 one-hot 来自 CIF
        elem_onehot = parse_cif_txt_to_onehot(s["cif"])

        label = s["label"]
        if label < 0 or label >= self.num_classes:
            label = max(0, min(label, self.num_classes - 1))

        return x_tensor, peaks_tensor, elem_onehot, label, s["name"]
