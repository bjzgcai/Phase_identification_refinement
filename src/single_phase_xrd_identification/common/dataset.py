# common/dataset.py
# 5维
import os
import re
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from ase.db import connect
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# -----------------------------
# 118 元素表（H=1 ... Og=118）
# -----------------------------
_ELEMENT_SYMBOLS = [
    "H","He",
    "Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn",
    "Nh","Fl","Mc","Lv","Ts","Og"
]
_SYM2IDX = {s: i for i, s in enumerate(_ELEMENT_SYMBOLS)}  # 0-based


def parse_formula_to_onehot(formula: str, dim: int = 118) -> torch.Tensor:
    """
    输入: "Ni2", "Li2Mn1.5Ni0.5O4"
    输出: 118维 one-hot（只标记出现过的元素，不管化学计量数）
    """
    v = torch.zeros(dim, dtype=torch.float32)
    if not isinstance(formula, str):
        return v
    formula = formula.strip()
    if not formula:
        return v

    # 抓元素符号：大写 + 可选小写
    tokens = re.findall(r"([A-Z][a-z]?)", formula)
    for sym in tokens:
        idx = _SYM2IDX.get(sym, None)
        if idx is not None:
            v[idx] = 1.0
    return v


def try_get_formula(row) -> str:
    """
    化学式在 row.formula
    同时为了兼容其它库，保留一些兜底读取方式。
    """
    f = getattr(row, "formula", "")
    if isinstance(f, str) and f.strip():
        return f.strip()

    kv = row.key_value_pairs or {}
    for k in ["Formula", "formula", "chemical_formula_sum", "chemical_formula"]:
        val = kv.get(k, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    sim = kv.get("simulation_param", None)
    if isinstance(sim, str) and sim.strip():
        try:
            obj = json.loads(sim)
            if isinstance(obj, dict):
                for k in ["Formula", "formula", "chemical_formula_sum", "chemical_formula"]:
                    val = obj.get(k, None)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
        except Exception:
            pass

    return ""


def upsample_angle_intensity(angle, intensity, *, x_min=10.0, x_max=80.0, n=3500):
    """
    - 把 (angle, intensity) 插值到固定 10~80° 的等间隔网格。
    - 自动补齐端点（如果 angle 不覆盖到 10 或 80）
    - 自动去重（angle 可能有重复点）
    """
    angle = np.asarray(angle, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)

    if angle.size == 0 or intensity.size == 0:
        return np.zeros(n, dtype=np.float32)

    # 对齐长度（防止某些数据不一致）
    m = min(len(angle), len(intensity))
    angle = angle[:m]
    intensity = intensity[:m]

    # 去掉 NaN
    mask = np.isfinite(angle) & np.isfinite(intensity)
    angle = angle[mask]
    intensity = intensity[mask]
    if angle.size == 0:
        return np.zeros(n, dtype=np.float32)

    # 按 angle 排序
    order = np.argsort(angle)
    angle = angle[order]
    intensity = intensity[order]

    # 去重 angle（只保留第一次出现）
    _, unique_idx = np.unique(angle, return_index=True)
    angle = angle[unique_idx]
    intensity = intensity[unique_idx]

    # 补端点
    if float(angle[0]) > x_min:
        angle = np.insert(angle, 0, x_min)
        intensity = np.insert(intensity, 0, float(intensity[0]))
    if float(angle[-1]) < x_max:
        angle = np.append(angle, x_max)
        intensity = np.append(intensity, float(intensity[-1]))

    # 插值到固定网格
    xnew = np.linspace(x_min, x_max, n, dtype=np.float32)
    f = interp1d(angle, intensity, kind="slinear", fill_value="extrapolate")
    ynew = f(xnew).astype(np.float32)
    return ynew


class XRDDataset(Dataset):

    def __init__(
        self,
        db_path: str,
        ids,
        *,
        target_dim: int = 3500,
        max_peaks: int = 48,
        augment: bool = False,
        return_id_str: bool = False,      
        return_elem_onehot: bool = True,
        num_classes: int = 100315,        # Task A：结构ID分类
        theta_min: float = 10.0,
        theta_max: float = 80.0,
    ):
        self.db_path = db_path
        self.ids = list(ids)
        self.target_dim = int(target_dim)
        self.max_peaks = int(max_peaks)
        self.augment = bool(augment)
        self.return_id_str = bool(return_id_str)
        self.return_elem_onehot = bool(return_elem_onehot)
        self.num_classes = int(num_classes)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        # peak token 用的“归一化位置”
        self.target_grid = np.linspace(0.0, 1.0, self.target_dim, dtype=np.float32)

    def __len__(self):
        return len(self.ids)

    # --------------------------
    # peak token
    # --------------------------
    def extract_peaks(self, y: np.ndarray) -> torch.Tensor:
        """
        5D peak token，

        输出 shape: [max_peaks, 5]
        每行:
        0: pos_norm                 (0~1)
        1: height_norm              (0~1)  这里直接用峰高(输入已0~1归一化)
        2: sin2_norm                (0~1)
        3: delta_sin2_norm          (0~1)
        4: delta_ratio_local_norm   (0~1)
        """
        # ✅ 选峰：完全照搬 or.py（不要平滑/动态阈值/配额）
        peaks, props = find_peaks(y, height=0.05, distance=10)
        if len(peaks) == 0:
            return torch.zeros((self.max_peaks, 5), dtype=torch.float32)

        peak_pos = self.target_grid[peaks].astype(np.float32)              # 0~1
        peak_h   = props["peak_heights"].astype(np.float32)                # 因为 y 已经归一化，所以它本身就在 0~1


        # ✅ 按峰高从大到小扫描，做“最小间隔=0.5°”的贪心选择（NMS）
        min_sep_deg = 0.5
        angles_deg_all = self.theta_min + peak_pos * (self.theta_max - self.theta_min)

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
                if len(sel) >= self.max_peaks:
                    break

        sel = np.array(sorted(sel), dtype=int)  # 最后按角度升序喂给 transformer

        peak_pos = peak_pos[sel]
        peak_h   = peak_h[sel]

        # ---- 计算 sin^2(theta) ----
        two_theta = self.theta_min + peak_pos * (self.theta_max - self.theta_min)   # degree
        theta_rad = np.deg2rad(two_theta / 2.0)
        sin2 = (np.sin(theta_rad) ** 2).astype(np.float32)

        if sin2.max() > sin2.min():
            sin2_norm = (sin2 - sin2.min()) / (sin2.max() - sin2.min())
        else:
            sin2_norm = np.zeros_like(sin2)

        # ---- delta_sin2 ----
        delta_sin2 = np.zeros_like(sin2)
        delta_sin2[1:] = sin2[1:] - sin2[:-1]

        if delta_sin2.max() > delta_sin2.min():
            delta_sin2_norm = (delta_sin2 - delta_sin2.min()) / (delta_sin2.max() - delta_sin2.min())
        else:
            delta_sin2_norm = np.zeros_like(delta_sin2)

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
            delta_ratio_norm = np.zeros_like(delta_ratio)

        feats = np.stack([peak_pos, peak_h, sin2_norm, delta_sin2_norm, delta_ratio_norm], axis=1).astype(np.float32)

        out = torch.zeros((self.max_peaks, 5), dtype=torch.float32)
        out[:len(feats)] = torch.from_numpy(feats)
        return out
    # --------------------------
    # 读谱
    # --------------------------
    def read_pattern(self, row) -> np.ndarray:

        # 1)  风格：row.angle / row.intensity
        angle = getattr(row, "angle", None)
        inten = getattr(row, "intensity", None)

        if angle is not None and inten is not None:
            try:
                # angle/inten 可能是 python list / numpy / 或字符串（有时会存成 repr）
                if isinstance(angle, str):
                    angle = json.loads(angle) if angle.strip().startswith("[") else angle
                if isinstance(inten, str):
                    inten = json.loads(inten) if inten.strip().startswith("[") else inten

                y = upsample_angle_intensity(
                    angle, inten,
                    x_min=self.theta_min, x_max=self.theta_max, n=self.target_dim
                )
                return y
            except Exception:
                pass

        # 2) 你的 db 风格：key_value_pairs
        kv = row.key_value_pairs or {}

        if "intensity" in kv and isinstance(kv["intensity"], str):
            y = np.fromstring(kv["intensity"], sep=",", dtype=np.float32)
            if y.size == self.target_dim:
                return y
            # 尝试拉伸到 target_dim
            if y.size > 2:
                xold = np.linspace(self.theta_min, self.theta_max, y.size, dtype=np.float32)
                f = interp1d(xold, y.astype(np.float32), kind="linear", fill_value="extrapolate")
                xnew = np.linspace(self.theta_min, self.theta_max, self.target_dim, dtype=np.float32)
                return f(xnew).astype(np.float32)

        if "latt_dis" in kv and isinstance(kv["latt_dis"], str):
            y = np.fromstring(kv["latt_dis"], sep=",", dtype=np.float32)
            if y.size == self.target_dim:
                return y
            if y.size > 2:
                xold = np.linspace(self.theta_min, self.theta_max, y.size, dtype=np.float32)
                f = interp1d(xold, y.astype(np.float32), kind="linear", fill_value="extrapolate")
                xnew = np.linspace(self.theta_min, self.theta_max, self.target_dim, dtype=np.float32)
                return f(xnew).astype(np.float32)

        # 3) 实在没有就给全零
        return np.zeros(self.target_dim, dtype=np.float32)

    def __getitem__(self, idx):
        real_id = int(self.ids[idx])

        with connect(self.db_path) as db:
            row = db.get(id=real_id)

        # --- 1) 读谱 ---
        x = self.read_pattern(row)

        # --- 2) 归一化到 0~1（用于稳定 peaks 提取）---
        x = x.astype(np.float32, copy=False)
        x_min = float(np.min(x)) if x.size else 0.0
        x_max = float(np.max(x)) if x.size else 0.0
        if x_max - x_min > 1e-12:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = np.zeros_like(x, dtype=np.float32)

        x_tensor = torch.from_numpy(x).float()
        peaks_tensor = self.extract_peaks(x)

        # --- 3) 元素 one-hot（118）---
        if self.return_elem_onehot:
            formula = try_get_formula(row)
            elem_onehot = parse_formula_to_onehot(formula, dim=118)
        else:
            elem_onehot = torch.zeros(118, dtype=torch.float32)

        # --- 4) label / id_str ---
        kv = row.key_value_pairs or {}

        if self.return_id_str:
            # 给 Stage1.5/Stage2 备用：如果没有 mp_id 就用 mp-{real_id}
            label = kv.get("mp_id", f"mp-{real_id}")
        else:
            # Task A：结构ID分类（0..100314）
            if "Label" in kv:
                label = int(kv["Label"])   # ❗不再 -1
            else:
                # 兜底：用 (real_id-1)，但一般你 db 里都有 Label
                label = real_id - 1

            # 防止 CUDA nll_loss 崩溃
            if label < 0 or label >= self.num_classes:
                label = max(0, min(label, self.num_classes - 1))

        return x_tensor, peaks_tensor, elem_onehot, label


# -----------------------------
# 自检：导出 CSV + 保存图片
# -----------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    db_path = os.path.join(root_dir, "data", "trainV.db")

    print(f"🧪 [Dataset 完整测试] 读取数据库: {os.path.basename(db_path)}")

    if os.path.exists(db_path):
        ds = XRDDataset(db_path, [20321], augment=True)

        # ✅ 现在返回 4 个
        y_aug, peaks, elem_onehot, label = ds[0]

        ds.augment = False
        y_raw, _, elem_onehot_raw, _ = ds[0]

        print(f"🧬 formula one-hot: shape={tuple(elem_onehot.shape)} ones={int(elem_onehot.sum().item())}")
        idxs = (elem_onehot > 0).nonzero(as_tuple=True)[0].tolist()
        elems_present = [_ELEMENT_SYMBOLS[i] for i in idxs]
        print("🧬 elements present:", elems_present)

        theta_min, theta_max = 10.0, 80.0
        x_axis = np.linspace(theta_min, theta_max, ds.target_dim)

        print(f"✅ 数据准备就绪: X轴范围 {theta_min}° - {theta_max}°")

        # 导出 CSV
        csv_filename = "xrd_demo_data.csv"
        print(f"💾 正在保存 CSV 文件: {csv_filename} ...")
        data_to_save = np.column_stack((x_axis, y_raw.numpy(), y_aug.numpy()))
        header = "2Theta,Intensity_Original,Intensity_Augmented"
        np.savetxt(csv_filename, data_to_save, delimiter=",", header=header, comments="", fmt="%.6f")
        print(f"   --> 保存成功: {csv_filename}")

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, y_raw.numpy(), color="blue", alpha=0.5, linewidth=2.0, )
        plt.fill_between(x_axis, y_raw.numpy(), color="blue", alpha=0.1)
        plt.plot(x_axis, y_aug.numpy(), color="red", linewidth=2.5, label="XRD")

        peak_locs = peaks[:, 0].numpy()  # pos_norm
        peak_angles = theta_min + peak_locs * (theta_max - theta_min)

        idx = np.rint(peak_locs * (ds.target_dim - 1)).astype(int)
        idx = np.clip(idx, 0, ds.target_dim - 1)

        # ✅ 用谱线真实强度作为 y
        y_scatter = y_aug.numpy()[idx]

        # 只画有效 token（pos>0 或者 y>0 都行）
        mask = peak_locs > 0
        plt.scatter(peak_angles[mask], y_scatter[mask], marker="x", s=50, color="black", zorder=5, label="Peak Tokens")

        plt.xlim(theta_min, theta_max)
        plt.xlabel("2θ (Degrees)", fontsize=12)
        plt.ylabel("Intensity (Normalized)", fontsize=12)
        # plt.title(f"Dataset Verification (ID=8, Label={label})", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)

        save_name = "dataset_check_angle.png"
        plt.savefig(save_name, dpi=200)
        print(f"🎉 图片已保存: {save_name}")