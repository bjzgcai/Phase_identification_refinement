import os, itertools, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from collections import deque, OrderedDict
from concurrent.futures import ProcessPoolExecutor

from pymatgen.core import Structure, Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pybaselines import Baseline

import argparse

# ===========================
# 全程记录日志（Rwp / frac / scale 曲线）
# ===========================
RWP_LOG = []
FRAC_LOG = []
SCALE_LOG = []
STEP_LOG = []

_ALT_EARLY_STOP_PATIENCE = 50
_ALT_SCORE_W_DATA = 1.0
_ALT_SCORE_W_STOICH = 0.1
ANG_MIN, ANG_MAX = 20.0, 160.0


def log_event(msg: str):
    """Event-driven logger. Always flush so tee/tail -f can show accepted improvements immediately."""
    print(msg, flush=True)


def save_xy_with_residual(out_path: str, x: np.ndarray, y_obs: np.ndarray, y_fit: np.ndarray):
    residual = y_obs - y_fit
    xy_out = np.column_stack([x, y_obs, y_fit, residual])
    np.savetxt(
        out_path, xy_out, fmt='%.6f',
        header='2Theta  Intensity_Obs  Intensity_Fit  Residual_ObsMinusFit'
    )


def save_fit_plot_with_residual(out_path: str, x: np.ndarray, y_obs: np.ndarray, y_fit: np.ndarray,
                                title: str, fit_label: str, text_lines: Optional[List[str]] = None):
    residual = y_obs - y_fit
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    ax1.plot(x, y_obs, lw=1.0, label='Experiment')
    ax1.plot(x, y_fit, lw=1.0, label=fit_label)
    ax1.set_ylabel('Normalized Intensity')
    ax1.set_title(title)
    ax1.legend()

    if text_lines:
        ax1.text(
            1.02, 0.98, '\n'.join(text_lines), transform=ax1.transAxes,
            fontsize=10, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

    ax2.plot(x, residual, lw=1.0, label='Residual (Obs - Fit)')
    ax2.axhline(0.0, lw=0.8, linestyle='--')
    ax2.set_xlabel('2θ (deg)')
    ax2.set_ylabel('Residual')
    ax2.legend()

    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_stage_outputs(ctx: 'RefinementContext', stage_name: str, out_dir: str, out_prefix: str,
                       title: str, fit_label: str):
    os.makedirs(out_dir, exist_ok=True)

    text_lines = ['Composition:']
    for f, w in zip(ctx.phase_structs.keys(), ctx.fr):
        text_lines.append(f'  {os.path.basename(f)}: {w*100:.2f}%')
    text_lines += ['', f'Rwp = {ctx.best_rwp:.2f}%']

    png_path = os.path.join(out_dir, f'{out_prefix}.png')
    save_fit_plot_with_residual(png_path, ctx.x_grid, ctx.y_obs, ctx.yfit, title, fit_label, text_lines)
    log_event(f'🖼️ 已保存{stage_name}图像：{png_path}')

    xy_path = os.path.join(out_dir, f'{out_prefix}.xy')
    save_xy_with_residual(xy_path, ctx.x_grid, ctx.y_obs, ctx.yfit)
    log_event(f'💾 已保存{stage_name}谱线：{xy_path}')

    txt_path = os.path.join(out_dir, f'{out_prefix}.txt')
    with open(txt_path, 'w', encoding='utf-8') as fw:
        fw.write(f'=== {stage_name} Refinement Result ===\n')
        fw.write(f'Rwp : {ctx.best_rwp:.3f}%\n')
        fw.write('TCH params: ' + ', '.join([f'{k}={v:.6g}' for k, v in ctx.tch_params.items()]) + '\n')
        fw.write('Global zero shift: ' + f'{ctx.global_zshift:.6f} deg\n')
        fw.write('Residual definition: Intensity_Obs - Intensity_Fit\n')
        fw.write('\nPhases:\n')
        for f, w, s in zip(ctx.phase_structs.keys(), ctx.fr, ctx.sf):
            fw.write(f'  {os.path.basename(f):<28s} frac={w*100:6.2f}% | scale={float(s):.4f}\n')
    log_event(f'🧾 已保存{stage_name}文本报告：{txt_path}')

    cif_dir = os.path.join(out_dir, 'cifs')
    os.makedirs(cif_dir, exist_ok=True)
    stage_suffix = out_prefix.replace('_Refined', '')
    for fpath, struct in ctx.phase_structs.items():
        base = os.path.basename(fpath)
        name, _ = os.path.splitext(base)
        out_path = os.path.join(cif_dir, f'{name}_{stage_suffix}.cif')
        export_st = apply_dw_to_structure(struct, ctx.phase_dw_dicts[fpath])
        export_cif_with_biso(export_st, out_path)
    log_event(f'💾 已导出{stage_name} CIF 文件至 {cif_dir}/')

# ===========================
# v44 core helpers
# ===========================
def sync_biso(struct):
    for site in struct.sites:
        if 'Biso' not in site.properties:
            if 'Uiso' in site.properties:
                site.properties['Biso'] = float(site.properties['Uiso']) * 8.0 * np.pi * np.pi
            else:
                site.properties['Biso'] = 0.78956
        else:
            site.properties['Biso'] = float(site.properties['Biso'])
    return struct


def export_cif_with_biso(struct: Structure, out_path: str):
    b_vals = [float(s.properties.get('Biso', 0.78956)) for s in struct.sites]
    writer = CifWriter(struct, symprec=None)
    writer.write_file(out_path)
    with open(out_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    final_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == 'loop_' and i + 1 < len(lines) and '_atom_site_' in lines[i + 1]:
            raw_headers = []
            i += 1
            while i < len(lines) and lines[i].strip().startswith('_atom_site_'):
                tag = lines[i].strip()
                if tag not in ['_atom_site_U_iso_or_equiv', '_atom_site_B_iso_or_equiv']:
                    raw_headers.append(tag)
                i += 1

            try:
                occ_idx = raw_headers.index('_atom_site_occupancy')
            except ValueError:
                occ_idx = len(raw_headers) - 1

            new_headers = list(raw_headers)
            new_headers.insert(occ_idx + 1, '_atom_site_B_iso_or_equiv')

            data_rows = []
            atom_count = 0
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('_', 'loop_', 'data_')):
                tokens = lines[i].split()
                if len(tokens) >= len(raw_headers) and atom_count < len(b_vals):
                    current_tokens = tokens[:len(raw_headers)]
                    try:
                        occ_val = float(current_tokens[occ_idx])
                        current_tokens[occ_idx] = f'{occ_val:.4f}'
                    except (ValueError, IndexError):
                        pass
                    current_tokens.insert(occ_idx + 1, f'{b_vals[atom_count]:.5f}')
                    data_rows.append(''.join([f'{t:<16}' for t in current_tokens]).rstrip() + '\n')
                    atom_count += 1
                i += 1

            final_lines.append('loop_\n')
            for h in new_headers:
                final_lines.append(f' {h}\n')
            final_lines.extend(data_rows)
            if i < len(lines):
                final_lines.append(lines[i])
                i += 1
        else:
            final_lines.append(line)
            i += 1

    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)


def apply_dw_to_structure(struct: Structure, dw_dict: dict) -> Structure:
    b_vals = []
    for site in struct.sites:
        val = np.mean([dw_dict.get(str(el), 0.78956) for el in site.species.elements])
        b_vals.append(float(val))
    new_props = {k: list(v) for k, v in struct.site_properties.items()}
    new_props['Biso'] = b_vals
    return Structure(struct.lattice, [s.species for s in struct.sites], [s.frac_coords for s in struct.sites], site_properties=new_props)


def read_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        data = np.loadtxt(path, comments=['#', '%', ';', '!', '@', "'"])
    except ValueError:
        x_list, y_list = [], []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                for c in ['#', '%', ';', '!', '@', "'"]:
                    line = line.split(c)[0]
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        x_list.append(float(parts[0]))
                        y_list.append(float(parts[1]))
                    except ValueError:
                        continue
        data = np.column_stack((x_list, y_list))

    x, y_raw = data[:, 0], data[:, 1]
    fitter = Baseline(x_data=x)
    y_bg, _ = fitter.snip(y_raw, max_half_window=30, decreasing=True, smooth_half_window=3)
    y_net = y_raw - y_bg
    y_net = np.clip(y_net, 0, None)
    if y_net.max() > 0:
        y_net /= y_net.max()
    return x, y_net


def calc_r_factors(y_obs: np.ndarray, y_calc: np.ndarray, num_params: int = 20) -> float:
    num = np.sum((y_obs - y_calc) ** 2)
    den = np.sum(y_obs ** 2)
    if den < 1e-12:
        return 999.0
    return float(np.sqrt(num / den) * 100.0)


def md_factor_for_hkl(struct: Structure, hkl, axis_uvw, r: float) -> float:
    r = float(np.clip(r, 1e-3, 1e3))
    h, k, l = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
    u, v, w = (float(axis_uvw[0]), float(axis_uvw[1]), float(axis_uvw[2]))
    if abs(u) + abs(v) + abs(w) < 1e-12:
        return 1.0
    rl = struct.lattice.reciprocal_lattice
    n_cart = rl.get_cartesian_coords([h, k, l])
    d_cart = struct.lattice.get_cartesian_coords([u, v, w])
    n_norm = np.linalg.norm(n_cart)
    d_norm = np.linalg.norm(d_cart)
    if n_norm < 1e-12 or d_norm < 1e-12:
        return 1.0
    cos_alpha = float(np.dot(n_cart, d_cart) / (n_norm * d_norm))
    cos2 = np.clip(cos_alpha * cos_alpha, 0.0, 1.0)
    sin2 = 1.0 - cos2
    return float((r * r * cos2 + (1.0 / r) * sin2) ** (-1.5))


def shift_spectrum(y: torch.Tensor, shift_deg: torch.Tensor, step_deg: float) -> torch.Tensor:
    n = y.numel()
    shift_pix = shift_deg / torch.clamp(step_deg, min=1e-9)
    idx = torch.arange(n, device=y.device, dtype=torch.float32) - shift_pix
    i0 = torch.clamp(torch.floor(idx).long(), 0, n - 1)
    i1 = torch.clamp(i0 + 1, 0, n - 1)
    frac = idx - i0.float()
    return (1 - frac) * y[i0] + frac * y[i1]


def synth_profile_po(two_theta: np.ndarray, structure: Structure,
                     wl=1.5406,
                     U=0.003, V=0.001, W=0.020, X=0.020, Y=0.010,
                     broad_base=0.08,
                     po_axis=(0, 0, 1), po_r: float = 1.0,
                     enable_po: bool = False) -> np.ndarray:
    U, V, W, X, Y = [max(float(v), 0.0) for v in (U, V, W, X, Y)]
    calc = XRDCalculator(wavelength=wl)
    pat = calc.get_pattern(structure, (float(two_theta.min()), float(two_theta.max())))
    y = np.zeros_like(two_theta)
    if len(pat.x) == 0:
        return y
    deg2rad = np.pi / 180.0
    for t0, I0, hkls in zip(pat.x, pat.y, pat.hkls):
        if enable_po and hkls:
            facs = []
            for item in hkls:
                hkl = item.get('hkl', None)
                if hkl is None:
                    continue
                facs.append(md_factor_for_hkl(structure, hkl, po_axis, po_r))
            if facs:
                I0 = I0 * float(np.mean(facs))
        theta = 0.5 * t0 * deg2rad
        tanth, costh = np.tan(theta), max(np.cos(theta), 1e-8)
        H_G2 = U * tanth ** 2 + V * tanth + W
        H_L = X / costh + Y * tanth
        H = (H_G2 ** 5 + 2.69269 * H_G2 ** 4 * H_L + 2.42843 * H_G2 ** 3 * H_L ** 2 +
             4.47163 * H_G2 ** 2 * H_L ** 3 + 0.07842 * H_G2 * H_L ** 4 + H_L ** 5) ** (1 / 5)
        H = np.sqrt(H ** 2 + broad_base ** 2)
        ratio = H_L / max(H, 1e-10)
        eta = np.clip(1.36603 * ratio - 0.47719 * ratio ** 2 + 0.11116 * ratio ** 3, 0.0, 1.0)
        z = (two_theta - t0) / max(H, 1e-10)
        g = np.exp(-4 * np.log(2) * z ** 2)
        l = 1.0 / (1.0 + (2 * z) ** 2)
        y += I0 * (eta * l + (1 - eta) * g)
    if y.max() > 0:
        y = y / y.max()
    return y


class TorchRietveld(torch.nn.Module):
    def __init__(self, exp_x: np.ndarray, exp_y: np.ndarray,
                 patterns: List[np.ndarray], bg_degree: int, device: torch.device,
                 freeze_scale: bool = False,
                 freeze_zero_shift: bool = False,
                 init_zero_shift: float = 0.0):
        super().__init__()
        self.x = torch.tensor(exp_x, dtype=torch.float32, device=device)
        self.y = torch.tensor(exp_y, dtype=torch.float32, device=device)
        self.patterns = [torch.tensor(p, dtype=torch.float32, device=device) for p in patterns]
        n_phase = len(self.patterns)
        self.logits = torch.nn.Parameter(torch.zeros(n_phase, device=device))
        init_z_t = torch.tensor(init_zero_shift, dtype=torch.float32, device=device)
        if freeze_zero_shift:
            self.register_buffer('zero_shift', init_z_t.clone())
        else:
            self.zero_shift = torch.nn.Parameter(init_z_t.clone())
        if freeze_scale:
            self.register_buffer('scale_factors', torch.ones(n_phase, device=device))
            self._freeze_scale = True
        else:
            self.log_scale = torch.nn.Parameter(torch.zeros(n_phase, device=device))
            self._freeze_scale = False

    def forward(self):
        weights = torch.softmax(self.logits, dim=0)
        step = torch.clamp(self.x[1] - self.x[0], min=1e-9)
        if self._freeze_scale:
            s_pos = torch.clamp(self.scale_factors, min=1e-4, max=5.0)
        else:
            s_pos = torch.clamp(torch.exp(self.log_scale), min=1e-4, max=5.0)
        raw = weights * s_pos
        frac = raw / torch.clamp(raw.sum(), min=1e-9)
        amp = s_pos.sum()
        mix = torch.zeros_like(self.patterns[0])
        for f, p in zip(frac, self.patterns):
            shifted = shift_spectrum(p, self.zero_shift, step)
            mix += f * shifted
        y_pred = amp * mix
        return y_pred, frac, s_pos


def torch_refine(exp_y: np.ndarray, profiles: List[np.ndarray], device: torch.device,
                 bg_degree=5, epochs=100, lr=5e-3, weight_decay=1e-4, lbfgs=True, mode='fit',
                 freeze_scale=False, early_stop=True, patience=_ALT_EARLY_STOP_PATIENCE, min_delta=1e-4,
                 lbfgs_lr=0.3, lbfgs_max_iter=40,
                 main_bias: float = 0.0,
                 stoich_penalty_per_phase: Optional[List[float]] = None,
                 stoich_phase_weights: Optional[List[float]] = None,
                 lambda_stoich: float = 0.0,
                 global_logits: Optional[torch.nn.Parameter] = None,
                 freeze_zero_shift=False, init_zero_shift=0.0,
                 log_improvements: bool = True,
                 log_context: str = ''):
    model = TorchRietveld(np.arange(len(exp_y)), exp_y, profiles, bg_degree, device,
                          freeze_scale, freeze_zero_shift, init_zero_shift).to(device)
    if global_logits is not None:
        with torch.no_grad():
            n_phase = len(model.patterns)
            if global_logits.numel() == n_phase:
                model.logits.data.copy_(global_logits.data)
    exp_y_t = torch.tensor(exp_y, dtype=torch.float32, device=device)
    mse = torch.nn.MSELoss(reduction='none')
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if main_bias != 0.0:
        with torch.no_grad():
            if model.logits.numel() > 0:
                model.logits.data[0] += float(main_bias)
    best = {'loss': 1e9, 'pred': None, 'w': None, 's': None, 'z': init_zero_shift}
    no_improve = 0
    for ep in range(epochs):
        opt.zero_grad()
        y_pred, frac, s = model()
        data_loss = torch.mean(mse(y_pred, exp_y_t))
        if lambda_stoich > 0.0 and stoich_penalty_per_phase is not None:
            pen_vec = torch.tensor(stoich_penalty_per_phase, dtype=torch.float32, device=device)
            alpha_vec = torch.tensor(stoich_phase_weights or [1.0] * len(pen_vec), dtype=torch.float32, device=device)
            stoich_term = torch.sum(frac * s * alpha_vec * pen_vec)
            if mode == 'fit':
                loss = data_loss
            else:
                loss = data_loss + lambda_stoich * stoich_term
        else:
            loss = data_loss
        cur_metric = calc_r_factors(exp_y_t.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        if ep in (int(epochs * 0.5), int(epochs * 0.75)):
            for g in opt.param_groups:
                g['lr'] *= 0.5
        if cur_metric + 1e-8 < best['loss'] - min_delta:
            prev_best = best['loss']
            rwp_now = calc_r_factors(
                exp_y_t.detach().cpu().numpy(),
                y_pred.detach().cpu().numpy()
            )
            best.update(loss=cur_metric, pred=y_pred.detach().cpu().numpy(),
                        w=frac.detach().cpu().numpy(),
                        s=(s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else np.ones_like(frac.detach().cpu().numpy())),
                        z=model.zero_shift.detach().cpu().item())
            no_improve = 0
            if log_improvements and prev_best < 1e8:
                prefix = f"{log_context} | " if log_context else ""
                log_event(
                    f"    [torch_refine] improve | {prefix}mode={mode} | ep={ep:03d} | "
                    f"metric: {prev_best:.6f} -> {cur_metric:.6f} | "
                    f"Rwp={rwp_now:.2f}% | z={best['z']:.4f}"
                )
        else:
            no_improve += 1
        if early_stop and no_improve >= patience:
            break
    if lbfgs:
        opt2 = torch.optim.LBFGS(model.parameters(), lr=lbfgs_lr, max_iter=lbfgs_max_iter,
                                 tolerance_grad=1e-7, tolerance_change=1e-9)
        def closure():
            opt2.zero_grad()
            y2, frac2, s2 = model()
            data_loss2 = torch.mean(mse(y2, exp_y_t))
            if lambda_stoich > 0.0 and stoich_penalty_per_phase is not None:
                pen_vec = torch.tensor(stoich_penalty_per_phase, dtype=torch.float32, device=device)
                alpha_vec = torch.tensor(stoich_phase_weights or [1.0] * len(pen_vec), dtype=torch.float32, device=device)
                stoich_term2 = torch.sum(frac2 * s2 * alpha_vec * pen_vec)
                loss2 = data_loss2 + lambda_stoich * stoich_term2
            else:
                loss2 = data_loss2
            loss2.backward()
            return loss2
        opt2.step(closure)
        with torch.no_grad():
            y2, w2, s2 = model()
            data_loss2_eval = torch.mean((y2 - exp_y_t) ** 2)
            metric2 = calc_r_factors(exp_y_t.detach().cpu().numpy(), y2.detach().cpu().numpy())
            if metric2 + 1e-12 < best['loss'] - min_delta:
                prev_best = best['loss']
                rwp2_now = calc_r_factors(exp_y_t.detach().cpu().numpy(), y2.detach().cpu().numpy())
                best.update(pred=y2.detach().cpu().numpy(), w=w2.detach().cpu().numpy(),
                            s=(s2.detach().cpu().numpy() if isinstance(s2, torch.Tensor) else np.ones_like(w2.detach().cpu().numpy())),
                            z=model.zero_shift.detach().cpu().item(), loss=metric2)
                if log_improvements and prev_best < 1e8:
                    prefix = f"{log_context} | " if log_context else ""
                    log_event(
                        f"    [torch_refine] improve | {prefix}mode={mode} | stage=LBFGS | "
                        f"metric: {prev_best:.6f} -> {metric2:.6f} | "
                        f"Rwp={rwp2_now:.2f}% | z={best['z']:.4f}"
                    )
    if best['pred'] is None:
        with torch.no_grad():
            y_last, frac_last, s_last = model()
        best['pred'] = y_last.detach().cpu().numpy()
        best['w'] = frac_last.detach().cpu().numpy()
        best['s'] = s_last.detach().cpu().numpy()
        best['loss'] = 0.0
    Rwp = calc_r_factors(exp_y, best['pred'])
    fracs = np.atleast_1d(best['w'] / np.clip(np.sum(best['w']), 1e-12, None))
    s_final = np.atleast_1d(best['s'])
    return best['pred'], fracs, s_final, Rwp, best['z']


def _synth_profile_worker(args):
    key, st_obj, x_grid, wl, tpars, ax, r, broad_base, enable_po, dw_dict = args
    st = Structure.from_dict(st_obj) if isinstance(st_obj, dict) else st_obj
    U = float(np.clip(tpars['U'], 0.0001, 0.15))
    V = float(np.clip(tpars['V'], -0.10, 0.10))
    W = float(np.clip(tpars['W'], 0.0001, 0.15))
    X = float(np.clip(tpars['X'], 0.0001, 0.15))
    Y = float(np.clip(tpars['Y'], 0.0001, 0.25))
    kmin, kmax = float(x_grid.min()), float(x_grid.max())
    calc = XRDCalculator(wavelength=wl, debye_waller_factors=dw_dict)
    pat = calc.get_pattern(st, (kmin, kmax))
    t_list, I_list, hkls_all = np.array(pat.x), np.array(pat.y), np.array(pat.hkls, dtype=object)
    y = np.zeros_like(x_grid)
    if len(t_list) == 0:
        return key, y
    deg2rad = np.pi / 180.0
    for t0, I0, hkls in zip(t_list, I_list, hkls_all):
        if enable_po and hkls is not None:
            facs = []
            for h in hkls:
                hkl = h.get('hkl', None)
                if hkl is None:
                    continue
                facs.append(md_factor_for_hkl(st, hkl, ax, r))
            if facs:
                I0 *= float(np.mean(facs))
        theta = 0.5 * t0 * deg2rad
        tanth, costh = np.tan(theta), max(np.cos(theta), 1e-8)
        H_G2 = U * tanth ** 2 + V * tanth + W
        H_L = X / costh + Y * tanth
        H = (H_G2 ** 5 + 2.69269 * H_G2 ** 4 * H_L + 2.42843 * H_G2 ** 3 * H_L ** 2 +
             4.47163 * H_G2 ** 2 * H_L ** 3 + 0.07842 * H_G2 * H_L ** 4 + H_L ** 5) ** (1 / 5)
        H = np.sqrt(H ** 2 + broad_base ** 2)
        ratio = H_L / max(H, 1e-10)
        eta = np.clip(1.36603 * ratio - 0.47719 * ratio ** 2 + 0.11116 * ratio ** 3, 0.0, 1.0)
        z = (x_grid - t0) / max(H, 1e-10)
        g = np.exp(-4 * np.log(2) * z ** 2)
        l = 1.0 / (1.0 + (2 * z) ** 2)
        y += I0 * (eta * l + (1 - eta) * g)
    if y.max() > 0:
        y /= y.max()
    return key, y


def get_cell_params(struct: Structure):
    lat = struct.lattice
    return lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma


def set_cell_abc(struct: Structure, a: float, b: float, c: float) -> Structure:
    al, be, ga = struct.lattice.alpha, struct.lattice.beta, struct.lattice.gamma
    new_lat = Lattice.from_parameters(a, b, c, al, be, ga)
    site_props = {k: list(v) for k, v in struct.site_properties.items()}
    return Structure(new_lat, [s.species for s in struct.sites], [s.frac_coords for s in struct.sites], site_properties=site_props)


def set_cell_abc_angles(struct: Structure, a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> Structure:
    new_lat = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    site_props = {k: list(v) for k, v in struct.site_properties.items()}
    return Structure(new_lat, [s.species for s in struct.sites], [s.frac_coords for s in struct.sites], site_properties=site_props)


def get_equivalent_groups(struct: Structure):
    sga = SpacegroupAnalyzer(struct, symprec=5e-4, angle_tolerance=3.0)
    symm = sync_biso(sga.get_symmetrized_structure())
    return [list(group) for group in symm.equivalent_indices]


def structure_with_shifted_group(struct: Structure, group_indices: List[int], dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
    lattice = struct.lattice
    species = [s.species for s in struct]
    fracs = [s.frac_coords.copy() for s in struct]
    site_props = {k: list(v) for k, v in struct.site_properties.items()}
    for idx in group_indices:
        fx, fy, fz = fracs[idx]
        fracs[idx] = [(fx + dx) % 1.0, (fy + dy) % 1.0, (fz + dz) % 1.0]
    return Structure(lattice, species, fracs, site_properties=site_props)


def get_site_occ_dict(site) -> Dict[str, float]:
    return {str(el): float(frac) for el, frac in site.species.items()}


def normalize_with_vac(occ: Dict[str, float]) -> Dict[str, float]:
    total = sum(occ.values())
    if total > 1.0:
        s = total if total > 0 else 1.0
        occ = {k: v / s for k, v in occ.items()}
        vac = 0.0
    else:
        vac = 1.0 - total
    occ2 = dict(occ)
    occ2['Vac'] = vac
    return occ2


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def structure_with_mixed_occupancy(struct: Structure, site_index: int, occ_mix: Dict[str, float]) -> Structure:
    lattice = struct.lattice
    fracs = [s.frac_coords.copy() for s in struct]
    site_props = {k: list(v) for k, v in struct.site_properties.items()}
    new_species = []
    for i, s in enumerate(struct):
        if i != site_index:
            new_species.append(s.species)
            continue
        occ = {k: float(v) for k, v in occ_mix.items() if k != 'Vac'}
        occ = {k: clamp01(v) for k, v in occ.items()}
        occ2 = normalize_with_vac(occ)
        d = {k: v for k, v in occ2.items() if k != 'Vac' and v > 1e-12}
        new_species.append(d)
    return Structure(lattice, new_species, fracs, site_properties=site_props)


def list_mixed_sites(struct: Structure) -> List[int]:
    idxs = []
    for i, s in enumerate(struct.sites):
        d = get_site_occ_dict(s)
        if len(d) > 1:
            idxs.append(i)
        elif len(d) == 1 and abs(1.0 - list(d.values())[0]) > 1e-6:
            idxs.append(i)
    return idxs


def normalized_composition_vector(struct: Structure, target_keys: List[str]) -> np.ndarray:
    cdict = struct.composition.get_el_amt_dict()
    vec = np.array([float(cdict.get(k, 0.0)) for k in target_keys], dtype=float)
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec


def stoich_penalty_for_phase(struct: Structure, target: Dict[str, float]) -> float:
    keys = sorted(set(target.keys()))
    tvec = np.array([float(target[k]) for k in keys], dtype=float)
    ts = tvec.sum()
    tvec = (tvec / ts) if ts > 0 else tvec
    cvec = normalized_composition_vector(struct, keys)
    diff = cvec - tvec
    return float(np.dot(diff, diff))

# ===========================
# RL data structures
# ===========================
@dataclass
class RefinementContext:
    x_grid: np.ndarray
    y_obs: np.ndarray
    wavelength: float
    phase_structs: Dict[str, Structure]
    phase_dw_dicts: Dict[str, Dict[str, float]]
    tch_params: Dict[str, float]
    po_r: Dict[str, float]
    po_axes: Dict[str, Tuple[int, int, int]]
    yfit: Optional[np.ndarray] = None
    fr: Optional[np.ndarray] = None
    sf: Optional[np.ndarray] = None
    profiles: Optional[List[np.ndarray]] = None
    best_score: float = 1e9
    best_rwp: float = 1e9
    global_zshift: float = 0.0
    lambda_stoich: float = 0.0
    stoich_phase_key: Optional[str] = None
    stoich_target: Optional[Dict[str, float]] = None
    device: Any = None
    bg_degree: int = 5
    broad_base: float = 0.08
    enable_po: bool = False
    stage_name: str = '初始化'
    stage_id: int = 0
    loop_idx: int = 0
    total_loops: int = 1
    improved_in_loop: bool = False
    main_bias: float = 0.0


@dataclass
class StepState:
    cell_step: float
    angle_step: float
    tch_step: float
    po_step: float
    pos_step: float
    mix_step: float
    min_cell_step: float
    min_angle_step: float
    min_tch_step: float
    min_po_step: float
    min_pos_step: float
    min_mix_step: float
    @property
    def b_step(self) -> float:
        return self.pos_step * 30.0


@dataclass
class StageConfig:
    name: str
    stage_id: int
    loops: int
    cell_scale: float
    angle_scale: float
    tch_scale: float
    pos_scale: float
    mix_scale: float
    po_scale: float
    xyz_mode: str
    stage_patience: int = 8
    stage_min_delta: float = 0.01
    idle_patience: int = 3
    step_exhaust_patience: int = 2


@dataclass
class Candidate:
    module_name: str
    phase_key: Optional[str] = None
    target_type: Optional[str] = None
    target_id: Optional[Any] = None
    axis: Optional[str] = None
    direction: float = 1.0
    step_scale: float = 1.0
    payload: Optional[dict] = None
    rl_score: float = 0.0


@dataclass
class CandidateEvalResult:
    accepted: bool
    score_try: float
    rwp_try: float
    new_phase_structs: Optional[Dict[str, Structure]] = None
    new_phase_dw_dicts: Optional[Dict[str, Dict[str, float]]] = None
    new_tch_params: Optional[Dict[str, float]] = None
    new_po_r: Optional[Dict[str, float]] = None
    yfit_try: Optional[np.ndarray] = None
    fr_try: Optional[np.ndarray] = None
    sf_try: Optional[np.ndarray] = None
    profiles_try: Optional[List[np.ndarray]] = None


def format_candidate_adjustment(cand: Candidate) -> str:
    phase = os.path.basename(cand.phase_key) if cand.phase_key else ''
    if cand.module_name == 'cell':
        return f"cell {phase}.{cand.axis} {cand.direction * cand.step_scale:+.1f}step"
    if cand.module_name == 'angle':
        return f"angle {phase}.{cand.target_id} {cand.direction * cand.step_scale:+.1f}step"
    if cand.module_name == 'tch':
        return f"tch {cand.target_id} {cand.direction * cand.step_scale:+.1f}step"
    if cand.module_name == 'po':
        return f"po {phase} {cand.direction * cand.step_scale:+.1f}step"
    if cand.module_name == 'xyz':
        gid = cand.target_id if cand.target_id is not None else '?'
        return f"xyz {phase}.group{gid}.{cand.axis} {cand.direction * cand.step_scale:+.1f}step"
    if cand.module_name == 'biso':
        return f"biso {phase}.{cand.target_id} {cand.direction * cand.step_scale:+.1f}step"
    if cand.module_name == 'occ':
        sid = cand.target_id if cand.target_id is not None else '?'
        elem = cand.payload.get('element', '?') if cand.payload else '?'
        return f"occ {phase}.site{sid}.{elem} {cand.direction * cand.step_scale:+.1f}step"
    return cand.module_name


def format_corr_summary(files: List[str], corrs: List[float]) -> str:
    return ', '.join(f"{os.path.basename(f)}:{c:.3f}" for f, c in zip(files, corrs))


class ProfileCacheManager:
    def __init__(self, max_cache=256, num_workers=None):
        self.cache = OrderedDict()
        self.max_cache = max_cache
        self.pool = ProcessPoolExecutor(max_workers=num_workers or os.cpu_count())

    def phase_profile_key(self, phase_key, st, tpars, po_r_dict, po_axes, wavelength, enable_po, broad_base, dw_dict):
        tkey = (round(tpars['U'], 8), round(tpars['V'], 8), round(tpars['W'], 8), round(tpars['X'], 8), round(tpars['Y'], 8))
        dw_key = tuple(sorted((str(el), round(float(b), 8)) for el, b in (dw_dict or {}).items()))
        return (
            phase_key,
            id(st),
            tkey,
            round(po_r_dict.get(phase_key, 1.0), 8),
            po_axes.get(phase_key, (0, 0, 1)),
            round(wavelength, 8),
            bool(enable_po),
            round(broad_base, 8),
            dw_key,
        )

    def make_profiles(self, ctx: RefinementContext, phase_structs, tpars, po_r_dict, dw_dicts):
        profs = [None] * len(phase_structs)
        items_to_compute, need_idx, cache_keys = [], [], []
        keys = list(phase_structs.keys())
        for i, key in enumerate(keys):
            st = phase_structs[key]
            phase_dw = dw_dicts.get(key, {})
            ck = self.phase_profile_key(key, st, tpars, po_r_dict, ctx.po_axes, ctx.wavelength, ctx.enable_po, ctx.broad_base, phase_dw)
            cache_keys.append(ck)
            if ck in self.cache:
                profs[i] = self.cache[ck]
                self.cache.move_to_end(ck)
            else:
                items_to_compute.append((
                    key, st.as_dict(), ctx.x_grid, ctx.wavelength, tpars,
                    ctx.po_axes.get(key, (0, 0, 1)), po_r_dict.get(key, 1.0),
                    ctx.broad_base, ctx.enable_po, phase_dw
                ))
                need_idx.append(i)
        if items_to_compute:
            results = list(self.pool.map(_synth_profile_worker, items_to_compute))
            got = {k: y for (k, y) in results}
            for i in need_idx:
                ck, key = cache_keys[i], keys[i]
                profs[i] = self.cache[ck] = got[key]
                while len(self.cache) > self.max_cache:
                    self.cache.popitem(last=False)
        return profs


def build_penalty_vectors(ctx: RefinementContext, phase_structs: Dict[str, Structure], verbose: bool = False):
    pen_list, alpha_list = None, None
    if ctx.lambda_stoich > 0.0 and ctx.stoich_target is not None:
        pen_list, alpha_list = [], []
        for k, st in phase_structs.items():
            pen = stoich_penalty_for_phase(st, ctx.stoich_target)
            is_main = ctx.stoich_phase_key and (os.path.basename(k) == os.path.basename(ctx.stoich_phase_key))
            alpha_list.append(0.1 if is_main else 0.8)
            pen_list.append(pen)
        if verbose and not hasattr(build_penalty_vectors, '_stoich_printed'):
            log_event(f"\n📘 [StoichPenalty] λ_stoich = {ctx.lambda_stoich:.3f}")
            log_event(f"👉 已识别主相为：{ctx.stoich_phase_key}")
            log_event('🧩 各相的化学计量约束权重：')
            for k in phase_structs.keys():
                is_main = ctx.stoich_phase_key and (os.path.basename(k) == os.path.basename(ctx.stoich_phase_key))
                w = 0.1 if is_main else 0.8
                tag = '主相' if is_main else '杂相'
                log_event(f"   ├─ {os.path.basename(k):<25s} | 类型: {tag:<3s} | λ_phase = {w*ctx.lambda_stoich:.3f}")
            log_event('-------------------------------------------------------')
            build_penalty_vectors._stoich_printed = True
    return pen_list, alpha_list


def inner_once(ctx: RefinementContext, cache_mgr: ProfileCacheManager, phase_structs: Dict[str, Structure], tch_params: Dict[str, float], po_r_dict: Dict[str, float], *, freeze_scale=False, freeze_zero_shift=True, zshift_val=0.0, dw_dicts_override=None, lambda_stoich=None, verbose_stoich=False):
    dw_to_use = dw_dicts_override if dw_dicts_override is not None else ctx.phase_dw_dicts
    profiles = cache_mgr.make_profiles(ctx, phase_structs, tch_params, po_r_dict, dw_to_use)
    lam = ctx.lambda_stoich if lambda_stoich is None else lambda_stoich
    pen_list, alpha_list = (build_penalty_vectors(ctx, phase_structs, verbose=verbose_stoich) if lam > 0.0 else (None, None))
    mode = 'fit' if lam == 0.0 else 'full'
    stage_main_bias = ctx.main_bias if int(getattr(ctx, 'stage_id', 0)) == 0 else 0.0
    yfit, fr, sf, Rwp, z_out = torch_refine(
        ctx.y_obs, profiles, device=ctx.device, bg_degree=ctx.bg_degree,
        freeze_scale=freeze_scale, epochs=250 if '粗调' in ctx.stage_name else 200 if '微调' in ctx.stage_name else 150,
        lbfgs_lr=0.30 if '粗调' in ctx.stage_name or '微调' in ctx.stage_name else 0.25,
        lr=5e-3, weight_decay=1e-4,
        main_bias=stage_main_bias,
        stoich_penalty_per_phase=pen_list,
        stoich_phase_weights=alpha_list,
        lambda_stoich=lam,
        mode=mode,
        freeze_zero_shift=freeze_zero_shift,
        init_zero_shift=zshift_val,
        log_improvements=False,
        log_context=f"{ctx.stage_name}",
    )
    stoich_term = float(np.sum(np.array(pen_list) * np.array(alpha_list) * np.array(fr) * np.array(sf))) if pen_list else 0.0
    score = _ALT_SCORE_W_DATA * Rwp + _ALT_SCORE_W_STOICH * stoich_term
    RWP_LOG.append(float(Rwp))
    FRAC_LOG.append([float(x) for x in fr])
    SCALE_LOG.append([float(x) for x in sf])
    STEP_LOG.append(ctx.stage_name)
    return score, yfit, fr, sf, Rwp, profiles, z_out


class CandidateScorerNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class RLRanker:
    def __init__(self, state_dim: int = 32, hidden_dim: int = 128, lr: float = 1e-3,
                 gamma: float = 0.95, epsilon: float = 0.05, memory_size: int = 5000,
                 batch_size: int = 64, target_replace_iter: int = 50, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = int(state_dim)
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_replace_iter = target_replace_iter
        self.learn_step_counter = 0
        self.policy_net = CandidateScorerNet(self.state_dim, hidden_dim).to(self.device)
        self.target_net = CandidateScorerNet(self.state_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

    def encode_global_state(self, ctx: RefinementContext, steps: StepState, stage: StageConfig) -> np.ndarray:
        fr_main = float(ctx.fr[0]) if ctx.fr is not None and len(ctx.fr) > 0 else 1.0
        sf_main = float(ctx.sf[0]) if ctx.sf is not None and len(ctx.sf) > 0 else 1.0
        return np.array([
            stage.stage_id / 2.0,
            ctx.loop_idx / max(1, ctx.total_loops),
            min(ctx.best_score / 100.0, 10.0),
            min(ctx.best_rwp / 100.0, 10.0),
            ctx.global_zshift,
            steps.cell_step, steps.angle_step, steps.tch_step,
            steps.po_step, steps.pos_step, steps.mix_step,
            fr_main, sf_main,
            len(ctx.phase_structs) / 10.0,
        ], dtype=np.float32)

    def encode_candidate_state(self, ctx: RefinementContext, steps: StepState, stage: StageConfig, cand: Candidate) -> np.ndarray:
        mod_map = {'cell': 0, 'angle': 1, 'tch': 2, 'po': 3, 'xyz': 4, 'biso': 5, 'occ': 6}
        mod_id = mod_map[cand.module_name] / 6.0
        phase_frac = 0.0
        if ctx.fr is not None and cand.phase_key is not None:
            keys = list(ctx.phase_structs.keys())
            if cand.phase_key in keys:
                idx = keys.index(cand.phase_key)
                if idx < len(ctx.fr):
                    phase_frac = float(ctx.fr[idx])
        axis_map = {'a': 0.0, 'b': 0.2, 'c': 0.4, 'x': 0.6, 'y': 0.8, 'z': 1.0, None: -1.0}
        return np.array([mod_id, phase_frac, axis_map.get(cand.axis, -1.0), cand.direction, cand.step_scale], dtype=np.float32)

    def _fix_state_dim(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.size < self.state_dim:
            vec = np.concatenate([vec, np.zeros(self.state_dim - vec.size, dtype=np.float32)])
        elif vec.size > self.state_dim:
            vec = vec[:self.state_dim]
        return vec

    def build_state_vec(self, ctx: RefinementContext, steps: StepState, stage: StageConfig, cand: Candidate) -> np.ndarray:
        vec = np.concatenate([self.encode_global_state(ctx, steps, stage), self.encode_candidate_state(ctx, steps, stage, cand)])
        return self._fix_state_dim(vec)

    def score_candidates(self, ctx: RefinementContext, steps: StepState, stage: StageConfig, candidates: List[Candidate]) -> List[Candidate]:
        scored = []
        for cand in candidates:
            state_vec = self.build_state_vec(ctx, steps, stage, cand)
            with torch.no_grad():
                s = self.policy_net(torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0))
                cand.rl_score = float(s.item())
            if random.random() < self.epsilon:
                cand.rl_score += random.uniform(-0.05, 0.05)
            scored.append(cand)
        scored.sort(key=lambda x: x.rl_score, reverse=True)
        return scored

    def store_transition(self, s, r, s_next):
        self.memory.append((self._fix_state_dim(s), float(r), self._fix_state_dim(s_next)))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_counter += 1
        batch = random.sample(self.memory, self.batch_size)
        states, rewards, next_states = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        q_eval = self.policy_net(states)
        with torch.no_grad():
            q_next = self.target_net(next_states)
            q_target = rewards + self.gamma * q_next
        loss = nn.MSELoss()(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_after_module(self, state_before: np.ndarray, score_before: float, score_after: float, state_after: np.ndarray, module_name: str):
        reward = float(np.clip(score_before - score_after, -1.0, 1.0))
        self.store_transition(state_before, reward, state_after)
        self.learn()


class CandidateBuilder:
    def build_cell_candidates(self, ctx, steps, stage):
        if steps.cell_step < steps.min_cell_step:
            return []
        cands = []
        for phase_key in ctx.phase_structs.keys():
            for axis in ['a', 'b', 'c']:
                for direction in [-1.0, +1.0]:
                    for sc in [0.5, 1.0, 1.5]:
                        cands.append(Candidate('cell', phase_key, 'cell_axis', axis, axis, direction, sc))
        return cands

    def build_angle_candidates(self, ctx, steps, stage):
        if steps.angle_step < steps.min_angle_step:
            return []
        cands = []
        for phase_key in ctx.phase_structs.keys():
            for axis in ['alpha', 'beta', 'gamma']:
                for direction in [-1.0, +1.0]:
                    for sc in [0.5, 1.0, 1.5]:
                        cands.append(Candidate('angle', phase_key, 'cell_angle', axis, None, direction, sc))
        return cands

    def build_tch_candidates(self, ctx, steps, stage):
        if steps.tch_step < steps.min_tch_step:
            return []
        cands = []
        for param in ['U', 'V', 'W', 'X', 'Y']:
            for direction in [-1.0, +1.0]:
                for sc in [0.5, 1.0, 1.5]:
                    cands.append(Candidate('tch', None, 'tch_param', param, None, direction, sc))
        return cands

    def build_po_candidates(self, ctx, steps, stage):
        if steps.po_step < steps.min_po_step:
            return []
        cands = []
        for phase_key in ctx.phase_structs.keys():
            for direction in [-1.0, +1.0]:
                for sc in [0.5, 1.0, 1.5]:
                    cands.append(Candidate('po', phase_key, 'po_r', 'po_r', None, direction, sc))
        return cands

    def get_xyz_groups_for_stage(self, ctx, phase_key, stage, fixed_groups_map):
        st = ctx.phase_structs[phase_key]
        if stage.xyz_mode == 'fixed':
            return fixed_groups_map[phase_key]
        elif stage.xyz_mode == 'symmetry':
            return get_equivalent_groups(st)
        elif stage.xyz_mode == 'atomwise':
            return [[i] for i in range(len(st.sites))]
        raise ValueError(f'Unknown xyz_mode: {stage.xyz_mode}')

    def build_xyz_candidates(self, ctx, steps, stage, fixed_groups_map):
        if steps.pos_step < steps.min_pos_step:
            return []
        cands = []
        for phase_key in ctx.phase_structs.keys():
            groups = self.get_xyz_groups_for_stage(ctx, phase_key, stage, fixed_groups_map)
            for gidx, group in enumerate(groups):
                for axis in ['x', 'y', 'z']:
                    for direction in [-1.0, +1.0]:
                        for sc in [0.5, 1.0, 1.5]:
                            cands.append(Candidate('xyz', phase_key, 'atom_group', gidx, axis, direction, sc, {'group': group}))
        return cands

    def build_biso_candidates(self, ctx, steps, stage):
        if steps.pos_step < steps.min_pos_step:
            return []
        cands = []
        for phase_key, dw_dict in ctx.phase_dw_dicts.items():
            for elem in dw_dict.keys():
                for direction in [-1.0, +1.0]:
                    for sc in [0.5, 1.0, 1.5]:
                        cands.append(Candidate('biso', phase_key, 'element', elem, None, direction, sc))
        return cands

    def build_occ_candidates(self, ctx, steps, stage):
        if steps.mix_step < steps.min_mix_step:
            return []
        cands = []
        for phase_key, st in ctx.phase_structs.items():
            site_indices = list_mixed_sites(st)
            for sidx in site_indices:
                occ0 = normalize_with_vac(get_site_occ_dict(st.sites[sidx]))
                for elem in occ0.keys():
                    for direction in [-1.0, +1.0]:
                        for sc in [0.5, 1.0, 1.5]:
                            cands.append(Candidate('occ', phase_key, 'site_occ', sidx, None, direction, sc, {'element': elem}))
        return cands


def accept_if_better(score_try: float, best_score: float, tol: float = 1e-6) -> bool:
    return bool(score_try + tol < best_score)


def accept_fit_if_better(rwp_try: float, best_rwp: float, tol: float = 1e-6) -> bool:
    return bool(rwp_try + tol < best_rwp)


def clone_phase_dw_dicts(phase_dw_dicts):
    return {k: dict(v) for k, v in phase_dw_dicts.items()}


def eval_cell_candidate(ctx, cache_mgr, cand, steps):
    phase_key = cand.phase_key
    st0 = ctx.phase_structs[phase_key]
    a, b, c, alpha, beta, gamma = get_cell_params(st0)
    delta = steps.cell_step * cand.step_scale * cand.direction
    a_try, b_try, c_try = a, b, c
    if cand.axis == 'a':
        a_try = max(1e-4, a * (1.0 + delta))
    elif cand.axis == 'b':
        b_try = max(1e-4, b * (1.0 + delta))
    elif cand.axis == 'c':
        c_try = max(1e-4, c * (1.0 + delta))
    st_try = sync_biso(set_cell_abc_angles(st0, a_try, b_try, c_try, alpha, beta, gamma))
    phase_structs_try = dict(ctx.phase_structs)
    phase_structs_try[phase_key] = st_try
    score_try, yfit_try, fr_try, sf_try, rwp_try, profiles_try, _ = inner_once(ctx, cache_mgr, phase_structs_try, ctx.tch_params, ctx.po_r, freeze_zero_shift=True, zshift_val=ctx.global_zshift, lambda_stoich=0.0)
    return CandidateEvalResult(accept_fit_if_better(rwp_try, ctx.best_rwp), rwp_try, rwp_try, new_phase_structs=phase_structs_try, yfit_try=yfit_try, fr_try=fr_try, sf_try=sf_try, profiles_try=profiles_try)


def eval_angle_candidate(ctx, cache_mgr, cand, steps):
    phase_key = cand.phase_key
    st0 = ctx.phase_structs[phase_key]
    a, b, c, alpha, beta, gamma = get_cell_params(st0)
    delta = steps.angle_step * cand.step_scale * cand.direction
    alpha_try, beta_try, gamma_try = alpha, beta, gamma
    if cand.target_id == 'alpha':
        alpha_try = float(np.clip(alpha + delta, ANG_MIN, ANG_MAX))
    elif cand.target_id == 'beta':
        beta_try = float(np.clip(beta + delta, ANG_MIN, ANG_MAX))
    elif cand.target_id == 'gamma':
        gamma_try = float(np.clip(gamma + delta, ANG_MIN, ANG_MAX))
    st_try = sync_biso(set_cell_abc_angles(st0, a, b, c, alpha_try, beta_try, gamma_try))
    phase_structs_try = dict(ctx.phase_structs)
    phase_structs_try[phase_key] = st_try
    score_try, yfit_try, fr_try, sf_try, rwp_try, profiles_try, _ = inner_once(ctx, cache_mgr, phase_structs_try, ctx.tch_params, ctx.po_r, freeze_zero_shift=True, zshift_val=ctx.global_zshift, lambda_stoich=0.0)
    return CandidateEvalResult(accept_fit_if_better(rwp_try, ctx.best_rwp), rwp_try, rwp_try, new_phase_structs=phase_structs_try, yfit_try=yfit_try, fr_try=fr_try, sf_try=sf_try, profiles_try=profiles_try)


def eval_tch_candidate(ctx, cache_mgr, cand, steps):
    param = cand.target_id
    delta = steps.tch_step * cand.step_scale * cand.direction
    tpars_try = dict(ctx.tch_params)
    tpars_try[param] = tpars_try[param] * (1.0 + delta)
    tpars_try['U'] = float(np.clip(tpars_try['U'], 0.0001, 0.15))
    tpars_try['V'] = float(np.clip(tpars_try['V'], -0.10, 0.10))
    tpars_try['W'] = float(np.clip(tpars_try['W'], 0.0001, 0.15))
    tpars_try['X'] = float(np.clip(tpars_try['X'], 0.0001, 0.15))
    tpars_try['Y'] = float(np.clip(tpars_try['Y'], 0.0001, 0.25))
    score_try, yfit_try, fr_try, sf_try, rwp_try, profiles_try, _ = inner_once(ctx, cache_mgr, ctx.phase_structs, tpars_try, ctx.po_r, freeze_zero_shift=True, zshift_val=ctx.global_zshift, lambda_stoich=0.0)
    return CandidateEvalResult(accept_fit_if_better(rwp_try, ctx.best_rwp), rwp_try, rwp_try, new_tch_params=tpars_try, yfit_try=yfit_try, fr_try=fr_try, sf_try=sf_try, profiles_try=profiles_try)


def eval_po_candidate(ctx, cache_mgr, cand, steps):
    phase_key = cand.phase_key
    r0 = ctx.po_r[phase_key]
    delta = steps.po_step * cand.step_scale * cand.direction
    r_try = float(np.clip(r0 * (1.0 + delta), 0.2, 5.0))
    po_r_try = dict(ctx.po_r)
    po_r_try[phase_key] = r_try
    score_try, yfit_try, fr_try, sf_try, rwp_try, profiles_try, _ = inner_once(ctx, cache_mgr, ctx.phase_structs, ctx.tch_params, po_r_try, freeze_zero_shift=True, zshift_val=ctx.global_zshift, lambda_stoich=0.0)
    return CandidateEvalResult(accept_fit_if_better(rwp_try, ctx.best_rwp), rwp_try, rwp_try, new_po_r=po_r_try, yfit_try=yfit_try, fr_try=fr_try, sf_try=sf_try, profiles_try=profiles_try)


def eval_xyz_candidate(ctx, cache_mgr, cand, steps):
    phase_key = cand.phase_key
    st0 = ctx.phase_structs[phase_key]
    group = cand.payload['group']
    delta = steps.pos_step * cand.step_scale * cand.direction
    dx = dy = dz = 0.0
    if cand.axis == 'x':
        dx = delta
    elif cand.axis == 'y':
        dy = delta
    elif cand.axis == 'z':
        dz = delta
    st_try = sync_biso(structure_with_shifted_group(st0, group, dx, dy, dz))
    phase_structs_try = dict(ctx.phase_structs)
    phase_structs_try[phase_key] = st_try
    score_try, yfit_try, fr_try, sf_try, rwp_try, profiles_try, _ = inner_once(ctx, cache_mgr, phase_structs_try, ctx.tch_params, ctx.po_r, freeze_zero_shift=True, zshift_val=ctx.global_zshift, lambda_stoich=0.0)
    return CandidateEvalResult(accept_fit_if_better(rwp_try, ctx.best_rwp), rwp_try, rwp_try, new_phase_structs=phase_structs_try, yfit_try=yfit_try, fr_try=fr_try, sf_try=sf_try, profiles_try=profiles_try)


def eval_biso_candidate(ctx, cache_mgr, cand, steps):
    phase_key = cand.phase_key
    elem = cand.target_id
    dw_dicts_try = clone_phase_dw_dicts(ctx.phase_dw_dicts)
    B0 = float(dw_dicts_try[phase_key][elem])
    delta = steps.b_step * cand.step_scale * cand.direction
    B_try = float(np.clip(B0 + delta, 0.039, 3.94))
    dw_dicts_try[phase_key][elem] = B_try
    score_try, yfit_try, fr_try, sf_try, rwp_try, profiles_try, _ = inner_once(ctx, cache_mgr, ctx.phase_structs, ctx.tch_params, ctx.po_r, freeze_zero_shift=True, zshift_val=ctx.global_zshift, dw_dicts_override=dw_dicts_try, lambda_stoich=0.0)
    return CandidateEvalResult(accept_fit_if_better(rwp_try, ctx.best_rwp), rwp_try, rwp_try, new_phase_dw_dicts=dw_dicts_try, yfit_try=yfit_try, fr_try=fr_try, sf_try=sf_try, profiles_try=profiles_try)


def eval_occ_candidate(ctx, cache_mgr, cand, steps):
    phase_key = cand.phase_key
    site_idx = cand.target_id
    elem = cand.payload['element']
    st0 = ctx.phase_structs[phase_key]
    occ0 = normalize_with_vac(get_site_occ_dict(st0.sites[site_idx]))
    occ_try = dict(occ0)
    delta = steps.mix_step * cand.step_scale * cand.direction
    occ_try[elem] = clamp01(occ_try.get(elem, 0.0) + delta)
    occ_try = normalize_with_vac({k: v for k, v in occ_try.items() if k != 'Vac'})
    st_try = sync_biso(structure_with_mixed_occupancy(st0, site_idx, occ_try))
    phase_structs_try = dict(ctx.phase_structs)
    phase_structs_try[phase_key] = st_try
    score_try, yfit_try, fr_try, sf_try, rwp_try, profiles_try, _ = inner_once(ctx, cache_mgr, phase_structs_try, ctx.tch_params, ctx.po_r, freeze_scale=True, freeze_zero_shift=True, zshift_val=ctx.global_zshift)
    return CandidateEvalResult(accept_if_better(score_try, ctx.best_score), score_try, rwp_try, new_phase_structs=phase_structs_try, yfit_try=yfit_try, fr_try=fr_try, sf_try=sf_try, profiles_try=profiles_try)


def shrink_cell_step(steps): steps.cell_step *= 0.8

def shrink_angle_step(steps): steps.angle_step *= 0.8

def shrink_tch_step(steps): steps.tch_step *= 0.8

def shrink_po_step(steps): steps.po_step *= 0.8

def shrink_pos_step(steps): steps.pos_step *= 0.8

def shrink_mix_step(steps): steps.mix_step *= 0.8


class ModuleExecutor:
    def __init__(self, candidate_builder, rl_ranker, cache_mgr):
        self.builder = candidate_builder
        self.rl_ranker = rl_ranker
        self.cache_mgr = cache_mgr

    def _apply_result_to_ctx(self, ctx, result):
        if result.new_phase_structs is not None:
            ctx.phase_structs = result.new_phase_structs
        if result.new_phase_dw_dicts is not None:
            ctx.phase_dw_dicts = result.new_phase_dw_dicts
        if result.new_tch_params is not None:
            ctx.tch_params = result.new_tch_params
        if result.new_po_r is not None:
            ctx.po_r = result.new_po_r
        ctx.best_score = result.score_try
        ctx.best_rwp = result.rwp_try
        ctx.yfit = result.yfit_try
        ctx.fr = result.fr_try
        ctx.sf = result.sf_try
        ctx.profiles = result.profiles_try
        ctx.improved_in_loop = True

    def _run_module_generic(self, ctx, steps, stage, module_name, candidates, evaluator_fn, shrink_fn):
        if len(candidates) == 0:
            return ctx
        score_before = ctx.best_score
        state_before = None
        if self.rl_ranker is not None:
            state_before = self.rl_ranker.encode_global_state(ctx, steps, stage)
            candidates = self.rl_ranker.score_candidates(ctx, steps, stage, candidates)
        improved = False
        for cand in candidates:
            prev_rwp = ctx.best_rwp
            result = evaluator_fn(ctx, self.cache_mgr, cand, steps)
            if result.accepted:
                improved = True
                adjustment = format_candidate_adjustment(cand)
                self._apply_result_to_ctx(ctx, result)
                log_event(
                    f"[{stage.name}] {adjustment} | Rwp {prev_rwp:.2f}% -> {ctx.best_rwp:.2f}%"
                )
        if not improved:
            shrink_fn(steps)
        if self.rl_ranker is not None and state_before is not None:
            state_after = self.rl_ranker.encode_global_state(ctx, steps, stage)
            self.rl_ranker.update_after_module(state_before, score_before, ctx.best_score, state_after, module_name)
        return ctx

    def run_cell_module(self, ctx, steps, stage):
        return self._run_module_generic(ctx, steps, stage, 'cell', self.builder.build_cell_candidates(ctx, steps, stage), eval_cell_candidate, shrink_cell_step)

    def run_angle_module(self, ctx, steps, stage):
        return self._run_module_generic(ctx, steps, stage, 'angle', self.builder.build_angle_candidates(ctx, steps, stage), eval_angle_candidate, shrink_angle_step)

    def run_tch_module(self, ctx, steps, stage):
        return self._run_module_generic(ctx, steps, stage, 'tch', self.builder.build_tch_candidates(ctx, steps, stage), eval_tch_candidate, shrink_tch_step)

    def run_po_module(self, ctx, steps, stage):
        return self._run_module_generic(ctx, steps, stage, 'po', self.builder.build_po_candidates(ctx, steps, stage), eval_po_candidate, shrink_po_step)

    def run_b_loop_module(self, ctx, steps, stage, fixed_groups_map):
        score_before = ctx.best_score
        state_before = self.rl_ranker.encode_global_state(ctx, steps, stage) if self.rl_ranker is not None else None
        bloop_improved = False
        loop_tag = f"B-Loop {ctx.loop_idx + 1:03d}"

        xyz_candidates = self.builder.build_xyz_candidates(ctx, steps, stage, fixed_groups_map)
        if xyz_candidates:
            if self.rl_ranker is not None:
                xyz_candidates = self.rl_ranker.score_candidates(ctx, steps, stage, xyz_candidates)
            for cand in xyz_candidates:
                prev_rwp = ctx.best_rwp
                result = eval_xyz_candidate(ctx, self.cache_mgr, cand, steps)
                if result.accepted:
                    bloop_improved = True
                    self._apply_result_to_ctx(ctx, result)
                    gid = cand.target_id if cand.target_id is not None else '?'
                    delta = steps.pos_step * cand.step_scale * cand.direction
                    log_event(
                        f"[{stage.name} | {loop_tag}] ✅ atoms {os.path.basename(cand.phase_key)} "
                        f"group#{gid} {cand.axis} {delta:+.4f} → Rwp={ctx.best_rwp:.2f}%  "
                        f"pos_step={steps.pos_step:.4f} mix_step={steps.mix_step:.3f}"
                    )

        biso_candidates = self.builder.build_biso_candidates(ctx, steps, stage)
        if biso_candidates:
            if self.rl_ranker is not None:
                biso_candidates = self.rl_ranker.score_candidates(ctx, steps, stage, biso_candidates)
            best_biso_result = None
            best_biso_cand = None
            best_biso_score = ctx.best_rwp
            for cand in biso_candidates:
                result = eval_biso_candidate(ctx, self.cache_mgr, cand, steps)
                if result.rwp_try + 1e-4 < best_biso_score:
                    best_biso_score = result.rwp_try
                    best_biso_result = result
                    best_biso_cand = cand
            if best_biso_result is not None:
                bloop_improved = True
                self._apply_result_to_ctx(ctx, best_biso_result)
                sign = '+' if best_biso_cand.direction > 0 else '-'
                log_event(
                    f"[{stage.name} | {loop_tag}] ✅ Biso ({best_biso_cand.target_id} {sign}) optimized → Rwp={ctx.best_rwp:.2f}%"
                )

        occ_candidates = self.builder.build_occ_candidates(ctx, steps, stage)
        if occ_candidates:
            if self.rl_ranker is not None:
                occ_candidates = self.rl_ranker.score_candidates(ctx, steps, stage, occ_candidates)
            best_occ_result = None
            best_occ_cand = None
            best_occ_score = ctx.best_rwp
            for cand in occ_candidates:
                result = eval_occ_candidate(ctx, self.cache_mgr, cand, steps)
                if result.rwp_try + 1e-4 < best_occ_score:
                    best_occ_score = result.rwp_try
                    best_occ_result = result
                    best_occ_cand = cand
            if best_occ_result is not None:
                bloop_improved = True
                self._apply_result_to_ctx(ctx, best_occ_result)
                elem = best_occ_cand.payload.get('element', '?') if best_occ_cand.payload else '?'
                delta = steps.mix_step * best_occ_cand.step_scale * best_occ_cand.direction
                log_event(
                    f"[{stage.name} | {loop_tag}] ✅ mix {os.path.basename(best_occ_cand.phase_key)} "
                    f"site#{best_occ_cand.target_id} {elem} {delta:+.3f} → Rwp={ctx.best_rwp:.2f}%"
                )

        if not bloop_improved:
            old_pos, old_mix = steps.pos_step, steps.mix_step
            shrink_pos_step(steps)
            shrink_mix_step(steps)
            log_event(
                f"[{stage.name} | {loop_tag}] ↘️ 步长衰减：pos {old_pos:.4f}->{steps.pos_step:.4f} | "
                f"mix {old_mix:.3f}->{steps.mix_step:.3f} | Rwp={ctx.best_rwp:.2f}% | score={ctx.best_score:.4f}"
            )

        if self.rl_ranker is not None and state_before is not None:
            state_after = self.rl_ranker.encode_global_state(ctx, steps, stage)
            self.rl_ranker.update_after_module(state_before, score_before, ctx.best_score, state_after, 'bloop')
        return ctx


def build_fixed_groups_map(phase_structs: Dict[str, Structure]) -> Dict[str, List[List[int]]]:
    fixed_groups_map = {}
    for key, st in phase_structs.items():
        sga0 = SpacegroupAnalyzer(st, symprec=5e-4, angle_tolerance=3.0)
        symm0 = sga0.get_symmetrized_structure()
        fixed_groups_map[key] = [list(g) for g in symm0.equivalent_indices]
    return fixed_groups_map


def steps_near_min(steps: StepState, ratio: float = 1.05) -> bool:
    return (
        steps.cell_step <= steps.min_cell_step * ratio and
        steps.angle_step <= steps.min_angle_step * ratio and
        steps.tch_step <= steps.min_tch_step * ratio and
        steps.po_step <= steps.min_po_step * ratio and
        steps.pos_step <= steps.min_pos_step * ratio and
        steps.mix_step <= steps.min_mix_step * ratio
    )


class OuterRefineV44RL:
    def __init__(self, use_rl_guidance: bool = True, num_workers=None):
        self.use_rl_guidance = use_rl_guidance
        self.cache_mgr = ProfileCacheManager(max_cache=256, num_workers=num_workers)
        self.builder = CandidateBuilder()
        self.rl_ranker = RLRanker(state_dim=32) if use_rl_guidance else None
        self.executor = ModuleExecutor(self.builder, self.rl_ranker, self.cache_mgr)

    def build_stage_configs(self, stage_loops=(80, 80, 80)):
        return [
            StageConfig('粗调 (Stage 1)', 0, stage_loops[0], 1.5, 1.5, 1.8, 4.0, 2.5, 1.6, 'fixed',
                        stage_patience=6, stage_min_delta=0.03, idle_patience=2, step_exhaust_patience=2),
            StageConfig('微调 (Stage 2)', 1, stage_loops[1], 1.0, 1.2, 1.2, 3.0, 2.0, 1.2, 'fixed',
                        stage_patience=8, stage_min_delta=0.015, idle_patience=3, step_exhaust_patience=2),
            StageConfig('精调 (Stage 3)', 2, stage_loops[2], 0.7, 1.0, 0.9, 2.5, 1.8, 1.0, 'fixed',
                        stage_patience=10, stage_min_delta=0.008, idle_patience=3, step_exhaust_patience=2),
        ]

    def init_step_state(self, stage: StageConfig,
                        init_cell_step_frac=0.0025, min_cell_step_frac=0.0002,
                        init_angle_step_deg=0.3, min_angle_step_deg=0.05,
                        init_tch_step_frac=0.20, min_tch_step_frac=0.05,
                        init_pos_step=0.005, min_pos_step=0.002,
                        init_mix_step=0.08, min_mix_step=0.004,
                        init_po_step_frac=0.25, min_po_step_frac=0.05):
        return StepState(
            cell_step=init_cell_step_frac * stage.cell_scale,
            angle_step=init_angle_step_deg * stage.angle_scale,
            tch_step=init_tch_step_frac * stage.tch_scale,
            po_step=init_po_step_frac * stage.po_scale,
            pos_step=init_pos_step * stage.pos_scale,
            mix_step=init_mix_step * stage.mix_scale,
            min_cell_step=min_cell_step_frac,
            min_angle_step=min_angle_step_deg,
            min_tch_step=min_tch_step_frac,
            min_po_step=min_po_step_frac,
            min_pos_step=min_pos_step,
            min_mix_step=min_mix_step,
        )

    def instrument_align(self, ctx: RefinementContext):
        rwp_before = ctx.best_rwp if np.isfinite(ctx.best_rwp) else 1e9
        best_z_score = 1e9
        best_z_val = ctx.global_zshift
        for z_try in np.linspace(-0.5, 0.5, 21):
            score_inst, _, _, _, _, _, _ = inner_once(
                ctx, self.cache_mgr, ctx.phase_structs, ctx.tch_params, ctx.po_r,
                freeze_zero_shift=True, zshift_val=z_try
            )
            if score_inst < best_z_score:
                best_z_score = score_inst
                best_z_val = z_try
        score2, yfit2, fr2, sf2, rwp2, profiles2, z_final = inner_once(
            ctx, self.cache_mgr, ctx.phase_structs, ctx.tch_params, ctx.po_r,
            freeze_zero_shift=False, zshift_val=best_z_val,
            verbose_stoich=False,
        )
        ctx.global_zshift = z_final
        ctx.best_score = score2
        ctx.best_rwp = rwp2
        ctx.yfit = yfit2
        ctx.fr = fr2
        ctx.sf = sf2
        ctx.profiles = profiles2
        log_event(f"[{ctx.stage_name}] Instrument Align | zero_shift={best_z_val:+.4f}° | Rwp {rwp_before:.2f}% -> {ctx.best_rwp:.2f}%")
        log_event(f"[{ctx.stage_name}] 设备零点偏移已锁定 | zero_shift={ctx.global_zshift:+.4f}°")
        return ctx

    def run(self, ctx: RefinementContext, stage_loops=(100, 80, 70), init_cell_step_frac=0.0025, min_cell_step_frac=0.0002,
            init_angle_step_deg=0.3, min_angle_step_deg=0.05,
            init_tch_step_frac=0.20, min_tch_step_frac=0.05,
            init_pos_step=0.005, min_pos_step=0.002,
            init_mix_step=0.08, min_mix_step=0.004,
            init_po_step_frac=0.25, min_po_step_frac=0.05):
        score0, yfit0, fr0, sf0, rwp0, profiles0, _ = inner_once(
            ctx, self.cache_mgr, ctx.phase_structs, ctx.tch_params, ctx.po_r,
            freeze_zero_shift=True, zshift_val=ctx.global_zshift
        )
        ctx.best_score = score0
        ctx.best_rwp = rwp0
        ctx.yfit = yfit0
        ctx.fr = fr0
        ctx.sf = sf0
        ctx.profiles = profiles0
        log_event(f"初始化完成 | Rwp={ctx.best_rwp:.2f}%")

        fixed_groups_map = build_fixed_groups_map(ctx.phase_structs)
        total_groups = sum(len(v) for v in fixed_groups_map.values())
        log_event(f"Wyckoff分组已冻结 | phases={len(fixed_groups_map)} | groups={total_groups}")

        for stage in self.build_stage_configs(stage_loops):
            log_event(f"\n进入{stage.name} | loops={stage.loops}")
            ctx.stage_name = stage.name
            ctx.stage_id = stage.stage_id
            ctx.total_loops = stage.loops

            ctx = self.instrument_align(ctx)

            log_event(f"▶️ {stage.name} StepA：结构拟合阶段")
            score_a, yfit_a, fr_a, sf_a, rwp_a, profiles_a, _ = inner_once(
                ctx, self.cache_mgr, ctx.phase_structs, ctx.tch_params, ctx.po_r,
                freeze_scale=False,
                freeze_zero_shift=True,
                zshift_val=ctx.global_zshift,
                lambda_stoich=0.0,
                verbose_stoich=False,
            )
            ctx.best_score = score_a
            ctx.best_rwp = rwp_a
            ctx.yfit = yfit_a
            ctx.fr = fr_a
            ctx.sf = sf_a
            ctx.profiles = profiles_a
            log_event(f"✅ StepA 完成：Rwp={ctx.best_rwp:.2f}% | score={ctx.best_score:.2f}")
            ctx.best_score = ctx.best_rwp

            if stage.name == '粗调 (Stage 1)':
                log_event(f"{stage.name} StepB：化学计量修正（λ_stoich={ctx.lambda_stoich})")
                score_b, yfit_b, fr_b, sf_b, rwp_b, profiles_b, _ = inner_once(
                    ctx, self.cache_mgr, ctx.phase_structs, ctx.tch_params, ctx.po_r,
                    freeze_scale=False,
                    freeze_zero_shift=True,
                    zshift_val=ctx.global_zshift,
                    lambda_stoich=ctx.lambda_stoich,
                    verbose_stoich=True,
                )
                ctx.best_score = score_b
                ctx.best_rwp = rwp_b
                ctx.yfit = yfit_b
                ctx.fr = fr_b
                ctx.sf = sf_b
                ctx.profiles = profiles_b
                log_event(f"✅ StepB 完成：Rwp={ctx.best_rwp:.2f}% | score={ctx.best_score:.2f}")
                ctx.best_score = ctx.best_rwp
            else:
                log_event(f"⛔ 跳过 {stage.name} 的 StepB（避免破坏结构）")

            steps = self.init_step_state(stage, init_cell_step_frac, min_cell_step_frac,
                                         init_angle_step_deg, min_angle_step_deg,
                                         init_tch_step_frac, min_tch_step_frac,
                                         init_pos_step, min_pos_step,
                                         init_mix_step, min_mix_step,
                                         init_po_step_frac, min_po_step_frac)

            # A 段：先完整执行 cell / angle / tch / po
            for loop in range(stage.loops):
                ctx.loop_idx = loop
                ctx.improved_in_loop = False

                prev_a_steps = (steps.cell_step, steps.angle_step, steps.tch_step, steps.po_step)
                score_before_a = ctx.best_score

                ctx = self.executor.run_cell_module(ctx, steps, stage)
                ctx = self.executor.run_angle_module(ctx, steps, stage)
                ctx = self.executor.run_tch_module(ctx, steps, stage)
                ctx = self.executor.run_po_module(ctx, steps, stage)

                if ctx.best_score >= score_before_a - 1e-12:
                    if (steps.cell_step <= steps.min_cell_step and
                        steps.angle_step <= steps.min_angle_step and
                        steps.tch_step <= steps.min_tch_step and
                        steps.po_step <= steps.min_po_step):
                        log_event(f"[{stage.name} | A-Loop {loop + 1:03d}] ⛳ 已到最小步长阈值，结束 A 段。")
                        break
                    if (steps.cell_step, steps.angle_step, steps.tch_step, steps.po_step) != prev_a_steps:
                        log_event(
                            f"[{stage.name} | A-Loop {loop + 1:03d}] ↘️ 步长衰减："
                            f"cell {prev_a_steps[0]:.5f}->{steps.cell_step:.5f} | "
                            f"angle {prev_a_steps[1]:.3f}->{steps.angle_step:.3f} | "
                            f"tch {prev_a_steps[2]:.3f}->{steps.tch_step:.3f} | "
                            f"po {prev_a_steps[3]:.3f}->{steps.po_step:.3f} | "
                            f"Rwp={ctx.best_rwp:.2f}% | score={ctx.best_score:.2f}"
                        )

            # B 段：再执行 atoms / Biso / occ
            for loop in range(stage.loops):
                ctx.loop_idx = loop
                ctx.improved_in_loop = False

                prev_b_steps = (steps.pos_step, steps.mix_step)
                score_before_b = ctx.best_score

                ctx = self.executor.run_b_loop_module(ctx, steps, stage, fixed_groups_map)

                if ctx.best_score >= score_before_b - 1e-12:
                    if steps.pos_step <= steps.min_pos_step and steps.mix_step <= steps.min_mix_step:
                        log_event(f"[{stage.name} | B-Loop {loop + 1:03d}] ⛳ 已到最小步长阈值，结束 B 段。")
                        break
                    if (steps.pos_step, steps.mix_step) != prev_b_steps:
                        log_event(
                            f"[{stage.name} | B-Loop {loop + 1:03d}] ↘️ 步长衰减："
                            f"pos {prev_b_steps[0]:.4f}->{steps.pos_step:.4f} | "
                            f"mix {prev_b_steps[1]:.3f}->{steps.mix_step:.3f} | "
                            f"Rwp={ctx.best_rwp:.2f}% | score={ctx.best_score:.2f}"
                        )

            log_event(f"{stage.name} 完成 | Rwp={ctx.best_rwp:.2f}% | score={ctx.best_score:.2f}")

            if stage.name == '粗调 (Stage 1)':
                save_stage_outputs(ctx, '粗调阶段', 'stage1_output', 'Stage1_Refined', 'Stage 1 Refinement Result', 'Stage1 Fit')
            elif stage.name == '微调 (Stage 2)':
                save_stage_outputs(ctx, '微调阶段', 'stage2_output', 'Stage2_Refined', 'Stage 2 Refinement Result', 'Stage2 Fit')
            elif stage.name == '精调 (Stage 3)':
                save_stage_outputs(ctx, '精调阶段', 'stage3_output', 'Stage3_Refined', 'Stage 3 Refinement Result', 'Stage3 Fit')
        return ctx



def prepare_phase_dw_dicts(phase_structs: Dict[str, Structure]) -> Dict[str, Dict[str, float]]:
    phase_dw_dicts = {}
    for key, st in phase_structs.items():
        dw = {}
        for site in st.sites:
            b_val = float(site.properties.get('Biso', site.properties.get('Uiso', 0.01) * 8.0 * np.pi ** 2))
            for el in site.species.elements:
                sym = str(el)
                dw.setdefault(sym, []).append(b_val)
        phase_dw_dicts[key] = {k: float(np.mean(v)) for k, v in dw.items()}
    return phase_dw_dicts


def ensure_biso_on_structure(st: Structure) -> Structure:
    return sync_biso(st)


def detect_main_phase_by_corr(x: np.ndarray, y: np.ndarray, files: List[str], wavelength: float, broad_base: float, tch_init=(0.003, 0.001, 0.020, 0.020, 0.010)):
    """Detect the main phase by profile-observation correlation, return reordered files and diagnostics."""
    if len(files) <= 1:
        only = list(files)
        return only, (only[0] if only else None), ([1.0] if only else []), 0
    U0, V0, W0, X0, Y0 = tch_init
    corrs = []
    for f in files:
        st = ensure_biso_on_structure(Structure.from_file(f))
        prof = synth_profile_po(
            x, st, wl=wavelength, U=U0, V=V0, W=W0, X=X0, Y=Y0,
            broad_base=broad_base, enable_po=False
        )
        c = float(np.corrcoef(y, prof)[0, 1])
        if not np.isfinite(c):
            c = -1.0
        corrs.append(c)
    main_idx = int(np.argmax(corrs))
    reordered_indices = [main_idx] + [i for i in range(len(files)) if i != main_idx]
    reordered = [files[i] for i in reordered_indices]
    reordered_corrs = [corrs[i] for i in reordered_indices]
    return reordered, files[main_idx], reordered_corrs, main_idx


def initial_combo_search(x, y, main_cif, top, wavelength, bg_degree, broad_base, main_bias, single_phase=False, max_phases_in_mix=4, main_selection='fixed'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_event(f"🖥️ 设备：{device}")
    U0, V0, W0, X0, Y0 = 0.003, 0.001, 0.020, 0.020, 0.010
    best = {'files': [main_cif], 'yfit': None, 'rwp': 1e9, 'fr': None, 'sf': None, 'main_phase': os.path.basename(main_cif), 'corrs': []}
    pools = [[]] if single_phase else [[]] + [list(c) for k in range(1, max_phases_in_mix) for c in itertools.combinations([f for f, _, _ in top], k)]
    for combo in pools:
        raw_files = [main_cif] + list(combo)
        if main_selection == 'pearson':
            files, detected_main, corrs, main_idx_raw = detect_main_phase_by_corr(
                x, y, raw_files, wavelength=wavelength, broad_base=broad_base,
                tch_init=(U0, V0, W0, X0, Y0)
            )
        else:
            files = raw_files
            detected_main = main_cif
            main_idx_raw = 0
            corrs = []
            for f in files:
                st_corr = ensure_biso_on_structure(Structure.from_file(f))
                prof_corr = synth_profile_po(
                    x, st_corr, wl=wavelength, U=U0, V=V0, W=W0, X=X0, Y=Y0,
                    broad_base=broad_base, enable_po=False
                )
                c = float(np.corrcoef(y, prof_corr)[0, 1])
                corrs.append(c if np.isfinite(c) else -1.0)
        structs = [ensure_biso_on_structure(Structure.from_file(f)) for f in files]
        profiles = [synth_profile_po(x, st, wl=wavelength, U=U0, V=V0, W=W0, X=X0, Y=Y0, broad_base=broad_base, enable_po=False) for st in structs]
        yfit, fr, sf, Rwp0_tmp, _ = torch_refine(
            y, profiles, device=device, bg_degree=bg_degree, epochs=200,
            lbfgs_lr=0.3, lbfgs_max_iter=60,
            main_bias=main_bias if len(files) > 1 else 0.0,
            log_improvements=False,
            log_context='combo'
        )
        if Rwp0_tmp < best['rwp']:
            old_rwp = best['rwp']
            best = {
                'files': files, 'yfit': yfit, 'rwp': Rwp0_tmp, 'fr': fr, 'sf': sf,
                'main_phase': os.path.basename(detected_main) if detected_main else None,
                'corrs': corrs,
            }
    corr_msg = format_corr_summary(best['files'], best['corrs'])
    combo_msg = ' + '.join(os.path.basename(f) for f in best['files'])
    if main_selection == 'pearson':
        log_event(f"主相识别 | main={best['main_phase']} | pearson=({corr_msg})")
    else:
        log_event(f"主相固定 | main={best['main_phase']} | strategy=model_top1 | corr=({corr_msg})")
    log_event(f"初选最佳组合 | combo={combo_msg} | Rwp={best['rwp']:.2f}%")
    return best


def main(xy_file=None, main_cif='Li6PS5Cl.cif', imp_dir='impure_phase', wavelength=1.5406,
         tch_init=(0.003, 0.001, 0.020, 0.020, 0.010), broad_base=0.08, bg_degree=5,
         max_candidates=6, max_phases_in_mix=4, stage_loops=(100, 80, 70), single_phase=False,
         init_pos_step=0.005, min_pos_step=0.002, init_mix_step=0.05, min_mix_step=0.01,
         init_angle_step_deg=0.3, min_angle_step_deg=0.05,
         init_cell_step_frac=0.0025, min_cell_step_frac=0.0002,
         init_tch_step_frac=0.15, min_tch_step_frac=0.05,
         enable_po=True, init_po_step_frac=0.25, min_po_step_frac=0.05,
         po_r_bounds=(0.3, 3.0), po_axes_user=None, po_r_init_user=None,
         lambda_stoich=0.5, stoich_phase=None, stoich_target=None,
         num_workers: int = os.cpu_count(), main_bias: float = 1.0,
         use_rl_guidance: bool = True, main_selection: str = 'fixed'):
    if xy_file is None:
        xy_files = [f for f in os.listdir('.') if f.lower().endswith('.xy')]
        if not xy_files:
            raise FileNotFoundError('未发现 .xy 文件')
        xy_file = sorted(xy_files)[0]
    x, y = read_xy(xy_file)
    if not os.path.exists(main_cif):
        raise FileNotFoundError(f'主相 {main_cif} 不存在')
    st_main = ensure_biso_on_structure(Structure.from_file(main_cif))
    imp_files = []
    if (not single_phase) and imp_dir and isinstance(imp_dir, str) and os.path.isdir(imp_dir):
        main_abs = os.path.abspath(main_cif)
        imp_files = [
            os.path.join(imp_dir, f)
            for f in os.listdir(imp_dir)
            if f.lower().endswith('.cif') and os.path.abspath(os.path.join(imp_dir, f)) != main_abs
        ]
        imp_files.sort()
    U0, V0, W0, X0, Y0 = tch_init
    cands = []
    for f in imp_files:
        st = ensure_biso_on_structure(Structure.from_file(f))
        prof = synth_profile_po(x, st, wl=wavelength, U=U0, V=V0, W=W0, X=X0, Y=Y0, broad_base=broad_base, enable_po=False)
        c = np.corrcoef(y, prof)[0, 1]
        cands.append((f, prof, c))
    cands.sort(key=lambda z: z[2], reverse=True)
    top = cands[:max_candidates]
    best = initial_combo_search(x, y, main_cif, top, wavelength, bg_degree, broad_base, main_bias, single_phase=single_phase, max_phases_in_mix=max_phases_in_mix, main_selection=main_selection)
    detected_main_phase = best.get('main_phase', os.path.basename(main_cif))
    phase_structs = {f: ensure_biso_on_structure(Structure.from_file(f)) for f in best['files']}
    tch_dict = {'U': U0, 'V': V0, 'W': W0, 'X': X0, 'Y': Y0}
    po_axes = {k: (0, 0, 1) for k in phase_structs.keys()}
    if po_axes_user:
        phase_basename_map = {os.path.basename(k): k for k in po_axes.keys()}
        for k, v in po_axes_user.items():
            phase_key = k if k in po_axes else phase_basename_map.get(os.path.basename(k))
            if phase_key is not None:
                po_axes[phase_key] = tuple(v)
    po_r_init = {k: 1.0 for k in phase_structs.keys()}
    if po_r_init_user:
        phase_basename_map = {os.path.basename(k): k for k in po_r_init.keys()}
        for k, v in po_r_init_user.items():
            phase_key = k if k in po_r_init else phase_basename_map.get(os.path.basename(k))
            if phase_key is not None:
                po_r_init[phase_key] = float(v)
    if stoich_phase is None:
        stoich_phase = detected_main_phase
    else:
        stoich_phase = os.path.basename(stoich_phase)

    phase_file_map = {os.path.basename(k): k for k in phase_structs.keys()}
    stoich_phase_file = phase_file_map.get(stoich_phase, phase_file_map.get(detected_main_phase, main_cif))

    if stoich_target is None:
        comp = phase_structs.get(stoich_phase_file, st_main).composition.get_el_amt_dict()
        stoich_target = {k: float(v) for k, v in comp.items() if v > 1e-6}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = RefinementContext(
        x_grid=x,
        y_obs=y,
        wavelength=wavelength,
        phase_structs=phase_structs,
        phase_dw_dicts=prepare_phase_dw_dicts(phase_structs),
        tch_params=tch_dict,
        po_r=po_r_init,
        po_axes=po_axes,
        lambda_stoich=lambda_stoich,
        stoich_phase_key=stoich_phase,
        stoich_target=stoich_target,
        device=device,
        bg_degree=bg_degree,
        broad_base=broad_base,
        enable_po=enable_po,
        main_bias=main_bias,
    )
    log_event(f"主相偏置 | target={ctx.stoich_phase_key} | bias={ctx.main_bias:.3f} | 策略=初始组合+Stage1")
    runner = OuterRefineV44RL(use_rl_guidance=use_rl_guidance, num_workers=num_workers)
    ctx = runner.run(ctx, stage_loops=stage_loops,
                     init_cell_step_frac=init_cell_step_frac, min_cell_step_frac=min_cell_step_frac,
                     init_angle_step_deg=init_angle_step_deg, min_angle_step_deg=min_angle_step_deg,
                     init_tch_step_frac=init_tch_step_frac, min_tch_step_frac=min_tch_step_frac,
                     init_pos_step=init_pos_step, min_pos_step=min_pos_step,
                     init_mix_step=init_mix_step, min_mix_step=min_mix_step,
                     init_po_step_frac=init_po_step_frac, min_po_step_frac=min_po_step_frac)
    refined_structs = ctx.phase_structs
    tch_final = ctx.tch_params
    yfit_final = ctx.yfit
    fr_final = ctx.fr
    sf_final = ctx.sf
    rwp_final = ctx.best_rwp
    po_r_final = ctx.po_r
    po_axes_final = ctx.po_axes
    text_lines = ['Composition:']
    for f, w in zip(refined_structs.keys(), fr_final):
        text_lines.append(f'  {os.path.basename(f)}: {w*100:.2f}%')
    text_lines += ['', f'Rwp = {rwp_final:.2f}%']
    save_fit_plot_with_residual('yfsf_Refined.png', x, y, yfit_final, 'yfsf RL-enhanced Refinement Result', 'Final Fit', text_lines)
    log_event('🖼️ 拟合图已保存：yfsf_Refined.png')
    save_xy_with_residual('yfsf_Refined.xy', x, y, yfit_final)
    log_event('💾 已保存拟合曲线数据：yfsf_Refined.xy')
    os.makedirs('yfsf_refined_cifs', exist_ok=True)
    with open('yfsf_Refined.txt', 'w', encoding='utf-8') as fw:
        fw.write('=== yfsf_Refined ===\n')
        fw.write(f'XY file   : {xy_file}\n')
        fw.write(f'Final Rwp : {rwp_final:.3f}%\n')
        fw.write('TCH params (final): ' + ', '.join([f'{k}={v:.6g}' for k, v in tch_final.items()]) + '\n')
        fw.write('Residual definition: Intensity_Obs - Intensity_Fit\n')
        fw.write('\nPhases (fractions & per-phase scales):\n')
        for f, w, s in zip(refined_structs.keys(), fr_final, sf_final):
            fw.write(f'  {os.path.basename(f):<28s} frac={w*100:6.2f}% | scale={float(s):.4f}\n')
        fw.write('\nPreferred Orientation (March–Dollase):\n')
        fw.write(f'  bounds: r in [{po_r_bounds[0]:.2f}, {po_r_bounds[1]:.2f}]\n')
        for f in refined_structs.keys():
            axis = po_axes_final.get(f, (0, 0, 1))
            rfin = po_r_final.get(f, 1.0)
            fw.write(f'  {os.path.basename(f):<28s} axis=[{axis[0]},{axis[1]},{axis[2]}] | r={rfin:.4f}\n')
        fw.write(f'\nStoichiometry target (phase={stoich_phase}): {stoich_target}\n')
        fw.write('\nExported CIFs:\n')
    for fpath, struct in refined_structs.items():
        base = os.path.basename(fpath)
        name, _ = os.path.splitext(base)
        out_path = os.path.join('yfsf_refined_cifs', f'{name}_refined.cif')
        st_out = apply_dw_to_structure(struct, ctx.phase_dw_dicts[fpath])
        export_cif_with_biso(st_out, out_path)
        log_event(f'💾 已导出精修 CIF: {out_path}')
        with open('yfsf_Refined.txt', 'a', encoding='utf-8') as fw:
            fw.write(f'  {out_path}\n')
    log_event(f'\n📈 最终指标：Rwp = {rwp_final:.2f}%')
    log_event('📊 相分数（softmax）与每相独立 scale：')
    for f, w, s in zip(refined_structs.keys(), fr_final, sf_final):
        log_event(f'   {os.path.basename(f):<28s} frac={w*100:6.2f}% | scale={float(s):.3f}')
    if enable_po:
        log_event('📌 择优取向（March–Dollase）参数：')
        for f in refined_structs.keys():
            axis = po_axes_final.get(f, (0, 0, 1))
            rfin = po_r_final.get(f, 1.0)
            log_event(f'   {os.path.basename(f):<28s} axis=[{axis[0]},{axis[1]},{axis[2]}] | r={rfin:.4f}')
    return ctx


if __name__ == '__main__':
    def parse_stage_loops(value: str) -> Tuple[int, int, int]:
        parts = [p.strip() for p in value.split(',') if p.strip()]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError('stage loops must be three comma-separated integers, e.g. 100,80,70')
        try:
            loops = tuple(int(p) for p in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError('stage loops must be integers') from exc
        if any(v <= 0 for v in loops):
            raise argparse.ArgumentTypeError('stage loops must be positive')
        return loops

    parser = argparse.ArgumentParser()
    parser.add_argument('--xy', type=str, help='实验谱文件 (.xy)')
    parser.add_argument('--main', type=str, help='主相 CIF 文件')
    parser.add_argument('--imp', type=str, help='杂相目录')
    parser.add_argument('--num-workers', type=int, default=os.cpu_count(), help='并行进程数')
    parser.add_argument('--max-candidates', type=int, default=10, help='参与初始组合搜索的候选杂相数量；默认 10，对应每个样品 1+10 个 CIF 全读入筛选')
    parser.add_argument('--max-phases-in-mix', type=int, default=4, help='初始组合搜索的最大总相数，默认 4')
    parser.add_argument('--stage-loops', type=parse_stage_loops, default=(100, 80, 70), help='三阶段循环次数，格式如 100,80,70')
    parser.add_argument('--main-bias', type=float, default=1.0, help='主相偏置系数 (用于相组合筛选与后续 torch_refine 初始化)')
    parser.add_argument('--main-selection', choices=['fixed', 'pearson'], default='fixed', help='fixed: 固定传入 --main 为主相；pearson: 使用旧版 Pearson 自动重排主相')
    parser.add_argument('--stoich-phase', type=str, default=None, help='指定化学计量约束参考相，默认自动使用相关性最高的主相')
    parser.add_argument('--stoich', type=str, default=None, help='目标化学计量比，例如 "Li:6,S:5,P:1,Cl:1"')
    parser.add_argument('--lambda-stoich', type=float, default=0.5, help='化学计量约束强度 λ_stoich (默认 0.5)')
    parser.add_argument('--wl', type=float, default=1.5406, help='X射线波长')
    parser.add_argument('--po-axes', type=str, default=None, help='PO轴向设定，格式如 主相文件名.cif:1,3,0')
    parser.add_argument('--no-rl', action='store_true', help='关闭 RL 排序，固定候选顺序')
    parser.add_argument('--single-phase', action='store_true', help='只精修传入的 --main CIF，不使用 --imp 杂相')
    parser.add_argument('--disable-po', action='store_true', help='关闭 March-Dollase 择优取向精修')
    args = parser.parse_args()
    po_axes_dict = None
    if args.po_axes:
        try:
            cif_name, hkl_str = args.po_axes.split(':')
            h, k, l = map(int, hkl_str.split(','))
            po_axes_dict = {cif_name: (h, k, l)}
        except Exception as e:
            log_event(f'⚠️ 解析 --po-axes 参数失败 ({args.po_axes}): {e}')
    main(
        xy_file=args.xy,
        main_cif=args.main,
        imp_dir=args.imp,
        main_bias=args.main_bias,
        num_workers=args.num_workers,
        max_candidates=args.max_candidates,
        max_phases_in_mix=args.max_phases_in_mix,
        stage_loops=args.stage_loops,
        single_phase=args.single_phase,
        stoich_phase=args.stoich_phase,
        stoich_target=None if args.stoich is None else {k: float(v) for k, v in (pair.split(':') for pair in args.stoich.split(','))},
        lambda_stoich=args.lambda_stoich,
        wavelength=args.wl,
        po_axes_user=po_axes_dict,
        enable_po=(not args.disable_po),
        use_rl_guidance=(not args.no_rl),
        main_selection=args.main_selection,
    )
