# Non noise model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# FFT 多视角：低通滤波
# -----------------------------
def fft_lowpass(x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """
    简单 FFT 低通滤波（保留前 keep_ratio 的低频分量）
    x: [B, L]
    return: [B, L]
    """
    x = x.float()
    X = torch.fft.rfft(x, dim=-1)          # [B, L//2+1] complex
    Lf = X.shape[-1]
    k = max(1, int(Lf * keep_ratio))
    Xf = torch.zeros_like(X)
    Xf[..., :k] = X[..., :k]
    xr = torch.fft.irfft(Xf, n=x.shape[-1], dim=-1)
    return xr


# -----------------------------
# 固定 1D Sin-Cos 位置编码
# -----------------------------
def build_sincos_1d_pos_embed(length: int, dim: int, device=None) -> torch.Tensor:
    """
    return: [1, length, dim] (float32)
    """
    if dim % 2 != 0:
        raise ValueError(f"pos_embed dim must be even, got {dim}")
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)  # [L,1]
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim)
    )  # [dim/2]
    pe = torch.zeros(length, dim, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1,L,D]


# -----------------------------
# 多尺度卷积字典
# -----------------------------
class MultiScaleConvBank(nn.Module):
    """
    输入:  [B, C_in, L]
    输出:  [B, C_out, L]
    """
    def __init__(self, c_in: int, c_each: int = 32, kernel_sizes=(17, 33, 65, 129, 257), c_out: int = 64):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(c_in, c_each, k, padding=k // 2),
                nn.BatchNorm1d(c_each),
                nn.ReLU(inplace=True),
            )
            for k in kernel_sizes
        ])
        c_cat = c_each * len(kernel_sizes)
        self.fuse = nn.Sequential(
            nn.Conv1d(c_cat, c_out, kernel_size=1),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        outs = [b(x) for b in self.branches]          # list of [B,c_each,L]
        x = torch.cat(outs, dim=1)                    # [B,c_cat,L]
        x = self.fuse(x)                              # [B,c_out,L]
        return x


class ResidualBlock(nn.Module):
    """1D ResNet Block 用于提取局部峰形特征"""
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class PerceiverXRD(nn.Module):
    """
    - 曲线流：FFT 多视角(4通道) -> 多尺度卷积字典 -> ResBlocks -> 投影到 latent_dim
    - peak token：MLP -> latent_dim
    - 元素 one-hot：1) Elem token（拼进 context）
                 2) FiLM 条件化（让 one-hot 真正影响曲线特征）
    - 融合：Perceiver cross-attn -> 4层 Transformer -> 强分类头（BN+大 Dropout，抄 XQueryer）
    """
    def __init__(self, input_dim=3500, num_latents=128, latent_dim=256, num_classes=230,
                 num_heads=8, ff_dim=1024, dropout=0.2, max_peaks=48, elem_dim=118):
        super().__init__()

        self.input_dim = int(input_dim)
        self.num_latents = int(num_latents)
        self.latent_dim = int(latent_dim)
        self.num_classes = int(num_classes)
        self.max_peaks = int(max_peaks)
        self.elem_dim = int(elem_dim)

        # ===== Stream 1: 曲线流（FFT 4通道 + 多尺度卷积字典）=====
        self.msbank = MultiScaleConvBank(c_in=4, c_each=32, kernel_sizes=(17, 33, 65, 129, 257), c_out=64)
        self.cnn_tail = nn.Sequential(
            ResidualBlock(64, 64, kernel_size=7),
            ResidualBlock(64, 64, kernel_size=7),
        )
        self.curve_proj = nn.Linear(64, latent_dim)

        # 固定 sin-cos 位置编码（register_buffer，不训练）
        pe = build_sincos_1d_pos_embed(self.input_dim, self.latent_dim)
        self.register_buffer("curve_pos_embed", pe, persistent=False)  # [1,L,D]

        # ===== Stream 2: 峰值流 (Peak Tokens) =====
        self.peak_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim)
        )
        # ===== Peak-guided Attention：peak tokens 做 Query 去读曲线（Key/Value）=====
        self.peak_query_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # 把原始 peak token + peak-attended curve token 融合（更稳）
        self.peak_fuse = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

        # ===== Stream 3: 元素 one-hot =====
        # 3.1 Elem token（拼进 context）
        self.elem_encoder = nn.Sequential(
            nn.Linear(elem_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim)
        )
        # 3.2 FiLM：elem -> (gamma,beta) 条件化曲线特征（让 one-hot 真正发挥作用）
        self.elem_film = nn.Sequential(
            nn.Linear(elem_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim * 2),
        )

        # ===== Fusion Core (Perceiver) =====
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # ===== 更强分类头（抄 XQueryer：BN + 大 Dropout）=====
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

    def forward(self, x_curve, x_peaks, elem=None, return_embedding: bool = False):
        """
        x_curve: [B, L]          (L=3500)
        x_peaks: [B, P, 5]       (P=max_peaks)
        elem:    [B, 118]
        """
        b = x_curve.shape[0]
        device = x_curve.device

        # ===== 0) elem 兜底 =====
        if elem is None:
            elem = torch.zeros((b, self.elem_dim), device=device, dtype=torch.float32)
        else:
            elem = elem.to(device=device, dtype=torch.float32)

        # ===== 1) 曲线 FFT 4 视角 =====
        # 说明：x_curve 通常已经归一化到 0~1（dataset 做的），这里不再 /100
        x0 = x_curve.float()
        x1 = fft_lowpass(x0, 0.70)
        x2 = fft_lowpass(x0, 0.40)
        x3 = fft_lowpass(x0, 0.10)
        x4c = torch.stack([x0, x1, x2, x3], dim=1)              # [B,4,L]

        # 多尺度卷积字典 + ResBlocks
        feat = self.msbank(x4c)                                 # [B,64,L]
        feat = self.cnn_tail(feat)                              # [B,64,L]
        feat = feat.permute(0, 2, 1).contiguous()               # [B,L,64]

        # 投影到 latent_dim + 固定位置编码
        feat_curve = self.curve_proj(feat)                      # [B,L,D]
        pos = self.curve_pos_embed.to(device=device, dtype=feat_curve.dtype)  # [1,L,D]
        feat_curve = feat_curve + pos

        # ===== 2) FiLM：元素条件化曲线特征（关键：让 one-hot 真正影响谱线理解）=====
        film = self.elem_film(elem).to(dtype=feat_curve.dtype)  # [B,2D]
        gamma, beta = film.chunk(2, dim=-1)                     # [B,D], [B,D]
        gamma = gamma.unsqueeze(1)                               # [B,1,D]
        beta  = beta.unsqueeze(1)                                # [B,1,D]
        feat_curve = feat_curve * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)

        # ===== 3) Peak tokens =====
        x_peaks = x_peaks.to(device=device, dtype=feat_curve.dtype)
        feat_peaks = self.peak_encoder(x_peaks)                 # [B,P,D]

        # ===== 3.1) Peak-guided Attention：peak tokens 做 Query 去读曲线特征 =====
        # Query: feat_peaks [B,P,D]
        # Key/Value: feat_curve [B,L,D]
        peak_attended, _ = self.peak_query_attn(feat_peaks, feat_curve, feat_curve)  # [B,P,D]

        # ===== 3.2) 融合：原始峰 token + 峰感知曲线 token =====
        feat_peaks = self.peak_fuse(torch.cat([feat_peaks, peak_attended], dim=-1))  # [B,P,D]


        # ===== 4) Elem token（拼进 context）=====
        elem_tok = self.elem_encoder(elem).to(dtype=feat_curve.dtype).unsqueeze(1)  # [B,1,D]

        # ===== 5) Context 拼接： [ElemToken, Curve, Peaks] =====
        context = torch.cat([elem_tok, feat_curve, feat_peaks], dim=1)  # [B, 1+L+P, D]

        # ===== 6) Cross Attention：Latents 读 Context =====
        latents = self.latents.unsqueeze(0).expand(b, -1, -1).contiguous()  # [B, num_latents, D]
        latents, _ = self.cross_attn(latents, context, context)

        # ===== 7) Transformer 推理 =====
        latents = self.transformer(latents)
        global_feat = latents.mean(dim=1)  # [B,D]

        if return_embedding:
            return global_feat
        return self.classifier(global_feat)

    def to_logits(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding)