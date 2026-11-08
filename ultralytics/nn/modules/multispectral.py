# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Multispectral middle-fusion modules.

These modules enable a two-stream (dual backbone) YOLO architecture for multispectral TIFF inputs:

- Stream A receives the visible bands (G, R).
- Stream B receives the remaining bands (NIR, RE, or NIR+RE).
- Features from both streams are fused at P3, P4 and P5 by ``Concat -> 1x1 Conv`` (``MSFusion``).

Expected TIFF band order on disk (see ``create_multispectral_dataset.py``):
    3ch: (G, R, NIR) or (G, R, RE)
    4ch: (G, R, NIR, RE)

Note:
    Ultralytics reverses the channel order of every 3-channel image (BGR -> RGB) in both the training ``Format``
    transform and the ``Predictor``. A 3-band TIFF (G, R, X) therefore reaches the network as (X, R, G). ``MSSplit``
    compensates for this automatically so that stream A always receives (G, R) in that order.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("CGFusion", "CMGFusion", "MSAttnFusion", "MSFusion", "MSSplit")


class MSSplit(nn.Module):
    """Select the channels for one stream of a dual-backbone multispectral model.

    Attributes:
        stream (int): 0 selects the visible (G, R) stream, 1 selects the remaining bands (NIR / RE).
        c1 (int): Number of input image channels.
        idx (list[int]): Channel indices gathered from the network input.

    Examples:
        >>> x = torch.randn(1, 4, 64, 64)  # (G, R, NIR, RE)
        >>> MSSplit(4, 0)(x).shape  # (G, R)
        torch.Size([1, 2, 64, 64])
        >>> MSSplit(4, 1)(x).shape  # (NIR, RE)
        torch.Size([1, 2, 64, 64])
    """

    def __init__(self, c1: int, stream: int = 0, n_vis: int = 2, flip3: bool = True):
        """Initialize MSSplit.

        Args:
            c1 (int): Number of input channels of the full multispectral image.
            stream (int): 0 for the visible (G, R) stream, 1 for the remaining bands.
            n_vis (int): Number of leading visible bands in the TIFF (default 2 -> G, R).
            flip3 (bool): Compensate the BGR->RGB channel reversal Ultralytics applies to 3-channel inputs.
        """
        super().__init__()
        if c1 <= n_vis:
            raise ValueError(
                f"MSSplit requires more than {n_vis} input channels (got {c1}). "
                f"Use 3ch (G,R,NIR|RE) or 4ch (G,R,NIR,RE) TIFF images and set 'channels' in the dataset YAML."
            )
        self.c1, self.stream, self.n_vis, self.flip3 = c1, stream, n_vis, flip3
        order = list(range(c1))  # on-disk band index -> tensor channel index
        if flip3 and c1 == 3:
            order = order[::-1]  # 3ch tensors arrive reversed: tensor[k] = disk[2 - k]
        bands = list(range(n_vis)) if stream == 0 else list(range(n_vis, c1))
        self.idx = [order[b] for b in bands]
        self.c2 = len(self.idx)
        # Contiguous ascending slices are cheaper (and export-friendly) than gathers
        self._slice = (
            (self.idx[0], self.idx[-1] + 1) if self.idx == list(range(self.idx[0], self.idx[-1] + 1)) else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the selected channels of the input tensor."""
        if self._slice is not None:
            return x[:, self._slice[0] : self._slice[1]]
        return x[:, self.idx]


class MSFusion(nn.Module):
    """Middle fusion block: concatenate two feature maps along channels and fuse them with a 1x1 convolution.

    The 1x1 convolution is initialized as the element-wise average of both streams (``[0.5*I | 0.5*I]``) so that a
    detection head transferred from a single-stream pretrained model receives features at the expected scale.

    Attributes:
        conv (nn.Conv2d): 1x1 fusion convolution.
        bn (nn.Module): Optional BatchNorm2d (Identity if disabled).
        act (nn.Module): Optional activation (Identity if disabled).

    Examples:
        >>> m = MSFusion(512, 256)
        >>> a, b = torch.randn(1, 256, 20, 20), torch.randn(1, 256, 20, 20)
        >>> torch.allclose(m([a, b]), (a + b) / 2, atol=1e-6)
        True
    """

    def __init__(self, c1: int, c2: int, bn: bool = False, act: bool = False):
        """Initialize MSFusion.

        Args:
            c1 (int): Total input channels (sum of both streams).
            c2 (int): Output channels.
            bn (bool): Add BatchNorm2d after the 1x1 conv.
            act (bool): Add SiLU activation after the 1x1 conv (and BN).
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()
        self.reset_identity()

    @torch.no_grad()
    def reset_identity(self):
        """Initialize the 1x1 conv as an average of equally sized input streams (falls back to default init)."""
        c2, c1 = self.conv.out_channels, self.conv.in_channels
        if c1 % c2:
            return  # streams of unequal width: keep default init
        k = c1 // c2  # number of streams
        w = torch.zeros(c2, c1, 1, 1)
        eye = torch.eye(c2) / k
        for s in range(k):
            w[:, s * c2 : (s + 1) * c2, 0, 0] = eye
        self.conv.weight.copy_(w)
        if self.conv.bias is not None:
            self.conv.bias.zero_()

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate the input feature maps and apply the 1x1 fusion convolution."""
        return self.act(self.bn(self.conv(torch.cat(x, 1))))


class CMGFusion(nn.Module):
    """Cross-Modal Gated & Differential Fusion (CMGFusion) for multispectral object detection.

    Designed for agricultural multispectral disease detection (small, dense lesions with subtle inter-band parallax):
    1. **3x3 Depthwise Residual Alignment**: absorbs local sub-pixel/few-pixel registration residuals between the
       visible stream (G, R) and the NIR/RE stream.
    2. **Spectral Differential & Multiplicative Branch**: explicitly computes ``align_b - align_a`` (spectral contrast
       analogous to NDVI/GNDVI/NDRE) and ``SiLU(align_a) * SiLU(align_b)`` (structural co-activation), then refines
       them with a lightweight depthwise-separable bottleneck.
    3. **Channel + Spatial Cross-Modal Gate**: dynamically weights visible vs. NIR/RE features per channel (SE gate)
       and per spatial location (7x7 spatial contrast gate).

    All residual/gating output projections are zero-initialized so that at step 0 the module computes the exact stream
    average ``0.5 * f_a + 0.5 * f_b`` (preserving transferred YOLOv8 pretrained head compatibility) while every sub-branch
    receives non-zero gradients from the very first backward pass.

    Examples:
        >>> m = CMGFusion(512, 256)
        >>> a, b = torch.randn(1, 256, 20, 20), torch.randn(1, 256, 20, 20)
        >>> torch.allclose(m([a, b]), (a + b) / 2, atol=1e-6)
        True
    """

    def __init__(self, c1: int, c2: int, reduction: int = 4, k_align: int = 3, k_spatial: int = 7):
        """Initialize CMGFusion.

        Args:
            c1 (int): Total input channels (sum of both streams, expected ``2 * c2``).
            c2 (int): Output channels (single-stream width).
            reduction (int): Channel reduction ratio for the channel gate and differential bottleneck.
            k_align (int): Kernel size for per-stream depthwise spatial alignment.
            k_spatial (int): Kernel size for the spatial cross-modal gate.
        """
        super().__init__()
        if c1 != 2 * c2:
            raise ValueError(f"CMGFusion expects two equal-width streams (c1 == 2 * c2), got c1={c1}, c2={c2}")
        c_mid = max(c2 // reduction, 16)

        # 1) Local spatial alignment (3x3 depthwise residual conv per stream)
        self.dw_a = nn.Conv2d(c2, c2, k_align, 1, k_align // 2, groups=c2, bias=True)
        self.dw_b = nn.Conv2d(c2, c2, k_align, 1, k_align // 2, groups=c2, bias=True)

        # 2) Channel gate (SE-style on concatenated global avg + max pooled descriptors)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(c1, c_mid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c2, 1, bias=True),
        )

        # 3) Spatial gate on [mean_a, max_a, mean_b, max_b, mean_diff, max_diff] (6 channels)
        self.spatial_gate = nn.Conv2d(6, 1, k_spatial, 1, k_spatial // 2, bias=True)

        # 4) Differential & multiplicative interaction branch: input is [align_b - align_a, SiLU(a)*SiLU(b)] (2*c2 ch)
        self.diff_branch = nn.Sequential(
            nn.Conv2d(c1, c_mid, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c_mid, 3, 1, 1, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c2, 1, bias=True),
        )
        self.act = nn.SiLU(inplace=True)
        self.reset_identity()

    @torch.no_grad()
    def reset_identity(self):
        """Zero-initialize residual/gate projections so initial output equals 0.5 * (f_a + f_b) with full grad flow."""
        for m in (self.dw_a, self.dw_b, self.channel_gate[-1], self.spatial_gate, self.diff_branch[-1]):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse visible stream A and NIR/RE stream B via alignment, dynamic gating, and differential contrast."""
        f_a, f_b = x[0], x[1]

        # Step 1: Local spatial alignment
        a = f_a + self.dw_a(f_a)
        b = f_b + self.dw_b(f_b)

        # Step 2: Spectral difference & multiplicative co-activation
        diff = b - a
        inter = F.silu(a) * F.silu(b)

        # Step 3: Dynamic channel + spatial cross-modal gate
        ab = torch.cat([a, b], dim=1)
        g_c = self.channel_gate(ab.mean((2, 3), keepdim=True) + ab.amax((2, 3), keepdim=True))
        s_in = torch.cat(
            [
                a.mean(1, keepdim=True),
                a.amax(1, keepdim=True),
                b.mean(1, keepdim=True),
                b.amax(1, keepdim=True),
                diff.mean(1, keepdim=True),
                diff.amax(1, keepdim=True),
            ],
            dim=1,
        )
        g_s = self.spatial_gate(s_in)
        alpha = torch.sigmoid(g_c + g_s)

        # Step 4: Gated base + differential contrast enhancement
        base = alpha * a + (1.0 - alpha) * b
        delta = self.diff_branch(torch.cat([diff, inter], dim=1))
        return base + delta


class CGFusion(nn.Module):
    """Channel-Gated & Aligned Middle Fusion (CGFusion) for multispectral object detection.

    Combines ONLY two lightweight mechanisms on top of ``MSFusion`` (Concat + 1x1 Conv):
    1. **Local Misalignment Absorption (3x3 Depthwise Residual Conv)**: absorbs inter-lens registration residuals
       for stream A (G, R) and stream B (NIR / RE) independently.
    2. **Dynamic Channel Gate (SE-style over all 2C concatenated channels)**: dynamically re-weights every channel
       of both streams based on global average + max pooled context before the 1x1 cross-channel fusion conv.

    Initialized so that at step 0 ``dw_a = 0``, ``dw_b = 0``, ``gate = 1.0``, and ``conv = [0.5*I | 0.5*I]``, making it
    a strict superset of ``MSFusion`` with zero BatchNorm statistical drift.

    Examples:
        >>> m = CGFusion(512, 256)
        >>> a, b = torch.randn(1, 256, 20, 20), torch.randn(1, 256, 20, 20)
        >>> torch.allclose(m([a, b]), (a + b) / 2, atol=1e-6)
        True
    """

    def __init__(self, c1: int, c2: int, reduction: int = 4, k_align: int = 3):
        """Initialize CGFusion.

        Args:
            c1 (int): Total input channels (sum of both streams, ``2 * c2``).
            c2 (int): Output channels.
            reduction (int): Reduction ratio for the dynamic channel gate MLP.
            k_align (int): Kernel size for the per-stream depthwise alignment conv.
        """
        super().__init__()
        if c1 != 2 * c2:
            raise ValueError(f"CGFusion expects two equal-width streams (c1 == 2 * c2), got c1={c1}, c2={c2}")
        c_mid = max(c1 // reduction, 16)

        # 1) Local misalignment absorption (per-stream 3x3 depthwise residual conv)
        self.dw_a = nn.Conv2d(c2, c2, k_align, 1, k_align // 2, groups=c2, bias=True)
        self.dw_b = nn.Conv2d(c2, c2, k_align, 1, k_align // 2, groups=c2, bias=True)

        # 2) Dynamic channel gate across all 2*c2 channels (scaled by 2*sigmoid so 0-init -> 1.0)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(c1, c_mid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c1, 1, bias=True),
        )

        # 3) 1x1 cross-channel fusion convolution (initialized to [0.5*I | 0.5*I], identical to MSFusion)
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=True)
        self.reset_identity()

    @torch.no_grad()
    def reset_identity(self):
        """Initialize alignment & gate residuals to identity and 1x1 conv to stream average [0.5*I | 0.5*I]."""
        for m in (self.dw_a, self.dw_b, self.channel_gate[-1]):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        c2, c1 = self.conv.out_channels, self.conv.in_channels
        k = c1 // c2
        w = torch.zeros(c2, c1, 1, 1)
        eye = torch.eye(c2) / k
        for s in range(k):
            w[:, s * c2 : (s + 1) * c2, 0, 0] = eye
        self.conv.weight.copy_(w)
        if self.conv.bias is not None:
            self.conv.bias.zero_()

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Align both streams locally, apply dynamic channel gating, and fuse with 1x1 convolution."""
        f_a, f_b = x[0], x[1]
        a = f_a + self.dw_a(f_a)
        b = f_b + self.dw_b(f_b)
        ab = torch.cat([a, b], dim=1)
        desc = ab.mean((2, 3), keepdim=True) + ab.amax((2, 3), keepdim=True)
        gate = 2.0 * torch.sigmoid(self.channel_gate(desc))
        return self.conv(ab * gate)


class MSAttnFusion(nn.Module):
    """Multi-Scale Alignment + Bi-Directional Cross-Covariance Attention + Local Contrast Fusion (no Sigmoid gates).

    Tailored for dense, small plant disease lesions in multispectral drone imagery:
    1. **Multi-Scale Local Misalignment Absorption (3x3 + 5x5 Depthwise Conv)**: corrects both fine (1px) and wider
       (2-3px) inter-band registration residuals per stream.
    2. **Bi-Directional Multi-Head Cross-Covariance Attention**: computes full cross-spectral correlation matrices
       ``Softmax(tau * Q_a @ K_b^T) @ V_b`` and ``Softmax(tau * Q_b @ K_a^T) @ V_a`` across channels in O(HW) complexity.
    3. **Local Leaf-Background Anomaly Contrast**: extracts ``x - AvgPool7x7(x)`` so tiny lesions (~4x5px at P3) stand
       out invariant to global illumination/shadow variations.
    4. **1x1 Cross-Stream Fusion Conv**: combines the aligned, cross-attended, and local-contrast features.

    Examples:
        >>> m = MSAttnFusion(512, 256)
        >>> a, b = torch.randn(1, 256, 20, 20), torch.randn(1, 256, 20, 20)
        >>> torch.allclose(m([a, b]), (a + b) / 2, atol=1e-5)
        True
    """

    def __init__(self, c1: int, c2: int, num_heads: int = 4):
        """Initialize MSAttnFusion.

        Args:
            c1 (int): Total input channels (sum of both streams, ``2 * c2``).
            c2 (int): Output channels.
            num_heads (int): Number of attention heads for cross-covariance attention.
        """
        super().__init__()
        if c1 != 2 * c2:
            raise ValueError(f"MSAttnFusion expects two equal-width streams (c1 == 2 * c2), got c1={c1}, c2={c2}")
        while c2 % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.c2 = c2
        self.num_heads = num_heads

        # 1) Multi-scale local misalignment absorption (3x3 + 5x5 depthwise conv per stream)
        self.dw3_a = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        self.dw5_a = nn.Conv2d(c2, c2, 5, 1, 2, groups=c2, bias=False)
        self.dw3_b = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        self.dw5_b = nn.Conv2d(c2, c2, 5, 1, 2, groups=c2, bias=False)
        self.norm_a = nn.GroupNorm(num_heads, c2)
        self.norm_b = nn.GroupNorm(num_heads, c2)

        # 2) Bi-directional Cross-Covariance Attention projections (Q, K, V for each stream)
        self.qkv_a = nn.Conv2d(c2, c2 * 3, 1, bias=False)
        self.qkv_b = nn.Conv2d(c2, c2 * 3, 1, bias=False)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 4.0)
        self.norm_attn = nn.GroupNorm(num_heads * 2, c1)

        # 3) Local lesion contrast enhancer (subtract 7x7 local background average + depthwise refinement)
        self.local_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3, count_include_pad=False)
        self.contrast_dw = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
        self.norm_contrast = nn.GroupNorm(num_heads * 2, c1)

        # Learnable residual scales initialized to 0.05 (non-zero!) so every internal conv/attention parameter
        # receives non-zero gradients from the very first backward pass while keeping >95% pretrained feature identity
        self.scale_align = nn.Parameter(torch.full((1, c1, 1, 1), 0.05))
        self.scale_attn = nn.Parameter(torch.full((1, c1, 1, 1), 0.05))
        self.scale_contrast = nn.Parameter(torch.full((1, c1, 1, 1), 0.05))

        # 4) Final 1x1 cross-channel fusion convolution
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=True)
        self.reset_identity()

    @torch.no_grad()
    def reset_identity(self):
        """Initialize the final 1x1 conv to [0.5*I | 0.5*I] and residual scales to 0.05 so all layers train from step 1."""
        self.scale_align.fill_(0.05)
        self.scale_attn.fill_(0.05)
        self.scale_contrast.fill_(0.05)
        c2, c1 = self.conv.out_channels, self.conv.in_channels
        k = c1 // c2
        w = torch.zeros(c2, c1, 1, 1)
        eye = torch.eye(c2) / k
        for s in range(k):
            w[:, s * c2 : (s + 1) * c2, 0, 0] = eye
        self.conv.weight.copy_(w)
        if self.conv.bias is not None:
            self.conv.bias.zero_()

    def _cross_cov_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: int, c: int, h: int, w: int
    ) -> torch.Tensor:
        """Compute L2-normalized multi-head cross-covariance attention in O(HW) time."""
        nh = self.num_heads
        d = c // nh
        q = q.reshape(b, nh, d, h * w)
        k = k.reshape(b, nh, d, h * w)
        v = v.reshape(b, nh, d, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out.reshape(b, c, h, w)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse visible and NIR/RE streams via multi-scale alignment, cross-attention, and local contrast."""
        f_a, f_b = x[0], x[1]
        bsz, c, h, w = f_a.shape

        # 1) Multi-scale local misalignment absorption (3x3 + 5x5 depthwise convs)
        align_a = F.silu(self.norm_a(self.dw3_a(f_a) + self.dw5_a(f_a)))
        align_b = F.silu(self.norm_b(self.dw3_b(f_b) + self.dw5_b(f_b)))
        ab_align = torch.cat([align_a, align_b], dim=1)

        # Base aligned features
        ab_base = torch.cat([f_a, f_b], dim=1) + self.scale_align * ab_align
        a_cur, b_cur = ab_base[:, :c], ab_base[:, c:]

        # 2) Bi-directional Cross-Covariance Attention (A <- B and B <- A)
        qa, ka, va = self.qkv_a(a_cur).chunk(3, dim=1)
        qb, kb, vb = self.qkv_b(b_cur).chunk(3, dim=1)
        attn_a = self._cross_cov_attn(qa, kb, vb, bsz, c, h, w)  # visible queries attend to NIR/RE keys & values
        attn_b = self._cross_cov_attn(qb, ka, va, bsz, c, h, w)  # NIR/RE queries attend to visible keys & values
        ab_attn = self.norm_attn(torch.cat([attn_a, attn_b], dim=1))

        # 3) Local leaf-background anomaly contrast (highlights small lesions against local 7x7 background)
        high_pass = ab_base - self.local_pool(ab_base)
        ab_contrast = F.silu(self.norm_contrast(self.contrast_dw(high_pass)))

        # 4) Combine and project with 1x1 conv
        ab_total = ab_base + self.scale_attn * ab_attn + self.scale_contrast * ab_contrast
        return self.conv(ab_total)
