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
from torch import nn

__all__ = ("MSFusion", "MSSplit")


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
