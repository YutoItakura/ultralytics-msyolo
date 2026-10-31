# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Utilities for the YOLOv8 multispectral middle-fusion model (``yolov8-msmid.yaml``).

Converts official single-stream YOLOv8 checkpoints (``yolov8{n,s,m,l,x}.pt``) into dual-stream ``yolov8{scale}-msmid.pt``
checkpoints:

- Both backbones (stream A: G,R / stream B: NIR,RE) are initialized from the YOLOv8 backbone (layers 0-9).
- The stem conv input channels are re-mapped from RGB so that, per output filter, the sum over input channels is
  preserved (a grey input therefore produces exactly the same stem activation as the original model):
    stream A (G, R):  w_G = W_G + W_B / 2,  w_R = W_R + W_B / 2
    stream B (n ch):  w_k = (W_R + W_G + W_B) / n
- The MSFusion 1x1 convs are initialized as the average of both streams.
- The neck/head (layers 10-22) is copied with an index offset of +16.

Examples:
    >>> from ultralytics.utils.multispectral import convert_yolov8_to_msmid
    >>> convert_yolov8_to_msmid("yolov8m.pt", "yolov8m-msmid.pt", ch=4)
    >>> from ultralytics import YOLO
    >>> model = YOLO("yolov8m-msmid.pt")
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

from ultralytics.utils import LOGGER

MSMID_REPO = "YutoItakura/ultralytics-msyolo"  # GitHub repo hosting pre-converted weights in weights/
MSMID_BRANCH = "main"
MSMID_NAME_RE = re.compile(r"yolov8([nslmx])-msmid\.pt")

N_BACKBONE = 10  # YOLOv8 backbone layers 0-9
OFFSET_A, OFFSET_B, OFFSET_HEAD = 2, 13, 16  # layer index offsets in yolov8-msmid.yaml
RGB = {"R": 0, "G": 1, "B": 2}  # channel order of official YOLOv8 stem weights (trained on RGB)


def _map_key(k: str) -> list[str]:
    """Map a yolov8.yaml state_dict key to its destination key(s) in yolov8-msmid.yaml."""
    parts = k.split(".")
    if parts[0] != "model" or not parts[1].isdigit():
        return []
    i = int(parts[1])
    rest = ".".join(parts[2:])
    if i < N_BACKBONE:
        return [f"model.{i + OFFSET_A}.{rest}", f"model.{i + OFFSET_B}.{rest}"]
    return [f"model.{i + OFFSET_HEAD}.{rest}"]


def _stem_weights(w: torch.Tensor, n_a: int, n_b: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build stream A (G, R) and stream B (n_b bands) stem conv weights from an RGB stem conv weight."""
    wr, wg, wb = (w[:, RGB[c] : RGB[c] + 1] for c in "RGB")
    assert n_a == 2, "stream A is expected to receive exactly 2 bands (G, R)"
    w_a = torch.cat([wg + wb / 2, wr + wb / 2], 1)  # (G, R) order as stored in the TIFF
    w_b = w.sum(1, keepdim=True).repeat(1, n_b, 1, 1) / n_b
    return w_a, w_b


def convert_yolov8_to_msmid(
    src: str | Path = "yolov8m.pt",
    dst: str | Path | None = None,
    ch: int = 4,
    cfg: str | None = None,
    verbose: bool = True,
) -> Path:
    """Convert an official YOLOv8 detection checkpoint into a dual-stream middle-fusion checkpoint.

    Args:
        src (str | Path): Source YOLOv8 checkpoint (downloaded automatically if it's an official name).
        dst (str | Path, optional): Output path. Defaults to ``yolov8{scale}-msmid.pt`` next to the source.
        ch (int): Number of input channels of the multispectral TIFFs (3 or 4).
        cfg (str, optional): Model YAML. Defaults to ``yolov8{scale}-msmid.yaml``.
        verbose (bool): Print transfer summary.

    Returns:
        (Path): Path to the saved checkpoint.
    """
    from ultralytics import __version__
    from ultralytics.nn.modules import MSFusion, MSSplit
    from ultralytics.nn.tasks import DetectionModel, load_checkpoint

    src_model, ckpt_args = load_checkpoint(src, fuse=False)
    scale = src_model.yaml.get("scale") or re.search(r"yolov8([nslmx])", Path(str(src)).stem).group(1)
    cfg = cfg or f"yolov8{scale}-msmid.yaml"
    dst = Path(dst) if dst else Path(str(src)).with_name(f"yolov8{scale}-msmid.pt")

    model = DetectionModel(cfg, ch=ch, nc=src_model.yaml["nc"], verbose=False)
    model.names = deepcopy(src_model.names)
    csd = src_model.float().state_dict()
    sd = model.state_dict()

    # Stream split channel counts
    splits = [m for m in model.model if isinstance(m, MSSplit)]
    n_a, n_b = splits[0].c2, splits[1].c2

    loaded, skipped = set(), []
    for k, v in csd.items():
        for k2 in _map_key(k):
            if k2 not in sd:
                skipped.append(k2)
                continue
            if k == "model.0.conv.weight":  # stems: remap RGB -> (G, R) / rest
                w_a, w_b = _stem_weights(v, n_a, n_b)
                v2 = w_a if k2.startswith(f"model.{OFFSET_A}.") else w_b
            else:
                v2 = v
            if sd[k2].shape != v2.shape:
                skipped.append(f"{k2} {tuple(v2.shape)} -> {tuple(sd[k2].shape)}")
                continue
            sd[k2].copy_(v2)
            loaded.add(k2)

    fusion_keys = {
        f"model.{m.i}.{n}" for m in model.model if isinstance(m, MSFusion) for n, _ in m.named_parameters()
    } | {f"model.{m.i}.{n}" for m in model.model if isinstance(m, MSFusion) for n, _ in m.named_buffers()}
    missing = [k for k in sd if k not in loaded and k not in fusion_keys]
    if verbose:
        LOGGER.info(
            f"MSMid conversion {src} -> {dst}: transferred {len(loaded)}/{len(sd)} tensors "
            f"({len(fusion_keys)} fusion tensors identity-initialized), ch={ch} (A={n_a}, B={n_b})"
        )
    if missing or skipped:
        raise RuntimeError(f"Incomplete weight transfer. missing={missing[:10]} skipped={skipped[:10]}")

    model.args = {**ckpt_args, "model": cfg}
    model.pt_path = str(dst)
    model.task = "detect"
    ckpt = {
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "epoch": -1,
        "best_fitness": None,
        "model": deepcopy(model).half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": model.args,
        "train_metrics": None,
        "train_results": None,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    return dst


def get_msmid_weights(name: str, file: str | Path) -> str:
    """Resolve ``yolov8{scale}-msmid.pt``: download from the GitHub repo, or build it from the official YOLOv8 weights.

    Args:
        name (str): File name, e.g. ``yolov8m-msmid.pt``.
        file (str | Path): Requested local path.

    Returns:
        (str): Local path to the checkpoint.
    """
    from ultralytics.utils.downloads import attempt_download_asset, safe_download

    file = Path(file)
    url = f"https://github.com/{MSMID_REPO}/raw/{MSMID_BRANCH}/weights/{name}"
    try:
        safe_download(url=url, file=file, min_bytes=1e5, retry=1)
        if file.exists():
            return str(file)
    except Exception as e:
        LOGGER.info(f"Pre-converted {name} not available ({e}); converting from official YOLOv8 weights instead.")
    scale = MSMID_NAME_RE.fullmatch(name).group(1)
    src = attempt_download_asset(f"yolov8{scale}.pt")
    return str(convert_yolov8_to_msmid(src, file))
