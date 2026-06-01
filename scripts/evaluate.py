#!/usr/bin/env python3
"""Proper mIoU evaluation over the WHOLE validation split.

Unlike visualize.py (which reports point accuracy on a single frame), this runs
the model on every val frame and reports the SemanticKITTI benchmark metric:
per-class IoU and their mean (mIoU), via mmdet3d's SegMetric.

Why mIoU and not accuracy: accuracy is dominated by the few huge classes (road,
vegetation, building) and hides failures on small classes. IoU is per-class and
penalises both missed points and false positives; mIoU is their unweighted mean,
so every class counts equally. See docs/04_evaluation_miou.md.

Usage:
    python3 scripts/evaluate.py                              # epoch_5, val split
    python3 scripts/evaluate.py --checkpoint .../epoch_10.pth
    python3 scripts/evaluate.py --config configs/cylinder3d_seq00_weighted.py \
                                --checkpoint .../cylinder3d_seq00_weighted/epoch_10.pth
"""
import argparse
import os

import torch

# trusted local checkpoint embeds an mmengine ConfigDict -> full unpickle
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

try:
    from mmdet3d.utils import register_all_modules
    register_all_modules()
except Exception:                               # pragma: no cover
    import mmdet3d  # noqa: F401

from mmengine.config import Config
from mmengine.runner import Runner

DEF_CFG = os.path.expanduser(
    "~/Autonomy/lidarseg/configs/cylinder3d_seq00.py")
DEF_CKPT = os.path.expanduser(
    "~/Autonomy/mmdetection3d/work_dirs/"
    "cylinder3d_4xb4-3x_semantickitti/epoch_5.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEF_CFG)
    ap.add_argument("--checkpoint", default=DEF_CKPT)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    cfg.work_dir = os.path.join(os.path.dirname(args.checkpoint), "eval")

    Runner.from_cfg(cfg).test()


if __name__ == "__main__":
    main()
