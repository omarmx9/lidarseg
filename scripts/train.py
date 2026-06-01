#!/usr/bin/env python3
"""Train (or resume) Cylinder3D with our clean seq-00 config.

This is a thin, patch-safe replacement for `mmdet3d/tools/train.py`:
  * applies the torch.load(weights_only=False) patch so resuming/warm-starting
    from our own checkpoints works under PyTorch >= 2.6, and
  * reads everything else (data paths, batch size, schedule) from the config,
    so there is no wall of --cfg-options to remember.

Examples:
    python3 scripts/train.py                              # from scratch
    python3 scripts/train.py --resume                     # resume latest in work_dir
    python3 scripts/train.py --resume .../epoch_5.pth     # resume a specific ckpt
    python3 scripts/train.py --config configs/cylinder3d_seq00_weighted.py
"""
import argparse
import os

import torch

# Our checkpoints embed an mmengine ConfigDict; PyTorch >= 2.6 defaults
# torch.load to weights_only=True and would reject it. Trusted local files.
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

try:                                            # populate the mmdet3d registry
    from mmdet3d.utils import register_all_modules
    register_all_modules()
except Exception:                               # pragma: no cover - version drift
    import mmdet3d  # noqa: F401

from mmengine.config import Config
from mmengine.runner import Runner

DEF_CFG = os.path.expanduser(
    "~/Autonomy/lidarseg/configs/cylinder3d_seq00.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEF_CFG)
    ap.add_argument("--resume", nargs="?", const=True, default=False,
                    help="resume latest in work_dir, or a given checkpoint path")
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.resume is True:
        cfg.resume = True
    elif args.resume:
        cfg.resume = True
        cfg.load_from = args.resume

    Runner.from_cfg(cfg).train()


if __name__ == "__main__":
    main()
