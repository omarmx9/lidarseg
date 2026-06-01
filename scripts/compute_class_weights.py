#!/usr/bin/env python3
"""Compute per-class loss weights from the training labels.

Scans the .label files referenced by the train pkl, counts how many points each
of the 19 classes has, and prints a ready-to-paste `class_weight` vector of
length 20 (the trailing 1.0 is the ignore slot, index 19).

Two weighting schemes are printed so you can compare:
  * inverse-frequency      w_c = N_total / N_c           (aggressive)
  * median-frequency       w_c = median(freq) / freq_c   (gentler, common in seg)
both are clipped and normalised so the most common class sits near 1.0.

Usage:
    python3 scripts/compute_class_weights.py                 # sample 400 frames
    python3 scripts/compute_class_weights.py --max-frames 0  # use every frame
Paste the printed vector into configs/cylinder3d_seq00_weighted.py.
"""
import argparse
import os
import pickle

import numpy as np

CLASSES = (
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
)

# raw SemanticKITTI label id -> 0..18 learning id (19 = ignore)
LABEL_MAP = {
    0: 19, 1: 19, 10: 0, 11: 1, 13: 4, 15: 2, 16: 4, 18: 3, 20: 4, 30: 5,
    31: 6, 32: 7, 40: 8, 44: 9, 48: 10, 49: 11, 50: 12, 51: 13, 52: 19,
    60: 8, 70: 14, 71: 15, 72: 16, 80: 17, 81: 18, 99: 19, 252: 0, 253: 6,
    254: 5, 255: 7, 256: 4, 257: 4, 258: 3, 259: 4,
}
IGNORE = 19
LUT = np.full(260, IGNORE, dtype=np.int64)
for _k, _v in LABEL_MAP.items():
    LUT[_k] = _v


def fmt_vector(weights, names):
    """Pretty-print a length-20 python list with a comment per class."""
    lines = ["CLASS_WEIGHT = ["]
    for i, w in enumerate(weights):
        name = names[i] if i < len(names) else "ignore"
        lines.append(f"    {w:5.2f},   # {i:<2d} {name}")
    lines.append("    1.00,   # 19 ignore (unused)")
    lines.append("]")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root",
                    default=os.path.expanduser("~/Autonomy/semantickitti/dataset"))
    ap.add_argument("--pkl", default=None,
                    help="train infos pkl (default: <data-root>/semantickitti_infos_train_seq00.pkl)")
    ap.add_argument("--max-frames", type=int, default=400,
                    help="sample at most this many frames evenly (0 = all)")
    args = ap.parse_args()

    pkl = args.pkl or os.path.join(args.data_root,
                                   "semantickitti_infos_train_seq00.pkl")
    with open(pkl, "rb") as f:
        entries = pickle.load(f)["data_list"]

    if args.max_frames and len(entries) > args.max_frames:
        step = len(entries) / args.max_frames
        entries = [entries[int(i * step)] for i in range(args.max_frames)]

    counts = np.zeros(19, dtype=np.int64)
    for e in entries:
        lbl_path = os.path.join(args.data_root, e["pts_semantic_mask_path"])
        raw = np.fromfile(lbl_path, dtype=np.uint32) & 0xFFFF
        learn = LUT[raw]
        valid = learn < IGNORE
        binc = np.bincount(learn[valid], minlength=19)
        counts += binc

    total = counts.sum()
    freq = counts / max(total, 1)

    print(f"\nScanned {len(entries)} frames, {total:,} labelled points.\n")
    print(f"{'id':>2}  {'class':<14} {'points':>12} {'freq':>8}")
    for i in range(19):
        print(f"{i:>2}  {CLASSES[i]:<14} {counts[i]:>12,} {freq[i]:>8.4f}")

    # avoid div-by-zero for classes absent in the sample
    safe = np.where(counts > 0, freq, freq[freq > 0].min() if (freq > 0).any() else 1.0)

    inv = 1.0 / safe
    inv = inv / inv.min()                     # most common class -> 1.0
    inv = np.clip(inv, 1.0, 50.0)

    med = np.median(safe)
    medf = med / safe
    medf = np.clip(medf, 0.5, 50.0)

    print("\n--- inverse-frequency (aggressive) ---")
    print(fmt_vector(inv, CLASSES))
    print("\n--- median-frequency (gentler, recommended starting point) ---")
    print(fmt_vector(medf, CLASSES))
    print("\nPaste one of the above into configs/cylinder3d_seq00_weighted.py")


if __name__ == "__main__":
    main()
