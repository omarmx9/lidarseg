#!/usr/bin/env python3
"""Filter the full SemanticKITTI train infos down to sequence 00 and split 80/20.

`create_data` builds `semantickitti_infos_train.pkl` covering ALL training
sequences (00-07, 09, 10 = 19130 frames), but we only downloaded sequence 00.
This selects the 4541 seq-00 frames and splits them:

    first 80% (3632 frames) -> train
    last  20% ( 909 frames) -> val

Contiguous (not random) on purpose: it mimics a real "drive earlier vs. drive
later" split, so val frames are genuinely unseen stretches of road rather than
neighbours of train frames. Pass --random for an i.i.d. shuffle instead.

Output (written next to the source pkl by default):
    semantickitti_infos_train_seq00.pkl
    semantickitti_infos_val_seq00.pkl

The pkl format is a plain dict:  {"metainfo": {...}, "data_list": [ {...}, ... ]}
where each entry has  lidar_points.lidar_path  and  pts_semantic_mask_path.
"""
import argparse
import os
import pickle
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root",
                    default=os.path.expanduser("~/Autonomy/semantickitti/dataset"))
    ap.add_argument("--src", default=None,
                    help="full train infos pkl "
                         "(default: <data-root>/semantickitti_infos_train.pkl)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the seq00 pkls (default: <data-root>)")
    ap.add_argument("--seq", default="00", help="sequence id to extract")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--random", action="store_true",
                    help="shuffle before splitting (default: contiguous)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = args.src or os.path.join(args.data_root, "semantickitti_infos_train.pkl")
    out_dir = args.out_dir or args.data_root
    os.makedirs(out_dir, exist_ok=True)

    with open(src, "rb") as f:
        data = pickle.load(f)
    metainfo = data["metainfo"]

    needle = f"sequences/{args.seq}/"
    seq = [e for e in data["data_list"]
           if needle in e["lidar_points"]["lidar_path"]]
    seq.sort(key=lambda e: e["lidar_points"]["lidar_path"])
    n = len(seq)
    if n == 0:
        raise SystemExit(f"No frames for sequence {args.seq} found in {src}")

    if args.random:
        random.Random(args.seed).shuffle(seq)
    n_train = int(args.train_frac * n)
    train, val = seq[:n_train], seq[n_train:]

    for name, lst in [("train", train), ("val", val)]:
        out = os.path.join(out_dir, f"semantickitti_infos_{name}_seq{args.seq}.pkl")
        with open(out, "wb") as f:
            pickle.dump({"metainfo": metainfo, "data_list": lst}, f)
        print(f"  wrote {out}  ({len(lst)} frames)")
    print(f"seq {args.seq}: {n} frames -> {len(train)} train / {len(val)} val "
          f"({'random' if args.random else 'contiguous'} split)")


if __name__ == "__main__":
    main()
