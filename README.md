# lidarseg — Cylinder3D on SemanticKITTI (training & data project)

The **training / dataset side** of the 3D LiDAR semantic-segmentation project,
restructured for easy edits and follow-ups. The live RViz2 demo lives separately
in the ROS 2 package `~/Autonomy/ros2_ws/src/kitti_seg_sim/`; this repo is where
you **prepare data, train, weight classes, and evaluate**.

> New to LiDAR data or this project? Read the docs in order — they assume a 2D
> vision background and explain everything from the raw `.bin` up.

---

## 📚 Documentation (read in order)

| # | Doc | Answers |
|---|-----|---------|
| 1 | [docs/01_dataset_preparation.md](docs/01_dataset_preparation.md) | What LiDAR data *is*, every file's datatype, the raw→19-class mapping, the pkl index, and the 80/20 split. |
| 2 | [docs/02_model_and_training.md](docs/02_model_and_training.md) | Cylinder3D's architecture and **every training decision + why** (batch size, fp32, LR, schedule, the CE+Lovász loss). |
| 3 | [docs/03_class_weighting.md](docs/03_class_weighting.md) | What class weighting is, **how to optimize one class alone**, and the data-driven helper. |
| 4 | [docs/04_evaluation_miou.md](docs/04_evaluation_miou.md) | Why **mIoU**, not the 88 % point accuracy, is the metric that matters — with the one command to compute it. |

---

## 🗂 Project layout

```
lidarseg/
├── README.md                      ← you are here
├── env.sh                         source first: CUDA 13.0 + project paths
├── Makefile                       make split | weights | train | resume | eval | viz
├── docs/                          the four explainer docs above
├── configs/
│   ├── cylinder3d_seq00.py        clean training config (inherits stock, ~4 overrides)
│   └── cylinder3d_seq00_weighted.py   per-class-weighted variant (Task 2)
└── scripts/
    ├── split_seq00.py             filter full infos → seq00, 80/20 split (verified reproducible)
    ├── compute_class_weights.py   scan labels → ready-to-paste class_weight vector
    ├── train.py / train.sh        train (patch-safe Runner; no --cfg-options wall)
    ├── resume.sh                  resume the latest checkpoint
    ├── evaluate.py / evaluate.sh  per-class IoU + mIoU over the whole val split
    └── visualize.py               GT / prediction / error PNGs + PLYs for one frame
```

**Why this shape:** the old workflow was one giant `tools/train.py … --cfg-options
<12 lines>` command plus loose scripts. Now every knob lives in a named config,
each task is one `make` target, and the four docs explain the *why*. The heavy
lifting still uses the installed `mmdetection3d`; this project is the clean front
door to it.

---

## ⚙️ Prerequisites (already set up on this machine)

- PyTorch 2.12.0+cu130, **CUDA toolkit 13.0** (`CUDA_HOME=/usr/local/cuda-13.0`)
- mmdet3d 1.4.0 · mmcv 2.1.0 · mmdet 3.3.0 · spconv-cu120 · `numpy<2`
- Data at `~/Autonomy/semantickitti/dataset/sequences/00/`
- Baseline checkpoint at
  `~/Autonomy/mmdetection3d/work_dirs/cylinder3d_4xb4-3x_semantickitti/epoch_5.pth`

See the repo-root `PROGRESS.md` for how that environment was built (it was not
trivial).

---

## 🚀 Quickstart

```bash
cd ~/Autonomy/lidarseg
source env.sh

# (re)generate the seq-00 train/val pkls  — verified to match what's on disk
make split

# resume training the baseline toward convergence
make resume

# proper evaluation: per-class IoU + mIoU over the 909 val frames
make eval

# render a frame (GT / prediction / error) to ~/Autonomy/viz/
make viz
```

### Class-weighting experiment (Task 2)
```bash
# 1. get data-driven weights (optional)
make weights
# 2. paste into configs/cylinder3d_seq00_weighted.py  (or just bump one class)
# 3. warm-start train the weighted variant
python3 scripts/train.py --config configs/cylinder3d_seq00_weighted.py
# 4. compare mIoU vs baseline
python3 scripts/evaluate.py --config configs/cylinder3d_seq00_weighted.py \
        --checkpoint ~/Autonomy/mmdetection3d/work_dirs/cylinder3d_seq00_weighted/epoch_5.pth
```

---

## 🔧 Editing guide

| Want to… | Edit |
|----------|------|
| Point at different data / split | `configs/cylinder3d_seq00.py` (`data_root`, `ann_file`) |
| Change batch size / workers | `configs/cylinder3d_seq00.py` → `train_dataloader` |
| Tune class weights (one or all) | `configs/cylinder3d_seq00_weighted.py` → `CLASS_WEIGHT` |
| Re-make the seq-00 split | `scripts/split_seq00.py` (`--random`, `--train-frac`) |
| Compute weights from data | `scripts/compute_class_weights.py` |
| Recolor visualizations | `scripts/visualize.py` → `PALETTE` |

After editing a config, just re-run `make train` / `make eval` — no rebuild step.

---

## ⚠️ Stack gotchas (don't relearn the hard way)

- **fp32 only.** Cylinder3D + AMP/fp16 crashes (spconv `feats_reduce_kernel`
  has no Half kernel). The config stays fp32 on purpose.
- **`weights_only` checkpoints.** Our checkpoints embed an mmengine `ConfigDict`;
  `train.py`/`evaluate.py` patch `torch.load(weights_only=False)` so loading
  works under PyTorch ≥2.6. Don't remove that patch.
- **Our val = tail of seq 00**, not the official seq-08 benchmark — mIoU here is
  for comparing *your own* runs, not the public leaderboard (see doc 4 §6).
