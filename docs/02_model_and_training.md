# Task 1 (Part B) — Model choice & training decisions, explained

> Continues from [01_dataset_preparation.md](01_dataset_preparation.md). Here:
> **what model we used, how it's built, and the reason behind every training
> setting.** Each number below comes straight from
> [`configs/cylinder3d_seq00.py`](../configs/cylinder3d_seq00.py) (our overrides)
> and the stock Cylinder3D config it inherits.

---

## 1. The model: Cylinder3D — and why it was chosen

**Cylinder3D** is a 3D sparse-convolution U-Net designed *specifically* for
rotating-LiDAR semantic segmentation. Three reasons it's a good first choice
here:

1. **It's built for the sensor.** As covered in Part A, it voxelizes in
   **cylindrical** coordinates (ρ, θ, z), which matches how a spinning LiDAR
   samples the world (dense near, sparse far). Cartesian-voxel methods waste
   capacity on empty far-away cubes.
2. **It's sparse and therefore cheap.** A LiDAR scan is ~99% empty space.
   Cylinder3D uses **sparse convolutions** (via `spconv`) that only compute where
   points exist, so a 480×360×32 grid is tractable on an **8 GB laptop GPU**.
3. **It's in MMDetection3D with a ready SemanticKITTI recipe**, so we get a
   tested config, data pipeline, and pretrained-style schedule instead of
   building from zero.

The trade-off we accepted: Cylinder3D is **fp32-only** in this stack (see §4),
and a single scan still takes ~0.28 s to infer. For learning the full 3D-seg
pipeline on a laptop, that's a fine price.

---

## 2. How the network is wired (data → logits)

```mermaid
%%{init: {"theme":"dark","themeVariables":{"primaryTextColor":"#fff","lineColor":"#00BFFF","primaryColor":"#1f2937","secondaryColor":"#374151","tertiaryColor":"#4b5563"}}}%%
flowchart LR
    P["points N×4\n(x,y,z,intensity)"]:::b --> V["cylindrical voxelize\n480×360×32"]:::o
    V --> E["SegVFE\nper-voxel features → 16-D"]:::g
    E --> B["Asymm3DSpconv\n3D sparse U-Net (base 32)"]:::g
    B --> H["Cylinder3DHead\n128→20 logits per point"]:::p
    H --> L["CE + Lovász loss\n(vs per-point labels)"]:::pk
    classDef b fill:#2196F3,color:#fff
    classDef o fill:#FF9800,color:#fff
    classDef g fill:#4CAF50,color:#fff
    classDef p fill:#9C27B0,color:#fff
    classDef pk fill:#E91E63,color:#fff
```

Piece by piece (all values from the config):

| Stage | Config | What it does |
|-------|--------|--------------|
| **Voxelization** | `voxel_type='cylindrical'`, `point_cloud_range=[0,-π,-4, 50,π,2]`, `grid_shape=[480,360,32]` | bucket points into cylindrical cells; range = 0–50 m radius, full 360°, −4…+2 m height |
| **Voxel encoder** | `SegVFE`, `in_channels=6`, `feat_channels=[64,128,256,256]`, `feat_compression=16` | turns the points inside each voxel (their coords, intensity, and offset to the voxel centre — 6 numbers) into one **16-D feature per voxel** |
| **Backbone** | `Asymm3DSpconv`, `base_channels=32`, `input_channels=16` | an **asymmetric residual 3D sparse U-Net** — downsample→upsample with skip connections — that mixes information across neighbouring voxels |
| **Head** | `Cylinder3DHead`, `channels=128`, `num_classes=20` | projects each point's feature to **20 class logits** (19 classes + ignore slot) |

`num_classes=20`, not 19, because the ignore index (19) is a real output channel;
the loss just never *targets* it. (This 20 vs 19 detail matters for class
weighting — see [03_class_weighting.md](03_class_weighting.md).)

---

## 3. The loss: cross-entropy **plus** Lovász — why both

```python
loss_ce     = CrossEntropyLoss(use_sigmoid=False, loss_weight=1.0, class_weight=None)
loss_lovasz = LovaszLoss(loss_weight=1.0, reduction='none')
total = loss_ce + loss_lovasz
```

- **Cross-entropy** is the standard per-point classifier loss: "for this point,
  push the probability of the correct class up." It's stable and easy to
  optimize, but it treats every *point* equally — so the few giant classes
  (road, vegetation) dominate the gradient.
- **Lovász-softmax** is a clever surrogate that **directly optimizes IoU** (the
  metric we actually care about — see [04_evaluation_miou.md](04_evaluation_miou.md)).
  Because IoU is *per-class*, Lovász cares about a small class's region as much
  as a big one's. It partially counteracts class imbalance on its own.

Using both = "classify each point correctly" (CE) **and** "make each class's
predicted region overlap the truth well" (Lovász). This pairing is the standard
Cylinder3D recipe and a big reason it handles imbalance reasonably even before we
add class weights. `class_weight=None` here means **no manual weighting yet** —
that's exactly the lever Task 2 turns on.

---

## 4. The decisions we changed for an 8 GB laptop — and why

The config name `cylinder3d_**4xb4**-3x` means the authors trained on **4 GPUs ×
batch 4 = effective batch 16**, "3x" schedule (36 epochs). We have **one 8 GB
GPU**. Here is every deviation and its reason:

| Setting | Stock | Ours | Why |
|---------|-------|------|-----|
| **Precision** | (fp32) | **fp32, forced** | fp16/AMP **crashes**: spconv's `feats_reduce_kernel` has no Half implementation. This is non-negotiable in this stack. |
| **batch_size** | 4 ×4 GPU | **1** | A full scan + the 480×360×32 sparse grid in fp32 is what fits 8 GB (~3 GB used). |
| **num_workers** | — | **4** | overlap CPU data loading with GPU compute so the GPU isn't starved. |
| **data_root / ann_file** | `data/semantickitti`, all-seq pkl | our paths + `*_seq00.pkl` | point training at our filtered seq-00 split. |
| **everything else** | inherited | inherited | architecture, LR, schedule, augmentation all kept stock so results stay comparable to the paper. |

These are the *only* knobs in `cylinder3d_seq00.py` — the whole point of the
restructure (Task 4) is that the config now says, in 4 short blocks, exactly what
differs from the reference. No more giant `--cfg-options` string.

### Optimizer & schedule (inherited — explained so you know what's running)

| Thing | Value | Meaning / why |
|-------|-------|---------------|
| Optimizer | `AdamW`, `lr=0.001`, `weight_decay=0.01` | AdamW = Adam with decoupled weight decay; robust default for segmentation, little tuning needed. |
| Warmup | `LinearLR`, `start_factor=0.001`, first **1000 iters** | ramp LR from ~0 to full over 1000 steps so early updates don't blow up. |
| Main schedule | `MultiStepLR`, `milestones=[30]`, `gamma=0.1`, `max_epochs=36` | hold LR, then drop ×0.1 at epoch 30 to fine-tune. |
| Loop | `EpochBasedTrainLoop`, `val_interval=1` | validate every epoch. |
| Checkpoints | `CheckpointHook(interval=5)` | save every 5 epochs (hence `epoch_5.pth`). |

> ⚠️ **Honest caveat worth knowing:** that `lr=0.001` was tuned for **effective
> batch 16**. We run **effective batch 1** — 16× smaller — but kept the same LR.
> In practice it trained fine (loss 3.58 → 1.38 in 400 iters, 88% point accuracy
> by epoch 5), but a smaller batch with a large LR makes gradients noisier. If
> convergence ever looks unstable, the textbook fix is to **lower the LR**
> (roughly linearly with batch size, so ~6e-5), or accumulate gradients over
> several scans to fake a bigger batch. We didn't need to, but that's the lever.

---

## 5. What "training" actually did, and where it stopped

- Ran `tools/train.py` with the config → ~**0.41 s/iter**, ~**3 GB VRAM**.
- Full 36 epochs ≈ **15 hours** on this laptop. We **stopped at epoch 5**
  (`epoch_5.pth`) — enough to prove the pipeline and get a usable model.
- At epoch 5, on val frame `004000`: **88.4 % point accuracy**. Big surfaces
  (road 98 %, vegetation 97 %, building 94 %) are already near-perfect; the weak
  spots are **small/rare classes** — `car` (54 %), `trunk`/`traffic-sign` (~0 %,
  only a handful of points in that frame).

That weakness pattern is the whole motivation for **Task 2 (class weighting)**
and the reason **point accuracy is a misleading headline** — which is **Task 3
(mIoU)**.

---

## 6. Inference path (how predictions are produced)

Both the visualizer and the ROS node call mmdet3d's `inference_segmentor`:

```python
from mmdet3d.apis import init_model, inference_segmentor
model = init_model(CONFIG, "epoch_5.pth", device="cuda:0")   # build + load weights
result, _ = inference_segmentor(model, "000000.bin")          # forward one scan
pred = result.pred_pts_seg.pts_semantic_mask.cpu().numpy()    # int per point, 0..18
```

- `test_cfg=dict(mode='whole')` → the whole scan is segmented in one forward
  pass (no tiling).
- The checkpoint stores an mmengine `ConfigDict`, so `torch.load` must run with
  `weights_only=False` (PyTorch ≥2.6 defaults to `True` and would refuse it).
  Every script here applies that one-line patch — it's safe because the
  checkpoint is our own trusted file.

---

## 7. How to (re)train with the restructured project

```bash
cd ~/Autonomy/lidarseg
source env.sh

# resume the existing run toward convergence
make resume                      # = scripts/resume.sh  (latest ckpt in work_dir)

# or train fresh
make train                       # = scripts/train.sh   (config = cylinder3d_seq00.py)

# train the class-weighted variant (Task 2)
python3 scripts/train.py --config configs/cylinder3d_seq00_weighted.py
```

Everything that used to be a 12-line `--cfg-options` wall now lives in the
config file. Next:
**[03_class_weighting.md](03_class_weighting.md)** — turning the `class_weight`
lever, including optimizing one class on its own.
