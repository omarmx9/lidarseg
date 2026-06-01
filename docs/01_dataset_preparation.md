# Task 1 (Part A) — Dataset Preparation, explained from scratch

> You said you've worked on 2D images but never on LiDAR sensor data. This file
> assumes exactly that. It explains **what the data is**, **what every file
> contains (datatypes)**, **how it gets loaded**, and **how we split it** — and
> why each choice was made.

---

## 1. What is this data, physically?

A spinning LiDAR (the KITTI car used a **Velodyne HDL-64E**) sits on the roof and
rotates 10 times a second. On each rotation, 64 laser beams fire outward and
measure the **time-of-flight** of the reflection. That gives a distance for each
beam at each angle → a **3D point**. One full rotation = one **frame** =
**~120,000 points** spread over a full 360°.

So unlike a camera image, a LiDAR frame is **not a grid of pixels**. It is an
**unordered set of points** floating in 3D space — a "point cloud". There is no
row/column; point #5 and point #6 are just two measurements that happen to be
next to each other in the file. This is the single biggest mental shift from 2D
vision:

| 2D image | LiDAR point cloud |
|----------|-------------------|
| Dense grid `H×W×3` | Unordered list `N×4` |
| Every pixel filled | Empty space is just *absent* points |
| Neighbours = adjacent pixels | Neighbours = nearby in 3D (must be searched) |
| `N` fixed | `N` varies per frame (~100k–130k) |
| Channels = R,G,B | Channels = x, y, z, **intensity** |

**Intensity** (the 4th number) is how strongly the laser came back — a rough
measure of surface reflectivity (road paint, license plates and signs reflect
strongly; matte/dark surfaces weakly). It is a real, useful signal.

---

## 2. The dataset: SemanticKITTI sequence 00

**KITTI** is a famous autonomous-driving dataset recorded in Karlsruhe, Germany.
**SemanticKITTI** took KITTI's LiDAR "odometry" sequences (22 drives, numbered
`00`–`21`) and added a **semantic label for every single point**. That hand
labelling is what makes it usable for segmentation.

We only downloaded **sequence 00** — one ~7-minute suburban drive:

| Property | Value |
|----------|------:|
| Frames (scans) | **4541** |
| Points per frame | ~124,668 (frame 0) |
| Velodyne data | 8.8 GB |
| Labels | part of a 179 MB zip |

Why only seq 00: the full velodyne dataset is **~80 GB**; we had a ~10 GB
bandwidth budget, so we pulled just seq 00 (see PROGRESS.md for the ZIP-range
trick). Seq 00 alone is plenty to learn the pipeline end-to-end.

### The files for one frame `000000`

```
sequences/00/
├── velodyne/000000.bin    ← the point cloud   (geometry + intensity)
├── labels/000000.label    ← per-point class   (the answer key)
├── image_2/000000.png     ← left color camera (NOT used for training)
├── calib.txt              ← sensor calibration (NOT used for training)
├── poses.txt, times.txt   ← ego-motion / timestamps (not used here)
```

> **Important:** training is **LiDAR-only** (`use_camera=False, use_lidar=True`).
> The camera image + `calib.txt` are only used by the ROS2 overlay
> (`kitti_seg_sim`) to *display* results on the photo. The model never sees a
> pixel.

---

## 3. Datatypes — exactly what's in each file

### `velodyne/000000.bin` — the point cloud
A **flat binary blob of `float32`**, no header. It's just `x,y,z,intensity`
repeated for every point:

```
[x0 y0 z0 i0  x1 y1 z1 i1  x2 y2 z2 i2  ... ]   all float32, little-endian
```

Read it in Python:
```python
import numpy as np
pts = np.fromfile("000000.bin", dtype=np.float32).reshape(-1, 4)
# pts[:, 0:3] = XYZ in metres (sensor frame), pts[:, 3] = intensity (0..1)
```
Sanity check on file size: `1,994,688 bytes ÷ 4 (bytes/float) ÷ 4 (floats/point)
= 124,668 points`. Coordinates are in **metres**, in the **LiDAR's own frame**
(x = forward, y = left, z = up).

### `labels/000000.label` — the answer key
A flat binary of **`uint32`, one per point**, in the *same order* as the `.bin`.
Each 32-bit value packs **two** numbers:

```
 31 ............ 16 | 15 ............ 0
   instance id      |   semantic id
```

- **Lower 16 bits** = the **semantic class** (what we want).
- **Upper 16 bits** = the **instance id** (which specific car/person — used for
  *panoptic* segmentation; we ignore it here).

Read it and keep only the class:
```python
raw = np.fromfile("000000.label", dtype=np.uint32) & 0xFFFF   # mask off instance
```

### `calib.txt` (reference only — used by the ROS overlay, not training)
Plain text. `P2` = the 3×4 projection matrix of the left color camera; `Tr` =
the 4×4 transform from the LiDAR frame to the camera frame. To paint points on
the photo: `uv = P2 · Tr · [x y z 1]ᵀ`. Not part of model training.

---

## 4. From 28 raw ids → 19 "learning" classes

SemanticKITTI's raw labels use sparse ids like `10, 11, 18, 40, 252…` (up to
259). Two problems:
1. The ids are **not contiguous** (a network wants `0,1,2,…`).
2. There are **moving-object duplicates**: a parked car is `10`, a *moving* car
   is `252`. For semantic (not panoptic) segmentation we don't care if it moves.

So we **remap** raw ids → a clean **0..18** range (19 classes), with a special
**id 19 = "ignore"** for everything unlabelled or irrelevant. This mapping is the
`labels_map` / `LUT` you'll see in the config and in `colormap.py`:

```python
LABEL_MAP = {10:0, 252:0,     # parked car + moving car  -> "car"
             11:1,            # bicycle
             ...,
             40:8, 60:8,      # road + lane-marking       -> "road"
             0:19, 1:19, ...} # unlabelled / outlier      -> ignore
```

The 19 classes (this is the order the model's output channels follow):

```
0 car          5 person       10 sidewalk      15 trunk
1 bicycle      6 bicyclist     11 other-ground  16 terrain
2 motorcycle   7 motorcyclist  12 building      17 pole
3 truck        8 road          13 fence         18 traffic-sign
4 other-veh.   9 parking       14 vegetation    (19 = ignore)
```

> **Cosmetic note:** the stock mmdet3d config misspells two names
> (`trunck`, `terrian`). Our `colormap.py` uses the correct `trunk`, `terrain`.
> The **order/ids are identical**, so it has zero effect on training — names are
> just display strings.

The "ignore" class (19) is special: during training the loss **skips** those
points entirely (`ignore_index=19`), and during scoring they don't count. They're
points the human annotators couldn't label confidently.

---

## 5. Building the index: `create_data` → `.pkl`

MMDetection3D doesn't read `sequences/00/...` directly during training. It first
builds an **index file** (a "`.pkl`") that lists, for every frame, where its
point cloud and label file are. This is done once:

```bash
cd ~/Autonomy/mmdetection3d
python3 -m tools.create_data semantickitti \
    --root-path ~/Autonomy/semantickitti/dataset \
    --out-dir   ~/Autonomy/semantickitti/dataset \
    --extra-tag semantickitti
```

This produced `semantickitti_infos_{train,val,test}.pkl`. The structure is a
plain Python dict (we verified this):

```python
{
  "metainfo": {"DATASET": "SemanticKitti"},
  "data_list": [
     {
       "lidar_points": {"lidar_path": "sequences/00/velodyne/000000.bin",
                        "num_pts_feats": 4},
       "pts_semantic_mask_path": "sequences/00/labels/000000.label",
       "sample_id": ...
     },
     ...                      # one dict per frame
  ]
}
```

So the pkl is just a **table of contents** — paths, not data. Light to load,
and the heavy `.bin`/`.label` files are read lazily during training.

---

## 6. Splitting: why we made our own train/val

The official `create_data` train pkl lists **all** training sequences
(00–07, 09, 10) = **19,130 frames** — but we only *have* seq 00. So we wrote
[`scripts/split_seq00.py`](../scripts/split_seq00.py) to:

1. **Filter** the big train pkl down to entries whose path contains
   `sequences/00/` → 4541 frames.
2. **Split 80/20**, *contiguously*: first **3632** frames → train, last **909**
   → val.

```
frames 000000 ............. 003631 | 003632 ......... 004540
        train  (3632, 80%)         |      val (909, 20%)
```

**Why contiguous, not a random shuffle?** Frame 100 and frame 101 are 0.1 s
apart — almost the same scene. If we shuffled randomly, near-identical frames
would land in *both* train and val, and val accuracy would be inflated (the model
effectively "saw" the answer). A contiguous split puts a *later stretch of the
drive* in val, so val is genuinely less-seen geometry. It's the honest choice
for a single sequence. (`--random` is available if you ever want i.i.d. splits
across many sequences.)

> This script is **verified** to reproduce the exact pkls already on disk
> (identical 3632/909 path lists), so it's a faithful record of how the split
> was made — and re-runnable if you add data.

Result: `semantickitti_infos_train_seq00.pkl` (3632) and
`semantickitti_infos_val_seq00.pkl` (909), both pointing only at seq 00.

---

## 7. How a frame is loaded during training (the data pipeline)

For each frame, mmdet3d runs a **pipeline** of transforms (think of it like a
torchvision `Compose`, but for point clouds). From our config, the **train**
pipeline is:

| Step | What it does | Why |
|------|--------------|-----|
| `LoadPointsFromFile(load_dim=4, use_dim=4)` | read the `.bin` → `N×4` array | get x,y,z,intensity |
| `LoadAnnotations3D(with_seg_3d=True, seg_offset=65536)` | read the `.label`, take lower 16 bits | get per-point class |
| `PointSegClassMapping` | apply the raw→0..18 remap | clean contiguous ids |
| `RandomFlip3D(0.5, 0.5)` | randomly mirror across X and/or Y | **augmentation** |
| `GlobalRotScaleTrans(rot ±45°, scale 0.95–1.05, trans σ=0.1)` | random rotate/scale/jitter the whole cloud | **augmentation** |
| `Pack3DDetInputs(keys=[points, pts_semantic_mask])` | bundle into the model's input format | hand off to model |

The **val/test** pipeline is the same minus the two augmentation steps (you
never augment evaluation data — you want to measure the real thing).

### Why these augmentations specifically?
- **Flip / rotate / scale** are *label-preserving* for LiDAR: a car rotated 30°
  or mirrored left-right is still a car, and every point keeps its class. They
  cheaply multiply our effective data — important when we only have one
  sequence. Rotation especially teaches orientation-invariance, which matters
  because the same object appears at any heading as the car drives.
- **Translation jitter (σ=0.1 m)** makes the voxel grid boundaries fall
  differently each time, so the model doesn't overfit to a fixed voxel alignment.
- We **do not** add color/intensity noise or dropout here — keeping it simple and
  matching the stock Cylinder3D recipe.

---

## 8. The geometry step that makes Cylinder3D special: cylindrical voxels

Before the network, points are grouped into **voxels** (3D cells). Most methods
use a **cartesian** grid (cubes of fixed size in x,y,z). Cylinder3D instead uses
a **cylindrical** grid — it partitions space by **(radius, angle, height)**
instead of (x, y, z):

```
point_cloud_range = [0, -π, -4,  50, +π, 2]      # [ρ_min,θ_min,z_min, ρ_max,θ_max,z_max]
grid_shape        = [480, 360, 32]               # cells along (ρ, θ, z)
```

So space is sliced into 480 radial rings × 360 angular wedges × 32 height
layers. **Why cylindrical?** A spinning LiDAR's points get **sparser with
distance** (beams fan out). Cartesian cubes far away end up nearly empty while
nearby cubes are jammed. Cylindrical cells *grow* with radius, which keeps a more
**even number of points per cell** from near to far — a much better match to how
the sensor actually samples the world. This is the core idea of the paper and a
big reason it works well on LiDAR. (More on the model itself in
[02_model_and_training.md](02_model_and_training.md).)

---

## 9. One-screen summary

```
.bin  (float32  N×4: x,y,z,intensity)   ─┐
.label(uint32   N: instance<<16 | class) ─┤ create_data → infos_*.pkl (path index)
                                          │        │
              raw ids ──LUT──► 0..18 (+19 ignore)  │ split_seq00.py (filter + 80/20)
                                          │        ▼
                          train pipeline: load → map → flip/rot/scale → pack
                                          │        ▼
                          cylindrical voxels (480×360×32) → Cylinder3D
```

**Datatypes at a glance:** points `float32`, labels `uint32` (class in low 16
bits), learning ids `int64` in `0..18`, ignore = `19`, index files are pickled
`dict`s of paths.

Next: **[02_model_and_training.md](02_model_and_training.md)** — the model and
every training decision.
