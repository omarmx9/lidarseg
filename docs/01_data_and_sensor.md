# The data & the sensor — everything before the model

This is the single reference for **what the data is, where it comes from, how it's
prepared, and how to visualize it.** It's organized around the questions a reader
actually asks. Every number is measured from the files on disk (seq 00, frame
004000 for the figures) by the scripts under [`scripts/`](../scripts/) — no model,
no GPU needed for any of it.

Three parts:
1. **The sensor** — what a LiDAR is, how it measures, what a scan looks like.
2. **The dataset on disk** — files, datatypes, label mapping, the split, the load
   pipeline.
3. **Seeing it** — projecting a 3D segmentation onto the camera photo.

---

# Part 1 — The sensor

## What is LiDAR data, and how is it different from a 2D image?

A spinning LiDAR (KITTI used a **Velodyne HDL-64E**) sits on the car roof and
rotates 10×/second. On each rotation, 64 laser beams fire outward and measure the
**time-of-flight** of each reflection → a distance per beam per angle → a **3D
point**. One full rotation = one **frame** ≈ **~125,000 points** over a full 360°.

Unlike a camera image, a LiDAR frame is **not a grid of pixels**. It's an
**unordered set of points** in 3D — a "point cloud." There is no row/column; point
#5 and #6 are just two measurements that happen to sit next to each other in the
file. That's the biggest mental shift coming from 2D vision:

| 2D image | LiDAR point cloud |
|----------|-------------------|
| Dense grid `H×W×3` | Unordered list `N×4` |
| Every pixel filled | Empty space is just *absent* points |
| Neighbours = adjacent pixels | Neighbours = nearby in 3D (must be searched) |
| `N` fixed | `N` varies per frame (~100k–130k) |
| Channels = R,G,B | Channels = x, y, z, **intensity** |

**Intensity** (the 4th number) is how strongly the laser came back — a rough
measure of surface reflectivity (road paint, plates, and signs reflect strongly;
matte/dark surfaces weakly). It's a real, useful signal the model gets for free.

## Is there a "3D camera"?

No. KITTI's sensor roof has **two kinds of sensor**, and only one is 3D:

| Sensor | What it gives | 2D or 3D? |
|--------|---------------|-----------|
| **Velodyne HDL-64E LiDAR** | a 3D point cloud (x, y, z + intensity) | **3D** |
| **Cameras** (2 grayscale + 2 color, in stereo pairs) | flat photos | **2D** |

The cameras are **ordinary 2D cameras** — not depth/RGB-D cameras. KITTI has a
stereo pair, and stereo *can* estimate depth in principle, but **this project never
uses that**: the camera is only a normal 2D photo (the left color camera, "cam 2")
that we paint the segmentation onto (Part 3). **All the 3D comes from the LiDAR.**

## How does the Velodyne HDL-64E work?

Picture a **vertical stack of 64 laser beams**, each fixed at a slightly different
up/down angle, on a head that **spins ~10×/second**. As it spins:

- each of the 64 lasers sweeps out one **horizontal circle** → a **"ring,"**
- 64 lasers → **64 rings**, nested like ripples on a pond,
- one full 360° turn (~0.1 s) = **one frame** ≈ 125,000 points.

So the beams form a **vertical fan** that rotates. Measured from the actual points
of frame 004000 by [`scripts/sensor_geometry.py`](../scripts/sensor_geometry.py):

```
125,528 points
vertical FOV : +6.1° (up)  →  −25.7° (down)   = 31.8° tall
horizontal   : full 360°
range        : 1.2 m near  →  80 m far
~points/beam : 1,961  (× 64 beams ≈ 125k)
```

The fan points **mostly downward** (about +6° up to −26° down) because the sensor
sits ~1.7 m up and the interesting stuff (road, cars, curbs) is *below* it. It can
barely see above horizontal, so the **sky and the tops of nearby tall buildings
are simply not measured.**

A thin front-facing slice — *distance ahead* vs *height*, colored by which beam
(elevation angle) produced each point:

![side fan](images/sensor_side_fan.png)

- The **flat line at z ≈ −1.7 m** is the **road** (the LiDAR is the origin, the
  road is 1.7 m below it).
- **Steep-down beams** (purple/blue, ≈ −20°) hit the road **close**; **shallow
  beams** (orange/red, near 0°) reach the road **far away** — that's why a flat
  road becomes rings at increasing distance.
- The vertical clumps at 30–45 m are **buildings/vegetation**.
- Dashed lines mark the **top (+6.1°)** and **bottom (−25.7°)** beams — the edges
  of the vertical FOV.

## How is the distance to each point actually measured?

A LiDAR measures distance like sonar, but with **light**: it fires a short pulse
and times how long until the reflection returns
([`scripts/dataset_overview.py`](../scripts/dataset_overview.py)):

![time of flight](images/dist_tof_schematic.png)

```
        c · Δt
   d = ────────          c = speed of light ≈ 3 × 10⁸ m/s,  Δt = round-trip time
          2
```

The `÷2` is because the light travels to the object **and back**. What's
remarkable is the **timing precision required**:

| distance | round-trip time Δt |
|----------|--------------------|
| 50 m | **334 ns** |
| 80 m (far edge of this scan) | 534 ns |
| **to resolve 2 cm** | the clock must tick to **~133 ps** (picoseconds) |

Light moves ~30 cm per nanosecond, so to place a point to a few centimetres the
electronics time the echo to **hundreds of picoseconds** — repeated ~1.3 million
times a second. The strength of the returning pulse is also recorded — that's the
**intensity** channel.

## How does a measured distance become (x, y, z)?

The sensor doesn't natively know `x, y, z`. For each return it knows:

- **range `r`** — the time-of-flight distance,
- **azimuth `α`** — the head's spin angle when it fired,
- **elevation `ω`** — *which* of the 64 beams fired (each has a fixed angle).

That's a point in **spherical coordinates**, converted to cartesian with
trigonometry:

![spherical to cartesian](images/dist_spherical.png)

```
x = r · cos(ω) · cos(α)      (forward)
y = r · cos(ω) · sin(α)      (left)
z = r · sin(ω)               (up)
```

And you can always **invert** it to recover the measured distance:

```
r          = √(x² + y² + z²)     ← the line-of-sight range (what was timed)
√(x² + y²) =                      ← distance along the ground (top-down / BEV)
```

The overview script reconstructs `x,y,z` from `(r, α, ω)` and matches the stored
values to **1.1 × 10⁻⁵ m** — the spherical model is exact; the residual is just
`float32` rounding. (Don't mix the two "distances": **range** `√(x²+y²+z²)` is the
slant distance the laser measured; **ground distance** `√(x²+y²)` is its shadow on
the ground.)

## Why is the data "striped" and sparse far away?

Two consequences of the 64-beam geometry explain most of what you'll see:

1. **Rings / scan-lines.** 64 discrete beams → projecting onto the camera or the
   ground shows **horizontal stripes**, each stripe one beam (visible in Part 3).
2. **Density collapses with distance.** Beams **fan apart** as they travel: two
   rings ~10 cm apart at 5 m are **several metres apart at 40 m**. Half the points
   in a scan are within **5.9 m**, 90 % within **15.7 m**:

![range histogram](images/dist_range_histogram.png)

```
0–10 m : ~93,900 points        ← three-quarters of the whole scan
10–20 m: ~17,800
20–30 m:  ~6,000
30–40 m:  ~1,700
40–50 m:  ~1,100
```

![bev rings](images/dist_bev_rings.png)

A car at 40 m might be only a **handful of points** while the road at the bumper
has tens of thousands. This single fact is why far/small objects are hard to
segment — and why Cylinder3D voxelizes in **cylindrical** coordinates, whose cells
grow with range (see the end of Part 2).

The 64 beams are even visible directly in the stored numbers — histogram every
point's elevation angle and it clumps into **discrete spikes**, one per beam:

![elevation peaks](images/dist_elevation_peaks.png)

## Can LiDAR be viewed as a 2D image?

Yes — the bridge from 2D vision. Because every point has an **azimuth** and an
**elevation**, you can **unwrap** the 360° scan into a flat **range image**:

- **rows** = the 64 beams (top beam → bottom beam),
- **columns** = the spin angle (rear → right → front → left → rear),
- **pixel value** = the range (or the class).

Colored by distance, then by **segmentation class** (*the LiDAR as a normal 2D
image*):

![range image depth](images/sensor_range_image_depth.png)
![range image class](images/sensor_range_image_class.png)

**Magenta road fills the bottom rows**, **green vegetation and yellow buildings
sit higher**, **blue cars** are blobs, and the **top rows are empty** (beams shot
into the sky). A whole family of methods (RangeNet++, SqueezeSeg) segment *this*
2D image with ordinary 2D CNNs.

> **But our model, Cylinder3D, does *not* use the range image.** It works on the
> true 3D points in cylindrical voxels. The range image is a *representation* tool
> — a great way to see the data and an alternative approach worth knowing.

## What's on every point?

One LiDAR point = **4 floats**: `(x, y, z, intensity)`.

| field | unit | meaning | useful for |
|-------|------|---------|-----------|
| `x` | m | forward | where things are |
| `y` | m | left | where things are |
| `z` | m | up | **height** → road vs wall vs overhang |
| `intensity` | 0–1 | return strength | road paint, signs, plates reflect strongly |

There is **no color** on a LiDAR point — color only exists in the separate camera
photo. That's why the camera overlay in Part 3 is a *fusion* of the two sensors.

---

# Part 2 — The dataset on disk

## What is SemanticKITTI, and what did we actually download?

**KITTI** is a well-known autonomous-driving dataset recorded in Karlsruhe,
Germany. **SemanticKITTI** took KITTI's LiDAR "odometry" drives (22 sequences,
`00`–`21`) and added a **semantic label for every single point** — that hand
labelling is what makes it usable for segmentation.

This project uses **sequence 00 only** — one ~7-minute suburban drive:

| Property | Value |
|----------|------:|
| Frames (scans) | **4541** |
| Points per frame | ~124,668 (frame 0) |
| Velodyne data | 8.8 GB |
| Labels | part of a 179 MB zip |

**Why only seq 00:** the full velodyne dataset is ~80 GB against a ~10 GB
bandwidth budget (see `PROGRESS.md` for the ZIP-range trick). One sequence is
plenty to learn the full pipeline end-to-end.

## What's in each file for one frame?

For frame `000000`:

```
sequences/00/
├── velodyne/000000.bin    ← the point cloud   (geometry + intensity)
├── labels/000000.label    ← per-point class   (the answer key)
├── image_2/000000.png     ← left color camera (NOT used for training)
├── calib.txt              ← sensor calibration (NOT used for training)
├── poses.txt, times.txt   ← ego-motion / timestamps (not used here)
```

> Training is **LiDAR-only** (`use_camera=False, use_lidar=True`). The camera image
> + `calib.txt` are only used by the ROS 2 overlay to *display* results on the
> photo (Part 3). The model never sees a pixel.

### `velodyne/000000.bin` — datatype
A **flat binary blob of `float32`**, no header — just `x,y,z,intensity` repeated:

```python
import numpy as np
pts = np.fromfile("000000.bin", dtype=np.float32).reshape(-1, 4)
# pts[:, 0:3] = XYZ in metres (sensor frame), pts[:, 3] = intensity (0..1)
```

Sanity check: `1,994,688 bytes ÷ 4 (bytes/float) ÷ 4 (floats/point) = 124,668
points`. Coordinates are **metres** in the LiDAR frame (x forward, y left, z up).

### `labels/000000.label` — datatype
A flat binary of **`uint32`, one per point**, in the *same order* as the `.bin`.
Each value packs **two** numbers:

```
 31 ............ 16 | 15 ............ 0
   instance id      |   semantic id
```

- **Lower 16 bits** = the **semantic class** (what we want).
- **Upper 16 bits** = the **instance id** (which specific car/person — *panoptic*
  info; ignored here).

```python
raw = np.fromfile("000000.label", dtype=np.uint32) & 0xFFFF   # mask off instance
```

### `calib.txt`
Plain text, used only by the overlay. `P2` = the left camera's 3×4 projection
matrix; `Tr` = the 4×4 transform LiDAR→camera. To paint points on the photo:
`uv = P2 · Tr · [x y z 1]ᵀ` (Part 3).

## How do 28 raw label ids become 19 classes?

SemanticKITTI's raw ids are sparse (`10, 11, 18, 40, 252…`, up to 259). Two
problems: they're **not contiguous** (a network wants `0,1,2,…`), and there are
**moving-object duplicates** (parked car `10`, moving car `252`). For semantic (not
panoptic) segmentation we don't care if it moves, so raw ids are **remapped** →
**0..18** (19 classes), with **id 19 = "ignore"**:

```python
LABEL_MAP = {10:0, 252:0,     # parked + moving car  -> "car"
             11:1,            # bicycle
             40:8, 60:8,      # road + lane-marking   -> "road"
             0:19, 1:19, ...} # unlabelled / outlier  -> ignore
```

```
0 car          5 person       10 sidewalk      15 trunk
1 bicycle      6 bicyclist     11 other-ground  16 terrain
2 motorcycle   7 motorcyclist  12 building      17 pole
3 truck        8 road          13 fence         18 traffic-sign
4 other-veh.   9 parking       14 vegetation    (19 = ignore)
```

> The stock mmdet3d config misspells two names (`trunck`, `terrian`); our
> `colormap.py` uses correct `trunk`, `terrain`. Ids/order are identical, so it has
> zero effect on training — names are just display strings.

The **ignore** class (19) is special: the loss **skips** those points
(`ignore_index=19`) and scoring excludes them — they're points annotators couldn't
label confidently.

## What is the `.pkl` index, and why is it needed?

MMDetection3D doesn't read `sequences/00/...` directly. It first builds an **index
file** (a `.pkl`) listing, for every frame, where its point cloud and label live.
Built once:

```bash
cd ~/Autonomy/mmdetection3d
python3 -m tools.create_data semantickitti \
    --root-path ~/Autonomy/semantickitti/dataset \
    --out-dir   ~/Autonomy/semantickitti/dataset \
    --extra-tag semantickitti
```

It's a plain Python dict — a **table of contents** of paths, not data:

```python
{"metainfo": {"DATASET": "SemanticKitti"},
 "data_list": [
   {"lidar_points": {"lidar_path": "sequences/00/velodyne/000000.bin",
                     "num_pts_feats": 4},
    "pts_semantic_mask_path": "sequences/00/labels/000000.label"},
   ...]}                      # one dict per frame
```

Light to load; the heavy `.bin`/`.label` files are read lazily during training.

## Why a custom 80/20 split, and why contiguous instead of random?

The official train pkl lists **all** training sequences = 19,130 frames, but only
seq 00 is on disk. So [`scripts/split_seq00.py`](../scripts/split_seq00.py):

1. **Filters** the big pkl to `sequences/00/` → 4541 frames.
2. **Splits 80/20 contiguously**: first **3632** → train, last **909** → val.

```
frames 000000 ............. 003631 | 003632 ......... 004540
        train  (3632, 80%)         |      val (909, 20%)
```

**Why contiguous, not shuffled?** Frame 100 and 101 are 0.1 s apart — almost the
same scene. A random shuffle would put near-identical frames in *both* train and
val, inflating val accuracy. A contiguous split puts a *later stretch of the drive*
in val, so val is genuinely less-seen geometry — the honest choice for one
sequence. (`--random` exists for i.i.d. splits across many sequences.) The script
is **verified** to reproduce the exact pkls on disk.

## What happens when a frame is loaded for training?

mmdet3d runs a per-frame **pipeline** (like a torchvision `Compose` for point
clouds). The **train** pipeline:

| Step | What it does | Why |
|------|--------------|-----|
| `LoadPointsFromFile(load_dim=4, use_dim=4)` | read `.bin` → `N×4` | get x,y,z,intensity |
| `LoadAnnotations3D(with_seg_3d=True, seg_offset=65536)` | read `.label`, low 16 bits | get per-point class |
| `PointSegClassMapping` | apply raw→0..18 remap | clean contiguous ids |
| `RandomFlip3D(0.5, 0.5)` | mirror across X and/or Y | **augmentation** |
| `GlobalRotScaleTrans(rot ±45°, scale 0.95–1.05, trans σ=0.1)` | rotate/scale/jitter | **augmentation** |
| `Pack3DDetInputs(keys=[points, pts_semantic_mask])` | bundle into model input | hand off |

The **val/test** pipeline is the same minus the two augmentation steps — you never
augment evaluation data. Flip/rotate/scale are *label-preserving* for LiDAR (a
rotated or mirrored car is still a car, every point keeps its class), so they
cheaply multiply effective data — important with one sequence. Translation jitter
(σ=0.1 m) shifts the voxel-grid boundaries so the model doesn't overfit a fixed
alignment.

## What makes Cylinder3D's voxelization special?

Before the network, points are grouped into **voxels** (3D cells). Most methods use
a **cartesian** grid (fixed cubes in x,y,z). Cylinder3D partitions by **(radius,
angle, height)**:

```
point_cloud_range = [0, -π, -4,  50, +π, 2]   # [ρ,θ,z min] [ρ,θ,z max]
grid_shape        = [480, 360, 32]            # cells along (ρ, θ, z)
```

480 radial rings × 360 angular wedges × 32 height layers. As Part 1 showed, a
spinning LiDAR's points get **sparser with distance** — cartesian cubes far away
end up nearly empty while nearby cubes are jammed. **Cylindrical cells grow with
radius**, keeping a more **even number of points per cell** from near to far — a
much better match to how the sensor samples the world. This is the core idea of the
paper. (The model that consumes these voxels:
[02_model_and_training.md](02_model_and_training.md).)

---

# Part 3 — Seeing it: 3D segmentation on the camera photo

How do you draw a 3D LiDAR segmentation onto a flat camera photo? Every image below
was generated by
[`scripts/project_to_camera.py`](../scripts/project_to_camera.py) on frame 004000.

> Regenerate: `python3 scripts/project_to_camera.py --frame 004000`
> (add `--source gt` to skip the model — no GPU needed).

## The core idea in one sentence

The LiDAR gives a **3D point + a class** for ~125k points; the camera gives a
**flat photo**. For each 3D point we ask *"if the camera saw this point, which
pixel would it hit?"* — then paint that pixel with the point's class color. That
"which pixel" question is **projection**, and `calib.txt` hands us the exact
matrices to do it.

## Two coordinate frames, two matrices

- **LiDAR (velodyne) frame** — origin at the laser, x forward, y left, z up.
- **Camera frame** — origin at the camera, looking down its own +z.

| Matrix | Shape | Meaning |
|--------|-------|---------|
| `Tr` | 4×4 | rigid transform **velodyne → camera frame** |
| `P2` | 3×4 | **camera projection**: 3D camera point → 2D pixel |

```
P2 = [718.86    0    607.19   45.38      # fx, cx
        0    718.86  185.22   -0.11      # fy, cy
        0      0       1       0.0038]    # fx=fy=718.86 px, centre (607,185)
```

## The projection equation

For one point `X = [x, y, z]` in homogeneous form `[x, y, z, 1]`:

```
[u·s, v·s, s]ᵀ  =  P2 · Tr · [x, y, z, 1]ᵀ
```

1. **`Tr · X` → camera frame** `[Xc, Yc, Zc]`, where **`Zc` is the depth**.
2. **`P2 · (camera point)` → `[u·s, v·s, s]`**, a *perspective* projection where
   `s` equals the depth, so **divide by it** for the pixel:

```python
cam = (Tr @ [x, y, z, 1]ᵀ)          # → camera frame, depth = cam.z
uvw = (P2 @ cam)                     # → [u·s, v·s, s]
u, v = uvw[0]/uvw[2], uvw[1]/uvw[2]  # → pixel, divide by depth
```

The divide-by-depth is what makes far things small and near things large — exactly
how a real lens works.

## Why only ~15 % of points show up

The LiDAR spins **360°**; the camera looks **forward through a ~82° window**. A
point is kept only if `depth > 0.5 m` (in front of the camera) **and** the pixel
lands inside the image (`0 ≤ u < 1241`, `0 ≤ v < 376`). For this frame that's
**19,181 of 125,528 points (15.3 %)**:

![LiDAR is 360°, camera sees a wedge](images/step2_lidar_bev.png)

## Building the overlay, bit by bit

| Step | Image | What it shows |
|------|-------|---------------|
| 1 — raw photo | ![raw](images/step1_camera_raw.png) | the canvas |
| 3 — project geometry | ![geom](images/step3_projected_geometry.png) | in-FOV points as white dots; the **stripes** are laser rings — proves the geometry before any color |
| 4 — color by distance | ![depth](images/step4_projected_depth.png) | near red → far blue; the smooth gradient confirms the depth maths |
| 5 — color by class | ![class](images/step5_projected_class.png) | the LiDAR segmentation, as sparse class-colored dots |

Then **densify** (`dense_overlay()`): sort points **far → near** (painter's
algorithm, so near surfaces paint over far ones), stamp each as a **3×3** square so
the sparse cloud reads as regions, and **alpha-blend (α≈0.55)** so the photo shows
through:

| Ground truth | Model prediction |
|---|---|
| ![gt](images/step6_overlay_gt.png) | ![pred](images/step7_overlay_pred.png) |

Side by side — **RAW | GROUND TRUTH | PREDICTION** — you can eyeball where the
epoch-5 model agrees (road, vegetation, building) and where it's weak (the car,
small objects):

![raw vs gt vs pred](images/compare_raw_gt_pred.png)

Class color key, and the stacked pipeline hero image:

![legend](images/legend.png)
![pipeline](images/pipeline.png)

## Recap

```
3D point [x,y,z] ──Tr──► camera frame [Xc,Yc,Zc] ──P2──► [u·s, v·s, s]
                                       depth = Zc            │ ÷ s
                         keep if depth>0.5 and inside image → pixel (u,v)
                         far→near paint, 3×3 stamp, α-blend → overlay
```

One matrix multiply per point, divide by depth, cull to the frustum, paint
nearest-last. Next: **[02_model_and_training.md](02_model_and_training.md)** — the
network that consumes these points, and every training decision.
