#!/usr/bin/env python3
"""Step-by-step visualization of how a 3D LiDAR segmentation is shown on a 2D photo.

Produces a series of PNGs that build up the projection, bit by bit, plus a
stitched "pipeline" figure for the README:

  step1_camera_raw.png        the left color camera photo (input)
  step2_lidar_bev.png         the 360 deg LiDAR scan from above, + camera FOV wedge
  step3_projected_geometry.png the 3D points that fall in the camera, as white dots
  step4_projected_depth.png   the same points colored by distance (near=red, far=blue)
  step5_projected_class.png   the same points colored by SEGMENTATION class
  step6_overlay_gt.png        dense ground-truth segmentation blended onto the photo
  step7_overlay_pred.png      dense MODEL-PREDICTED segmentation blended onto the photo
  compare_raw_gt_pred.png     RAW | GROUND TRUTH | PREDICTION, side by side
  legend.png                  the class color key
  pipeline.png                all the key steps stacked, with titles (README hero)

Math (KITTI odometry):  uv = P2 @ Tr @ [x y z 1]^T ,  depth = (Tr @ x)_z
  Tr : 4x4 velodyne -> camera frame    (from calib.txt)
  P2 : 3x4 camera projection (intrinsics + stereo baseline term)

Usage:
    python3 scripts/project_to_camera.py                 # frame 004000, GT + prediction
    python3 scripts/project_to_camera.py --frame 000750
    python3 scripts/project_to_camera.py --source gt     # skip the model (no GPU needed)
"""
import argparse
import os

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Wedge

# --------------------------------------------------------------------------- #
# class names + palette + raw->learning label map (kept in sync with colormap)
# --------------------------------------------------------------------------- #
CLASSES = (
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
)
PALETTE = np.array([
    [100, 150, 245], [100, 230, 245], [30, 60, 150], [80, 30, 180],
    [100, 80, 250], [155, 30, 30], [255, 40, 200], [150, 30, 90],
    [255, 0, 255], [255, 150, 255], [75, 0, 75], [175, 0, 75],
    [255, 200, 0], [255, 120, 50], [0, 175, 0], [135, 60, 0],
    [150, 240, 80], [255, 240, 150], [255, 0, 0],
], dtype=np.uint8)
IGNORE = 19
LABEL_MAP = {
    0: 19, 1: 19, 10: 0, 11: 1, 13: 4, 15: 2, 16: 4, 18: 3, 20: 4, 30: 5,
    31: 6, 32: 7, 40: 8, 44: 9, 48: 10, 49: 11, 50: 12, 51: 13, 52: 19,
    60: 8, 70: 14, 71: 15, 72: 16, 80: 17, 81: 18, 99: 19, 252: 0, 253: 6,
    254: 5, 255: 7, 256: 4, 257: 4, 258: 3, 259: 4,
}
LUT = np.full(260, IGNORE, dtype=np.int64)
for _k, _v in LABEL_MAP.items():
    LUT[_k] = _v

HOME = os.path.expanduser("~")
DATA = f"{HOME}/Autonomy/semantickitti/dataset/sequences/00"
OUT = f"{HOME}/Autonomy/lidarseg/docs/images"
CONFIG = f"{HOME}/Autonomy/lidarseg/configs/cylinder3d_seq00.py"
CKPT = (f"{HOME}/Autonomy/mmdetection3d/work_dirs/"
        f"cylinder3d_4xb4-3x_semantickitti/epoch_5.pth")


# --------------------------------------------------------------------------- #
# calibration + projection
# --------------------------------------------------------------------------- #
def parse_calib(path):
    P2 = Tr = None
    with open(path) as f:
        for line in f:
            if line.startswith("P2:"):
                P2 = np.array(line.split()[1:], float).reshape(3, 4)
            elif line.startswith("Tr:"):
                t = np.array(line.split()[1:], float).reshape(3, 4)
                Tr = np.eye(4)
                Tr[:3, :4] = t
    return P2, Tr


def project(pts, P2, Tr, w, h, min_depth=0.5):
    """3D velodyne points -> (u, v, depth, mask of points inside the image)."""
    n = len(pts)
    cam = (Tr @ np.hstack([pts, np.ones((n, 1))]).T).T      # velo -> camera
    uvw = (P2 @ cam.T).T                                    # camera -> pixels
    z = uvw[:, 2]
    depth = cam[:, 2]                                       # forward distance
    with np.errstate(divide="ignore", invalid="ignore"):
        u = uvw[:, 0] / z
        v = uvw[:, 1] / z
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    mask = ((depth > min_depth) & (z > 0) &
            (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h))
    return ui, vi, depth, mask


def dense_overlay(img_bgr, ui, vi, labels, depth, alpha=0.55):
    """Paint class-colored 3x3 stamps, near points on top, alpha-blended."""
    h, w = img_bgr.shape[:2]
    order = np.argsort(-depth)                              # far first
    ui, vi, lab = ui[order], vi[order], labels[order]
    cols = PALETTE[:, ::-1][lab]                            # RGB->BGR for cv2
    paint = img_bgr.copy()
    drawn = np.zeros((h, w), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            uu = np.clip(ui + dx, 0, w - 1)
            vv = np.clip(vi + dy, 0, h - 1)
            paint[vv, uu] = cols
            drawn[vv, uu] = True
    out = img_bgr.copy()
    blend = (alpha * paint.astype(np.float32) +
             (1 - alpha) * img_bgr.astype(np.float32)).astype(np.uint8)
    out[drawn] = blend[drawn]
    return out


# --------------------------------------------------------------------------- #
# small plotting helpers (matplotlib, dark theme for README)
# --------------------------------------------------------------------------- #
def _new_ax_for_image(img_rgb, title):
    h, w = img_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 110, h / 110))
    fig.patch.set_facecolor("#0d0d14")
    ax.imshow(img_rgb)
    ax.set_title(title, color="white", fontsize=13, pad=6)
    ax.axis("off")
    return fig, ax


def save_image_panel(img_rgb, title, path):
    fig, _ = _new_ax_for_image(img_rgb, title)
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#0d0d14", bbox_inches="tight")
    plt.close()
    print(f"  saved {os.path.basename(path)}")


def save_scatter_panel(img_rgb, u, v, c, title, path, cmap=None, s=3):
    fig, ax = _new_ax_for_image(img_rgb, title)
    sc = ax.scatter(u, v, c=c, s=s, cmap=cmap, marker="s",
                    linewidths=0, alpha=0.9)
    if cmap is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("distance from sensor (m)", color="white", fontsize=9)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#0d0d14", bbox_inches="tight")
    plt.close()
    print(f"  saved {os.path.basename(path)}")


def save_bev(pts, gt, P2, Tr, w, h, path):
    """Top-down view of the whole 360 deg scan, + the camera's field of view."""
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0d0d14")
    ax.set_facecolor("#0d0d14")
    rng = (np.abs(pts[:, 0]) < 50) & (np.abs(pts[:, 1]) < 50)
    x, y, g = pts[rng, 0], pts[rng, 1], gt[rng]
    valid = g < IGNORE
    cols = np.full((len(x), 3), 0.25)
    cols[valid] = PALETTE[g[valid]] / 255.0
    # plot x forward (up), y left -> use (-y, x) so forward points up
    ax.scatter(-y[~valid], x[~valid], s=0.5, c="#404040")
    ax.scatter(-y[valid], x[valid], s=0.7, c=cols[valid])
    # camera FOV wedge (camera looks along +x; HFOV from P2 focal length)
    fx = P2[0, 0]
    hfov = np.degrees(2 * np.arctan((w / 2) / fx))
    ax.add_patch(Wedge((0, 0), 50, 90 - hfov / 2, 90 + hfov / 2,
                       color="white", alpha=0.10))
    ax.plot(0, 0, "^", color="red", ms=12)                 # the sensor
    ax.text(2, 2, "LiDAR", color="red", fontsize=11)
    ax.text(0, 47, f"camera FOV  ~{hfov:.0f}°", color="white",
            ha="center", fontsize=11)
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.set_title("Step 2 — the LiDAR sees 360°; the camera sees only the wedge",
                 color="white", fontsize=13)
    ax.set_xlabel("left  ←  metres  →  right", color="white")
    ax.set_ylabel("metres (forward ↑)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#444")
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#0d0d14")
    plt.close()
    print(f"  saved {os.path.basename(path)}")


def save_legend(present, path):
    fig = plt.figure(figsize=(10, 1.6))
    fig.patch.set_facecolor("#0d0d14")
    handles = [Patch(facecolor=PALETTE[c] / 255.0, label=CLASSES[c])
               for c in present]
    fig.legend(handles=handles, loc="center", ncol=min(len(present), 7),
               facecolor="#0d0d14", edgecolor="white", labelcolor="white",
               fontsize=11, framealpha=1.0, title="classes in this frame",
               title_fontsize=12)
    fig.gca().axis("off")
    plt.savefig(path, dpi=120, facecolor="#0d0d14", bbox_inches="tight")
    plt.close()
    print(f"  saved {os.path.basename(path)}")


def save_pipeline(panels, present, frame, path):
    """Stack labelled RGB panels vertically with a legend -> README hero."""
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.0 * n))
    fig.patch.set_facecolor("#0d0d14")
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=14, pad=6, loc="left")
        ax.axis("off")
    handles = [Patch(facecolor=PALETTE[c] / 255.0, label=CLASSES[c])
               for c in present]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(present), 7), facecolor="#0d0d14",
               edgecolor="white", labelcolor="white", fontsize=10,
               framealpha=1.0)
    fig.suptitle(f"From 3D LiDAR segmentation to a 2D camera overlay "
                 f"— SemanticKITTI seq 00, frame {frame}",
                 color="white", fontsize=17, y=0.997)
    plt.tight_layout(rect=[0, 0.05, 1, 0.985])
    plt.savefig(path, dpi=95, facecolor="#0d0d14")
    plt.close()
    print(f"  saved {os.path.basename(path)}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="004000")
    ap.add_argument("--source", choices=["gt", "both"], default="both",
                    help="'gt' = labels only (no GPU); 'both' = also run the model")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    bin_path = f"{DATA}/velodyne/{args.frame}.bin"
    lbl_path = f"{DATA}/labels/{args.frame}.label"
    img_path = f"{DATA}/image_2/{args.frame}.png"
    for p in (bin_path, lbl_path, img_path):
        assert os.path.exists(p), f"missing {p}"

    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    raw = np.fromfile(lbl_path, dtype=np.uint32) & 0xFFFF
    gt = LUT[raw]
    img_bgr = cv2.imread(img_path)
    img_rgb = img_bgr[:, :, ::-1]
    h, w = img_bgr.shape[:2]
    print(f"frame {args.frame}: {len(pts):,} points, image {w}x{h}")

    # ---- optional model prediction --------------------------------------- #
    pred = None
    if args.source == "both":
        try:
            import torch
            _orig = torch.load
            torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
            from mmdet3d.apis import init_model, inference_segmentor
            print("loading model (one-time)...")
            model = init_model(CONFIG, CKPT, device="cuda:0")
            result, _ = inference_segmentor(model, bin_path)
            pred = result.pred_pts_seg.pts_semantic_mask.cpu().numpy().astype(np.int64)
            print("model prediction done.")
        except Exception as e:                              # graceful fallback
            print(f"[warn] model unavailable ({type(e).__name__}: {e});"
                  f" producing ground-truth panels only.")
            pred = None

    # ---- project once; reuse for every panel ----------------------------- #
    P2, Tr = parse_calib(f"{DATA}/calib.txt")
    ui, vi, depth, mask = project(pts, P2, Tr, w, h)
    in_cam = mask & (gt < IGNORE)
    u, v, d = ui[in_cam], vi[in_cam], depth[in_cam]
    gt_in = gt[in_cam]
    print(f"{in_cam.sum():,} of {len(pts):,} points land in the camera FOV "
          f"({100*in_cam.sum()/len(pts):.1f}%)")

    # ---- the bit-by-bit panels ------------------------------------------- #
    save_image_panel(img_rgb, "Step 1 — the raw left color camera photo",
                     f"{OUT}/step1_camera_raw.png")
    save_bev(pts, gt, P2, Tr, w, h, f"{OUT}/step2_lidar_bev.png")
    save_scatter_panel(img_rgb, u, v, "white",
                       "Step 3 — the 3D points that fall inside the camera (geometry only)",
                       f"{OUT}/step3_projected_geometry.png", s=2)
    save_scatter_panel(img_rgb, u, v, d,
                       "Step 4 — same points colored by distance (near=red, far=blue)",
                       f"{OUT}/step4_projected_depth.png", cmap="turbo_r", s=4)
    save_scatter_panel(img_rgb, u, v, PALETTE[gt_in] / 255.0,
                       "Step 5 — same points colored by SEGMENTATION class (ground truth)",
                       f"{OUT}/step5_projected_class.png", s=5)
    ov_gt = dense_overlay(img_bgr, u, v, gt_in, d)[:, :, ::-1]
    save_image_panel(ov_gt, "Step 6 — dense ground-truth segmentation blended onto the photo",
                     f"{OUT}/step6_overlay_gt.png")

    present = sorted(set(gt_in.tolist()))
    panels = [
        (img_rgb, "1 - raw camera"),
        (img_rgb, "2 - 3D points projected in (see scatter steps)"),
    ]

    if pred is not None:
        pred_in = pred[in_cam]
        save_scatter_panel(img_rgb, u, v, PALETTE[pred_in] / 255.0,
                           "Step 5b — points colored by the MODEL's prediction",
                           f"{OUT}/step5_projected_pred.png", s=5)
        ov_pred = dense_overlay(img_bgr, u, v, pred_in, d)[:, :, ::-1]
        save_image_panel(ov_pred,
                         "Step 7 — dense MODEL-PREDICTED segmentation blended onto the photo",
                         f"{OUT}/step7_overlay_pred.png")
        # RAW | GT | PRED
        sep = np.full((h, 6, 3), 60, np.uint8)
        trio = np.hstack([img_bgr, sep, ov_gt[:, :, ::-1], sep,
                          ov_pred[:, :, ::-1]])[:, :, ::-1]
        save_image_panel(trio, "RAW          |          GROUND TRUTH          |          MODEL PREDICTION",
                         f"{OUT}/compare_raw_gt_pred.png")
        present = sorted(set(gt_in.tolist()) | set(pred_in.tolist()))
        hero = [(img_rgb, "Step 1 — raw camera photo"),
                (ov_gt, "Step 6 — ground-truth segmentation overlay"),
                (ov_pred, "Step 7 — model-predicted segmentation overlay")]
    else:
        hero = [(img_rgb, "Step 1 — raw camera photo"),
                (ov_gt, "Step 6 — ground-truth segmentation overlay")]

    save_legend(present, f"{OUT}/legend.png")
    save_pipeline(hero, present, args.frame, f"{OUT}/pipeline.png")
    print(f"\nAll images in {OUT}/")


if __name__ == "__main__":
    main()
