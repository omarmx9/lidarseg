#!/usr/bin/env python3
"""
Visualize Cylinder3D semantic segmentation on SemanticKITTI sequence 00.

Runs the trained model on one validation frame and renders three views:
  1. Ground-truth labels
  2. Model prediction
  3. Error map (green = correct, red = wrong)

Outputs PNGs (viewable in the IDE) and .ply files (interactive in any viewer).

Usage:
    python3 visualize_seg.py                 # default frame
    python3 visualize_seg.py --frame 004000  # specific frame
    python3 visualize_seg.py --interactive   # pop a live 3D window
"""
import argparse
import os
import numpy as np
import open3d as o3d
import torch

# PyTorch 2.6+ defaults torch.load to weights_only=True, which rejects the
# mmengine ConfigDict stored in our own checkpoints. This file is ours and
# trusted, so force the legacy full-unpickle behaviour.
_orig_torch_load = torch.load
def _full_load(*a, **k):
    k["weights_only"] = False
    return _orig_torch_load(*a, **k)
torch.load = _full_load

from mmdet3d.apis import init_model, inference_segmentor

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
MM = os.path.expanduser("~/Autonomy/mmdetection3d")
DATA = os.path.expanduser("~/Autonomy/semantickitti/dataset/sequences/00")
CONFIG = os.path.expanduser("~/Autonomy/lidarseg/configs/cylinder3d_seq00.py")
CKPT = f"{MM}/work_dirs/cylinder3d_4xb4-3x_semantickitti/epoch_5.pth"
OUT = os.path.expanduser("~/Autonomy/viz")

# ----------------------------------------------------------------------------
# SemanticKITTI 19-class palette + names (from mmdet3d METAINFO)
# ----------------------------------------------------------------------------
CLASSES = ('car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
           'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
           'other-ground', 'building', 'fence', 'vegetation', 'trunk',
           'terrain', 'pole', 'traffic-sign')
PALETTE = np.array([
    [100, 150, 245], [100, 230, 245], [30, 60, 150], [80, 30, 180],
    [100, 80, 250], [155, 30, 30], [255, 40, 200], [150, 30, 90],
    [255, 0, 255], [255, 150, 255], [75, 0, 75], [175, 0, 75],
    [255, 200, 0], [255, 120, 50], [0, 175, 0], [135, 60, 0],
    [150, 240, 80], [255, 240, 150], [255, 0, 0],
], dtype=np.float32) / 255.0
IGNORE = 19  # unlabeled / ignore index

# Raw SemanticKITTI id -> 19-class learning id (from dataset config labels_map)
LABEL_MAP = {
    0: 19, 1: 19, 10: 0, 11: 1, 13: 4, 15: 2, 16: 4, 18: 3, 20: 4, 30: 5,
    31: 6, 32: 7, 40: 8, 44: 9, 48: 10, 49: 11, 50: 12, 51: 13, 52: 19,
    60: 8, 70: 14, 71: 15, 72: 16, 80: 17, 81: 18, 99: 19, 252: 0, 253: 6,
    254: 5, 255: 7, 256: 4, 257: 4, 258: 3, 259: 4,
}


def map_raw_labels(raw):
    """Map raw SemanticKITTI label ids to 0..18 (19 = ignore)."""
    out = np.full(raw.shape, IGNORE, dtype=np.int64)
    for k, v in LABEL_MAP.items():
        out[raw == k] = v
    return out


def colorize(labels):
    """Map class ids to RGB. Ignore index -> dark grey."""
    colors = np.full((len(labels), 3), 0.2, dtype=np.float32)
    valid = labels < IGNORE
    colors[valid] = PALETTE[labels[valid]]
    return colors


def render_png(pts, colors, path, title):
    """Offscreen-render a colored point cloud to a PNG (top-down-ish view)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1600, height=1000)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.background_color = np.array([0.05, 0.05, 0.08])

    # bird's-eye-ish view: look down with slight tilt
    ctr = vis.get_view_control()
    ctr.set_front([0.0, -0.45, 0.9])
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_zoom(0.42)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path, do_render=True)
    vis.destroy_window()
    print(f"  saved {path}  ({title})")


def make_comparison(frame, acc, present, out_dir):
    """Stitch the 3 renders into one labeled figure with a class legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    imgs = [
        (f"{out_dir}/frame{frame}_1_groundtruth.png", "Ground Truth"),
        (f"{out_dir}/frame{frame}_2_prediction.png", "Your Model (epoch 5)"),
        (f"{out_dir}/frame{frame}_3_errormap.png", "Error  (green=correct, red=wrong)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(16, 22))
    fig.patch.set_facecolor("#0d0d14")
    for ax, (path, title) in zip(axes, imgs):
        ax.imshow(plt.imread(path))
        ax.set_title(title, color="white", fontsize=18, pad=8)
        ax.axis("off")

    # legend only for classes present in this frame
    handles = [Patch(facecolor=PALETTE[c], label=CLASSES[c]) for c in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(present), 6),
               facecolor="#0d0d14", edgecolor="white", labelcolor="white",
               fontsize=12, framealpha=1.0)
    fig.suptitle(f"SemanticKITTI seq 00 — frame {frame}   |   point accuracy {acc:.1f}%",
                 color="white", fontsize=22, y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out = f"{out_dir}/frame{frame}_COMPARISON.png"
    plt.savefig(out, dpi=90, facecolor="#0d0d14")
    plt.close()
    print(f"  saved {out}  (stitched comparison + legend)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="004000", help="frame id, e.g. 004000")
    ap.add_argument("--interactive", action="store_true",
                    help="open a live 3D window instead of saving PNGs")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    bin_path = f"{DATA}/velodyne/{args.frame}.bin"
    label_path = f"{DATA}/labels/{args.frame}.label"
    assert os.path.exists(bin_path), f"missing {bin_path}"

    # --- load model + run inference ---
    print(f"Loading model from {os.path.basename(CKPT)} ...")
    model = init_model(CONFIG, CKPT, device="cuda:0")
    print(f"Running inference on frame {args.frame} ...")
    result, _ = inference_segmentor(model, bin_path)
    pred = result.pred_pts_seg.pts_semantic_mask.cpu().numpy().astype(np.int64)

    # --- load points + ground truth ---
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    raw = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
    gt = map_raw_labels(raw)

    # --- accuracy on labeled points ---
    valid = gt < IGNORE
    acc = (pred[valid] == gt[valid]).mean() * 100
    print(f"\nPoint accuracy on labeled points: {acc:.1f}%")

    # per-class presence
    present = sorted(set(gt[valid].tolist()))
    print("Classes present in this frame:")
    for c in present:
        name = CLASSES[c]
        n = int((gt == c).sum())
        ca = (pred[gt == c] == c).mean() * 100
        print(f"  {name:14s} {n:6d} pts   {ca:5.1f}% correct")

    if args.interactive:
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(pts)
        gt_pcd.colors = o3d.utility.Vector3dVector(colorize(gt))
        print("\nOpening interactive window (close it to exit)...")
        o3d.visualization.draw_geometries([gt_pcd], window_name=f"GT - frame {args.frame}")
        return

    # --- render all three views ---
    print()
    render_png(pts, colorize(gt), f"{OUT}/frame{args.frame}_1_groundtruth.png", "ground truth")
    render_png(pts, colorize(pred), f"{OUT}/frame{args.frame}_2_prediction.png", "prediction")

    err = np.tile(np.array([[0.0, 0.7, 0.0]]), (len(pts), 1))  # green = correct
    wrong = valid & (pred != gt)
    err[wrong] = [0.9, 0.0, 0.0]  # red = wrong
    err[~valid] = [0.15, 0.15, 0.15]  # grey = unlabeled
    render_png(pts, err, f"{OUT}/frame{args.frame}_3_errormap.png", "error map")

    # --- save ply for interactive viewing ---
    for name, lab in [("groundtruth", gt), ("prediction", pred)]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colorize(lab))
        o3d.io.write_point_cloud(f"{OUT}/frame{args.frame}_{name}.ply", pcd)

    make_comparison(args.frame, acc, present, OUT)

    print(f"\nDone. PNGs and PLYs in {OUT}/")


if __name__ == "__main__":
    main()
