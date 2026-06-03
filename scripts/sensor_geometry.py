#!/usr/bin/env python3
"""Visualize the Velodyne HDL-64E sensor geometry behind one SemanticKITTI frame.

Pure geometry from the .bin (+ .label for class colors) — no model, no GPU.
Shows WHY the data looks the way it does:

  sensor_side_fan.png        a side ("from the side of the car") slice showing the
                             64-beam vertical fan and the vertical field of view
  sensor_range_image_depth.png  the whole 360deg scan UNWRAPPED into a 2D image
                             (rows = beams/elevation, cols = azimuth), colored by range
  sensor_range_image_class.png  same unwrapping, colored by SEGMENTATION class

It also prints the real numbers (vertical FOV, ranges, beams) measured from the
points themselves.

Usage:
    python3 scripts/sensor_geometry.py                 # frame 004000
    python3 scripts/sensor_geometry.py --frame 000750
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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


def side_fan(x, y, z, el_deg, path):
    """Front slice (|azimuth|<3deg): range vs height, colored by beam elevation."""
    az = np.degrees(np.arctan2(y, x))
    sl = (np.abs(az) < 3) & (np.sqrt(x**2 + y**2) < 45)
    r = np.sqrt(x[sl]**2 + y[sl]**2)
    zz, ee = z[sl], el_deg[sl]
    el_max, el_min = el_deg.max(), el_deg.min()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d0d14"); ax.set_facecolor("#0d0d14")
    sc = ax.scatter(r, zz, c=ee, s=8, cmap="turbo")
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("beam elevation angle (deg)", color="white")
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    # the top and bottom beam directions, drawn from the sensor
    rr = np.array([0, 45])
    ax.plot(rr, rr * np.tan(np.radians(el_max)), "--", color="#00e5ff", lw=1.5,
            label=f"top beam  {el_max:+.1f}°")
    ax.plot(rr, rr * np.tan(np.radians(el_min)), "--", color="#ff4081", lw=1.5,
            label=f"bottom beam {el_min:+.1f}°")
    ax.plot(0, 0, "^", color="red", ms=14)
    ax.text(0.5, 0.5, "LiDAR", color="red")
    ax.set_xlim(0, 45); ax.set_ylim(-5, 8)
    ax.set_xlabel("horizontal distance in front (m)", color="white")
    ax.set_ylabel("height z (m)", color="white")
    ax.set_title(f"Side view — the 64 beams form a vertical fan "
                 f"(vertical FOV ≈ {el_max - el_min:.1f}°)",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.legend(facecolor="#0d0d14", edgecolor="#444", labelcolor="white", loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#0d0d14")
    plt.close()
    print(f"  saved {os.path.basename(path)}")


def range_images(x, y, z, gt, H, W, depth_path, class_path):
    """Spherical projection: unwrap the 360deg scan into a 2D (beam x azimuth) image."""
    r = np.sqrt(x**2 + y**2)
    d = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)                     # -pi..pi
    el = np.arctan2(z, r)                     # elevation
    el_min, el_max = el.min(), el.max()

    col = ((az + np.pi) / (2 * np.pi) * (W - 1)).astype(np.int32)
    row = ((el_max - el) / (el_max - el_min + 1e-9) * (H - 1)).astype(np.int32)
    col = np.clip(col, 0, W - 1); row = np.clip(row, 0, H - 1)

    depth_img = np.full((H, W), np.nan)
    class_img = np.full((H, W), -1, dtype=np.int64)
    order = np.argsort(-d)                    # far first, nearest wins the pixel
    for rr, cc, dd, gg in zip(row[order], col[order], d[order], gt[order]):
        depth_img[rr, cc] = dd
        class_img[rr, cc] = gg

    # depth range image
    fig, ax = plt.subplots(figsize=(14, 3.2))
    fig.patch.set_facecolor("#0d0d14")
    im = ax.imshow(depth_img, cmap="turbo", aspect="auto")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("range (m)", color="white")
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax.set_title("Range image — the 360° scan unwrapped: rows = the 64 beams "
                 "(top→bottom), columns = spin angle", color="white", fontsize=12)
    ax.set_xlabel("azimuth  (rear → right → front → left → rear)", color="white")
    ax.set_ylabel("beam / elevation", color="white")
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(depth_path, dpi=120, facecolor="#0d0d14")
    plt.close()
    print(f"  saved {os.path.basename(depth_path)}")

    # class range image
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    valid = class_img >= 0
    safe = np.where((class_img >= 0) & (class_img < IGNORE), class_img, 0)
    rgb[:] = PALETTE[safe]
    rgb[~valid] = (15, 15, 20)
    rgb[(class_img == IGNORE)] = (40, 40, 40)
    fig, ax = plt.subplots(figsize=(14, 3.2))
    fig.patch.set_facecolor("#0d0d14")
    ax.imshow(rgb, aspect="auto")
    ax.set_title("Same range image, colored by SEGMENTATION class "
                 "(this is LiDAR shown as a 2D image)", color="white", fontsize=12)
    ax.set_xlabel("azimuth (full 360°)", color="white")
    ax.set_ylabel("beam / elevation", color="white")
    ax.tick_params(colors="white")
    present = sorted(set(gt[gt < IGNORE].tolist()))
    handles = [Patch(facecolor=PALETTE[c] / 255.0, label=CLASSES[c]) for c in present]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=min(len(present), 8), facecolor="#0d0d14", edgecolor="#444",
              labelcolor="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(class_path, dpi=120, facecolor="#0d0d14", bbox_inches="tight")
    plt.close()
    print(f"  saved {os.path.basename(class_path)}")
    return el_min, el_max, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="004000")
    ap.add_argument("--rows", type=int, default=64)     # HDL-64E has 64 beams
    ap.add_argument("--cols", type=int, default=1024)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pts = np.fromfile(f"{DATA}/velodyne/{args.frame}.bin",
                      dtype=np.float32).reshape(-1, 4)
    x, y, z, inten = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    raw = np.fromfile(f"{DATA}/labels/{args.frame}.label", dtype=np.uint32) & 0xFFFF
    gt = LUT[raw]

    r = np.sqrt(x**2 + y**2)
    d = np.sqrt(x**2 + y**2 + z**2)
    el = np.degrees(np.arctan2(z, r))
    az = np.degrees(np.arctan2(y, x))

    print(f"frame {args.frame}: {len(pts):,} points")
    print(f"  vertical FOV   : {el.min():+.1f}° (down)  to  {el.max():+.1f}° (up)"
          f"   = {el.max()-el.min():.1f}° tall")
    print(f"  horizontal     : {az.min():.0f}° .. {az.max():.0f}°  (full 360°)")
    print(f"  range          : {r[r>0].min():.1f} m near  →  {d.max():.0f} m far")
    print(f"  intensity      : {inten.min():.2f} .. {inten.max():.2f}")
    print(f"  ~points/beam   : {len(pts)//args.rows:,}  (over {args.rows} beams)")

    side_fan(x, y, z, el, f"{OUT}/sensor_side_fan.png")
    range_images(x, y, z, gt, args.rows, args.cols,
                 f"{OUT}/sensor_range_image_depth.png",
                 f"{OUT}/sensor_range_image_class.png")
    print(f"\nImages in {OUT}/")


if __name__ == "__main__":
    main()