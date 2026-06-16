"""Project labeled LiDAR points into the camera image and build overlays.

KITTI odometry projection: uv = P2 @ Tr @ [x_velo; 1], depth = (Tr.x)_z.
"""
import cv2
import numpy as np

from kitti_seg_sim.colormap import PALETTE_BGR, IGNORE


def project_to_image(pts, P2, Tr, width, height, min_depth=0.5):
    """Return (ui, vi, depth, mask) for points landing inside the image."""
    n = len(pts)
    cam = (Tr @ np.hstack([pts, np.ones((n, 1))]).T).T   # (N,4) velo->cam
    uvw = (P2 @ cam.T).T                                  # (N,3)
    z = uvw[:, 2]
    depth = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = uvw[:, 0] / z
        v = uvw[:, 1] / z
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    mask = ((depth > min_depth) & (z > 0) &
            (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height))
    return ui, vi, depth, mask


def make_overlay(pts, labels, img, P2, Tr, alpha=0.6):
    """Paint projected, class-colored points onto a copy of img.

    Returns (overlay_bgr, n_points_drawn). Near points overwrite far ones.
    """
    h, w = img.shape[:2]
    ui, vi, depth, mask = project_to_image(pts, P2, Tr, w, h)
    mask = mask & (labels < IGNORE)
    ui, vi, lab, d = ui[mask], vi[mask], labels[mask], depth[mask]

    order = np.argsort(-d)                 # far first (painter's algorithm)
    ui, vi, lab = ui[order], vi[order], lab[order]
    cols = PALETTE_BGR[lab]

    paint = img.copy()
    drawn = np.zeros((h, w), bool)
    for dy in (-1, 0, 1):                  # 3x3 stamp so points are visible
        for dx in (-1, 0, 1):
            uu = np.clip(ui + dx, 0, w - 1)
            vv = np.clip(vi + dy, 0, h - 1)
            paint[vv, uu] = cols
            drawn[vv, uu] = True

    out = img.copy()
    blended = (alpha * paint.astype(np.float32) +
               (1 - alpha) * img.astype(np.float32)).astype(np.uint8)
    out[drawn] = blended[drawn]
    return out, int(mask.sum())


def label_image(img, text):
    """Draw a white label with a black outline in the top-left corner."""
    cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)


def side_by_side(raw, overlay):
    """Stack [RAW / SEGMENTATION] vertically with labels + separator."""
    raw_l, ov_l = raw.copy(), overlay.copy()
    label_image(raw_l, "RAW")
    label_image(ov_l, "SEGMENTATION")
    sep = np.full((4, raw.shape[1], 3), 60, np.uint8)
    return np.vstack([raw_l, sep, ov_l])
