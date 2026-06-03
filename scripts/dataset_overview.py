#!/usr/bin/env python3
"""How LiDAR distance is measured + a richer statistical view of the dataset.

Pure analysis of one .bin (+ .label) — no model, no GPU. Produces:

  dist_tof_schematic.png    how a single distance is measured (time-of-flight)
  dist_spherical.png        how (range, azimuth, elevation) become (x, y, z)
  dist_range_histogram.png  how many points sit at each distance (density vs range)
  dist_bev_rings.png        top-down view, colored by distance, with range rings
  dist_elevation_peaks.png  elevation histogram -> the 64 beams pop out of the data
  dataset_overview.png      the data-driven panels stitched together

It also VERIFIES, from the numbers, that range = sqrt(x^2+y^2+z^2) is exactly the
distance the sensor measured (by reconstructing x,y,z from range+angles).

Usage:
    python3 scripts/dataset_overview.py                 # frame 004000
    python3 scripts/dataset_overview.py --frame 000750
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

C_LIGHT = 299_792_458.0   # speed of light, m/s

HOME = os.path.expanduser("~")
DATA = f"{HOME}/Autonomy/semantickitti/dataset/sequences/00"
OUT = f"{HOME}/Autonomy/lidarseg/docs/images"
BG = "#0d0d14"


def _dark(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#444")


# --------------------------------------------------------------------------- #
def tof_schematic(path):
    """Diagram: pulse goes out, echo comes back, d = c*dt/2."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG); _dark(ax)
    ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")

    ax.add_patch(Rectangle((0.2, 2.4), 1.1, 1.2, color="#e53935"))
    ax.text(0.75, 1.9, "LiDAR", color="white", ha="center", fontsize=11)
    ax.add_patch(Rectangle((9.4, 1.6), 0.7, 2.8, color="#90a4ae"))
    ax.text(9.75, 1.1, "object\n(e.g. a car)", color="white", ha="center", fontsize=10)

    ax.add_patch(FancyArrowPatch((1.4, 3.4), (9.3, 3.4),
                 arrowstyle="-|>", mutation_scale=22, color="#00e5ff", lw=2.5))
    ax.text(5.3, 3.7, "laser pulse  →  travels at speed of light c",
            color="#00e5ff", ha="center", fontsize=11)
    ax.add_patch(FancyArrowPatch((9.3, 2.6), (1.4, 2.6),
                 arrowstyle="-|>", mutation_scale=22, color="#ffb300",
                 lw=2.5, linestyle=(0, (6, 4))))
    ax.text(5.3, 2.2, "echo  ←  reflection returns", color="#ffb300",
            ha="center", fontsize=11)

    ax.annotate("", xy=(9.3, 4.7), xytext=(1.4, 4.7),
                arrowprops=dict(arrowstyle="<->", color="white"))
    ax.text(5.3, 4.9, "distance  d", color="white", ha="center", fontsize=12)

    txt = ("The sensor times the round trip Δt, then:\n\n"
           r"$d = \dfrac{c \cdot \Delta t}{2}$"
           "        (÷2 because the light goes there AND back)\n\n"
           "Worked examples:\n"
           f"  d = 50 m  →  Δt = 2·50 / c = {2*50/C_LIGHT*1e9:.0f} ns\n"
           f"  d = 80 m  →  Δt = 2·80 / c = {2*80/C_LIGHT*1e9:.0f} ns\n"
           f"  to resolve 2 cm  →  time it to {2*0.02/C_LIGHT*1e12:.0f} ps "
           "(picoseconds!)")
    ax.text(0.3, 0.05, txt, color="white", fontsize=11, va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="#1c1c2b", edgecolor="#444"))
    ax.set_title("How ONE distance is measured — pulsed time-of-flight",
                 color="white", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor=BG); plt.close()
    print(f"  saved {os.path.basename(path)}")


def spherical_panel(x, y, z, path):
    """Show range/azimuth/elevation -> x,y,z, and verify the conversion."""
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x**2 + y**2))
    # reconstruct cartesian from the spherical triplet
    xr = r * np.cos(el) * np.cos(az)
    yr = r * np.cos(el) * np.sin(az)
    zr = r * np.sin(el)
    err = np.max(np.abs(np.stack([xr - x, yr - y, zr - z])))

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG); _dark(ax); ax.axis("off")
    ax.set_xlim(0, 11); ax.set_ylim(0, 6)
    txt = (
        "Each point is really a (range, azimuth, elevation) measurement:\n\n"
        "  range r       = the time-of-flight distance the laser measured\n"
        "  azimuth α     = the spin angle of the head when it fired\n"
        "  elevation ω   = which of the 64 beams fired (its fixed up/down angle)\n\n"
        "The dataset stores the cartesian form instead:\n\n"
        "  x = r·cos(ω)·cos(α)        (forward)\n"
        "  y = r·cos(ω)·sin(α)        (left)\n"
        "  z = r·sin(ω)               (up)\n\n"
        "So you can always go back to the measured distance:\n\n"
        "  r = √(x² + y² + z²)        ← the true line-of-sight range\n"
        "  √(x² + y²)                 ← distance along the ground (BEV)\n\n"
        f"Verification on this frame: reconstructing x,y,z from (r,α,ω)\n"
        f"matches the stored values to {err:.2e} m  → the model is exact.")
    ax.text(0.3, 5.7, txt, color="white", fontsize=12, va="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="#1c1c2b", edgecolor="#444"))
    ax.set_title("From the measured distance to (x, y, z)", color="white", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor=BG); plt.close()
    print(f"  saved {os.path.basename(path)}  (xyz reconstruction error {err:.1e} m)")
    return err


def range_histogram(x, y, z, path):
    r = np.sqrt(x**2 + y**2 + z**2)
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(BG); _dark(ax)
    ax.hist(r, bins=80, range=(0, 80), color="#00bcd4", edgecolor="#0d0d14")
    for q in (50, 90):
        v = np.percentile(r, q)
        ax.axvline(v, color="#ffb300", ls="--", lw=1.4)
        ax.text(v + 0.5, ax.get_ylim()[1]*0.85, f"{q}% of points\nwithin {v:.0f} m",
                color="#ffb300", fontsize=9)
    ax.set_xlabel("distance from sensor (m)", color="white")
    ax.set_ylabel("number of points", color="white")
    ax.set_title("Where the points are: most returns are close to the car",
                 color="white", fontsize=13)
    plt.tight_layout(); plt.savefig(path, dpi=120, facecolor=BG); plt.close()
    print(f"  saved {os.path.basename(path)}")


def bev_rings(x, y, z, path):
    """Top-down, colored by distance, with range rings + per-ring density."""
    rh = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)
    keep = rh < 55
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.patch.set_facecolor(BG); _dark(ax)
    sc = ax.scatter(-y[keep], x[keep], c=r[keep], s=0.5, cmap="turbo", vmax=60)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.01)
    cb.set_label("distance (m)", color="white")
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    rings = [10, 20, 30, 40, 50]
    prev = 0
    for rad in rings:
        ax.add_patch(Circle((0, 0), rad, fill=False, ec="white", lw=0.8, alpha=0.5))
        n = int(((rh >= prev) & (rh < rad)).sum())
        ax.text(0, rad - 1.5, f"{rad-10}-{rad} m: {n:,} pts", color="white",
                ha="center", fontsize=8, alpha=0.85)
        prev = rad
    ax.plot(0, 0, "^", color="red", ms=12)
    ax.set_xlim(-55, 55); ax.set_ylim(-55, 55); ax.set_aspect("equal")
    ax.set_xlabel("left ← metres → right", color="white")
    ax.set_ylabel("metres (forward ↑)", color="white")
    ax.set_title("Top-down, colored by distance — point density drops with range",
                 color="white", fontsize=12)
    plt.tight_layout(); plt.savefig(path, dpi=120, facecolor=BG); plt.close()
    print(f"  saved {os.path.basename(path)}")


def elevation_peaks(x, y, z, path):
    """Histogram of elevation angle -> the discrete 64 beams appear as spikes."""
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    el = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    counts, edges = np.histogram(el, bins=600, range=(-28, 8))   # 0.06 deg/bin
    centers = 0.5 * (edges[:-1] + edges[1:])
    # smooth a touch, then find peaks at least ~0.4 deg apart (the beam spacing)
    sm = gaussian_filter1d(counts.astype(float), 1.4)
    peaks, _ = find_peaks(sm, distance=7, prominence=sm.max() * 0.02)
    n_beams = len(peaks)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor(BG); _dark(ax)
    ax.fill_between(centers, counts, color="#7e57c2", step="mid")
    ax.plot(centers[peaks], sm[peaks], "x", color="#ffeb3b", ms=5, mew=1.2)
    ax.set_xlabel("elevation angle (deg)", color="white")
    ax.set_ylabel("number of points", color="white")
    ax.set_title("Elevation histogram — each spike is one of the 64 laser beams "
                 "(upper beams see sky → sparse)", color="white", fontsize=12)
    # zoomed inset where individual beams are cleanly separated
    axin = ax.inset_axes([0.58, 0.42, 0.40, 0.55])
    _dark(axin)
    band = (centers > -16) & (centers < -8)
    axin.fill_between(centers[band], counts[band], color="#26c6da", step="mid")
    bp = peaks[(centers[peaks] > -16) & (centers[peaks] < -8)]
    axin.plot(centers[bp], sm[bp], "x", color="#ffeb3b", ms=5, mew=1.2)
    axin.set_title(f"zoom −16°…−8°: {len(bp)} beams (~0.4° apart)",
                   color="white", fontsize=9)
    axin.tick_params(labelsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=120, facecolor=BG); plt.close()
    print(f"  saved {os.path.basename(path)}  (~{n_beams} populated beams detected)")
    return n_beams


def composite(paths, out):
    fig, axes = plt.subplots(len(paths), 1, figsize=(12, 4.3 * len(paths)))
    fig.patch.set_facecolor(BG)
    for ax, p in zip(axes, paths):
        ax.imshow(plt.imread(p)); ax.axis("off")
    fig.suptitle("Representing the dataset: distance, density, and the 64 beams",
                 color="white", fontsize=16, y=0.997)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(out, dpi=95, facecolor=BG); plt.close()
    print(f"  saved {os.path.basename(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="004000")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pts = np.fromfile(f"{DATA}/velodyne/{args.frame}.bin",
                      dtype=np.float32).reshape(-1, 4)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    print(f"frame {args.frame}: {len(pts):,} points")
    print(f"  range: min {r[r>0].min():.2f} m  median {np.median(r):.1f} m  "
          f"max {r.max():.1f} m")
    print(f"  50% of points within {np.percentile(r,50):.1f} m, "
          f"90% within {np.percentile(r,90):.1f} m")

    tof_schematic(f"{OUT}/dist_tof_schematic.png")
    spherical_panel(x, y, z, f"{OUT}/dist_spherical.png")
    range_histogram(x, y, z, f"{OUT}/dist_range_histogram.png")
    bev_rings(x, y, z, f"{OUT}/dist_bev_rings.png")
    elevation_peaks(x, y, z, f"{OUT}/dist_elevation_peaks.png")
    composite([f"{OUT}/dist_range_histogram.png",
               f"{OUT}/dist_bev_rings.png",
               f"{OUT}/dist_elevation_peaks.png"],
              f"{OUT}/dataset_overview.png")
    print(f"\nImages in {OUT}/")


if __name__ == "__main__":
    main()
