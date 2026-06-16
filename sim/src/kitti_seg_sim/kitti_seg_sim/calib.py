"""KITTI odometry calibration parsing and small math helpers."""
import numpy as np


def parse_calib(path):
    """Read calib.txt -> (P2 3x4 projection, Tr 4x4 velodyne->camera)."""
    P2 = Tr = None
    with open(path) as f:
        for line in f:
            if line.startswith("P2:"):
                P2 = np.array(line.split()[1:], float).reshape(3, 4)
            elif line.startswith("Tr:"):
                t = np.array(line.split()[1:], float).reshape(3, 4)
                Tr = np.eye(4)
                Tr[:3, :4] = t
    if P2 is None or Tr is None:
        raise ValueError(f"P2 and/or Tr not found in {path}")
    return P2, Tr


def rot_to_quat(R):
    """3x3 rotation matrix -> quaternion (x, y, z, w)."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return x, y, z, w
