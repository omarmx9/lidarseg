"""SemanticKITTI 19-class color palette and raw->learning label mapping.

Matches the mmdetection3d SemanticKITTI convention (class 19 = ignore/unlabeled).
Edit the palette here to recolor every output (cloud, overlay, legend) at once.
"""
import numpy as np

CLASSES = (
    "car", "bicycle", "motorcycle", "truck", "bus", "person", "bicyclist",
    "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building",
    "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
)

# RGB, 0-255, one row per class id 0..18
PALETTE = np.array([
    [100, 150, 245], [100, 230, 245], [30, 60, 150], [80, 30, 180],
    [100, 80, 250], [155, 30, 30], [255, 40, 200], [150, 30, 90],
    [255, 0, 255], [255, 150, 255], [75, 0, 75], [175, 0, 75],
    [255, 200, 0], [255, 120, 50], [0, 175, 0], [135, 60, 0],
    [150, 240, 80], [255, 240, 150], [255, 0, 0],
], dtype=np.uint8)

PALETTE_BGR = PALETTE[:, ::-1].copy()  # OpenCV uses BGR
IGNORE = 19                            # ignore / unlabeled id

# raw SemanticKITTI label id -> 0..18 learning id (19 = ignore)
LABEL_MAP = {
    0: 19, 1: 19, 10: 0, 11: 1, 13: 4, 15: 2, 16: 4, 18: 3, 20: 4, 30: 5,
    31: 6, 32: 7, 40: 8, 44: 9, 48: 10, 49: 11, 50: 12, 51: 13, 52: 19,
    60: 8, 70: 14, 71: 15, 72: 16, 80: 17, 81: 18, 99: 19, 252: 0, 253: 6,
    254: 5, 255: 7, 256: 4, 257: 4, 258: 3, 259: 4,
}

# lookup table for fast raw -> learning id mapping
LUT = np.full(260, IGNORE, dtype=np.int64)
for _k, _v in LABEL_MAP.items():
    LUT[_k] = _v
