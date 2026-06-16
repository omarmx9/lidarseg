"""Build a colored sensor_msgs/PointCloud2 from points + class labels."""
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from kitti_seg_sim.colormap import PALETTE, IGNORE

_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
]


def make_pointcloud2(pts, labels, stamp, frame_id="velodyne"):
    """pts (N,3) float, labels (N,) int -> PointCloud2 with packed rgb field.

    RViz colors this by setting the PointCloud2 color transformer to RGB8.
    """
    rgb = np.full((len(labels), 3), 50, dtype=np.uint32)
    valid = labels < IGNORE
    rgb[valid] = PALETTE[labels[valid]]
    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]

    cloud = np.zeros(len(pts), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "f4")])
    cloud["x"] = pts[:, 0]
    cloud["y"] = pts[:, 1]
    cloud["z"] = pts[:, 2]
    cloud["rgb"] = packed.astype(np.uint32).view(np.float32)

    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height = 1
    msg.width = len(pts)
    msg.fields = _FIELDS
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * len(pts)
    msg.is_dense = True
    msg.data = cloud.tobytes()
    return msg
