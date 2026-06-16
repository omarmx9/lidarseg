"""ROS2 node: stream SemanticKITTI seq 00 and publish live segmentation.

Publishes per frame:
  /kitti/points          PointCloud2  xyz + rgb (colored by class)
  /kitti/camera/image    Image        raw left color camera
  /kitti/camera/overlay  Image        segmentation projected onto the image
  /kitti/camera/compare  Image        RAW | SEGMENTATION side by side
  /kitti/camera/info     CameraInfo   from calib P2
  static TF              velodyne -> camera (from calib Tr)
"""
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster

from kitti_seg_sim.calib import parse_calib, rot_to_quat
from kitti_seg_sim.colormap import LUT
from kitti_seg_sim.pointcloud import make_pointcloud2
from kitti_seg_sim.projection import make_overlay, side_by_side

_HOME = os.path.expanduser("~")
DEF_DATA = f"{_HOME}/Autonomy/semantickitti/dataset/sequences/00"
DEF_MM = f"{_HOME}/Autonomy/mmdetection3d"
DEF_CONFIG = f"{DEF_MM}/configs/cylinder3d/cylinder3d_4xb4-3x_semantickitti.py"
DEF_CKPT = f"{DEF_MM}/work_dirs/cylinder3d_4xb4-3x_semantickitti/epoch_5.pth"


class KittiSegPlayer(Node):
    def __init__(self):
        super().__init__("kitti_seg_player")

        self.declare_parameter("data_root", DEF_DATA)
        self.declare_parameter("color_source", "pred")   # pred | gt
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("end_frame", -1)          # -1 = last available
        self.declare_parameter("loop", True)
        self.declare_parameter("model_config", DEF_CONFIG)
        self.declare_parameter("model_checkpoint", DEF_CKPT)
        self.declare_parameter("device", "cuda:0")

        g = self.get_parameter
        self.root = g("data_root").value
        self.color_source = g("color_source").value
        self.loop = g("loop").value
        self.start = g("start_frame").value
        self.end = g("end_frame").value
        rate = g("rate_hz").value

        self.velo_dir = f"{self.root}/velodyne"
        self.label_dir = f"{self.root}/labels"
        self.img_dir = f"{self.root}/image_2"

        self.pub_pc = self.create_publisher(PointCloud2, "/kitti/points", 5)
        self.pub_img = self.create_publisher(Image, "/kitti/camera/image", 5)
        self.pub_info = self.create_publisher(CameraInfo, "/kitti/camera/info", 5)
        self.pub_overlay = self.create_publisher(Image, "/kitti/camera/overlay", 5)
        self.pub_compare = self.create_publisher(Image, "/kitti/camera/compare", 5)
        self.bridge = CvBridge()

        self.P2, self.Tr = parse_calib(f"{self.root}/calib.txt")
        self._publish_static_tf()

        self.model = None
        if self.color_source == "pred":
            from kitti_seg_sim.inference import SegModel
            self.get_logger().info("Loading Cylinder3D (live inference)...")
            self.model = SegModel(
                g("model_config").value, g("model_checkpoint").value,
                g("device").value)
            self.get_logger().info("Model ready.")

        n_velo = len([f for f in os.listdir(self.velo_dir) if f.endswith(".bin")])
        last = n_velo - 1 if self.end < 0 else self.end
        self.frames = list(range(self.start, last + 1))
        self.idx = 0
        self.get_logger().info(
            f"Playing frames {self.frames[0]}..{self.frames[-1]} "
            f"({len(self.frames)} frames) at {rate} Hz, source={self.color_source}")

        self.timer = self.create_timer(1.0 / rate, self.tick)

    # ------------------------------------------------------------------ TF
    def _publish_static_tf(self):
        # Tr maps velodyne -> camera (x_cam = R x_velo + t).
        # Pose of camera in velodyne frame = (R^T, -R^T t).
        R = self.Tr[:3, :3]
        t = self.Tr[:3, 3]
        Rc = R.T
        tc = -Rc @ t
        qx, qy, qz, qw = rot_to_quat(Rc)

        self.static_br = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "velodyne"
        tf.child_frame_id = "camera"
        tf.transform.translation.x = float(tc[0])
        tf.transform.translation.y = float(tc[1])
        tf.transform.translation.z = float(tc[2])
        tf.transform.rotation.x = float(qx)
        tf.transform.rotation.y = float(qy)
        tf.transform.rotation.z = float(qz)
        tf.transform.rotation.w = float(qw)
        self.static_br.sendTransform(tf)

    def _camera_info(self, stamp, w, h):
        info = CameraInfo()
        info.header = Header(stamp=stamp, frame_id="camera")
        info.width = w
        info.height = h
        fx, fy = self.P2[0, 0], self.P2[1, 1]
        cx, cy = self.P2[0, 2], self.P2[1, 2]
        info.k = [fx, 0., cx, 0., fy, cy, 0., 0., 1.]
        info.p = list(self.P2.flatten())
        info.distortion_model = "plumb_bob"
        info.d = [0., 0., 0., 0., 0.]
        return info

    # ---------------------------------------------------------------- loop
    def tick(self):
        if self.idx >= len(self.frames):
            if self.loop:
                self.idx = 0
            else:
                self.get_logger().info("Done.")
                self.timer.cancel()
                return

        fid = self.frames[self.idx]
        name = f"{fid:06d}"
        bin_path = f"{self.velo_dir}/{name}.bin"
        img_path = f"{self.img_dir}/{name}.png"
        if not os.path.exists(bin_path) or not os.path.exists(img_path):
            self.idx += 1   # image may not be downloaded yet; skip
            return

        stamp = self.get_clock().now().to_msg()

        pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        if self.color_source == "pred":
            labels = self.model.predict(bin_path)
        else:
            raw = np.fromfile(f"{self.label_dir}/{name}.label",
                              dtype=np.uint32) & 0xFFFF
            labels = LUT[raw]
        self.pub_pc.publish(make_pointcloud2(pts, labels, stamp))

        img = cv2.imread(img_path)  # BGR
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header = Header(stamp=stamp, frame_id="camera")
        self.pub_img.publish(img_msg)
        self.pub_info.publish(self._camera_info(stamp, img.shape[1], img.shape[0]))

        overlay, n_proj = make_overlay(pts, labels, img, self.P2, self.Tr)
        ov_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        ov_msg.header = Header(stamp=stamp, frame_id="camera")
        self.pub_overlay.publish(ov_msg)

        cmp_msg = self.bridge.cv2_to_imgmsg(side_by_side(img, overlay),
                                            encoding="bgr8")
        cmp_msg.header = Header(stamp=stamp, frame_id="camera")
        self.pub_compare.publish(cmp_msg)

        if self.idx % 20 == 0:
            self.get_logger().info(
                f"frame {name}  ({len(pts)} pts, {n_proj} projected)")
        self.idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = KittiSegPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
