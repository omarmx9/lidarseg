import os
from glob import glob

from setuptools import setup

package_name = "kitti_seg_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"),
            glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="omarmx9",
    maintainer_email="omar.mx909@gmail.com",
    description="Live SemanticKITTI segmentation simulation for RViz2.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "player_node = kitti_seg_sim.player_node:main",
        ],
    },
)
