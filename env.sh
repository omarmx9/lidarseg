# Source this before training / evaluating:  source env.sh
# Sets the CUDA toolkit (13.0, matches PyTorch's cu130 build) and project paths.
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export MMDET3D="$HOME/Autonomy/mmdetection3d"
export KITTI_ROOT="$HOME/Autonomy/semantickitti/dataset"
export LIDARSEG="$HOME/Autonomy/lidarseg"
