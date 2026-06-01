#!/usr/bin/env bash
# Compute per-class IoU + mIoU over the whole val split (SemanticKITTI metric).
#   ./evaluate.sh
#   ./evaluate.sh --checkpoint ~/Autonomy/mmdetection3d/work_dirs/.../epoch_10.pth
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/../env.sh"
exec python3 "$DIR/evaluate.py" "$@"
