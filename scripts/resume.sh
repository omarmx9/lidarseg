#!/usr/bin/env bash
# Resume training. With no arg, resumes the latest checkpoint in work_dir;
# pass a path to resume a specific one:
#   ./resume.sh
#   ./resume.sh ~/Autonomy/mmdetection3d/work_dirs/cylinder3d_4xb4-3x_semantickitti/epoch_5.pth
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/../env.sh"
exec python3 "$DIR/train.py" --resume "$@"
