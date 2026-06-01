#!/usr/bin/env bash
# Train Cylinder3D on SemanticKITTI seq 00 (fp32, batch_size=1) with our config.
# All extra args pass straight through, e.g.:
#   ./train.sh --config ../configs/cylinder3d_seq00_weighted.py
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/../env.sh"
exec python3 "$DIR/train.py" "$@"
