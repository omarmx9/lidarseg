# Convenience front-door for the LiDAR segmentation project.
# Inline-recipe form (target: ; cmd) so no TAB characters are needed.

.PHONY: help split weights train resume eval viz

help: ; @echo "targets: split | weights | train | resume | eval | viz"

## regenerate the seq-00 train/val pkls from the full train infos
split:   ; python3 scripts/split_seq00.py

## scan train labels and print per-class loss weights
weights: ; python3 scripts/compute_class_weights.py

## train Cylinder3D (fp32, bs=1) with configs/cylinder3d_seq00.py
train:   ; bash scripts/train.sh

## resume the latest checkpoint in work_dir
resume:  ; bash scripts/resume.sh

## per-class IoU + mIoU over the whole val split (SemanticKITTI metric)
eval:    ; bash scripts/evaluate.sh

## render GT / prediction / error PNGs + PLYs for one frame
viz:     ; python3 scripts/visualize.py --frame 004000
