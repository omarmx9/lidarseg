"""Cylinder3D on SemanticKITTI sequence 00 — our clean training config.

Inherits the stock mmdetection3d Cylinder3D config and overrides ONLY what our
single-GPU 8 GB / seq-00-only setup needs, so training becomes one command
instead of a wall of `--cfg-options`.

Everything not set here (model architecture, losses, LR schedule, augmentation)
is inherited unchanged from the base config below. Read that file to see the
full picture; read THIS file to see what we changed and why.
"""

# Absolute path so this config works no matter which directory you launch from.
_base_ = (
    '/home/ox/Autonomy/mmdetection3d/configs/cylinder3d/'
    'cylinder3d_4xb4-3x_semantickitti.py'
)

# --- where the data + our filtered/split pkls live -------------------------
data_root = '/home/ox/Autonomy/semantickitti/dataset'

# batch_size=1 + fp32 is what fits 8 GB. Cylinder3D CANNOT use fp16/AMP:
# spconv's feats_reduce_kernel is not implemented for Half. The base config
# already uses a plain OptimWrapper (fp32), so we don't touch the optimizer.
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file='semantickitti_infos_train_seq00.pkl'),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='semantickitti_infos_val_seq00.pkl'),
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='semantickitti_infos_val_seq00.pkl'),
)

# Keep checkpoints in the existing run folder so epoch_5.pth stays resumable.
work_dir = (
    '/home/ox/Autonomy/mmdetection3d/work_dirs/'
    'cylinder3d_4xb4-3x_semantickitti'
)
