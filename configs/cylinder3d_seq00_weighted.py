"""Variant config: per-class-weighted cross-entropy.

This is the file to edit when you want to push a weak class (e.g. `car`) or the
rare ones. It inherits our seq-00 config and changes ONE thing: the per-class
weights in the cross-entropy loss.

How to use:
  1. (optional) Generate a data-driven vector:
         python3 scripts/compute_class_weights.py
     and paste its output over CLASS_WEIGHT below.
  2. To optimize ONE class alone, leave everything at 1.0 and raise just that
     class — e.g. car only:
         CLASS_WEIGHT = [1.0] * 20
         CLASS_WEIGHT[0] = 5.0          # car
  3. Train:
         python3 scripts/train.py --config configs/cylinder3d_seq00_weighted.py

The vector length MUST be 20 = num_classes (19 real classes + index 19 = ignore).
Index 19's value is never used (those points are ignored by the loss) but must be
present so the length matches the network's 20 output channels.

See docs/03_class_weighting.md for the full explanation.
"""

_base_ = './cylinder3d_seq00.py'

#            weight  id  class                 note
CLASS_WEIGHT = [
    3.0,   # 0   car                 boosted (weak at epoch 5: ~54%)
    5.0,   # 1   bicycle             rare
    5.0,   # 2   motorcycle          rare
    5.0,   # 3   truck               rare
    5.0,   # 4   other-vehicle/bus   rare
    5.0,   # 5   person              rare
    5.0,   # 6   bicyclist           rare
    5.0,   # 7   motorcyclist        rare
    1.0,   # 8   road                abundant
    3.0,   # 9   parking             uncommon
    1.0,   # 10  sidewalk            abundant
    4.0,   # 11  other-ground        uncommon
    1.0,   # 12  building            abundant
    2.0,   # 13  fence               common-ish
    1.0,   # 14  vegetation          abundant
    4.0,   # 15  trunk               small/sparse
    1.0,   # 16  terrain             abundant
    3.0,   # 17  pole                small/sparse
    4.0,   # 18  traffic-sign        small/sparse
    1.0,   # 19  ignore              (unused; must be present)
]

model = dict(
    decode_head=dict(
        loss_ce=dict(class_weight=CLASS_WEIGHT)),
)

# Warm-start from the baseline checkpoint (loads weights, restarts the LR
# schedule) and write to a SEPARATE folder so the baseline run is preserved.
load_from = (
    '/home/ox/Autonomy/mmdetection3d/work_dirs/'
    'cylinder3d_4xb4-3x_semantickitti/epoch_5.pth'
)
work_dir = (
    '/home/ox/Autonomy/mmdetection3d/work_dirs/cylinder3d_seq00_weighted'
)
