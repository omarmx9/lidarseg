# Task 2 — Class weighting (and yes, you can tune ONE class alone)

> Your question: *"explain class weighting, and can I try optimizing a specific
> class alone — like the class weights in the model alone?"*
>
> **Short answer: yes.** There is a single vector in the loss that controls how
> much each class matters. You can raise just one entry to make the model "care
> more" about that one class, leaving everything else untouched. This file
> explains what it is, exactly where it lives, how to set it (including the
> one-class recipe), and the traps to avoid.

---

## 1. The problem class weighting solves: imbalance

Count the points per class in seq 00 and you get something wildly lopsided:

```
road, vegetation, building, sidewalk, terrain   →  MILLIONS of points
car, fence, pole                                 →  thousands
bicycle, person, trunk, traffic-sign, motorcyclist → tens to hundreds
```

By default, **cross-entropy treats every point equally**. So the gradient is
dominated by the giant classes — the model can reach high *point accuracy* just
by nailing road and vegetation, while basically ignoring `traffic-sign`. That's
exactly what we saw at epoch 5 (road 98 %, car 54 %, trunk/traffic-sign ~0 %).

**Class weighting** rebalances this: you multiply each class's contribution to
the loss by a weight `w_c`. Give a rare/weak class a bigger weight and each of
its points produces a bigger gradient, so the model is *forced* to pay attention
to it.

Formally, weighted cross-entropy for a point with true class `c`:

```
loss = − w_c · log( p_c )
```

`w_c = 1` for all classes = the default (no weighting). Raise `w_car` and every
car point pushes harder.

---

## 2. Exactly where it lives

One field, in the segmentation head's CE loss:

```python
model = dict(
    decode_head=dict(
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            class_weight=None,        # ← THIS. None = all weights = 1.0
        )))
```

In our project you don't edit the stock config — you use the ready-made variant
[`configs/cylinder3d_seq00_weighted.py`](../configs/cylinder3d_seq00_weighted.py),
which sets exactly this and nothing else.

### The length-20 rule (important, easy to get wrong)
`class_weight` must be a list of **length 20**, not 19 — because the network has
**20 output channels** (19 real classes + index 19 = ignore; see
[02_model_and_training.md](02_model_and_training.md) §2). The order matches the
class ids:

```
index:  0 car  1 bicycle  2 motorcycle ... 18 traffic-sign  | 19 ignore
```

The value at index 19 is **never used** (those points are excluded by
`ignore_index=19`), but it **must be present** so the list length equals the
number of channels — otherwise PyTorch's `cross_entropy` throws a shape error.
We keep it `1.0` by convention.

---

## 3. ✅ Optimizing ONE class alone — the recipe

This is precisely what you asked. To make the model focus on, say, **`car`**
(id 0) and change nothing else:

```python
# in configs/cylinder3d_seq00_weighted.py
CLASS_WEIGHT = [1.0] * 20      # everybody normal...
CLASS_WEIGHT[0] = 5.0          # ...except car: 5× the gradient

model = dict(decode_head=dict(loss_ce=dict(class_weight=CLASS_WEIGHT)))
```

That's it. `car` points now contribute 5× more to the loss; every other class
stays at its default 1.0. Train, then check whether `car`'s IoU went up (use
`scripts/evaluate.py` — Task 3).

Want to push **pole** (17) instead? `CLASS_WEIGHT[17] = 5.0`. Two classes at
once? Set both. The mapping of id→class is printed at the top of the weighted
config and in [01_dataset_preparation.md](01_dataset_preparation.md) §4.

**How big a weight?** Start gentle and increase:
`2 → 3 → 5 → 8`. Below ~2 you'll barely see a change; above ~10 the model often
*over*-predicts that class (it starts labelling fences and walls as "car" to
avoid ever missing one — your **false positives** explode and IoU can actually
drop). Sweep a couple of values and compare IoU.

---

## 4. Weighting ALL classes at once (data-driven)

If instead you want to lift *all* the weak classes together, don't guess —
compute weights from the actual label frequencies with the helper:

```bash
cd ~/Autonomy/lidarseg
python3 scripts/compute_class_weights.py            # samples 400 train frames
python3 scripts/compute_class_weights.py --max-frames 0   # use every frame (slower)
```

It scans the `.label` files, counts points per class, and prints two
ready-to-paste vectors:

| Scheme | Formula | Character |
|--------|---------|-----------|
| **inverse-frequency** | `w_c = N_total / N_c` (clipped, normalised) | aggressive — huge weights for the tiniest classes |
| **median-frequency** | `w_c = median(freq) / freq_c` (clipped) | gentler, the usual segmentation default — **start here** |

Paste one block over `CLASS_WEIGHT` in the weighted config and train. Median-freq
is the safer first try; inverse-freq can destabilise training if a class is
extremely rare.

---

## 5. Warm-start so you don't restart from zero

Retraining 36 epochs to test a weight is wasteful. The weighted config
**warm-starts from the existing checkpoint** and writes to a *separate* folder so
your baseline run is preserved:

```python
load_from = '.../cylinder3d_4xb4-3x_semantickitti/epoch_5.pth'   # load weights, fresh schedule
work_dir  = '.../cylinder3d_seq00_weighted'                       # don't overwrite baseline
```

`load_from` (vs `--resume`) loads the **weights** but restarts the optimizer and
LR schedule — the right choice when you've *changed the loss*, because the old
optimizer state no longer matches. A few epochs of warm-started fine-tuning is
usually enough to see whether a weight helps.

```bash
python3 scripts/train.py --config configs/cylinder3d_seq00_weighted.py
# then compare:
python3 scripts/evaluate.py   # baseline (epoch_5)
python3 scripts/evaluate.py --config configs/cylinder3d_seq00_weighted.py \
        --checkpoint .../cylinder3d_seq00_weighted/epoch_5.pth
```

---

## 6. Traps & honest caveats

- **Weighting moves attention; it does NOT add information.** If `traffic-sign`
  has only 6 points in a frame and a few hundred in all of seq 00, no weight
  conjures detail that isn't there. For the truly tiny classes, the real fix is
  **more data** (more sequences) or **oversampling/copy-paste augmentation**, not
  a bigger weight.
- **Watch false positives, not just recall.** A high weight raises a class's
  *recall* (it stops missing them) but usually hurts *precision* (it starts
  hallucinating them). IoU captures both — that's why you must judge with
  **mIoU / per-class IoU**, not point accuracy. (Task 3.)
- **Change one thing at a time.** If you bump a weight *and* the LR *and* add
  augmentation, you won't know which helped. Tune the weight vector alone first.
- **Lovász already helps.** Remember the loss is CE **+ Lovász**, and Lovász is
  already imbalance-aware. So sometimes modest CE weights (2–3×) are enough;
  going extreme fights against an already-balanced term.
- **`car` is a special case.** Its 54 % at epoch 5 is partly *undertraining*
  (only 5 epochs), not pure imbalance — `car` isn't actually that rare. Try
  **more epochs first** (`make resume`), then weighting, so you don't over-correct
  a problem that was about to fix itself.

---

## 7. TL;DR

- The lever is `model.decode_head.loss_ce.class_weight`, a **length-20** list.
- **One class alone:** `[1.0]*20`, then set that class's index higher (3–5×).
- **All classes:** generate weights with `scripts/compute_class_weights.py`.
- Use the **weighted config** (warm-starts, separate work_dir), then **judge with
  IoU**, sweep a couple of values, change one knob at a time.

Next: **[04_evaluation_miou.md](04_evaluation_miou.md)** — why IoU/mIoU is the
metric that tells you whether the weighting actually worked.
