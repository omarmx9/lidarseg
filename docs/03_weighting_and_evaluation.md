# Measuring & improving the model — the questions

Two tightly-linked topics: **how to judge the model honestly** (mIoU, not point
accuracy), and **the one lever that lifts weak classes** (class weighting). They
belong together because you can't tune weighting without the right metric to tell
you whether it worked.

Part A — **Evaluation:** why accuracy lies, what IoU/mIoU are, how to compute them.
Part B — **Class weighting:** the length-20 vector, tuning one class, what
`CLASS_WEIGHT[0]=5.0` actually means, the data-driven helper, and the traps.

---

# Part A — Evaluation: mIoU vs point accuracy

## Why is the 88 % point accuracy misleading?

```
point accuracy = (correctly predicted points) / (all labelled points)
```

It's an **average over points**, so the classes with the most points dominate. In a
street scan, road + vegetation + building are the overwhelming majority — get those
right and you score ~90 % **even if you completely fail every small class** (car,
pole, sign, person). That's exactly epoch 5: 88.4 % overall, but `trunk` and
`traffic-sign` at ~0 %. Accuracy answers *"what fraction of points did I get
right?"* — but for perception you care about *"did I find each kind of thing,
without hallucinating it?"* That's a per-class question, and that's IoU.

## What is IoU, per class?

For **one class**, look only at that class and count three things:

- **TP** (true positive): is *c* and predicted *c* ✓
- **FP** (false positive): predicted *c* but actually something else ✗
- **FN** (false negative): is *c* but predicted as something else ✗

```
            TP                  (overlap of prediction and truth)
IoU_c = ───────────   =   ─────────────────────────────────────────────
         TP + FP + FN      (everything either side called class c)
```

The property accuracy lacks: IoU **punishes both kinds of mistake** — missing real
points (FN) *and* inventing fake ones (FP). You cannot cheat it by over-predicting.

## What is mIoU?

**mIoU** = the **plain average of IoU over the 19 classes** (ignore excluded):

```
mIoU = (IoU_car + IoU_bicycle + ... + IoU_traffic-sign) / 19
```

Because it's an *unweighted* average over **classes** (not points), a tiny class
counts exactly as much as road. Fail `traffic-sign` and mIoU takes a full 1/19 hit
— even though those points are a rounding error for accuracy.

## Worked example: why accuracy lies

A scan with **100,000 road points** and **100 car points**, model predicts
**everything = road**:

| Metric | Calculation | Result |
|--------|-------------|-------:|
| **Point accuracy** | 100,000 / 100,100 | **99.9 %** 🎉 |
| IoU road | 100,000 / (100,000 + 100 FP) | 99.9 % |
| IoU car | 0 / (0 + 100 FN) | **0 %** |
| **mIoU** | (99.9 + 0) / 2 | **50 %** 😬 |

Same predictions. Accuracy says "basically perfect"; mIoU says "you completely
missed cars." **mIoU is telling the truth** — which is why every LiDAR-seg paper and
the SemanticKITTI leaderboard rank by mIoU, never accuracy.

## Which metric should I use?

**Headline = mIoU; always read the per-class IoU table underneath.**

- **mIoU** — the single score to track across experiments, and the field standard
  (published Cylinder3D is ~67 % mIoU on the full benchmark).
- **Per-class IoU** — to diagnose. When you bump `car`'s weight (Part B), you want
  **car IoU** up *without* **building/fence IoU** falling (the false-positive
  side-effect). Only the per-class view shows that trade.
- **Point accuracy** — keep as a secondary "is anything totally broken" pulse check.

## How do I compute it here?

mmdet3d's `SegMetric` computes per-class IoU + mIoU over the whole split.
[`scripts/evaluate.py`](../scripts/evaluate.py) runs it on **all 909 val frames**:

```bash
cd ~/Autonomy/lidarseg && source env.sh
make eval                         # baseline epoch_5 on the val split
```

Output (illustrative layout — real numbers come from the run):

```
classes   car  bicycle ... vegetation pole traffic-sign  miou  acc
results  61.2     0.0  ...      88.9   34.1         0.0  41.7 88.6
```

## Is our mIoU leaderboard-comparable?

No. The official benchmark validates on **sequence 08**; we only have seq 00, so our
val is the **tail of seq 00** (frames 3632–4540) — same city, drive, and weather as
training. So our mIoU is a perfectly good **internal** metric ("is model B better
than model A?") but **not** comparable to leaderboard numbers. Adding sequence 08
later would give a benchmark-comparable val set.

---

# Part B — Class weighting: the lever for weak classes

Can you make the model "care more" about one class? **Yes** — a single vector in the
loss controls how much each class matters.

## What problem does class weighting solve?

Points per class in seq 00 are wildly lopsided:

```
road, vegetation, building, sidewalk, terrain   →  MILLIONS of points
car, fence, pole                                 →  thousands
bicycle, person, trunk, traffic-sign, motorcyclist → tens to hundreds
```

By default **cross-entropy treats every point equally**, so the giant classes
dominate the gradient — the model hits high accuracy by nailing road and vegetation
while ignoring `traffic-sign`. **Class weighting** multiplies each class's loss
contribution by `w_c`; a bigger weight on a weak class means each of its points
produces a bigger gradient:

```
loss = − w_c · log( p_c )           # w_c = 1 for all = the default (no weighting)
```

## Where does the weight live, and why length 20?

One field, in the head's CE loss — set via the ready-made variant
[`configs/cylinder3d_seq00_weighted.py`](../configs/cylinder3d_seq00_weighted.py):

```python
model = dict(decode_head=dict(loss_ce=dict(class_weight=None)))  # None = all 1.0
```

`class_weight` must be **length 20**, not 19, because the network has **20 output
channels** (19 classes + index 19 = ignore; see
[02_model_and_training.md](02_model_and_training.md)). Index 19's value is never used
(`ignore_index=19` excludes it) but **must be present** or PyTorch's `cross_entropy`
throws a shape error — keep it `1.0`.

## How do I optimize ONE class alone?

To focus on **`car`** (id 0) and change nothing else:

```python
CLASS_WEIGHT = [1.0] * 20      # everybody normal...
CLASS_WEIGHT[0] = 5.0          # ...except car: 5× the gradient
model = dict(decode_head=dict(loss_ce=dict(class_weight=CLASS_WEIGHT)))
```

`car` points now contribute 5× more to the loss; everything else stays at 1.0.
Train, then check whether `car`'s IoU rose (Part A). **pole** instead?
`CLASS_WEIGHT[17] = 5.0`. **How big?** Start gentle: `2 → 3 → 5 → 8`. Below ~2 you'll
barely notice; above ~10 the model **over**-predicts (it labels walls "car" to never
miss one → false positives explode and IoU can drop).

## What does `CLASS_WEIGHT[0] = 5.0` actually mean?

### It does NOT replace anything the model learned
`class_weight` is a **multiplier on the loss**, not a weight inside the network. The
millions of learned parameters are untouched. `5.0` for car = "a mistake on a car
point counts **5× as much**." One class just shouts louder.

### Why a bigger number is *logical*: the loss is a SUM
Training minimises the **total** loss, summed **over every point**:

```
total_loss = Σ_points  w_class(point) · ( −log p_correct(point) )
```

Classes with the **most points** contribute the **most loss**, so the optimizer
spends almost all its effort there:

| class | # points | avg per-point loss | **total** (w=1) |
|-------|---------:|-------------------:|----------------:|
| road  | 100,000  | 0.10               | **10,000** |
| car   |   1,000  | 0.50               | **500** |

Road is **20× louder** than car purely from frequency. Apply `w_car = 5.0`:

```
car  = 1,000 × 0.50 × 5  = 2,500     (was 500)
road = 100,000 × 0.10 × 1 = 10,000   (unchanged)
```

Car went from 1/20th of road's voice to ~1/4. You fight a frequency imbalance
(itself a multiplication) with a multiplication.

### Where does "5" come from?
The *principled* weight is **inverse frequency** (`w_c ∝ 1/frequency_c`). To make car
exactly as loud as road above you'd need `w_car = 100`. So why **5**, not 100? Full
inverse-frequency is usually **too aggressive** — the model gets so afraid of missing
a car it stamps "car" everywhere (recall up, **precision collapses**) and training
wobbles. A gentle **3–8× nudge** gives the weak class a real say without dominating.
Think **volume knob**, not switch: `1.0` normal, `5.0` 5× louder, `100` deafening.

> **Equivalent intuitions.** Gradient view: each car point pushes the weights 5×
> harder per step. Resampling view: `w_car = 5` ≈ *duplicating every car point 5
> times* — same effect on the sum, without touching the files.

## How do I weight ALL classes at once (data-driven)?

Don't guess — compute weights from actual label frequencies:

```bash
python3 scripts/compute_class_weights.py            # samples 400 train frames
python3 scripts/compute_class_weights.py --max-frames 0   # every frame (slower)
```

It prints two ready-to-paste vectors:

| Scheme | Formula | Character |
|--------|---------|-----------|
| **inverse-frequency** | `w_c = N_total / N_c` (clipped, normalised) | aggressive — huge weights for the tiniest classes |
| **median-frequency** | `w_c = median(freq) / freq_c` (clipped) | gentler, the usual default — **start here** |

## How do I avoid retraining from scratch (warm start)?

The weighted config **warm-starts from the existing checkpoint** and writes to a
*separate* folder so the baseline survives:

```python
load_from = '.../cylinder3d_4xb4-3x_semantickitti/epoch_5.pth'   # weights, fresh schedule
work_dir  = '.../cylinder3d_seq00_weighted'                       # don't overwrite baseline
```

`load_from` (vs `--resume`) loads the **weights** but restarts the optimizer/LR — the
right choice when you've *changed the loss*. A few epochs of fine-tuning is enough to
see whether a weight helps:

```bash
python3 scripts/train.py --config configs/cylinder3d_seq00_weighted.py
python3 scripts/evaluate.py --config configs/cylinder3d_seq00_weighted.py \
        --checkpoint .../cylinder3d_seq00_weighted/epoch_5.pth
```

## What are the traps?

- **Weighting moves attention; it does NOT add information.** If `traffic-sign` has
  6 points in a frame, no weight conjures detail that isn't there — the real fix is
  **more data** or **copy-paste augmentation**.
- **Watch false positives, not just recall.** High weight raises *recall* but usually
  hurts *precision* — judge with **mIoU / per-class IoU** (Part A), never accuracy.
- **Change one thing at a time** — weight alone before LR or augmentation.
- **Lovász already helps.** The loss is CE **+ Lovász** (already imbalance-aware), so
  modest CE weights (2–3×) are often enough.
- **`car` is a special case.** Its 54 % at epoch 5 is partly *undertraining* (only 5
  epochs), not pure imbalance. Try **more epochs first** (`make resume`), then
  weighting.

---

## TL;DR

- **Judge with mIoU**, debug with per-class IoU, keep accuracy as a pulse check;
  compute with `make eval`. Our seq-00-tail val compares your own runs, not the
  leaderboard.
- **The lever** is `model.decode_head.loss_ce.class_weight`, a **length-20** list.
  One class: `[1.0]*20` then bump its index (3–5×). All classes:
  `scripts/compute_class_weights.py`. Use the weighted config (warm-start), sweep a
  couple of values, and **judge the result by IoU**.
