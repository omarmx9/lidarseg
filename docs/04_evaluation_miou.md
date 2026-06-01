# Task 3 — "Proper" evaluation: mIoU vs. point accuracy

> Your question: *"explain the proper mIoU you keep mentioning — what do you
> suggest, and especially why choose it?"*
>
> Short version: the **88.4 %** we've been quoting is **point accuracy**, and for
> segmentation that number is **flattering and misleading**. The metric the whole
> field uses instead is **mIoU** (mean Intersection-over-Union). This file
> explains both, shows with a concrete example why accuracy lies, says what I
> recommend, and gives you the one command to compute it.

---

## 1. What we've been reporting: point accuracy

```
point accuracy = (correctly predicted points) / (all labelled points)
```

Simple, but it has a fatal flaw for imbalanced data: it's an **average over
points**, so the classes with the most points dominate it. In a street scan,
road + vegetation + building are the overwhelming majority of points. Get those
right and you score ~90 % **even if you completely fail every small class**
(car, pole, sign, person). That's exactly our situation at epoch 5: 88.4 %
overall, but `trunk` and `traffic-sign` at ~0 %.

Accuracy answers *"what fraction of points did I get right?"* — but for
perception you actually care about *"did I find each kind of thing, without
hallucinating it?"* That's a per-class question, and that's IoU.

---

## 2. The proper metric: IoU, per class

For **one class**, look only at that class and count three things:

- **TP** (true positive): points that are class *c* and predicted *c* ✓
- **FP** (false positive): points predicted *c* but are actually something else ✗
- **FN** (false negative): points that are *c* but predicted as something else ✗

```
            TP                  (overlap of prediction and truth)
IoU_c = ───────────   =   ─────────────────────────────────────────────
         TP + FP + FN      (everything either side called class c)
```

It's the **overlap** between "where the model says class *c* is" and "where
class *c* truly is", divided by their **union**. IoU = 1.0 means a perfect
match; 0.0 means no overlap at all.

The key property accuracy lacks: IoU **punishes both kinds of mistake** — missing
real points (FN) *and* inventing fake ones (FP). You cannot cheat it by
over-predicting a class.

**mIoU** = the **plain average of IoU over the 19 classes** (the ignore class is
excluded):

```
mIoU = (IoU_car + IoU_bicycle + ... + IoU_traffic-sign) / 19
```

Because it's an *unweighted* average over **classes** (not points), a tiny class
counts exactly as much as road. Fail `traffic-sign` and your mIoU takes a full
1/19 hit — even though those points are a rounding error for accuracy.

---

## 3. Why accuracy lies — a worked example

Imagine a scan with **100,000 road points** and **100 car points**, and a lazy
model that predicts **everything = road**:

| Metric | Calculation | Result |
|--------|-------------|-------:|
| **Point accuracy** | 100,000 right / 100,100 total | **99.9 %** 🎉 |
| IoU road | TP 100,000 / (100,000 + 100 FP + 0 FN) | 99.9 % |
| IoU car | TP 0 / (0 + 0 + 100 FN) | **0 %** |
| **mIoU** | (99.9 + 0) / 2 | **50 %** 😬 |

Same model, same predictions. Accuracy says "99.9 %, basically perfect." mIoU
says "50 %, you completely missed cars." **mIoU is telling the truth.** This is
why every LiDAR-segmentation paper and the SemanticKITTI leaderboard rank by
mIoU, never by accuracy.

---

## 4. What I suggest — and why

**Use mIoU as the headline number, but always read the per-class IoU table
underneath it.**

- **mIoU** as the single score to track across experiments (did epoch 10 beat
  epoch 5? did class weighting help overall?). It's the field standard, so it
  also lets you compare against published Cylinder3D numbers (~67 % mIoU on the
  full benchmark) to see how far a 1-sequence, 5-epoch model is from a real one.
- **Per-class IoU** to actually diagnose. mIoU going up is nice, but the table
  tells you *which* class moved. When you bump `car`'s weight (Task 2), you want
  to see **car IoU** rise without **building/fence IoU** falling (the false-
  positive side-effect). Only the per-class view shows that trade.
- **Keep point accuracy too**, but as a *secondary* sanity number, not the
  headline. It's fine for a quick "is anything totally broken" glance.

So: **mIoU = the score you optimize and report; per-class IoU = how you debug;
accuracy = a quick pulse check.**

---

## 5. How to compute it here (one command)

mmdet3d's `SegMetric` computes per-class IoU + mIoU over a whole split. Our
[`scripts/evaluate.py`](../scripts/evaluate.py) runs it on **all 909 val
frames** (not one frame like the visualizer):

```bash
cd ~/Autonomy/lidarseg
source env.sh
make eval                         # baseline epoch_5 on the val split
# or explicitly:
python3 scripts/evaluate.py --checkpoint ~/Autonomy/mmdetection3d/work_dirs/\
cylinder3d_4xb4-3x_semantickitti/epoch_5.pth
```

You'll get a table like (illustrative layout — real numbers come from the run):

```
classes   car  bicycle ... vegetation pole traffic-sign  miou  acc
results  61.2     0.0  ...      88.9   34.1         0.0  41.7 88.6
```

Then compare a new checkpoint or the weighted variant the same way and watch
**miou** and the weak columns.

> Note: this runs the model over 909 frames at a few fps, so expect a couple of
> minutes on the GPU. I've **set the script up but not run it yet** — say the
> word and I'll run `make eval` to get your real baseline mIoU.

---

## 6. One honest caveat about *our* val split

The **official** SemanticKITTI benchmark validates on **sequence 08** and tests
on the hidden sequences 11–21. We only have seq 00, so our val split is the
**tail of seq 00** (frames 3632–4540). That means:

- Our mIoU is a perfectly good **internal** metric for "is model B better than
  model A on data it didn't train on" — use it freely to compare your own runs.
- It is **not** directly comparable to leaderboard numbers, because it's a
  different, easier, single-environment split (same city, same drive, same
  weather as training). Treat our mIoU as *relative progress*, not an absolute
  benchmark score.

When bandwidth allows, adding sequence 08 would give you a benchmark-comparable
val set — a good future step.

---

## 7. TL;DR

- **88.4 % was point accuracy** — inflated by the few huge classes.
- **IoU** per class = `TP / (TP+FP+FN)`; it punishes misses *and* hallucinations.
- **mIoU** = mean IoU over the 19 classes; every class counts equally → it
  exposes failures on small classes that accuracy hides. It's the field standard.
- **Suggestion:** report mIoU, debug with per-class IoU, keep accuracy as a
  secondary check. Compute it with `make eval`.
- Our seq-00-tail val is good for **comparing your own runs**, not for comparing
  to the public leaderboard.
