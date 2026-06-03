# Task (extra) — Turning your recorded video into a README "GIF"

> You'll record a clip (e.g. the live RViz2 sim playing) and want that looping,
> auto-playing "GIF effect" in the README. Here's exactly how — with a one-command
> helper, and the trade-offs between a real GIF and an embedded MP4.

There are **two ways** to get a moving image in a README. Pick by file size:

| | **Animated GIF** | **MP4 video** |
|---|---|---|
| Plays automatically + loops | ✅ always | ✅ (GitHub) |
| Renders on GitHub, npm, VS Code preview | ✅ everywhere | ⚠️ GitHub yes, some renderers no |
| File size for ~8 s clip | **big** (5–30 MB) | **small** (0.5–3 MB) |
| Colors / quality | 256 colors, can band | full quality |
| Sound | ❌ | ✅ |
| Best for | short loops (≤10 s), guaranteed rendering | longer/HQ clips, keeping the repo light |

**Rule of thumb:** short loop you want to *just work* anywhere → **GIF**. Longer
or higher-quality, and you're OK relying on GitHub → **MP4**.

---

## Option A — make an optimized GIF (one command)

`ffmpeg` is already installed (v4.4.2). Use the helper
[`scripts/make_gif.sh`](../scripts/make_gif.sh):

```bash
cd ~/Autonomy/lidarseg/scripts

# simplest: whole clip, 12 fps, 900 px wide -> demo.gif next to the input
./make_gif.sh ~/Videos/rviz_demo.mp4

# put it straight into the images folder, tune fps + width
./make_gif.sh ~/Videos/rviz_demo.mp4 ../docs/images/demo.gif 15 1000

# trim to a 6-second highlight starting at 3 s (keeps the file small)
./make_gif.sh ~/Videos/rviz_demo.mp4 ../docs/images/demo.gif 12 900 00:00:03 6
```

**Why this is good quality:** the script runs ffmpeg **twice** — pass 1
(`palettegen`) studies *your* clip and builds an optimal 256-color palette; pass 2
(`paletteuse`) applies it with dithering. That's dramatically smaller and cleaner
than a naive one-pass GIF.

**Knobs that control size** (size ≈ fps × width² × duration):
- `FPS` — drop to **10** for talking-head/slow clips; 12–15 for smooth motion.
- `WIDTH` — **800–1000 px** is plenty for a README; 1280 if you must.
- `START`/`DURATION` — **trim ruthlessly**. A tight 6-second loop beats 30 s.

### Embed it in the README
```markdown
![Live RViz2 segmentation demo](docs/images/demo.gif)
```
(Use `images/demo.gif` if you're writing from inside `docs/`.)

### If the GIF is still too big (>~10 MB)
- Lower `FPS` and `WIDTH` first (biggest wins).
- Install a dedicated optimizer for an extra ~30–50 % off:
  ```bash
  sudo apt install gifsicle
  gifsicle -O3 --lossy=80 docs/images/demo.gif -o docs/images/demo.gif
  ```
- Or install **gifski** (best-looking GIFs) and feed it frames/MP4:
  ```bash
  sudo apt install gifski    # or: cargo install gifski
  gifski --fps 15 --width 1000 -o demo.gif ~/Videos/rviz_demo.mp4
  ```

---

## Option B — keep it an MP4 (smaller, GitHub auto-plays it)

GitHub renders video two ways:

**B1 — the easy way (recommended for GitHub):** open your repo on github.com,
edit the README (or open an issue), and **drag-and-drop the `.mp4` into the text
box**. GitHub uploads it and inserts a URL like
`https://github.com/<user>/<repo>/assets/<id>`. That link auto-plays, loops on
hover, and works in the rendered README. You don't commit the file at all.

**B2 — an HTML5 tag (works in GitHub READMEs, not all renderers):**
```html
<video src="docs/images/demo.mp4" autoplay loop muted playsinline width="900">
</video>
```
Keep it `muted` or browsers won't autoplay. Note: plain `![](demo.mp4)` does
**not** work — video needs the `<video>` tag or the drag-drop asset URL.

**Shrink an MP4 before committing** (often 2–5× smaller, no visible loss):
```bash
ffmpeg -i in.mp4 -vf "scale=1000:-2,fps=24" -c:v libx264 -crf 28 -preset slow \
       -an -movflags +faststart docs/images/demo.mp4
```
(`-an` drops audio; `-crf 28` is the quality/size dial — higher = smaller.)

---

## Recording the clip in the first place

On this Ubuntu box, easy screen recorders:
- **GNOME built-in:** `Ctrl`+`Alt`+`Shift`+`R` starts/stops a screen recording
  (saves a `.webm` to `~/Videos`). Convert with `make_gif.sh` (it eats `.webm`
  too) or `ffmpeg`.
- **ffmpeg directly** (X11), e.g. a 1280×720 region:
  ```bash
  ffmpeg -f x11grab -framerate 30 -video_size 1280x720 -i :0.0+100,100 \
         -c:v libx264 -crf 23 ~/Videos/rviz_demo.mp4
  ```
- For the RViz2 sim specifically: launch it
  (`ros2 launch kitti_seg_sim sim.launch.py`), record the RViz window, then trim
  to a clean loop with the `START`/`DURATION` args above.

---

## Recap

```
record (.mp4/.webm)
      │
      ├─ short loop, must render everywhere ──► scripts/make_gif.sh ──► docs/images/demo.gif
      │                                          ![alt](docs/images/demo.gif)
      │
      └─ longer / HQ / keep repo light ───────► shrink mp4 ──► drag-drop on GitHub
                                                 (or <video> tag)
```

When your clip is ready, hand me the file path and I'll run `make_gif.sh` with
sensible settings, drop the result into `docs/images/`, and wire it into the
README for you.
