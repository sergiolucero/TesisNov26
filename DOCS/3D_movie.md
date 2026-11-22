# The 3D plume movie: a run that cannot finish without a picture

A 3D optimization run ends with `J` and `q*`. The number says whether the policy
is cheap; the certificate says whether it is safe; neither says **what the policy
does to the plume**. This document specifies the `movies/` artifact that every
`full3D_colab_run.py` run now produces, and why each choice was made.

Files: `SCRIPTS/plume_movie3d.py` (capture/render/encode), `SCRIPTS/movie3d_from_policy.py`
(re-render from a finished run), `tests/test_movie3d.py` (22 tests).

## 1. The requirement: *always*

"Always" is the whole design constraint, so it is enforced structurally rather
than by discipline:

| risk | what prevents a lost artifact |
| :--- | :--- |
| `--plots` not passed | the movie does **not** live in `_make_plots`; it is its own stage |
| matplotlib missing/broken | the renderer uses numpy + Pillow only; no figure stack, no 3D toolkit |
| no display / headless VM | nothing is drawn to a screen; frames are arrays |
| ffmpeg missing or without libx264 | an animated GIF is written independently |
| Pillow missing | the raw field stacks are still saved as `fields.npz` |
| renderer raises at frame 37 of 48 | `make_movie` catches everything, writes `movie3d_error.txt`, still saves the stacks |
| a later stage crashes | the movie is rendered **before** `metrics.json` is written, so its paths are in the metrics |
| run crashes before the movie | `SCRIPTS/movie3d_from_policy.py` re-renders from `policy.json` + `run_config.json` |

Artifacts land in `<run>/movies/`:

```
plume3d.mp4            h264, ~0.5-3 MB, the thing you actually watch
plume3d.gif            same frames, plays anywhere
plume3d_contact.png    8 evenly spaced frames, the version that survives LaTeX
fields.npz             the raw (T, nz, ny, nx) stacks + metadata
movie3d_manifest.json  what was written, in what time, and any error
```

`metrics.json` gains `movie: {dir, mp4, gif, contact, fields, enabled}` so the
existence of the movie is machine-checkable, and a missing movie is a recorded
`error`, never silence.

## 2. Capture: the iterates, not a re-simulation

`transport_run_3d(..., capture_n=N)` keeps `N` snapshots inside the implicit-Euler
loop and returns them as `C_frames` with `frame_times` (the last snapshot is
`C_final` bit for bit). Two consequences:

* The movie shows the same numbers the objective was built from. Re-solving for
  rendering would be a second numerical experiment that nobody verified.
* Cost is one `copy()` per captured step: 24 frames on a 32x16x8 grid is 393 kB
  and ~0.1 s; on 160x96x16 it is ~23 MB and ~1 s. Nothing is stored unless asked.

Both policies are solved **on the same scenario** (held-out scenario 0 of the test
archive) so the only difference between the two panels is pumping:
`q = 0` versus `q = q*`.

## 3. Rendering: rotating maximum-intensity projection

For each height band `k` and image column `u`, the ray runs along
`e2 = (-sin t, cos t)` and the image plane's horizontal axis is
`e1 = (cos t, sin t)`:

```
x = cx + (u - U/2) du cos t - s sin t
y = cy + (u - U/2) du sin t + s cos t
```

The maximum over the ray gives the pixel; the depth `s` at which it occurred gives
the shading (`1 - 0.55 * depth`, nearer is brighter), which is what makes a
projection read as a volume rather than a silhouette.

Three deliberate choices:

1. **The vertical axis is z, not depth.** The first implementation resampled the
   *depth* axis into the image rows; the resulting picture looked like a plume
   lying on the floor. `tests/test_movie3d.py::test_vertical_axis_is_height_not_depth`
   pins the correct behaviour (low-z blob at the bottom of the frame).
2. **One colour scale for the whole movie**, fixed by the 99.5th percentile of
   the concentrations over both stacks. Per-frame autoscaling would draw a
   decaying plume at constant brightness -- exactly the distortion this artifact
   must not have.
3. **Linear by default, log available.** On a log scale a one-decade difference
   (peak 10.6 mg/L treated versus 97.6 untreated, *measured* in
   `out3d_colab/run_full`) compresses into two neighbouring colours and the two
   panels look alike; on a linear scale the effect is unmistakable. `--log` exists
   for showing the plume *edge*, which is where a log scale is the honest choice.

The injection cell is by construction the brightest object in the domain: the
source is a mass *rate* (10 m3/d at 100 mg/L) into one cell, so its equilibrium
concentration is tens of grams per litre, one to two decades above the plume. The
percentile-based scale keeps it from blacking out everything that matters, and the
colour bar states the value so the saturation is visible rather than hidden.

## 4. Frame furniture, and what each item is for

* **Axis triad** in the panel corner, rotating with the camera. Without it a
  turntable is ambiguous: the viewer cannot separate plume motion from camera
  motion. The default orbit is 40 degrees over the movie for the same reason --
  not a full spin.
* **Wells** drawn as their screened interval (`z_bot`-`z_top`), green + labelled
  with the rate when pumping, grey when shut in. Partially penetrating wells are
  a 3D feature; drawing them full-height would erase it.
* **`src` and `reg`** markers: the injection point and the compliance point.
* **Trace strip** along the bottom: the compliance-point concentration through
  time for both policies, with the regulatory limit in red and a cursor at the
  current frame. This is the bridge between the pictures and the certificate --
  the frame where the untreated curve crosses the red line is the cleanup time
  the optimizer is paying for.

## 5. Cost

Measured on the `full` preset (24x12x6, 24 frames, 3 CPUs):

| stage | cost |
| :--- | :--- |
| capture (2 policies) | 0.1 s each |
| render 24 frames at 460 px panels | ~4 s |
| encode mp4 + gif + contact sheet | ~1 s |

Under 10 s of a ~650 s run, <2 %. The whole movie stage is skipped only when
`cfg["movie"] = false`, and then `metrics.json` records
`{"enabled": false, "reason": "cfg.movie = false"}` so the absence is explicit.

## 6. Re-rendering an old run

```
python3 SCRIPTS/movie3d_from_policy.py --run out3d_colab/run_paper
python3 SCRIPTS/movie3d_from_policy.py --run <dir> --frames 32 --scenario 3 --log
python3 SCRIPTS/plume_movie3d.py --npz <run>/movies/fields.npz --out /tmp/m --fps 8
```

The first path re-solves two transports (seconds) through the run's own
constructors; the second renders from the saved stacks and touches no solver, so a
cosmetic change (panel width, fps, scale) costs nothing.

## 7. What the movie does *not* claim

* A maximum-intensity projection is a *max*, not an average: a thin filament of
  high concentration looks as bright as a thick lobe of the same peak. Read it as
  "where is the plume", never as "how much mass is here" -- the mass ledger in
  `metrics.json` is the quantitative statement.
* One scenario. The policy is certified on 40-400 scenarios; the movie is one of
  them, chosen for legibility, and labelled `scenario 0 of N` on the frame.
* The colour bar saturates above the 99.5th percentile. The plume core in the
  untreated panel is often above the top colour; the bar says so.

## 8. Measured example: `plume3d_size_mid` (8,000 cells, 32 frames)

`full3D_colab_run.py --preset size_mid --budget 90` produced, in 4,486 s:

| | untreated | under q\* (42.19 m³/d, 5 of 6 wells) |
| :--- | ---: | ---: |
| peak concentration, final frame | 44.8 mg/L | 6.0 mg/L |
| mean concentration, final frame | 2.94 mg/L | 0.19 mg/L |
| cells above the 1 mg/L limit | 35.7 % | 6.3 % |

(`SCRIPTS/plume_movie3d.stack_stats` on `movies/fields.npz`, so the numbers are
reproducible from the artifact rather than read off the image.) The 32-frame
contact sheet paired with the 0.93 certified reliability reads correctly: the
policy removes most of the plume but leaves a residual that still reaches the
compliance point on the violating scenarios.

## 9. Second measured example: the GPU policy (one well, 37.27 m³/d)

Same geometry, same scenario, but the certified policy from the torch/T4 run
(`out3d_colab/plume3d_size_mid_gpu/movies/`), 32 frames:

| | untreated | under q\* |
| :--- | ---: | ---: |
| peak concentration, final frame | 44.8 mg/L | **3.7 mg/L** |
| mean concentration, final frame | 2.94 mg/L | **0.14 mg/L** |
| cells above the 1 mg/L limit | 35.7 % | **3.9 %** |

This is the picture that matches the certificate (13/400 violations, PASS): one
well at 37.27 m³/d holds the plume in place instead of spreading the same total
pumping over five wells, and the compliance marker at x = 185 m stays clear. The
five-well policy of §8 (peak 6.0 mg/L, 6.3 % above the limit) is visibly the
weaker design — which is why the policy question has to be settled on 400 fresh
fields and not on the pictures, however suggestive they are.
