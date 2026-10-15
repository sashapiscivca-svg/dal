# Architecture

dal follows one person through video after something else (an operator, or a person detector such as [hae](https://github.com/sashapiscivca-svg/hae)) has pointed at them. It returns a box, a confidence and a "lost" flag per frame, and accepts new boxes from a detector at any time. It does not detect people by itself, it does not identify anyone, and it does not control any vehicle, camera or gimbal.

```
                      ┌──────────── every frame ─────────────────────────────────────────────────┐
frame t ──► ego-motion (phase correlation, frame t-1 → t, downscaled) ──► (dx, dy)               │
        ──► Kalman predict: constant velocity + (dx, dy); velocity damped while lost ──► centre  │
        ──► win × win patch (48 px) ──► MOSSE response ──► peak offset, PSR → conf               │
        ──► lost if conf < 0.25 or the Mahalanobis gate rejects the measurement                  │
        ──► Kalman update and filter update only when found ──► {box, conf, lost}               │
                      └──────────────────────────────────────────────────────────────────────────┘
detector box every K frames (optional) ──► correct(): Kalman measurement with small noise;
                                           after a loss: re-learn the filter at the detection
```

## Components

| Module | What it does |
|---|---|
| `dal.ego` | Camera motion by phase correlation of whole frames (OpenCV), downscaled (`ego_scale`, 0.5 by default; 0.3 recommended on a Raspberry Pi) to an FFT-friendly size. No parameters, no learning. |
| `dal.kalman.KalmanBank` | Vectorised Kalman filter for N targets, state `(cx, cy, vx, vy, w, h)`: constant velocity with white acceleration, random-walk size, measurement noise scaled by 1 / confidence, camera shift added in the prediction, Joseph-form update, Mahalanobis gate, velocity clamp and damping. |
| `dal.classic.mosse` | MOSSE correlation filter (Bolme et al., 2010) in numpy: log + normalise + Hann window, initial training on the first patch and 8 small affine perturbations of it, running update with forgetting rate 0.05, sub-pixel parabolic peak, PSR confidence. Multi-channel form (MCCF) for feature stacks. |
| `dal.classic.features` | Patch features: `gray` (log luminance) and `grad` (luminance + 4 gradient-orientation channels, HOG-like). Others (`rgb`, `lumcol`, `lcn`, `hot`, `dog`) are kept for experiments. |
| `dal.classic.tracker.ClassicTracker` | The loop above, plus all options (re-capture, detector corrections, scale search, anchor and memory-bank filters, forward-backward check, multi-hypothesis peak choice). Everything except the core loop is off by default. |
| `dal.classic.profiles` | `visible`, `thermal`, `v1` and the experimental `thermal_exp` settings. |
| `dal.metrics`, `dal.longterm` | Frame metrics (IoU, centre hits, recall / precision / F1, COCO AP, NWD) and long-term metrics (Survival@N, losses, drift, false captures, re-acquisition, bootstrap). |
| `dal.clips`, `dal.bench`, `dal.cli` | Clip formats, benchmark runners, command line. |
| `dal.onnx_tracker.OnnxTracker` | The learned tracker through ONNX Runtime, without PyTorch. |
| `dal.learned` | DalNet, LIF neurons, the PyTorch tracker, synthetic data, training and the stateful ONNX export. |

## Profiles

| Profile | Settings | Chosen because |
|---|---|---|
| `visible` | `features="grad"`, Kalman on | Best on RGB people-in-water sets, consistently on every third of MOBDrone (+0.055 F1 IoU) and on the hardest SeaDronesSee clips (centre F1 of the worst quarter 0.760 → 0.883). |
| `thermal` | luminance, `mode="no_kalman"` | Best on held-out thermal locations; every alternative that won on tuning locations (hot-spot and gradient features, context negatives, anchor filter, update guard) lost there. |
| `v1` | luminance, Kalman on | The first validated tracker; best F1 by IoU on SeaDronesSee (0.778). |

`make_tracker("visible", **overrides)` builds a tracker; `dal track --profile auto` picks `thermal` for grayscale video (including gray video stored as RGB) and `visible` otherwise.

## Design decisions and what the evidence said

These are condensed from the project's decision log. "Held-out" means clips or locations that were not used to choose the setting.

1. **Measurement and estimation are separate.** The correlation filter (or network) measures; the Kalman filter estimates. In the learned tracker the Kalman filter added 0.06–0.08 NWD-AUC on synthetic sequences and 0.13 on occluded frames. In the classic tracker it changes F1 by at most ±0.01 on RGB, and on thermal footage the tracker is better without it, hence the `thermal` profile.
2. **Robust Kalman settings beat retraining.** The first learned tracker had a mean centre error of 40–48 px: one bad measurement accelerated the velocity (prior 10 px / frame, gate 12σ) and a lost target flew away. A weak velocity prior (2), a clamp (8 px / frame), a 5–8σ gate and damping 0.85 while lost raised NWD-AUC from 0.633 to 0.698 and cut the error to 11 px without retraining.
3. **Camera motion without learning.** Phase correlation needs no parameters and no data. On a Raspberry Pi Zero 2 W it is the most expensive part of a classic step (74 %), so the downscaled frame is rounded to FFT-friendly sizes and the previous frame is cached; `ego_scale=0.3` cut it from 29 to 20 ms with no loss on the thermal hard set (centre F1 0.846 vs 0.842); 0.25 was too coarse (0.811).
4. **A classic baseline was set up to check the learned tracker.** The owner's rule: if the training-free tracker trails the learned one by at most 0.03 F1 (centre), it may become the default. It did not trail at all: it was ahead on all three sets (by 0.02–0.19 centre F1). The classic tracker became the default.
5. **Gradients, not colour.** Colour channels (`rgb`, `lumcol`) gave no gain of 0.02 F1; gradient channels did on RGB and hurt on thermal.
6. **No scale estimation.** A DSST-like slow scale filter cut F1 by IoU from 0.778 to 0.647 on SeaDronesSee; with a dead zone and a pull to the template size it stopped hurting but gained at most 0.015. The ceiling, with the true size substituted, is +0.06 (SeaDronesSee) to +0.20 (MOBDrone, IoU-hit share), and the annotated boxes change with pose (w and h diverge), not as a uniform scale. The box size therefore stays the size from frame 0 (or the last detector correction).
7. **Re-capture is safe but small.** Searching a grid around the prediction after a loss, accepting only a confident peak in the central half of a patch, never created a false find on empty background and changed F1 by ±0.01. It is on in `WITH_DETECTOR` and off otherwise.
8. **Long-term holding needs outside help.** Forward-backward checks, multi-hypothesis peak choice and a memory bank of templates did not raise Survival@N by 1 point. Simulated detector corrections every 15 frames did (WiSARD thermal Survival@100 69 % → 94–100 %). With corrections, a far correction or one after a loss should hard-reset the Kalman filter and re-learn the correlation filter (`corr_hard`, `corr_reinit`).
9. **The tracker state is not persisted.** There is no save / load of a target model, by design (see [RESPONSIBLE_USE.md](../RESPONSIBLE_USE.md)).

## Detector corrections

- `correct(box, conf=1.0, r_scale=0.25)`: the box is a Kalman measurement with a quarter of the usual noise variance; velocity is kept. In `no_kalman` mode the box is taken as is. If the tracker was lost, the correlation filter is re-trained on the last frame at the box. Call it after `step(frame)` on the frame the detector saw.
- `associate(dets, gate_k=1.5, verify_pk=0.0, grow=0.1)`: picks the nearest detection within `gate_k` box sizes (the gate grows by 10 % per missed frame, up to ×6). Outside the gate, a detection is accepted only if the filter's own response there has a peak of at least `verify_pk` (off by default).
- `correct_from(dets)`: `associate` + `correct`.

## The learned tracker (DalNet)

```
template 32×32 ──► backbone (stride 4) ──► centre 3×3 features = kernel (once per target)
search crop S×S ──► backbone ──► depth-wise correlation with the kernel ──┐
frame difference (t − aligned t−1) ──► 1×1 ──► LIF ──► 3×3 ──► LIF ───────┴─► fuse ──► heads: centre logit, sub-cell offset, log size
                                     membranes carried between frames (motion memory), re-aligned when the crop moves
```

- **Backbone:** space-to-depth, 3×3 convolutions, stride 4, shared by the template and the search crop. **Correlation:** depth-wise, SiamFC / NanoTrack style. **Heads:** CenterNet-style peak, no NMS.
- **Motion module:** two layers of leaky integrate-and-fire neurons with a surrogate gradient (or a non-spiking twin with the same recurrence and parameter count). The membranes are explicit inputs and outputs, re-aligned by whole cells when the search crop moves.
- **Sizes:** n / t / s / m / l / x (36,639 to 1,156,391 parameters; 0.022 to 2.86 GMAC per step). Only n, t, s and m were trained; n ≈ t ≈ s on real clips, m was worse.
- **Training:** truncated BPTT over 6 frames, AdamW with cosine schedule, weight EMA; teacher forcing with a perturbed prediction that imitates the Kalman error. Data: 70 % real SeaDronesSee windows, 30 % synthetic sequences (real person patches pasted onto real backgrounds with random target and camera motion, noise and occlusions).
- **Stateful ONNX:** `<name>.onnx` is one step (`cur`, `diff`, `kern`, `state_in.m1`, `state_in.m2` → `cls`, `off`, `size`, `state_out.m1`, `state_out.m2`; dynamic crop size); `<name>.tpl.onnx` encodes the template. Crop re-alignment of the state stays outside the graph (`realign_np`). The initial state is zeros. Metadata keys `dal.*` carry the search size, presets and licence. ONNX matches PyTorch to 5.7e-6.
- **Spiking vs non-spiking:** on synthetic sequences the spiking module scored 0.701 vs 0.679 NWD-AUC for the twin (two seeds, spread about 0.007), but the twin differs in nonlinearity too, so the benefit of spikes is not proven. With T = 1 inner step both have the same graph cost: 0.8 ms per step on a laptop CPU for n. More inner steps (T = 2, 4) did not help (0.703, 0.696 vs 0.698).
