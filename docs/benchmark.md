# Benchmark protocol

dal has two benchmarks. **Short clips** measure, frame by frame, whether the box is on the person (recall, precision, F1). **Long-term holding** measures whether the person is still held after N frames (Survival@N) and how the tracker fails (drift, silent losses, re-acquisition). Both run from the command line on any folder of annotated clips (`dal bench short|survival`) and are implemented in `src/dal/bench.py`, `src/dal/metrics.py` and `src/dal/longterm.py`.

## Common protocol

- **Initialisation:** the ground-truth box of frame 0. The tracker then sees frames 1..T only. No detector, unless corrections are switched on explicitly (`--correct-every`).
- **Scale:** every clip is downscaled so that the person in frame 0 is about **24 px** (`s = clip(24 / max(w0, h0), 1/16, 1)`; frames are never upscaled). The prepare scripts store clips at this scale. The trackers were tuned for this size.
- **Colour:** SeaDronesSee and WiSARD clips are 8-bit grayscale; MOBDrone short clips are RGB (the `v1` and `thermal` profiles use their luminance; `visible` uses luminance plus gradients).
- **Hit rules:** `iou` = IoU ≥ 0.5; `center` = the predicted centre lies within `max(w, h) / 2 + 2` px of the target centre. On 6–25 px targets IoU is very strict (a 3 px offset on a 12 px target gives 0.6), so both are reported.
- **Reported frames:** a frame counts as reported when the tracker does not declare the target lost. Recall = hits / all frames; precision = hits / reported frames.
- **Confidence presets:** `balanced` maximises F1 and `sensitive` maximises F2 over the confidence threshold; for the classic tracker the best threshold is 0 (frames below the PSR threshold are already declared lost).
- **Tuning discipline:** settings were tried on every third clip (`--every 3 --offset 0`) of SeaDronesSee and MOBDrone and checked on the other two thirds (`--offset 1`, `--offset 2`). WiSARD settings were tried on separate tuning locations (Baker, Carnation, FHL); the evaluation locations (Karen, MtErie, Airfield) were never used for tuning.

## Data preparation

| Set | Source and licence | Command | Clips; frames |
|---|---|---|---|
| `data/sds_val` | SeaDronesSee SOT val, CC0 1.0 | `python scripts/prepare_seadronessee.py --split val --out data/sds_val` | 70; 13,312 (+70 init frames) |
| `data/mobdrone_clips` | MOBDrone, CC BY 4.0 | `python scripts/prepare_mobdrone.py clips --zip MOBDrone_videos.zip --ann annotations_person_coco_classes.json --out data/mobdrone_clips` | 119; 33,423 |
| `data/mobdrone_hold` | MOBDrone, CC BY 4.0 | `python scripts/prepare_mobdrone.py long --zip ... --ann ... --out data/mobdrone_hold --split hold` | 24 tracks; 16,348 |
| `data/wisard_ir_val` | WiSARD thermal, MIT | `python scripts/prepare_wisard_ir.py --root WiSARD --sites Karen MtErie Airfield --out data/wisard_ir_val` | 20; 2,398 |
| `data/wisard_ir_val_hard` | WiSARD thermal, MIT | `... --sites Karen MtErie Airfield --top 12 --min-len 30 --out data/wisard_ir_val_hard` | 52; 3,843 |
| `data/wisard_ir_long` | WiSARD thermal, MIT | `... --sites Karen MtErie Airfield --top 3 --min-len 100 --max-len 1200 --out data/wisard_ir_long` and `... --sites Baker Carnation FHL --top 2 --min-len 150 --max-len 1200 --out data/wisard_ir_long --prefix tune_` | 16 tracks (10 + 6), 9 sequences |

SeaDronesSee frames are read from the dataset's public archive with HTTP Range requests, so only the frames of the requested clips are downloaded (3,267 unique frames, 2.3 GB, for all 70 validation clips; `--clips 1-4,33-41` takes 286 frames, 160 MB). The converted 70-clip set takes a few GB more, because clips whose person is already smaller than 24 px stay at full 4K resolution. MOBDrone and WiSARD must be downloaded from their pages first ([datasets.md](datasets.md)). WiSARD has no track ids: its clips are **pseudo-tracks** linked by IoU, and some of them jump from one person to another.

## Short clips

```bash
dal bench short data/sds_val            --tracker classic:v1 --tracker classic:thermal --tracker classic:visible --json runs/short_sds.json
dal bench short data/mobdrone_clips     --tracker classic:v1 --tracker classic:thermal --tracker classic:visible
dal bench short data/wisard_ir_val      --tracker classic:v1 --tracker classic:thermal --tracker classic:visible
dal bench short data/wisard_ir_val_hard --tracker classic:thermal
dal bench short data/sds_val --tracker onnx:models/dal-n                        # learned, balanced preset (lost_thr 0.1)
dal bench short data/sds_val --tracker onnx:models/dal-n --kw '{"lost_thr": 0.03}'   # learned, sensitive preset
dal bench short data/sds_val --tracker classic:visible --every 3 --offset 1      # a held-out third only
```

Documented results (F1 IoU / F1 centre; share of frames declared lost):

| Set | `v1` | `thermal` | `visible` | `dal-n` sensitive | `dal-n` balanced |
|---|---|---|---|---|---|
| SeaDronesSee val (70) | 0.778 / 0.931 (0.08) | 0.767 / 0.937 (0.06) | 0.765 / 0.964 (0.02) | 0.734 / 0.913 (0.04) | 0.750 / 0.866 (0.15) |
| MOBDrone (119) | 0.417 / 0.979 (0.04) | 0.416 / 0.976 (0.04) | 0.472 / 0.997 (0.00) | 0.387 / 0.837 (0.14) | |
| WiSARD thermal val (20) | 0.523 / 0.739 (0.03) | 0.529 / 0.794 (0.04) | 0.474 / 0.701 (0.00) | 0.339 / 0.545 (0.06) | |
| WiSARD thermal val hard (52) | | 0.627 / 0.841 | | | |

Recall and precision for each cell, AP (COCO style, needs `pycocotools`), success AUC and P@5/10 px are in the JSON written by `--json`. On SeaDronesSee the `v1` recall / precision by IoU is 0.746 / 0.814 and by centre 0.892 / 0.973.

## Long-term holding (Survival@N)

- **Miss:** the centre is farther than `max(w, h) / 2 + 2` px from the person (the `center` rule).
- **Loss:** at least `x = 10` consecutive misses (`--x`). Shorter glitches are not losses. A loss ends when the person is hit 5 frames in a row.
- **Survival@N:** among tracks at least N frames long, the share whose first loss happens at frame N or later. Shown only when at least 3 tracks are that long; `n` is the number of tracks.
- **Mean time to first loss:** mean of min(first loss, track length), so tracks without a loss are censored at their length.
- **Drift:** median and p95 centre error by track age.
- **False captures:** loss events during which the tracker did not declare a loss on at least half of the frames ("confidently wrong"); for MOBDrone also events where its centre sat inside another person.
- **Re-acquisition:** share of loss events after which the person is found again within 30 / 100 / 300 frames.
- **Confidence intervals:** 95 % percentile bootstrap over **sequences** (videos), 500 resamples, seed 0; tracks from one video are resampled together.

```bash
dal bench survival data/mobdrone_hold --tracker classic:v1 --tracker classic:visible --json runs/surv_mob.json
dal bench survival data/sds_val        --tracker classic:v1 --tracker classic:visible
dal bench survival data/wisard_ir_long --tracker classic:thermal --tracker classic:visible
```

Documented results (all tracks of each set; the full summaries are in [`results/survival_reference.json`](../results/survival_reference.json)):

| Set | Tracker | @50 | @100 | @200 | @500 |
|---|---|---|---|---|---|
| MOBDrone hold (24) | `v1` | 96 % [88–100] | 96 % [88–100] | 92 % [79–100] | 79 % [58–95] (n=19) |
| | `visible` | 100 % | 100 % | 100 % | 100 % (n=19) |
| SeaDronesSee val (70) | `v1` | 97 % [93–100] | 94 % [89–99] | 64 % [42–84] (n=22) | |
| | `visible` | 99 % [96–100] | 97 % [92–100] | 68 % [45–86] (n=22) | |
| WiSARD thermal (16 tracks, 9 sequences) | `thermal` | 75 % [59–91] | 69 % [50–88] | 75 % [55–100] (n=8) | |
| | `visible` | 75 % [59–91] | 69 % [44–89] | 88 % [67–100] (n=8) | |

The original research log also reports SeaDronesSee on its held-out two thirds only (46 clips): `visible` Survival@100 96 % [89–100], @200 57 % [29–82] (n=14). The WiSARD set mixes 10 held-out and 6 tuning-location tracks; only 5 sequences are from held-out locations, which is too few for firm conclusions.

### Detector corrections

`--correct-every K` simulates a detector that re-points the person every K frames: the ground-truth box with noise (centre σ = 0.1 × size, size × exp(N(0, 0.1))). This is an **upper bound** for a real detector, which also misses people, raises false alarms and needs association. With K ≤ 10 a loss (10 misses) cannot happen by construction, so use K ≥ 15.

```bash
dal bench survival data/sds_val        --tracker classic:visible --kw '{"corr_hard": 2.0, "corr_reinit": true, "recapture": true}' --correct-every 15
dal bench survival data/wisard_ir_long --tracker classic:thermal --kw '{"corr_reinit": true}' --correct-every 15
```

| Set | Without corrections | Simulated detector every 15 frames |
|---|---|---|
| SeaDronesSee val, `visible` → `visible` + `WITH_DETECTOR` | @200: 68 % (n=22) | @200: 91 % [78–100] |
| WiSARD thermal, `visible` → `visible` + `WITH_DETECTOR` | @100: 69 % | @100: 94 % [79–100] |
| WiSARD thermal, `thermal` → `thermal` + `corr_reinit` | @100: 69 % | @100: 100 % |
| MOBDrone, box IoU ≥ 0.5 share | 49 % of frames | 96 % of frames |

`WITH_DETECTOR` (`dal.classic.WITH_DETECTOR`) is `corr_hard=2.0` (a correction farther than 2 box sizes, or one after a loss, resets the Kalman filter), `corr_reinit=True` (the correlation filter is re-learned at the corrected box) and `recapture=True`. Without the hard reset, MOBDrone box accuracy at K = 15 / 30 / 60 was 93 / 87 / 78 % instead of 96 / 94 / 90 %. Corrections every 30 or 60 frames helped much less (WiSARD @100: 88 % and 75 %).

A real detector in the loop is not evaluated in this release: every number here is reproducible from public data and code, and no detector model is part of it.

## Quick check

To check that this repository reproduces the original research code, we prepared 13 SeaDronesSee validation clips and ran both:

```bash
python scripts/prepare_seadronessee.py --split val --clips 1-4,33-41 --out data/sds_val_quick       # about 160 MB of frames
dal bench short data/sds_val_quick --tracker classic:v1 --tracker classic:thermal --tracker classic:visible --tracker onnx:models/dal-n
dal bench survival data/sds_val_quick --tracker classic:v1 --tracker classic:visible
```

| Tracker (13 clips, 1,846 frames) | F1 IoU | F1 centre | lost | AP50 | original code |
|---|---|---|---|---|---|
| `classic:v1` | 0.823 | 1.000 | 0.00 | 0.725 | identical |
| `classic:thermal` | 0.820 | 1.000 | 0.00 | 0.720 | identical |
| `classic:visible` | 0.837 | 1.000 | 0.00 | 0.740 | identical |
| `onnx:models/dal-n` (balanced) | 0.807 | 0.930 | 0.07 | 0.768 | identical to the PyTorch checkpoint |

Survival@50 and @100 are 100 % for both classic profiles on this subset (clips are 143 frames long), and the p95 drift is 10.5–10.8 px at 101–142 frames. These clips are easy (calm water, no losses); the full 70 clips give the numbers above. Per-frame boxes agree with the stored outputs of the original long-term runs within 0.33 px, with identical lost flags; the small difference comes from later changes to the camera-motion downscaling, which were made after those runs.

## Your own clips

Any folder of clips works: an image folder with `groundtruth.txt` (`x,y,w,h` per line, top-left corner), a video with a `.txt` sidecar, an `.npz` (`frames` or `enc`/`enc_off`, `boxes` as cx, cy, w, h), or a `.clip.json` piece of a longer video. See `src/dal/clips.py`. Use `--target-px 24` if your clips are not already scaled, and put a `meta.json` with a `group` in each folder if several clips come from the same video, so that the bootstrap treats them as one sequence.

## Speed

`dal bench` prints the time per frame including decoding, which is not a clean latency figure. The published x86 numbers were measured with the tracker alone on 200 synthetic 640×512 frames, one thread. The Raspberry Pi Zero 2 W numbers were measured on the device (Raspberry Pi OS Bookworm 64-bit, Python 3.11, OpenCV 5.0, one thread, 60 frames of a 640×512 WiSARD thermal sequence, `ego_scale` 0.5, no heatsink): `thermal` 31.2 ms p50 (35.8 ms p95) and `visible` 53.4 ms p50 (54.0 ms p95) per track and frame; with 4 tracks 124.9 and 172.8 ms. A profile on the Pi attributed 74 % of a step to phase correlation; `ego_scale=0.3` with FFT-friendly sizes reduced that part from 29 to 20 ms. The learned `dal-n` network step alone took 21.4 ms (1 thread) and 10.8 ms (4 threads) on the Pi.
