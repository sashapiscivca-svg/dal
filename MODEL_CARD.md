# Model card: dal-n v0.1.0 (learned tracker, research release)

> [!NOTE]
> **dal-n is not the recommended tracker.** On every benchmark we ran, the training-free classic tracker in the same package was at least as accurate (table below). dal-n is released because the comparison is part of the result, and so that others can reproduce it, fine-tune it or beat it.

## Summary

| | |
|---|---|
| Task | single-object tracking of a person (one box per frame), given the box in the first frame |
| Architecture | DalNet size `n`: Siamese stride-4 backbone, depth-wise correlation, a spiking motion module (two layers of leaky integrate-and-fire neurons, T = 1), CenterNet-style head; inside a Kalman filter + camera-motion loop (`OnnxTracker`) |
| Parameters / compute | 36,639 parameters; 0.022 GMAC per step at a 96×96 search crop |
| Input | grayscale (1 channel), values / 255; template 32×32 around the initial box; search crop 64 or 96 px |
| Files | `models/dal-n/dal-n.onnx` (one step, 158 KB), `dal-n.tpl.onnx` (template encoder, 63 KB), `dal-n.safetensors` (PyTorch weights, 153 KB) + `config.json`; SHA-256 in `models/SHA256SUMS` |
| Presets | `balanced`: `lost_thr` 0.1 (default); `sensitive`: `lost_thr` 0.03 (stored in the ONNX metadata) |
| Licence | MIT (code and weights) |
| Intended use | research on lightweight tracking for civilian search and rescue; see [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) |

```bash
pip install -e ".[onnx]"
dal track flight.mp4 --init 812,430,22,18 --tracker onnx --model models/dal-n
```

```python
from dal.onnx_tracker import OnnxTracker
trk = OnnxTracker.from_model("models/dal-n", preset="balanced")
trk.init(frame0, (cx, cy, w, h))
r = trk.step(frame1)        # {"box", "conf", "lost", "S"}
```

## Training

- **From scratch.** No pretrained backbone, teacher or third-party weights.
- **Recipe (v3):** 1,500 optimisation steps × 16 sequences × 6 frames, truncated BPTT through the motion state, AdamW (lr 2e-3 with warm-up and cosine decay, weight decay 1e-4), weight EMA 0.99, seed 0, one run; 980 s on one NVIDIA T4.
- **Data:** each batch was, with probability 0.7, real windows and otherwise synthetic sequences.
  - Real: 1,880 windows of 7 frames from 47 of the 58 SeaDronesSee SOT training clips (every 5th clip held out), grayscale, downscaled so the person is about 24 px (`scripts/build_sds_windows.py --split train --dev-every 5 --per-clip 40 --no-tpl`). SeaDronesSee is CC0 1.0.
  - Synthetic: real person patches and real backgrounds from HIT-UAV trainval (CC0 1.0) and AIResQ train (CC BY 4.0), pasted with random target and camera motion, sensor noise and occlusions.
- **Teacher forcing:** the search crop is centred on a perturbed prediction (previous ground truth + a random fraction of the velocity + Gaussian noise of 1–5 px) to imitate Kalman errors; the previous frame is aligned with the true camera shift plus 0.5 px noise.

To retrain: `scripts/build_sds_windows.py` and `scripts/train.py` (the command is in its docstring); export with `scripts/export_onnx.py`.

## Evaluation

Protocol in [docs/benchmark.md](docs/benchmark.md): initialised from the frame-0 ground truth, no detector afterwards, F1 per frame by IoU ≥ 0.5 and by centre (within half the larger target side + 2 px).

| Set (clips; frames) | dal-n balanced | dal-n sensitive | best classic profile |
|---|---|---|---|
| SeaDronesSee val (70; 13,312) | 0.750 / 0.866, lost 0.15 | 0.734 / 0.913, lost 0.04 | `v1` 0.778 / 0.931; `visible` 0.765 / 0.964 |
| MOBDrone (119; 33,423) | | 0.387 / 0.837, lost 0.14 | `visible` 0.472 / 0.997 |
| WiSARD thermal, held-out locations (20; 2,398) | | 0.339 / 0.545, lost 0.06 | `thermal` 0.529 / 0.794 |

Entries are F1 by IoU / F1 by centre. On SeaDronesSee val, AP50 (COCO style, frames as images) is 0.674 for dal-n balanced and 0.688 for the classic `v1`. Long-term holding (Survival@N) was not measured for dal-n.

Other findings from the same project:

- **Real data matters.** The same recipe without real windows (synthetic only) reached AP50 0.425 on SeaDronesSee val; with 50 % real windows, 0.652 (500 steps each).
- **Spikes:** on synthetic sequences the spiking motion module beat a non-spiking twin with the same parameter count by 0.022 NWD-AUC (two seeds), which does not isolate the effect of spikes. On a CPU there is no speed-up.
- **Sizes:** t and s (102 k and 204 k parameters) were no better than n on real clips; m was worse.

## Speed

| Hardware | Network step only (ONNX Runtime) |
|---|---|
| x86 laptop CPU (Intel Core i3-1115G4), 1 thread | 0.8 ms p50 |
| Raspberry Pi Zero 2 W, 1 thread (measured on the device) | 21.4 ms p50 (24.0 ms p95) |
| Raspberry Pi Zero 2 W, 4 threads | 10.8 ms p50 |

The full tracker step adds camera-motion estimation, cropping and the Kalman filter (on the Pi the camera-motion part alone was 20–29 ms per frame).

## Limitations

- Trained once (seed 0), for 1,500 steps; it may be under-trained. "The classic tracker is better" holds for this budget only.
- SeaDronesSee training and validation clips come from the same lake; the real training data is people in water, RGB converted to gray. Thermal performance on land is poor (WiSARD centre F1 0.55).
- The box size is predicted relative to the template, but IoU-based scores stay low on 6–25 px targets.
- The ONNX graphs were exported with the TorchScript-based exporter (opset 17). State re-alignment between crops happens outside the graph and must be reproduced by any other runtime (`realign_np` in `dal/onnx_tracker.py` is the reference).
- No identity or attribute outputs; it follows one box and nothing else.

## Attribution

Training data: SeaDronesSee (CC0 1.0); HIT-UAV (CC0 1.0); AIResQ Benchmark, Büttner, J., Zenodo 2025, doi:10.5281/zenodo.17405074, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (person patches were cropped, rescaled and pasted onto other frames). Full references: [NOTICE.md](NOTICE.md).
