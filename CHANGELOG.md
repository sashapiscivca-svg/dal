# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/). Model files are versioned together with the code: a change to any released weight is a new minor version at least.

## [0.1.0] — 2026-10-15

First public release.

### Tracker
- Classic, training-free single-object tracker: MOSSE correlation filter (single- and multi-channel), Kalman filter, camera-motion compensation by phase correlation, PSR confidence and loss detection.
- Profiles `visible` (gradient features + Kalman) and `thermal` (luminance, no Kalman), plus `v1`; `make_tracker()`; detector corrections `correct()`, `associate()`, `correct_from()` and the `WITH_DETECTOR` settings.
- Experimental options, off by default: re-capture, scale search, anchor and memory-bank filters, context negatives, forward-backward check, multi-hypothesis peak choice.

### Benchmarks
- Short-clip metrics: recall / precision / F1 by IoU and by centre, confidence presets, COCO AP, success AUC, NWD.
- Long-term holding: Survival@N, time to first loss, drift by track age, false captures, re-acquisition, bootstrap CIs over sequences.
- `dal bench short|survival` on folders of clips (`.npz`, image folders with `groundtruth.txt`, videos with a sidecar, `.clip.json` video pieces).
- Dataset preparation: SeaDronesSee SOT (selective download over HTTP Range), MOBDrone (short clips and long tracks), WiSARD thermal (pseudo-tracks).
- `results/survival_reference.json`: Survival@N summaries of the documented runs.

### Learned tracker (research)
- `dal-n` v0.1.0: DalNet n with a spiking (LIF) motion module, 36,639 parameters; stateful ONNX (step + template), safetensors, model card.
- PyTorch model, synthetic sequence generator, real-window sampler, training loop and stateful ONNX export; `OnnxTracker` runs without PyTorch.

### Command line and docs
- `dal track` (annotated video and per-frame JSON), `dal bench`, `dal info`.
- README (English, Ukrainian), architecture, benchmark protocol, datasets and licences, responsible-use statement, model card, notice.
