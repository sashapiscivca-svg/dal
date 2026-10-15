# NOTICE — third-party material and attribution

The dal source code and the dal-n weights are released under the [MIT License](LICENSE), Copyright (c) 2026 The dal authors. This file lists the third-party material that dal relies on, its licence and the attribution it requires. Licence statements were read on each dataset's primary page (README, Zenodo record or the authors' site). Check the current terms on those pages before you redistribute anything.

## 1. Datasets

**Only a few SeaDronesSee frames are redistributed in this repository** (the README animation `docs/assets/dal-hero.gif` and `docs/assets/dal-hero-still.jpg`: downscaled crops of validation frames with our tracking boxes drawn on them). All other data is downloaded by the user from its official source with the scripts in `scripts/`.

| Dataset | Licence | Use in dal | Attribution |
|---|---|---|---|
| **SeaDronesSee** | CC0 1.0 | benchmark (SOT validation clips); real training windows for dal-n (SOT training clips); README animation | Varga, L. A., Kiefer, B., Messmer, M., Zell, A. *SeaDronesSee: A Maritime Benchmark for Detecting Humans in Open Water.* IEEE/CVF WACV 2022, pp. 3686–3696, doi:[10.1109/WACV51458.2022.00374](https://doi.org/10.1109/WACV51458.2022.00374). https://github.com/Ben93kie/SeaDronesSee. The README states: "Code is MIT license, dataset is Creative Commons CC0 1.0 Universal (CC0 1.0)". CC0 does not require attribution; we cite it as good practice. |
| **MOBDrone** | CC BY 4.0 | benchmark only | Cafarelli, D., Ciampi, L., Vadicamo, L., Gennaro, C., Berton, A., Paterni, M., Benvenuti, C., Passera, M., Falchi, F. *MOBDrone: a large-scale drone-view dataset for man overboard detection.* Zenodo (2022), doi:[10.5281/zenodo.5996890](https://doi.org/10.5281/zenodo.5996890); paper: *MOBDrone: a Drone Video Dataset for Man OverBoard Rescue*, ICIAP 2022, pp. 633–644, doi:[10.1007/978-3-031-06430-2_53](https://doi.org/10.1007/978-3-031-06430-2_53). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Nothing is redistributed; `scripts/prepare_mobdrone.py` cuts clips from the user's own download and downscales them. |
| **WiSARD** (thermal part) | MIT | benchmark only | Broyles, D., Hayner, C. R., Leung, K. *WiSARD: A Labeled Visual and Thermal Image Dataset for Wilderness Search and Rescue.* IEEE/RSJ IROS 2022, pp. 9467–9474, doi:[10.1109/IROS47612.2022.9981298](https://doi.org/10.1109/IROS47612.2022.9981298). https://sites.google.com/uw.edu/wisard. Distributed under the MIT License, copyright (c) 2022 Daniel Broyles, Christopher Hayner, Karen Leung. Nothing is redistributed; `scripts/prepare_wisard_ir.py` links the published boxes into pseudo-tracks. |
| **HIT-UAV** | CC0 1.0 | person patches and backgrounds for synthetic training sequences of dal-n; synthetic test sequences | Suo, J. et al. *HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection.* Scientific Data 10, 227 (2023), doi:[10.1038/s41597-023-02066-6](https://doi.org/10.1038/s41597-023-02066-6). Data: Zenodo, doi:[10.5281/zenodo.7633134](https://doi.org/10.5281/zenodo.7633134). |
| **AIResQ Benchmark** | CC BY 4.0 | person patches and backgrounds for synthetic training sequences of dal-n | Büttner, J. *AIResQ Benchmark.* Zenodo (2025), doi:[10.5281/zenodo.17405074](https://doi.org/10.5281/zenodo.17405074). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: person boxes were cropped with a small margin, rescaled and pasted with soft edges onto other frames to make synthetic sequences; nothing is redistributed. |

Whether attribution duties extend to model weights trained on these datasets is not settled in law. We give the attribution anyway; the ONNX metadata of dal-n names its training data (`dal.training_data`).

Datasets considered and rejected (non-commercial or unclear terms, military subject matter, project decision) are listed in [docs/datasets.md](docs/datasets.md).

## 2. Pretrained weights

None. dal-n was trained from random initialisation. No ImageNet or other third-party checkpoints, teachers or backbones were used. The classic tracker has no learned weights at all.

## 3. Published ideas the design draws on

All code in this repository was written by the dal authors; no third-party source code was copied. The design draws on these published ideas:

- MOSSE correlation filters — Bolme, Beveridge, Draper, Lui (CVPR 2010); multi-channel correlation filters — Kiani Galoogahi, Sim, Lucey (ICCV 2013); context-aware correlation filters — Mueller, Smith, Ghanem (CVPR 2017); DSST scale estimation — Danelljan et al. (BMVC 2014)
- Kalman filtering in SORT — Bewley et al. (ICIP 2016); confidence-scaled measurement noise (NSA Kalman, GIAOTracker, 2021); camera-motion compensation in BoT-SORT — Aharon, Orfaig, Bobrovsky (2022)
- HOG-like gradient orientation channels — Dalal, Triggs (CVPR 2005)
- SiamFC — Bertinetto et al. (ECCVW 2016); depth-wise correlation as in SiamRPN++ / NanoTrack; CenterNet — Zhou, Wang, Krähenbühl (2019)
- Leaky integrate-and-fire neurons with surrogate gradients — Neftci, Mostafa, Zenke (2019); learnable leak (PLIF) — Fang et al. (2021)
- Normalized Wasserstein Distance for tiny objects — Wang, Xu, Yang, Yu (2021)
- AdamW and SGDR — Loshchilov and Hutter (2017, 2019); weight EMA

## 4. Runtime dependencies (installed by the user, not bundled)

| Package | Licence |
|---|---|
| numpy | BSD-3-Clause (and bundled components) |
| opencv-python-headless | Apache-2.0. The wheel bundles FFmpeg under LGPL-2.1; this matters if you redistribute the wheel itself. |
| onnxruntime, onnx (optional) | MIT; Apache-2.0 |
| pycocotools, scipy (optional, benchmarks and dataset preparation) | BSD-2-Clause; BSD-3-Clause |
| torch, safetensors (optional, training) | BSD-style; Apache-2.0 |
| matplotlib, pillow (optional, figures) | PSF-based Matplotlib licence; MIT-CMU (HPND) |

dal does not ship these packages. `pip` installs them from their publishers under their own licences.
