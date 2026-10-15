# Datasets and licences

Rules: **civilian** data only (search and rescue, agriculture, nature); the licence of the **data itself** must allow commercial use and derivative works (CC0, CC BY, MIT, Apache-2.0, CDLA-Permissive). Non-commercial (NC), no-derivatives (ND), "academic use only", "licence not stated" and "on request / by subscription" mean **not used**. No datasets of military origin or funding. No third-party trained models or weights. Licence statements below were read on each dataset's primary page (README, Zenodo record or the authors' site); check the current terms before you redistribute anything.

## Used

| Dataset | Licence (where stated) | Content | How dal uses it | Redistributed here |
|---|---|---|---|---|
| **SeaDronesSee** SOT / MOT (Varga et al., WACV 2022) | **CC0 1.0** — README of [github.com/Ben93kie/SeaDronesSee](https://github.com/Ben93kie/SeaDronesSee): "dataset is Creative Commons CC0 1.0 Universal" | RGB drone video of people in open water (Lake Constance), SOT: 58 train / 70 val clips | benchmark (70 val clips); real training windows for dal-n (47 of the 58 train clips; every 5th held out); README animation | 36 downscaled frames in `docs/assets/dal-hero.gif` and one in `dal-hero-still.jpg` |
| **MOBDrone** (Cafarelli et al., ICIAP 2022) | **CC BY 4.0** — [Zenodo 5996890](https://doi.org/10.5281/zenodo.5996890) | 66 Full-HD drone videos of people in the water, 10–60 m, COCO boxes with track ids | benchmark only: 119 short clips; 24 long tracks | nothing |
| **WiSARD** (Broyles, Hayner, Leung, IROS 2022), thermal part | **MIT** — [sites.google.com/uw.edu/wisard](https://sites.google.com/uw.edu/wisard) | thermal and visible drone flights in wilderness, YOLO person boxes per frame, no track ids | benchmark only: pseudo-tracks linked by IoU from 3 held-out locations (Karen, MtErie, Airfield); 3 tuning locations (Baker, Carnation, FHL) | nothing |
| **HIT-UAV** (Suo et al., Scientific Data 2023) | **CC0 1.0** — [Zenodo 7633134](https://doi.org/10.5281/zenodo.7633134) (the GitHub copy states CC BY 4.0; we follow the stricter, with attribution) | thermal drone images with person boxes | person patches and backgrounds for the synthetic sequences that trained dal-n (trainval); synthetic test sequences (test) | nothing |
| **AIResQ Benchmark** (Büttner, 2025) | **CC BY 4.0** — [Zenodo 17405074](https://doi.org/10.5281/zenodo.17405074) | real thermal rescue footage with person boxes | person patches and backgrounds for the synthetic sequences that trained dal-n (train split) | nothing |

The selective SeaDronesSee download (`scripts/prepare_seadronessee.py`) uses the dataset authors' own public Nextcloud shares, the ones their README and dataset page point to: the SOT annotations (linked from [seadronessee.cs.uni-tuebingen.de/dataset](https://seadronessee.cs.uni-tuebingen.de/dataset)) and the multi-object-tracking frame archive that the SOT README names as the source of the frames.

Whether attribution duties extend to weights trained on CC BY data is not settled in law; we give the attribution anyway (here, in [NOTICE.md](../NOTICE.md), in the [model card](../MODEL_CARD.md) and in the ONNX metadata of dal-n).

## Considered and not used

| Dataset | Reason |
|---|---|
| BIRDSAI / Conservation Drones | licence (CDLA-Permissive) would allow it, but the project excluded it by decision (anti-poaching surveillance) |
| AnimalTrack, DanceTrack, GMOT-40, SportsMOT, SA-FARI | non-commercial terms |
| MOT17 / MOT20, VisDrone-MOT, GOT-10k, LSOTB-TIR | non-commercial or NC-SA terms |
| UAVDT | research use only |
| BDD100K | commercial use for members only |
| LaSOT, TrackingNet, TAO, UAV123, DTB70, VOT, AU-AIR (MIT covers code only), EventVOT, VisEvent, NOMAD | data licence not confirmed on a primary page, so treated as not allowed |
| eTraM (CC BY-SA 4.0) | road traffic, not needed; share-alike kept separate |
| Anti-UAV and similar | military subject matter; not considered |
| AUTH-Sheep (CC BY 4.0), ICAERUS sheep and livestock (CC BY 4.0, to be confirmed), KABR (CC0) | accepted in principle for animal tracking, but not used in this release |

## Gaps

There are no openly licensed, hand-labelled **tracks** of people on land in thermal video that we know of, which is why the thermal numbers rest on pseudo-tracks from WiSARD detection labels. There are no clean-licence tracks for flooded land, cattle, or event cameras from drones either.

Personal data: the images show people, mostly tiny and unidentifiable. The legal question of personal data in these datasets was not assessed by a lawyer; we redistribute only a few downscaled SeaDronesSee frames for the README.
