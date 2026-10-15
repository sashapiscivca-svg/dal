import json

import cv2
import numpy as np

from dal.learned.patches import PatchBank
from dal.learned.synth import SynthConfig, make_sequence


def _make_set(tmp_path, n=6):
    (tmp_path / "img").mkdir()
    rng = np.random.default_rng(0)
    images, anns = [], []
    for i in range(n):
        im = rng.integers(20, 80, (120, 160), dtype=np.uint8)
        x, y = 30 + 10 * i, 40
        im[y:y + 12, x:x + 8] = 230                         # "person"
        cv2.imwrite(str(tmp_path / "img" / f"{i}.png"), im)
        images.append({"id": i, "file_name": f"{i}.png", "height": 120, "width": 160})
        anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [x, y, 8, 12], "ignore": 0, "iscrowd": 0})
        anns.append({"id": 100 + i, "image_id": i, "category_id": 3, "bbox": [5, 5, 8, 12], "ignore": 1, "iscrowd": 0})
    js = tmp_path / "ann.json"
    js.write_text(json.dumps({"images": images, "annotations": anns, "categories": [{"id": 0, "name": "person"}]}))
    return js, tmp_path / "img"


def test_bank_samples_bright_patch_with_alpha_and_skips_ignore(tmp_path):
    js, d = _make_set(tmp_path)
    bank = PatchBank([(js, d)], person_ids={0}, preload=6)
    rng = np.random.default_rng(1)
    for _ in range(10):
        patch, alpha = bank.sample_patch(rng, scale=(1.0, 1.0))
        assert patch.shape == alpha.shape and patch.dtype == np.uint8
        assert alpha.max() == 1.0 and alpha.min() < 1.0     # soft edge
        assert patch.max() > 200                            # really a person, not an ignore region
    bg = bank.sample_bg(rng)
    assert bg.ndim == 2 and bg.shape == (120, 160)


def test_alpha_composite_in_sequence_keeps_background_at_edges(tmp_path):
    js, d = _make_set(tmp_path)
    bank = PatchBank([(js, d)], person_ids={0}, preload=6)
    rng = np.random.default_rng(2)
    patch, alpha = bank.sample_patch(rng)
    bg = bank.sample_bg(rng)
    cfg = SynthConfig(frames=6, out_hw=(100, 120), cam_speed=0.0, target_speed=(0.0, 0.0), noise=0.0, occlusions=0)
    seq = make_sequence(bg, patch, cfg, seed=3, alpha=alpha)
    assert seq["frames"].shape == (6, 100, 120) and seq["visible"].all()
