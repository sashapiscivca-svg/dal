import json

import cv2
import numpy as np

from dal.learned.patches import PatchBank
from dal.learned.evaluate import evaluate, make_eval_sequences
from dal.learned.train import TrainConfig, train


def _bank(tmp_path):
    (tmp_path / "img").mkdir()
    rng = np.random.default_rng(0)
    images, anns = [], []
    for i in range(8):
        im = rng.integers(20, 70, (160, 200), dtype=np.uint8)
        im[50:62, 60:68] = 240
        cv2.imwrite(str(tmp_path / "img" / f"{i}.png"), im)
        images.append({"id": i, "file_name": f"{i}.png", "height": 160, "width": 200})
        anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [60, 50, 8, 12], "ignore": 0, "iscrowd": 0})
    (tmp_path / "a.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": []}))
    return PatchBank([(tmp_path / "a.json", tmp_path / "img")], preload=8)


def test_loss_decreases_on_easy_data_and_eval_runs(tmp_path):
    bank = _bank(tmp_path)
    net, info = train(TrainConfig(size="n", steps=60, batch=8, L=4, warmup=5, threads=2), bank, log=lambda *_: None)
    h = info["loss"]
    assert np.mean(h[-10:]) < np.mean(h[:10]) * 0.8, (np.mean(h[:10]), np.mean(h[-10:]))
    seqs = make_eval_sequences(bank, n=2, S=96, frames=12, occlusions=1)
    r = evaluate(net, seqs, S=96, modes=("full", "kalman_only"))
    assert set(r) == {"full", "kalman_only"} and "auc_nwd" in r["full"] and 0.0 <= r["full"]["lost_rate"] <= 1.0


def test_training_mixes_real_windows(tmp_path):
    from dal.learned.real import RealSampler
    from test_real_sampler import _windows
    bank = _bank(tmp_path)
    real = RealSampler(_windows(n=6, L=4), L=4, seed=0)
    net, info = train(TrainConfig(size="n", steps=6, batch=4, L=4, warmup=2, threads=2, real_frac=1.0, search_choices=(48, 64)),
                      bank, log=lambda *_: None, real=real)
    assert len(info["loss"]) == 6 and np.isfinite(info["loss"]).all()


def test_train_accepts_device_and_returns_cpu_weights(tmp_path):
    bank = _bank(tmp_path)
    cfg = TrainConfig(size="n", steps=3, batch=4, L=3, warmup=1, threads=1, device="cpu")
    assert cfg.device == "cpu"
    net, _ = train(cfg, bank, log=lambda *_: None)
    assert all(p.device.type == "cpu" for p in net.parameters())     # weights are always returned on the CPU: the checkpoint is portable
