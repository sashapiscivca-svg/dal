"""A bank of person patches and background frames from COCO-format detection datasets, held in memory.

A patch is a person box with a small margin and a soft alpha mask (blurred edges, so the paste does not look "stuck on"). Patch
and background come from *different* frames, so the target never coincides with a person already in the background (who stays as
a distractor).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class PatchBank:
    def __init__(self, sources, person_ids=frozenset({0}), preload: int = 400, margin: int = 2, size_range=(6, 40), seed: int = 0,
                 channels: int = 1):
        """`sources`: list of `(annotations.json, image directory)`. Only `person_ids`; `ignore` / `iscrowd` boxes are skipped.
        `channels=3` loads RGB frames (grayscale files are replicated to 3 channels)."""
        self.margin, self.size_range, self.channels = margin, size_range, channels
        items = []
        for js, d in sources:
            coco = json.loads(Path(js).read_text())
            by_img: dict[int, list] = {}
            for a in coco["annotations"]:
                if a["category_id"] in person_ids and not a.get("ignore", 0) and not a.get("iscrowd", 0):
                    by_img.setdefault(a["image_id"], []).append(a["bbox"])
            for im in coco["images"]:
                if im["id"] in by_img:
                    items.append((Path(d) / im["file_name"], by_img[im["id"]]))
        if not items:
            raise ValueError("no usable person boxes in the sources")
        rng = np.random.default_rng(seed)
        rng.shuffle(items)
        self.images: list[np.ndarray] = []
        self.boxes: list[list] = []
        for path, bxs in items[:preload]:
            g = cv2.imread(str(path), cv2.IMREAD_COLOR if channels == 3 else cv2.IMREAD_GRAYSCALE)
            if g is not None:
                if channels == 3:
                    g = np.ascontiguousarray(g[..., ::-1])      # BGR -> RGB
                self.images.append(g)
                self.boxes.append(bxs)
        if len(self.images) < 2 and len(items) >= 2:
            raise ValueError("could not read the images")

    def __len__(self) -> int:
        return len(self.images)

    def sample_bg(self, rng: np.random.Generator) -> np.ndarray:
        return self.images[int(rng.integers(len(self.images)))]

    def sample_patch(self, rng: np.random.Generator, scale=(0.8, 1.4)):
        """`(patch uint8 (h, w[, 3]), alpha float32 (h, w))`; `scale` is a random appearance scale."""
        lo, hi = self.size_range
        for _ in range(100):
            i = int(rng.integers(len(self.images)))
            img, bxs = self.images[i], self.boxes[i]
            x, y, w, h = bxs[int(rng.integers(len(bxs)))]
            m = self.margin
            x0, y0, x1, y1 = int(x) - m, int(y) - m, int(x + w) + m, int(y + h) + m
            if not (lo <= w <= hi and lo <= h <= hi) or x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
                continue
            patch = img[y0:y1, x0:x1]
            s = float(rng.uniform(*scale))
            if s != 1.0:
                patch = cv2.resize(patch, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            return patch, self._alpha(patch.shape, max(1, int(round(m * s))))
        raise RuntimeError("no suitable patch found")

    @staticmethod
    def _alpha(shape, ramp: int) -> np.ndarray:
        h, w = shape[:2]

        def edge(n):
            e = np.ones(n, np.float32)
            r = np.linspace(0.15, 1.0, ramp + 1, dtype=np.float32)[:-1] if ramp > 0 else np.ones(0, np.float32)
            k = min(len(r), n // 2)
            e[:k], e[n - k:] = r[:k], r[:k][::-1]
            return e

        return np.outer(edge(h), edge(w)).astype(np.float32)


def merge_banks(banks: list[PatchBank]) -> PatchBank:
    b = banks[0]
    for p in banks[1:]:
        b.images += p.images
        b.boxes += p.boxes
    return b
