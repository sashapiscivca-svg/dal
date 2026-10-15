"""A vectorised bank of Kalman filters for N targets at once. State: `(cx, cy, vx, vy, w, h)`.

The centre follows a constant-velocity model with white acceleration noise; the size is a random walk. Pure numpy. Ideas:
SORT (Bewley et al., 2016) for the state layout; NSA Kalman (GIAOTracker, 2022) for scaling the measurement noise by
confidence; BoT-SORT (2022) for camera-motion compensation. Written from scratch.

The defaults are the robust settings that fixed runaway tracks in early experiments: a small velocity prior, a velocity clamp,
a Mahalanobis gate and velocity damping while coasting without measurements.
"""
from __future__ import annotations

import numpy as np

_H = np.zeros((4, 6))
_H[0, 0] = _H[1, 1] = _H[2, 4] = _H[3, 5] = 1.0


class KalmanBank:
    def __init__(self, meas_std: float = 2.0, accel_std: float = 0.5, size_std: float = 0.3, vel_std0: float = 2.0,
                 ego_std: float = 0.5, min_conf: float = 0.05, vmax: float = 8.0):
        self.meas_std, self.accel_std, self.size_std = meas_std, accel_std, size_std
        self.vel_std0, self.ego_std, self.min_conf, self.vmax = vel_std0, ego_std, min_conf, vmax
        self.x = np.zeros((0, 6))
        self.P = np.zeros((0, 6, 6))

    def __len__(self) -> int:
        return len(self.x)

    def init(self, boxes: np.ndarray) -> None:
        """`boxes (N, 4)`: cx, cy, w, h. Replaces the whole bank; velocities start at zero."""
        b = np.asarray(boxes, float).reshape(-1, 4)
        self.x = np.zeros((len(b), 6))
        self.x[:, [0, 1, 4, 5]] = b
        p = np.diag([self.meas_std ** 2] * 2 + [self.vel_std0 ** 2] * 2 + [self.meas_std ** 2] * 2)
        self.P = np.repeat(p[None], len(b), 0)

    def predict(self, dt: float = 1.0, ego: np.ndarray | None = None, damp: float = 1.0) -> None:
        """Prediction step. `ego` `(2,)` or `(N, 2)` is the apparent image shift caused by camera motion, in pixels (added to
        the centre). `damp < 1` damps the velocity while coasting, so a lost target does not fly away at its last speed.
        The velocity is always clamped to `vmax` pixels per frame."""
        if not len(self.x):
            return
        F = np.eye(6)
        F[0, 2] = F[1, 3] = dt
        F[2, 2] = F[3, 3] = damp
        q = self.accel_std ** 2
        Q = np.zeros((6, 6))
        for p, v in ((0, 2), (1, 3)):                       # discrete white-acceleration noise
            Q[p, p], Q[p, v], Q[v, p], Q[v, v] = q * dt ** 4 / 4, q * dt ** 3 / 2, q * dt ** 3 / 2, q * dt ** 2
        Q[4, 4] = Q[5, 5] = (self.size_std ** 2) * dt
        self.x = self.x @ F.T
        self.x[:, 2:4] = np.clip(self.x[:, 2:4], -self.vmax, self.vmax)
        self.P = F @ self.P @ F.T + Q
        if ego is not None:
            e = np.broadcast_to(np.asarray(ego, float), (len(self.x), 2))
            self.x[:, :2] += e
            self.P[:, 0, 0] += self.ego_std ** 2             # the camera-motion estimate is uncertain too
            self.P[:, 1, 1] += self.ego_std ** 2

    def _R(self, conf: np.ndarray, r_scale: float = 1.0) -> np.ndarray:
        s = 1.0 / np.clip(np.asarray(conf, float), self.min_conf, 1.0)   # NSA: lower confidence -> larger noise
        return r_scale * (self.meas_std ** 2) * s[:, None, None] * np.eye(4)[None]

    def update(self, idx: np.ndarray, z: np.ndarray, conf: np.ndarray, r_scale: float = 1.0) -> None:
        """Measurement update of targets `idx` with boxes `z (n, 4)`. `r_scale < 1` marks a measurement more precise than usual
        (for example a detector box) by scaling the noise variance."""
        idx = np.asarray(idx, int)
        if not len(idx):
            return
        z = np.asarray(z, float).reshape(-1, 4)
        P = self.P[idx]
        R = self._R(conf, r_scale)
        S = _H @ P @ _H.T + R
        K = P @ _H.T @ np.linalg.inv(S)
        y = z - self.x[idx] @ _H.T
        self.x[idx] = self.x[idx] + np.einsum("nij,nj->ni", K, y)
        I_KH = np.eye(6)[None] - K @ _H
        self.P[idx] = I_KH @ P @ I_KH.transpose(0, 2, 1) + K @ R @ K.transpose(0, 2, 1)   # Joseph form
        self.x[idx, 2:4] = np.clip(self.x[idx, 2:4], -self.vmax, self.vmax)

    def mahalanobis(self, z: np.ndarray) -> np.ndarray:
        """Mahalanobis distance of measurement `z (N, 4)` to each corresponding target (for gating)."""
        z = np.asarray(z, float).reshape(-1, 4)
        S = _H @ self.P @ _H.T + self._R(np.ones(len(self.x)))
        y = z - self.x @ _H.T
        return np.sqrt(np.einsum("ni,nij,nj->n", y, np.linalg.inv(S), y))

    def box(self) -> np.ndarray:
        return self.x[:, [0, 1, 4, 5]].copy()

    def velocity(self) -> np.ndarray:
        return self.x[:, 2:4].copy()

    def pos_std(self) -> np.ndarray:
        return np.sqrt(self.P[:, 0, 0] + self.P[:, 1, 1])
