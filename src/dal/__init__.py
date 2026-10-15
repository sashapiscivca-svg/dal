"""dal: a lightweight single-object tracker for search-and-rescue drone video (civilian use only).

Given an initial box (from an operator or a person detector), dal keeps a box on that target in the following frames. The
recommended tracker is classic and training-free (MOSSE correlation filter + Kalman filter + camera-motion compensation) and
needs only numpy and OpenCV::

    from dal import make_tracker
    trk = make_tracker("visible")          # or "thermal"
    trk.init(frame0, (cx, cy, w, h))
    for frame in frames:
        r = trk.step(frame)                # {"box": (cx, cy, w, h), "conf": 0..1, "lost": bool}

dal reports where the target is in the image. It does not control a vehicle, a camera or a gimbal.
"""
__version__ = "0.1.0"

from dal.classic import PROFILES, ClassicTracker, make_tracker  # noqa: E402

__all__ = ["PROFILES", "ClassicTracker", "__version__", "make_tracker"]
