"""Classic (training-free) tracker: MOSSE + Kalman + camera-motion compensation. Needs only numpy and OpenCV."""
from dal.classic.profiles import PROFILES, WITH_DETECTOR, make_tracker
from dal.classic.tracker import ClassicTracker

__all__ = ["PROFILES", "WITH_DETECTOR", "ClassicTracker", "make_tracker"]
