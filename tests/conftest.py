"""Skip the learned-path tests when PyTorch is not installed (the runtime CI job checks the numpy + OpenCV tracker alone)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TORCH_TESTS = ["test_colour.py", "test_export.py", "test_learned_tracker.py", "test_lif.py", "test_model.py", "test_presets.py",
               "test_real_sampler.py", "test_sampler_loss.py", "test_train_smoke.py"]

try:
    import torch  # noqa: F401
    collect_ignore = []
except ImportError:
    collect_ignore = TORCH_TESTS
