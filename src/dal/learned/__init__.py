"""The learned tracker (secondary path): DalNet, a tiny Siamese correlator with a spiking (LIF) or non-spiking motion module,
plus training, synthetic data and stateful ONNX export. Needs PyTorch (`pip install "dal[train]"`).

On every benchmark we ran, the classic tracker (`dal.classic`) was at least as accurate, so this path is kept for research and
reproducibility, not as the recommended tracker. For inference without PyTorch use `dal.onnx_tracker.OnnxTracker`.
"""
