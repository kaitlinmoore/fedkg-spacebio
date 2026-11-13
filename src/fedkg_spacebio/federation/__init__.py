"""Federated training on NVIDIA FLARE (phase 3) and intermittent links (phase 3b).

FLARE job recipes (Job Recipe API, FL simulator), a numpy FedAvg reference used to check
FLARE's result, and (stretch) a FLUID-style store-and-forward shim with simulated
loss-of-signal windows, adapted from Casaletto et al. 2025 with attribution.
"""
