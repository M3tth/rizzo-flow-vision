"""Test-wide setup, applied before any test module imports MLX."""

import os

# TF32 rounding on NVIDIA GPUs exceeds the exact-equivalence tolerances of test_mlx.py.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import rizzo_flow  # noqa: F401  (prepares the MLX stack on Windows)
