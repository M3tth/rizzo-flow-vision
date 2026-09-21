"""Rizzo Flow: typed decisions over local Spark logits."""

from .runtime import prepare

prepare()  # before anything imports the MLX stack (Windows needs a stub and the CUDA DLL path)

from .responses import Response
from .schema import Request

__all__ = ["Request", "Response"]
