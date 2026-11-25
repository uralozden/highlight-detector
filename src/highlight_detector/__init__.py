"""Find the action in long gameplay recordings. Local, fast, no GPU, no API keys."""

from .detector import detect
from .scoring import DEFAULT_WEIGHTS, Highlight, Options

__all__ = ["detect", "Highlight", "Options", "DEFAULT_WEIGHTS", "__version__"]
__version__ = "0.1.0"
