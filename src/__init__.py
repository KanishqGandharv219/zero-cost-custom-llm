"""zero-cost-custom-llm package exports.

Uses optional imports so lightweight scripts (for example data prep) can run
without requiring every training dependency at import time.
"""

from .config import GPTConfig

try:
    from .dataset import TokenDataset
except ModuleNotFoundError:
    TokenDataset = None  # type: ignore[assignment]

try:
    from .tokenizer import Tokenizer
except ModuleNotFoundError:
    Tokenizer = None  # type: ignore[assignment]

try:
    from .model import GPT
except ModuleNotFoundError:
    GPT = None  # type: ignore[assignment]

__all__ = ["GPTConfig", "TokenDataset", "Tokenizer", "GPT"]
