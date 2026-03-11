"""
Dataset – Memory-mapped token loader
-------------------------------------
Reads pre-tokenized .bin files (uint16 numpy arrays) via memory-mapping
so that even large datasets can be used without loading everything into RAM.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """
    PyTorch Dataset that reads a flat binary file of token IDs
    and returns (input, target) chunks for next-token prediction.

    The binary file is expected to be a flat array of np.uint16 token IDs
    created during the data-preparation step.

    Usage:
        ds = TokenDataset("data/processed/train.bin", block_size=128)
        x, y = ds[0]   # x: (128,)  y: (128,)
    """

    def __init__(self, bin_path: str, block_size: int = 128):
        """
        Args:
            bin_path:    Path to .bin file with uint16 token IDs.
            block_size:  Context length (number of tokens per sample).
        """
        assert os.path.exists(bin_path), f"Data file not found: {bin_path}"
        self.block_size = block_size
        # Memory-map the file so we never load the whole thing into RAM
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.data)
        print(f"Loaded {self.num_tokens:,} tokens from {bin_path}")

    def __len__(self) -> int:
        # Number of non-overlapping chunks
        return (self.num_tokens - 1) // self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size + 1  # +1 because target is shifted by 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        x = chunk[:-1]   # input tokens
        y = chunk[1:]     # target tokens (shifted by 1)
        return x, y
