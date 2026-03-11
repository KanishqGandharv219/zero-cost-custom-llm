"""
GPT Model Configuration
-----------------------
All hyperparameters for the model, training, and data pipeline.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- Model Architecture ---
    vocab_size: int = 32_000       # BPE vocabulary size
    embed_dim: int = 256           # Embedding / hidden dimension
    num_heads: int = 8             # Number of attention heads
    num_layers: int = 4            # Number of transformer blocks
    block_size: int = 128          # Maximum context length (sequence length)
    dropout: float = 0.1           # Dropout rate

    # --- Training ---
    learning_rate: float = 3e-4    # Adam learning rate
    batch_size: int = 8            # Micro-batch size
    num_epochs: int = 10           # Training epochs
    grad_accum_steps: int = 4      # Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)
    weight_decay: float = 0.01     # AdamW weight decay
    warmup_steps: int = 100        # LR warmup steps
    max_grad_norm: float = 1.0     # Gradient clipping

    # --- Data ---
    data_dir: str = "data/processed"   # Directory containing train.bin / valid.bin
    train_file: str = "train.bin"
    valid_file: str = "valid.bin"

    # --- Checkpointing ---
    checkpoint_dir: str = "models"
    save_every: int = 1            # Save a checkpoint every N epochs
    log_interval: int = 50         # Print loss every N steps

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        return self.embed_dim // self.num_heads
