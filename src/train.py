"""
Training Script
---------------
End-to-end training loop for the GPT model.
Supports gradient accumulation, mixed-precision (AMP), checkpointing,
and validation loss logging.

Usage (from project root):
    python -m src.train
"""

import os
import time
import math
import torch
from torch.utils.data import DataLoader

from .config import GPTConfig
from .model import GPT
from .dataset import TokenDataset


def get_lr(step: int, config: GPTConfig) -> float:
    """Linear warmup followed by cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    # Cosine decay to 10% of max LR
    decay_ratio = (step - config.warmup_steps) / max(1, 1000 - config.warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * 0.1 + coeff * (config.learning_rate * 0.9)


@torch.no_grad()
def estimate_loss(model: GPT, val_loader: DataLoader, device: torch.device, max_batches: int = 50) -> float:
    """Compute average loss on the validation set."""
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def train(config: GPTConfig | None = None):
    """Main training function."""
    if config is None:
        config = GPTConfig()

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────
    train_path = os.path.join(config.data_dir, config.train_file)
    valid_path = os.path.join(config.data_dir, config.valid_file)

    train_ds = TokenDataset(train_path, block_size=config.block_size)
    val_ds = TokenDataset(valid_path, block_size=config.block_size)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = GPT(config).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ── AMP scaler (mixed precision) ──────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    global_step = 0

    for epoch in range(1, config.num_epochs + 1):
        t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)

            # Forward (with AMP)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _, loss = model(x, y)
                loss = loss / config.grad_accum_steps  # scale for accumulation

            scaler.scale(loss).backward()
            running_loss += loss.item() * config.grad_accum_steps

            # Optimizer step after accumulation
            if step % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # LR schedule
                lr = get_lr(global_step, config)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % config.log_interval == 0:
                    avg = running_loss / config.log_interval
                    print(f"  epoch {epoch} | step {global_step} | loss {avg:.4f} | lr {lr:.2e}")
                    running_loss = 0.0

        # ── End of epoch ──────────────────────────────────────────────────
        elapsed = time.time() - t0
        val_loss = estimate_loss(model, val_loader, device)
        print(f"Epoch {epoch}/{config.num_epochs}  | val_loss {val_loss:.4f} | time {elapsed:.1f}s")

        # Checkpoint
        if epoch % config.save_every == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"gpt_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

    print("Training complete!")


if __name__ == "__main__":
    train()
