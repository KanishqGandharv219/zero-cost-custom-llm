"""
GPT Model – From Scratch
-------------------------
A small GPT-style transformer language model built entirely in PyTorch.
Architecture:  Token Embedding + Positional Embedding
             → N × TransformerBlock (LayerNorm → MHA → Residual → LayerNorm → FFN → Residual)
             → LayerNorm → Linear Head → Logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.embed_dim = config.embed_dim

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Causal mask – upper-triangular matrix of -inf
        # Registered as a buffer so it moves with the model to GPU
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.block_size, config.block_size), diagonal=1).bool(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, embed_dim

        # QKV  → (B, T, 3*C) → 3 × (B, T, C)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale                  # (B, H, T, T)
        attn = attn.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum  → (B, H, T, head_dim) → (B, T, C)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = 4 * config.embed_dim  # Standard 4× expansion
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (GPT-2 style)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # Residual around attention
        x = x + self.ffn(self.ln2(x))    # Residual around FFN
        return x


# ---------------------------------------------------------------------------
# Full GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    A small GPT language model.

    Parameters are set via GPTConfig.  Call `model(input_ids)` for logits
    or `model.generate(...)` for autoregressive text generation.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # --- Embeddings ---
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.block_size, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        # --- Transformer blocks ---
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)]
        )

        # --- Output head ---
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying (optional but standard): share embedding and head weights
        self.head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)
        print(f"GPT model initialized – {self.count_parameters() / 1e6:.2f}M parameters")

    # ----- helpers ----------------------------------------------------------

    @staticmethod
    def _init_weights(module: nn.Module):
        """Xavier-style init for linear / embedding layers."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ----- forward ----------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: (B, T) token indices
            targets:   (B, T) target token indices (optional)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar CrossEntropy loss (if targets given), else None
        """
        B, T = input_ids.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Token + positional embeddings
        positions = torch.arange(T, device=input_ids.device)  # (T,)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)  # (B, T, C)
        x = self.drop(x)

        # Transformer
        x = self.blocks(x)
        x = self.ln_f(x)

        # Logits
        logits = self.head(x)  # (B, T, vocab_size)

        # Loss (optional)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    # ----- generation -------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            input_ids:      (B, T) starting token IDs
            max_new_tokens: number of tokens to generate
            temperature:    sampling temperature (1.0 = neutral)
            top_k:          if set, only sample from top-k logits

        Returns:
            (B, T + max_new_tokens) generated token IDs
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = input_ids[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # last position

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids
