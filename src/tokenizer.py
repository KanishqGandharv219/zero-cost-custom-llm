"""
Tokenizer Wrapper
-----------------
Wraps tiktoken (GPT-2 BPE) for encoding / decoding text.
Later, this module can be extended to train a custom BPE tokenizer
on our Indian-language corpus using HuggingFace `tokenizers`.
"""

import tiktoken


class Tokenizer:
    """
    Thin wrapper around a tiktoken encoding.

    Usage:
        tok = Tokenizer()              # defaults to GPT-2 encoding
        ids = tok.encode("Hello!")     # list[int]
        text = tok.decode(ids)         # str
    """

    def __init__(self, encoding_name: str = "gpt2"):
        """
        Args:
            encoding_name: Name of the tiktoken encoding to load.
                           Use "gpt2" for GPT-2's byte-level BPE.
        """
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs."""
        return self.enc.encode_ordinary(text)

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back to text."""
        return self.enc.decode(token_ids)

    def __len__(self) -> int:
        return self.vocab_size


# -- Future: Custom BPE Training -----------------------------------------------
# To train your own tokenizer on Indic/English text:
#
#   from tokenizers import Tokenizer as HFTokenizer, models, trainers, pre_tokenizers
#
#   tokenizer = HFTokenizer(models.BPE())
#   tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
#   trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=["
