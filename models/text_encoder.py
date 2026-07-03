import torch
import torch.nn as nn


class MaskedMeanPooling(nn.Module):
    """
    Averages only real (non-padding) token embeddings.
    Input:  token_embeddings [B, seq_len, embed_dim]
            mask             [B, seq_len]  — 1=real, 0=padding
    Output: [B, embed_dim]

    Unchanged from v2.
    """
    def forward(self, token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask         = mask.unsqueeze(-1)                   # [B, seq_len, 1]
        masked_emb   = token_embeddings * mask              # [B, seq_len, D]
        sum_emb      = masked_emb.sum(dim=1)                # [B, D]
        token_counts = mask.sum(dim=1)                      # [B, 1]
        return sum_emb / (token_counts + 1e-8)              # [B, D]


class TextEncoder(nn.Module):
    """
    Embedding + masked mean pooling + Linear projection.

    v2 change: vocab_size upgraded from 1000 (char-level) to 49408 (CLIP BPE).
    The tokenizer lives in data/libero_dataset.py — this module only holds
    the learnable embedding table and projection.

    Input:  tokens [B, seq_len]  (CLIP token ids, long)
            mask   [B, seq_len]  (1=real, 0=pad, float)
    Output: [B, embed_dim]

    Unchanged from v2.
    """
    def __init__(self, vocab_size: int = 49408, embed_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pool      = MaskedMeanPooling()
        self.proj      = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens.long())  # [B, seq_len, embed_dim]
        x = self.pool(x, mask)             # [B, embed_dim]
        x = self.proj(x)                   # [B, embed_dim]
        return x
