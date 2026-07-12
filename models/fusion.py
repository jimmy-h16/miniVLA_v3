# models/fusion.py
# v3 upgrade: replace TransformerFusion (4 tokens, shared modality_embedding)
#             with ObservationEncoder (5 tokens, per-modality embeddings)
#
# KEY CHANGES vs v2:
#   1. Renamed TransformerFusion → ObservationEncoder
#   2. 4 tokens → 5 tokens: added task_token slot
#   3. Shared modality_embedding → separate per-modality embeddings
#      (camera_embedding, state_embedding, text_embedding, task_embedding)
#      Each camera gets the SAME camera_embedding (agentview and wrist are
#      both “camera” type — their visual difference is already encoded by
#      their independent ResNetSpatialEncoder weights)
#   4. forward() signature adds task_feat argument

import torch
import torch.nn as nn


# ── v2 reference — kept for diff clarity, not used by MiniVLA v3 ───────────
class TransformerFusion(nn.Module):
    """v2 fusion module. Kept for reference only. Replaced by ObservationEncoder."""
    pass


# ── NEW in v3 ───────────────────────────────────────────────────────────────
class ObservationEncoder(nn.Module):
    """
    Cross-modal fusion of 5 observation tokens via TransformerEncoder.

    Replaces TransformerFusion from v2.

    Each modality vector [B, D] receives a dedicated learnable embedding
    (its “identity badge”), then all 5 tokens are stacked and fused via
    self-attention. This lets image, wrist, state, text, and task tokens
    all attend to each other in a single joint representation.

    Pipeline:
      1. Add per-modality embeddings to each input vector
             img_feat   += camera_embedding    [D]
             wrist_feat += camera_embedding    [D]   (same embedding type — both cameras)
             state_feat += state_embedding     [D]
             txt_feat   += text_embedding      [D]
             task_feat  += task_embedding      [D]
      2. Stack 5 tokens                        [B, 5, D]
      3. TransformerEncoder (self-attention)   [B, 5, D]   ← memory_tokens
      4. Mean pool across 5 tokens             [B, D]
      5. out_proj (Linear + LayerNorm)         [B, D]      ← obs_summary

    forward() returns (memory_tokens, obs_summary) — same interface as v2
    TransformerFusion so ActionQueryDecoder is unchanged.

    Args:
        dim_model  : embedding dimension (must match all encoders)
        nhead      : attention heads for TransformerEncoder
        num_layers : number of TransformerEncoder layers
    """
    NUM_TOKENS: int = 5  # img, wrist, state, text, task

    def __init__(
        self,
        dim_model:  int = 256,
        nhead:      int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        # ── TODO 1: Per-modality learnable embeddings ─────────────────────────
        # Each is an nn.Parameter of shape [dim_model] (NOT [1, 1, dim_model] —
        # broadcasting is handled in forward via .unsqueeze(0)).
        #
        # Note: agentview and wrist share ONE camera_embedding (same modality type).
        # State, text, and task each have their own.
        #
        self.camera_embedding = nn.Parameter(torch.randn(dim_model))
        self.state_embedding  = nn.Parameter(torch.randn(dim_model))
        self.text_embedding   = nn.Parameter(torch.randn(dim_model))
        self.task_embedding   = nn.Parameter(torch.randn(dim_model))
        # raise NotImplementedError("TODO 1: define four per-modality embedding parameters")

        # ── TODO 2: TransformerEncoder for cross-modal attention ──────────────
        # Same structure as v2 TransformerFusion — just handles 5 tokens instead of 4.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model, nhead=nhead,
            dim_feedforward=4*dim_model, dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # raise NotImplementedError("TODO 2: define self.encoder")

        # ── TODO 3: Output projection (same as v2) ────────────────────────────
        # Linear + LayerNorm applied to the mean-pooled summary.
        self.out_proj = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.LayerNorm(dim_model),
        )
        # raise NotImplementedError("TODO 3: define self.out_proj")

    def forward(
        self,
        img_feat:   torch.Tensor,   # [B, dim_model]  agentview  — from ResNetSpatialEncoder
        wrist_feat: torch.Tensor,   # [B, dim_model]  eye_in_hand — from ResNetSpatialEncoder
        state_feat: torch.Tensor,   # [B, dim_model]  — from StateEncoder
        txt_feat:   torch.Tensor,   # [B, dim_model]  — from TextEncoder
        task_feat:  torch.Tensor,   # [B, dim_model]  — from TaskEncoder  (NEW in v3)
    ):
        """
        Returns:
            memory_tokens : [B, 5, dim_model]  — per-token representations (for ActionQueryDecoder)
            obs_summary   : [B, dim_model]     — single fused observation vector

        TODO 4: Implement forward.
        Steps:
          A. Add modality embeddings (broadcast [D] over batch with .unsqueeze(0)):
               img_feat   = img_feat   + self.camera_embedding.unsqueeze(0)  # [B, D]
               wrist_feat = wrist_feat + self.camera_embedding.unsqueeze(0)
               state_feat = state_feat + self.state_embedding.unsqueeze(0)
               txt_feat   = txt_feat   + self.text_embedding.unsqueeze(0)
               task_feat  = task_feat  + self.task_embedding.unsqueeze(0)

          B. Stack into token sequence:
               tokens = torch.stack([img_feat, wrist_feat, state_feat, txt_feat, task_feat], dim=1)
               # → [B, 5, D]

          C. Run TransformerEncoder:
               memory_tokens = self.encoder(tokens)  # [B, 5, D]

          D. Mean pool + out_proj for summary:
               fused = memory_tokens.mean(dim=1)     # [B, D]
               obs_summary = self.out_proj(fused)    # [B, D]

          E. Return (memory_tokens, obs_summary)
        """
        # raise NotImplementedError("TODO 4: implement ObservationEncoder.forward")
        img_feat   = img_feat   + self.camera_embedding.unsqueeze(0)  # [B, D]
        wrist_feat = wrist_feat + self.camera_embedding.unsqueeze(0)
        state_feat = state_feat + self.state_embedding.unsqueeze(0)
        txt_feat   = txt_feat   + self.text_embedding.unsqueeze(0)
        task_feat  = task_feat  + self.task_embedding.unsqueeze(0)
        
        tokens = torch.stack([img_feat, wrist_feat, state_feat, txt_feat, task_feat], dim=1)
        
        memory_tokens = self.encoder(tokens)
        
        fused = memory_tokens.mean(dim=1)
        obs_summary = self.out_proj(fused)
        
        return memory_tokens, obs_summary