# models/mini_vla.py
# v3: wires ResNetSpatialEncoder + ObservationEncoder + TaskEncoder together.
#
# Signature change vs v2:
#   forward(...) now takes task_idx [B] as an additional argument.

import torch
import torch.nn as nn

from models.image_encoder import ResNetSpatialEncoder
from models.text_encoder  import TextEncoder
from models.state_encoder import StateEncoder
from models.fusion        import ObservationEncoder
from models.task_encoder  import TaskEncoder
from models.action_head   import ActionQueryDecoder


class MiniVLA(nn.Module):
    """
    Mini-VLA v3
    ===========
    Inputs:
        image       [B, 3, H, W]   agentview_rgb       (128×128)
        wrist_image [B, 3, H, W]   eye_in_hand_rgb     (128×128)
        tokens      [B, seq_len]   CLIP token ids
        mask        [B, seq_len]   attention mask
        state       [B, state_dim] proprioceptive state
        task_idx    [B]            stable task index (long)     ← NEW in v3

    Pipeline:
        image       → image_encoder  (ResNetSpatialEncoder)  [B, D]
        wrist_image → wrist_encoder  (ResNetSpatialEncoder)  [B, D]   (independent weights)
        state       → state_encoder  (StateEncoder)          [B, D]
        tokens+mask → text_encoder   (TextEncoder)           [B, D]
        task_idx    → task_encoder   (TaskEncoder)           [B, D]   NEW
                         ↓
              ObservationEncoder  →  memory_tokens [B, 5, D]
                         ↓
             ActionQueryDecoder  →  [B, chunk_size, action_dim]
    """
    def __init__(
        self,
        dim_model:      int = 256,
        vocab_size:     int = 49408,    # CLIP BPE vocab
        state_dim:      int = 8,
        chunk_size:     int = 16,
        action_dim:     int = 7,        # xyz + rpy + gripper
        nhead:          int = 4,
        fusion_layers:  int = 2,
        decoder_layers: int = 2,
        num_tasks:      int = 10,       # NEW: number of tasks in your training suite
        pretrained_backbone: bool = True,  # NEW: ImageNet init for ResNet18
    ):
        super().__init__()

        # --- Image encoders (pretrained ResNet18, independent weights) ---
        # TODO 1: Replace SmallImageEncoder with ResNetSpatialEncoder.
        # Same pattern as v2 — two instances, different weights.
        self.image_encoder = ResNetSpatialEncoder(embed_dim=dim_model, pretrained=pretrained_backbone)
        self.wrist_encoder = ResNetSpatialEncoder(embed_dim=dim_model, pretrained=pretrained_backbone)
        # raise NotImplementedError("TODO 1: instantiate image_encoder and wrist_encoder")

        # --- Other encoders (unchanged from v2) ---
        # TODO 2: Instantiate state_encoder and text_encoder exactly as in v2.
        self.state_encoder = StateEncoder(state_dim=state_dim, embed_dim=dim_model)
        self.text_encoder  = TextEncoder(vocab_size=vocab_size, embed_dim=dim_model)
        # raise NotImplementedError("TODO 2: instantiate state_encoder and text_encoder")

        # --- Task encoder (NEW in v3) ---
        # TODO 3: Instantiate task_encoder.
        self.task_encoder = TaskEncoder(num_tasks=num_tasks, embed_dim=dim_model)
        # raise NotImplementedError("TODO 3: instantiate task_encoder")

        # --- Observation encoder (replaces TransformerFusion) ---
        # TODO 4: Instantiate ObservationEncoder.
        # Note: no num_tokens argument needed — ObservationEncoder always uses 5 tokens.
        self.obs_encoder = ObservationEncoder(
            dim_model=dim_model, nhead=nhead, num_layers=fusion_layers,
        )
        # raise NotImplementedError("TODO 4: instantiate obs_encoder")

        # --- Action head (unchanged from v2) ---
        # TODO 5: Instantiate action_head exactly as in v2.
        self.action_head = ActionQueryDecoder(
            dim_model=dim_model, chunk_size=chunk_size,
            action_dim=action_dim, nhead=nhead, num_layers=decoder_layers,
        )
        # raise NotImplementedError("TODO 5: instantiate action_head")

    def forward(
        self,
        image:       torch.Tensor,  # [B, 3, H, W]  agentview
        wrist_image: torch.Tensor,  # [B, 3, H, W]  eye_in_hand
        tokens:      torch.Tensor,  # [B, seq_len]  CLIP ids
        mask:        torch.Tensor,  # [B, seq_len]  attention mask
        state:       torch.Tensor,  # [B, state_dim]
        task_idx:    torch.Tensor,  # [B]           long  ← NEW in v3
    ) -> torch.Tensor:
        """
        Returns:
            actions : [B, chunk_size, action_dim]

        TODO 6: Implement forward — mirrors v2 but with task_idx.
        Steps:
          A. img_feat   = self.image_encoder(image)           # [B, D]
          B. wrist_feat = self.wrist_encoder(wrist_image)     # [B, D]
          C. state_feat = self.state_encoder(state)           # [B, D]
          D. txt_feat   = self.text_encoder(tokens, mask)     # [B, D]
          E. task_feat  = self.task_encoder(task_idx)         # [B, D]  NEW
          F. memory_tokens, _ = self.obs_encoder(
                 img_feat, wrist_feat, state_feat, txt_feat, task_feat
             )                                                # [B, 5, D]
          G. actions = self.action_head(memory_tokens)        # [B, 16, 7]
          H. return actions
        """
        # raise NotImplementedError("TODO 6: implement MiniVLA.forward")
        img_feat   = self.image_encoder(image)           # [B, D]
        wrist_feat = self.wrist_encoder(wrist_image)     # [B, D]
        state_feat = self.state_encoder(state)           # [B, D]
        txt_feat   = self.text_encoder(tokens, mask)     # [B, D]
        task_feat  = self.task_encoder(task_idx)         # [B, D]  NEW
        memory_tokens, _ = self.obs_encoder(
            img_feat, wrist_feat, state_feat, txt_feat, task_feat
        )                                                # [B, 5, D]
        actions = self.action_head(memory_tokens)        # [B, 16, 7]
        return actions

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_backbone_stage(self, stage: int) -> None:
        """
        Convenience method to apply freeze stage to BOTH image encoders at once.

        stage 1 — freeze backbone entirely   (start of training)
        stage 2 — unfreeze layer4 only       (after warmup)
        stage 3 — unfreeze full backbone     (final fine-tuning, use lower LR)

        Usage in train3.py:
            model.set_backbone_stage(1)   # before training loop
            # ... after N epochs ...
            model.set_backbone_stage(2)
        """
        assert stage in (1, 2, 3), "stage must be 1, 2, or 3"
        for enc in (self.image_encoder, self.wrist_encoder):
            if stage == 1:
                enc.freeze_all()
            elif stage == 2:
                enc.unfreeze_last_layer()
            else:
                enc.unfreeze_all()
