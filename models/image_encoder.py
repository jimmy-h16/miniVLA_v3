# models/image_encoder.py
# v3 upgrade: replace SmallImageEncoder (3-layer CNN + GAP) with
#             ResNetSpatialEncoder (pretrained ResNet18 + spatial tokens + 2D pos enc
#             + within-camera TransformerEncoder + mean pool)
#
# OUTPUT SHAPE CHANGE:
#   v2: SmallImageEncoder  → [B, D]          (one global vector per camera)
#   v3: ResNetSpatialEncoder → [B, D]         (one pooled spatial-aware vector per camera)
#
# The output shape is STILL [B, D], so ObservationEncoder (fusion.py) is unchanged
# in its expected per-camera input. Spatial understanding is fully contained here.

import torch
import torch.nn as nn
from torchvision import models


# ── v2 reference — kept for diff clarity, not used by MiniVLA v3 ───────────
class SmallImageEncoder(nn.Module):
    """
    v2 CNN encoder. Kept for reference only.
    Replaced by ResNetSpatialEncoder in v3.
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.flat(self.gap(self.cnn(x))))


# ── NEW in v3 ───────────────────────────────────────────────────────────────
class ResNetSpatialEncoder(nn.Module):
    """
    Pretrained ResNet18 backbone → spatial tokens → within-camera attention → pooled [B, D].

    Full pipeline for a 128×128 input image:

        image [B, 3, 128, 128]
          ↓  ResNet18 backbone (conv1…layer4, no avgpool/fc)   stride=32
        feat_map  [B, 512, 4, 4]          ← 4×4 spatial grid = 16 patches
          ↓  1×1 Conv projection
        projected [B, dim_model, 4, 4]
          ↓  flatten spatial dims + permute
        tokens    [B, 16, dim_model]       ← 16 spatial tokens
          ↓  add 2D positional encoding (row_embed [4, D/2] + col_embed [4, D/2])
        tokens_pe [B, 16, dim_model]
          ↓  LayerNorm
          ↓  within-camera TransformerEncoder (local spatial attention, 2 layers)
        attended  [B, 16, dim_model]
          ↓  mean pool across 16 tokens
        output    [B, dim_model]           ← one rich spatial-aware vector

    Freeze control (standard pretrained backbone fine-tuning protocol):
        Stage 1 — freeze_all()          : only proj, pos_enc, norm, spatial_encoder train
        Stage 2 — unfreeze_last_layer() : backbone.layer4 also unfreezes
        Stage 3 — unfreeze_all()        : full backbone fine-tuning (use low LR)

    Args:
        embed_dim  : output dimension, must match dim_model everywhere
        pretrained : load ImageNet weights (True for training, False for unit tests)
    """
    # Spatial grid size for 128×128 input through ResNet18 (stride=32)
    H_FEAT: int = 4
    W_FEAT: int = 4
    RESNET_CHANNELS: int = 512  # ResNet18 layer4 output channels

    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.embed_dim = embed_dim

        # ── TODO 1: Build self.backbone from pretrained ResNet18 ──────────────
        # Goal: keep conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4.
        # Strip avgpool and fc (we don’t want global pooling — we keep spatial dims).
        #
        # Pattern:
        #   base = models.resnet18(pretrained=pretrained)
        #   self.backbone = nn.Sequential(
        #       base.conv1, base.bn1, base.relu, base.maxpool,
        #       base.layer1, base.layer2, base.layer3, base.layer4,
        #   )
        #
        # Verify: backbone(torch.zeros(1,3,128,128)).shape  → should be [1, 512, 4, 4]
        # raise NotImplementedError("TODO 1: build self.backbone")
        
        base = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4, 
        )
        #layer1,2,3,4 such module network build with sequential
        # [B, 512, 4, 4]

        print(self.backbone(torch.zeros(1,3,128,128)).shape)
        # ── TODO 2: 1×1 Conv projection: 512 → embed_dim ─────────────────────
        # Maps ResNet’s 512-channel feature map to the model’s working dimension.
        # Hint: nn.Conv2d(self.RESNET_CHANNELS, embed_dim, kernel_size=1)
        # raise NotImplementedError("TODO 2: define self.proj")
        
        
        self.proj = nn.Conv2d(self.RESNET_CHANNELS, embed_dim, kernel_size=1)
        # [B, 256, 4, 4]

        # ── TODO 3: Learnable 2D positional encoding ──────────────────────────
        # Each of the 4×4=16 spatial patches needs a unique position signal.
        # We use SEPARATE row and column embeddings (each embed_dim//2),
        # then concatenate to form the full embed_dim position vector.
        #
        #   self.row_embed = nn.Parameter(torch.randn(self.H_FEAT, embed_dim // 2))
        #   self.col_embed = nn.Parameter(torch.randn(self.W_FEAT, embed_dim // 2))
        #
        # In forward() you will combine them like this:
        #   row part: self.row_embed.unsqueeze(1).expand(H, W, D/2)  → [4, 4, D/2]
        #   col part: self.col_embed.unsqueeze(0).expand(H, W, D/2)  → [4, 4, D/2]
        #   concat on dim=-1  → [4, 4, D]  then reshape to [16, D]
        #   unsqueeze(0)      → [1, 16, D]  (broadcasts over batch)
        # raise NotImplementedError("TODO 3: define self.row_embed and self.col_embed")
        
        self.row_embed = nn.Parameter(torch.randn(self.H_FEAT, embed_dim // 2))
        self.col_embed = nn.Parameter(torch.randn(self.W_FEAT, embed_dim // 2))


        # ── TODO 4: LayerNorm on token sequence ──────────────────────────────
        # Applied after adding positional encoding, before spatial attention.
        # Hint: nn.LayerNorm(embed_dim)
        # raise NotImplementedError("TODO 4: define self.norm")
        self.norm = nn.LayerNorm(embed_dim)
        
        # ── TODO 5: Within-camera TransformerEncoder (local spatial attention) 
        # This lets the model understand relationships WITHIN one camera view
        # (e.g. gripper relative to bowl) before tokens enter ObservationEncoder.
        # Use 2 layers, nhead=4, dim_feedforward=4*embed_dim, batch_first=True.
        #
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4,
            dim_feedforward=4*embed_dim, dropout=0.1, batch_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=2)
        
        # raise NotImplementedError("TODO 5: define self.spatial_encoder")

    def _build_pos_encoding(self) -> torch.Tensor:
        """
        Build the 2D positional encoding tensor from row_embed and col_embed.

        Returns:
            pos : [1, H_FEAT*W_FEAT, embed_dim]  — ready to add to token sequence

        TODO 6: Implement this helper.
        Steps:
          A. Expand row_embed [H, D/2] → [H, W, D/2]  using .unsqueeze(1).expand(-1, W, -1)
          B. Expand col_embed [W, D/2] → [H, W, D/2]  using .unsqueeze(0).expand(H, -1, -1)
          C. torch.cat([row_part, col_part], dim=-1)   → [H, W, D]
          D. .reshape(H*W, D).unsqueeze(0)             → [1, H*W, D]
        """
        # raise NotImplementedError("TODO 6: implement _build_pos_encoding")
        
        row_embed = self.row_embed.unsqueeze(1).expand(-1, self.W_FEAT, -1)
        col_embed = self.col_embed.unsqueeze(0).expand(self.H_FEAT, -1, -1)
        concatFeature = torch.cat([row_embed,col_embed], dim=-1)
        concatFeature = concatFeature.reshape(1,self.H_FEAT * self.W_FEAT,self.embed_dim,)
        return concatFeature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, 128, 128]  normalised image
                (if using pretrained=True, normalise with ImageNet mean/std
                 mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] before passing in)
        Returns:
            feat : [B, embed_dim]  spatial-aware image summary vector

        TODO 7: Implement the forward pass.
        Steps:
          A. feat_map = self.backbone(x)                      # [B, 512, 4, 4]
          B. proj     = self.proj(feat_map)                   # [B, D, 4, 4]
          C. tokens   = proj.flatten(2).permute(0, 2, 1)     # [B, 16, D]
                        ^ flatten H and W dims, then move channel to last
          D. pos      = self._build_pos_encoding()            # [1, 16, D]
          E. tokens   = self.norm(tokens + pos)               # [B, 16, D]
          F. attended = self.spatial_encoder(tokens)          # [B, 16, D]
          G. return attended.mean(dim=1)                      # [B, D]  mean pool
        """
        # raise NotImplementedError("TODO 7: implement forward")
        
        featureMap = self.backbone(x)
        projectedFeature = self.proj(featureMap)
        tokens = projectedFeature.flatten(2).permute(0, 2, 1)
        pos = self._build_pos_encoding() 
        tokens   = self.norm(tokens + pos)  
        attended = self.spatial_encoder(tokens)  
        return attended.mean(dim=1)
    
    # ── Freeze control ───────────────────────────────────────────────────────

    def freeze_all(self) -> None:
        """
        TODO 8: Freeze the entire ResNet backbone.
        Only self.backbone parameters are frozen —
        proj, row_embed, col_embed, norm, spatial_encoder remain trainable.

        Call this at the START of training.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False


    def unfreeze_last_layer(self) -> None:
        """
        TODO 9: Unfreeze backbone.layer4 only (the last residual block).
        First freeze everything via freeze_all(), then selectively unfreeze layer4.

        Call this after a few warmup epochs (Stage 2 of training).
        """
        self.freeze_all()
        for p in self.backbone[-1].parameters():  # backbone[-1] is layer4
            p.requires_grad = True


    def unfreeze_all(self) -> None:
        """
        TODO 10: Unfreeze the entire backbone for full fine-tuning.
        Use with a reduced LR (e.g. 1e-5) to avoid destroying pretrained features.

        Call this in the final training stage (Stage 3).
        """
        for p in self.backbone.parameters():
            p.requires_grad = True
