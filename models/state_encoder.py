import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """
    2-layer MLP for robot proprioceptive state encoding.
    Input:  [B, state_dim]   (ee_pos[3] + ee_ori[3] + gripper[2] = 8)
    Output: [B, embed_dim]

    Unchanged from v2.
    """
    def __init__(self, state_dim: int = 8, embed_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
