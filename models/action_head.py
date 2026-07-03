import torch
import torch.nn as nn


class ActionQueryDecoder(nn.Module):
    """
    Replaces ActionChunkHead from v1.

    Instead of one flat MLP outputting chunk_size * action_dim numbers at once,
    we have chunk_size LEARNABLE queries - one per future timestep - that each
    attend to the fused observation memory and independently predict one action.

    Pipeline:
      1. query_embed  [1, chunk_size, dim_model]  - learnable, shared across batch
      2. expand to    [B, chunk_size, dim_model]
      3. TransformerDecoder: queries (tgt) attend to memory_tokens (memory)
      4. action_out Linear -> [B, chunk_size, action_dim]

    Args:
        dim_model  : must match ObservationEncoder dim_model
        chunk_size : number of future action steps (16)
        action_dim : degrees of freedom per step (7: xyz + rpy + gripper)
        nhead      : attention heads
        num_layers : TransformerDecoder layers

    Unchanged from v2 (memory_tokens shape changes from [B,4,D] to [B,5,D]
    but TransformerDecoder handles variable memory sequence length natively).
    """
    def __init__(
        self,
        dim_model:  int = 256,
        chunk_size: int = 16,
        action_dim: int = 7,
        nhead:      int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.queryEmbed = nn.Parameter(torch.randn(1, chunk_size, dim_model))

        decoderLayer = nn.TransformerDecoderLayer(
            d_model=dim_model, nhead=nhead,
            dim_feedforward=4 * dim_model, dropout=0.1, batch_first=True,
        )
        self.decoder   = nn.TransformerDecoder(decoder_layer=decoderLayer, num_layers=num_layers)
        self.actionProj = nn.Linear(dim_model, action_dim)

    def forward(self, memory_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory_tokens: [B, num_tokens, dim_model]  from ObservationEncoder
                           num_tokens = 5 in v3 (img, wrist, state, text, task)
        Returns:
            pred_action_chunk: [B, chunk_size, action_dim]
        """
        B = memory_tokens.shape[0]
        queries        = self.queryEmbed.expand(B, -1, -1)              # [B, 16, D]
        decodedQueries = self.decoder(tgt=queries, memory=memory_tokens) # [B, 16, D]
        predActionChunk = self.actionProj(decodedQueries)                # [B, 16, 7]
        return predActionChunk
