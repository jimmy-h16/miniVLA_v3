# models/task_encoder.py
# NEW in v3: encodes a task index (stable integer) as a learned embedding vector.
#
# Why task index instead of task string?
#   The task string is already encoded by TextEncoder. TaskEncoder provides a
#   SEPARATE, compact conditioning signal so the model can learn task-specific
#   behaviour beyond what the language captures (e.g. task-specific action style).
#
# Why is the index STABLE?
#   We map language_instruction → int via TASK_NAME_TO_IDX (see data/libero_dataset.py).
#   This dict is built from LIBERO’s canonical task list, so task 3 is always task 3
#   regardless of how many tasks you load at train or eval time.

import torch
import torch.nn as nn


class TaskEncoder(nn.Module):
    """
    Learnable embedding lookup: task_idx (int) → task token [B, embed_dim].

    This is intentionally the simplest possible task encoder — one embedding
    row per task, trained end-to-end. No language model, no hashing.

    Args:
        num_tasks : total number of tasks in your training suite
                    (e.g. 10 for libero_10, 90 for libero_90)
                    MUST cover every task_idx seen at train AND eval time.
        embed_dim : must match dim_model everywhere else

    Input:  task_idx [B]          long tensor, values in [0, num_tasks)
    Output: task_feat [B, embed_dim]
    """
    def __init__(self, num_tasks: int = 10, embed_dim: int = 256):
        super().__init__()

        # TODO 1: Define the embedding table
        # One embedding vector per task, dimension = embed_dim.
        # Hint: nn.Embedding(num_tasks, embed_dim)
        # self.embedding = ...
        raise NotImplementedError("TaskEncoder.__init__: define self.embedding")

        # TODO 2: Define an output projection (optional but consistent with other encoders)
        # A single Linear layer: embed_dim → embed_dim
        # This gives the task token its own learnable projection like state/text encoders.
        # self.proj = ...
        raise NotImplementedError("TaskEncoder.__init__: define self.proj")

    def forward(self, task_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_idx : [B]  long tensor — each element is a task index
        Returns:
            task_feat : [B, embed_dim]
        """
        # TODO 3: Forward pass
        # Step A: look up embedding for each task_idx  → [B, embed_dim]
        # Step B: pass through self.proj              → [B, embed_dim]
        # Hint: self.embedding(task_idx.long())
        raise NotImplementedError("TaskEncoder.forward: implement lookup + projection")
