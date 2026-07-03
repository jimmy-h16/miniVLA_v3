# data/libero_dataset.py
# v3 upgrade: stable task index via TASK_NAME_TO_IDX (string lookup, not list position)
#
# KEY CHANGE vs v2:
#   v2: task_idx = position of this file in the local hdf5_files list
#       → breaks when you train on 10 tasks but evaluate on 1 (position changes!)
#   v3: task_idx = TASK_NAME_TO_IDX[language_instruction from HDF5 attrs]
#       → always stable; task 3 is task 3 regardless of how many files you load
#
# All other logic (chunking, masking, observation loading) is unchanged from v2.

import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from data.task_map import get_task_idx


class LiberoDataset(Dataset):
    """
    Loads LIBERO HDF5 demonstration files into (observation, action_chunk, task_idx) tuples.

    Observations returned per sample:
        agentview_rgb   : [3, H, W]  float32, normalised to [0, 1]
        eye_in_hand_rgb : [3, H, W]  float32, normalised to [0, 1]
        robot_state     : [state_dim] float32

    Actions returned:
        action_chunk    : [chunk_size, action_dim]  float32
        action_mask     : [chunk_size]               float32 (1=valid, 0=padding)

    Task:
        task_idx        : int  — stable index from TASK_NAME_TO_IDX,
                                 independent of local file list order

    Args:
        hdf5_files  : list of paths to .hdf5 demo files (any subset of the suite)
        chunk_size  : number of future actions per sample (default 16)
        image_size  : spatial size to resize images to (default 128)
    """

    def __init__(
        self,
        hdf5_files: list[str],
        chunk_size: int = 16,
        image_size: int = 128,
    ):
        self.chunk_size = chunk_size
        self.image_size = image_size

        # Build flat index: list of (hdf5_path, episode_key, start_step, task_idx)
        self.samples: list[tuple[str, str, int, int]] = []

        for hdf5_path in hdf5_files:
            with h5py.File(hdf5_path, "r") as f:
                # ── v3: read task instruction from HDF5 attrs (stable string lookup) ──
                # language_instruction is stored in f["data"].attrs["problem_info"]
                # as a JSON string: {"language_instruction": "...", ...}
                problem_info = json.loads(f["data"].attrs["problem_info"])
                instruction  = problem_info["language_instruction"]
                task_idx     = get_task_idx(instruction)  # raises KeyError if not in map

                for ep_key in f["data"].keys():
                    T = f["data"][ep_key]["actions"].shape[0]
                    for t in range(T):
                        self.samples.append((hdf5_path, ep_key, t, task_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        hdf5_path, ep_key, t, task_idx = self.samples[idx]

        with h5py.File(hdf5_path, "r") as f:
            ep = f["data"][ep_key]
            T  = ep["actions"].shape[0]

            # ── Observations at step t ─────────────────────────────────────
            def load_image(key: str) -> torch.Tensor:
                """Load image, normalise to [0,1], CHW layout."""
                img = ep["obs"][key][t]          # [H, W, 3]  uint8
                img = img.astype(np.float32) / 255.0
                return torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]

            agentview   = load_image("agentview_rgb")
            eye_in_hand = load_image("eye_in_hand_rgb")

            robot_state = torch.from_numpy(
                ep["obs"]["robot0_eef_pos"][t].astype(np.float32)
            )  # shape depends on state_dim in your dataset

            # ── Action chunk [t : t + chunk_size] ─────────────────────────
            end = min(t + self.chunk_size, T)
            raw_chunk = ep["actions"][t:end].astype(np.float32)    # [L, action_dim]
            pad_len   = self.chunk_size - raw_chunk.shape[0]

            if pad_len > 0:
                padding   = np.zeros((pad_len, raw_chunk.shape[1]), dtype=np.float32)
                raw_chunk = np.concatenate([raw_chunk, padding], axis=0)

            action_chunk = torch.from_numpy(raw_chunk)            # [chunk_size, action_dim]

            # Mask: 1 for real steps, 0 for padding
            mask_vals    = [1.0] * (end - t) + [0.0] * pad_len
            action_mask  = torch.tensor(mask_vals, dtype=torch.float32)

        return {
            "agentview_rgb":   agentview,    # [3, H, W]
            "eye_in_hand_rgb": eye_in_hand,  # [3, H, W]
            "robot_state":     robot_state,  # [state_dim]
            "action_chunk":    action_chunk, # [chunk_size, action_dim]
            "action_mask":     action_mask,  # [chunk_size]
            "task_idx":        task_idx,     # int  (stable, suite-based)
        }
