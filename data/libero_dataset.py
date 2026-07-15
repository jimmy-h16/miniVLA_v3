# data/libero_dataset.py
# v3 upgrade: stable task index via TASK_NAME_TO_IDX (string lookup, not list position)
# v3 text upgrade: tokenize the real HDF5 language instruction once per file.
#
# KEY CHANGE vs v2:
#   v2: task_idx = position of this file in the local hdf5_files list
#       → breaks when you train on 10 tasks but evaluate on 1 (position changes!)
#   v3: task_idx = TASK_NAME_TO_IDX[language_instruction from HDF5 attrs]
#       → always stable; task 3 is task 3 regardless of how many files you load
#
# TEXT CONDITIONING:
#   The language instruction is stored in HDF5 problem_info. We tokenize it once
#   during __init__, cache input_ids / attention_mask per HDF5 file, and return
#   them for every timestep sample. This replaces zero-padded dummy tokens in
#   train3.py.
#
# All other logic (chunking, masking, observation loading) is unchanged from v2.

import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import CLIPTokenizer

from data.task_map import get_task_idx


# Preprocessing expected by the ImageNet-pretrained ResNet18 backbone.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LiberoDataset(Dataset):
    """
    Loads LIBERO HDF5 demonstration files into
    (observation, action_chunk, task_idx, text tokens) tuples.

    Observations returned per sample:
        agentview_rgb   : [3, H, W]  float32, ImageNet-normalised
        eye_in_hand_rgb : [3, H, W]  float32, ImageNet-normalised
        robot_state     : [state_dim] float32

    Text returned per sample:
        tokens          : [seq_len]  int64 CLIP input ids
        text_mask       : [seq_len]  float32 CLIP attention mask

    Actions returned:
        action_chunk    : [chunk_size, action_dim]  float32
        action_mask     : [chunk_size]               float32 (1=valid, 0=padding)

    Task:
        task_idx        : int — stable index from TASK_NAME_TO_IDX,
                                independent of local file list order

    Args:
        hdf5_files  : list of paths to .hdf5 demo files (any subset of the suite)
        chunk_size  : number of future actions per sample (default 16)
        image_size  : retained for interface compatibility (default 128)
        tokenizer_name : Hugging Face CLIP tokenizer identifier
        max_text_len : fixed CLIP sequence length (default 77)
    """

    def __init__(
        self,
        hdf5_files: list[str],
        chunk_size: int = 16,
        image_size: int = 128,
        tokenizer_name: str = "openai/clip-vit-base-patch32",
        max_text_len: int = 77,
    ):
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.max_text_len = max_text_len

        # Tokenizer is constructed once per Dataset, never per sample.
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)

        # Build flat index:
        # (hdf5_path, episode_key, start_step, task_idx, tokens, text_mask)
        self.samples: list[
            tuple[str, str, int, int, torch.Tensor, torch.Tensor]
        ] = []

        for hdf5_path in hdf5_files:
            with h5py.File(hdf5_path, "r") as f:
                # Read the real task language from the HDF5 metadata.
                # problem_info is JSON: {"language_instruction": "...", ...}
                problem_info = json.loads(f["data"].attrs["problem_info"])
                instruction = problem_info["language_instruction"].strip()

                # Stable categorical task identity for TaskEncoder.
                task_idx = get_task_idx(instruction)

                # Real language conditioning for TextEncoder.
                # Cache once per HDF5 file because every sample from this file
                # shares the same instruction.
                enc = self.tokenizer(
                    instruction,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_text_len,
                    return_tensors="pt",
                )
                tokens = enc["input_ids"].squeeze(0).long()               # [77]
                text_mask = enc["attention_mask"].squeeze(0).float()     # [77]

                for ep_key in f["data"].keys():
                    T = f["data"][ep_key]["actions"].shape[0]
                    for t in range(T):
                        self.samples.append(
                            (
                                hdf5_path,
                                ep_key,
                                t,
                                task_idx,
                                tokens,
                                text_mask,
                            )
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        (
            hdf5_path,
            ep_key,
            t,
            task_idx,
            tokens,
            text_mask,
        ) = self.samples[idx]

        with h5py.File(hdf5_path, "r") as f:
            ep = f["data"][ep_key]
            T = ep["actions"].shape[0]

            # ── Observations at step t ─────────────────────────────────────
            def load_image(key: str) -> torch.Tensor:
                """Load RGB image and apply ImageNet normalisation in CHW layout."""
                img = ep["obs"][key][t]  # [H, W, 3], uint8
                img = img.astype(np.float32) / 255.0
                img = (img - IMAGENET_MEAN) / IMAGENET_STD
                return torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]

            agentview = load_image("agentview_rgb")
            eye_in_hand = load_image("eye_in_hand_rgb")

            eef_pos = ep["obs"]["ee_pos"][t]                    # [3]
            eef_ori = ep["obs"]["ee_ori"][t]                    # [3]
            gripper = ep["obs"]["gripper_states"][t]            # [2]
            state = np.concatenate([eef_pos, eef_ori, gripper]) # [8]
            state = torch.from_numpy(state).float()

            # ── Action chunk [t : t + chunk_size] ──────────────────────────
            end = min(t + self.chunk_size, T)
            raw_chunk = ep["actions"][t:end].astype(np.float32)  # [L, 7]
            pad_len = self.chunk_size - raw_chunk.shape[0]

            if pad_len > 0:
                padding = np.zeros(
                    (pad_len, raw_chunk.shape[1]),
                    dtype=np.float32,
                )
                raw_chunk = np.concatenate([raw_chunk, padding], axis=0)

            action_chunk = torch.from_numpy(raw_chunk)  # [chunk_size, 7]

            # Mask: 1 for real actions, 0 for end-of-trajectory padding.
            mask_vals = [1.0] * (end - t) + [0.0] * pad_len
            action_mask = torch.tensor(mask_vals, dtype=torch.float32)

        return {
            "agentview_rgb": agentview,          # [3, H, W], float32
            "eye_in_hand_rgb": eye_in_hand,      # [3, H, W], float32
            "robot_state": state,                # [8], float32
            "tokens": tokens.clone(),            # [77], int64
            "text_mask": text_mask.clone(),      # [77], float32
            "action_chunk": action_chunk,        # [chunk_size, 7], float32
            "action_mask": action_mask,          # [chunk_size], float32
            "task_idx": task_idx,                # int, stable suite index
        }
