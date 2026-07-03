# miniVLA v3

Upgrade from [miniVLA_v2](https://github.com/jimmy-h16/miniVLA_v2).

## What's New in v3

| Component | v2 | v3 |
|---|---|---|
| Image Encoder | `SmallImageEncoder` (3-layer CNN + GAP → `[B, D]`) | `ResNetSpatialEncoder` (pretrained ResNet18 → `[B, 16, D]` spatial tokens) |
| Spatial Awareness | None (global average pool loses position) | 4×4 grid with 2D row/col positional encoding |
| Within-camera fusion | N/A | 2-layer TransformerEncoder inside image encoder → pooled `[B, D]` |
| Backbone freeze control | N/A | `freeze_all()` / `unfreeze_last_layer()` / `unfreeze_all()` |
| Fusion module | `TransformerFusion` (4 tokens: img, wrist, state, text) | `ObservationEncoder` (5 tokens: img, wrist, state, text, task) |
| Per-modality embeddings | Single shared `modality_embedding` | Separate `camera_embedding`, `state_embedding`, `text_embedding`, `task_embedding` |
| Task conditioning | None | `TaskEncoder` — learned embedding lookup by stable task index |
| Task index source | N/A | Mapped from `language_instruction` string in HDF5 attrs via `TASK_NAME_TO_IDX` |

## Architecture

```
agentview_rgb   → ResNetSpatialEncoder  →  [B, 16, D] → TransformerEncoder → pool → [B, D] + camera_embed
eye_in_hand_rgb → ResNetSpatialEncoder  →  [B, 16, D] → TransformerEncoder → pool → [B, D] + camera_embed
state           → StateEncoder          →  [B, D]                                          + state_embed
text (CLIP)     → TextEncoder           →  [B, D]                                          + text_embed
task_idx        → TaskEncoder           →  [B, D]                                          + task_embed
                                              ↓
                                   stack → obs_tokens [B, 5, D]
                                              ↓
                                   ObservationEncoder (TransformerEncoder, cross-modal attention)
                                              ↓
                                   memory_tokens [B, 5, D]  +  obs_summary [B, D]
                                              ↓
                               16 learnable action queries
                                   ActionQueryDecoder (TransformerDecoder)
                                              ↓
                                   [B, chunk_size=16, action_dim=7]
```

### ResNet18 Spatial Tokens
The pretrained ResNet18 backbone (ImageNet weights) outputs a `4×4` feature map for `128×128` inputs (stride-32). A 1×1 Conv projects channels to `dim_model`. Each of the 16 spatial patches gets a 2D positional encoding from learnable `row_embed` and `col_embed` (each `dim_model//2`, concatenated). A small 2-layer TransformerEncoder runs within each camera for local spatial attention, then mean pooling produces one rich summary vector per camera.

### ObservationEncoder
Five modality tokens (img, wrist, state, text, task) each receive a dedicated learnable embedding before the joint TransformerEncoder. This replaces v2's `TransformerFusion` and adds the task token slot.

### Task Conditioning
The task index is resolved by mapping the `language_instruction` string (stored in HDF5 `attrs`) to a stable integer via `TASK_NAME_TO_IDX` (built from LIBERO's canonical task lists). This means task 3 is always task 3 regardless of how many tasks you load at train or eval time.

## Setup

```bash
conda env create -f environment.yml
conda activate miniVLA
pip install transformers
```

## Training

```bash
python train3.py
```
