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

## Lessons Learned and Reflection

### 1. Spatial positional embeddings

A standard categorical embedding learns one independent vector for every
category. If the 16 cells of a `4×4` feature map were treated as unrelated
categories, the positional embedding would require `16 × 256 = 4,096`
parameters for an embedding dimension of 256.

This project instead learns separate row and column embeddings. Each row and
column has a 128-dimensional vector, and the two vectors are concatenated at
each spatial location:

```text
position(row=i, col=j) = concat(row_embedding[i], col_embedding[j])
```

This design has several advantages:

1. **Parameter efficiency.** A `4×4` grid requires only
   `4 × 128 + 4 × 128 = 1,024` parameters, four times fewer than learning 16
   independent 256-dimensional vectors.
2. **Explicit spatial structure.** Cells in the same row share the same row
   component, while cells in the same column share the same column component.
   With 16 unrelated categorical vectors, the model is not directly told that
   two neighboring cells share a row or column.
3. **Correct parameter sharing with `expand()`.** PyTorch's `expand()` creates
   broadcasted views of the same row and column parameters; it does not create
   independent learnable copies. During backpropagation, gradients from every
   use of a row or column accumulate into its original shared parameter.

The main lesson is that the structure of an embedding should reflect the
structure of the data. An image grid is two-dimensional, so explicitly
representing rows and columns provides a more useful inductive bias than
treating every grid cell as an unrelated category.

### 2. Transforming Conv2d features into embeddings

A convolutional network normally outputs a feature map in `[B, C, H, W]`
format because convolution operates naturally over channels and spatial
dimensions. A Transformer instead expects a sequence in `[B, N, C]` format,
where `N` is the number of tokens and `C` is the feature dimension of each
token.

For an image feature map, every `(H, W)` location becomes one token, while the
values across all channels at that location become the token's feature vector.
The conversion is therefore:

```python
tokens = feature_map.flatten(2).permute(0, 2, 1)
```

```text
[B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
```

In this project, the ResNet18 output `[B, 512, 4, 4]` is first projected to
`[B, 256, 4, 4]` and then rearranged into 16 tokens of 256 features:
`[B, 16, 256]`. Conceptually, the transpose collects the value from every
channel at one spatial position into a single token. Positional embeddings are
then needed because flattening assigns the grid cells a sequence order, but
sequence order alone does not explicitly preserve their two-dimensional row
and column relationships.

### 3. Final reflection

The project progressed gradually from v1 to v2 and finally to v3. Each version
increased the model's spatial perception and improved how visual observations,
robot state, language, and task identity are fused together. These changes were
intended to give the policy a stronger understanding of the relationship
between the gripper, target object, and destination during grasp-and-place
tasks.

Across these versions, the increase in model complexity was accompanied by an
increase in robustness and task success. The richer spatial and multimodal
representations improved the policy's ability to combine different inputs and
perform grasp-and-place behavior under a wider range of task conditions.
Overall, v3 provides a stronger foundation for robust single-task behavior and
for scaling the same policy across multiple tasks, while consistent
preprocessing, sufficient training data, and responsive closed-loop control
remain important for realizing the architecture's full capability.
