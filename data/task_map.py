# data/task_map.py
# Canonical task-name → integer index mapping for LIBERO suites.
#
# WHY THIS FILE EXISTS:
#   HDF5 files store the task as a language string in attrs["problem_info"].
#   There is NO integer index inside the HDF5 — we must assign one.
#   This file provides a STABLE mapping so task 3 is always task 3,
#   regardless of how many tasks you load at train or eval time.
#
# USAGE:
#   from data.task_map import TASK_NAME_TO_IDX, NUM_TASKS
#
#   task_idx = TASK_NAME_TO_IDX[instruction_string]  # int, 0-based
#
# IMPORTANT: LIBERO_SUITE must match the suite you're training on.
#   Switch to "libero_90" if you move to the full 90-task benchmark.

# LIBERO_SUITE = "libero_10"

# Canonical ordered list — copied verbatim from:
#   Lifelong-Robot-Learning/LIBERO: libero/libero/benchmark/libero_suite_task_map.py
# DO NOT reorder. Indices are baked into saved checkpoints.
LIBERO_SUITE = "libero_spatial"

LIBERO_SPATIAL_TASKS = [
"pick up the black bowl between the plate and the ramekin and place it on the plate",
"pick up the black bowl next to the ramekin and place it on the plate",
"pick up the black bowl from table center and place it on the plate",
"pick up the black bowl on the cookie box and place it on the plate",
"pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
"pick up the black bowl on the ramekin and place it on the plate",
"pick up the black bowl next to the cookie box and place it on the plate",
"pick up the black bowl on the stove and place it on the plate",
"pick up the black bowl next to the plate and place it on the plate",
"pick up the black bowl on the wooden cabinet and place it on the plate",
]

_SUITE_MAP = {
    "libero_spatial": LIBERO_SPATIAL_TASKS,
}

_task_list = _SUITE_MAP[LIBERO_SUITE]
NUM_TASKS: int = len(_task_list)

# Primary lookup: instruction string → stable int index
TASK_NAME_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(_task_list)}
# Reverse lookup (useful for logging / debugging)
IDX_TO_TASK_NAME: dict[int, str] = {i: name for name, i in TASK_NAME_TO_IDX.items()}


def get_task_idx(instruction: str) -> int:
    """
    Look up stable integer index for a LIBERO task instruction string.

    Args:
        instruction : language_instruction string read from HDF5 attrs
    Returns:
        int in range [0, NUM_TASKS)
    Raises:
        KeyError with a helpful message if the instruction is not found
                 (e.g. wrong suite, or instruction contains extra whitespace)
    """
    key = instruction.strip()
    print(key)
    if key not in TASK_NAME_TO_IDX:
        raise KeyError(
            f"Task instruction not found in {LIBERO_SUITE} map:\n"
            f"  '{key}'\n"
            f"Check that LIBERO_SUITE='{LIBERO_SUITE}' matches your data, "
            f"or that the instruction string has no leading/trailing whitespace."
        )
    return TASK_NAME_TO_IDX[key]
