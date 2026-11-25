# Input schemas

Inputs are external to the package. Use identical candidate row order across state offsets, scores, labels and pair references for a given pool.

## State rows and readout checkpoints

The state file stores contiguous row-major little-endian FP16 response states of shape `[state_rows, width]`. Supply the width explicitly. The integer `offsets.npy` has one more item than the number of candidates, starts at zero, and ends at the state-row count. Each pair of offsets delimits one nonempty response. The checkpoint accepted by `score` is a trusted local PyTorch tensor state dict or a checkpoint with the readout tensors under `head`.

## Candidate binding and scores

| Key | Type and shape |
| --- | --- |
| `labels` | Binary integer or Boolean `[candidate]` |
| `question_index` | Dense zero-based integer `[candidate]`, in original-question order |
| `groups` | Optional integer `[question, group, at most 8]` pointing into the candidate array. Each row belongs to its question. Pad with `-1` |
| score keys | Finite floating point `[candidate]` or `[seed, candidate]`; `gap` takes matching Head and Tail arrays |

Store these arrays in an NPZ file. The entire cube for one question is resampled as one unit. Use the same ordered seed rows for all arms of a paired family. Original text and candidate labels can remain separate. The code consumes only the numeric binding needed by the declared comparison.

## Training order

Provide separate train and development bindings. Pair orders are integer NPY arrays of shape `[pair, 2]` for one shared order or `[5, pair, 2]` for epoch-specific orders. A pair lists the positive candidate first and the negative from the same question second. `--initial` accepts the matching trusted local tensor checkpoint.

## Phase map

`p32-fit` takes two finite floating-point NPY arrays of shape `[matched_training_row, 32]`. The output NPZ stores `x_mean`, `x_scale`, `y_mean`, `y_scale`, and `W_P`. `p32-apply` takes another matrix in the mapped source phase and writes the corresponding target-phase representations.

## Budget family

For each pool supply its candidate binding, positive integer `lengths[candidate]`, and scores for `F`, `E` and `L4`/`S4` through `L64`/`S64` using denominators 4, 8, 16, 32 and 64. All arm arrays have `[seed, candidate]` shape. Supply the pools in the family order and the declared width. Results retain individual tolerance outcomes for every arm.
