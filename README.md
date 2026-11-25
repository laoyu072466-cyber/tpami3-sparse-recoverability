# TPAMI3 Sparse Readout

Runnable research code for sparse response-state selection, Sparse-LME, shared C0/C1 readouts, P32 alignment, and paired evaluation. The included synthetic examples exercise the interfaces on a CPU.

The published materials are limited to code, input schemas, and synthetic fixtures. Empirical reproduction requires externally supplied training and evaluation data, candidate bindings, score streams, model weights, and states. Manuscripts and large caches are kept separately. See [the reproduction map](docs/reproduction.md), [input schemas](schemas/inputs.md), and [copyright and external methods](docs/rights.md).

## CPU installation

Use Python 3.11 or 3.12 with the pinned CPU environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
```

The base package also installs with `python -m pip install .` for NumPy-only statistics, view and P32 entry points.

## CPU checks

```bash
tpami3 --help
python -m unittest discover -s tests -v
python scripts/cpu_smoke.py
```

The smoke script generates temporary mock state rows and scores, checks the CLI, and renders a synthetic budget plot. The tensor tests cover the readout and loss on mock inputs.

## Entry points

| Command | Purpose |
| --- | --- |
| `tpami3 views` | Compute response-local view indices for a supplied length |
| `tpami3 train` | Fit a readout from supplied state files, development binding and fixed training pairs |
| `tpami3 score` | Apply a caller-supplied frozen readout to a chosen view |
| `tpami3 p32-fit`, `tpami3 p32-apply` | Fit and apply the 32-dimensional phase map |
| `tpami3 stats`, `tpami3 gap` | Compare aligned score streams with original-question resampling |
| `tpami3 access`, `tpami3 budget` | Recompute state access and the Task44 paired budget family |
| `figures/source/plot_budget.py` | Plot a budget result supplied at runtime |

Run each command with `--help` for its explicit paths. `DATA_ROOT` resolves relative input paths. `MODEL_PATH` supplies a trusted local readout when `score` has no `--model-path` argument. Output paths are explicit.
