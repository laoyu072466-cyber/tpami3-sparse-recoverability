# Reproduction map

| Component | Code | Inputs needed for an empirical run |
| --- | --- | --- |
| Sparse-LME and VCons-LME C0/C1 | `views.py`, `states.py`, `readout.py`, `training.py` | Frozen response-only states, exact ordered pairs, development binding and starting head tensors |
| Task44 budget accounting | `views.py`, `budget.py` | Original response lengths, group membership, seed-aligned scores per arm |
| BinHead/BinTail phase alignment and P32 | `alignment.py` | Row-matched training features, correct phase direction and frozen scoring tail |
| Question-level comparison used for frozen pools | `metrics.py`, `bootstrap.py` | Ordered labels, candidates by original question and fixed group membership |
| Figure generation | `figures/source/plot_budget.py` | Budget output supplied at execution time |

The Sparse-LME head projects each state to 128 dimensions, applies layer normalization and a 32-dimensional token map, pools with unit-temperature LogMeanExp, and uses the two-layer scalar tail. C0 averages the two pairwise softplus objectives. C1 adds `0.1 * Huber(head_margin - tail_margin, delta=1)` to the shared objective. The optional initial tensors and fixed pair orders determine the historical initialization and training sequence. `configs/example/recipe.json` records the settings with a synthetic state width.

BinHead25 and BinTail25 select the first and last in each of `ceil(T / 4)` integer bins. Last25 selects a contiguous terminal slice. Spread25 selects the lower midpoint of each bin. The state access fraction divides selected rows across the pool by all available response rows. FP16 byte accounting applies to the state payload.

Pair-Macro uses strict positive over negative comparisons within each mixed-label question and averages those questions equally. Best averages tied-winner credit within each group, then averages each question's groups. Pooled AUROC assigns half credit to ties. For question bootstrap, one draw's multiplicities apply to both candidates in each pooled AUROC comparison. The same draws apply across all methods and seeds in a paired pool. Seed-level metrics are averaged after measurement.

By convention `stats --contrast LABEL LEFT RIGHT` computes LEFT minus RIGHT and calibrates a family separately for each metric. Pass the contrast family specified by the input protocol. `max-t` uses fixed bootstrap standard errors. `budget` instead uses the Task44 replicate-specific studentization and maximum across its supplied cell family. Provide the cells in the declared order.

For example, generate inputs and run comparisons in a scratch directory.

```bash
python examples/synthetic/generate.py synthetic_run
tpami3 stats --input synthetic_run/mock_scores.npz --score scores --score alternative \
  --contrast example alternative scores --interval bonferroni --draws 64
tpami3 budget --input synthetic_run/mock_budget.npz --width 6 --draws 64 --output synthetic_budget.json
python figures/source/plot_budget.py --input synthetic_budget.json --output synthetic_budget.png
```
