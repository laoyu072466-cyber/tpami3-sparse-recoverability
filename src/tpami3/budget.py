"""State counts and paired Best comparisons for the Task44 budget grid."""

import numpy as np

from .bootstrap import _validate_weights, percentile_intervals, question_weights


DENOMINATORS = (4, 8, 16, 32, 64)
BUDGET_ARMS = tuple(f"{p}{d}" for d in DENOMINATORS for p in ("L", "S"))
ARMS = ("F", "E", *BUDGET_ARMS)


def state_access(lengths, arm, width):
    lengths = np.asarray(lengths)
    if (lengths.ndim != 1 or not lengths.size or lengths.dtype.kind not in "iu" or (lengths < 1).any()
            or arm not in ARMS or width < 1 or (lengths > np.iinfo(np.int32).max).any()):
        raise ValueError("Expected positive integer response lengths, state width and a budget arm.")
    lengths = lengths.astype(np.int64)
    if arm == "F":
        counts, last = lengths, np.ones(len(lengths), dtype=bool)
    elif arm == "E":
        counts, last = np.ones_like(lengths), np.ones(len(lengths), dtype=bool)
    else:
        denominator = int(arm[1:])
        counts = (lengths + denominator - 1) // denominator
        last = np.ones(len(lengths), dtype=bool) if arm.startswith("L") else (counts - 1) * lengths // counts == lengths - 1
    states, total = int(counts.sum()), int(lengths.sum())
    return {
        "candidates": len(lengths), "states": states, "full_states": total, "real_ratio": states / total,
        "mean_candidate_ratio": float(np.mean(counts / lengths)),
        "mean_k": float(counts.mean()), "median_k": float(np.median(counts)),
        "p90_k": float(np.quantile(counts, 0.9)), "k1_fraction": float(np.mean(counts == 1)),
        "endpoint_exact_fraction": float(np.mean((counts == 1) & last)),
        "contains_endpoint_fraction": float(last.mean()), "input_fp16_bytes": 2 * width * states,
    }


def _studentized(contrasts, weights):
    n = len(contrasts)
    point = contrasts.mean(axis=0)
    scale = contrasts.std(axis=0, ddof=1) / np.sqrt(n)
    centered = contrasts - point
    shifts = weights @ centered / n
    sample_se = np.sqrt(np.maximum(weights @ (centered ** 2) / n - shifts ** 2, 0) / (n - 1))
    use = scale > 0
    z = np.abs(shifts[:, use])
    s = sample_se[:, use]
    ts = np.divide(z, s, out=np.where(z <= 1e-12, 0.0, np.inf), where=s > 1e-14)
    maximum = ts.max(axis=1) if ts.shape[1] else np.zeros(len(weights))
    return point, scale, maximum


def budget_intervals(cells, alpha=0.05):
    """Calibrate the Task44 family across cells with aligned draw indices."""
    if not cells or not 0 < alpha < 1:
        raise ValueError("Expected a nonempty budget family and alpha in (0,1).")
    stats = []
    for differences, weights in cells:
        d = np.asarray(differences, dtype=float)
        if d.ndim != 2 or len(d) < 2 or not d.shape[1] or not np.isfinite(d).all():
            raise ValueError("Budget contrasts need finite [question, arm] values and at least two questions.")
        stats.append(_studentized(d, _validate_weights(weights, len(d))))
    if len({len(row[2]) for row in stats}) != 1:
        raise ValueError("All cells need the same number of draws.")
    critical = float(np.quantile(np.maximum.reduce([r[2] for r in stats]), 1 - alpha, method="higher"))
    bounds = []
    for point, se, _ in stats:
        radius = np.zeros_like(se)
        np.multiply(se, critical, out=radius, where=se > 0)
        bounds.append(np.column_stack((point - radius, point + radius)))
    return bounds, critical


def replay_budget(pools, width, draws, seed):
    """Use supplied scores and bindings to recalculate a declared budget family."""
    rng = np.random.default_rng(seed)
    cells, inputs = [], []
    for name, pool, lengths, scores in pools:
        arrays = {arm: np.asarray(scores[arm]) for arm in ARMS}
        if any(a.ndim != 2 or not len(a) or a.shape != arrays["F"].shape for a in arrays.values()):
            raise ValueError("All budget arms need aligned [seed, candidate] score matrices.")
        if len(lengths) != len(pool.labels):
            raise ValueError("Response lengths do not match the candidate binding.")
        best = {arm: np.mean([pool.per_question_best(row) for row in a], axis=0) for arm, a in arrays.items()}
        differences = np.column_stack([best["F"] - best[a] for a in BUDGET_ARMS])
        w = question_weights(pool.n_questions, draws, rng)
        cells.append((differences, w))
        inputs.append((name, lengths, best))
    bands, critical = budget_intervals(cells)
    rows, frontier = [], []
    for (name, lengths, best), (difference, w), bounds in zip(inputs, cells, bands):
        nominal = percentile_intervals(w @ difference / len(difference))
        endpoint_effect = np.column_stack([best[a] - best["E"] for a in BUDGET_ARMS])
        endpoint_ci = percentile_intervals(w @ endpoint_effect / len(endpoint_effect))
        pool_rows = {}
        for k, arm in enumerate(BUDGET_ARMS):
            lo, hi = 100 * bounds[k]
            row = {"pool": name, "arm": arm, **state_access(lengths, arm, width),
                   "best_pct": 100 * float(best[arm].mean()), "full_minus_sparse_pp": 100 * float(difference[:, k].mean()),
                   "full_best_pct": 100 * float(best["F"].mean()), "endpoint_best_pct": 100 * float(best["E"].mean()),
                   "nominal_pp": (100 * nominal[k]).tolist(), "simultaneous_pp": [float(lo), float(hi)],
                   "sparse_minus_endpoint_pp": 100 * float(endpoint_effect[:, k].mean()),
                   "sparse_minus_endpoint_nominal_pp": (100 * endpoint_ci[k]).tolist()}
            for threshold in (2, 5):
                status = "UNRESOLVED"
                if np.isfinite(hi) and hi < threshold:
                    status = "WITHIN_TOLERANCE_SUPPORTED"
                elif np.isfinite(lo) and lo > threshold:
                    status = "LOSS_CONFIRMED"
                row[f"tolerance_{threshold}_pp"] = status
            rows.append(row)
            pool_rows[arm] = row
        for prefix in ("L", "S"):
            for threshold in (2, 5):
                supported = [d for d in reversed(DENOMINATORS)
                             if pool_rows[f"{prefix}{d}"][f"tolerance_{threshold}_pp"] == "WITHIN_TOLERANCE_SUPPORTED"]
                chosen = supported[0] if supported else None
                frontier.append({"pool": name, "arm_prefix": prefix, "tolerance_pp": threshold,
                                 "supported_denominators": supported, "smallest_budget_denominator": chosen,
                                 "smallest_real_ratio": pool_rows[f"{prefix}{chosen}"]["real_ratio"] if chosen else None})
    return {"critical": critical, "comparisons": len(rows), "rows": rows, "frontier": frontier}
