"""Paired resampling at original-question level and simultaneous intervals."""

from operator import index

import numpy as np


def question_weights(questions, draws, seed):
    """Return counts in same-size PCG64 resamples of the original questions."""
    questions, draws = index(questions), index(draws)
    if questions < 1 or draws < 1:
        raise ValueError("Expected positive question and draw counts.")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    weights = np.empty((draws, questions), dtype=np.int64)
    for i, sample in enumerate(weights):
        sample[:] = np.bincount(rng.integers(questions, size=questions), minlength=questions)
    return weights


def _validate_weights(weights, questions):
    a = np.asarray(weights)
    if (a.ndim != 2 or a.shape[1] != questions or not len(a)
            or a.dtype.kind not in "iu" or (a < 0).any() or not (a.sum(axis=1) == questions).all()):
        raise ValueError("Expected nonnegative integer question counts and the original resample size.")
    return a


def _ratio(numerator, denominator, weights):
    n = weights @ numerator
    d = weights @ denominator
    return np.divide(n, d, out=np.full(len(weights), np.nan), where=d != 0)


def metric_samples(pool, scores, weights, metric):
    """Return the metric on the original sample and on each common bootstrap draw."""
    w = _validate_weights(weights, pool.n_questions)
    all_weights = np.vstack((np.ones(pool.n_questions, dtype=np.int64), w))
    if metric == "auroc":
        values = pool.weighted_auc(scores, all_weights)
    elif metric == "disagreement":
        values = _ratio(*pool.pair_disagreements(*scores), all_weights)
    else:
        if metric not in ("best", "pair"):
            raise ValueError(f"Unknown metric {metric}.")
        per_question = pool.per_question_best(scores) if metric == "best" else pool.per_question_pair(scores)
        included = np.isfinite(per_question)
        values = _ratio(np.where(included, per_question, 0), included, all_weights)
    return float(values[0]), values[1:]


def _samples(samples, alpha):
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or not all(values.shape) or not 0 < alpha < 1:
        raise ValueError("Expected a nonempty [draw, contrast] matrix and alpha in (0,1).")
    return values


def percentile_intervals(samples, alpha=0.05, bonferroni=False):
    """Two-sided empirical percentile intervals, adjusted for a declared family if requested."""
    values = _samples(samples, alpha)
    tail = alpha / (2 * (values.shape[1] if bonferroni else 1))
    result = []
    for column in values.T:
        finite = column[np.isfinite(column)]
        result.append(np.quantile(finite, [tail, 1 - tail]) if finite.size else [np.nan, np.nan])
    return np.asarray(result)


def max_t_intervals(points, samples, alpha=0.05):
    """Use the draw-wise maximum absolute centered statistic and fixed bootstrap SEs."""
    a = _samples(samples, alpha)
    points = np.asarray(points, dtype=float)
    if (points.shape != (a.shape[1],) or len(a) < 2
            or not np.isfinite(a).all() or not np.isfinite(points).all()):
        raise ValueError("A max-|t| family needs complete finite aligned draws and point estimates.")
    se = a.std(axis=0, ddof=1)
    scaled = np.divide(np.abs(a - points), se, out=np.zeros_like(a), where=se > 0)
    critical = float(np.quantile(scaled.max(axis=1), 1 - alpha))
    radius = critical * se
    return np.column_stack((points - radius, points + radius)), critical
