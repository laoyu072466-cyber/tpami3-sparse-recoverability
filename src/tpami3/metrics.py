"""Tie-aware candidate metrics evaluated on an explicit question binding."""

import numpy as np


class CandidatePool:
    def __init__(self, labels, question_index, groups=None):
        y, q = np.asarray(labels), np.asarray(question_index)
        if (y.ndim != 1 or not len(y) or q.shape != y.shape or q.dtype.kind not in "iu"
                or y.dtype.kind not in "biu" or not np.isin(y, (0, 1)).all()):
            raise ValueError("Expected binary labels and an aligned integer question index.")
        unique_q = np.unique(q)
        if not np.array_equal(unique_q, np.arange(len(unique_q))):
            raise ValueError("Question indices must be dense and zero-based.")
        self.labels = y.astype(bool, copy=True)
        self.question_index = q.astype(np.int64, copy=True)
        self.n_questions = len(unique_q)
        order = np.argsort(q, kind="stable")
        split = np.cumsum(np.bincount(self.question_index))[:-1]
        self._rows = np.split(order, split)
        self._positive = [rows[self.labels[rows]] for rows in self._rows]
        self._negative = [rows[~self.labels[rows]] for rows in self._rows]
        self.groups = None if groups is None else self._validate_groups(groups)

    def _validate_groups(self, groups):
        values = np.asarray(groups)
        if (values.dtype.kind not in "iu" or values.ndim != 3 or len(values) != self.n_questions
                or not 1 <= values.shape[-1] <= 8 or (values >= len(self.labels)).any()
                or (values < -1).any()):
            raise ValueError("Groups must have shape [question, group, up to 8] and valid candidate IDs.")
        values = values.astype(np.int64, copy=True)
        live = values >= 0
        owners = self.question_index[np.maximum(values, 0)]
        if (live & (owners != np.arange(self.n_questions)[:, None, None])).any():
            raise ValueError("A group crosses original questions.")
        ordered = np.sort(values, axis=-1)
        if ((ordered[..., 1:] >= 0) & (np.diff(ordered, axis=-1) == 0)).any():
            raise ValueError("A group repeats a candidate.")
        if not live.any(axis=-1).any(axis=-1).all():
            raise ValueError("Every original question must have a nonempty group.")
        return values

    def _score(self, scores):
        a = np.asarray(scores, dtype=np.float64)
        if a.shape != self.labels.shape or not np.isfinite(a).all():
            raise ValueError("Expected a finite score for every bound candidate.")
        return a

    def per_question_pair(self, scores):
        """Strict positive-over-negative accuracy on each mixed-label question."""
        values = self._score(scores)
        result = np.full(self.n_questions, np.nan)
        for k, (positive, negative) in enumerate(zip(self._positive, self._negative)):
            if len(positive) and len(negative):
                wins = np.searchsorted(np.sort(values[negative]), values[positive], side="left").sum()
                result[k] = wins / (len(positive) * len(negative))
        return result

    def per_question_best(self, scores):
        """Uniform correct-label credit across tied winners of each supplied group."""
        values = self._score(scores)
        if self.groups is None:
            raise ValueError("Best requires explicit within-question groups.")
        live = self.groups >= 0
        rows = np.maximum(self.groups, 0)
        picked = np.where(live, values[rows], -np.inf)
        tied = live & (picked == picked.max(axis=-1, keepdims=True))
        count = tied.sum(axis=-1)
        credit = np.divide((tied & self.labels[rows]).sum(axis=-1), count,
                           out=np.zeros(count.shape, dtype=float), where=count > 0)
        return credit.sum(axis=-1) / (count > 0).sum(axis=-1)

    def pair_disagreements(self, left, right):
        """Return per-question counts of mismatching margin signs and all eligible pairs."""
        left, right = self._score(left), self._score(right)
        mismatches = np.zeros(self.n_questions, dtype=np.int64)
        total = np.zeros_like(mismatches)
        for k, (positive, negative) in enumerate(zip(self._positive, self._negative)):
            total[k] = len(positive) * len(negative)
            for index in positive:
                a = np.sign(left[index] - left[negative])
                b = np.sign(right[index] - right[negative])
                mismatches[k] += np.count_nonzero(a != b)
        return mismatches, total

    def weighted_auc(self, scores, weights):
        """Pooled AUROC with half credit for ties; reweight both candidates in a pair."""
        values = self._score(scores)
        _, rank = np.unique(values, return_inverse=True)
        size = int(rank.max()) + 1
        pos, neg = rank[self.labels], rank[~self.labels]
        qpos = self.question_index[self.labels]
        qneg = self.question_index[~self.labels]
        result = np.full(len(weights), np.nan)
        for i, weight in enumerate(weights):
            p = np.bincount(pos, weights=weight[qpos], minlength=size)
            n = np.bincount(neg, weights=weight[qneg], minlength=size)
            total = p.sum() * n.sum()
            if total:
                result[i] = (p @ (np.cumsum(n) - 0.5 * n)) / total
        return result
