import unittest

import numpy as np

from tpami3.bootstrap import max_t_intervals, metric_samples, percentile_intervals, question_weights
from tpami3.budget import ARMS, budget_intervals, replay_budget
from tpami3.metrics import CandidatePool


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, 0, 1, 1, 0, 0])
        self.q = np.array([0, 0, 0, 1, 1, 2])
        self.groups = np.array([[[0, 1], [2, 1]], [[3, 4], [-1, -1]], [[5, -1], [-1, -1]]])
        self.pool = CandidatePool(self.y, self.q, self.groups)
        self.scores = np.array([0.2, 0.2, 0.7, 0.2, 0.5, 0.2])
        self.weights = np.array([[3, 0, 0], [0, 2, 1], [0, 0, 3], [1, 1, 1]])

    def test_strict_pair_macro_and_best_ties(self):
        pair, draws = metric_samples(self.pool, self.scores, self.weights, "pair")
        best, best_draws = metric_samples(self.pool, self.scores, self.weights, "best")
        self.assertEqual(pair, 0.25)
        np.testing.assert_allclose(draws, [0.5, 0, np.nan, 0.25], equal_nan=True)
        self.assertEqual(best, 0.25)
        np.testing.assert_allclose(best_draws, [0.75, 0, 0, 0.25])

    def test_auroc_reweights_both_candidates(self):
        point, draws = metric_samples(self.pool, self.scores, self.weights, "auroc")
        expected = []
        for w in np.vstack([[1, 1, 1], self.weights]):
            repeat = np.repeat(np.arange(len(self.q)), w[self.q])
            pos = self.scores[repeat[self.y[repeat] == 1]]
            neg = self.scores[repeat[self.y[repeat] == 0]]
            gap = pos[:, None] - neg[None, :]
            expected.append(((gap > 0) + 0.5 * (gap == 0)).mean() if gap.size else np.nan)
        np.testing.assert_allclose([point, *draws], expected, atol=1e-15, equal_nan=True)

    def test_sign_disagreement_ties(self):
        alt = self.scores.copy()
        alt[0] = 0.3
        mismatches, total = self.pool.pair_disagreements(self.scores, alt)
        np.testing.assert_array_equal(mismatches, [1, 0, 0])
        np.testing.assert_array_equal(total, [2, 1, 0])
        self.assertEqual(metric_samples(self.pool, (self.scores, alt), self.weights, "disagreement")[0], 1 / 3)

    def test_question_cluster_and_family_intervals(self):
        expected = np.random.default_rng(19).integers(3, size=(11, 3))
        weights = question_weights(3, 11, 19)
        for w, row in zip(weights, expected):
            np.testing.assert_array_equal(w, np.bincount(row, minlength=3))
        samples = np.column_stack([np.arange(11), np.arange(11) ** 2])
        ci = percentile_intervals(samples, bonferroni=True)
        np.testing.assert_allclose(ci, np.quantile(samples, [0.0125, 0.9875], axis=0).T)
        points = [5, 25]
        bounds, critical = max_t_intervals(points, samples)
        se = samples.std(axis=0, ddof=1)
        self.assertAlmostEqual(critical, np.quantile(np.max(np.abs(samples - points) / se, axis=1), 0.95))
        np.testing.assert_allclose(bounds.mean(axis=1), points)

    def test_task44_studentized_and_frontier(self):
        constant = np.column_stack([np.ones(3), np.zeros(3)])
        bounds, critical = budget_intervals([(constant, self.weights)])
        np.testing.assert_array_equal(bounds[0], [[1, 1], [0, 0]])
        self.assertEqual(critical, 0)
        scores = {arm: self.scores[None, :] for arm in ARMS}
        result = replay_budget([("synthetic", self.pool, [1, 4, 5, 8, 9, 2], scores)], 6, 21, 22)
        self.assertEqual(result["comparisons"], 10)
        self.assertEqual(result["frontier"][0]["smallest_budget_denominator"], 64)
        self.assertEqual(result["rows"][0]["simultaneous_pp"], [0, 0])

    def test_misbinding_refused(self):
        wrong = self.groups.copy()
        wrong[0, 0, 1] = 3
        with self.assertRaisesRegex(ValueError, "crosses"):
            CandidatePool(self.y, self.q, wrong)
