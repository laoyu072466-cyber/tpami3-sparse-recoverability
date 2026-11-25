import importlib.util
import math
import unittest

import numpy as np

from tpami3.alignment import fit_p32


class AlignmentTests(unittest.TestCase):
    def test_p32_fit_with_known_ridge_solution(self):
        source = np.vstack([np.eye(32), -np.eye(32)])
        target = 5 + 3 * source
        fitted = fit_p32(source, target)
        np.testing.assert_allclose(fitted.W_P, np.eye(32) / 1.01, atol=1e-13)
        np.testing.assert_allclose(fitted.transform(source), 5 + 3 * source / 1.01, atol=1e-13)
        source[:, 0] = 2
        self.assertEqual(fit_p32(source, target).x_scale[0], 1.0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install the readout extra for CPU tensor tests.")
class ReadoutTests(unittest.TestCase):
    def test_shared_initialization_and_masked_log_mean(self):
        import torch
        from tpami3.readout import SparseLME

        mask = torch.tensor([[True, True, False], [True, False, False]])
        states = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 10
        torch.manual_seed(42)
        readout = SparseLME(6)
        twin = SparseLME(6)
        twin.load_state_dict(readout.state_dict())
        torch.testing.assert_close(readout(states, mask), twin(states, mask))
        features = readout.phi(readout.proj_ln(readout.proj(states)))
        expected = torch.stack([features[:2].logsumexp(dim=0) - math.log(2), features[2]])
        torch.testing.assert_close(readout.pooled(states, mask), expected)
        with self.assertRaises(ValueError):
            readout(states, mask & False)

    def test_c0_c1_loss_and_gradients(self):
        import torch
        from tpami3.readout import dual_view_loss

        a = torch.tensor([0.0, -2.0, 1.0], requires_grad=True)
        b = torch.tensor([0.0, 1.5, 1.25], requires_grad=True)
        base, parts0 = dual_view_loss(a, b, "C0")
        c1, parts1 = dual_view_loss(a, b, "C1")
        manual = float(np.logaddexp(0, -a.detach().numpy()).mean()
                       + np.logaddexp(0, -b.detach().numpy()).mean()) / 2
        self.assertAlmostEqual(base.item(), manual, places=6)
        self.assertAlmostEqual(parts1["huber"].item(), (3.0 + 0.03125) / 3, places=6)
        self.assertAlmostEqual((c1 - base).item(), 0.1 * parts1["huber"].item(), places=6)
        self.assertFalse(parts0["huber"].requires_grad)
        c1.backward()
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())

    def test_supplied_pair_order_requires_one_question(self):
        from tpami3.metrics import CandidatePool
        from tpami3.training import validate_orders

        pool = CandidatePool([1, 0, 1, 0], [0, 0, 1, 1])
        self.assertEqual(validate_orders([[0, 1], [2, 3]], pool).shape, (1, 2, 2))
        with self.assertRaisesRegex(ValueError, "same original question"):
            validate_orders([[0, 3]], pool)
