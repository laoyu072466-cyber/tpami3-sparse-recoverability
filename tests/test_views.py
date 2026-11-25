import tempfile
import unittest
from pathlib import Path

import numpy as np

from tpami3.budget import state_access
from tpami3.states import PackedStates
from tpami3.views import VIEWS, view_positions


class ViewTests(unittest.TestCase):
    def test_exact_boundaries(self):
        cases = {"last": [6, 7, 8], "spread": [1, 4, 7], "bin_head": [0, 3, 6], "bin_tail": [2, 5, 8]}
        for view, expected in cases.items():
            np.testing.assert_array_equal(view_positions(9, view), expected)
        for view in VIEWS:
            np.testing.assert_array_equal(view_positions(1, view, 64), [0])

    def test_short_and_large_responses(self):
        for length in (2, 3, 7, 8, 11, 65, 100003):
            for denominator in (1, 4, 8, 64):
                count = (length + denominator - 1) // denominator
                head = view_positions(length, "bin_head", denominator)
                tail = view_positions(length, "bin_tail", denominator)
                spread = view_positions(length, "spread", denominator)
                self.assertEqual(len(head), count)
                self.assertEqual((head[0], tail[-1]), (0, length - 1))
                np.testing.assert_array_equal(head[1:], tail[:-1] + 1)
                self.assertTrue(np.all((head <= spread) & (spread <= tail)))
        with self.assertRaises(ValueError):
            view_positions(0, "last")

    def test_raw_read_and_repeated_candidate_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = np.arange(18, dtype="<f2").reshape(9, 2)
            state.tofile(root / "states.bin")
            np.save(root / "offsets.npy", [0, 5, 9])
            packed = PackedStates.open_fp16(root / "states.bin", root / "offsets.npy", 2)
            values, mask = packed.batch([1, 0, 0], "bin_tail")
            np.testing.assert_array_equal(values, state[[8, 1, 4, 1, 4]])
            np.testing.assert_array_equal(mask, [[1, 0], [1, 1], [1, 1]])
            self.assertEqual(values.dtype, np.float32)
            with self.assertRaises(ValueError):
                PackedStates.open_fp16(root / "states.bin", root / "offsets.npy", 3)

    def test_access_is_token_weighted(self):
        last = state_access([1, 5, 9], "L4", 6)
        spread = state_access([1, 5, 9], "S4", 6)
        self.assertEqual(last["states"], 6)
        self.assertEqual(last["real_ratio"], 0.4)
        self.assertEqual(last["contains_endpoint_fraction"], 1.0)
        self.assertEqual(spread["contains_endpoint_fraction"], 1 / 3)
        self.assertEqual(spread["input_fp16_bytes"], 72)
