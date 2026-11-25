"""Response-local, equally budgeted deterministic views."""

from operator import index

import numpy as np


VIEWS = ("full", "endpoint", "last", "spread", "bin_head", "bin_tail")


def view_positions(length, view, denominator=4):
    """Return sorted zero-based positions for a single nonempty response."""
    length, denominator = index(length), index(denominator)
    if length < 1 or denominator < 1 or view not in VIEWS:
        raise ValueError("Expected a positive response length, denominator and known view.")
    if length > np.iinfo(np.int32).max:
        raise ValueError("Response length exceeds the supported integer range.")
    if view == "full":
        return np.arange(length, dtype=np.int64)
    if view == "endpoint":
        return np.array([length - 1], dtype=np.int64)
    count = (length + denominator - 1) // denominator
    steps = np.arange(count, dtype=np.int64)
    if view == "last":
        return length - count + steps
    lower = steps * length // count
    upper = (steps + 1) * length // count - 1
    if view == "spread":
        return (lower + upper) // 2
    return lower if view == "bin_head" else upper
