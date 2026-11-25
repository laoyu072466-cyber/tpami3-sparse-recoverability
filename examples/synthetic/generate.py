"""Generate tiny numerical fixtures with no external inputs."""

import argparse
from pathlib import Path

import numpy as np

from tpami3.budget import ARMS


def generate(directory):
    directory.mkdir()
    rng = np.random.default_rng(731)
    lengths = np.array([2, 6, 5, 7, 3, 1], dtype=np.int64)
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)
    question_index = np.repeat(np.arange(3), 2)
    groups = np.arange(6).reshape(3, 1, 2)
    base = np.array([0.5, 0.5, 0.2, 0.3, -0.2, 0.9])
    alternative = np.array([0.6, 0.4, 0.2, 0.3, 0.0, 0.9])
    np.save(directory / "lengths.npy", lengths)
    np.save(directory / "offsets.npy", np.r_[0, np.cumsum(lengths)])
    rng.normal(size=(lengths.sum(), 6)).astype("<f2").tofile(directory / "mock_states.bin")
    np.savez(directory / "mock_scores.npz", labels=labels, question_index=question_index, groups=groups,
             scores=base, alternative=alternative, head=base, tail=alternative)
    scored = {arm: np.stack([base if arm == "F" or i % 2 == 0 else alternative] * 3)
              for i, arm in enumerate(ARMS)}
    np.savez(directory / "mock_budget.npz", labels=labels, question_index=question_index,
             groups=groups, lengths=lengths, **scored)
    x = rng.normal(size=(48, 32))
    np.save(directory / "mock_source.npy", x)
    np.save(directory / "mock_target.npy", x * 1.4 + 0.2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    generate(p.parse_args().directory)
