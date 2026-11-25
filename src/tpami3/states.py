"""Read caller-supplied response-state rows with explicit offsets."""

from pathlib import Path

import numpy as np

from .views import view_positions


class PackedStates:
    def __init__(self, states, offsets):
        values = np.asanyarray(states)
        bounds = np.asarray(offsets)
        if values.ndim != 2 or values.shape[1] == 0 or values.dtype.kind != "f":
            raise ValueError("States must be a floating-point matrix with positive width.")
        if bounds.ndim != 1 or len(bounds) < 2 or bounds.dtype.kind not in "iu":
            raise ValueError("Offsets must be a one-dimensional integer array.")
        if bounds[0] != 0 or bounds[-1] != len(values) or np.any(bounds[1:] <= bounds[:-1]):
            raise ValueError("Offsets must start at zero and delimit nonempty candidates.")
        self.values = values
        self.offsets = bounds.astype(np.int64, copy=True)
        self.width = values.shape[1]

    @classmethod
    def open_fp16(cls, states_path, offsets_path, width):
        """Use little-endian float16 rows and a NumPy-format offsets sidecar."""
        if width < 1:
            raise ValueError("State width must be positive.")
        offsets = np.load(offsets_path, allow_pickle=False)
        if offsets.ndim != 1 or len(offsets) < 2 or offsets.dtype.kind not in "iu":
            raise ValueError("Offsets must be a one-dimensional integer array.")
        rows = int(offsets[-1])
        if rows < 1 or Path(states_path).stat().st_size != rows * width * 2:
            raise ValueError("State file size does not match the final offset and width.")
        values = np.memmap(states_path, dtype="<f2", shape=(rows, width), mode="r")
        return cls(values, offsets)

    def __len__(self):
        return len(self.offsets) - 1

    def batch(self, rows, view, denominator=4):
        """Return compact float32 token rows and a batch-by-time validity mask."""
        rows = np.asarray(rows)
        if (rows.ndim != 1 or not rows.size or rows.dtype.kind not in "iu"
                or np.any(rows < 0) or np.any(rows >= len(self))):
            raise ValueError("Expected a nonempty vector of valid candidate indices.")
        parts = []
        for row in rows:
            start, end = self.offsets[int(row):int(row) + 2]
            positions = view_positions(end - start, view, denominator)
            parts.append(self.values[start + positions])
        sizes = np.array([len(part) for part in parts])
        valid = np.arange(sizes.max())[None, :] < sizes[:, None]
        result = np.concatenate(parts).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("Selected hidden-state rows must be finite.")
        return result, valid
