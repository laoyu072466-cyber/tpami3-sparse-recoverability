"""Exercise the installed CLI, readout and figure code in an empty temporary directory."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    import numpy as np
    import torch
    from tpami3.readout import SparseLME

    source = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "MPLCONFIGDIR": str(root / "matplotlib")}

        def command(*args):
            subprocess.run([sys.executable, *args], cwd=root, env=env, check=True, stdout=subprocess.DEVNULL)

        def cli(*args):
            command("-m", "tpami3", *args)

        command(str(source / "examples/synthetic/generate.py"), "synthetic")
        for name in ("views", "train", "score", "stats", "gap", "p32-fit", "p32-apply", "budget", "access"):
            cli(name, "--help")
        cli("stats", "--input", "synthetic/mock_scores.npz", "--score", "scores", "--score", "alternative",
            "--contrast", "example", "alternative", "scores", "--interval", "max-t", "--draws", "64",
            "--output", "stats.json")
        cli("gap", "--input", "synthetic/mock_scores.npz", "--draws", "64", "--output", "gap.json")
        cli("budget", "--input", "synthetic/mock_budget.npz", "--width", "6", "--draws", "64",
            "--output", "budget.json")
        cli("access", "--lengths", "synthetic/lengths.npy", "--width", "6", "--output", "access.json")
        cli("p32-fit", "--source", "synthetic/mock_source.npy", "--target", "synthetic/mock_target.npy",
            "--output", "p32.npz")
        cli("p32-apply", "--source", "synthetic/mock_source.npy", "--map", "p32.npz", "--output", "mapped.npy")
        torch.manual_seed(42)
        torch.save(SparseLME(6).state_dict(), root / "mock_head.pt")
        cli("score", "--states", "synthetic/mock_states.bin", "--offsets", "synthetic/offsets.npy", "--width", "6",
            "--model-path", "mock_head.pt", "--output", "mock_scores.npy")
        command(str(source / "figures/source/plot_budget.py"), "--input", "budget.json", "--output", "budget.png")
        assert "example" in json.loads((root / "stats.json").read_text())["contrasts"]
        assert json.loads((root / "budget.json").read_text())["comparisons"] == 10
        assert np.load(root / "mapped.npy", allow_pickle=False).shape == (48, 32)
        assert np.isfinite(np.load(root / "mock_scores.npy", allow_pickle=False)).all()
        assert (root / "budget.png").stat().st_size > 0
    print("CPU_SMOKE_PASS")


if __name__ == "__main__":
    main()
