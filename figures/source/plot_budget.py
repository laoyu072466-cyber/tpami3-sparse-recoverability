"""Plot Best versus response states read from a caller-supplied budget result."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.suffix not in (".png", ".pdf", ".svg"):
        p.error("Output extension must be .png, .pdf or .svg.")
    groups = defaultdict(list)
    for row in json.loads(args.input.read_text(encoding="utf-8"))["rows"]:
        groups[row["pool"], row["arm"][0]].append((100 * row["real_ratio"], row["best_pct"]))
    if not groups:
        p.error("The supplied result has no budget rows.")
    fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
    for (pool, prefix), values in sorted(groups.items()):
        x, y = zip(*sorted(values))
        ax.plot(x, y, marker="o", markersize=3, label=f"{pool} · {dict(L='Last', S='Spread')[prefix]}")
    ax.set(xlabel="Response states read (%)", ylabel="Best (%)", xscale="log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    with args.output.open("xb") as stream:
        fig.savefig(stream, format=args.output.suffix[1:], dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
