"""Command-line entry points using caller-provided input and output paths."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import os

import numpy as np

from .alignment import PhaseMap, fit_p32
from .bootstrap import max_t_intervals, metric_samples, percentile_intervals, question_weights
from .budget import ARMS, replay_budget, state_access
from .metrics import CandidatePool
from .states import PackedStates
from .views import VIEWS, view_positions


def _input(path):
    if path.is_absolute():
        return path
    return Path(os.environ.get("DATA_ROOT", ".")) / path


def _arrays(path):
    with np.load(_input(path), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _pool(arrays):
    return CandidatePool(arrays["labels"], arrays["question_index"], arrays.get("groups"))


def _clean(data):
    if isinstance(data, dict):
        return {key: _clean(item) for key, item in data.items()}
    if isinstance(data, (list, tuple, np.ndarray)):
        return [_clean(value) for value in data]
    if isinstance(data, (float, np.floating)):
        return float(data) if np.isfinite(data) else None
    return data


def _emit(data, output=None):
    text = json.dumps(_clean(data), indent=2, allow_nan=False) + "\n"
    if output is None:
        print(text, end="")
    else:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(text)


def _save_array(path, values):
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def _path(parser, flag, **kwargs):
    parser.add_argument(flag, type=Path, **kwargs)


def _draw_options(parser):
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2025)


def _state_options(parser, prefix=""):
    _path(parser, f"--{prefix}states", required=True)
    _path(parser, f"--{prefix}offsets", required=True)


def _packed(args, prefix=""):
    states = _input(getattr(args, prefix + "states"))
    offsets = _input(getattr(args, prefix + "offsets"))
    return PackedStates.open_fp16(states, offsets, args.width)


def _views(args):
    _emit({view: view_positions(args.length, view, args.denominator).tolist() for view in VIEWS})


def _stats(args):
    arrays = _arrays(args.input)
    pool = _pool(arrays)
    weights = question_weights(pool.n_questions, args.draws, args.seed)
    results, contrasts = {}, {}
    for key in args.score or ["scores"]:
        matrix = np.atleast_2d(arrays[key])
        if matrix.ndim != 2 or not len(matrix):
            raise ValueError("Scores must have shape [candidate] or [seed, candidate].")
        results[key] = {}
        for name in ("pair", "best", "auroc") if pool.groups is not None else ("pair", "auroc"):
            summaries = [metric_samples(pool, row, weights, name) for row in matrix]
            point, draws = zip(*summaries)
            results[key][name] = (float(np.mean(point)), np.mean(draws, axis=0))
    for label, left, right in args.contrast:
        if label in contrasts:
            raise ValueError("Contrast names must be unique.")
        if np.atleast_2d(arrays[left]).shape != np.atleast_2d(arrays[right]).shape:
            raise ValueError("Paired score streams must also align by seed.")
        contrasts[label] = {name: (point - results[right][name][0], d - results[right][name][1])
                            for name, (point, d) in results[left].items()}
    report = {"unit": "original question", "draws": args.draws, "seed": args.seed, "absolute": {},
              "contrast_method": args.interval, "contrasts": {}}
    for key, values in results.items():
        report["absolute"][key] = {
            name: {"point": point, "interval": percentile_intervals(d[:, None])[0].tolist()}
            for name, (point, d) in values.items()
        }
    if contrasts:
        for name in next(iter(contrasts.values())):
            points = [entry[name][0] for entry in contrasts.values()]
            samples = np.column_stack([entry[name][1] for entry in contrasts.values()])
            if args.interval == "max-t":
                ci, _ = max_t_intervals(points, samples)
            else:
                ci = percentile_intervals(samples, bonferroni=args.interval == "bonferroni")
            for label, point, interval in zip(contrasts, points, ci):
                report["contrasts"].setdefault(label, {})[name] = {"point": point, "interval": interval.tolist()}
    _emit(report, args.output)


def _gap(args):
    arrays = _arrays(args.input)
    pool = _pool(arrays)
    w = question_weights(pool.n_questions, args.draws, args.seed)
    head, tail = np.atleast_2d(arrays[args.head_key]), np.atleast_2d(arrays[args.tail_key])
    if head.ndim != 2 or not len(head) or head.shape != tail.shape:
        raise ValueError("Head and Tail scores must align by seed and candidate.")
    points, samples = zip(*(metric_samples(pool, pair, w, "disagreement") for pair in zip(head, tail)))
    point, draws = float(np.mean(points)), np.mean(samples, axis=0)
    _emit({"point": point, "interval": percentile_intervals(draws[:, None])[0].tolist()}, args.output)


def _budget(args):
    pools = []
    for path in args.input:
        arrays = _arrays(path)
        pools.append((path.stem, _pool(arrays), arrays["lengths"], {arm: arrays[arm] for arm in ARMS}))
    _emit(replay_budget(pools, args.width, args.draws, args.seed), args.output)


def _access(args):
    lengths = np.load(_input(args.lengths), allow_pickle=False)
    _emit({arm: state_access(lengths, arm, args.width) for arm in ARMS}, args.output)


def _fit(args):
    fit = fit_p32(np.load(_input(args.source), allow_pickle=False), np.load(_input(args.target), allow_pickle=False))
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **asdict(fit))


def _apply(args):
    fit = PhaseMap(**_arrays(args.map))
    _save_array(args.output, fit.transform(np.load(_input(args.source), allow_pickle=False)))


def _score(args):
    from .readout import SparseLME, load_weights, score_candidates

    model_path = args.model_path or os.environ.get("MODEL_PATH")
    if model_path is None:
        raise ValueError("Provide --model-path or set MODEL_PATH to a trusted local readout.")
    model = SparseLME(args.width)
    load_weights(model, Path(model_path))
    _save_array(args.output, score_candidates(model, _packed(args), args.view, args.batch_size, args.denominator))


def _train(args):
    import torch
    from .readout import SparseLME, load_weights
    from .training import train_readout

    torch.manual_seed(args.seed)
    model = SparseLME(args.width)
    if args.initial is not None:
        load_weights(model, _input(args.initial))
    train_readout(model, _packed(args, "train_"), _pool(_arrays(args.train_binding)),
                  _packed(args, "dev_"), _pool(_arrays(args.dev_binding)),
                  np.load(_input(args.pair_orders), allow_pickle=False), args.arm, args.view,
                  args.denominator, args.seed, args.output)


def parser():
    root = argparse.ArgumentParser(prog="tpami3", description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    views = sub.add_parser("views", help="Show deterministic indices for a response length.")
    views.add_argument("length", type=int)
    views.add_argument("--denominator", type=int, default=4)
    views.set_defaults(handler=_views)

    stats = sub.add_parser("stats", help="Evaluate aligned scores with paired question bootstrap.")
    _path(stats, "--input", required=True)
    _path(stats, "--output")
    stats.add_argument("--score", action="append", metavar="NPZ_KEY")
    stats.add_argument("--contrast", action="append", nargs=3, default=[], metavar=("NAME", "LEFT", "RIGHT"))
    stats.add_argument("--interval", choices=("percentile", "bonferroni", "max-t"), default="percentile")
    _draw_options(stats)
    stats.set_defaults(handler=_stats)

    gap = sub.add_parser("gap", help="Evaluate aligned margin-sign disagreement.")
    _path(gap, "--input", required=True)
    _path(gap, "--output")
    gap.add_argument("--head-key", default="head")
    gap.add_argument("--tail-key", default="tail")
    _draw_options(gap)
    gap.set_defaults(handler=_gap)

    budget = sub.add_parser("budget", help="Recompute a complete Task44-style budget family.")
    _path(budget, "--input", required=True, action="append", help="Pool NPZ in the intended resampling order.")
    _path(budget, "--output")
    budget.add_argument("--width", type=int, required=True)
    _draw_options(budget)
    budget.set_defaults(handler=_budget)

    access = sub.add_parser("access", help="Calculate the state read counts for each budget arm.")
    _path(access, "--lengths", required=True)
    _path(access, "--output")
    access.add_argument("--width", type=int, required=True)
    access.set_defaults(handler=_access)

    fit = sub.add_parser("p32-fit", help="Fit P32 on explicitly supplied matched training representations.")
    for flag in ("--source", "--target", "--output"):
        _path(fit, flag, required=True)
    fit.set_defaults(handler=_fit)
    apply = sub.add_parser("p32-apply", help="Apply a stored P32 map to supplied representations.")
    for flag in ("--source", "--map", "--output"):
        _path(apply, flag, required=True)
    apply.set_defaults(handler=_apply)

    score = sub.add_parser("score", help="Apply a local frozen head to explicit external state rows.")
    _state_options(score)
    _path(score, "--model-path", help="Trusted local tensor checkpoint. Defaults to MODEL_PATH.")
    _path(score, "--output", required=True)
    score.add_argument("--batch-size", type=int, default=48)
    score.set_defaults(handler=_score)
    train = sub.add_parser("train", help="Fit a sparse or C0/C1 readout with fixed external pair orders.")
    _state_options(train, "train-")
    _state_options(train, "dev-")
    for flag in ("--train-binding", "--dev-binding", "--pair-orders", "--output"):
        _path(train, flag, required=True)
    _path(train, "--initial", help="Trusted local starting tensors for the readout.")
    train.add_argument("--arm", choices=("sparse", "C0", "C1"), required=True)
    train.add_argument("--seed", type=int, choices=(42, 123, 456), required=True)
    train.set_defaults(handler=_train)
    for command in (score, train):
        command.add_argument("--width", type=int, required=True)
        command.add_argument("--view", choices=VIEWS, default="bin_head")
        command.add_argument("--denominator", type=int, default=4)
    return root


def main(argv=None):
    builder = parser()
    args = builder.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        builder.error(str(error))
    return 0
