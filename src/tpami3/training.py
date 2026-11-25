"""Train a readout using external, already fixed ranking-pair orders."""

import json

import numpy as np
import torch
from torch.nn import functional as F

from .readout import CONSISTENCY, dual_view_loss, score_candidates


SEEDS = (42, 123, 456)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_PAIRS = 32
CLIP_NORM = 1.0
EPOCHS = 5
PATIENCE = 2


def validate_orders(orders, pool):
    orders = np.asarray(orders)
    if orders.ndim == 2:
        orders = orders[None, ...]
    if (
        orders.ndim != 3 or orders.shape[0] not in (1, EPOCHS) or not orders.shape[1]
        or orders.shape[2] != 2 or orders.dtype.kind not in "iu"
        or (orders < 0).any() or (orders >= len(pool.labels)).any()
    ):
        raise ValueError("Expected valid [pair, 2] indices or one such order per epoch.")
    positive, negative = orders[..., 0], orders[..., 1]
    if (
        not pool.labels[positive].all() or pool.labels[negative].any()
        or not (pool.question_index[positive] == pool.question_index[negative]).all()
    ):
        raise ValueError("Each pair must be a positive and negative from the same original question.")
    return orders


def train_readout(model, train_states, train_pool, dev_states, dev_pool, orders,
                  arm, view, denominator, seed, output):
    if len(train_states) != len(train_pool.labels) or len(dev_states) != len(dev_pool.labels):
        raise ValueError("State rows and pool bindings must contain the same number of candidates.")
    if train_states.width != dev_states.width or arm not in ("sparse", *CONSISTENCY):
        raise ValueError("Expected aligned state widths and a sparse, C0 or C1 arm.")
    orders = validate_orders(orders, train_pool)
    views = (view,) if arm == "sparse" else ("bin_head", "bin_tail")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    output.mkdir()
    best, selected, history = -np.inf, None, []

    for epoch in range(EPOCHS):
        model.train()
        order = orders[0 if len(orders) == 1 else epoch]
        for start in range(0, len(order), BATCH_PAIRS):
            positive, negative = order[start:start + BATCH_PAIRS].T
            rows = np.concatenate((positive, negative))
            optimizer.zero_grad(set_to_none=True)
            gaps = []
            for current in views:
                features, mask = train_states.batch(rows, current, denominator)
                scores = model(torch.from_numpy(features), torch.from_numpy(mask))
                gaps.append(scores[:len(positive)] - scores[len(positive):])
            loss = F.softplus(-gaps[0]).mean() if arm == "sparse" else dual_view_loss(*gaps, arm)[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=True)
            optimizer.step()

        dev_scores = {current: score_candidates(model, dev_states, current, denominator=denominator)
                      for current in views}
        per_view = {}
        for current, values in dev_scores.items():
            scores = dev_pool.per_question_pair(values)
            scores = scores[np.isfinite(scores)]
            if not len(scores):
                raise ValueError("Development Pair-Macro requires a mixed-label question.")
            per_view[current] = float(scores.mean())
        score = float(np.mean(list(per_view.values())))
        record = {"epoch": epoch + 1, "strict_pair_macro": per_view, "selection": score}
        history.append(record)
        epoch_name = f"epoch_{epoch + 1:02d}"
        with (output / (epoch_name + ".pt")).open("xb") as stream:
            torch.save({"head": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "torch_rng_state": torch.get_rng_state(), "seed": seed, **record}, stream)
        with (output / (epoch_name + "_dev.npz")).open("xb") as stream:
            np.savez_compressed(stream, **dev_scores)
        if score > best:
            best, selected = score, epoch + 1
        if epoch + 1 - selected >= PATIENCE:
            break
    with (output / "selection.json").open("x", encoding="utf-8") as stream:
        json.dump({"arm": arm, "views": list(views), "denominator": denominator,
                   "seed": seed, "selected_epoch": selected, "history": history},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")
