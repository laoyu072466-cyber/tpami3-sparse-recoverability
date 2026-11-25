"""Sparse LogMeanExp readout and shared C0/C1 objectives."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CONSISTENCY = {"C0": 0.0, "C1": 0.1}


class SparseLME(nn.Module):
    def __init__(self, width):
        super().__init__()
        if width < 1:
            raise ValueError("State width must be positive.")
        self.proj = nn.Linear(width, 128, bias=False)
        self.proj_ln = nn.LayerNorm(128)
        self.phi = nn.Sequential(nn.Linear(128, 32), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(32, 186), nn.ReLU(), nn.Linear(186, 1))

    def pooled(self, states, valid):
        """Pool compact token rows in the row-major order of true mask entries."""
        if (
            valid.ndim != 2 or valid.dtype != torch.bool or not valid.numel()
            or not valid.any(dim=1).all() or states.ndim != 2
            or states.shape[1] != self.proj.in_features or len(states) != int(valid.sum())
        ):
            raise ValueError("Expected compact token rows matching a nonempty boolean mask.")
        features = self.phi(self.proj_ln(self.proj(states)))
        padded = features.new_zeros((*valid.shape, features.shape[-1]))
        padded[valid] = features
        masked = padded.masked_fill(~valid[..., None], -torch.inf)
        return torch.logsumexp(masked, dim=1) - valid.sum(dim=1).float().log()[:, None]

    def forward(self, states, valid):
        return self.rho(self.pooled(states, valid)).squeeze(-1)


def load_weights(model, model_path):
    """Load a tensor state dict or the head tensors of a compatible checkpoint."""
    weight_data = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(weight_data, dict):
        raise ValueError("Expected a readout state dictionary.")
    model.load_state_dict(weight_data.get("head", weight_data), strict=True)


def dual_view_loss(head_margin, tail_margin, arm):
    if (arm not in CONSISTENCY or head_margin.shape != tail_margin.shape
            or head_margin.ndim != 1 or not head_margin.numel()):
        raise ValueError("Expected a C0/C1 arm and aligned, nonempty margin vectors.")
    rank_head = F.softplus(-head_margin).mean()
    rank_tail = F.softplus(-tail_margin).mean()
    weight = CONSISTENCY[arm]
    with torch.set_grad_enabled(torch.is_grad_enabled() and bool(weight)):
        consistency = F.huber_loss(head_margin, tail_margin, delta=1.0)
    loss = (rank_head + rank_tail) / 2 + weight * consistency
    return loss, {"rank_head": rank_head, "rank_tail": rank_tail, "huber": consistency}


def score_candidates(model, packed, view, batch_size=48, denominator=4):
    if batch_size < 1:
        raise ValueError("Candidate batch size must be positive.")
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(packed), batch_size):
            rows = range(start, min(start + batch_size, len(packed)))
            states, mask = packed.batch(rows, view, denominator)
            chunks.append(model(torch.from_numpy(states), torch.from_numpy(mask)).numpy())
    return np.concatenate(chunks)
