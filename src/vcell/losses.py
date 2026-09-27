import torch
from itertools import combinations
from torch.nn import functional as F


def cross_student_contrastive(z_a, z_b, perturbations, temperature=.2):
    """Same row across students is positive. Other rows of the same target are excluded.

    Distinct targets are only approximate negatives (they can share pathways).
    No claim of biological class labels. All-masked batches return zero safely.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    n = len(z_a)
    if n < 2:
        return (z_a.sum() + z_b.sum()) * 0
    logits = F.normalize(z_a, dim=-1) @ F.normalize(z_b, dim=-1).T / temperature
    same_target = perturbations[:, None] == perturbations[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=z_a.device)
    allowed = ~same_target | eye
    valid = (~same_target).any(dim=1)
    if not valid.any():
        return (z_a.sum() + z_b.sum()) * 0
    logits = logits.masked_fill(~allowed, -1e4)
    labels = torch.arange(n, device=z_a.device)
    a = F.cross_entropy(logits[valid], labels[valid])
    b = F.cross_entropy(logits.T[valid], labels[valid])
    return (a + b) / 2


def mutual_loss(a, b):
    # Each network receives a fixed peer target within this optimization step.
    return .5 * (F.mse_loss(a, b.detach()) + F.mse_loss(b, a.detach()))


def student_losses(outputs, perturbations, temperature, use_peer, use_contrast):
    """Average over pairs, so adding students does not amplify loss weights."""
    pairs = list(combinations(outputs, 2))
    zero = outputs[0][0].sum() * 0
    peer = torch.stack([mutual_loss(a[0], b[0]) for a, b in pairs]).mean() if use_peer and pairs else zero
    contrast = torch.stack([cross_student_contrastive(a[1], b[1], perturbations, temperature)
                            for a, b in pairs]).mean() if use_contrast and pairs else zero
    return peer, contrast
