"""Validation objective and ensemble selection shared by training and agent."""
from itertools import combinations
import numpy as np
from scipy.optimize import minimize


def row_weights(meta):
    # Equal contexts, equal perturbations within context, equal batches per perturbation.
    counts = meta.groupby(["context", "perturbation"]).row_id.transform("count").to_numpy()
    per_context = meta.groupby("context").perturbation.transform("nunique").to_numpy()
    return 1. / (meta.context.nunique() * per_context * counts)


def mse(pred, truth, weights):
    return float(np.sum(np.mean((pred - truth) ** 2, axis=1) * weights))


def fit_mix(predictions, truth, weights):
    """Small convex simplex fit. Inputs are validation rows only."""
    errors = np.stack(predictions).astype(np.float64) - truth
    gram = np.einsum("irg,jrg,r->ij", errors, errors, weights) / truth.shape[1]
    n = len(predictions)
    result = minimize(lambda w: float(w @ gram @ w), np.full(n, 1. / n),
                      jac=lambda w: 2 * gram @ w, method="SLSQP",
                      bounds=[(0., 1.)] * n,
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                                   "jac": lambda w: np.ones(n)},
                      options={"ftol": 1e-10, "maxiter": 100})
    if not result.success:
        raise RuntimeError(result.message)
    w = np.maximum(result.x, 0)
    return w / w.sum()


def combine(predictions, weights):
    return np.einsum("i,irg->rg", weights, np.stack(predictions)).astype(np.float32)


def summarize(predictions, truth, meta):
    """No test data or raw expression is returned to the agent."""
    names, arrays = list(predictions), list(predictions.values())
    rw = row_weights(meta)
    scores = {name: mse(pred, truth, rw) for name, pred in predictions.items()}
    weights = fit_mix(arrays, truth, rw)
    equal_score = mse(np.mean(arrays, axis=0), truth, rw)
    mixed_score = mse(combine(arrays, weights), truth, rw)
    pairs = []
    for i, j in combinations(range(len(arrays)), 2):
        ea, eb = (arrays[i] - truth).ravel(), (arrays[j] - truth).ravel()
        corr = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 1e-12 and eb.std() > 1e-12 else None
        pairs.append({"a": names[i], "b": names[j], "error_correlation": corr,
                      "disagreement_rmse": float(np.sqrt(np.mean((arrays[i] - arrays[j]) ** 2)))})
    return {"individual_mse": scores, "ranking": sorted(scores, key=scores.get),
            "mean_mse": equal_score, "valmix_mse": mixed_score,
            "ensemble_gain_over_best_individual": min(scores.values()) - mixed_score,
            "weights": dict(zip(names, weights.tolist())), "pairs": pairs}

