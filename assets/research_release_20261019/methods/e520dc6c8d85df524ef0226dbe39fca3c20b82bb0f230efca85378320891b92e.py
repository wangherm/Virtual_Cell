# EVOLVE-BLOCK-START
import numpy as np


def fit_predict(data, budget):
    """
    Target-shrinkage ridge (TSR):
    - Compute global scalar alpha mapping support to response on train+cal.
    - For each target, compute a ridge estimate of alpha_target using all rows
      of that target, shrunk towards the global alpha.
    - The shrinkage strength lambda is selected by 5-fold cross-validation
      on train+cal rows only (never on query labels).
    - Predict query as alpha_target * query_support.
    If lambda becomes very large, the model reduces to the global calibrated
    support baseline, which is known to be strong.
    """
    # Load legal arrays
    train_y = np.asarray(data['train_y'], dtype=np.float64)
    cal_y = np.asarray(data['calibration_y'], dtype=np.float64)
    train_s = np.asarray(data['train_support'], dtype=np.float64)
    cal_s = np.asarray(data['calibration_support'], dtype=np.float64)
    query_s = np.asarray(data['query_support'], dtype=np.float64)

    train_targets = np.asarray(data['train_targets'])
    cal_targets = np.asarray(data['calibration_targets'])
    query_targets = np.asarray(data['query_targets'])

    # Combine train and calibration as permitted labelled source
    Y = np.vstack([train_y, cal_y])
    S = np.vstack([train_s, cal_s])
    targets = np.concatenate([train_targets, cal_targets])

    n, G = Y.shape
    nq = query_s.shape[0]

    # Global scalar alpha (same as baseline)
    ssq_global = np.sum(S ** 2)
    sy_global = np.sum(S * Y)
    alpha_global = sy_global / max(ssq_global, 1e-12)

    # Candidate shrinkage strengths for target-specific alphas
    lam_grid = np.array([0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0,
                         1e4, 1e5, 1e6], dtype=np.float64)

    # 5-fold cross-validation on train+cal rows to select lambda
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    best_lam = lam_grid[-1]  # default to global model
    best_cv_mse = np.inf

    for lam in lam_grid:
        total_mse = 0.0
        total_count = 0

        for fold in folds:
            val_idx = fold
            tr_idx = np.setdiff1d(indices, val_idx)

            S_tr = S[tr_idx]
            Y_tr = Y[tr_idx]
            targets_tr = targets[tr_idx]

            # Compute global alpha on inner training
            ssq_tr = np.sum(S_tr ** 2)
            sy_tr = np.sum(S_tr * Y_tr)
            alpha_tr_global = sy_tr / max(ssq_tr, 1e-12)

            # Precompute per-target sufficient statistics on inner training
            unique_tr = np.unique(targets_tr)
            target_stats = {}
            for t in unique_tr:
                mask = targets_tr == t
                S_t = S_tr[mask]
                Y_t = Y_tr[mask]
                s2_t = np.sum(S_t ** 2)
                sy_t = np.sum(S_t * Y_t)
                # Ridge estimate with shrinkage to global
                denom = s2_t + lam
                num = sy_t + lam * alpha_tr_global
                alpha_t = num / denom if denom > 1e-12 else alpha_tr_global
                target_stats[t] = alpha_t

            # Validate on held-out rows
            S_val = S[val_idx]
            Y_val = Y[val_idx]
            targets_val = targets[val_idx]

            for i in range(len(val_idx)):
                t = targets_val[i]
                # If target not seen in training, use global alpha
                alpha_i = target_stats.get(t, alpha_tr_global)
                pred = S_val[i] * alpha_i
                total_mse += np.sum((pred - Y_val[i]) ** 2)
                total_count += G

        cv_mse = total_mse / max(total_count, 1)
        if cv_mse < best_cv_mse - 1e-12:
            best_cv_mse = cv_mse
            best_lam = lam

    # Refit on all train+cal with selected lambda
    # Global alpha already computed as alpha_global
    unique_all = np.unique(targets)
    alpha_targets = {}
    for t in unique_all:
        mask = targets == t
        S_t = S[mask]
        Y_t = Y[mask]
        s2_t = np.sum(S_t ** 2)
        sy_t = np.sum(S_t * Y_t)
        denom = s2_t + best_lam
        num = sy_t + best_lam * alpha_global
        alpha_t = num / denom if denom > 1e-12 else alpha_global
        alpha_targets[t] = alpha_t

    # Predict query
    pred_query = np.zeros((nq, G), dtype=np.float64)
    for i in range(nq):
        t = query_targets[i]
        a = alpha_targets.get(t, alpha_global)
        pred_query[i] = query_s[i] * a

    # Safety clip and cast
    pred_query = np.nan_to_num(pred_query, nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = np.percentile(np.abs(Y), 99.9) * 3.0
    if max_abs > 0:
        pred_query = np.clip(pred_query, -max_abs, max_abs)

    return pred_query.astype(np.float32)
# EVOLVE-BLOCK-END
