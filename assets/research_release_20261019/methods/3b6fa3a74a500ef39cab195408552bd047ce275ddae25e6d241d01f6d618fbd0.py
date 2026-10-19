# EVOLVE-BLOCK-START
import numpy as np


def fit_predict(data, budget):
    """
    Novel target/context scalar ridge model.

    The core assumption is y_i = alpha_i * s_i, where alpha_i is a row-specific
    scalar that may vary by source context and target. We estimate alpha_i per
    training row by least squares, then shrink these estimates toward a global
    alpha using ridge regression on one-hot context/target features. The penalty
    is selected by 5-fold cross-validation on the original MSE. This transforms
    the problem from 1809 gene-level regressions to a low-dimensional random
    effects model on scalars, avoiding per-gene overfitting while still adapting
    to donor/target magnitude differences.
    """
    # --- load and combine permitted sources ---
    train_y = np.asarray(data['train_y'], dtype=np.float64)
    cal_y = np.asarray(data['calibration_y'], dtype=np.float64)
    train_s = np.asarray(data['train_support'], dtype=np.float64)
    cal_s = np.asarray(data['calibration_support'], dtype=np.float64)
    train_ctx = np.asarray(data['train_source_context'], dtype=str)
    cal_ctx = np.asarray(data['calibration_source_context'], dtype=str)
    train_tgt = np.asarray(data['train_targets'], dtype=str)
    cal_tgt = np.asarray(data['calibration_targets'], dtype=str)

    query_s = np.asarray(data['query_support'], dtype=np.float64)
    query_ctx = np.asarray(data['query_source_context'], dtype=str)
    query_tgt = np.asarray(data['query_targets'], dtype=str)

    def _clean(s):
        return s.strip()

    Y_all = np.vstack([train_y, cal_y])
    S_all = np.vstack([train_s, cal_s])
    ctx_all = np.array([_clean(c) for c in np.concatenate([train_ctx, cal_ctx])])
    tgt_all = np.array([_clean(t) for t in np.concatenate([train_tgt, cal_tgt])])
    n, G = Y_all.shape

    # --- row-wise scalar estimates ---
    sum_s2 = np.sum(S_all ** 2, axis=1)
    sum_sy = np.sum(S_all * Y_all, axis=1)
    valid_mask = sum_s2 > 1e-12
    # alpha_i = sum(s*y) / sum(s^2) per row (OLS)
    alpha_i = np.zeros(n, dtype=np.float64)
    alpha_i[valid_mask] = sum_sy[valid_mask] / np.maximum(sum_s2[valid_mask], 1e-12)

    # global alpha using all valid rows
    total_s2 = np.sum(sum_s2[valid_mask])
    total_sy = np.sum(sum_sy[valid_mask])
    global_alpha_all = total_sy / total_s2 if total_s2 > 1e-12 else 0.0

    # --- create design matrix for context/target indicators ---
    unique_ctx = np.unique(ctx_all)
    unique_tgt = np.unique(tgt_all)
    C = len(unique_ctx)
    T = len(unique_tgt)
    ctx_to_idx = {c: i for i, c in enumerate(unique_ctx)}
    tgt_to_idx = {t: i for i, t in enumerate(unique_tgt)}

    X_all = np.zeros((n, C + T), dtype=np.float64)
    for i in range(n):
        if ctx_all[i] in ctx_to_idx:
            X_all[i, ctx_to_idx[ctx_all[i]]] = 1.0
        if tgt_all[i] in tgt_to_idx:
            X_all[i, C + tgt_to_idx[tgt_all[i]]] = 1.0

    # --- cross-validation on original MSE to select ridge penalty ---
    rng = np.random.RandomState(42)
    all_indices = np.arange(n)
    rng.shuffle(all_indices)
    folds = np.array_split(all_indices, 5)

    lam_grid = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6, 1e7, 1e8],
                        dtype=np.float64)
    best_lam = 1e6  # default to heavy shrinkage (near global alpha)
    best_cv = np.inf

    for lam in lam_grid:
        total_mse = 0.0
        total_count = 0
        for fold in folds:
            val_idx = fold
            tr_idx = np.setdiff1d(all_indices, val_idx)

            # use only valid rows for fitting scalar regression
            tr_valid_all = tr_idx[valid_mask[tr_idx]]
            if len(tr_valid_all) == 0:
                continue

            alpha_g_tr = np.sum(sum_sy[tr_valid_all]) / max(np.sum(sum_s2[tr_valid_all]), 1e-12)
            z_tr = alpha_i[tr_valid_all] - alpha_g_tr
            X_tr = X_all[tr_valid_all]

            # ridge solve: (X'X + lam*I) coef = X'z
            A = X_tr.T @ X_tr + lam * np.eye(X_tr.shape[1])
            b = X_tr.T @ z_tr
            coef = np.linalg.solve(A, b)

            # predict all validation rows, including zero-support rows
            for i in val_idx:
                z_pred = X_all[i] @ coef
                alpha_pred = alpha_g_tr + z_pred
                pred = alpha_pred * S_all[i]
                total_mse += np.sum((pred - Y_all[i]) ** 2)
                total_count += G

        if total_count > 0:
            cv_mse = total_mse / total_count
        else:
            cv_mse = np.inf

        if cv_mse < best_cv - 1e-12:
            best_cv = cv_mse
            best_lam = lam

    # --- refit on all valid training rows with selected lambda ---
    valid_all = np.where(valid_mask)[0]
    alpha_i_valid = alpha_i[valid_all]
    z_valid = alpha_i_valid - global_alpha_all
    X_valid = X_all[valid_all]

    A = X_valid.T @ X_valid + best_lam * np.eye(X_valid.shape[1])
    b = X_valid.T @ z_valid
    coef_final = np.linalg.solve(A, b)

    # --- predict query ---
    query_s = np.asarray(data['query_support'], dtype=np.float64)
    query_ctx_clean = np.array([_clean(c) for c in query_ctx])
    query_tgt_clean = np.array([_clean(t) for t in query_tgt])
    nq = query_s.shape[0]
    preds = np.zeros_like(query_s, dtype=np.float64)

    for i in range(nq):
        # build query feature row (zeros if context/target unseen)
        x = np.zeros(C + T, dtype=np.float64)
        if query_ctx_clean[i] in ctx_to_idx:
            x[ctx_to_idx[query_ctx_clean[i]]] = 1.0
        if query_tgt_clean[i] in tgt_to_idx:
            x[C + tgt_to_idx[query_tgt_clean[i]]] = 1.0

        z_q = x @ coef_final
        alpha_q = global_alpha_all + z_q
        preds[i] = alpha_q * query_s[i]

    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    return preds.astype(np.float32)
# EVOLVE-BLOCK-END
