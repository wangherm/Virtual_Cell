# EVOLVE-BLOCK-START
import numpy as np


def fit_predict(data, budget):
    """
    Hierarchical ridge per-gene shrinkage blended with global support scaling.

    Steps:
    1. Compute a per-gene ridge coefficient shrunken toward the global OLS scalar.
       The ridge penalty is selected by 5-fold cross-validation on training rows.
    2. Compute a global OLS scalar (same as baseline) using all train+cal.
    3. Choose a blending weight w in [0,1] on the calibration set by minimizing
       squared error between the weighted sum of global and ridge predictions and
       the calibration labels.
    4. Apply the same weighted combination to query support.
    """
    # --- legal arrays ---
    train_y = np.asarray(data['train_y'], dtype=np.float64)
    cal_y = np.asarray(data['calibration_y'], dtype=np.float64)
    train_s = np.asarray(data['train_support'], dtype=np.float64)
    cal_s = np.asarray(data['calibration_support'], dtype=np.float64)
    query_s = np.asarray(data['query_support'], dtype=np.float64)

    Y_all = np.vstack([train_y, cal_y])
    S_all = np.vstack([train_s, cal_s])
    n_all, G = Y_all.shape

    # --- global scalar from all train+cal (baseline alpha) ---
    sum_s2_all = np.sum(S_all ** 2)
    sum_sy_all = np.sum(S_all * Y_all)
    alpha_global_scalar = sum_sy_all / max(sum_s2_all, 1e-12)

    # --- per-gene sufficient statistics for all train+cal ---
    sum_s2_g_all = np.sum(S_all ** 2, axis=0)   # (G,)
    sum_sy_g_all = np.sum(S_all * Y_all, axis=0) # (G,)

    # --- cross-validation on training rows to choose ridge penalty lambda ---
    train_n = train_y.shape[0]
    rng = np.random.RandomState(42)
    indices = np.arange(train_n)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    lam_grid = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0,
                         1e6, 1e7], dtype=np.float64)
    best_lam = lam_grid[0]
    best_cv = np.inf

    for lam in lam_grid:
        total_mse = 0.0
        total_count = 0
        for fold in folds:
            val_idx = fold
            tr_idx = np.setdiff1d(indices, val_idx)
            tr_s = train_s[tr_idx]
            tr_y = train_y[tr_idx]
            val_s = train_s[val_idx]
            val_y = train_y[val_idx]

            # per-gene sufficient stats on inner training fold
            inner_sum_s2 = np.sum(tr_s ** 2, axis=0)   # (G,)
            inner_sum_sy = np.sum(tr_s * tr_y, axis=0) # (G,)
            inner_alpha_global = np.sum(inner_sum_sy) / max(np.sum(inner_sum_s2), 1e-12)

            # shrunken per-gene coefficients for this lambda
            denom = inner_sum_s2 + lam
            num = inner_sum_sy + lam * inner_alpha_global
            alpha_g = num / denom

            # predict validation rows
            pred = val_s * alpha_g[None, :]
            total_mse += np.sum((pred - val_y) ** 2)
            total_count += val_y.size

        cv_mse = total_mse / total_count if total_count > 0 else np.inf
        if cv_mse < best_cv - 1e-12:
            best_cv = cv_mse
            best_lam = lam

    # --- refit per-gene ridge on all train+cal using selected lambda ---
    denom_final = sum_s2_g_all + best_lam
    num_final = sum_sy_g_all + best_lam * alpha_global_scalar
    alpha_g_ridge = num_final / denom_final

    # --- build predictions on calibration for blending ---
    pred_global_cal = alpha_global_scalar * cal_s
    pred_ridge_cal = alpha_g_ridge[None, :] * cal_s

    # --- choose blending weight w by grid search on calibration MSE ---
    w_grid = np.linspace(0.0, 1.0, 101)
    best_w = 0.0
    best_mse_cal = np.inf
    for w in w_grid:
        pred_blend = w * pred_ridge_cal + (1.0 - w) * pred_global_cal
        mse = np.mean((pred_blend - cal_y) ** 2)
        if mse < best_mse_cal:
            best_mse_cal = mse
            best_w = w

    # --- predict query using same blend ---
    pred_global_q = alpha_global_scalar * query_s
    pred_ridge_q = alpha_g_ridge[None, :] * query_s
    pred_q = best_w * pred_ridge_q + (1.0 - best_w) * pred_global_q

    pred_q = np.nan_to_num(pred_q, nan=0.0, posinf=0.0, neginf=0.0)
    return pred_q.astype(np.float32)
# EVOLVE-BLOCK-END
