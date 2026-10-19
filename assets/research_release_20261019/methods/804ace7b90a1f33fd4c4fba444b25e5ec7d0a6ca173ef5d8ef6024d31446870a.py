# EVOLVE-BLOCK-START
import numpy as np


def fit_predict(data, budget):
    # --- combine legal training and calibration sources ---
    train_y = np.asarray(data['train_y'], dtype=np.float64)
    cal_y = np.asarray(data['calibration_y'], dtype=np.float64)
    train_s = np.asarray(data['train_support'], dtype=np.float64)
    cal_s = np.asarray(data['calibration_support'], dtype=np.float64)
    train_t = np.asarray(data['train_targets'], dtype=str)
    cal_t = np.asarray(data['calibration_targets'], dtype=str)
    train_c = np.asarray(data['train_source_context'], dtype=str)
    cal_c = np.asarray(data['calibration_source_context'], dtype=str)

    Y = np.vstack([train_y, cal_y])
    S = np.vstack([train_s, cal_s])
    targets = np.concatenate([train_t, cal_t])
    contexts = np.concatenate([train_c, cal_c])

    query_s = np.asarray(data['query_support'], dtype=np.float64)
    query_targets = np.asarray(data['query_targets'], dtype=str)
    query_contexts = np.asarray(data['query_source_context'], dtype=str)

    n, G = Y.shape
    nq = query_s.shape[0]

    # --- hyperparameter grids ---
    ridge_scales = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], dtype=np.float64)
    residual_scales = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0], dtype=np.float64)

    # --- helper: fit from a training subset and predict a validation subset ---
    def fit_predict_subset(
        S_tr, Y_tr, T_tr, C_tr,
        S_val, T_val, C_val,
        ridge_scale, residual_scale
    ):
        n_tr = S_tr.shape[0]
        if n_tr == 0:
            return np.zeros_like(S_val, dtype=np.float64)

        # global least squares coefficient
        sum_sq = np.sum(S_tr * S_tr)
        sum_sy = np.sum(Y_tr * S_tr)
        if sum_sq > 1e-12:
            alpha0 = float(sum_sy / sum_sq)
        else:
            alpha0 = 0.0
        alpha0 = max(min(alpha0, 10.0), -10.0)

        # per-gene sufficient statistics
        r = np.sum(Y_tr * S_tr, axis=0)               # (G,)
        d = np.sum(S_tr * S_tr, axis=0) + 1e-12       # (G,)

        # unregularized per-gene least squares and noise variance estimate
        alpha_ls = r / d
        Y_sq = np.sum(Y_tr * Y_tr, axis=0)
        SSE = np.maximum(Y_sq - r * alpha_ls, 0.0)
        sigma2 = np.maximum(SSE / max(n_tr - 1, 1), 1e-12)

        # prior variance of per-gene coefficients
        tau2 = float(np.var(alpha_ls))
        tau2 = max(tau2, 1e-12)

        # per-gene empirical Bayes shrinkage penalty
        lam = ridge_scale * sigma2 / tau2
        A = (r + lam * alpha0) / (d + lam)

        # residual offsets for target-context pairs
        base_tr = S_tr * A[None, :]
        resid_tr = Y_tr - base_tr

        # aggregate residuals by target-context and by target
        offset_tc = {}
        count_tc = {}
        offset_t = {}
        count_t = {}

        for i in range(n_tr):
            t = T_tr[i]
            c = C_tr[i]
            key = (t, c)
            if key not in offset_tc:
                offset_tc[key] = np.zeros(G, dtype=np.float64)
                count_tc[key] = 0
            offset_tc[key] += resid_tr[i]
            count_tc[key] += 1

            if t not in offset_t:
                offset_t[t] = np.zeros(G, dtype=np.float64)
                count_t[t] = 0
            offset_t[t] += resid_tr[i]
            count_t[t] += 1

        # convert sums to means
        for key in offset_tc:
            offset_tc[key] /= count_tc[key]
        for t in offset_t:
            offset_t[t] /= count_t[t]

        # predict validation subset
        pred_val = S_val * A[None, :]
        for i in range(len(T_val)):
            t = T_val[i]
            c = C_val[i]
            key = (t, c)
            if key in offset_tc:
                off = offset_tc[key]
            elif t in offset_t:
                off = offset_t[t]
            else:
                off = np.zeros(G, dtype=np.float64)
            pred_val[i] += residual_scale * off

        return pred_val

    # --- 5-fold cross-validation to select ridge_scale and residual_scale ---
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    best_score = float('inf')
    best_ridge = 1.0
    best_resid = 0.0

    for rs in ridge_scales:
        for res in residual_scales:
            fold_mses = []
            for fold in folds:
                val_idx = fold
                tr_idx = np.setdiff1d(indices, val_idx)

                pred_val = fit_predict_subset(
                    S[tr_idx], Y[tr_idx], targets[tr_idx], contexts[tr_idx],
                    S[val_idx], targets[val_idx], contexts[val_idx],
                    rs, res
                )

                mse = float(np.mean((pred_val - Y[val_idx]) ** 2))
                fold_mses.append(mse)

            avg_mse = float(np.mean(fold_mses))
            if avg_mse < best_score:
                best_score = avg_mse
                best_ridge = rs
                best_resid = res

    # --- final fit on all train+calibration with selected hyperparameters ---
    # compute global coefficient
    sum_sq = np.sum(S * S)
    sum_sy = np.sum(Y * S)
    if sum_sq > 1e-12:
        alpha0 = float(sum_sy / sum_sq)
    else:
        alpha0 = 0.0
    alpha0 = max(min(alpha0, 10.0), -10.0)

    r = np.sum(Y * S, axis=0)
    d = np.sum(S * S, axis=0) + 1e-12
    alpha_ls = r / d
    Y_sq = np.sum(Y * Y, axis=0)
    SSE = np.maximum(Y_sq - r * alpha_ls, 0.0)
    sigma2 = np.maximum(SSE / max(n - 1, 1), 1e-12)
    tau2 = float(np.var(alpha_ls))
    tau2 = max(tau2, 1e-12)
    lam = best_ridge * sigma2 / tau2
    A_final = (r + lam * alpha0) / (d + lam)

    # residual offsets from full data
    base_full = S * A_final[None, :]
    resid_full = Y - base_full

    offset_tc_full = {}
    count_tc_full = {}
    offset_t_full = {}
    count_t_full = {}

    for i in range(n):
        t = targets[i]
        c = contexts[i]
        key = (t, c)
        if key not in offset_tc_full:
            offset_tc_full[key] = np.zeros(G, dtype=np.float64)
            count_tc_full[key] = 0
        offset_tc_full[key] += resid_full[i]
        count_tc_full[key] += 1

        if t not in offset_t_full:
            offset_t_full[t] = np.zeros(G, dtype=np.float64)
            count_t_full[t] = 0
        offset_t_full[t] += resid_full[i]
        count_t_full[t] += 1

    for key in offset_tc_full:
        offset_tc_full[key] /= count_tc_full[key]
    for t in offset_t_full:
        offset_t_full[t] /= count_t_full[t]

    # predict query
    pred = query_s * A_final[None, :]
    for i in range(nq):
        t = query_targets[i]
        c = query_contexts[i]
        key = (t, c)
        if key in offset_tc_full:
            off = offset_tc_full[key]
        elif t in offset_t_full:
            off = offset_t_full[t]
        else:
            off = np.zeros(G, dtype=np.float64)
        pred[i] += best_resid * off

    # --- finite, clipped, float32 output ---
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    max_abs = np.percentile(np.abs(Y), 99.9) * 3.0
    if max_abs > 0:
        pred = np.clip(pred, -max_abs, max_abs)

    return pred.astype(np.float32)
# EVOLVE-BLOCK-END
