"""Return original-unit, ordered gene mean-deltas. Only evolve the marked method."""
import numpy as np
import torch

# EVOLVE-BLOCK-START

def _safe_str(x):
    if x is None:
        return None
    if isinstance(x, bytes):
        x = x.decode('utf-8', 'ignore')
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip().lower()
    if s in ('nan', 'none', ''):
        return None
    return s


def _nnls(A, b, iters=2000):
    """Coordinate descent non-negative least squares."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, k = A.shape
    x = np.zeros(k, dtype=np.float64)
    if k == 0:
        return x
    norms = np.einsum('ij,ij->j', A, A)
    for _ in range(iters):
        improved = False
        for j in range(k):
            if norms[j] < 1e-12:
                continue
            r = b - A @ x
            xj_new = max(0.0, x[j] + (A[:, j] @ r) / norms[j])
            if abs(xj_new - x[j]) > 1e-12:
                x[j] = xj_new
                improved = True
        if not improved:
            break
    return x


def _fit_support_context(y, support, contexts, val_support, val_contexts):
    """Context-aware demeaned support baseline with expanded alpha grid."""
    out = np.zeros_like(val_support, dtype=np.float64)
    grid = np.array([0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0], dtype=np.float64)
    base = {}

    for ctx in set(contexts):
        mask = np.array([c == ctx for c in contexts], dtype=bool)
        if np.sum(mask) == 0:
            continue
        yy = y[mask]
        ss = support[mask]
        mu_y = yy.mean(axis=0)
        mu_s = ss.mean(axis=0)
        alpha = min(grid, key=lambda v: float(np.square(mu_y + v * (ss - mu_s) - yy).mean()))
        base[ctx] = (mu_y, alpha, mu_s)

    mu_y_all = y.mean(axis=0)
    mu_s_all = support.mean(axis=0)
    alpha_all = min(grid, key=lambda v: float(np.square(mu_y_all + v * (support - mu_s_all) - y).mean()))

    for i, ctx in enumerate(val_contexts):
        if ctx in base:
            mu_y, alpha, mu_s = base[ctx]
        else:
            mu_y, alpha, mu_s = mu_y_all, alpha_all, mu_s_all
        out[i] = mu_y + alpha * (val_support[i] - mu_s)

    return out


def _fit_prior_pred(y, prior, val_prior):
    """Global scaled prior with intercept."""
    A = np.column_stack([prior.ravel(), np.ones(prior.size, dtype=np.float64)])
    coef, _, _, _ = np.linalg.lstsq(A, y.ravel(), rcond=None)
    return coef[0] * val_prior + coef[1]


def _fit_xridge_pred(y, x, val_x, rank=8, penalty=10.0):
    """Reduced-rank ridge regression from row covariates to gene outputs."""
    n_tr = y.shape[0]
    G = y.shape[1]
    y_mean = y.mean(axis=0)
    yc = y - y_mean

    _, _, Vt = np.linalg.svd(yc, full_matrices=False)
    r = min(rank, n_tr, len(Vt))
    if r == 0:
        return np.tile(y_mean, (len(val_x), 1))
    basis = Vt[:r]
    proj_tr = yc @ basis.T

    design_tr = np.c_[x, np.ones(n_tr, dtype=np.float64)]
    reg = np.eye(design_tr.shape[1], dtype=np.float64) * penalty
    reg[-1, -1] = 0.0
    try:
        coef = np.linalg.solve(design_tr.T @ design_tr + reg, design_tr.T @ proj_tr)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(design_tr.T @ design_tr + reg, design_tr.T @ proj_tr, rcond=None)[0]

    design_val = np.c_[val_x, np.ones(len(val_x), dtype=np.float64)]
    return (design_val @ coef) @ basis + y_mean


def _fit_target_resid_ridge(y, support_pred, tidx, T, val_tidx, lam=None):
    """Ridge one-hot target effects on residual y - support_pred."""
    R = y - support_pred
    n_tr = len(tidx)
    H_tr = np.zeros((n_tr, T), dtype=np.float64)
    valid = tidx >= 0
    if valid.any():
        H_tr[np.arange(n_tr)[valid], tidx[valid]] = 1.0

    if lam is None:
        lam = max(0.3, 0.4 * T / max(n_tr, 1))
    A = H_tr.T @ H_tr + lam * np.eye(T, dtype=np.float64)
    B = H_tr.T @ R
    try:
        V = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        V = np.linalg.lstsq(A, B, rcond=None)[0]

    n_val = len(val_tidx)
    H_val = np.zeros((n_val, T), dtype=np.float64)
    valid_v = val_tidx >= 0
    if valid_v.any():
        H_val[np.arange(n_val)[valid_v], val_tidx[valid_v]] = 1.0
    return H_val @ V


def _fit_direct_target_ridge(y, tidx, T, val_tidx, lam=0.5):
    """Direct one-hot target main effects on y, without explicit intercept."""
    n_tr = len(tidx)
    H_tr = np.zeros((n_tr, T), dtype=np.float64)
    valid = tidx >= 0
    if valid.any():
        H_tr[np.arange(n_tr)[valid], tidx[valid]] = 1.0

    A = H_tr.T @ H_tr + lam * np.eye(T, dtype=np.float64)
    B = H_tr.T @ y
    try:
        V = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        V = np.linalg.lstsq(A, B, rcond=None)[0]

    n_val = len(val_tidx)
    H_val = np.zeros((n_val, T), dtype=np.float64)
    valid_v = val_tidx >= 0
    if valid_v.any():
        H_val[np.arange(n_val)[valid_v], val_tidx[valid_v]] = 1.0
    return H_val @ V


def _fit_target_self_pred(y, support_pred, tidx, target_gene_idx, val_tidx, shrink=0.7):
    """
    Diagonal self correction: for each target, estimate mean residual at the
    output gene index corresponding to that target and add it as a sparse boost.
    """
    R = y - support_pred
    n_tr = len(tidx)
    G = y.shape[1]
    sums = np.zeros((len(target_gene_idx), G), dtype=np.float64)
    counts = np.zeros(len(target_gene_idx), dtype=np.float64)

    for i in range(n_tr):
        t = tidx[i]
        if t >= 0:
            sums[t] += R[i]
            counts[t] += 1.0

    D = sums / np.maximum(counts[:, None], 1e-9)
    shrink_factor = counts / (counts + shrink)
    D_shrunk = D * shrink_factor[:, None]

    out = np.zeros((len(val_tidx), G), dtype=np.float64)
    for i, t in enumerate(val_tidx):
        if t < 0:
            continue
        g = target_gene_idx[t]
        if g >= 0:
            out[i, g] = D_shrunk[t, g]
    return out


def _fit_target_support_ridge(y, support, tidx, T, val_support, val_tidx, lam=3.0, global_shrink=0.5):
    """
    Per-target ridge regression from support to full gene output.
    For each target t, fit a ridge model using only rows with that target,
    then predict for validation rows with the same target. If a target has
    fewer than 2 training rows, fall back to a global model blended with
    the target-specific intercept.
    """
    n_tr = len(support)
    G = y.shape[1]
    X_tr = np.c_[support, np.ones(n_tr, dtype=np.float64)]
    reg_global = np.eye(X_tr.shape[1], dtype=np.float64) * lam
    reg_global[-1, -1] = 0.0
    try:
        coef_global = np.linalg.solve(X_tr.T @ X_tr + reg_global, X_tr.T @ y)
    except np.linalg.LinAlgError:
        coef_global = np.linalg.lstsq(X_tr.T @ X_tr + reg_global, X_tr.T @ y, rcond=None)[0]

    pred = np.zeros((len(val_support), G), dtype=np.float64)
    X_val = np.c_[val_support, np.ones(len(val_support), dtype=np.float64)]
    pred_global = X_val @ coef_global

    for t in range(T):
        mask_tr = tidx == t
        if mask_tr.sum() == 0:
            continue
        supp_t = support[mask_tr]
        y_t = y[mask_tr]
        n_t = len(supp_t)

        if n_t >= 2:
            X_t = np.c_[supp_t, np.ones(n_t, dtype=np.float64)]
            reg_t = np.eye(X_t.shape[1], dtype=np.float64) * lam
            reg_t[-1, -1] = 0.0
            try:
                coef_t = np.linalg.solve(X_t.T @ X_t + reg_t, X_t.T @ y_t)
            except np.linalg.LinAlgError:
                coef_t = np.linalg.lstsq(X_t.T @ X_t + reg_t, X_t.T @ y_t, rcond=None)[0]
            blend = min(1.0, n_t / 6.0)
            coef_blended = blend * coef_t + (1.0 - blend) * coef_global
        else:
            y_mean_t = y_t.mean(axis=0)
            supp_mean_t = supp_t.mean(axis=0)
            coef_blended = coef_global.copy()
            coef_blended[-1] = y_mean_t - (supp_mean_t @ coef_global[:-1])

        mask_val = val_tidx == t
        if mask_val.any():
            pred[mask_val] = X_val[mask_val] @ coef_blended

    mask_unseen = val_tidx < 0
    for t in range(T):
        if (tidx == t).sum() == 0:
            mask_unseen |= (val_tidx == t)
    pred[mask_unseen] = pred_global[mask_unseen]
    return pred


def _fit_target_resid_svd(R, tidx, T, val_tidx, rank=4, shrink=0.8):
    """Low-rank SVD denoiser for target effects."""
    G = R.shape[1]
    sums = np.zeros((T, G), dtype=np.float64)
    counts = np.zeros(T, dtype=np.float64)
    for i in range(len(tidx)):
        t = tidx[i]
        if t >= 0:
            sums[t] += R[i]
            counts[t] += 1.0
    counts = np.maximum(counts, 1e-9)
    V_raw = sums / counts[:, None]
    shrink_factor = counts / (counts + shrink)
    V_raw = V_raw * shrink_factor[:, None]

    V_mean = V_raw.mean(axis=0, keepdims=True)
    V_c = V_raw - V_mean
    U, s, Vh = np.linalg.svd(V_c, full_matrices=False)
    rank = max(0, min(rank, len(s)))
    if rank == 0:
        V_smooth = np.tile(V_mean[0], (T, 1))
    else:
        V_smooth = (U[:, :rank] * s[:rank]) @ Vh[:rank, :] + V_mean

    out = np.zeros((len(val_tidx), G), dtype=np.float64)
    for i, t in enumerate(val_tidx):
        if t >= 0:
            out[i] = V_smooth[t]
    return out


def _fit_profile_resid(R, P_train, P_val, lam=6.0):
    """Fit a single scaling coefficient per gene from profile to residual."""
    G = R.shape[1]
    out = np.zeros_like(P_val, dtype=np.float64)
    for g in range(G):
        p = P_train[:, g]
        denom = float(np.dot(p, p) + lam)
        if denom > 1e-12:
            a = float(np.dot(p, R[:, g]) / denom)
            out[:, g] = a * P_val[:, g]
    return out


def _extract_aux_profiles(data, target_to_idx, target_token_fn, contexts_ref, contexts_query, G):
    """Build global and context-specific auxiliary target profiles if available."""
    aux_delta = data.get('aux_delta')
    if aux_delta is None:
        return False, None, None

    aux_delta = np.asarray(aux_delta, dtype=np.float64)
    if aux_delta.ndim != 2 or aux_delta.shape[1] != G:
        return False, None, None

    n_aux = aux_delta.shape[0]
    if n_aux == 0:
        return False, None, None

    T = len(target_to_idx)
    if T == 0:
        return False, None, None

    aux_target_raw = data.get('aux_target')
    if aux_target_raw is None or len(aux_target_raw) != n_aux:
        return False, None, None
    aux_target_raw = np.asarray(aux_target_raw)

    aux_context_raw = data.get('aux_context')
    if aux_context_raw is not None and len(aux_context_raw) == n_aux:
        aux_contexts = [_safe_str(c) or 'global' for c in np.asarray(aux_context_raw)]
    else:
        aux_contexts = ['global'] * n_aux

    keep = np.ones(n_aux, dtype=bool)
    if data.get('aux_stimulus') is not None and len(data['aux_stimulus']) == n_aux:
        stims = [_safe_str(s) for s in np.asarray(data['aux_stimulus'])]
        if any(s and 'ifng' in s for s in stims):
            keep &= np.array([bool(s and 'ifng' in s) for s in stims])
    if data.get('aux_modality') is not None and len(data['aux_modality']) == n_aux:
        mods = [_safe_str(m) for m in np.asarray(data['aux_modality'])]
        if any(m and 'crispr' in m for m in mods):
            keep &= np.array([bool(m and 'crispr' in m) for m in mods])
    if keep.sum() < max(5, int(0.05 * n_aux)) and n_aux > 0:
        keep[:] = True

    aux_tidx = np.full(n_aux, -1, dtype=np.int64)
    for i, t in enumerate(aux_target_raw):
        tok = target_token_fn(t)
        if tok is not None:
            aux_tidx[i] = target_to_idx.get(tok, -1)

    global_prof = np.zeros((T, G), dtype=np.float64)
    global_counts = np.zeros(T, dtype=np.float64)
    for i in range(n_aux):
        t = aux_tidx[i]
        if keep[i] and t >= 0:
            global_prof[t] += aux_delta[i]
            global_counts[t] += 1.0
    global_prof = global_prof / np.maximum(global_counts[:, None], 1.0)

    all_ctx = set(contexts_ref) | set(contexts_query) | set(aux_contexts)
    ctx_profs = {}
    for ctx in all_ctx:
        sums = np.zeros((T, G), dtype=np.float64)
        counts = np.zeros(T, dtype=np.float64)
        for i in range(n_aux):
            t = aux_tidx[i]
            if keep[i] and t >= 0 and aux_contexts[i] == ctx:
                sums[t] += aux_delta[i]
                counts[t] += 1.0
        prof = np.zeros((T, G), dtype=np.float64)
        for t in range(T):
            if counts[t] > 0:
                prof[t] = sums[t] / counts[t]
            else:
                prof[t] = global_prof[t]
        ctx_profs[ctx] = prof

    return True, global_prof, ctx_profs


def _prof_matrix(tidx, prof):
    P = np.zeros((len(tidx), prof.shape[1]), dtype=np.float64)
    for i, t in enumerate(tidx):
        if t >= 0:
            P[i] = prof[t]
    return P


def _prof_matrix_ctx(tidx, contexts, ctx_profs, fallback_prof):
    P = np.zeros((len(tidx), fallback_prof.shape[1]), dtype=np.float64)
    for i, t in enumerate(tidx):
        if t >= 0:
            prof = ctx_profs.get(contexts[i], fallback_prof)
            P[i] = prof[t]
    return P


def _select_context_target_correction(y_cal, pred_cal, tidx_cal, target_gene_idx,
                                      cal_contexts, context, global_fallback):
    """
    Per-context per-target correction selection using rank+amplitude-aware objective.
    """
    corrections = {}
    T = len(target_gene_idx)
    y_scale = float(np.mean(np.abs(y_cal)) + 1e-9)
    s_grid = np.array([0.25, 0.4, 0.6, 0.8, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0], dtype=np.float64)
    b_grid = np.array([-1.0, -0.7, -0.5, -0.3, -0.2, -0.1, -0.05, -0.02, 0.0,
                       0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 4.0], dtype=np.float64) * y_scale

    for t in range(T):
        mask = (tidx_cal == t) & (np.array(cal_contexts) == context)
        if mask.sum() < 2:
            corrections[t] = global_fallback.get(t, (1.0, 0.0))
            continue
        rows = np.where(mask)[0]
        genes = target_gene_idx[tidx_cal[rows]]
        valid_rows = genes >= 0
        if valid_rows.sum() < 1:
            corrections[t] = global_fallback.get(t, (1.0, 0.0))
            continue
        rows = rows[valid_rows]
        genes = genes[valid_rows]
        y_t = y_cal[rows, genes]
        p_t = pred_cal[rows, genes]

        full_pred = pred_cal[rows].copy()
        true_abs = float(np.mean(np.abs(y_t))) + 1e-9
        base_mse = float(np.mean((full_pred - y_cal[rows]) ** 2)) + 1e-12

        best_score = -np.inf
        best_s, best_b = global_fallback.get(t, (1.0, 0.0))
        for s in s_grid:
            for b in b_grid:
                temp = full_pred.copy()
                temp[np.arange(len(rows)), genes] = s * p_t + b
                win = float(np.mean(np.argmax(temp, axis=1) == genes))
                mse = float(np.mean((temp - y_cal[rows]) ** 2))
                rel_mse = mse / base_mse
                amp_ratio = float(np.mean(np.abs(s * p_t + b)) / true_abs)
                # Higher weight on amplitude, lower MSE penalty than prior baseline
                score = win + 0.6 * min(amp_ratio, 1.2) - 0.08 * max(0.0, rel_mse - 1.0)
                if score > best_score + 1e-12:
                    best_score = score
                    best_s, best_b = float(s), float(b)
        corrections[t] = (best_s, best_b)

    return corrections


def _global_fallback_corrections(cal_y, cal_pred, cal_tidx, target_gene_idx):
    """Shrunken least-squares fallback per target."""
    T = len(target_gene_idx)
    corrections = {}
    for t in range(T):
        mask = cal_tidx == t
        if mask.sum() < 2:
            corrections[t] = (1.0, 0.0)
            continue
        rows = np.where(mask)[0]
        genes = target_gene_idx[cal_tidx[rows]]
        valid_rows = genes >= 0
        if valid_rows.sum() < 1:
            corrections[t] = (1.0, 0.0)
            continue
        rows = rows[valid_rows]
        genes = genes[valid_rows]
        y_t = cal_y[rows, genes]
        p_t = cal_pred[rows, genes]
        A = np.c_[p_t, np.ones_like(p_t)]
        coef, _, _, _ = np.linalg.lstsq(A, y_t, rcond=None)
        # Shrink multiplicative coefficient toward 1
        n = len(rows)
        shrink = n / (n + 2.0)
        s = shrink * coef[0] + (1.0 - shrink) * 1.0
        s = max(0.2, min(s, 5.0))
        b = shrink * coef[1]
        b = max(-10.0, min(b, 10.0))
        corrections[t] = (float(s), float(b))
    return corrections


def fit_predict(data, budget):
    # ---------- Stage 1: Data extraction and alignment ----------
    train_y = np.asarray(data['train_y'], dtype=np.float64)
    cal_y = np.asarray(data['calibration_y'], dtype=np.float64)
    y_ref = np.concatenate([train_y, cal_y], axis=0)
    n_train = train_y.shape[0]
    n_cal = cal_y.shape[0]
    n_ref = n_train + n_cal
    G = y_ref.shape[1]

    train_support = np.asarray(data['train_support'], dtype=np.float64)
    cal_support = np.asarray(data['calibration_support'], dtype=np.float64)
    support_ref = np.concatenate([train_support, cal_support], axis=0)
    query_support = np.asarray(data['query_support'], dtype=np.float64)
    n_query = query_support.shape[0]

    def ctx_to_str(c):
        s = _safe_str(c)
        return 'global' if s is None else s

    source_context = data.get('source_context')
    if source_context is not None and len(source_context) == n_ref:
        ref_contexts = [ctx_to_str(c) for c in np.asarray(source_context)]
    else:
        ref_contexts = ['global'] * n_ref

    query_context = data.get('query_context')
    if query_context is not None and len(query_context) == n_query:
        query_contexts = [ctx_to_str(c) for c in np.asarray(query_context)]
    else:
        query_contexts = ['global'] * n_query

    genes_list = [str(g) for g in data['genes']]
    gene_lower_to_idx = {g.lower(): i for i, g in enumerate(genes_list)}

    alias_map = {}
    alias_labels = data.get('target_alias_labels')
    alias_symbols = data.get('target_alias_symbols')
    if alias_labels is not None and alias_symbols is not None:
        for lab, sym in zip(alias_labels, alias_symbols):
            l = _safe_str(lab)
            s = _safe_str(sym)
            if l is not None and s is not None:
                alias_map[l] = s
                alias_map[s] = s

    def target_token(t):
        c = _safe_str(t)
        if c is None:
            return None
        if c in gene_lower_to_idx:
            return f"gene:{gene_lower_to_idx[c]}"
        if c in alias_map:
            sym = alias_map[c]
            if sym in gene_lower_to_idx:
                return f"gene:{gene_lower_to_idx[sym]}"
        return c

    train_targets = np.asarray(data['train_targets'])
    cal_targets = np.asarray(data['calibration_targets'])
    ref_targets_raw = np.concatenate([train_targets, cal_targets])

    query_targets_raw = data.get('query_targets')
    if query_targets_raw is not None:
        query_targets_raw = np.asarray(query_targets_raw)
    else:
        query_targets_raw = np.array([None] * n_query, dtype=object)

    all_raw = list(ref_targets_raw) + list(query_targets_raw)
    unique_tokens = []
    for t in all_raw:
        tok = target_token(t)
        if tok is not None:
            unique_tokens.append(tok)
    unique_tokens = sorted(set(unique_tokens))
    target_to_idx = {u: i for i, u in enumerate(unique_tokens)}
    T = len(unique_tokens)

    def map_targets(raw):
        out = np.full(len(raw), -1, dtype=np.int64)
        for i, t in enumerate(raw):
            tok = target_token(t)
            if tok is not None:
                out[i] = target_to_idx.get(tok, -1)
        return out

    ref_tidx = map_targets(ref_targets_raw)
    query_tidx = map_targets(query_targets_raw)

    target_gene_idx = np.full(T, -1, dtype=np.int64)
    for tok, idx in target_to_idx.items():
        if tok.startswith('gene:'):
            try:
                target_gene_idx[idx] = int(tok.split(':')[1])
            except Exception:
                pass

    use_prior = False
    prior_ref = None
    query_prior = None
    if (data.get('train_prior') is not None and
        data.get('calibration_prior') is not None and
        data.get('query_prior') is not None):
        train_prior = np.asarray(data['train_prior'], dtype=np.float64)
        cal_prior = np.asarray(data['calibration_prior'], dtype=np.float64)
        query_prior = np.asarray(data['query_prior'], dtype=np.float64)
        if (train_prior.ndim == 2 and cal_prior.ndim == 2 and query_prior.ndim == 2 and
            train_prior.shape[0] == n_train and cal_prior.shape[0] == n_cal and
            query_prior.shape[0] == n_query and train_prior.shape[1] == G):
            prior_ref = np.concatenate([train_prior, cal_prior], axis=0)
            use_prior = True

    use_x = False
    x_ref = None
    query_x = None
    if (data.get('train_x') is not None and
        data.get('calibration_x') is not None and
        data.get('query_x') is not None):
        train_x = np.asarray(data['train_x'], dtype=np.float64)
        cal_x = np.asarray(data['calibration_x'], dtype=np.float64)
        query_x = np.asarray(data['query_x'], dtype=np.float64)
        if (train_x.ndim == 2 and cal_x.ndim == 2 and query_x.ndim == 2 and
            train_x.shape[0] == n_train and cal_x.shape[0] == n_cal and
            query_x.shape[0] == n_query):
            x_ref = np.concatenate([train_x, cal_x], axis=0)
            use_x = True

    aux_available, aux_global_prof, aux_ctx_profs = _extract_aux_profiles(
        data, target_to_idx, target_token, ref_contexts, query_contexts, G
    )

    # ---------- Stage 2: Base learners ----------
    base_names = ['support']
    if T > 0:
        base_names += ['target_resid', 'target_self', 'target_svd', 'direct_target', 'target_support_ridge']
        if aux_available:
            base_names += ['aux_global', 'aux_context']
    if use_prior:
        base_names.append('prior')
    if use_x:
        base_names.append('xridge')
    base_names.append('const')
    base_idx = {name: i for i, name in enumerate(base_names[:-1])}
    n_bases = len(base_names) - 1

    oof_preds = [np.zeros((n_ref, G), dtype=np.float64) for _ in range(n_bases)]
    const_oof = np.ones((n_ref, G), dtype=np.float64)

    # ---------- Stage 3: Out-of-fold base predictions ----------
    for fold in [0, 1]:
        tr_idx = np.array([i for i in range(n_ref) if i % 2 != fold], dtype=np.int64)
        val_idx = np.array([i for i in range(n_ref) if i % 2 == fold], dtype=np.int64)
        if len(tr_idx) == 0 or len(val_idx) == 0:
            continue

        y_tr = y_ref[tr_idx]
        support_tr = support_ref[tr_idx]
        ctx_tr = [ref_contexts[i] for i in tr_idx]
        support_val = support_ref[val_idx]
        ctx_val = [ref_contexts[i] for i in val_idx]

        support_val_pred = _fit_support_context(y_tr, support_tr, ctx_tr, support_val, ctx_val)
        oof_preds[base_idx['support']][val_idx] = support_val_pred

        if T > 0:
            support_tr_pred = _fit_support_context(y_tr, support_tr, ctx_tr, support_tr, ctx_tr)
            tidx_tr = ref_tidx[tr_idx]
            tidx_val = ref_tidx[val_idx]
            R_tr = y_tr - support_tr_pred

            oof_preds[base_idx['target_resid']][val_idx] = (
                support_val_pred + _fit_target_resid_ridge(y_tr, support_tr_pred, tidx_tr, T, tidx_val)
            )
            oof_preds[base_idx['target_self']][val_idx] = (
                support_val_pred + _fit_target_self_pred(y_tr, support_tr_pred, tidx_tr,
                                                         target_gene_idx, tidx_val, shrink=0.7)
            )
            oof_preds[base_idx['target_svd']][val_idx] = (
                support_val_pred + _fit_target_resid_svd(R_tr, tidx_tr, T, tidx_val, rank=4, shrink=0.8)
            )
            oof_preds[base_idx['direct_target']][val_idx] = (
                _fit_direct_target_ridge(y_tr, tidx_tr, T, tidx_val, lam=0.5)
            )
            oof_preds[base_idx['target_support_ridge']][val_idx] = (
                _fit_target_support_ridge(y_tr, support_tr, tidx_tr, T,
                                          support_val, tidx_val, lam=3.0, global_shrink=0.5)
            )

            if aux_available:
                P_tr_global = _prof_matrix(tidx_tr, aux_global_prof)
                P_val_global = _prof_matrix(tidx_val, aux_global_prof)
                oof_preds[base_idx['aux_global']][val_idx] = (
                    support_val_pred + _fit_profile_resid(R_tr, P_tr_global, P_val_global, lam=6.0)
                )
                P_tr_ctx = _prof_matrix_ctx(tidx_tr, ctx_tr, aux_ctx_profs, aux_global_prof)
                P_val_ctx = _prof_matrix_ctx(tidx_val, ctx_val, aux_ctx_profs, aux_global_prof)
                oof_preds[base_idx['aux_context']][val_idx] = (
                    support_val_pred + _fit_profile_resid(R_tr, P_tr_ctx, P_val_ctx, lam=6.0)
                )

        if use_prior:
            oof_preds[base_idx['prior']][val_idx] = _fit_prior_pred(
                y_tr, prior_ref[tr_idx], prior_ref[val_idx]
            )
        if use_x:
            oof_preds[base_idx['xridge']][val_idx] = _fit_xridge_pred(
                y_tr, x_ref[tr_idx], x_ref[val_idx]
            )

    # ---------- Stage 4: Stacking weights ----------
    A_oof_full = np.column_stack([pred.ravel() for pred in oof_preds])
    A_oof_full = np.column_stack([A_oof_full, const_oof.ravel()])
    b_oof = y_ref.ravel()
    w_global = _nnls(A_oof_full, b_oof, iters=2000)

    w_context = {}
    unique_ref_contexts = set(ref_contexts)
    for ctx in unique_ref_contexts:
        mask = np.array([c == ctx for c in ref_contexts], dtype=bool)
        if mask.sum() < 4:
            continue
        A_ctx = np.column_stack([p[mask].ravel() for p in oof_preds])
        A_ctx = np.column_stack([A_ctx, const_oof[mask].ravel()])
        b_ctx = y_ref[mask].ravel()
        w_context[ctx] = _nnls(A_ctx, b_ctx, iters=2000)

    # ---------- Stage 5: Final full-data base predictions ----------
    support_query = _fit_support_context(y_ref, support_ref, ref_contexts, query_support, query_contexts)
    final_preds = [support_query]

    if T > 0:
        support_full = _fit_support_context(y_ref, support_ref, ref_contexts, support_ref, ref_contexts)
        R_full = y_ref - support_full

        final_preds.append(
            support_query + _fit_target_resid_ridge(y_ref, support_full, ref_tidx, T, query_tidx)
        )
        final_preds.append(
            support_query + _fit_target_self_pred(y_ref, support_full, ref_tidx,
                                                  target_gene_idx, query_tidx, shrink=0.7)
        )
        final_preds.append(
            support_query + _fit_target_resid_svd(R_full, ref_tidx, T, query_tidx, rank=4, shrink=0.8)
        )
        final_preds.append(_fit_direct_target_ridge(y_ref, ref_tidx, T, query_tidx, lam=0.5))
        final_preds.append(
            _fit_target_support_ridge(y_ref, support_ref, ref_tidx, T,
                                      query_support, query_tidx, lam=3.0, global_shrink=0.5)
        )

        if aux_available:
            P_ref_global = _prof_matrix(ref_tidx, aux_global_prof)
            P_query_global = _prof_matrix(query_tidx, aux_global_prof)
            final_preds.append(
                support_query + _fit_profile_resid(R_full, P_ref_global, P_query_global, lam=6.0)
            )
            P_ref_ctx = _prof_matrix_ctx(ref_tidx, ref_contexts, aux_ctx_profs, aux_global_prof)
            P_query_ctx = _prof_matrix_ctx(query_tidx, query_contexts, aux_ctx_profs, aux_global_prof)
            final_preds.append(
                support_query + _fit_profile_resid(R_full, P_ref_ctx, P_query_ctx, lam=6.0)
            )

    if use_prior:
        final_preds.append(_fit_prior_pred(y_ref, prior_ref, query_prior))
    if use_x:
        final_preds.append(_fit_xridge_pred(y_ref, x_ref, query_x))

    final_preds.append(np.ones((n_query, G), dtype=np.float64))

    pred = np.zeros((n_query, G), dtype=np.float64)
    for i in range(n_query):
        ctx = query_contexts[i]
        w = w_context.get(ctx, w_global)
        for wi, pi in zip(w, final_preds):
            pred[i] += wi * pi[i]

    # ---------- Stage 6: Calibration corrections ----------
    oof_stacked = np.zeros((n_ref, G), dtype=np.float64)
    for i in range(n_ref):
        ctx = ref_contexts[i]
        w = w_context.get(ctx, w_global)
        for wi, oi in zip(w[:-1], oof_preds):
            oof_stacked[i] += wi * oi[i]
        oof_stacked[i] += w[-1] * const_oof[i]

    cal_pred = oof_stacked[n_train:n_ref]
    cal_tidx = ref_tidx[n_train:n_ref]
    cal_contexts = [ref_contexts[i] for i in range(n_train, n_ref)]

    global_corrections = {}
    if T > 0:
        global_corrections = _global_fallback_corrections(
            cal_y, cal_pred, cal_tidx, target_gene_idx
        )

    context_corrections = {}
    for ctx in set(cal_contexts):
        corr = _select_context_target_correction(
            cal_y, cal_pred, cal_tidx, target_gene_idx,
            cal_contexts, ctx, global_corrections
        )
        context_corrections[ctx] = corr

    valid_q = (query_tidx >= 0)
    if T > 0:
        valid_q = valid_q & (target_gene_idx[query_tidx] >= 0)
    if np.any(valid_q):
        q_rows = np.where(valid_q)[0]
        q_genes = target_gene_idx[query_tidx[valid_q]]
        q_ctxs = [query_contexts[i] for i in q_rows]
        for row_idx, gene_idx, t, ctx in zip(q_rows, q_genes, query_tidx[valid_q], q_ctxs):
            if ctx in context_corrections and t in context_corrections[ctx]:
                s_t, b_t = context_corrections[ctx][t]
            elif t in global_corrections:
                s_t, b_t = global_corrections[t]
            else:
                s_t, b_t = 1.0, 0.0
            pred[row_idx, gene_idx] = s_t * pred[row_idx, gene_idx] + b_t

    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return pred.astype(np.float32)

# EVOLVE-BLOCK-END
