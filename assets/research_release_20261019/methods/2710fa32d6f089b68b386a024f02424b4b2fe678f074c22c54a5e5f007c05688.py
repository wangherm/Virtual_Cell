# EVOLVE-BLOCK-START
import numpy as np


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


def _canonical_target(t, gene_lower_to_idx, alias_map):
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


def _build_target_index(raw_targets, gene_lower_to_idx, alias_map):
    tokens = []
    for t in raw_targets:
        tok = _canonical_target(t, gene_lower_to_idx, alias_map)
        if tok is not None:
            tokens.append(tok)
    unique = sorted(set(tokens))
    idx = {u: i for i, u in enumerate(unique)}
    T = len(unique)
    target_gene_idx = np.full(T, -1, dtype=np.int64)
    for tok, i in idx.items():
        if tok.startswith('gene:'):
            try:
                target_gene_idx[i] = int(tok.split(':')[1])
            except Exception:
                pass
    return idx, target_gene_idx


def _map_targets(raw_targets, target_to_idx, gene_lower_to_idx, alias_map):
    out = np.full(len(raw_targets), -1, dtype=np.int64)
    for i, t in enumerate(raw_targets):
        tok = _canonical_target(t, gene_lower_to_idx, alias_map)
        if tok is not None and tok in target_to_idx:
            out[i] = target_to_idx[tok]
    return out


class PreparationStage:
    def __init__(self, data):
        self.data = data
        self.G = len(data['genes'])
        self.genes_list = [str(g) for g in data['genes']]
        self.gene_lower_to_idx = {g.lower(): i for i, g in enumerate(self.genes_list)}
        self.gene_lower_to_idx.update(
            {str(s).lower(): i for i, s in enumerate(data['gene_symbols']) if str(s)}
        )
        self.alias_map = {}
        alias_labels = data.get('target_alias_labels')
        alias_symbols = data.get('target_alias_symbols')
        if alias_labels is not None and alias_symbols is not None:
            for lab, sym in zip(alias_labels, alias_symbols):
                l = _safe_str(lab)
                s = _safe_str(sym)
                if l is not None and s is not None:
                    self.alias_map[l] = s
                    self.alias_map[s] = s

    def run(self):
        data = self.data
        self.train_y = np.asarray(data['train_y'], dtype=np.float64)
        self.cal_y = np.asarray(data['calibration_y'], dtype=np.float64)
        self.y_ref = np.concatenate([self.train_y, self.cal_y], axis=0)
        self.n_train = self.train_y.shape[0]
        self.n_cal = self.cal_y.shape[0]
        self.n_ref = self.n_train + self.n_cal

        self.train_support = np.asarray(data['train_support'], dtype=np.float64)
        self.cal_support = np.asarray(data['calibration_support'], dtype=np.float64)
        self.support_ref = np.concatenate([self.train_support, self.cal_support], axis=0)
        self.query_support = np.asarray(data['query_support'], dtype=np.float64)
        self.n_query = self.query_support.shape[0]

        def ctx_str(c):
            s = _safe_str(c)
            return 'global' if s is None else s

        src_ctx = data.get('source_context')
        if src_ctx is not None and len(src_ctx) == self.n_ref:
            self.ref_contexts = [ctx_str(c) for c in np.asarray(src_ctx)]
        else:
            self.ref_contexts = ['global'] * self.n_ref

        q_ctx = data.get('query_context')
        if q_ctx is not None and len(q_ctx) == self.n_query:
            self.query_contexts = [ctx_str(c) for c in np.asarray(q_ctx)]
        else:
            self.query_contexts = ['global'] * self.n_query

        # targets
        train_targets = np.asarray(data['train_targets'])
        cal_targets = np.asarray(data['calibration_targets'])
        ref_targets = np.concatenate([train_targets, cal_targets])
        query_targets = data.get('query_targets')
        if query_targets is not None:
            query_targets = np.asarray(query_targets)
        else:
            query_targets = np.array([None] * self.n_query, dtype=object)

        self.target_to_idx, self.target_gene_idx = _build_target_index(
            list(ref_targets) + list(query_targets),
            self.gene_lower_to_idx,
            self.alias_map
        )
        self.T = len(self.target_to_idx)

        self.ref_tidx = _map_targets(ref_targets, self.target_to_idx,
                                     self.gene_lower_to_idx, self.alias_map)
        self.query_tidx = _map_targets(query_targets, self.target_to_idx,
                                       self.gene_lower_to_idx, self.alias_map)

        # aux target profiles
        self.aux_available = False
        self.aux_global_prof = None
        aux_delta = data.get('aux_delta')
        if aux_delta is not None:
            aux_delta = np.asarray(aux_delta, dtype=np.float64)
            if aux_delta.ndim == 2 and aux_delta.shape[1] == self.G and aux_delta.shape[0] > 0:
                aux_target = data.get('aux_target')
                if aux_target is not None and len(aux_target) == aux_delta.shape[0]:
                    aux_target = np.asarray(aux_target)
                    prof = np.zeros((self.T, self.G), dtype=np.float64)
                    counts = np.zeros(self.T, dtype=np.float64)
                    for i, t in enumerate(aux_target):
                        tok = _canonical_target(t, self.gene_lower_to_idx, self.alias_map)
                        if tok is not None and tok in self.target_to_idx:
                            ti = self.target_to_idx[tok]
                            prof[ti] += aux_delta[i]
                            counts[ti] += 1.0
                    prof = prof / np.maximum(counts[:, None], 1.0)
                    self.aux_global_prof = prof
                    self.aux_available = True

        # controls and priors (optional)
        self.use_control = False
        self.control_ref = None
        self.query_control = None
        if (data.get('train_control') is not None and
            data.get('calibration_control') is not None and
            data.get('query_control') is not None):
            train_control = np.asarray(data['train_control'], dtype=np.float64)
            cal_control = np.asarray(data['calibration_control'], dtype=np.float64)
            query_control = np.asarray(data['query_control'], dtype=np.float64)
            if (train_control.shape[0] == self.n_train and cal_control.shape[0] == self.n_cal and
                query_control.shape[0] == self.n_query and train_control.shape[1] == self.G):
                self.control_ref = np.concatenate([train_control, cal_control], axis=0)
                self.query_control = query_control
                self.use_control = True

        self.use_prior = False
        self.prior_ref = None
        self.query_prior = None
        if (data.get('train_prior') is not None and
            data.get('calibration_prior') is not None and
            data.get('query_prior') is not None):
            train_prior = np.asarray(data['train_prior'], dtype=np.float64)
            cal_prior = np.asarray(data['calibration_prior'], dtype=np.float64)
            query_prior = np.asarray(data['query_prior'], dtype=np.float64)
            if (train_prior.shape[0] == self.n_train and cal_prior.shape[0] == self.n_cal and
                query_prior.shape[0] == self.n_query and train_prior.shape[1] == self.G):
                self.prior_ref = np.concatenate([train_prior, cal_prior], axis=0)
                self.query_prior = query_prior
                self.use_prior = True


class DenoiseStage:
    """
    Context-aware affine denoiser from support to y.
    For each context, fits a shared slope and per-gene intercept using ridge regularization.
    This is equivalent to empirical Bayes shrinkage but with a single slope per context.
    """
    def __init__(self, prep, lam=1.0):
        self.prep = prep
        self.lam = lam
        self.mu_y_by_ctx = {}
        self.mu_s_by_ctx = {}
        self.beta_by_ctx = {}

    def _fit_context(self, ctx):
        mask = np.array([c == ctx for c in self.prep.ref_contexts], dtype=bool)
        if np.sum(mask) == 0:
            return None
        y = self.prep.y_ref[mask]
        s = self.prep.support_ref[mask]
        mu_y = y.mean(axis=0)
        mu_s = s.mean(axis=0)
        sy = s - mu_s
        yy = y - mu_y
        num = float(np.sum(sy * yy))
        den = float(np.sum(sy * sy)) + self.lam * self.prep.G
        beta = num / max(den, 1e-12)
        beta = float(np.clip(beta, 0.05, 2.0))
        return mu_y, mu_s, beta

    def run(self):
        unique_ctx = set(self.prep.ref_contexts)
        for ctx in unique_ctx:
            out = self._fit_context(ctx)
            if out is not None:
                self.mu_y_by_ctx[ctx], self.mu_s_by_ctx[ctx], self.beta_by_ctx[ctx] = out

        # global fallback
        global_ctx_mask = np.ones(self.prep.n_ref, dtype=bool)
        mu_y_g = self.prep.y_ref.mean(axis=0)
        mu_s_g = self.prep.support_ref.mean(axis=0)
        sy_g = self.prep.support_ref - mu_s_g
        yy_g = self.prep.y_ref - mu_y_g
        num_g = float(np.sum(sy_g * yy_g))
        den_g = float(np.sum(sy_g * sy_g)) + self.lam * self.prep.G
        self.global_beta = float(np.clip(num_g / max(den_g, 1e-12), 0.05, 2.0))
        self.global_mu_y = mu_y_g
        self.global_mu_s = mu_s_g

    def predict(self, support, contexts):
        n = support.shape[0]
        pred = np.zeros((n, self.prep.G), dtype=np.float64)
        for i in range(n):
            ctx = contexts[i]
            if ctx in self.beta_by_ctx:
                mu_y = self.mu_y_by_ctx[ctx]
                mu_s = self.mu_s_by_ctx[ctx]
                beta = self.beta_by_ctx[ctx]
            else:
                mu_y = self.global_mu_y
                mu_s = self.global_mu_s
                beta = self.global_beta
            pred[i] = mu_y + beta * (support[i] - mu_s)
        return pred


class TargetPriorStage:
    """
    Adds a scaled auxiliary target profile to improve target-specific signal,
    then applies a shrunken diagonal self-correction at the target gene.
    """
    def __init__(self, prep, denoise, prof_scale_lambda=8.0, self_shrink=0.5):
        self.prep = prep
        self.denoise = denoise
        self.prof_scale_lambda = prof_scale_lambda
        self.self_shrink = self_shrink

    def run(self):
        prep = self.prep
        denoised_ref = self.denoise.predict(prep.support_ref, prep.ref_contexts)
        denoised_query = self.denoise.predict(prep.query_support, prep.query_contexts)
        self.denoised_ref = denoised_ref
        self.base_query = denoised_query

        # auxiliary profile scaling
        if prep.aux_available and prep.T > 0:
            P_ref = np.zeros((prep.n_ref, prep.G), dtype=np.float64)
            P_query = np.zeros((prep.n_query, prep.G), dtype=np.float64)
            for i, t in enumerate(prep.ref_tidx):
                if t >= 0:
                    P_ref[i] = prep.aux_global_prof[t]
            for i, t in enumerate(prep.query_tidx):
                if t >= 0:
                    P_query[i] = prep.aux_global_prof[t]
            R = prep.y_ref - denoised_ref
            num = float(np.dot(P_ref.ravel(), R.ravel()))
            den = float(np.dot(P_ref.ravel(), P_ref.ravel())) + self.prof_scale_lambda * prep.G
            gamma = num / max(den, 1e-12)
            self.base_query = self.base_query + gamma * P_query
            self.denoised_ref = self.denoised_ref + gamma * P_ref
        else:
            P_query = np.zeros((prep.n_query, prep.G), dtype=np.float64)

        # target self correction from residuals
        self.diag_correction = np.zeros((prep.n_query, prep.G), dtype=np.float64)
        if prep.T > 0:
            R_diag = prep.y_ref - self.denoised_ref
            sums = np.zeros((prep.T, prep.G), dtype=np.float64)
            counts = np.zeros(prep.T, dtype=np.float64)
            for i, t in enumerate(prep.ref_tidx):
                if t >= 0:
                    sums[t] += R_diag[i]
                    counts[t] += 1.0
            counts = np.maximum(counts, 1e-9)
            mean_resid = sums / counts[:, None]
            shrink_factor = counts / (counts + self.self_shrink)
            mean_resid = mean_resid * shrink_factor[:, None]
            for i, t in enumerate(prep.query_tidx):
                if t < 0:
                    continue
                g = prep.target_gene_idx[t]
                if g >= 0:
                    self.diag_correction[i, g] = mean_resid[t, g]

        self.pred_query = self.base_query + self.diag_correction


class CalibrationStage:
    """
    Minimal post-processing to ensure finite values.
    More complex calibration is avoided because the reference set is tiny.
    """
    def __init__(self, prep, target_prior):
        self.prep = prep
        self.pred = target_prior.pred_query

    def run(self):
        self.pred = np.nan_to_num(self.pred, nan=0.0, posinf=0.0, neginf=0.0)
        return self.pred


def fit_predict(data, budget):
    """
    Structured pipeline for repeat prediction:
    Preparation -> Denoising -> Target prior -> Calibration.
    Returns float32 predictions for query rows.
    """
    prep = PreparationStage(data)
    prep.run()

    denoise = DenoiseStage(prep, lam=1.0)
    denoise.run()

    prior = TargetPriorStage(prep, denoise, prof_scale_lambda=8.0, self_shrink=0.5)
    prior.run()

    calib = CalibrationStage(prep, prior)
    pred = calib.run()

    return pred.astype(np.float32)
# EVOLVE-BLOCK-END
