# EVOLVE-BLOCK-START
import numpy as np


def fit_predict(data, budget):
    """
    Context-specific support scaling.
    For each cell-line context, estimate a scalar response scaling from
    legal train+calibration support/response pairs and use it to scale
    query support.
    """
    # Extract legal arrays
    train_y = np.asarray(data['train_y'], dtype=np.float64)
    cal_y = np.asarray(data['calibration_y'], dtype=np.float64)
    train_s = np.asarray(data['train_support'], dtype=np.float64)
    cal_s = np.asarray(data['calibration_support'], dtype=np.float64)
    query_s = np.asarray(data['query_support'], dtype=np.float64)

    # Source context strings
    train_ctx = np.asarray(data['train_source_context'], dtype=str)
    cal_ctx = np.asarray(data['calibration_source_context'], dtype=str)
    query_ctx = np.asarray(data['query_source_context'], dtype=str)

    # Initialize prediction
    pred = np.zeros_like(query_s, dtype=np.float64)

    # Global fallback coefficient in case a query context has no train/cal rows
    S_all = np.vstack([train_s, cal_s])
    Y_all = np.vstack([train_y, cal_y])
    sum_s2_global = np.sum(S_all ** 2)
    sum_sy_global = np.sum(S_all * Y_all)
    alpha_global = sum_sy_global / max(sum_s2_global, 1e-12)

    # Unique query contexts
    unique_query_ctx = np.unique(query_ctx)

    for ctx in unique_query_ctx:
        q_mask = (query_ctx == ctx)
        tr_mask = (train_ctx == ctx)
        ca_mask = (cal_ctx == ctx)

        # Gather legal source rows for this context
        if np.any(tr_mask) or np.any(ca_mask):
            S_ctx_list = []
            Y_ctx_list = []
            if np.any(tr_mask):
                S_ctx_list.append(train_s[tr_mask])
                Y_ctx_list.append(train_y[tr_mask])
            if np.any(ca_mask):
                S_ctx_list.append(cal_s[ca_mask])
                Y_ctx_list.append(cal_y[ca_mask])
            S_ctx = np.vstack(S_ctx_list)
            Y_ctx = np.vstack(Y_ctx_list)

            sum_s2_ctx = np.sum(S_ctx ** 2)
            sum_sy_ctx = np.sum(S_ctx * Y_ctx)
            alpha_ctx = sum_sy_ctx / max(sum_s2_ctx, 1e-12)
        else:
            # Rare case: no source rows for this context; use global
            alpha_ctx = alpha_global

        # Scale query support for rows of this context
        pred[q_mask] = query_s[q_mask] * alpha_ctx

    # Postprocess: finite cast to float32
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return pred.astype(np.float32)
# EVOLVE-BLOCK-END
