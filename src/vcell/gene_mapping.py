"""Explicit count aggregation for symbol-vocabulary encoder inputs only."""
import numpy as np


def collapse_symbol_counts(counts, symbols):
    """Sum exact duplicate symbols before normalization, preserving first order.

    This deliberately collapses distinct source features sharing a symbol for
    an encoder with one token per symbol. It does not assert that their Ensembl
    IDs are interchangeable. Never apply it to the prepared prediction targets.
    """
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != len(symbols):
        raise ValueError("Count columns must match the supplied gene symbols")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Expected finite nonnegative counts before symbol aggregation")
    if any(not isinstance(s, str) or not s.strip() for s in symbols):
        raise ValueError("Missing or empty gene symbol; resolve the source annotation explicitly")
    unique, groups, index = [], [], {}
    for column, symbol in enumerate(symbols):
        if symbol not in index:
            index[symbol] = len(unique)
            unique.append(symbol)
            groups.append([])
        groups[index[symbol]].append(column)
    if len(unique) == len(symbols):
        collapsed = counts
    else:
        # Accumulate in float64 to avoid integer overflow and unnecessary loss
        # of count precision. The model casts only after normalization/binning.
        collapsed = np.zeros((len(counts), len(unique)), dtype=np.float64)
        for i, columns in enumerate(groups):
            collapsed[:, i] = counts[:, columns].sum(axis=1, dtype=np.float64)
    if not np.allclose(collapsed.sum(1, dtype=np.float64), counts.sum(1, dtype=np.float64), rtol=1e-12, atol=1e-8):
        raise ValueError("Symbol aggregation changed a cell's full library total")
    audit = {
        "policy": "sum_raw_counts_by_exact_symbol_before_normalization",
        "input_columns": len(symbols), "unique_symbols": len(unique),
        "merged_columns": len(symbols) - len(unique),
        "duplicate_groups": [{"symbol": symbol, "source_columns": columns}
                             for symbol, columns in zip(unique, groups) if len(columns) > 1],
        "full_library_totals_preserved": True,
    }
    return collapsed, unique, audit
