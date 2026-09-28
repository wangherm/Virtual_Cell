import numpy as np
import pytest

from vcell.gene_mapping import collapse_symbol_counts


def test_counts_preserved_with_exact_symbols_and_stable_order():
    raw = np.array([[1, 2, 3, 4, 5], [7, 0, 8, 1, 9]], dtype=np.int32)
    before = raw.copy()
    counts, symbols, audit = collapse_symbol_counts(raw, ["A", "B", "A", "b", "B"])
    assert symbols == ["A", "B", "b"]  # Do not invent alias/case mappings.
    np.testing.assert_array_equal(counts, [[4, 7, 4], [15, 9, 1]])
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(counts.sum(1), raw.sum(1))
    assert audit["merged_columns"] == 2
    assert audit["duplicate_groups"] == [
        {"symbol": "A", "source_columns": [0, 2]}, {"symbol": "B", "source_columns": [1, 4]}]


def test_unique_symbols_are_unchanged_and_large_counts_do_not_overflow():
    raw = np.array([[2_000_000_000, 2_000_000_000, 10]], dtype=np.int32)
    same, symbols, audit = collapse_symbol_counts(raw, ["A", "B", "C"])
    np.testing.assert_array_equal(same, raw)
    assert audit["merged_columns"] == 0
    summed, _, _ = collapse_symbol_counts(raw, ["A", "A", "B"])
    np.testing.assert_array_equal(summed, [[4_000_000_000, 10]])


@pytest.mark.parametrize("symbols", [["A", ""], ["A", None], ["A", " "]])
def test_missing_symbols_are_not_silently_merged(symbols):
    with pytest.raises(ValueError, match="Missing or empty"):
        collapse_symbol_counts(np.ones((1, 2)), symbols)


@pytest.mark.parametrize("raw", [[[1., -2.]], [[1., np.nan]], [[1., np.inf]]])
def test_bad_counts_are_rejected(raw):
    with pytest.raises(ValueError, match="nonnegative"):
        collapse_symbol_counts(raw, ["A", "A"])
