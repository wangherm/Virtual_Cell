import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from vcell import challenge_data
from vcell.challenge_data import aggregate_training, prepare_student_c
from vcell.data import load_prepared
from vcell.synthetic import make_demo


def source_fixture(tmp_path, original):
    """Artificial source: fewer genes/targets than reference, never real VCC."""
    genes = original["genes"][:-2]
    targets = list(original["perturbations"][:2]) + ["NEW_TEST_TARGET"]
    names = ["non-targeting", *targets]
    records = [{"target_gene": p, "batch": b} for b in ["batch_A", "batch_B"] for p in names for _ in range(4)]
    counts = np.random.default_rng(41).poisson(3, (len(records), len(genes))).astype(np.float32)
    obs = pd.DataFrame(records, index=[f"cell_{i}" for i in range(len(records))])
    var = pd.DataFrame({"gene_id": genes}, index=[f"symbol_{i}" for i in range(len(genes))])
    path = tmp_path / "TEST_ONLY.h5ad"
    ad.AnnData(sparse.csr_matrix(counts), obs=obs, var=var).write_h5ad(path)
    return path, {"n_obs": len(obs), "n_vars": len(var), "gene_key": "gene_id", "perturbation_key": "target_gene",
                  "batch_key": "batch", "matrix": "X", "control": "non-targeting", "source": "TEST ONLY"}, counts, obs


def test_stream_aggregation_resume_matches_full_and_preserves_full_library(tmp_path, monkeypatch):
    ref = make_demo(tmp_path / "demo", genes=16, targets=4, cells=8)
    original = load_prepared(ref)
    path, source, counts, obs = source_fixture(tmp_path, original)
    full = aggregate_training(source, ref, tmp_path / "full", local_h5ad=path, chunk_rows=3, checkpoint_rows=6)
    original_save = challenge_data.atomic_npz
    def interrupt(path, **arrays):
        original_save(path, **arrays)
        if Path(path).name == "partial.npz":
            raise InterruptedError("TEST interruption after durable checkpoint")
    monkeypatch.setattr(challenge_data, "atomic_npz", interrupt)
    with pytest.raises(InterruptedError):
        aggregate_training(source, ref, tmp_path / "resume", local_h5ad=path, chunk_rows=3, checkpoint_rows=6)
    monkeypatch.setattr(challenge_data, "atomic_npz", original_save)
    resumed = aggregate_training(source, ref, tmp_path / "resume", local_h5ad=path, resume=True, chunk_rows=3, checkpoint_rows=6)
    with np.load(full) as a, np.load(resumed) as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        expected = np.log1p(counts.astype(np.float64) / counts.sum(1)[:, None] * 10000)
        for i, (batch, pert) in enumerate(zip(a["batches"], a["perturbations"])):
            ix = ((obs.batch == batch) & (obs.target_gene == pert)).to_numpy()
            np.testing.assert_allclose(a["sums"][i], expected[ix].sum(0), rtol=1e-6)
    prepared = prepare_student_c(ref, full, tmp_path / "c", min_cells=3, min_controls=3)
    c = load_prepared(prepared)
    assert set(c["meta"].loc[c["meta"].split == "train", "context"]) == {"H1_hESC"}
    assert len(c["genes"]) == 14
    assert len(c["splits"]["val"]) < len(original["splits"]["val"])
    mapping = dict(zip(original["meta"].row_id, range(len(original["meta"]))))
    for i in c["splits"]["val"]:
        np.testing.assert_array_equal(c["delta"][i], original["delta"][mapping[c["meta"].iloc[i].row_id], :-2])
    combined = load_prepared(prepare_student_c(ref, full, tmp_path / "combined", training_source="combined", min_cells=3, min_controls=3))
    assert set(combined["meta"].loc[combined["meta"].split == "train", "context"]) == {"H1_hESC", "K562", "RPE1"}
    assert len(combined["splits"]["val"]) == len(original["splits"]["val"])
    assert load_prepared(ref)["audit"]["fingerprint"] == original["audit"]["fingerprint"]
