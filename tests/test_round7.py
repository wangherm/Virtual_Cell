"""Fixed-update and native-tool software checks, not biological validation."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")
from test_qwen import setup
from test_specialization import challenge_fixture
from vcell.data import save_prepared, load_prepared
from vcell.fixed_training import response_basis, step_rows
from vcell.qwen import run_qwen, load_qwen_checkpoint, predict
from vcell.round7 import internal_partition, scalar_shrink, calibrated_mix, prepare_native_inputs
from vcell.train import tensors
from vcell.utils import write_json


def fixed(cfg):
    c = copy.deepcopy(cfg)
    c.update(max_steps=3, eval_every_steps=1, max_train_rows=None, max_val_rows=None,
             batch_size=2, gradient_accumulation=2, log_every=1)
    c["model"]["gradient_checkpointing"] = False
    return c


def test_exact_steps_endpoint_reload_and_resume(setup, monkeypatch):
    import vcell.fixed_training as trainer
    cfg, data = setup
    cfg = fixed(cfg)
    cfg["model"]["response_rank"] = 3
    reference = copy.deepcopy(cfg)
    reference["output_dir"] += "_reference"
    run_qwen(reference)
    save = trainer.atomic_torch_save
    def interrupt(obj, path):
        save(obj, path)
        if Path(path).name == "last.pt" and obj["optimizer_step"] == 1:
            raise InterruptedError("fixture")
    monkeypatch.setattr(trainer, "atomic_torch_save", interrupt)
    with pytest.raises(InterruptedError):
        run_qwen(cfg)
    monkeypatch.setattr(trainer, "atomic_torch_save", save)
    run_qwen(cfg, resume=True)
    folder = Path(cfg["output_dir"])/"supervised"
    endpoint = json.loads((folder/"fixed_endpoint.json").read_text())
    assert endpoint["optimizer_step"] == 3 and endpoint["examples_drawn"] == 12
    assert sum(endpoint["examples_by_context"].values()) == 12
    with np.load(folder/"fixed_predictions.npz") as a, np.load(Path(reference["output_dir"])/"supervised/fixed_predictions.npz") as b:
        np.testing.assert_allclose(a["delta"], b["delta"], atol=1e-6)
        expected = a["delta"].copy()
    model, saved = load_qwen_checkpoint(folder/"endpoint.pt")
    norm = {k:v.numpy() if torch.is_tensor(v) else v for k,v in saved["normalization"].items()}
    actual = predict(model, tensors(data,norm), data["splits"]["val"], "cpu", 2)*norm["scale"]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_basis_cannot_use_held_labels_and_sampling_has_full_passes(setup):
    _, data = setup
    train = data["splits"]["train"]
    a, _ = response_basis(data, train, 3)
    other = copy.deepcopy(data)
    other["delta"][data["splits"]["val"]] += 10000
    other["delta"][data["splits"]["test"]] -= 10000
    b, _ = response_basis(other, train, 3)
    np.testing.assert_allclose(a,b)
    np.testing.assert_allclose(a @ a.T, np.eye(3), atol=1e-5)
    stream = np.concatenate([step_rows(np.arange(5),17,s,4) for s in range(5)])
    for start in range(0,20,5):
        assert sorted(stream[start:start+5]) == list(range(5))


def test_global_calibration_and_zero_expert():
    truth = np.array([[1.,-2.], [2.,-3.]])
    weights = np.array([.5,.5])
    assert scalar_shrink(2*truth,truth,weights) == .5
    assert scalar_shrink(-truth,truth,weights) == 0
    mix = calibrated_mix([-truth,np.zeros_like(truth)],truth,weights)
    assert mix[1] > .999


def expand_batches(data, dest):
    # Real tests use three distinct synthetic batches without changing any split/context.
    m = data["meta"].copy()
    copies, base, delta, pi = [], [], [], []
    for i in range(3):
        x = m.copy()
        x["row_id"] = x.row_id+f"_rep{i}"
        x["batch"] = f"batch{i}"
        copies.append(x)
        base.append(data["baseline"]);delta.append(data["delta"]);pi.append(data["pert_idx"])
    save_prepared(dest,np.vstack(base),np.vstack(delta),np.concatenate(pi),data["genes"],data["perturbations"],
                  pd.concat(copies,ignore_index=True),copy.deepcopy(data["audit"]))
    return load_prepared(dest)


def test_round7_all_jobs_fixed_panel_calibration_resume(setup, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    runner = importlib.import_module("run_round7")
    cfg, data = setup
    data = expand_batches(data,tmp_path/"expanded")
    challenge_fixture(data,tmp_path/"challenge")
    write_json(tmp_path/"student.json",cfg)
    work=tmp_path/"work";work.mkdir()
    monkeypatch.setattr(runner.shutil,"disk_usage",lambda _:SimpleNamespace(free=100*1024**3))
    args=["--work-dir",str(work),"--data",str(tmp_path/"expanded"),"--challenge-data",str(tmp_path/"challenge"),
        "--student-config",str(tmp_path/"student.json"),"--model-dir",cfg["model"]["model_id"],"--device","cpu",
        "--max-steps","2","--eval-every-steps","1","--batch-size","2","--seeds","17","--traditional"]
    # Actual isolated concurrent Qwen processes on tiny artificial backbones.
    root=runner.main(args)
    assert json.loads((root/"COMPLETE.json").read_text())["students_completed"]==6
    assert json.loads((root/"initialization_audit.json").read_text())["paired_shared_weights_match"]
    assert (root/"calibration/internal_predictions.npz").exists()
    audit = json.loads((root/"calibration/audit.json").read_text())
    internal_summary = pd.read_csv(root/"students/internal_calibration/summary.csv").set_index("model")
    assert internal_summary.loc["mean_transfer", "mse_delta"] == pytest.approx(audit["outer_mse"]["mean_transfer_fit_only"])
    assert len(pd.read_csv(root/"comparison.csv"))==9
    runner.main(args+["--resume","--parallel-students","1"])
    with pytest.raises(ValueError,match="Resume"):
        runner.main(args+["--resume","--max-steps","3"])
    # Native-only recovery reuses all six predictions, even after legacy string serialization.
    native_audit = prepare_native_inputs(data, load_prepared(root/"prepared/original_only"), root/"native_inputs")
    from vcell.utils import file_sha256
    for entry in native_audit["datasets"]:
        file = root/"native_inputs"/entry["file"]
        with np.load(file, allow_pickle=False) as f:
            values = {key:f[key] for key in f.files}
        values["symbols"] = values["symbols"].astype(object)
        np.savez_compressed(file, **values)
        entry["sha256"] = file_sha256(file)
    write_json(root/"native_inputs/audit.json", native_audit)
    retry = importlib.import_module("retry_round7_native")
    before = {str(p.relative_to(root)):file_sha256(p) for p in root.rglob("*") if p.is_file()}
    def failed_native_jobs(jobs, output, gpu_slots, resume):
        assert gpu_slots == 0 and set(jobs) == {"celloracle","sctenifold"}
        assert all(j["resource"] == "cpu" and "qwen-run" not in j["cmd"] for j in jobs.values())
        (output/"logs").mkdir(exist_ok=True)
        (output/"logs/celloracle.log").write_text("fixture native failure\n")
        return {n:1 for n in jobs}
    monkeypatch.setattr(retry,"run_jobs",failed_native_jobs)
    recovery=["--work-dir",str(work),"--source-run",str(root),"--data",str(tmp_path/"expanded")]
    with pytest.raises(RuntimeError,match="Native jobs failed"):
        retry.main(recovery)
    supplement=work/"runs/round7_native_retry_01"
    assert json.loads((supplement/"INCOMPLETE.json").read_text())["students_trained"] == 0
    assert json.loads((supplement/"input_repair.json").read_text())["counts_library_cells_gene_mapping_identical"]
    assert before == {str(p.relative_to(root)):file_sha256(p) for p in root.rglob("*") if p.is_file()}
    import tarfile
    with tarfile.open(supplement/"round7_review.tar.gz") as archive:
        assert archive.extractfile("logs/celloracle.log").read() == b"fixture native failure\n"
    # A changed parent result may not be silently reused on resume.
    changed = root/"students/plain_plus_h1_seed17/supervised/fixed_endpoint.json"
    endpoint = json.loads(changed.read_text()); endpoint["optimizer_step"] = 1
    write_json(changed, endpoint)
    with pytest.raises(ValueError,match="finish fixed updates"):
        retry.main(recovery+["--resume"])


def test_native_controls_reconstruct_real_aggregation_and_skip_jurkat(setup,tmp_path):
    _,data=setup
    audit=prepare_native_inputs(data,data,tmp_path/"native",max_cells=25)
    assert {x["context"] for x in audit["datasets"]}=={"K562","RPE1","HepG2"}
    assert max(x["max_abs_delta_error"] for x in audit["raw_reconstruction_checks"])<2e-5
    for entry in audit["datasets"]:
        with np.load(tmp_path/"native"/entry["file"], allow_pickle=False) as f:
            for key in f.files:
                assert not f[key].dtype.hasobject
            assert f["symbols"].dtype.kind == "U"


def test_actual_sctenifold_native_api(tmp_path,monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path/"mpl"))
    pytest.importorskip("scTenifold")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    worker=importlib.import_module("run_traditional_teacher")
    rng=np.random.default_rng(17)
    raw=rng.poisson(4,size=(40,16)).astype(float)
    raw[:,1]=raw[:,0]*2+rng.poisson(1,40)
    symbols=np.array([f"GENE{i}" for i in range(16)])
    records,info=worker.run_sctenifold(raw,symbols,["GENE0"],tmp_path,17,1,16)
    assert records[0]["status"]=="complete"
    frame=pd.read_csv(tmp_path/"GENE0.csv")
    assert len(frame)==16 and np.isfinite(frame.Distance).all()
    assert "NOT delta" in info["prediction_space"]


def test_native_mixture_uses_calibration_labels_only(setup, tmp_path):
    from vcell.round7 import native_ensemble_audit
    _, original = setup
    data = expand_batches(original, tmp_path/"expanded")
    split = internal_partition(data)
    job = {"internal_split": split, "output_dir": str(tmp_path/"student")}
    folder = tmp_path/"student/supervised"
    folder.mkdir(parents=True)
    for key, filename in (("calibration", "calibration_predictions.npz"), ("outer", "fixed_predictions.npz")):
        rows = np.asarray(split[key], dtype=int)
        np.savez(folder/filename, genes=data["genes"], row_ids=data["meta"].iloc[rows].row_id.to_numpy(dtype="U"),
                 delta=data["delta"][rows]*1.7)
    inputs, native, out = tmp_path/"inputs", tmp_path/"native", tmp_path/"review"
    inputs.mkdir(); out.mkdir()
    entries = []
    for context in data["meta"].iloc[split["fit"]].context.unique():
        entries.append({"context": context})
        symbols = [f"G{i}" for i in range(len(data["genes"]))]
        pd.DataFrame({"symbol": symbols, "gene_id": data["genes"]}).to_csv(inputs/f"{context}_genes.csv", index=False)
        dest = native/"celloracle"/context
        dest.mkdir(parents=True)
        targets = data["meta"].query("context == @context").perturbation.unique()
        for target in targets:
            pd.DataFrame({"Gene": symbols, "delta": np.ones(len(symbols))}).to_csv(dest/f"{target}.csv", index=False)
        write_json(dest/"COMPLETE.json", {"records": [{"target": t, "status": "complete"} for t in targets]})
    write_json(inputs/"audit.json", {"datasets": entries})
    native_ensemble_audit(data, job, inputs, native, out)
    before = json.loads((out/"native_ensemble.json").read_text())
    assert before["status"] == "complete"
    assert sum(before["weights"].values()) == pytest.approx(1)
    assert min(before["weights"].values()) >= 0
    changed = copy.deepcopy(data)
    changed["delta"][split["outer"]] += 100
    changed["delta"][changed["splits"]["val"]] -= 100
    changed["delta"][changed["splits"]["test"]] += 100
    native_ensemble_audit(changed, job, inputs, native, out)
    after = json.loads((out/"native_ensemble.json").read_text())
    assert before["weights"] == after["weights"]
    assert before["celloracle_alpha"] == after["celloracle_alpha"]
    assert before["outer_mse"] != after["outer_mse"]
