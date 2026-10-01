"""Supplement an existing Round 7 run with native reviews; never train a student."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_round7 import REPO, run_jobs, package
from vcell.data import load_prepared
from vcell.round7 import prepare_native_inputs, native_review, internal_calibration, native_ensemble_audit
from vcell.specialization import compare_validation, make_modules
from vcell.train import source_fingerprint, simple_baselines
from vcell.utils import file_sha256, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_regenerated_inputs(previous, current):
    """Check numeric arrays and exported CSV mappings without unpickling old strings."""
    previous, current = Path(previous), Path(current)
    before, after = read(previous/"audit.json"), read(current/"audit.json")
    for key in ("source_fingerprint", "panel_genes"):
        if before[key] != after[key]:
            raise ValueError("Regenerated native inputs changed source/panel")
    old = {e["context"]: e for e in before["datasets"]}
    if set(old) != {e["context"] for e in after["datasets"]}:
        raise ValueError("Regenerated native context coverage changed")
    for entry in after["datasets"]:
        prior = old[entry["context"]]
        if file_sha256(previous/prior["file"]) != prior["sha256"]:
            raise ValueError("Original native input checksum mismatch")
        if entry["targets"] != prior["targets"]:
            raise ValueError("Regenerated native target order changed")
        with np.load(previous/prior["file"], allow_pickle=False) as a, np.load(current/entry["file"], allow_pickle=False) as b:
            for key in ("raw", "library", "cell_ids"):
                if not np.array_equal(a[key], b[key]):
                    raise ValueError(f"Regenerated native {key} changed")
            # Materialize EVERY regenerated array: object arrays must never escape again.
            for key in b.files:
                if b[key].dtype.hasobject:
                    raise ValueError("Object array in regenerated input")
        filename = entry["context"]+"_genes.csv"
        a, b = (pd.read_csv(p/filename, dtype=str, keep_default_na=False) for p in (previous,current))
        if not a.equals(b):
            raise ValueError("Regenerated native gene order/mapping changed")


def parent_results(source):
    """Validate completed student provenance and prediction alignment, not current code identity."""
    plan = read(source/"plan.json")
    datasets = {k:load_prepared(source/"prepared"/k) for k in ("original_only","plus_h1")}
    ref = datasets["original_only"]
    predictions = {k+"/"+n:v[d["splits"]["val"]] for k,d in datasets.items() for n,v in simple_baselines(d).items()}
    files = {"plan.json", "jobs.json", "comparison.csv", "native_inputs/audit.json"}
    internal = None
    count = 0
    for name, job in read(source/"jobs.json").items():
        if job["resource"] != "gpu":
            continue
        count += 1
        cfg_path = source/"configs"/(name+".json")
        cfg = read(cfg_path)
        student = source/"students"/name
        manifest, complete = read(student/"run_manifest.json"), read(student/"COMPLETE.json")
        if (manifest["source_fingerprint"] != plan["source_fingerprint"] or manifest["config"] != cfg
                or manifest["run_fingerprint"] != complete["run_fingerprint"]):
            raise ValueError(f"Student provenance mismatch: {name}")
        dataset = next((d for d in datasets.values() if d["audit"]["fingerprint"] == manifest["data_fingerprint"]), None)
        if dataset is None:
            raise ValueError(f"Student data mismatch: {name}")
        folder = student/"supervised"
        endpoint = read(folder/"fixed_endpoint.json")
        if endpoint["optimizer_step"] != plan["base_config"]["max_steps"]:
            raise ValueError(f"Student did not finish fixed updates: {name}")
        rows = np.asarray(cfg["internal_split"]["outer"]) if cfg.get("internal_split") else dataset["splits"]["val"]
        with np.load(folder/"fixed_predictions.npz", allow_pickle=False) as f:
            if (not np.array_equal(f["genes"], ref["genes"])
                    or not np.array_equal(f["row_ids"], dataset["meta"].iloc[rows].row_id.to_numpy(dtype="U"))
                    or str(f["data_fingerprint"]) != manifest["data_fingerprint"]):
                raise ValueError(f"Student prediction alignment mismatch: {name}")
            if cfg.get("internal_split"):
                internal = copy.deepcopy(cfg)
                internal["output_dir"] = str(student)
                files.add(str((folder/"calibration_predictions.npz").relative_to(source)))
            else:
                if not np.array_equal(dataset["meta"].iloc[rows].row_id,ref["meta"].iloc[ref["splits"]["val"]].row_id):
                    raise ValueError("Different primary validation rows")
                predictions[name] = f["delta"]
        for path in (cfg_path,student/"run_manifest.json",student/"COMPLETE.json",folder/"fixed_endpoint.json",folder/"fixed_predictions.npz"):
            files.add(str(path.relative_to(source)))
    if internal is None or not count:
        raise ValueError("Missing completed internal calibration student")
    for entry in read(source/"native_inputs/audit.json")["datasets"]:
        files.update(["native_inputs/"+entry["file"], "native_inputs/"+entry["context"]+"_genes.csv"])
    hashes = {name:file_sha256(source/name) for name in sorted(files)}
    return plan, datasets, predictions, internal, count, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="/root/autodl-tmp/vcell-work")
    parser.add_argument("--source-run")
    parser.add_argument("--data", help="Original prepared data if the source run used a custom --data path")
    parser.add_argument("--name", default="round7_native_retry_01")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        parser.error("Use a simple output name")
    work = Path(args.work_dir).resolve()
    source = Path(args.source_run or work/"runs/round7_01").resolve()
    root = work/"runs"/args.name
    if source == root:
        raise ValueError("Choose a separate native-review directory")
    parent, data, predictions, internal, count, hashes = parent_results(source)
    original = load_prepared(args.data or parent["base_config"]["data_dir"])
    if original["audit"]["fingerprint"] != parent["original_fingerprint"]:
        raise ValueError("Original preparation changed")
    plan = {"kind":"native-only supplement; zero student training", "parent_run":str(source),
        "parent_artifacts":hashes,"parent_source_fingerprint":parent["source_fingerprint"],
        "source_fingerprint":source_fingerprint(), "scripts":{n:file_sha256(REPO/"scripts"/n) for n in
            ("retry_round7_native.py","run_round7.py","run_traditional_teacher.py","run_traditional_autodl.sh")},
        "native_targets":parent["native_targets"], "test_evaluated":False}
    if root.exists() and any(root.iterdir()):
        if not args.resume or not (root/"plan.json").exists() or read(root/"plan.json") != plan:
            raise ValueError("Resume identical supplement or choose a fresh output name")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/"plan.json",plan)
    prepare_native_inputs(original,data["original_only"],root/"native_inputs")
    verify_regenerated_inputs(source/"native_inputs",root/"native_inputs")
    write_json(root/"input_repair.json",{"counts_library_cells_gene_mapping_identical":True,
        "change":"Regenerated Unicode metadata from original sources; no pickle loading", "test_evaluated":False})
    jobs={family:{"resource":"cpu","complete":str(root/"traditional"/family/"COMPLETE.json"),
        "cmd":["bash",str(REPO/"scripts/run_traditional_autodl.sh"),family,str(root/"native_inputs"),
               str(root/"traditional"/family),str(work),str(parent["native_targets"])]} for family in ("celloracle","sctenifold")}
    write_json(root/"jobs.json",jobs)
    finished=run_jobs(jobs,root,0,args.resume)
    try:
        reference=data["original_only"]
        modules,_=make_modules(reference,min(8,len(reference["genes"])))
        compare_validation(reference,predictions,modules,root)
        previous=pd.read_csv(source/"comparison.csv").set_index("model").mse_delta.sort_index()
        current=pd.read_csv(root/"comparison.csv").set_index("model").mse_delta.sort_index()
        if not previous.index.equals(current.index) or not np.allclose(previous,current,rtol=0,atol=1e-10):
            raise ValueError("Reused student comparison changed")
        internal_calibration(data["plus_h1"],internal,root/"calibration")
        native_review(reference,root/"native_inputs",root/"traditional",predictions,root)
        if finished.get("celloracle")==0:
            native_ensemble_audit(data["plus_h1"],internal,root/"native_inputs",root/"traditional",root/"calibration")
        failed=[n for n in jobs if finished.get(n)!=0]
        status={"kind":plan["kind"],"students_reused":count,"students_trained":0,"failed_jobs":failed,
            "parent_run":str(source),"parent_primary_scores_verified":True,"test_evaluated":False}
        write_json(root/("INCOMPLETE.json" if failed else "COMPLETE.json"),status)
        if not failed and (root/"INCOMPLETE.json").exists():
            (root/"INCOMPLETE.json").unlink()
    except Exception as error:
        write_json(root/"INCOMPLETE.json",{"phase":"reporting","error":str(error),"test_evaluated":False})
        raise
    finally:
        package(root)
    print(f"NATIVE SUPPLEMENT: {root}; students reused={count}; failed={failed}",flush=True)
    if failed:
        raise RuntimeError(f"Native jobs failed: {failed}; inspect included log tails")
    return root


if __name__ == "__main__":
    main()
