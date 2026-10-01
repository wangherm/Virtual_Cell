"""Round 7 data checks, internal calibration, native-teacher review and reporting."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from scipy.optimize import minimize

from .data import open_source, labels, row_selection
from .selection import row_weights, mse
from .specialization import reference_responses
from .utils import write_json, file_sha256


def internal_partition(data):
    """Disjoint batches inside each training context, no HepG2/Jurkat responses."""
    meta = data["meta"]
    parts = {k: [] for k in ("fit", "calibration", "outer")}
    for context, frame in meta.iloc[data["splits"]["train"]].groupby("context", sort=True):
        batches = np.asarray(sorted(frame.batch.unique()))
        if len(batches) < 3:
            raise ValueError(f"{context}: need at least 3 independent batch labels for calibration")
        batches = np.random.default_rng(1701).permutation(batches)
        count = max(1, len(batches)//5)
        for key, selected in (("calibration", batches[:count]), ("outer", batches[count:2*count]), ("fit", batches[2*count:])):
            parts[key].extend(frame.index[frame.batch.isin(selected)].tolist())
    seen = set(meta.iloc[parts["fit"]].perturbation)
    for key in ("calibration", "outer"):
        parts[key] = [i for i in parts[key] if meta.iloc[i].perturbation in seen]
    if any(not v for v in parts.values()):
        raise ValueError("Empty supported internal calibration partition")
    return {k: sorted(v) for k,v in parts.items()}


def prepare_native_inputs(original, panel, output, max_cells=512):
    """Read genuine controls only; record full-library counts and unique symbol mappings."""
    if float(original["audit"]["target_sum"]) != 10000.:
        raise ValueError("Native reviewer normalization requires prepared target_sum=10000")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "audit.json").exists():
        audit = json.loads((output / "audit.json").read_text())
        if audit["source_fingerprint"] != original["audit"]["fingerprint"] or audit["panel_genes"] != panel["genes"].tolist():
            raise ValueError("Traditional input cache changed")
        for d in audit["datasets"]:
            if file_sha256(output / d["file"]) != d["sha256"]:
                raise ValueError("Traditional input checksum mismatch")
        return audit
    manifest = original["audit"]["manifest"]
    sources = {s["id"]: s["path"] for s in original["audit"]["sources"]}
    datasets, raw_checks = [], []
    for entry in manifest["datasets"]:
        if entry["context"] not in set(original["meta"].query("split != 'test'").context):
            continue  # Never open Jurkat raw data in this review.
        a, genes = open_source({**entry, "path": sources[entry["id"]]}, ".")
        try:
            pert, batches = labels(a, entry)
            keep = row_selection(a, entry)
            controls = np.flatnonzero(keep & (pert == "__control__"))
            # Round-robin shuffled batches prevent the largest batch taking the whole sample.
            rng = np.random.default_rng(17)
            pools = [rng.permutation(controls[batches[controls] == b]).tolist() for b in sorted(set(batches[controls]))]
            chosen = []
            while len(chosen) < max_cells and any(pools):
                for pool in pools:
                    if pool and len(chosen) < max_cells:
                        chosen.append(pool.pop())
            chosen = np.sort(chosen)
            matrix = a.X if entry.get("count_layer", "X") == "X" else a.layers[entry["count_layer"]]
            raw = matrix[chosen, :]
            raw = raw.toarray() if sparse.issparse(raw) else np.asarray(raw)
            if not np.isfinite(raw).all() or (raw < 0).any() or not np.allclose(raw, np.rint(raw), rtol=0, atol=1e-5):
                raise ValueError("Native teachers require raw nonnegative integer counts")
            library = raw.sum(1).astype(np.float64)
            valid_cells = library > 0
            raw, library, chosen = raw[valid_cells], library[valid_cells], chosen[valid_cells]
            if len(raw) < 20:
                raise ValueError("Too few genuine control cells for native network inference")
            symbols = a.var["gene_name"].astype(str).to_numpy() if "gene_name" in a.var else genes.copy()
            counts = pd.Series(symbols).value_counts()
            unique = np.array([counts[s] == 1 and s not in {"", "nan", "None"} for s in symbols])
            selected_meta = original["meta"][original["meta"].dataset == entry["id"]]
            targets = sorted(selected_meta.perturbation.unique())
            # Prioritize targets also evaluated in HepG2, independent of effect sizes.
            val_targets = set(original["meta"].iloc[original["splits"]["val"]].perturbation)
            targets = sorted(targets, key=lambda t: (t not in val_targets, t))
            required = unique & (np.isin(genes, panel["genes"]) | np.isin(symbols, targets))
            variable = np.argsort(-raw.var(0), kind="stable")
            extra = [i for i in variable if unique[i] and not required[i]][:500]
            cols = np.sort(np.r_[np.flatnonzero(required), extra].astype(int))
            expressed = raw[:, cols].sum(0) > 0
            cols = cols[expressed]
            file = entry["context"] + ".npz"
            np.savez_compressed(output / file, raw=raw[:, cols].astype(np.float32), library=library,
                genes=np.asarray(genes[cols], dtype="U"), symbols=np.asarray(symbols[cols], dtype="U"),
                cell_ids=a.obs_names[chosen].to_numpy(dtype="U"))
            pd.DataFrame({"gene_id": genes[cols], "symbol": symbols[cols]}).to_csv(output / (entry["context"]+"_genes.csv"), index=False)
            datasets.append({"context": entry["context"], "dataset": entry["id"], "file": file,
                "sha256": file_sha256(output / file), "targets": targets, "n_controls": len(raw),
                "n_available_controls": len(controls), "ambiguous_symbol_columns_excluded": int((~unique).sum()),
                "count_layer": entry.get("count_layer", "X"), "source": sources[entry["id"]]})
            # Reconstruct one real aggregate in each source to check sign, order and full-library normalization.
            row = selected_meta.iloc[0]
            idx = np.array([list(genes).index(g) for g in original["genes"]])
            means = []
            for label in ("__control__", row.perturbation):
                cell_ix = np.flatnonzero(keep & (pert == label) & (batches == row.batch))
                total, n = np.zeros(len(idx)), 0
                for start in range(0, len(cell_ix), 512):
                    z = matrix[cell_ix[start:start+512], :]
                    z = z.toarray() if sparse.issparse(z) else np.asarray(z)
                    libs = z.sum(1)
                    z, libs = z[libs>0], libs[libs>0]
                    total += np.log1p(z[:, idx].astype(float) * (original["audit"]["target_sum"]/libs)[:,None]).sum(0)
                    n += len(z)
                if n == 0:
                    raise ValueError("Missing cells while checking raw delta")
                means.append(total/n)
            position = int(row.name)
            error = float(np.max(np.abs(means[1]-means[0]-original["delta"][position])))
            base_error = float(np.max(np.abs(means[0]-original["baseline"][position])))
            if max(error, base_error) > 2e-5:
                raise ValueError(f"Raw aggregation alignment/sign check failed: {entry['id']} {error} {base_error}")
            raw_checks.append({"row_id": row.row_id, "max_abs_delta_error": error, "max_abs_baseline_error": base_error})
        finally:
            a.file.close()
    audit = {"source_fingerprint": original["audit"]["fingerprint"], "panel_genes": panel["genes"].tolist(),
        "datasets": datasets, "raw_reconstruction_checks": raw_checks, "test_evaluated": False,
        "control_access": "Training and HepG2 control cells allowed; no perturbation labels fit native networks",
        "H1_native_network": "not run: existing H1 preparation contains aggregates, not a local raw-cell source",
        "symbol_policy": "exclude ambiguous symbols explicitly; never silently sum into a different output gene"}
    write_json(output / "audit.json", audit)
    return audit


def scalar_shrink(pred, truth, weights):
    numerator = np.sum(weights[:,None] * pred * truth)
    denominator = np.sum(weights[:,None] * pred * pred)
    return float(np.clip(numerator / max(denominator, 1e-30), 0, 1))


def calibrated_mix(predictions, truth, weights, penalty=1e-6):
    arrays = np.stack(predictions).astype(float)
    gram = np.einsum("irg,jrg,r->ij", arrays, arrays, weights) / truth.shape[1]
    cross = np.einsum("irg,rg,r->i", arrays, truth, weights) / truth.shape[1]
    gram += penalty * np.eye(len(arrays))
    fit = minimize(lambda w: float(w @ gram @ w - 2*w @ cross), np.ones(len(arrays))/len(arrays),
        jac=lambda w: 2*(gram @ w-cross), method="SLSQP", bounds=[(0,1)]*len(arrays),
        constraints={"type": "eq", "fun": lambda w: w.sum()-1}, options={"ftol": 1e-12, "maxiter": 200})
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x / fit.x.sum()


def internal_calibration(data, job, output):
    split = job["internal_split"]
    fit, cal, outer = (np.asarray(split[k], dtype=int) for k in ("fit", "calibration", "outer"))
    folder = Path(job["output_dir"]) / "supervised"
    def read(name, rows):
        with np.load(folder / name) as f:
            if not np.array_equal(f["genes"], data["genes"]) or not np.array_equal(f["row_ids"], data["meta"].iloc[rows].row_id.to_numpy(dtype="U")):
                raise ValueError("Calibration prediction alignment mismatch")
            return f["delta"]
    cp, op = read("calibration_predictions.npz", cal), read("fixed_predictions.npz", outer)
    ref, _ = reference_responses(data, fit)
    cr, ore = ref[data["pert_idx"][cal]], ref[data["pert_idx"][outer]]
    cw, ow = row_weights(data["meta"].iloc[cal]), row_weights(data["meta"].iloc[outer])
    alpha = scalar_shrink(cp, data["delta"][cal], cw)
    weights = calibrated_mix([cp, cr, np.zeros_like(cp)], data["delta"][cal], cw)
    methods = {"qwen_raw": op, "qwen_shrunk": op*alpha, "mean_transfer_fit_only": ore,
               "zero": np.zeros_like(op), "convex_qwen_reference_zero": weights[0]*op+weights[1]*ore}
    scores = {k: mse(v, data["delta"][outer], ow) for k,v in methods.items()}
    output = Path(output)
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "internal_predictions.npz", genes=data["genes"],
        fit_row_ids=data["meta"].iloc[fit].row_id.to_numpy(dtype="U"),
        calibration_row_ids=data["meta"].iloc[cal].row_id.to_numpy(dtype="U"),
        outer_row_ids=data["meta"].iloc[outer].row_id.to_numpy(dtype="U"),
        calibration_truth=data["delta"][cal], outer_truth=data["delta"][outer],
        calibration_prediction=cp, outer_prediction=op)
    write_json(output / "audit.json", {"alpha": alpha, "weights": dict(zip(["qwen", "reference", "zero"], weights.tolist())),
        "outer_mse": scores, "split": split, "test_evaluated": False,
        "task": "disjoint-batch transfer inside training contexts; NOT new-context generalization",
        "checkpoint": "fixed update; no calibration/outer label selection",
        "deployment": "diagnostic only; scalers not applied to a differently fitted full-data model"})
    return scores


def native_ensemble_audit(data, job, inputs, native_root, output):
    """Fit a joint signed mixture on calibration batches; score other training batches."""
    split = job["internal_split"]
    fit = np.asarray(split["fit"], dtype=int)
    reference, _ = reference_responses(data, fit)
    native = {}
    mapping = json.loads((Path(inputs)/"audit.json").read_text())
    for entry in mapping["datasets"]:
        context = entry["context"]
        folder = Path(native_root)/"celloracle"/context
        if not (folder/"COMPLETE.json").exists():
            continue
        genes = pd.read_csv(Path(inputs)/(context+"_genes.csv"))
        lookup = dict(zip(genes.symbol,genes.gene_id))
        for record in json.loads((folder/"COMPLETE.json").read_text())["records"]:
            if record["status"] != "complete":
                continue
            frame = pd.read_csv(folder/(record["target"]+".csv"))
            ids = frame.Gene.map(lookup)
            mask = ids.isin(data["genes"]).to_numpy()
            cols = np.asarray([list(data["genes"]).index(g) for g in ids[mask]])
            if len(cols):
                native[context,record["target"]] = cols,frame.delta.to_numpy()[mask]
    matrices, truths, weights, identities = {}, {}, {}, {}
    for key, filename in (("calibration","calibration_predictions.npz"),("outer","fixed_predictions.npz")):
        ix = np.asarray(split[key], dtype=int)
        with np.load(Path(job["output_dir"])/"supervised"/filename) as f:
            if not np.array_equal(f["row_ids"],data["meta"].iloc[ix].row_id.to_numpy(dtype="U")) or not np.array_equal(f["genes"],data["genes"]):
                raise ValueError("Native ensemble/student alignment mismatch")
            prediction=f["delta"]
        usable = [j for j,r in enumerate(ix) if (data["meta"].iloc[r].context,data["meta"].iloc[r].perturbation) in native]
        if not usable:
            write_json(Path(output)/"native_ensemble.json", {"status":"unavailable", "reason":f"No jointly supported {key} rows", "test_evaluated":False})
            return
        rw=row_weights(data["meta"].iloc[ix[usable]])
        x,y,w,ids = [],[],[],[]
        for position,j in enumerate(usable):
            r=ix[j]; row=data["meta"].iloc[r]
            cols,co=native[row.context,row.perturbation]
            x.append(np.column_stack([prediction[j,cols],co,reference[data["pert_idx"][r],cols],np.zeros(len(cols))]))
            y.append(data["delta"][r,cols]);w.append(np.full(len(cols),rw[position]/len(cols)))
            ids.append({"row_id":row.row_id,"context":row.context,"target":row.perturbation,"genes":data["genes"][cols].tolist()})
        matrices[key],truths[key],weights[key],identities[key]=np.vstack(x),np.concatenate(y),np.concatenate(w),ids
    names=["qwen","celloracle","reference","zero"]
    x,y,w=matrices["calibration"],truths["calibration"],weights["calibration"]
    mixture=calibrated_mix([x[:,i,None] for i in range(4)],y[:,None],w)
    alpha=scalar_shrink(x[:,1,None],y[:,None],w)
    x,y,w=matrices["outer"],truths["outer"],weights["outer"]
    scores={n:float(w @ ((x[:,i]-y)**2)) for i,n in enumerate(names)}
    scores["celloracle_shrunk"]=float(w @ ((alpha*x[:,1]-y)**2))
    scores["ensemble"]=float(w @ ((x @ mixture-y)**2))
    np.savez_compressed(Path(output)/"native_joint_predictions.npz", calibration=matrices["calibration"],
        calibration_truth=truths["calibration"],calibration_weights=weights["calibration"],
        outer=x,outer_truth=y,outer_weights=w,model_order=np.array(names))
    write_json(Path(output)/"native_ensemble.json", {"status":"complete", "weights":dict(zip(names,mixture.tolist())),
        "celloracle_alpha":alpha,"outer_mse":scores,"joint_panel_rows":identities,"test_evaluated":False,
        "scope":"Known-context, disjoint-batch diagnostic on jointly supported measured genes and targets only",
        "fit_membership":data["meta"].iloc[fit].row_id.tolist(),
        "distillation":"not auto-started: this diagnostic does not establish unseen-context teacher superiority"})


def native_review(data, inputs, native_root, student_predictions, output):
    """Matched supported-panel review; unsupported genes/targets are never zero-filled."""
    rows, signed = [], []
    mapping = json.loads((Path(inputs)/"audit.json").read_text())
    for entry in mapping["datasets"]:
        context = entry["context"]
        genes = pd.read_csv(Path(inputs)/(context+"_genes.csv"))
        symbol_to_id = dict(zip(genes.symbol, genes.gene_id))
        for family in ("celloracle", "sctenifold"):
            folder = Path(native_root)/family/context
            if not (folder/"COMPLETE.json").exists():
                continue
            info = json.loads((folder/"COMPLETE.json").read_text())
            for item in info["records"]:
                if item["status"] != "complete":
                    continue
                target = item["target"]
                ix = np.flatnonzero(((data["meta"].context==context)&(data["meta"].perturbation==target)&(data["meta"].split!="test")).to_numpy())
                if not len(ix):
                    continue
                frame = pd.read_csv(folder/(target+".csv"))
                mapped = frame.Gene.map(symbol_to_id)
                valid = mapped.isin(data["genes"]).to_numpy()
                ids = mapped[valid].tolist()
                cols = np.asarray([list(data["genes"]).index(g) for g in ids])
                if len(cols) < 3:
                    continue
                score = frame.loc[valid, "delta" if family=="celloracle" else "Distance"].to_numpy()
                truth = data["delta"][ix][:,cols].mean(0)
                correlation = spearmanr(np.abs(score), np.abs(truth)).statistic if np.std(score)>1e-12 and np.std(np.abs(truth))>1e-12 else None
                k = min(20, len(cols))
                overlap = len(set(np.argsort(-np.abs(score), kind="stable")[:k]) & set(np.argsort(-np.abs(truth), kind="stable")[:k])) / k
                record = {"family": family, "context": context, "target": target, "genes": len(cols),
                    "rank_spearman": float(correlation) if correlation is not None and np.isfinite(correlation) else None,
                    "top20_overlap": overlap, "split": data["meta"].iloc[ix[0]].split}
                if family == "celloracle":
                    record.update(mse_delta=float(np.mean((data["delta"][ix][:,cols]-score)**2)),
                                  mse_zero=float(np.mean(data["delta"][ix][:,cols]**2)))
                    signed.append((ix, cols, score))
                rows.append(record)
                if record["split"] == "val":
                    val_lookup = {r:i for i,r in enumerate(data["splits"]["val"])}
                    vi = [val_lookup[r] for r in ix]
                    for name, pred in student_predictions.items():
                        p = pred[vi][:,cols].mean(0)
                        r = spearmanr(np.abs(p), np.abs(truth)).statistic if np.std(p)>1e-12 and np.std(np.abs(truth))>1e-12 else None
                        rows.append({"family": name, "context": context, "target": target, "genes": len(cols),
                            "matched_to": family, "rank_spearman": float(r) if r is not None and np.isfinite(r) else None,
                            "mse_delta": float(np.mean((pred[vi][:,cols]-data["delta"][ix][:,cols])**2)), "split": "val"})
    output = Path(output)
    pd.DataFrame(rows).to_csv(output/"traditional_review.csv", index=False)
    # Global CellOracle amplitude fit uses training labels only, applied to HepG2 supported cells.
    numer, denom = 0., 0.
    for ix, cols, score in signed:
        if data["meta"].iloc[ix[0]].split == "train":
            truth = data["delta"][ix][:,cols].mean(0)
            numer += float(score @ truth)/len(cols)
            denom += float(score @ score)/len(cols)
    alpha = float(np.clip(numer/max(denom,1e-30), 0, 1)) if denom>0 else None
    calibrated = []
    for ix, cols, score in signed:
        if data["meta"].iloc[ix[0]].split == "val" and alpha is not None:
            calibrated.append({"target": data["meta"].iloc[ix[0]].perturbation,
                "mse_raw": float(np.mean((data["delta"][ix][:,cols]-score)**2)),
                "mse_calibrated": float(np.mean((data["delta"][ix][:,cols]-alpha*score)**2)),
                "mse_zero": float(np.mean(data["delta"][ix][:,cols]**2)), "genes": len(cols)})
    write_json(output/"celloracle_calibration.json", {"alpha": alpha, "fit": "original training context-target means only",
        "HepG2_supported_targets": calibrated, "test_evaluated": False,
        "limitation": "Different supported gene panels by context; this is a native-tool diagnostic, not the 1834-gene primary score",
        "sctenifold": "unsigned rankings only; excluded from expression mixture and distillation"})
    return rows
