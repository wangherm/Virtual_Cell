"""Static GO/Reactome co-annotation features; no expression labels are consulted."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .utils import file_sha256, write_json


def functional_vectors(perturbations, libraries, dimensions=128, neighbors=8):
    if dimensions < 2 or neighbors < 1:
        raise ValueError("Invalid functional graph dimensions")
    names = [str(p).upper() for p in perturbations]
    if len(set(names)) != len(names):
        raise ValueError("Ambiguous case-insensitive perturbation names")
    lookup = {name: i for i, name in enumerate(names)}
    ii, jj, terms = [], [], []
    for path in libraries:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            values = line.split("\t")
            if len(values) < 3:
                raise ValueError("Expected GMT gene-set annotations")
            members = sorted({lookup[g.upper()] for g in values[2:] if g.upper() in lookup})
            if not members:
                continue
            term = len(terms)
            terms.append(Path(path).stem + ":" + values[0])
            ii.extend(members)
            jj.extend([term] * len(members))
    membership = sparse.csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(names), len(terms)))
    known = np.asarray(membership.sum(1)).ravel() > 0
    if known.sum() < 2:
        raise ValueError("Fewer than two perturbations have functional annotations")
    inverse = 1 / np.sqrt(np.maximum(np.asarray(membership.sum(1)).ravel(), 1))
    unit = sparse.diags(inverse) @ membership
    # Edges mean shared annotated function, NOT a causal regulatory relationship.
    graph = (unit @ unit.T).toarray()
    np.fill_diagonal(graph, 0)
    for i in range(len(names)):
        keep = np.argsort(-graph[i], kind="stable")[:neighbors]
        remove = np.ones(len(names), dtype=bool)
        remove[keep] = False
        graph[i, remove] = 0
    graph /= np.maximum(graph.sum(1, keepdims=True), 1e-12)
    projection = np.zeros((len(terms), dimensions))
    for i, term in enumerate(terms):
        digest = hashlib.sha256(term.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        projection[i] = rng.choice([-1., 1.], dimensions) / np.sqrt(dimensions)
    features = unit @ projection
    features = .5 * features + .5 * graph @ features
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    features = np.column_stack([features, known.astype(float)]).astype(np.float32)
    return features, {"annotated": int(known.sum()), "total": len(names), "missing": [names[i] for i in np.flatnonzero(~known)],
                      "terms": terms, "directed_edges": int((graph > 0).sum()), "neighbors": neighbors,
                      "graph_kind": "shared GO/Reactome annotation cosine kNN; one diffusion step, not causal PPI"}


def prepare_function(data, assets, output):
    assets, output = Path(assets), Path(output)
    sources = json.loads((assets / "sources.json").read_text())
    paths = []
    for item in sources["files"]:
        path = assets / (item["library"] + ".gmt")
        if file_sha256(path) != item["sha256"]:
            raise ValueError("Functional annotation snapshot changed")
        paths.append(path)
    vectors, audit = functional_vectors(data["perturbations"], paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vectors=vectors, perturbations=data["perturbations"],
                        data_fingerprint=data["audit"]["fingerprint"])
    write_json(output.with_suffix(".json"), {**audit, "sources": sources, "sha256": file_sha256(output)})
    print(f"FUNCTIONAL GRAPH: {audit['annotated']}/{audit['total']} targets annotated; {audit['directed_edges']} edges", flush=True)
    return output


def load_function(spec, perturbations):
    path = Path(spec["path"])
    if file_sha256(path) != spec["sha256"]:
        raise ValueError("Functional feature checksum changed")
    with np.load(path, allow_pickle=False) as f:
        if not np.array_equal(f["perturbations"], perturbations):
            raise ValueError("Functional perturbation order mismatch")
        values = f["vectors"].astype(np.float32)
    if values.ndim != 2 or len(values) != len(perturbations) or not np.isfinite(values).all():
        raise ValueError("Invalid functional feature vectors")
    return values
