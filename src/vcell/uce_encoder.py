"""Pinned UCE-33 control embeddings, preserving upstream chromosome tokenization."""
import gc
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .uce_model import TransformerModel
from .utils import file_sha256, write_json


class PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Only a plain species-offset dictionary is accepted")


def cell_tokens(counts, token_ids, chroms, starts, rng, sample_size=1024):
    """Same log1p-count sampling and genomic ordering as upstream eval_data.py."""
    sequences = []
    for row in counts:
        weights = np.log1p(row.astype(np.float64))
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("UCE cell has no expressed mapped genes")
        selected = rng.choice(len(weights), size=sample_size, p=weights / weights.sum(), replace=True)
        selected = selected[np.argsort(chroms[selected])]
        chromosomes = np.unique(chroms[selected])
        rng.shuffle(chromosomes)
        seq = [3]
        for chrom in chromosomes:
            ix = selected[chroms[selected] == chrom]
            ix = ix[np.argsort(starts[ix])]
            seq.extend([143574 + int(chrom), *token_ids[ix].tolist(), 2])
        if len(seq) > 1536:
            raise ValueError("UCE sentence exceeds pretrained positional encoding")
        sequences.append(seq)
    tokens = np.zeros((len(sequences), max(map(len, sequences))), dtype=np.int64)
    mask = np.zeros_like(tokens, dtype=np.float32)
    for i, seq in enumerate(sequences):
        tokens[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1
    return torch.from_numpy(tokens), torch.from_numpy(mask)


class UCEEncoder:
    def __init__(self, model_dir, auxiliary_dir, device, seed=0):
        root, aux = Path(model_dir), Path(auxiliary_dir)
        cfg = json.loads((root / "config.json").read_text())
        if cfg != dict(token_dim=5120, d_model=1280, nhead=20, d_hid=5120, nlayers=33, output_dim=1280, dropout=.05):
            raise ValueError("Expected pinned UCE-33 architecture")
        # Avoid an additional randomly initialized multi-GB token table.
        with torch.device("meta"):
            model = TransformerModel(**cfg)
            model.pe_embedding = nn.Embedding(145469, 5120)
        weights = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(weights, strict=True, assign=True)
        self.model = model.to(device).eval().requires_grad_(False)
        del weights
        human = torch.load(aux / "protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
                           map_location="cpu", weights_only=True)
        symbols = list(dict.fromkeys(k.upper() for k in human))
        del human
        offsets = PlainUnpickler(io.BytesIO((aux / "species_offsets.pkl").read_bytes())).load()
        offset = offsets["human"]
        if not isinstance(offset, int) or offset < 4 or offset + len(symbols) > 143574:
            raise ValueError("Invalid human token offset")
        token_map = {s: i + offset for i, s in enumerate(symbols)}
        chrom = pd.read_csv(aux / "species_chrom.csv")
        codes = pd.Categorical(chrom.species + "_" + chrom.chromosome).codes
        chrom["code"] = codes
        human_chrom = chrom.loc[chrom.species == "human"].copy()
        human_chrom["gene_symbol"] = human_chrom.gene_symbol.str.upper()
        if human_chrom.gene_symbol.duplicated().any():
            raise ValueError("Ambiguous UCE chromosome annotation")
        self.mapping = {r.gene_symbol: (token_map[r.gene_symbol], int(r.code), int(r.start))
                        for r in human_chrom.itertuples() if r.gene_symbol in token_map}
        self.device, self.seed = device, seed
        gc.collect()

    @torch.inference_mode()
    def encode(self, counts, symbols):
        # Sum duplicate symbols in raw-count space, then apply log1p sampling.
        lookup = {}
        for i, symbol in enumerate(symbols):
            symbol = symbol.upper()
            if symbol in self.mapping:
                lookup.setdefault(symbol, []).append(i)
        if len(lookup) < 100:
            raise ValueError("Fewer than 100 genes overlap UCE human vocabulary")
        names = list(lookup)
        x = np.column_stack([counts[:, lookup[s]].sum(1) for s in names])
        ids, chroms, starts = np.array([self.mapping[s] for s in names]).T
        # Same seed per group makes interrupted group extraction reproducible.
        tokens, mask = cell_tokens(x, ids, chroms, starts, np.random.RandomState(self.seed))
        result = []
        for start in range(0, len(tokens), 2):
            tok, keep = tokens[start:start+2].to(self.device), mask[start:start+2].to(self.device)
            values = nn.functional.normalize(self.model.pe_embedding(tok).permute(1, 0, 2), dim=2)
            _, embedding = self.model(values, keep)
            result.append(embedding.float().cpu().numpy())
        result = np.concatenate(result)
        if result.shape != (len(counts), 1280) or not np.isfinite(result).all():
            raise ValueError("UCE real-weight inference produced invalid embeddings")
        self.coverage = {"input_columns": len(symbols), "mapped_unique_symbols": len(names)}
        return result


def export_uce(data, output, model_dir, auxiliary_dir, device="cuda", max_cells=8):
    from .teacher_pipeline import control_groups
    from .challenge_data import atomic_npz
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = {"data_fingerprint": data["audit"]["fingerprint"], "max_cells": max_cells,
            "checkpoint_sha256": file_sha256(Path(model_dir) / "pytorch_model.bin"),
            "encoder_sha256": file_sha256(__file__), "seed": 0}
    checkpoint = output.with_suffix(".partial.npz")
    features = np.full((len(data["meta"]), 1280), np.nan, dtype=np.float32)
    if checkpoint.exists():
        with np.load(checkpoint) as f:
            if json.loads(str(f["plan"].item())) != plan:
                raise ValueError("Interrupted UCE extraction changed; use a new run name")
            features = f["features"]
    encoder = UCEEncoder(model_dir, auxiliary_dir, torch.device(device))
    records = []
    for index, (rows, counts, genes, symbols, sample) in enumerate(control_groups(data, max_cells, 0), 1):
        sample["uce_mapping"] = {"input_columns": len(symbols),
            "mapped_unique_symbols": len({s.upper() for s in symbols if s.upper() in encoder.mapping}),
            "duplicate_policy": "sum raw counts before log1p sampling"}
        if not np.isfinite(features[rows]).all():
            features[rows] = encoder.encode(counts, symbols).mean(0)
            atomic_npz(checkpoint, features=features, plan=json.dumps(plan, sort_keys=True))
        records.append(sample)
        print(f"UCE control_group={index} dataset={sample['dataset']} batch={sample['batch']} cells={len(counts)}", flush=True)
    if not np.isfinite(features).all():
        raise ValueError("Missing UCE features")
    atomic_npz(output, features=features, row_ids=data["meta"].row_id.to_numpy(dtype="U"),
               data_fingerprint=data["audit"]["fingerprint"])
    write_json(output.with_suffix(".json"), {**plan, "feature_file_sha256": file_sha256(output),
               "artifact_kind": "frozen_control_features_NOT_predictions", "teacher_family": "uce",
               "control_samples": records, "adaptation": "UCE-33 frozen control encoder; VCell response head required",
               "source": "https://github.com/snap-stanford/UCE", "pretraining_overlap": "unverified"})
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output
