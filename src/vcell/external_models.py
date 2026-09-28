"""Optional native upstream adapters. No biological weights are bundled.

scGPT/scFoundation expose frozen control-cell representations here, followed by
a VCell conditional response head. They are not official perturbation decoders.
State exposes its actual pretrained State Transition expression predictions.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from .utils import file_sha256


def verify_source(cfg):
    root = Path(cfg["source_dir"]).resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != cfg["source_revision"]:
        raise ValueError(f"Upstream source mismatch: {actual}; expected {cfg['source_revision']}")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise ValueError("Upstream checkout has modified tracked files; record a clean pinned revision")
    return root


def verify_weights(cfg):
    actual = file_sha256(cfg["checkpoint"])
    if actual != cfg["checkpoint_sha256"]:
        raise ValueError("Teacher checkpoint SHA256 does not match configuration")
    return actual


def select_symbols(symbols, vocabulary):
    if len(set(symbols)) != len(symbols):
        raise ValueError("Duplicate gene symbols: resolve explicitly before using a symbol-vocabulary teacher")
    lookup = {name: i for i, name in enumerate(vocabulary)}
    src = [i for i, name in enumerate(symbols) if name in lookup]
    if len(src) < 2:
        raise ValueError("Fewer than two genes overlap the teacher vocabulary")
    return np.asarray(src), np.asarray([lookup[symbols[i]] for i in src])


class ScGPTEncoder:
    def __init__(self, cfg, device):
        verify_weights(cfg)
        from .scgpt_encoder import FrozenScGPT
        if cfg.get("implementation") != "vcell_frozen_scgpt_v1":
            raise ValueError("Use the verified scGPT launcher or set implementation: vcell_frozen_scgpt_v1")
        for key in ("args", "vocab"):
            if file_sha256(cfg[key]) != cfg[key + "_sha256"]:
                raise ValueError(f"scGPT {key} checksum mismatch")
        args = json.loads(Path(cfg["args"]).read_text())
        self.vocab = json.loads(Path(cfg["vocab"]).read_text())
        if not isinstance(self.vocab, dict) or sorted(self.vocab.values()) != list(range(len(self.vocab))):
            raise ValueError("Expected contiguous token IDs in official vocab.json")
        self.device, self.max_genes = device, int(cfg.get("max_input_genes", args["max_seq_len"]))
        if not 2 <= self.max_genes <= args["max_seq_len"]:
            raise ValueError("max_input_genes must be between 2 and the pretrained max_seq_len")
        self.n_bins = int(args["n_bins"])
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self.model = FrozenScGPT(self.vocab, args)
        raw = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
        # Fused FlashAttention QKV is the same parameter layout used by PyTorch MHA.
        raw = {k.replace(".self_attn.Wqkv.weight", ".self_attn.in_proj_weight")
               .replace(".self_attn.Wqkv.bias", ".self_attn.in_proj_bias"): v for k, v in raw.items()}
        for name in ("encoder", "value_encoder", "transformer_encoder"):
            prefix = name + "."
            weights = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
            # Reject partial/random encoder loads; task decoders are deliberately unused.
            getattr(self.model, name).load_state_dict(weights, strict=True)
        self.model.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, counts, symbols):
        from .scgpt_encoder import bin_values
        counts = np.asarray(counts, dtype=np.float32)
        if counts.ndim != 2 or counts.shape[1] != len(symbols) or not np.isfinite(counts).all() or (counts < 0).any() or (counts.sum(1) <= 0).any():
            raise ValueError("Expected finite nonnegative control counts with positive libraries")
        src, _ = select_symbols(symbols, self.vocab)
        ids = np.array([self.vocab[symbols[i]] for i in src])
        values = np.log1p(counts * (10000. / counts.sum(1))[:, None])
        output = []
        for row in values:
            # Bin across the measured full panel before vocabulary filtering and
            # random truncation. Do not preferentially retain high-expression genes.
            binned = bin_values(row, self.n_bins, self.rng)[src]
            ix = np.flatnonzero(row[src] > 0)
            if len(ix) == 0:
                raise ValueError("A control cell has no expressed genes in scGPT vocabulary")
            if len(ix) > self.max_genes:
                ix = np.sort(self.rng.choice(ix, self.max_genes, replace=False))
            token = torch.tensor(ids[ix][None], device=self.device)
            value = torch.tensor(binned[ix][None], device=self.device, dtype=torch.float32)
            encoded = self.model._encode(token, value, torch.zeros_like(token, dtype=torch.bool))
            output.append(encoded.mean(1).cpu().numpy()[0])
        return np.asarray(output, dtype=np.float32)


class ScFoundationEncoder:
    def __init__(self, cfg, device):
        root = verify_source(cfg)
        verify_weights(cfg)
        sys.path.insert(0, str(root / "model"))
        upstream = importlib.import_module("load")
        # Official MMF checkpoint structure. Load only an explicitly SHA-verified
        # official checkpoint; these historical checkpoints include Python metadata.
        raw = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        converted = upstream.convertconfig(raw["cell"])
        self.config = converted["config"]
        self.config.setdefault("qv_dim", self.config.get("dim_head", 64))
        self.config.setdefault("ppi_edge", None)
        self.config["device"] = str(device)
        self.model = upstream.select_model(self.config)
        self.model.load_state_dict(converted["model_state_dict"], strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.vocab = pd.read_csv(root / "OS_scRNA_gene_index.19264.tsv", sep="\t")["gene_name"].tolist()
        if len(self.vocab) != 19264 or len(set(self.vocab)) != 19264:
            raise ValueError("Expected the official 19,264-gene vocabulary")
        self.device = device

    @torch.no_grad()
    def encode(self, counts, symbols):
        src, dst = select_symbols(symbols, self.vocab)
        result = []
        for row in counts:
            # Follow official singlecell/F/f1 preprocessing on the mapped input.
            mapped = np.zeros(19264, dtype=np.float32)
            mapped[dst] = row[src]
            total = mapped.sum()
            if total <= 1:
                raise ValueError("Insufficient counts in scFoundation vocabulary")
            values = np.concatenate([np.log1p(mapped / total * 10000), [np.log10(total)] * 2])
            ids = np.flatnonzero(values > 0)
            x = torch.tensor(values[ids][None, :, None], device=self.device, dtype=torch.float32)
            positions = torch.tensor(ids[None], device=self.device)
            x = self.model.token_emb(x, output_weight=0) + self.model.pos_emb(positions)
            x = self.model.encoder(x, torch.zeros_like(positions, dtype=torch.bool))
            # Single-cell unpadded input keeps both resolution tokens last.
            z = torch.cat([x[:, -1], x[:, -2], x[:, :-2].amax(1), x[:, :-2].mean(1)], dim=1)
            result.append(z.cpu().numpy()[0])
        return np.asarray(result, dtype=np.float32)


class StatePredictor:
    def __init__(self, cfg, device):
        root = verify_source(cfg)
        verify_weights(cfg)
        sys.path.insert(0, str(root / "src"))
        from state.tx.models.state_transition import StateTransitionPerturbationModel
        self.model = StateTransitionPerturbationModel.load_from_checkpoint(
            cfg["checkpoint"], map_location="cpu", strict=True, weights_only=False)
        self.model.to(device).eval().requires_grad_(False)
        if self.model.output_space != "gene" or self.model.batch_encoder is not None:
            raise ValueError("Adapter supports gene-space State checkpoints without batch encoder only")
        if cfg["expression_space"] != "log1p_full_library_10000":
            raise ValueError("State checkpoint preprocessing must be audited as log1p_full_library_10000")
        self.genes = pd.read_csv(cfg["genes"], dtype=str)["gene"].tolist()
        if len(set(self.genes)) != len(self.genes) or len(self.genes) != self.model.input_dim or len(self.genes) != self.model.output_dim:
            raise ValueError("State gene order file does not match checkpoint dimensions")
        self.perts = torch.load(cfg["perturbation_map"], map_location="cpu", weights_only=True)
        self.device = device

    @torch.no_grad()
    def predict(self, counts, gene_ids, perturbation, output_genes):
        lookup = {name: i for i, name in enumerate(gene_ids)}
        if set(self.genes) - set(lookup) or set(output_genes) - set(self.genes):
            raise ValueError("State input/output gene panel incomplete; prepare a common panel, never zero-fill predictions")
        if perturbation not in self.perts:
            raise ValueError(f"State checkpoint has no perturbation {perturbation}")
        x = np.log1p(counts[:, [lookup[g] for g in self.genes]] * (10000. / counts.sum(1))[:, None])
        emb = torch.as_tensor(self.perts[perturbation], dtype=torch.float32, device=self.device).flatten()
        if emb.numel() != self.model.pert_dim:
            raise ValueError("State perturbation encoding dimension mismatch")
        outputs = []
        for start in range(0, len(x), self.model.cell_sentence_len):
            basal = torch.tensor(x[start:start + self.model.cell_sentence_len], device=self.device, dtype=torch.float32)
            result = self.model({"ctrl_cell_emb": basal, "pert_emb": emb.repeat(len(basal), 1)}, padded=False)
            result = result[0] if isinstance(result, tuple) else result
            outputs.append(result.cpu().numpy())
        pred = np.concatenate(outputs).mean(0)
        mapping = {g: i for i, g in enumerate(self.genes)}
        return pred[[mapping[g] for g in output_genes]]
