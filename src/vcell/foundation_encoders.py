"""Frozen, checksum-verified encoder paths for the exploratory four-teacher run.

Architecture/preprocessing sources and licenses are recorded in THIRD_PARTY_NOTICES.
Only inference components are loaded; every parameter on the encoding path is
loaded strictly. Task decoders are replaced by the separately trained VCell head.
"""
from pathlib import Path
import json
import math
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .external_models import verify_weights, select_symbols
from .gene_mapping import collapse_symbol_counts
from .utils import file_sha256


def verify_files(cfg, keys):
    verify_weights(cfg)
    for key in keys:
        if file_sha256(cfg[key]) != cfg[key + "_sha256"]:
            raise ValueError(f"Teacher {key} checksum mismatch")


def mapped_counts(counts, symbols, vocabulary):
    values, symbols, audit = collapse_symbol_counts(counts, symbols)
    src, dst = select_symbols(symbols, vocabulary)
    audit["vocabulary_overlap_symbols"] = len(src)
    return values[:, src], dst, audit


class SoftBins(nn.Module):
    def __init__(self, dim, bins, alpha, mask, pad):
        super().__init__()
        self.mlp, self.mlp2 = nn.Linear(1, bins), nn.Linear(bins, bins)
        self.emb, self.emb_mask, self.emb_pad = nn.Embedding(bins, dim), nn.Embedding(1, dim), nn.Embedding(1, dim)
        self.alpha, self.mask, self.pad = alpha, mask, pad

    def forward(self, values):
        h = F.leaky_relu(self.mlp(values), .1)
        z = F.softmax(self.alpha * h + self.mlp2(h), -1) @ self.emb.weight
        z = torch.where(values == self.mask, self.emb_mask.weight, z)
        return torch.where(values == self.pad, self.emb_pad.weight, z)


class FoundationTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["hidden_dim"]
        self.transformer_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d, cfg["heads"], d * 4, batch_first=True,
                                       norm_first=cfg["norm_first"])
            for _ in range(cfg["depth"])])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        for layer in self.transformer_encoder:
            x = layer(x)
        return self.norm(x)


class FoundationBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["encoder"]["hidden_dim"]
        if cfg["encoder"]["module_type"] != "transformer":
            raise ValueError("Expected official transformer cell encoder")
        self.token_emb = SoftBins(d, cfg["bin_num"], cfg["bin_alpha"], cfg["mask_token_id"], cfg["pad_token_id"])
        self.pos_emb = nn.Embedding(cfg["seq_len"] + 1, d)
        self.encoder = FoundationTransformer(cfg["encoder"])


class FoundationEncoder:
    def __init__(self, cfg, device):
        verify_files(cfg, ["vocab"])
        raw = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)["cell"]
        config = raw["config"]["model_config"]["mae_autobin"]
        self.model = FoundationBackbone(config)
        for name in ("token_emb", "pos_emb", "encoder"):
            prefix = "model." + name + "."
            getattr(self.model, name).load_state_dict(
                {k[len(prefix):]: v for k, v in raw["state_dict"].items() if k.startswith(prefix)}, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.vocab = pd.read_csv(cfg["vocab"], sep="\t")["gene_name"].tolist()
        if len(self.vocab) != 19264 or len(set(self.vocab)) != 19264:
            raise ValueError("Expected official 19,264-symbol vocabulary")
        self.device = device

    @torch.no_grad()
    def encode(self, counts, symbols):
        mapped, dst, self.last_input_audit = mapped_counts(counts, symbols, self.vocab)
        result = []
        for row in mapped:
            total = row.sum()
            if total <= 1 or np.count_nonzero(row) < 2:
                raise ValueError("Insufficient expressed scFoundation vocabulary genes")
            full = np.zeros(19264, dtype=np.float32)
            full[dst] = row
            values = np.r_[np.log1p(full / total * 10000), [np.log10(total)] * 2]
            ids = np.flatnonzero(values > 0)
            x = torch.tensor(values[ids][None, :, None], device=self.device, dtype=torch.float32)
            positions = torch.tensor(ids[None], device=self.device)
            x = self.model.encoder(self.model.token_emb(x) + self.model.pos_emb(positions))
            result.append(torch.cat([x[:, -1], x[:, -2], x[:, :-2].amax(1), x[:, :-2].mean(1)], 1).cpu().numpy()[0])
        return np.asarray(result, dtype=np.float32)


class PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name == "scalar":
            return np.core.multiarray.scalar
        if module == "numpy" and name == "dtype":
            return np.dtype
        raise ValueError("Geneformer dictionary must contain only plain Python values")


def plain_dictionary(path):
    with open(path, "rb") as f:
        value = PlainUnpickler(f).load()
    if not isinstance(value, dict):
        raise ValueError("Expected Geneformer dictionary")
    return value


class GeneformerEncoder:
    def __init__(self, cfg, device):
        from transformers import BertConfig, BertForMaskedLM
        from safetensors.torch import load_file
        verify_files(cfg, ["config", "vocab", "medians", "mapping"])
        self.vocab, self.medians = plain_dictionary(cfg["vocab"]), plain_dictionary(cfg["medians"])
        self.mapping = plain_dictionary(cfg["mapping"])
        self.model = BertForMaskedLM(BertConfig.from_json_file(cfg["config"]))
        weights = load_file(cfg["checkpoint"])
        positions = weights.pop("bert.embeddings.position_ids", None)
        if positions is not None and not torch.equal(positions, self.model.bert.embeddings.position_ids):
            raise ValueError("Geneformer saved position buffer differs from installed BERT")
        # HF safe serialization omits tied decoder tensors; restore only exact ties.
        weights.setdefault("cls.predictions.decoder.weight", weights["bert.embeddings.word_embeddings.weight"])
        weights.setdefault("cls.predictions.decoder.bias", weights["cls.predictions.bias"])
        self.model.load_state_dict(weights, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode(self, counts, gene_ids):
        # Collapse only author-declared Ensembl aliases; expression normalization
        # retains the original measured full-library denominator.
        full_total = np.asarray(counts, dtype=np.float64).sum(1)
        mapped_ids = [self.mapping.get(g, g) for g in gene_ids]
        values, ids, audit = collapse_symbol_counts(counts, mapped_ids)
        src = [i for i, g in enumerate(ids) if g in self.vocab and g in self.medians]
        if len(src) < 2 or (full_total <= 0).any():
            raise ValueError("Insufficient Geneformer Ensembl overlap/counts")
        audit["vocabulary_overlap_symbols"] = len(src)
        self.last_input_audit = audit
        tokens = np.array([self.vocab[ids[i]] for i in src])
        medians = np.array([self.medians[ids[i]] for i in src], dtype=np.float64)
        if not np.isfinite(medians).all() or (medians <= 0).any():
            raise ValueError("Invalid Geneformer nonzero gene medians")
        ranked = values[:, src] / full_total[:, None] * 10000 / medians
        result = []
        for row in ranked:
            positive = np.flatnonzero(row > 0)
            ix = positive[np.argsort(-row[positive], kind="stable")][:2048]
            if not len(ix):
                raise ValueError("No expressed Geneformer vocabulary genes")
            token = torch.tensor(tokens[ix][None], dtype=torch.long, device=self.device)
            hidden = self.model.bert(input_ids=token, attention_mask=torch.ones_like(token)).last_hidden_state
            result.append(hidden.mean(1).cpu().numpy()[0])
        return np.asarray(result, dtype=np.float32)
