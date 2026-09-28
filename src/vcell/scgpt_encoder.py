"""Inference-only scGPT encoder, compatible with the whole-human checkpoint.

Adapted from bowang-lab/scGPT (MIT), commit
cebd6fae655b9c585a4807daa3ac31bb764f06b4. See THIRD_PARTY_NOTICES.md.
Only the gene/value encoders and post-norm transformer are implemented; this
module is not scGPT's perturbation decoder or a reproduction of its benchmarks.
"""
import numpy as np
import torch
from torch import nn


class GeneEncoder(nn.Module):
    def __init__(self, tokens, width, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(tokens, width, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.enc_norm(self.embedding(x))


class ContinuousValueEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear1 = nn.Linear(1, width)
        self.linear2 = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(self.linear2(torch.relu(self.linear1(x.unsqueeze(-1).clamp(max=512)))))


class FrozenScGPT(nn.Module):
    def __init__(self, vocab, args):
        super().__init__()
        if args.get("pre_norm", False) or args.get("input_emb_style") != "continuous":
            raise ValueError("Only post-norm scGPT with continuous value embeddings is supported")
        if args.get("input_style") != "binned" or not args.get("no_cls", False):
            raise ValueError("This adapter expects binned inputs without a CLS token")
        width = args["embsize"]
        self.encoder = GeneEncoder(len(vocab), width, vocab[args["pad_token"]])
        self.value_encoder = ContinuousValueEncoder(width)
        layer = nn.TransformerEncoderLayer(width, args["nheads"], args["d_hid"],
                                           dropout=0., activation="relu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, args["nlayers"], enable_nested_tensor=False)

    def _encode(self, tokens, values, padding_mask):
        x = self.encoder(tokens) + self.value_encoder(values)
        return self.transformer_encoder(x, src_key_padding_mask=padding_mask)


def bin_values(row, n_bins, rng):
    """Upstream quantile/tie-spreading semantics with an explicit seeded RNG."""
    positive = np.flatnonzero(row > 0)
    result = np.zeros_like(row, dtype=np.float32)
    if not len(positive):
        return result
    values = row[positive]
    bins = np.quantile(values, np.linspace(0, 1, n_bins - 1))
    left = np.digitize(values, bins)
    right = np.digitize(values, bins, right=True)
    result[positive] = np.ceil(rng.random(len(values)) * (right - left) + left)
    return result
