"""State SE inference adapted from Arc Institute State.

Adapted from commit 9bbfe78a434a55205e4de834e1ea99f85f7a3add.
Original: https://github.com/ArcInstitute/state
License: CC BY-NC-SA 4.0, see licenses/STATE_CODE_LICENSE.txt.
Changes: frozen inference only, PyTorch SDPA dropout disabled at inference,
strict checkpoint loading, explicit raw-count input and duplicate aggregation.
"""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .foundation_encoders import verify_files, mapped_counts


class StateLayer(nn.Module):
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.heads = heads
        self.qkv_proj, self.out_proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.linear1, self.linear2 = nn.Linear(width, hidden), nn.Linear(hidden, width)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = [a.reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
                   for a in self.qkv_proj(x).chunk(3, -1)]
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
        x = self.norm1(x + self.out_proj(a.transpose(1, 2).reshape(b, n, d)))
        return self.norm2(x + self.linear2(F.gelu(self.linear1(x))))


class StateSkip(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.intermediate_dense, self.dense = nn.Linear(d, d * 2), nn.Linear(d * 2, d)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.layer_norm(x + self.dense(F.relu(self.intermediate_dense(x))))


class StateBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, t = cfg["d_model"], cfg["token_dim"]
        self.cls_token, self.dataset_token = nn.Parameter(torch.zeros(1, t)), nn.Parameter(torch.zeros(1, t))
        self.encoder = nn.Sequential(nn.Linear(t, d), nn.LayerNorm(d), nn.SiLU())
        self.transformer_encoder = nn.Module()
        self.transformer_encoder.layers = nn.ModuleList([StateLayer(d, cfg["nhead"], cfg["d_hid"]) for _ in range(cfg["nlayers"])])
        self.decoder = nn.Sequential(StateSkip(d), nn.Linear(d, cfg["output_dim"]))
        self.bin_encoder = nn.Embedding(10, d)
        self.count_encoder = nn.Sequential(nn.Linear(1, 512), nn.LeakyReLU(), nn.Linear(512, 10))
        self.width = d

    def forward(self, protein, counts):
        x = F.normalize(protein, dim=-1)
        x[:, 0] = self.cls_token
        x = torch.cat([x, self.dataset_token[None].expand(x.shape[0], -1, -1)], 1)
        x = self.encoder(x) * math.sqrt(self.width)
        c = F.softmax(self.count_encoder(counts[..., None]), -1) @ self.bin_encoder.weight
        x = x + torch.cat([c, torch.zeros_like(c[:, :1])], 1)
        for layer in self.transformer_encoder.layers:
            x = layer(x)
        return F.normalize(self.decoder(x[:, 0]), dim=-1)


class StateEncoder:
    def __init__(self, cfg, device):
        import yaml
        from safetensors.torch import load_file
        verify_files(cfg, ["config", "proteins"])
        config = yaml.safe_load(Path(cfg["config"]).read_text())
        if not config["model"]["counts"] or not config["model"]["dataset_correction"]:
            raise ValueError("This adapter requires the pinned count-aware State SE-100M")
        architecture = {**config["model"], "d_model": config["model"]["emsize"],
                        "token_dim": config["embeddings"][config["embeddings"]["current"]]["size"]}
        self.model = StateBackbone(architecture)
        weights = load_file(cfg["checkpoint"])
        # Safetensors may retain either alias of the shared input projection.
        for key in self.model.state_dict():
            if key.startswith("encoder.") and key not in weights:
                weights[key] = weights["gene_embedding_layer." + key[len("encoder."):]]
        self.model.load_state_dict({k: weights[k] for k in self.model.state_dict()}, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.proteins = torch.load(cfg["proteins"], map_location="cpu", weights_only=True)
        self.vocab = list(self.proteins)
        self.pad_length, self.cls_index = config["dataset"]["pad_length"], config["dataset"]["cls_token_idx"]
        self.device = device
        self.rng = np.random.default_rng(cfg.get("seed", 0))

    @torch.no_grad()
    def encode(self, counts, symbols):
        mapped, dst, self.last_input_audit = mapped_counts(counts, symbols, self.vocab)
        if len(dst) <= self.cls_index:
            raise ValueError("Too few State vocabulary genes")
        proteins = torch.stack([self.proteins[self.vocab[i]].float().flatten() for i in dst])
        result = []
        for row in mapped:
            log = np.log1p(row)
            if log.sum() <= 0:
                raise ValueError("No expressed State vocabulary genes")
            order = self.rng.permutation(len(row))
            rank = order[np.argsort(-log[order], kind="stable")]
            n = self.pad_length - 1
            if len(rank) < n:
                zero = rank[log[rank] == 0]
                if not len(zero):
                    raise ValueError("State requires enough vocabulary genes or unexpressed genes for padding")
                positive = rank[log[rank] > 0]
                rank = np.r_[positive, self.rng.choice(zero, n - len(positive), replace=True)]
            ids = np.r_[self.cls_index, rank[:n]]
            # Follow the pinned upstream collater, including its CLS count convention.
            values = torch.tensor((100 * log / log.sum())[ids][None], dtype=torch.float32, device=self.device)
            result.append(self.model(proteins[ids][None].to(self.device), values).cpu().numpy()[0])
        return np.asarray(result, dtype=np.float32)
