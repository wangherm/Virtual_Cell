"""Four response architectures, sized by configuration for either role."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 2, dim))

    def forward(self, x):
        return x + self.net(x)


class BaseResponse(nn.Module):
    def add_heads(self, hidden, genes, projection):
        self.output = nn.Linear(hidden, genes)
        self.project = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, projection))

    def heads(self, z):
        return self.output(z), F.normalize(self.project(z), dim=-1)


class ResidualResponse(BaseResponse):
    """Residual MLP with multiplicative context-target interaction."""
    def __init__(self, genes, perts, hidden=256, projection=32, dropout=.1):
        super().__init__()
        self.context = nn.Linear(genes, hidden)
        self.pert = nn.Embedding(perts, hidden)
        self.fuse = nn.Linear(hidden * 3, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(3)])
        self.add_heads(hidden, genes, projection)

    def forward(self, x, p):
        c, t = F.gelu(self.context(x)), self.pert(p)
        z = self.blocks(F.gelu(self.fuse(torch.cat([c, t, c * t], dim=-1))))
        return self.heads(z)


class ModuleResponse(BaseResponse):
    """Learned latent module tokens + target token; not a biological GRN."""
    def __init__(self, genes, perts, hidden=256, projection=32, dropout=.1):
        super().__init__()
        if hidden % 4:
            raise ValueError("Module hidden size must be divisible by 4")
        self.modules_count = 8
        self.context = nn.Linear(genes, hidden * self.modules_count)
        self.pert = nn.Embedding(perts, hidden)
        self.positions = nn.Parameter(torch.randn(1, self.modules_count + 1, hidden) * .02)
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.add_heads(hidden, genes, projection)

    def forward(self, x, p):
        context = self.context(x).reshape(len(x), self.modules_count, -1)
        tokens = torch.cat([self.pert(p)[:, None], context], dim=1) + self.positions
        z = self.norm(self.encoder(tokens)[:, 0])
        return self.heads(z)


class MLPResponse(BaseResponse):
    """Concatenation MLP."""
    def __init__(self, genes, perts, hidden=64, projection=32, dropout=.1):
        super().__init__()
        self.context = nn.Linear(genes, hidden)
        self.pert = nn.Embedding(perts, hidden)
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.add_heads(hidden, genes, projection)

    def forward(self, x, p):
        return self.heads(self.fuse(torch.cat([F.gelu(self.context(x)), self.pert(p)], -1)))


class BilinearResponse(BaseResponse):
    """Low-rank context-target interaction plus target main effect."""
    def __init__(self, genes, perts, hidden=64, projection=32, dropout=.1):
        super().__init__()
        self.context = nn.Linear(genes, hidden)
        self.pert = nn.Embedding(perts, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.add_heads(hidden, genes, projection)

    def forward(self, x, p):
        t = self.pert(p)
        z = self.drop(self.norm(t + torch.tanh(self.context(x)) * t))
        return self.heads(z)


MODEL_TYPES = {"residual": ResidualResponse, "module": ModuleResponse,
               "mlp": MLPResponse, "bilinear": BilinearResponse}


def make_model(spec, genes, perts, projection):
    kwargs = {"genes": genes, "perts": perts,
              "hidden": int(spec["hidden"]), "projection": int(projection),
              "dropout": float(spec["dropout"])}
    return MODEL_TYPES[spec["architecture"]](**kwargs), kwargs
