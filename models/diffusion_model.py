import math
from typing import Tuple

import torch
import torch.nn as nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.dim() == 0:
        t = t.unsqueeze(0)
    t = t.float().view(-1, 1)
    half = dim // 2
    freq = torch.exp(
        torch.arange(half, device=t.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    phase = t * freq
    emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class GraphMessageBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index
        rel = pos[src] - pos[dst]
        m_in = torch.cat([h[src], h[dst], rel], dim=-1)
        msg = self.msg_mlp(m_in)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        deg = torch.bincount(dst, minlength=h.shape[0]).clamp(min=1).unsqueeze(-1)
        agg = agg / deg

        out = self.upd_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + out)


class DiffusionGNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 6,
        hidden_dim: int = 128,
        n_layers: int = 4,
        cond_dim: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.pos_embed = nn.Linear(3, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([GraphMessageBlock(hidden_dim) for _ in range(n_layers)])
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def encode_context(
        self, x: torch.Tensor, pos: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        n = x.shape[0]
        h = self.node_embed(x) + self.pos_embed(pos)
        t_emb = sinusoidal_time_embedding(t=t.view(1), dim=self.hidden_dim)
        t_emb = self.time_proj(t_emb).view(1, -1).expand(n, -1)
        cond = cond.view(1, -1).float()
        c_emb = self.cond_proj(cond).expand(n, -1)
        return h + t_emb + c_emb

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode_context(x=x, pos=pos, t=t, cond=cond)
        for block in self.blocks:
            h = block(h=h, pos=pos, edge_index=edge_index)
        return self.noise_head(h)


def diffusion_step(
    model: DiffusionGNN,
    x: torch.Tensor,
    clean_pos: torch.Tensor,
    edge_index: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma = 0.05 + 0.35 * t.clamp(0.0, 1.0)
    noise = torch.randn_like(clean_pos)
    noisy_pos = clean_pos + sigma * noise
    pred_noise = model(x=x, pos=noisy_pos, edge_index=edge_index, t=t, cond=cond)
    return noisy_pos, noise, pred_noise
