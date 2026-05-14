from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion_model import GraphMessageBlock


class PropertyPredictor(nn.Module):
    def __init__(self, node_in_dim: int = 6, hidden_dim: int = 128, n_layers: int = 3) -> None:
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.pos_embed = nn.Linear(3, hidden_dim)
        self.blocks = nn.ModuleList([GraphMessageBlock(hidden_dim) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        h = self.node_embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h=h, pos=pos, edge_index=edge_index)
        g = h.mean(dim=0, keepdim=True)
        out = self.head(g)
        delta_g_h = 1.5 * torch.tanh(out[:, 0])
        thermo = torch.sigmoid(out[:, 1])
        kinetic = torch.sigmoid(out[:, 2])
        synthesis = torch.sigmoid(out[:, 3])
        return {
            "delta_g_h": delta_g_h.squeeze(0),
            "thermo": thermo.squeeze(0),
            "kinetic": kinetic.squeeze(0),
            "synthesis": synthesis.squeeze(0),
        }


def multitask_loss(
    diffusion_loss: torch.Tensor,
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    lambda_her: float = 0.6,
    lambda_thermo: float = 0.4,
    lambda_kinetic: float = 0.4,
    lambda_synthesis: float = 0.4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    her_regression = F.l1_loss(pred["delta_g_h"], target["delta_g_h"])
    her_objective = torch.abs(pred["delta_g_h"])
    her_loss = her_regression + 0.5 * her_objective

    thermo_regression = F.mse_loss(pred["thermo"], target["thermo"])
    kinetic_regression = F.mse_loss(pred["kinetic"], target["kinetic"])
    synthesis_regression = F.mse_loss(pred["synthesis"], target["synthesis"])

    thermo_obj = 1.0 - pred["thermo"]
    kinetic_obj = 1.0 - pred["kinetic"]
    synthesis_obj = 1.0 - pred["synthesis"]

    thermo_loss = thermo_regression + 0.2 * thermo_obj
    kinetic_loss = kinetic_regression + 0.2 * kinetic_obj
    synthesis_loss = synthesis_regression + 0.2 * synthesis_obj

    total = (
        diffusion_loss
        + lambda_her * her_loss
        + lambda_thermo * thermo_loss
        + lambda_kinetic * kinetic_loss
        + lambda_synthesis * synthesis_loss
    )
    details = {
        "diffusion": diffusion_loss.detach(),
        "her": her_loss.detach(),
        "thermo": thermo_loss.detach(),
        "kinetic": kinetic_loss.detach(),
        "synthesis": synthesis_loss.detach(),
    }
    return total, details
