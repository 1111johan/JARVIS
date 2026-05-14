import random
from typing import List, Optional

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from dataset.material_dataset import GraphData, MaterialDataset
from models.diffusion_model import DiffusionGNN
from utils.geo_utils import clamp_frac_coords, is_reasonable_structure


class StructureGenerator:
    def __init__(self, model: DiffusionGNN, n_steps: int = 60) -> None:
        self.model = model
        self.n_steps = n_steps

    def generate_from_seed(
        self,
        data: GraphData,
        device: torch.device,
        cond: Optional[torch.Tensor] = None,
    ) -> GraphData:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        n_atoms = data.pos.shape[0]
        pos = torch.rand((n_atoms, 3), device=device)
        pos[:, 2] = 0.5 + 0.02 * torch.randn(n_atoms, device=device)

        if cond is None:
            cond = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32, device=device)

        self.model.eval()
        with torch.no_grad():
            for step in reversed(range(self.n_steps)):
                t = torch.tensor([step / max(self.n_steps - 1, 1)], device=device)
                pred_noise = self.model(x=x, pos=pos, edge_index=edge_index, t=t, cond=cond)
                lr = 1.0 / self.n_steps
                eta = 0.02 * (step / max(self.n_steps - 1, 1))
                pos = pos - lr * pred_noise + eta * torch.randn_like(pos)
                pos[:, 0:2] = torch.remainder(pos[:, 0:2], 1.0)
                pos[:, 2] = torch.clamp(0.5 + 0.1 * (pos[:, 2] - 0.5), 0.45, 0.55)

        return GraphData(
            x=data.x,
            pos=pos.detach().cpu(),
            edge_index=data.edge_index,
            atomic_numbers=data.atomic_numbers,
            lattice=data.lattice,
            cif_path=data.cif_path,
            formula=data.formula,
            structure=data.structure,
        )

    def data_to_structure(
        self, generated_data: GraphData, reference_structure: Structure
    ) -> Structure:
        coords = generated_data.pos.detach().cpu().numpy()
        coords = clamp_frac_coords(coords)
        species = [site.specie for site in reference_structure]
        lat = reference_structure.lattice.matrix.copy()
        c_norm = np.linalg.norm(lat[2])
        if c_norm < 15.0:
            lat[2] = lat[2] * (15.0 / max(c_norm, 1e-6))
        lattice = Lattice(lat)
        new_structure = Structure(
            lattice=lattice, species=species, coords=coords, coords_are_cartesian=False
        )
        return new_structure

    def generate_multiple(
        self, dataset: MaterialDataset, n_generate: int, device: torch.device
    ) -> List[Structure]:
        generated: List[Structure] = []
        attempts = 0
        max_attempts = max(n_generate * 4, 80)
        rng = random.Random(42)

        while len(generated) < n_generate and attempts < max_attempts:
            idx = rng.randint(0, len(dataset) - 1)
            seed_data = dataset[idx]
            gen_data = self.generate_from_seed(data=seed_data, device=device)
            structure = self.data_to_structure(gen_data, seed_data.structure)
            if is_reasonable_structure(structure):
                generated.append(structure)
            attempts += 1

        if len(generated) < n_generate:
            for i in range(len(generated), n_generate):
                generated.append(dataset[i % len(dataset)].structure.copy())

        return generated
