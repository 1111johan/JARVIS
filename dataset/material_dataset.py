import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from pymatgen.core import Element, Structure
from torch.utils.data import Dataset


@dataclass
class GraphData:
    x: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    atomic_numbers: torch.Tensor
    lattice: torch.Tensor
    cif_path: str
    formula: str
    structure: Structure

    def to(self, device: torch.device) -> "GraphData":
        return GraphData(
            x=self.x.to(device),
            pos=self.pos.to(device),
            edge_index=self.edge_index.to(device),
            atomic_numbers=self.atomic_numbers.to(device),
            lattice=self.lattice.to(device),
            cif_path=self.cif_path,
            formula=self.formula,
            structure=self.structure,
        )


def _build_node_features(atomic_numbers: torch.Tensor) -> torch.Tensor:
    feats: List[List[float]] = []
    for z in atomic_numbers.tolist():
        elem = Element.from_Z(int(z))
        en = float(elem.X) if elem.X is not None else 0.0
        row = float(elem.row or 0)
        group = float(elem.group or 0)
        feats.append(
            [
                z / 100.0,
                en / 4.0,
                row / 7.0,
                group / 18.0,
                1.0 if elem.is_transition_metal else 0.0,
                1.0 if elem.symbol in {"S", "Se", "Te", "O"} else 0.0,
            ]
        )
    return torch.tensor(feats, dtype=torch.float32)


def _edge_index_from_structure(
    structure: Structure, cutoff: float = 5.0, max_neighbors: int = 16
) -> torch.Tensor:
    src, dst, _, dist = structure.get_neighbor_list(r=cutoff)
    n_atoms = len(structure)
    grouped: Dict[int, List[tuple]] = {i: [] for i in range(n_atoms)}
    for s, d, dis in zip(src, dst, dist):
        si = int(s)
        di = int(d)
        if si == di:
            continue
        grouped[si].append((di, float(dis)))

    edges: List[List[int]] = []
    for si in range(n_atoms):
        neighbors = sorted(grouped[si], key=lambda x: x[1])[:max_neighbors]
        for di, _ in neighbors:
            edges.append([si, di])

    if not edges and n_atoms == 1:
        edges = [[0, 0]]

    if not edges:
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    edges.append([i, j])

    if not edges:
        edges = [[0, 0]]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        edge_index = edge_index.view(-1, 2).t().contiguous()
    return edge_index


def structure_to_graph(
    structure: Structure, cif_path: str = "", cutoff: float = 5.0, max_neighbors: int = 16
) -> GraphData:
    atomic_numbers = torch.tensor(
        [site.specie.Z for site in structure], dtype=torch.long
    )
    x = _build_node_features(atomic_numbers)
    pos = torch.tensor(structure.frac_coords, dtype=torch.float32)
    edge_index = _edge_index_from_structure(
        structure=structure, cutoff=cutoff, max_neighbors=max_neighbors
    )
    lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    formula = structure.composition.reduced_formula
    return GraphData(
        x=x,
        pos=pos,
        edge_index=edge_index,
        atomic_numbers=atomic_numbers,
        lattice=lattice,
        cif_path=cif_path,
        formula=formula,
        structure=structure,
    )


class MaterialDataset(Dataset):
    def __init__(
        self,
        cif_dir: str = "data/cif",
        cutoff: float = 5.0,
        max_neighbors: int = 16,
        max_samples: Optional[int] = None,
        seed: int = 42,
        use_cache: bool = True,
    ) -> None:
        self.cif_dir = cif_dir
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.use_cache = use_cache
        self._cache: Dict[int, GraphData] = {}

        cif_paths = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))
        if max_samples is not None and len(cif_paths) > max_samples:
            rng = random.Random(seed)
            rng.shuffle(cif_paths)
            cif_paths = sorted(cif_paths[:max_samples])
        self.cif_paths = cif_paths

        if not self.cif_paths:
            raise FileNotFoundError(
                f"No CIF files found in {cif_dir}. Run download_jarvis_2d.py first."
            )

        sample = self._load_graph(0)
        self.node_feature_dim = sample.x.shape[-1]

    def _load_graph(self, idx: int) -> GraphData:
        path = self.cif_paths[idx]
        structure = Structure.from_file(path)
        return structure_to_graph(
            structure=structure,
            cif_path=path,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )

    def __len__(self) -> int:
        return len(self.cif_paths)

    def __getitem__(self, idx: int) -> GraphData:
        if self.use_cache and idx in self._cache:
            return self._cache[idx]
        graph = self._load_graph(idx)
        if self.use_cache:
            self._cache[idx] = graph
        return graph
