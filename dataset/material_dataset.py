import csv
import glob
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pymatgen.core import Element, Structure
from torch.utils.data import Dataset

from utils.geo_utils import proxy_targets


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
    jid: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    target: Dict[str, float] = field(default_factory=dict)

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
            jid=self.jid,
            metadata=self.metadata,
            target=self.target,
        )


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    if s in {"", "na", "none", "nan", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


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


def _targets_from_structure_and_metadata(
    structure: Structure, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    proxy = proxy_targets(structure)
    meta = metadata or {}

    fe = _safe_float(meta.get("formation_energy_peratom"))
    ehull = _safe_float(meta.get("ehull"))
    exf = _safe_float(meta.get("exfoliation_energy"))
    synth_data_label = _safe_float(meta.get("synthesis_data_label"))
    exp_formula_label = _safe_float(meta.get("exp_formula_label"))
    has_icsd = _safe_float(meta.get("has_icsd"))

    thermo_terms: List[float] = []
    if fe is not None:
        thermo_terms.append(float(np.clip(_sigmoid(-(fe + 0.10) / 0.25), 0.0, 1.0)))
    if ehull is not None:
        thermo_terms.append(float(np.clip(np.exp(-max(ehull, 0.0) / 0.05), 0.0, 1.0)))
    if exf is not None:
        thermo_terms.append(float(np.clip(np.exp(-max(exf - 80.0, 0.0) / 130.0), 0.0, 1.0)))
    thermo_data = float(np.mean(thermo_terms)) if thermo_terms else proxy["thermo"]
    thermo_target = float(np.clip(0.65 * thermo_data + 0.35 * proxy["thermo"], 0.0, 1.0))

    synth_terms: List[float] = []
    if synth_data_label is not None:
        synth_terms.append(0.90 if synth_data_label >= 0.5 else 0.10)
    elif exp_formula_label is not None or has_icsd is not None:
        exp_flag = int((exp_formula_label or 0.0) >= 0.5 or (has_icsd or 0.0) >= 0.5)
        synth_terms.append(0.90 if exp_flag else 0.10)
    if exf is not None:
        synth_terms.append(float(np.clip(np.exp(-max(exf - 100.0, 0.0) / 120.0), 0.0, 1.0)))
    if fe is not None:
        synth_terms.append(float(np.clip(_sigmoid(-(fe + 0.05) / 0.30), 0.0, 1.0)))
    synth_data = float(np.mean(synth_terms)) if synth_terms else proxy["synthesis"]
    synth_target = float(np.clip(0.70 * synth_data + 0.30 * proxy["synthesis"], 0.0, 1.0))

    kinetic_target = proxy["kinetic"]
    if exf is not None:
        exf_soft = float(np.clip(np.exp(-max(exf - 120.0, 0.0) / 160.0), 0.0, 1.0))
        kinetic_target = float(np.clip(0.80 * kinetic_target + 0.20 * exf_soft, 0.0, 1.0))

    return {
        "delta_g_h": proxy["delta_g_h"],
        "thermo": thermo_target,
        "kinetic": kinetic_target,
        "synthesis": synth_target,
    }


def _parse_jid_from_cif_path(path: str) -> str:
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def _load_metadata_map(csv_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if csv_path is None or not os.path.exists(csv_path):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jid = str(row.get("jid", "")).strip()
            if not jid:
                continue
            normalized = dict(row)
            for k in [
                "formation_energy_peratom",
                "ehull",
                "exfoliation_energy",
                "optb88vdw_bandgap",
                "hse_gap",
                "has_icsd",
                "exp_formula_label",
                "synthesis_data_label",
            ]:
                normalized[k] = _safe_float(row.get(k))
            out[jid] = normalized
    return out


def structure_to_graph(
    structure: Structure,
    cif_path: str = "",
    cutoff: float = 5.0,
    max_neighbors: int = 16,
    jid: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    target: Optional[Dict[str, float]] = None,
) -> GraphData:
    atomic_numbers = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    x = _build_node_features(atomic_numbers)
    pos = torch.tensor(structure.frac_coords, dtype=torch.float32)
    edge_index = _edge_index_from_structure(
        structure=structure, cutoff=cutoff, max_neighbors=max_neighbors
    )
    lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    formula = structure.composition.reduced_formula
    meta = metadata or {}
    tgt = target or _targets_from_structure_and_metadata(structure=structure, metadata=meta)
    return GraphData(
        x=x,
        pos=pos,
        edge_index=edge_index,
        atomic_numbers=atomic_numbers,
        lattice=lattice,
        cif_path=cif_path,
        formula=formula,
        structure=structure,
        jid=jid,
        metadata=meta,
        target=tgt,
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
        metadata_csv: Optional[str] = None,
    ) -> None:
        self.cif_dir = cif_dir
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.use_cache = use_cache
        self._cache: Dict[int, GraphData] = {}

        if metadata_csv is None:
            metadata_csv = os.path.join(os.path.dirname(cif_dir), "jarvis_dft_2d_metadata.csv")
        self.metadata_map = _load_metadata_map(metadata_csv)

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
        jid = _parse_jid_from_cif_path(path)
        metadata = self.metadata_map.get(jid, {})
        return structure_to_graph(
            structure=structure,
            cif_path=path,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            jid=jid,
            metadata=metadata,
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
