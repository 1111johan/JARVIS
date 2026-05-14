from typing import Dict, Iterable, Tuple

import numpy as np
from pymatgen.core import Structure


HER_TM = {"Mo", "W", "V", "Nb", "Ta", "Fe", "Co", "Ni"}
CHALCOGEN = {"S", "Se", "Te", "O"}
COMMON_2D = {
    "C",
    "B",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Se",
    "Te",
    "Mo",
    "W",
    "V",
    "Nb",
    "Ta",
    "Ti",
    "Zr",
    "Hf",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "In",
    "Ge",
    "Sn",
    "Bi",
}


def _min_distance(structure: Structure) -> float:
    d = structure.distance_matrix
    d = d + np.eye(d.shape[0]) * 1e6
    return float(np.min(d))


def _symbols(structure: Structure) -> Iterable[str]:
    return [site.specie.symbol for site in structure]


def estimate_delta_g_h(structure: Structure) -> float:
    symbols = set(_symbols(structure))
    min_dist = _min_distance(structure)

    # Baseline offset around mildly positive adsorption energy.
    delta = 0.35

    if len(symbols & HER_TM) > 0:
        delta -= 0.20
    if len(symbols & CHALCOGEN) > 0:
        delta -= 0.18
    if len(symbols & {"Fe", "Co", "Ni"}) > 0:
        delta -= 0.08

    # Reward TMDC-like element pairing.
    if len(symbols & {"Mo", "W", "V", "Nb", "Ta"}) > 0 and len(symbols & CHALCOGEN) > 0:
        delta -= 0.12

    # Geometric penalties.
    if min_dist < 1.0:
        delta += 0.45
    elif min_dist < 1.3:
        delta += 0.20
    elif min_dist > 3.4:
        delta += 0.10

    return float(np.clip(delta, -1.5, 1.5))


def estimate_thermodynamic_stability(structure: Structure) -> float:
    n = len(structure)
    vol_per_atom = structure.volume / max(n, 1)
    min_dist = _min_distance(structure)
    unique_elems = len(set(_symbols(structure)))

    score = 0.4
    if 8.0 <= vol_per_atom <= 45.0:
        score += 0.2
    if min_dist >= 1.2:
        score += 0.2
    if unique_elems <= 4:
        score += 0.1
    if structure.lattice.c >= 15.0:
        score += 0.1
    if min_dist < 1.0:
        score -= 0.4

    return float(np.clip(score, 0.0, 1.0))


def estimate_kinetic_stability(structure: Structure) -> float:
    d = structure.distance_matrix
    d = d + np.eye(d.shape[0]) * 1e6
    min_dist = float(np.min(d))
    avg_nn = float(np.mean(np.sort(d, axis=1)[:, :3]))

    score = 0.35
    if min_dist > 1.1:
        score += 0.25
    if 1.8 <= avg_nn <= 3.2:
        score += 0.25
    if structure.lattice.c >= 15.0:
        score += 0.1
    if min_dist < 0.95:
        score -= 0.45

    return float(np.clip(score, 0.0, 1.0))


def estimate_synthesizability(structure: Structure) -> float:
    syms = list(_symbols(structure))
    unique = set(syms)
    common_ratio = sum(1 for s in unique if s in COMMON_2D) / max(len(unique), 1)
    complexity_penalty = max(0.0, (len(unique) - 3) * 0.08)

    score = 0.35 + 0.45 * common_ratio
    if len(unique & {"Mo", "W", "V", "Nb", "Ta"}) > 0 and len(unique & CHALCOGEN) > 0:
        score += 0.15
    score -= complexity_penalty

    return float(np.clip(score, 0.0, 1.0))


def total_material_score(
    delta_g_h: float, thermo: float, kinetic: float, synth: float
) -> float:
    her_score = float(np.exp(-abs(delta_g_h) / 0.25))
    total = 0.4 * her_score + 0.25 * thermo + 0.2 * kinetic + 0.15 * synth
    return float(np.clip(total, 0.0, 1.0))


def is_reasonable_structure(structure: Structure) -> bool:
    if len(structure) < 2:
        return False
    if structure.lattice.c < 12.0:
        return False
    min_dist = _min_distance(structure)
    if min_dist < 0.8:
        return False
    return True


def proxy_targets(structure: Structure) -> Dict[str, float]:
    delta = estimate_delta_g_h(structure)
    thermo = estimate_thermodynamic_stability(structure)
    kinetic = estimate_kinetic_stability(structure)
    synth = estimate_synthesizability(structure)
    return {
        "delta_g_h": delta,
        "thermo": thermo,
        "kinetic": kinetic,
        "synthesis": synth,
        "total": total_material_score(delta, thermo, kinetic, synth),
    }


def clamp_frac_coords(coords: np.ndarray) -> np.ndarray:
    coords = coords.copy()
    coords[:, 0:2] = np.mod(coords[:, 0:2], 1.0)
    coords[:, 2] = np.clip(coords[:, 2], 0.45, 0.55)
    return coords


def to_tensor_targets(target: Dict[str, float], device) -> Dict[str, "torch.Tensor"]:
    import torch

    return {
        "delta_g_h": torch.tensor(target["delta_g_h"], dtype=torch.float32, device=device),
        "thermo": torch.tensor(target["thermo"], dtype=torch.float32, device=device),
        "kinetic": torch.tensor(target["kinetic"], dtype=torch.float32, device=device),
        "synthesis": torch.tensor(target["synthesis"], dtype=torch.float32, device=device),
    }
