import argparse
import glob
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

from dataset.material_dataset import MaterialDataset, structure_to_graph
from models.diffusion_model import DiffusionGNN
from models.optimization import PropertyPredictor
from models.structure_generator import StructureGenerator
from utils.geo_utils import proxy_targets, total_material_score
from utils.vis import (
    plot_generated_structures,
    plot_her_distribution,
    plot_stability_curve,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_cif_data(cif_dir: str) -> None:
    os.makedirs(cif_dir, exist_ok=True)
    if glob.glob(os.path.join(cif_dir, "*.cif")):
        return
    from download_jarvis_2d import main as download_main

    download_main()


def load_diffusion_model(ckpt_path: str, device: torch.device) -> DiffusionGNN:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = DiffusionGNN(
        node_in_dim=cfg["node_in_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        cond_dim=cfg.get("cond_dim", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_property_model(ckpt_path: str, device: torch.device) -> PropertyPredictor:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = PropertyPredictor(
        node_in_dim=cfg["node_in_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def evaluate_structure(
    structure: Structure, property_model: PropertyPredictor, device: torch.device
) -> Dict[str, float]:
    proxy = proxy_targets(structure)
    graph = structure_to_graph(structure=structure, cif_path="")
    g = graph.to(device)
    with torch.no_grad():
        pred = property_model(x=g.x, pos=g.pos, edge_index=g.edge_index)
    pred_delta = float(pred["delta_g_h"].item())
    pred_thermo = float(pred["thermo"].item())
    pred_kinetic = float(pred["kinetic"].item())
    pred_synth = float(pred["synthesis"].item())
    model_total = total_material_score(pred_delta, pred_thermo, pred_kinetic, pred_synth)
    return {
        "delta_g_h": proxy["delta_g_h"],
        "thermo": proxy["thermo"],
        "kinetic": proxy["kinetic"],
        "synthesis": proxy["synthesis"],
        "total_score": proxy["total"],
        "pred_delta_g_h": pred_delta,
        "pred_thermo": pred_thermo,
        "pred_kinetic": pred_kinetic,
        "pred_synthesis": pred_synth,
        "pred_total_score": model_total,
    }


def generate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_cif_data(args.cif_dir)

    os.makedirs("results", exist_ok=True)
    os.makedirs("generated_structures", exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = MaterialDataset(
        cif_dir=args.cif_dir,
        max_samples=args.max_seed_samples,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        seed=args.seed,
        use_cache=True,
    )

    diffusion_model = load_diffusion_model(args.diffusion_ckpt, device)
    property_model = load_property_model(args.property_ckpt, device)
    generator = StructureGenerator(model=diffusion_model, n_steps=args.reverse_steps)

    structures = generator.generate_multiple(dataset, n_generate=args.n_generate, device=device)

    rows: List[Dict[str, float]] = []
    for i, structure in enumerate(structures):
        path = os.path.join("generated_structures", f"candidate_{i:03d}.cif")
        structure.to(filename=path)
        metrics = evaluate_structure(structure=structure, property_model=property_model, device=device)
        rows.append(
            {
                "candidate_id": i,
                "formula": structure.composition.reduced_formula,
                "cif_path": path,
                **metrics,
            }
        )

    df = pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True)
    if len(df) < args.top_k:
        raise RuntimeError(
            f"Generated {len(df)} candidates, smaller than top_k={args.top_k}."
        )
    top_df = df.head(args.top_k).copy()

    for rank, row in top_df.reset_index(drop=True).iterrows():
        source = row["cif_path"]
        target = os.path.join("generated_structures", f"generated_{rank}.cif")
        Structure.from_file(source).to(filename=target)
        top_df.loc[top_df.index[rank], "top_cif_path"] = target

    top_csv = os.path.join("results", "top10_generated_materials.csv")
    top_df.to_csv(top_csv, index=False)

    plot_her_distribution(df=top_df, save_path=os.path.join("results", "her_performance.png"))
    plot_stability_curve(df=top_df, save_path=os.path.join("results", "stability_curve.png"))
    plot_df = top_df.copy()
    plot_df["cif_path"] = plot_df["top_cif_path"].fillna(plot_df["cif_path"])
    plot_generated_structures(
        df=plot_df,
        save_path=os.path.join("results", "generated_structures.png"),
    )

    print(f"Generated candidates: {len(df)}")
    print(f"Saved top-{args.top_k} CSV: {top_csv}")
    print("Saved figures:")
    print("  results/her_performance.png")
    print("  results/stability_curve.png")
    print("  results/generated_structures.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 2D materials with trained diffusion model.")
    parser.add_argument("--cif_dir", type=str, default="data/cif")
    parser.add_argument("--n_generate", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--reverse_steps", type=int, default=60)
    parser.add_argument("--max_seed_samples", type=int, default=400)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max_neighbors", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--diffusion_ckpt", type=str, default=os.path.join("checkpoints", "diffusion_model.pt")
    )
    parser.add_argument(
        "--property_ckpt", type=str, default=os.path.join("checkpoints", "property_model.pt")
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate(args)
