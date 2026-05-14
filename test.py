import argparse
import glob
import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

from dataset.material_dataset import MaterialDataset, structure_to_graph
from models.diffusion_model import DiffusionGNN
from models.optimization import PropertyPredictor
from models.structure_generator import StructureGenerator
from utils.geo_utils import proxy_targets, total_material_score
from utils.vis import plot_generated_structures, plot_her_distribution, plot_stability_curve


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_cif_data(cif_dir: str) -> None:
    os.makedirs(cif_dir, exist_ok=True)
    metadata_csv = os.path.join(os.path.dirname(cif_dir), "jarvis_dft_2d_metadata.csv")
    if glob.glob(os.path.join(cif_dir, "*.cif")) and os.path.exists(metadata_csv):
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
    pred_total = total_material_score(pred_delta, pred_thermo, pred_kinetic, pred_synth)

    delta_final = 0.50 * proxy["delta_g_h"] + 0.50 * pred_delta
    thermo_final = float(np.clip(0.60 * proxy["thermo"] + 0.40 * pred_thermo, 0.0, 1.0))
    kinetic_final = float(np.clip(0.60 * proxy["kinetic"] + 0.40 * pred_kinetic, 0.0, 1.0))
    synth_final = float(np.clip(0.60 * proxy["synthesis"] + 0.40 * pred_synth, 0.0, 1.0))
    total_final = total_material_score(delta_final, thermo_final, kinetic_final, synth_final)

    return {
        "delta_g_h_proxy": proxy["delta_g_h"],
        "thermo_proxy": proxy["thermo"],
        "kinetic_proxy": proxy["kinetic"],
        "synthesis_proxy": proxy["synthesis"],
        "total_proxy": proxy["total"],
        "pred_delta_g_h": pred_delta,
        "pred_thermo": pred_thermo,
        "pred_kinetic": pred_kinetic,
        "pred_synthesis": pred_synth,
        "pred_total": pred_total,
        "delta_g_h_final": delta_final,
        "thermo_final": thermo_final,
        "kinetic_final": kinetic_final,
        "synthesis_final": synth_final,
        "total_score": total_final,
    }


def summarize_metrics(df: pd.DataFrame, method_name: str) -> Dict[str, float]:
    if len(df) == 0:
        return {
            "Method": method_name,
            "Avg_HER_DeltaG_eV": np.nan,
            "Stability_Score": np.nan,
            "Synthesis_Success_Rate": np.nan,
            "Avg_Total_Score": np.nan,
            "N": 0,
        }
    stability = (df["thermo_final"] + df["kinetic_final"]) / 2.0
    return {
        "Method": method_name,
        "Avg_HER_DeltaG_eV": float(np.mean(np.abs(df["delta_g_h_final"]))),
        "Stability_Score": float(np.mean(stability)),
        "Synthesis_Success_Rate": float(np.mean(df["synthesis_final"] >= 0.60)),
        "Avg_Total_Score": float(np.mean(df["total_score"])),
        "N": int(len(df)),
    }


def evaluate_cif_folder(
    cif_dir: str, property_model: PropertyPredictor, device: torch.device, max_items: int = 100
) -> pd.DataFrame:
    cif_files = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))[:max_items]
    rows: List[Dict[str, float]] = []
    for i, path in enumerate(cif_files):
        try:
            structure = Structure.from_file(path)
            metrics = evaluate_structure(structure=structure, property_model=property_model, device=device)
            rows.append(
                {
                    "candidate_id": i,
                    "formula": structure.composition.reduced_formula,
                    "cif_path": path,
                    **metrics,
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def write_comparison_outputs(
    ours_df: pd.DataFrame, baseline_df: Optional[pd.DataFrame], out_dir: str
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = [summarize_metrics(ours_df, "Ours")]
    if baseline_df is not None and len(baseline_df) > 0:
        rows.insert(0, summarize_metrics(baseline_df, "baseline"))
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(out_dir, "baseline_comparison.csv"), index=False)

    md_lines = [
        "| Method | Avg HER DeltaG (eV) | Stability Score | Synthesis Success Rate | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in comp.iterrows():
        md_lines.append(
            f"| {r['Method']} | {r['Avg_HER_DeltaG_eV']:.4f} | {r['Stability_Score']:.4f} | "
            f"{r['Synthesis_Success_Rate']:.4f} | {int(r['N'])} |"
        )
    with open(os.path.join(out_dir, "baseline_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


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
        raise RuntimeError(f"Generated {len(df)} candidates, smaller than top_k={args.top_k}.")
    top_df = df.head(args.top_k).copy()

    for rank, row in top_df.reset_index(drop=True).iterrows():
        source = row["cif_path"]
        target = os.path.join("generated_structures", f"generated_{rank}.cif")
        Structure.from_file(source).to(filename=target)
        top_df.loc[top_df.index[rank], "top_cif_path"] = target

    top_csv = os.path.join("results", "top10_generated_materials.csv")
    top_df.to_csv(top_csv, index=False)

    plot_df = top_df.copy()
    plot_df["delta_g_h"] = plot_df["delta_g_h_final"]
    plot_df["thermo"] = plot_df["thermo_final"]
    plot_df["kinetic"] = plot_df["kinetic_final"]
    plot_df["synthesis"] = plot_df["synthesis_final"]
    plot_df["cif_path"] = plot_df["top_cif_path"].fillna(plot_df["cif_path"])

    plot_her_distribution(df=plot_df, save_path=os.path.join("results", "her_performance.png"))
    plot_stability_curve(df=plot_df, save_path=os.path.join("results", "stability_curve.png"))
    plot_generated_structures(df=plot_df, save_path=os.path.join("results", "generated_structures.png"))

    baseline_df: Optional[pd.DataFrame] = None
    if args.baseline_cif_dir and os.path.isdir(args.baseline_cif_dir):
        baseline_df = evaluate_cif_folder(
            cif_dir=args.baseline_cif_dir,
            property_model=property_model,
            device=device,
            max_items=args.baseline_max_items,
        )
        if len(baseline_df) > 0:
            baseline_df.to_csv(os.path.join("results", "baseline_evaluation.csv"), index=False)

    write_comparison_outputs(ours_df=top_df, baseline_df=baseline_df, out_dir="results")

    print(f"Generated candidates: {len(df)}")
    print(f"Saved top-{args.top_k} CSV: {top_csv}")
    print("Saved figures:")
    print("  results/her_performance.png")
    print("  results/stability_curve.png")
    print("  results/generated_structures.png")
    if baseline_df is not None and len(baseline_df) > 0:
        print("Saved baseline comparison: results/baseline_comparison.csv")


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
    parser.add_argument(
        "--baseline_cif_dir",
        type=str,
        default="d:/cursor_file/baseline_material_generation/generated_materials/cif_files",
    )
    parser.add_argument("--baseline_max_items", type=int, default=100)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate(args)
