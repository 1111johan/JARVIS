import os
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Structure


def _ensure_parent(save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


def plot_loss_curve(loss_history: Dict[str, List[float]], save_path: str) -> None:
    _ensure_parent(save_path)
    plt.figure(figsize=(10, 6))
    for key, values in loss_history.items():
        plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_her_distribution(df: pd.DataFrame, save_path: str) -> None:
    _ensure_parent(save_path)
    plt.figure(figsize=(8, 5))
    vals = df["delta_g_h"].values
    plt.hist(vals, bins=20, color="#3A86FF", alpha=0.85, edgecolor="black")
    plt.axvline(0.0, color="red", linestyle="--", linewidth=2, label="Target ΔG_H = 0 eV")
    plt.xlabel("Predicted / Proxy ΔG_H (eV)")
    plt.ylabel("Count")
    plt.title("HER Performance Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_stability_curve(df: pd.DataFrame, save_path: str) -> None:
    _ensure_parent(save_path)
    d = df.sort_values("total_score", ascending=False).reset_index(drop=True)
    x = np.arange(len(d))
    plt.figure(figsize=(9, 5))
    plt.plot(x, d["thermo"], marker="o", label="Thermodynamic Stability")
    plt.plot(x, d["kinetic"], marker="s", label="Kinetic Stability")
    plt.plot(x, d["synthesis"], marker="^", label="Synthesizability")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Ranked Candidate Index")
    plt.ylabel("Score")
    plt.title("Stability and Synthesis Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_generated_structures(df: pd.DataFrame, save_path: str) -> None:
    _ensure_parent(save_path)
    top = df.sort_values("total_score", ascending=False).head(10).reset_index(drop=True)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(top):
            ax.axis("off")
            continue
        cif_path = top.loc[i, "cif_path"]
        structure = Structure.from_file(cif_path)
        frac = structure.frac_coords
        ax.scatter(frac[:, 0], frac[:, 1], s=20, c=frac[:, 2], cmap="viridis")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"#{i+1} | ΔG={top.loc[i, 'delta_g_h']:.2f}")

    fig.suptitle("Top-10 Generated 2D Structures (Fractional XY Projection)")
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
