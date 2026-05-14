import argparse
import glob
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset.material_dataset import MaterialDataset
from models.diffusion_model import DiffusionGNN, diffusion_step
from models.optimization import PropertyPredictor, multitask_loss
from utils.geo_utils import proxy_targets, to_tensor_targets
from utils.vis import plot_loss_curve


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_cif_data(cif_dir: str) -> None:
    os.makedirs(cif_dir, exist_ok=True)
    cif_files = glob.glob(os.path.join(cif_dir, "*.cif"))
    if cif_files:
        return
    from download_jarvis_2d import main as download_main

    download_main()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_cif_data(args.cif_dir)

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    dataset = MaterialDataset(
        cif_dir=args.cif_dir,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        max_samples=args.max_samples,
        seed=args.seed,
        use_cache=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    diffusion_model = DiffusionGNN(
        node_in_dim=dataset.node_feature_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        cond_dim=3,
    ).to(device)
    property_model = PropertyPredictor(
        node_in_dim=dataset.node_feature_dim,
        hidden_dim=args.hidden_dim,
        n_layers=max(2, args.n_layers - 1),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(diffusion_model.parameters()) + list(property_model.parameters()),
        lr=args.lr,
        weight_decay=1e-6,
    )

    history: Dict[str, List[float]] = {
        "total": [],
        "diffusion": [],
        "her": [],
        "thermo": [],
        "kinetic": [],
        "synthesis": [],
    }

    for epoch in range(args.epochs):
        diffusion_model.train()
        property_model.train()

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        if args.steps_per_epoch > 0:
            indices = indices[: min(len(indices), args.steps_per_epoch)]

        sums = {k: 0.0 for k in history.keys()}
        pbar = tqdm(indices, desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=100)
        for idx in pbar:
            batch = dataset[idx].to(device)
            target_proxy = proxy_targets(dataset[idx].structure)
            target = to_tensor_targets(target_proxy, device=device)

            t = torch.rand(1, device=device)
            cond = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32, device=device)
            noisy_pos, true_noise, pred_noise = diffusion_step(
                model=diffusion_model,
                x=batch.x,
                clean_pos=batch.pos,
                edge_index=batch.edge_index,
                t=t,
                cond=cond,
            )
            diff_loss = F.mse_loss(pred_noise, true_noise)

            pred = property_model(x=batch.x, pos=noisy_pos, edge_index=batch.edge_index)
            total_loss, details = multitask_loss(
                diffusion_loss=diff_loss,
                pred=pred,
                target=target,
                lambda_her=args.lambda_her,
                lambda_thermo=args.lambda_thermo,
                lambda_kinetic=args.lambda_kinetic,
                lambda_synthesis=args.lambda_synthesis,
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(diffusion_model.parameters()) + list(property_model.parameters()), 2.0
            )
            optimizer.step()

            sums["total"] += float(total_loss.detach().item())
            for key in ["diffusion", "her", "thermo", "kinetic", "synthesis"]:
                sums[key] += float(details[key].item())

            pbar.set_postfix(loss=f"{sums['total'] / (pbar.n + 1):.4f}")

        denom = float(max(len(indices), 1))
        for key in history:
            history[key].append(sums[key] / denom)

        print(
            f"Epoch {epoch + 1:02d} | total={history['total'][-1]:.4f} "
            f"diff={history['diffusion'][-1]:.4f} her={history['her'][-1]:.4f} "
            f"thermo={history['thermo'][-1]:.4f} kinetic={history['kinetic'][-1]:.4f} "
            f"synth={history['synthesis'][-1]:.4f}"
        )

    torch.save(
        {
            "model_state": diffusion_model.state_dict(),
            "config": {
                "node_in_dim": dataset.node_feature_dim,
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "cond_dim": 3,
            },
        },
        os.path.join("checkpoints", "diffusion_model.pt"),
    )
    torch.save(
        {
            "model_state": property_model.state_dict(),
            "config": {
                "node_in_dim": dataset.node_feature_dim,
                "hidden_dim": args.hidden_dim,
                "n_layers": max(2, args.n_layers - 1),
            },
        },
        os.path.join("checkpoints", "property_model.pt"),
    )

    plot_loss_curve(history, os.path.join("results", "loss_curve.png"))
    print("Saved checkpoints to checkpoints/")
    print("Saved loss curve to results/loss_curve.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GNN diffusion model for 2D materials.")
    parser.add_argument("--cif_dir", type=str, default="data/cif")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=160)
    parser.add_argument("--max_samples", type=int, default=320)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max_neighbors", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_her", type=float, default=0.6)
    parser.add_argument("--lambda_thermo", type=float, default=0.4)
    parser.add_argument("--lambda_kinetic", type=float, default=0.4)
    parser.add_argument("--lambda_synthesis", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    train(parser.parse_args())
