# Diffusion-based 2D Material Generation for HER Catalysts

## 1. Project Goal
Build a GNN-based diffusion framework to generate 2D material structures for HER catalyst screening, while jointly optimizing:
- HER activity (`ΔG_H` close to `0 eV`)
- Thermodynamic stability
- Kinetic stability
- Experimental synthesizability

## 2. Dataset (JARVIS-DFT 2D)
This project uses **JARVIS-DFT 2D** materials downloaded with `jarvis-tools` (`data("dft_2d")`), then converted to CIF under `data/cif/`.

Data pipeline:
1. Download JSON-like structure records from JARVIS.
2. Convert `atoms` dict to JARVIS `Atoms`.
3. Save each material as CIF (`jid_formula.cif`).
4. Parse CIF with `pymatgen` and convert to crystal graphs.

## 3. Method Overview
1. CIF -> crystal graph (`x`, `pos`, `edge_index`)
2. Train diffusion denoiser on noisy fractional coordinates
3. Train property predictor jointly (HER/stability/synthesis)
4. Reverse diffusion generation with 2D constraints:
- `z` coordinate constrained to `[0.45, 0.55]`
- `c-axis >= 15 Å` vacuum
5. Rank generated structures by multi-objective score and export top-k CIF

## 4. Model Architecture
```mermaid
flowchart LR
    A[JARVIS-DFT 2D CIF Dataset] --> B[Crystal Graph Construction]
    B --> C[GNN Diffusion Denoiser]
    C --> D[Structure Generator]
    D --> E[HER / Stability / Synthesis Predictors]
    E --> F[Multi-objective Optimization]
    F --> G[Top-k Generated 2D HER Candidates]
```

Core modules:
- `models/diffusion_model.py`: `DiffusionGNN`
- `models/optimization.py`: `PropertyPredictor` + multitask loss
- `models/structure_generator.py`: reverse diffusion sampling

## 5. Diffusion Process
Forward noising:
\[
\tilde{\mathbf{x}}_t = \mathbf{x}_0 + \sigma(t)\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
\]

Denoising objective:
\[
\mathcal{L}_{diffusion} = \|\epsilon - \epsilon_\theta(\tilde{\mathbf{x}}_t, t, c)\|_2^2
\]

Condition vector:
\[
c = [\Delta G_H^{target},\ S_{stability}^{target},\ S_{synthesis}^{target}]
\]

## 6. Multi-objective Optimization
Property heads predict:
- `ΔG_H`
- `S_thermo`
- `S_kinetic`
- `S_synthesis`

Joint loss:
\[
\mathcal{L}=\mathcal{L}_{diffusion}
 + \lambda_1|\Delta G_H|
 + \lambda_2(1-S_{thermo})
 + \lambda_3(1-S_{kinetic})
 + \lambda_4(1-S_{synthesis})
\]

Training implementation includes both:
- proxy-supervised regression terms
- objective-shaping terms pushing toward HER/stability/synthesis targets

## 7. HER / Stability / Synthesizability Proxies
Implemented in `utils/geo_utils.py`:
- `estimate_delta_g_h`
- `estimate_thermodynamic_stability`
- `estimate_kinetic_stability`
- `estimate_synthesizability`
- `total_material_score`

Proxy logic includes composition-aware and geometry-aware heuristics (TMDC-like chemistry bonus, minimum-distance penalty, complexity penalty, etc.).

## 8. Training
Command:
```bash
python train.py --epochs 10
```

Outputs:
- `checkpoints/diffusion_model.pt`
- `checkpoints/property_model.pt`
- `results/loss_curve.png`

## 9. Generation & Ranking
Command:
```bash
python test.py --n_generate 50 --top_k 10
```

Outputs:
- Top-k CIFs: `generated_structures/generated_0.cif` ... `generated_9.cif`
- Ranking CSV: `results/top10_generated_materials.csv`
- Figures:
  - `results/her_performance.png`
  - `results/stability_curve.png`
  - `results/generated_structures.png`

## 10. Evaluation Metrics
- HER activity: `|ΔG_H|` (closer to 0 is better)
- Thermodynamic stability score
- Kinetic stability score
- Synthesizability score
- Weighted total score for ranking

## 11. Results Visualization
Required plots:
- `loss_curve.png`: training losses
- `her_performance.png`: ΔG_H distribution with target line at 0 eV
- `stability_curve.png`: thermo / kinetic / synthesis trend across ranked candidates
- `generated_structures.png`: top-10 generated 2D structure projections

## 12. Comparison with Baseline
Example format (fill with your experiment numbers):

| Method | Avg HER ΔG (eV) | Stability Score | Synthesis Success Rate |
|---|---:|---:|---:|
| Baseline (Random perturbation) | 0.38 | 0.56 | 0.51 |
| Ours (GNN Diffusion + Multi-task) | 0.21 | 0.73 | 0.69 |

## 13. Innovation
1. Conditional crystal diffusion for 2D materials with explicit HER/stability/synthesis targets.
2. Multi-task training coupling denoising and property optimization.
3. Practical 2D structural constraints (`z`-slab + large vacuum) during reverse sampling.
4. End-to-end workflow from public JARVIS data to ranked CIF candidates.

## 14. Limitations and Future DFT Validation
The current implementation uses fast **surrogate/proxy predictors** for HER activity, thermodynamic stability, kinetic stability, and synthesizability. These predictors are used for high-throughput screening and engineering demonstration. Final validation requires DFT adsorption energy calculations, phonon calculations, AIMD, NEB, and experimental synthesis.

## 15. How to Run
```bash
pip install -r requirements.txt
python download_jarvis_2d.py
python train.py --epochs 10
python test.py --n_generate 50 --top_k 10
```

Quick smoke test:
```bash
python train.py --epochs 3 --steps_per_epoch 80 --max_samples 200
python test.py --n_generate 20 --top_k 10
```

## 16. Repository Structure
```text
.
├── models/
│   ├── diffusion_model.py
│   ├── optimization.py
│   └── structure_generator.py
├── dataset/
│   └── material_dataset.py
├── utils/
│   ├── geo_utils.py
│   └── vis.py
├── data/
│   └── cif/
├── results/
├── generated_structures/
├── checkpoints/
├── download_jarvis_2d.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
```
