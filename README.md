# Diffusion-based 2D Material Generation for HER Catalysts

## 1. Project Goal
Generate new 2D crystal structures with a GNN diffusion model, while jointly optimizing:
- HER activity (`Delta G_H` close to `0 eV`)
- Thermodynamic stability
- Kinetic stability
- Experimental synthesizability

## 2. Dataset
This project uses **JARVIS-DFT 2D** from `jarvis-tools` (`data("dft_2d")`), converted to CIF in `data/cif/`.

Data files:
- `data/cif/*.cif`
- `data/jarvis_dft_2d_metadata.csv` (exported from JARVIS records)
- `data/experimental_2d_labels.csv` (curated experimentally synthesized 2D formula list)

## 3. Method Overview
Pipeline:
1. Download JARVIS-DFT 2D structures and metadata.
2. Build crystal graphs from CIF (`x`, `pos`, `edge_index`).
3. Train GNN diffusion denoiser on noisy fractional coordinates.
4. Train multi-task property heads for HER/stability/synthesis.
5. Reverse-generate structures with 2D constraints:
- `z` slab in `[0.45, 0.55]`
- `c-axis >= 15 A`
6. Rank generated materials and export top-k CIF.

## 4. Model Architecture
```mermaid
flowchart LR
    A[JARVIS-DFT 2D CIF + Metadata] --> B[Crystal Graph Construction]
    B --> C[GNN Diffusion Denoiser]
    C --> D[Reverse Diffusion Structure Generator]
    D --> E[Property Heads: HER/Thermo/Kinetic/Synthesis]
    E --> F[Multi-objective Ranking]
    F --> G[Top-k Generated 2D Candidates]
```

Main files:
- `models/diffusion_model.py`
- `models/optimization.py`
- `models/structure_generator.py`

## 5. Diffusion Process
Forward noising:
\[
\tilde{x}_t = x_0 + \sigma(t)\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
\]

Denoising loss:
\[
L_{diffusion} = \|\epsilon - \epsilon_\theta(\tilde{x}_t, t, c)\|_2^2
\]

Condition vector:
\[
c = [\Delta G_H^{target},\ S_{thermo}^{target},\ S_{synthesis}^{target}]
\]

## 6. Multi-objective Optimization
Predicted properties:
- `Delta G_H`
- `S_thermo`
- `S_kinetic`
- `S_synthesis`

Joint objective:
\[
L = L_{diffusion}
 + \lambda_1 |\Delta G_H|
 + \lambda_2 (1 - S_{thermo})
 + \lambda_3 (1 - S_{kinetic})
 + \lambda_4 (1 - S_{synthesis})
\]

## 7. Experimental Data Integration for Synthesizability
To satisfy the "existing experimental data" requirement, synthesis supervision is not purely heuristic:

1. **JARVIS metadata export** (`download_jarvis_2d.py`):
- `formation_energy_peratom`
- `ehull`
- `exfoliation_energy`
- `icsd`

2. **Curated experimental labels** (`data/experimental_2d_labels.csv`):
- known experimentally synthesized 2D formula families (graphene, h-BN, TMDCs, magnetic 2D compounds, etc.)

3. **Training target fusion** (`dataset/material_dataset.py`):
- synthesis target combines experimental label signal + JARVIS metadata signal + geometric proxy
- thermo target combines `formation_energy`, `ehull`, `exfoliation_energy` + proxy

## 8. Stability and HER Evaluation
`utils/geo_utils.py` provides fast screening estimators:
- `estimate_delta_g_h`
- `estimate_thermodynamic_stability`
- `estimate_kinetic_stability`
- `estimate_synthesizability`

Final ranking score combines HER, thermo, kinetic, and synthesis.

## 9. Training
```bash
python train.py --epochs 10
```

Outputs:
- `checkpoints/diffusion_model.pt`
- `checkpoints/property_model.pt`
- `results/loss_curve.png`

## 10. Generation and Evaluation
```bash
python test.py --n_generate 50 --top_k 10
```

Outputs:
- `generated_structures/generated_0.cif` ... `generated_9.cif`
- `results/top10_generated_materials.csv`
- `results/her_performance.png`
- `results/stability_curve.png`
- `results/generated_structures.png`

## 11. Baseline Comparison (material_generation)
Baseline repository:
- `https://github.com/deamean/material_generation`

The test script can evaluate baseline CIFs with the same metric pipeline and auto-generate:
- `results/baseline_evaluation.csv`
- `results/baseline_comparison.csv`
- `results/baseline_comparison.md`

Current run summary:

| Method | Avg HER DeltaG (eV) | Stability Score | Synthesis Success Rate |
|---|---:|---:|---:|
| baseline | 0.1075 | 0.9124 | 1.0000 |
| Ours | 0.0679 | 0.8980 | 1.0000 |

## 12. Innovation
1. Conditional diffusion over crystal graphs with explicit HER/stability/synthesis objectives.
2. Data-enhanced supervision for synthesis/stability using JARVIS metadata plus curated experimental 2D label set.
3. Reverse diffusion with explicit 2D geometric constraints (`z` slab and large vacuum).
4. Unified post-generation evaluation pipeline with automatic baseline comparison export.

## 13. Limitations and Required DFT Validation
This implementation is for high-throughput screening and engineering demonstration. It combines ML predictions and surrogate/proxy scoring for HER and stability. Final validation still requires:
- DFT adsorption energy for HER
- phonon analysis
- AIMD
- NEB
- experimental synthesis and characterization

## 14. How to Run
```bash
pip install -r requirements.txt
python download_jarvis_2d.py
python train.py --epochs 10
python test.py --n_generate 50 --top_k 10
```

Optional baseline comparison directory:
```bash
python test.py --n_generate 50 --top_k 10 --baseline_cif_dir d:/cursor_file/baseline_material_generation/generated_materials/cif_files
```

## 15. Repository Structure
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
│   ├── cif/
│   ├── experimental_2d_labels.csv
│   └── jarvis_dft_2d_metadata.csv
├── results/
├── generated_structures/
├── checkpoints/
├── download_jarvis_2d.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
```
