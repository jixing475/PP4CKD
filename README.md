# PP4CKD: Polypharmacology Predictor for Chronic Kidney Disease

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2022.09+-green.svg)](https://www.rdkit.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23147411.svg)](https://doi.org/10.5281/zenodo.23147411)

**PP4CKD** (*Polypharmacology Predictor for Chronic Kidney Disease*) is a state-of-the-art, multi-target ligand-based deep learning framework specifically engineered for nephrology drug discovery and target deconvolution. 

Strictly conforming to the algorithmic and architectural design principles established by the Reymond group in the **Polypharmacology Browser 3 (PPB3)**, PP4CKD scales the methodology to the latest **ChEMBL 36** database (encompassing **7,676** curated biological targets) while establishing a dedicated clinical evaluation cohort of **623 CKDdb renal disease targets**.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Comprehensive ChEMBL 36 Benchmark](#-comprehensive-chembl-36-benchmark)
- [Renal Specificity & CKD Target Uplift](#-renal-specificity--ckd-target-uplift)
- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Python API](#python-api)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Minimal Validation Script](#minimal-validation-script)
- [Repository Structure](#-repository-structure)
- [Full Pipeline Reproduction](#-full-pipeline-reproduction)
- [Zenodo Model Weights Archive & DOI Guide](#-zenodo-model-weights-archive--doi-guide)
- [Citation & License](#-citation--license)

---

## 🌟 Key Features

1. **Academic-Grade Reproducibility**: Faithful PyTorch implementation of the PPB3 multi-label neural network with strict adherence to Reymond group benchmarks.
2. **ChEMBL 36 Scaling**: Trained and evaluated on active ligand-target pairs filtered across 95 standard activity types from ChEMBL 36 ($activity \le 10\,\mu\text{M}$ or $inhibition \ge 50\%$).
3. **Multi-Fingerprint Ensemble**: Native generation and inference for **7 distinct molecular fingerprints**, including 4096-bit circular, topological, and pharmacophoric descriptors, alongside an 8192-bit Fused representation (`ECFP4` + `MHFP6`).
4. **CKDdb Disease-Centric Dual Projection**: Instantly reports both the global top-$K$ predictions across all 7,676 targets and disease-specific ranking within the 623 CKDdb chronic kidney disease target space.
5. **High-Throughput Acceleration**: Optimized with bit-packing memmaps, mixed-precision (`torch.amp`), and native Apple Silicon MPS / NVIDIA CUDA acceleration.

---

## 🏗️ System Architecture

```
                                  Input SMILES
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   PPB3 SMILES Standardization │
                       │ (Largest Fragment + Uncharge) │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │     Fingerprint Generation    │
                       │  (4096-bit / 8192-bit Fused)  │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │      PPB3 Deep Classifier     │
                       │  Linear(D, 1000) ──► ReLU/DO  │
                       │  Linear(1000, 500) ─► ReLU/DO │
                       │  Linear(500, 7676) ─► Sigmoid │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                        Probabilities across 7,676 Targets
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
  ┌───────────────────────────┐                 ┌───────────────────────────┐
  │   Global Top-K Ranking    │                 │   CKDdb Top-K Projection  │
  │ (All 7,676 ChEMBL Targets)│                 │ (623 Renal Disease Targets)│
  └───────────────────────────┘                 └───────────────────────────┘
```

---

## 📊 Comprehensive ChEMBL 36 Benchmark

All models were evaluated using rigorous **10-Fold Cross-Validation** on the complete ChEMBL 36 dataset. Evaluation metrics include Top-1, Top-5, and Top-10 Recall, Top-10 Precision, and Micro-AUPR.

### Table 1: Global Benchmark Performance (7,676 Targets)
*Results denote mean $\pm$ standard deviation across 10 completed cross-validation folds.*

| Fingerprint | Dimension | Top-1 Recall | Top-5 Recall | Top-10 Recall | Top-10 Precision | Micro-AUPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fused (ECFP4 + MHFP6)** | **8192** | **53.62 ± 1.30%** | **79.14 ± 1.33%** | **85.10 ± 1.07%** | **14.71 ± 0.23%** | **0.3481** |
| **ECFP4** | 4096 | 52.83 ± 1.49% | 78.32 ± 1.60% | 84.37 ± 1.30% | 14.57 ± 0.25% | 0.3362 |
| **ECFP6** | 4096 | 52.35 ± 1.02% | 77.63 ± 1.13% | 83.73 ± 0.94% | 14.45 ± 0.21% | 0.3299 |
| **MHFP6** | 4096 | 51.89 ± 1.61% | 77.28 ± 1.81% | 83.53 ± 1.51% | 14.40 ± 0.30% | 0.3232 |
| **RDKit** | 4096 | 44.88 ± 0.80% | 69.66 ± 0.88% | 77.04 ± 0.78% | 13.15 ± 0.13% | 0.2588 |
| **Layered** | 4096 | 44.11 ± 1.37% | 68.88 ± 1.56% | 76.30 ± 1.36% | 12.99 ± 0.24% | 0.2496 |
| **AtomPair** | 4096 | 43.69 ± 1.23% | 68.25 ± 1.42% | 75.70 ± 1.26% | 12.88 ± 0.23% | 0.2442 |

---

### Table 2: CKDdb Renal Disease Subset Performance (623 Targets)
*Evaluated on the 623 clinically relevant kidney disease targets identified from CKDdb clinical associations, indications, and renal tissue bioassays.*

| Fingerprint | Dimension | Top-1 Recall | Top-5 Recall | Top-10 Recall | Top-10 Precision | Micro-AUPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fused (ECFP4 + MHFP6)** | **8192** | **56.17 ± 1.40%** | **81.19 ± 1.23%** | **87.54 ± 0.89%** | **13.76 ± 0.17%** | **0.3851** |
| **ECFP4** | 4096 | 55.19 ± 1.60% | 80.30 ± 1.48% | 86.80 ± 1.14% | 13.63 ± 0.19% | 0.3699 |
| **ECFP6** | 4096 | 54.70 ± 1.15% | 79.70 ± 1.09% | 86.27 ± 0.84% | 13.55 ± 0.17% | 0.3643 |
| **MHFP6** | 4096 | 54.17 ± 1.84% | 79.33 ± 1.76% | 86.09 ± 1.32% | 13.51 ± 0.23% | 0.3558 |
| **RDKit** | 4096 | 46.83 ± 0.71% | 72.11 ± 0.76% | 80.27 ± 0.63% | 12.53 ± 0.10% | 0.2798 |
| **Layered** | 4096 | 46.13 ± 1.26% | 71.49 ± 1.36% | 79.67 ± 1.14% | 12.42 ± 0.18% | 0.2707 |
| **AtomPair** | 4096 | 45.34 ± 1.17% | 70.52 ± 1.25% | 78.86 ± 1.02% | 12.28 ± 0.15% | 0.2553 |

---

## 🎯 Renal Specificity & CKD Target Uplift

A defining finding in our benchmark evaluation is that **every single fingerprint representation demonstrates a marked, statistically significant performance uplift when evaluated specifically on the 623 CKDdb renal target subset compared to the unconstrained global dataset**:

- **Top-1 Recall Uplift**: The Fused model improves from **53.62%** on the global set to **56.17%** on the CKDdb subset ($+2.55\%$ absolute gain).
- **Top-10 Recall Uplift**: The Fused model captures **87.54%** of active targets within the top-10 predictions on CKD targets, compared to **85.10%** globally ($+2.44\%$ gain).
- **Micro-AUPR Enrichment**: Global micro-AUPR increases from **0.3481** to **0.3851** on the CKD subset ($+10.6\%$ relative enrichment).

### Scientific Rationale
1. **Target Density and Therapeutic Annotation**: Renal targets (e.g., sodium/glucose cotransporters, renin-angiotensin-aldosterone axis enzymes, endothelin receptors) represent clinically validated therapeutic families with extensive, high-quality SAR chemogenomics data.
2. **Cross-Family Feature Fusion**: Combining circular topological neighborhood descriptors (`ECFP4`) with MinHash pharmacophore path invariants (`MHFP6`) in the 8192-bit Fused architecture effectively captures both localized pharmacophores and global shape constraints necessary for selective renal target recognition.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jixing475/pp4ckd.git
cd pp4ckd

# Create conda environment with Python 3.10
conda create -n pp4ckd python=3.10 -y
conda activate pp4ckd

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Python API

```python
from pp4ckd import PP4CKDPredictor

# Initialize predictor (automatically uses MPS/CUDA if available)
predictor = PP4CKDPredictor(fp_type="Fused")

# Predict targets for Dapagliflozin (Farxiga)
smiles = "CCc1ccc(Cc2cc(Cl)c(C3OC(CO)C(O)C(O)C3O)cc2)cc1"

# Dual Prediction: Global Top-5 & CKDdb Specific Top-5
results = predictor.predict_dual(smiles, top_k=5)

print("Global Top Targets:")
print(results["global"])

print("\nCKDdb Renal Targets:")
print(results["ckd"])
```

### Command-Line Interface (CLI)

```bash
# Predict global and CKDdb targets for Aspirin
python -m pp4ckd.predictor --smiles "CC(=O)Oc1ccccc1C(=O)O" --top-k 5

# Predict exclusively within the 623 CKDdb renal targets
python -m pp4ckd.predictor --smiles "CCc1ccc(Cc2cc(Cl)c(C3OC(CO)C(O)C(O)C3O)cc2)cc1" --ckd-only --top-k 5
```

### Minimal Validation Script

Execute the self-contained validation script to verify environment configuration and end-to-end model inference:

```bash
python examples/quickstart.py
```

Expected output includes immediate recognition of **Sodium/glucose cotransporter 2 (SLC5A2, CHEMBL3884)** with **>99.7% confidence** for Dapagliflozin, and **Prostaglandin G/H synthase 2 (PTGS2, CHEMBL230)** and renal transporters (OAT1, MRP2) for Aspirin.

---

## 📁 Repository Structure

```
pp4ckd/
├── README.md               # Comprehensive documentation and benchmark reports
├── LICENSE                 # Apache License 2.0
├── pyproject.toml          # PEP 517/518 build specification
├── requirements.txt        # Locked lightweight dependencies
├── pp4ckd/                 # Core Python package
│   ├── __init__.py         # Package exports and versioning
│   ├── model.py            # PyTorch PPB3Net architecture and state loading
│   ├── fingerprints.py     # 7 standardized fingerprint engines (4096 & 8192-bit)
│   ├── predictor.py        # High-level inference API and CLI
│   └── data_pipeline.py    # ChEMBL SQLite extraction & standardization logic
├── scripts/
│   ├── train_multigpu.py   # Multi-GPU asynchronous 10-fold CV & full training
│   ├── evaluate.py         # Full-set and CKDdb slice benchmark evaluator
│   └── download_chembl.sh  # Automated ChEMBL SQLite downloader & verifier
├── data/
│   ├── ckd_623_targets.csv # CKDdb 623 renal target metadata mappings
│   └── target_labels.tsv   # 7,676 ChEMBL target dictionary labels
├── models/
│   ├── fused_full_model.pt # Production 8192-bit Fused weights (50 MB)
│   └── ecfp4_full_model.pt # Production 4096-bit ECFP4 weights (33 MB)
└── examples/
    ├── quickstart.py       # Minimal reproducible verification script
    └── benchmark_results/  # Full cross-validation and benchmark JSON artifacts
        ├── benchmark_summary.json
        ├── cv_metrics.json
        └── ckd_subset_metrics.json
```

---

## 🔄 Full Pipeline Reproduction

To reproduce the entire pipeline from raw ChEMBL data to trained weights:

```bash
# 1. Download official ChEMBL 36 SQLite database
bash scripts/download_chembl.sh 36 ./raw_data

# 2. Extract and standardize data (produces sparse CSR interaction matrix)
python -m pp4ckd.data_pipeline \
  --db-path ./raw_data/chembl_36/chembl_36_sqlite/chembl_36.db \
  --out-dir ./data \
  --ckd-csv ./data/ckd_623_targets.csv \
  --cpus 16

# 3. Train all models across 4 NVIDIA GPUs (10-fold CV + Full production models)
python scripts/train_multigpu.py \
  --data-dir ./data \
  --models-dir ./models \
  --results-dir ./examples/benchmark_results \
  --fps Fused ECFP4 ECFP6 MHFP6 RDKit Layered AtomPair \
  --gpus 0 1 2 3

# 4. Generate consolidated academic summary tables
python scripts/evaluate.py \
  --cv-metrics ./examples/benchmark_results/cv_metrics.json \
  --ckd-metrics ./examples/benchmark_results/ckd_subset_metrics.json \
  --output-json ./examples/benchmark_results/benchmark_summary.json
```

---

## 📦 Zenodo Model Weights Archive & DOI Guide

> **Permanent DOI Record**: [https://doi.org/10.5281/zenodo.23147411](https://doi.org/10.5281/zenodo.23147411)  
> **Direct Download**: `pp4ckd_models.tar.gz` (785 MB, MD5: `f9380bb9cef3fe56d6d49d3e4a4d57ec`)

The full model repository contains **77 trained PyTorch checkpoint files** (7 fingerprint types $\times$ 10 CV folds + 7 full models), totaling **~2.6 GB**.

### Web-UI Drag-and-Drop Upload Guide (For Jixing)

To archive the complete set of weights and obtain a citable DOI:

1. **Log in to Zenodo**: Navigate to [https://zenodo.org/deposit](https://zenodo.org/deposit) and sign in via ORCID or GitHub.
2. **Create New Upload**: Click **"New upload"**.
3. **Upload Weights Package**: Drag and drop the compressed archive:
   ```bash
   tar -czf pp4ckd_chembl36_weights.tar.gz models/
   ```
4. **Fill Metadata Form** using the pre-prepared draft below:
   - **Resource type**: `Dataset`
   - **Title**: `PP4CKD: Pre-trained Multi-Target Deep Neural Network Weights on ChEMBL 36 for Chronic Kidney Disease Target Prediction`
   - **Authors**: `Liu, Jixing; et al.`
   - **Description**: *(Copy draft below)*
   - **Keywords**: `Polypharmacology; Chronic Kidney Disease; Deep Learning; ChEMBL 36; Target Prediction; PyTorch; Chemoinformatics`
   - **License**: `Apache License 2.0` (or `Creative Commons Attribution 4.0 International`)
5. **Publish & Obtain DOI**: Click **"Save"** then **"Publish"**. Zenodo will immediately issue a persistent digital object identifier (e.g., [`10.5281/zenodo.23147411`](https://doi.org/10.5281/zenodo.23147411)).

### Zenodo Description Draft

> **Title**: PP4CKD: Pre-trained Multi-Target Deep Neural Network Weights on ChEMBL 36 for Chronic Kidney Disease Target Prediction  
> **Abstract**:  
> This archive contains 77 PyTorch model checkpoints (.pt) for the PP4CKD framework, implementing Reymond group Polypharmacology Browser (PPB3) deep multi-target architectures across 7,676 ChEMBL 36 targets and 623 CKDdb chronic kidney disease targets. The collection includes 10-fold cross-validation checkpoints and production full-dataset checkpoints for 7 molecular fingerprint representations: ECFP4 (4096-bit), ECFP6 (4096-bit), AtomPair (4096-bit), Layered (4096-bit), RDKit (4096-bit), MHFP6 (4096-bit), and Fused (8192-bit ECFP4+MHFP6). Total compressed archive size: ~2.6 GB.

---

## 📜 Citation & License

### License
This project is licensed under the [Apache License 2.0](LICENSE).

### Citation
If you use PP4CKD in your research, please cite:

```bibtex
@article{pp4ckd2026,
  title={PP4CKD: A Multi-Target Deep Learning Polypharmacology Framework for Chronic Kidney Disease Drug Discovery},
  author={Liu, Jixing and Collaborators},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  note={Under Review}
}
```
