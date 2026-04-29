# EpiLingo

> A unified multi-omics framework for highly multiplexed prediction of epigenetic modifications across DNA, RNA, and protein sequences.

---

## Table of Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Framework](#framework)
- [Supported Tasks](#supported-tasks)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Format](#data-format)
- [Training](#training)
- [Benchmarking](#benchmarking)
- [Interpretation and Explainability](#interpretation-and-explainability)
- [Outputs](#outputs)
- [Notes](#notes)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

**EpiLingo** is a unified sequence modeling framework for predicting epigenetic modification sites across **DNA, RNA, and protein** modalities.  
Unlike conventional predictors that are usually restricted to a single modification type, species, or omics layer, EpiLingo introduces a **shared multi-omics modeling architecture** that integrates heterogeneous biological sequences into a common latent space and refines them through **hierarchical conditional adaptation**.

The framework is designed for:

- **multi-omics epigenetic modification prediction**
- **multi-task learning across modification types**
- **cross-species generalization**
- **highly multiplexed sequence modeling**
- **systematic interpretability and benchmarking**

---

## Highlights

- Unified modeling across **DNA, RNA, and protein**
- Shared latent representation with **modality-specific projection**
- **Hierarchical conditional adapters**
  - omics adapter
  - species adapter
  - task adapter
- **Asymmetric context pooling** for upstream / center / downstream modeling
- Flexible training and evaluation pipelines for:
  - unified model benchmarking
  - cross-dataset evaluation
  - cross-species transfer analysis
  - ablation studies
- Interpretable analysis modules for:
  - latent space visualization
  - hierarchical adaptation analysis
  - position importance analysis
  - attribution analysis
  - motif discovery

---

## Framework

The core EpiLingo architecture consists of five major components:

1. **Backbone encoders**
   - DNA/RNA backbone: DNABERT-style encoder
   - Protein backbone: ProtBERT-style encoder

2. **Projection into a shared latent space**
   - modality-specific features are aligned into a unified hidden representation

3. **Hierarchical conditional adaptation**
   - omics-aware adaptation
   - species-aware adaptation
   - task-aware adaptation

4. **Asymmetric context pooling**
   - upstream, center, and downstream sequence contexts are explicitly modeled

5. **Conditional decoder**
   - final prediction is conditioned on omics, species, and task identity

---

## Supported Tasks

### DNA modifications
- 4mC
- 6mA

### RNA modifications
- m5C
- m6A
- Psi

### Protein PTMs
- Kac
- Kcr
- Phos
- Khib
- Ksu
- S-Nitro

---

## Repository Structure

```text
EpiLingo/
├── README.md
├── requirements.txt
├── data/
├── checkpoints/
├── outputs/
├── core/
│   └── EpilingoHierarchicalFramework.py
├── benchmarking/
│   ├── cross_species_eval_with_unified.py
│   ├── ablation_eval.py
│   └── unified_group_eval.py
├── interpretation/
│   ├── extract_latent_space.py
│   ├── analyze_hierarchical_adaptation.py
│   ├── analyze_position_importance.py
│   ├── plot_position_importance.py
│   ├── analyze_dna_rna_motifs.py
│   ├── analyze_attribution_dna_rna.py
│   ├── plot_attribution_dna_rna.py
│   ├── analyze_attribution_protein.py
│   └── plot_attribution_protein.py
└── docs/
```

---

## Installation

Create a clean Python environment and install dependencies:

```bash
conda create -n epilingo python=3.10
conda activate epilingo
pip install -r requirements.txt
```

Typical dependencies include:

- `torch`
- `transformers`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `openpyxl`
- `umap-learn`
- `logomaker`

---

## Data Format

Input files should be organized as:

```text
data/
├── DNA/
├── RNA/
└── PTM/
```

Each dataset file should contain two columns:

```text
label,seq
```

or

```text
label\tseq
```

### Supported naming pattern

```text
<Task>_<species>_train.csv
<Task>_<species>_dev.csv
```

Examples:

```text
6mA_arabidopsis_train.tsv
6mA_arabidopsis_dev.tsv
m6A_arabidopsis_train.tsv
Kac_oryza_train.csv
Phos_arabidopsis_dev.csv
```

---

## Training

Main training script:

```bash
python core/EpilingoHierarchicalFramework.py
```

This script supports:

- unified training across DNA, RNA, and protein
- shared vocabulary construction
- per-dataset evaluation
- ROC / PR curve data export

---

## Benchmarking

### 1. Cross-species / cross-dataset evaluation with unified models

```bash
python benchmarking/cross_species_eval_with_unified.py \
  --framework_path core/EpilingoHierarchicalFramework.py \
  --data_root ./data \
  --output_dir ./outputs/cross_species_with_unified_results
```

This script produces:

- DNA/RNA cross-dataset matrices
- Protein cross-dataset matrices
- unified DNA/RNA model results
- unified protein model results

---

### 2. Ablation analysis

```bash
python benchmarking/ablation_eval.py \
  --framework_path core/EpilingoHierarchicalFramework.py \
  --data_root ./data \
  --output_dir ./outputs/ablation_results
```

The ablation study evaluates the contribution of:

- projection to shared latent space
- omics adapter
- species adapter
- task adapter
- asymmetric context pooling
- conditional decoder

---


## Interpretation and Explainability

### 1. Latent space extraction

```bash
python interpretation/extract_latent_space.py \
  --framework_path core/EpilingoHierarchicalFramework.py \
  --checkpoint_path checkpoints/model.pt \
  --data_root ./data \
  --output_dir ./outputs/latent_space \
  --split valid \
  --omics all
```

---

### 2. Hierarchical adaptation analysis

```bash
python interpretation/analyze_hierarchical_adaptation.py \
  --framework_path core/EpilingoHierarchicalFramework.py \
  --checkpoint_path checkpoints/model.pt \
  --data_root ./data \
  --output_dir ./outputs/hierarchical_adaptation \
  --split valid \
  --omics all
```

This analysis compares representation geometry across stages:

- backbone
- projection
- omics adaptation
- species adaptation
- task adaptation

using metrics such as:

- inter-class distance
- intra-class compactness
- silhouette score
- Davies–Bouldin index

---

### 3. Position importance analysis

```bash
python interpretation/analyze_position_importance.py \
  --framework_path core/EpilingoHierarchicalFramework.py \
  --checkpoint_path checkpoints/model.pt \
  --data_root ./data \
  --output_dir ./outputs/position_importance \
  --split valid \
  --omics dna_rna
```

Plotting:

```bash
python interpretation/plot_position_importance.py \
  --input_dir ./outputs/position_importance
```

---

### 4. DNA/RNA motif and attribution analysis

```bash
python interpretation/analyze_dna_rna_motifs.py \
  --framework_path core/EpilingoHierarchicalFramework.py \
  --checkpoint_path checkpoints/model.pt \
  --data_root ./data \
  --output_dir ./outputs/motif_analysis \
  --split valid \
  --task_name 6mA
```

```bash
python interpretation/analyze_attribution_dna_rna.py \
  --framework_path core/EpilingoHierarchicalFramework.py \
  --checkpoint_path checkpoints/model.pt \
  --data_root ./data \
  --output_dir ./outputs/attr_dna_rna \
  --split valid
```

```bash
python interpretation/plot_attribution_dna_rna.py \
  --input_dir ./outputs/attr_dna_rna
```

---

### 5. Protein attribution analysis

```bash
python interpretation/analyze_attribution_protein.py \
  --framework_path core/EpilingoHierarchicalFramework.py \
  --checkpoint_path checkpoints/model.pt \
  --data_root ./data \
  --output_dir ./outputs/attr_protein \
  --split valid
```

```bash
python interpretation/plot_attribution_protein.py \
  --input_dir ./outputs/attr_protein
```

---

## Outputs

Typical outputs include:

- `metrics.csv`
- `raw_predictions.csv`
- `*_roc.csv`
- `*_pr.csv`
- `*_matrix.csv`
- `features.npy`
- `metadata.csv`
- attribution summaries
- motif fragment files
- publication-ready plots

---

## Notes

- DNA/RNA and protein backbones use different pretrained encoders.
- DNA/RNA inputs are tokenized in **6-mer format**.
- Protein inputs are modeled at the **residue level**.
- For offline HPC environments, pretrained backbone checkpoints may need to be downloaded in advance.
- When analyzing all omics jointly, raw backbone representations may not be directly comparable before projection because DNA/RNA and protein backbones can have different hidden dimensions.

---

## Citation

If you use EpiLingo in your work, please cite the corresponding manuscript.

```bibtex
@article{epilingo_placeholder,
  title   = {EpiLingo},
  author  = {Authors},
  journal = {To be updated},
  year    = {2026}
}
```

---

## Contact

For questions, bug reports, or collaboration interests, please open an issue or contact the authors.
