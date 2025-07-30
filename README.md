# EpiLingo: Large language model for highly multiplexed detection of multi-omics epigenetic modifications on single-base/residue resolution sequencing data
Epigenetic modifications, including DNA methylation, RNA modifications, and protein post-translational modifications (PTMs), play pivotal roles in plant development and environmental adaptation. However, current experimental and computational approaches are typically restricted to a single omics layer, modification type, or species, limiting their ability to support comprehensive regulatory analysis. Here, we introduce EpiLingo, a Transformer-based large language model framework that enables accurate prediction of 11 types of epigenetic modifications across DNA, RNA, and PTM layers in plants, using only primary sequence information without requiring structural inputs. By incorporating species-aware contextual encoding and task-specific decoders, EpiLingo achieves strong generalization performance across 31 prediction tasks spanning 15 plant species. This represents the first comprehensive prediction framework for plant epigenetic and epitranscriptomic modifications. Beyond enhancing predictive accuracy, EpiLingo reveals conserved sequence patterns and modification logic, offering a universal computational framework for decoding plant epigenetic regulation and promoting the integration of multi-omics data in precision plant science. 

![image](workflow.png)
### 🔍 Key Features
Cross-layer coverage: Predicts DNA methylation (4mC, 6mA), RNA modifications (m6A, m5C, Ψ), and protein post-translational modifications (e.g., lysine lactylation, crotonylation, phosphorylation, etc.)

Plant-specific: Fine-tuned across 15 plant species with species-aware tokenization and embedding

Transformer-based architecture: DNA/RNA models based on [DNABERT](https://github.com/jerryji1993/DNABERT); Protein models based on [ProteinBERT](https://github.com/nadavbra/protein_bert)

### 📁 Folder Structure
```
EpiLingo/
├── dna_rna/                # DNA & RNA modification prediction (based on DNABERT)
│   ├── run_dnarnamod.py    # Main training script
│   ├── data/               # Contains benchmark datasets
│   └── model/              # DNABERT finetuned weights
├── ptm/                    # PTM site prediction (based on ProteinBERT)
│   ├── run_ptmmod.py       # Protein model training and evaluation
│   ├── data/               # Protein sequence + label data
│   └── model/              # ProteinBERT finetuned weights
├── utils/                  # Shared preprocessing and evaluation functions
├── requirements.txt        # Required packages
└── README.md
```
### 🚀 How to Use (Detailed)
1. Install dependencies (recommended in a virtual environment):
```
pip install -r requirements.txt --ignore-installed
```
### 🧠 Model description
