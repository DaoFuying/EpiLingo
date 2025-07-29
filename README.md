# EpiLingo: Large language model for highly multiplexed detection of multi-omics epigenetic modifications on single-base/residue resolution sequencing data
Epigenetic modifications, including DNA methylation, RNA modifications, and protein post-translational modifications (PTMs), play pivotal roles in plant development and environmental adaptation. However, current experimental and computational approaches are typically restricted to a single omics layer, modification type, or species, limiting their ability to support comprehensive regulatory analysis. Here, we introduce EpiLingo, a Transformer-based large language model framework that enables accurate prediction of 11 types of epigenetic modifications across DNA, RNA, and PTM layers in plants, using only primary sequence information without requiring structural inputs. By incorporating species-aware contextual encoding and task-specific decoders, EpiLingo achieves strong generalization performance across 31 prediction tasks spanning 15 plant species. This represents the first comprehensive prediction framework for plant epigenetic and epitranscriptomic modifications. Beyond enhancing predictive accuracy, EpiLingo reveals conserved sequence patterns and modification logic, offering a universal computational framework for decoding plant epigenetic regulation and promoting the integration of multi-omics data in precision plant science. 

![image](workflow.png)

### 📁 EpiLingo Structure
```
EpiLingo/ 
├── BertADP.py # Main execution script 
├── BertADP/ # Directory containing the fine-tuned model
│ └── adapter_config.json
│ └── adapter_model.safetensors
├── example/ 
│ └── example.csv # Example input file with 10 sequences (5 positive, 5 negative) 
└── requirements.txt # List of required Python packages
```
