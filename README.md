# 💊 Darwin PBPK Platform

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17536674.svg)](https://doi.org/10.5281/zenodo.17536674)

**"Ciência rigorosa. Resultados honestos. Impacto real."**

## AI-Powered PBPK Prediction Platform

State-of-the-art deep learning platform for physiologically-based pharmacokinetic (PBPK) parameter prediction using multi-modal molecular representations.

### 🚀 Features

- ✅ **Multi-Modal Embeddings:** ChemBERTa 768d + Molecular graphs + RDKit descriptors
- ✅ **Advanced Architectures:** GNN (GAT + TransformerConv), Multi-task learning
- ✅ **Large Dataset:** 44,779 compounds (ChEMBL + TDC + KEC)
- ✅ **3 PBPK Parameters:** Fraction unbound (Fu), Volume of distribution (Vd), Clearance (CL)
- ✅ **PhysioQM Integration:** Physics-informed constraints
- ✅ **Production Ready:** Trained models, API endpoints

### 📊 Performance

- **Baseline MLP:** R² > 0.30 (ChemBERTa only)
- **GNN Model:** R² > 0.45 (Graphs + Attention)
- **Ensemble:** R² > 0.55 (Multi-modal fusion)

### 🧬 Architecture

**Embeddings:**
- ChemBERTa: `seyonec/ChemBERTa-zinc-base-v1` (768d)
- Pre-trained on ~100M molecules (ZINC, PubChem)

**Graphs:**
- PyTorch Geometric
- 20 node features (atom type, charge, aromaticity, etc.)
- 7 edge features (bond type, conjugation, ring, etc.)

**Descriptors:**
- RDKit: 25 molecular descriptors
- MW, LogP, TPSA, QED, HBA, HBD, etc.

**Models:**
- GAT: 4 attention heads, 3 layers
- TransformerConv: 4 heads, 3 layers
- Multi-task: Weighted loss (Fu, Vd, CL)

### 📚 Citation

If you use this software in your research, please cite:

```
Agourakis, D.C. (2025). Darwin PBPK Platform: AI-Powered Pharmacokinetic 
Prediction. Version 1.0.0 [Software]. Zenodo. 
https://doi.org/10.5281/zenodo.17536674
```

### 📖 Dataset

Large datasets (embeddings, graphs, ~1.7 GB) available at:
- **DOI:** https://doi.org/10.5281/zenodo.17541874 (to be published)
- **Contents:** ChemBERTa embeddings, molecular graphs, processed parquets

### 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train baseline MLP
python apps/training/01_baseline_mlp.py

# Train GNN model
python apps/training/02_gnn_model.py

# Make predictions (after training)
python apps/prediction/pbpk_predictor.py --smiles "CCO"
```

### 📊 Data Sources

- **ChEMBL:** Bioactivity and PK data
- **TDC (Therapeutics Data Commons):** ADMET benchmark datasets
- **KEC:** Curated literature extractions

### 📄 License

MIT License - See [LICENSE](LICENSE) file

### 🙏 Acknowledgments

Developed for computational drug discovery with Q1 scientific rigor at PUCRS.

---

**"Rigorous science. Honest results. Real impact."**

