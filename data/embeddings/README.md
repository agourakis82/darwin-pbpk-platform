# Darwin PBPK Platform - Training Datasets v1.0.0

**DOI:** [To be assigned after Zenodo upload]

## 📦 Contents

This dataset contains the processed training data for Darwin PBPK Platform v1.0.0:

### Files

1. **consolidated_pbpk_v1.parquet** (~1.5 MB)
   - Processed PBPK data for 44,779 compounds
   - Columns: SMILES, Fu, Vd, CL, scaffold, split (train/val/test)
   - Format: Parquet (pandas-readable)

2. **chemberta_embeddings_consolidated.npz** (~123 MB)
   - ChemBERTa embeddings (768-dimensional)
   - Pre-trained model: `seyonec/ChemBERTa-zinc-base-v1`
   - Format: NumPy compressed (.npz)
   - Shape: (44,779, 768)

3. **molecular_graphs.pkl** (~286 MB)
   - Molecular graphs in PyTorch Geometric format
   - 20 node features + 7 edge features
   - Format: Pickle (Python)
   - Total size: 44,779 graphs

**Total size:** ~410 MB (uncompressed)

## 🔬 Dataset Details

### Sources
- **ChEMBL:** Bioactivity and pharmacokinetic data
- **TDC (Therapeutics Data Commons):** ADMET benchmark datasets
- **KEC:** Curated literature extractions

### Split Strategy
- **Train:** 35,823 compounds (80%)
- **Validation:** 4,477 compounds (10%)
- **Test:** 4,479 compounds (10%)
- **Scaffold-based split:** Zero molecular leakage (23,806 unique scaffolds)

### PBPK Parameters
- **Fu (Fraction unbound):** 0.0 - 1.0
- **Vd (Volume of distribution):** L/kg (positive)
- **CL (Clearance):** L/h/kg (positive)

## 📖 Usage

### Loading the Data

```python
import pandas as pd
import numpy as np
import pickle
import torch

# Load parquet
df = pd.read_parquet('consolidated_pbpk_v1.parquet')

# Load embeddings
embeddings = np.load('chemberta_embeddings_consolidated.npz')
chemberta_emb = embeddings['embeddings']  # Shape: (44779, 768)

# Load graphs
with open('molecular_graphs.pkl', 'rb') as f:
    graphs = pickle.load(f)  # List of PyG Data objects
```

### Training Scripts

See the main repository for training scripts:
- `apps/training/01_baseline_mlp.py` - Baseline MLP model
- `apps/training/02_gnn_model.py` - GNN model with attention

## 🔗 Related Resources

- **Software Repository:** https://github.com/agourakis82/darwin-pbpk-platform
- **Software DOI:** https://doi.org/10.5281/zenodo.17536674
- **Documentation:** See `docs/` directory in main repository

## 📄 License

CC-BY-4.0 - See LICENSE file

## 🙏 Citation

If you use this dataset, please cite both the software and the dataset:

**Software:**
```
Agourakis, D.C. (2025). Darwin PBPK Platform: AI-Powered Pharmacokinetic 
Prediction. Version 1.0.0 [Software]. Zenodo. 
https://doi.org/10.5281/zenodo.17536674
```

**Dataset:**
```
Agourakis, D.C. (2025). Darwin PBPK Platform - Training Datasets v1.0.0 
[Dataset]. Zenodo. https://doi.org/10.5281/zenodo.YYYYYY
```

## 📧 Contact

For questions or issues, please open an issue on GitHub:
https://github.com/agourakis82/darwin-pbpk-platform/issues

---

**"Rigorous science. Honest results. Real impact."**

