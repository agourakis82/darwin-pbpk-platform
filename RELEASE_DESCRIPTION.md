# 💊 Darwin PBPK Platform v1.0.0 - Production Release

**"Ciência rigorosa. Resultados honestos. Impacto real."**

## 🚀 Features

### Core Architecture
- ✅ **Multi-modal molecular representations**
  - ChemBERTa embeddings (768d, pre-trained on 100M molecules)
  - Molecular graphs (PyTorch Geometric, 20 node + 7 edge features)
  - RDKit descriptors (25 molecular features)
- ✅ **Advanced GNN architectures**
  - GAT (Graph Attention Networks, 4 heads, 3 layers)
  - TransformerConv (4 heads, 3 layers)
  - Attention-based pooling
- ✅ **Multi-task learning**
  - Fraction unbound (Fu)
  - Volume of distribution (Vd)
  - Clearance (CL)
  - Weighted loss function for imbalanced data

### Dataset
- ✅ **44,779 compounds**
  - ChEMBL: Bioactivity and pharmacokinetic data
  - TDC (Therapeutics Data Commons): ADMET benchmarks
  - KEC: Curated literature extractions
- ✅ **Scaffold-based split** (zero molecular leakage)
  - Train: 35,823 (80%)
  - Validation: 4,477 (10%)
  - Test: 4,479 (10%)
  - 23,806 unique scaffolds

### Performance Targets
- **Baseline MLP:** R² > 0.30 (ChemBERTa embeddings only)
- **GNN Model:** R² > 0.45 (Graph-based with attention)
- **Ensemble:** R² > 0.55 (Multi-modal fusion)

### Advanced Features
- ✅ **PhysioQM integration:** Physics-informed constraints
- ✅ **Evidential uncertainty:** Quantify prediction confidence
- ✅ **KEC-PINN:** Entropy-Curvature-Coherence informed learning
- ✅ **Multi-modal fusion:** Optimal combination of representations

## 📊 Code Statistics

- **Files:** 56
- **Lines of Python code:** 7,601
- **Total lines:** 14,826 (including docs)
- **Modules:** 30+ Python modules
- **Scripts:** Training pipeline, data processing, validation

## 🧬 Technical Specifications

**Model Parameters:**
- Baseline MLP: ~560K parameters
- GNN Model: ~1.6M parameters
- Input features: 820 total (768 ChemBERTa + 27 Graph + 25 RDKit)

**Training Configuration:**
- Batch size: 256 (MLP), 128 (GNN)
- Learning rate: 3e-4 (MLP), 1e-4 (GNN)
- Optimizer: AdamW with weight decay
- Scheduler: ReduceLROnPlateau
- Gradient clipping: max_norm=1.0

**Data Handling:**
- Missing data support (82% Fu, 81% Vd, 10% CL)
- Logit/log1p transforms for bounded/positive values
- Weighted multi-task loss for imbalanced targets

## 📚 Citation

If you use this software in your research, please cite:

```
Agourakis, D.C. (2025). Darwin PBPK Platform: AI-Powered Pharmacokinetic 
Prediction. Version 1.0.0 [Software]. Zenodo. 
https://doi.org/10.5281/zenodo.17536674
```

## 📖 Data Availability

Large training datasets (1.7 GB) available at separate Zenodo record:
- **DOI:** https://doi.org/10.5281/zenodo.YYYYYY
- **Contents:** 
  - ChemBERTa embeddings (1 GB, 44,779 molecules × 768d)
  - Molecular graphs (500 MB, PyG format)
  - Processed parquets (100 MB, splits + descriptors)

## 🔬 Validation

- **Scaffold split:** Zero leakage (23,806 unique scaffolds)
- **Cross-validation:** 10-fold on validation set
- **Test set:** Held-out for final evaluation
- **Uncertainty:** Evidential deep learning for calibrated confidence

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Acknowledgments

Developed for computational drug discovery with Q1 scientific rigor at PUCRS. Targets publication in Nature Machine Intelligence and Journal of Medicinal Chemistry.

---

**"Rigorous science. Honest results. Real impact."**

