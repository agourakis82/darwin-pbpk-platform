# Fractal Dimension Analysis of Blood Cell Images - Proof of Concept

## 🎯 Hypothesis

**The fractal dimension (df) of blood microstructure correlates with pharmacokinetic
parameters, enabling image-based prediction of drug distribution.**

## ✅ Status: POC COMPLETE - AWAITING VALIDATION DATA

### Completed Tasks
- [x] Literature review complete
- [x] Dataset download (BCCD, NIH Malaria)
- [x] Fractal dimension calculation (box-counting)
- [x] Statistical analysis (t-test, Cohen's d)
- [x] Theoretical model development (df → h → PK)

### Pending (requires external data)
- [ ] Obtain paired image + PK data
- [ ] Empirical validation of df → h relationship
- [ ] Clinical validation study

## 🔬 Key Finding

**df_edge distinguishes pathological from normal cells (p < 0.001, d = -0.54)**

| Condition | df_edge | Interpretation |
|-----------|---------|----------------|
| Uninfected | 1.712 ± 0.036 | More complex boundaries |
| Parasitized | 1.691 ± 0.042 | Simpler boundaries (membrane altered) |

## 📐 Theoretical Model

```
df (from image) → h (heterogeneity) → PK parameters

h = α(2-df_edge) + β(2-df_dist) + γ|1-R|

k(t) = k₀ × t^(-h)  [Kopelman, 1986]
```

See `THEORETICAL_MODEL.md` for full derivation.


## 📁 Files

```
analysis/fractal_poc/
├── README.md                    # This file
├── THEORETICAL_MODEL.md         # Full theoretical derivation
├── RESULTS_SUMMARY.md           # Summary of findings
│
├── Core Algorithms:
│   ├── fractal_dimension.py     # Box-counting algorithm
│   ├── advanced_fractal.py      # Cell distribution analysis
│   └── fractal_pk_model.py      # Theoretical PK model
│
├── Analysis Scripts:
│   ├── run_poc.py               # Download BCCD + basic analysis
│   ├── run_advanced_poc.py      # Cell distribution analysis
│   ├── malaria_fractal_analysis.py  # Malaria comparison
│   ├── compare_datasets.py      # Multi-dataset comparison
│   └── demo_theoretical_model.py    # Model demonstration
│
├── Data Scripts:
│   ├── download_malaria_nih.py  # Download NIH malaria dataset
│   ├── download_pathological.py # Download pathological datasets
│   └── download_leukemia_kaggle.py  # Kaggle leukemia dataset
│
├── data/                        # Downloaded datasets
│   ├── BCCD_Dataset-master/     # 364 normal blood smears
│   ├── malaria_cells/           # 27,558 cells (infected+normal)
│   └── synthetic_pathological/  # 20 test images
│
└── results/                     # Analysis results (JSON)
    ├── fractal_analysis_results.json
    ├── advanced_fractal_results.json
    ├── dataset_comparison.json
    └── malaria_fractal_analysis.json
```

## 📚 References

1. Kopelman R. (1986) "Rate Processes on Fractals" - J. Stat. Phys. 42:185-200
2. Kopelman R. (1988) "Fractal Reaction Kinetics" - Science 241:1620-1626
3. Jung et al. (2023) "Fractal Kinetic Implementation in Population PK" - Pharmaceutics 15:304
4. NIH Malaria Dataset - Rajaraman et al. (2018)

## 🔮 Future Work

To move from theoretical to validated model:

1. **Collaboration**: Partner with clinical center for paired image+PK data
2. **Validation Study**: N≥50 subjects, healthy vs pathological
3. **Model Calibration**: Fit α, β, γ coefficients empirically
4. **Integration**: Add to Darwin platform as experimental feature

## ⚠️ Disclaimer

**This is a theoretical proof-of-concept.**
Predictions from this model should NOT be used for clinical decisions
until validated with empirical data.

