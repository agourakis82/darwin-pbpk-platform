# Fractal Analysis of Blood Cell Images - PoC Results

## Executive Summary

This Proof of Concept demonstrates that **fractal dimension analysis can detect 
pathological changes in blood cell images** with statistical significance.

## Key Finding

**The fractal dimension of cell edges (df_edge) significantly differs between
malaria-infected and uninfected red blood cells.**

| Metric | Parasitized | Uninfected | Cohen's d | p-value |
|--------|-------------|------------|-----------|---------|
| df_edge | 1.691 ± 0.042 | 1.712 ± 0.036 | -0.54 | < 0.001 |
| area_ratio | 0.301 ± 0.063 | 0.276 ± 0.053 | +0.44 | < 0.001 |

## Biological Interpretation

1. **Infected cells have LOWER df_edge** (simpler, smoother boundaries)
2. This is consistent with **membrane distortion** caused by malaria parasite
3. Altered membrane properties → Changed **permeability** and **deformability**

## Connection to Pharmacokinetics

### Theoretical Framework (Kopelman, 1986)

In heterogeneous/fractal media, reaction rate constants are time-dependent:

```
k(t) = k₀ × t^(-h)

where h = 1 - (d_s/2) and d_s is the spectral dimension
```

### Proposed Connection

```
df_edge → membrane heterogeneity → permeability → drug diffusion → PK parameters
```

If blood microstructure (captured by df) correlates with tissue heterogeneity,
then df could predict the fractal exponent h in PK models.

## Datasets Used

1. **BCCD Dataset** - Normal blood smears (364 images)
2. **NIH Malaria Dataset** - Parasitized (13,780) + Uninfected (13,780) cells

## Methodology

1. Image segmentation and edge detection (Sobel filter)
2. Box-counting algorithm for fractal dimension
3. Statistical comparison (t-test, Cohen's d effect size)

## Limitations

1. No direct PK data paired with images (yet)
2. Malaria is not representative of all pathologies
3. Need validation with other diseases (leukemia, sepsis)

## Next Steps

1. **Obtain paired data**: Images + PK parameters from same patients
2. **Develop theoretical model**: df → h → PK parameters
3. **Validate with leukemia data**: ALL-IDB or C-NMC datasets
4. **Clinical collaboration**: Partner with hospital/research center

## Files

- `fractal_dimension.py` - Core box-counting algorithm
- `advanced_fractal.py` - Cell distribution analysis
- `malaria_fractal_analysis.py` - Malaria dataset analysis
- `results/malaria_fractal_analysis.json` - Full results

## Citation

If using this analysis, cite:
- Kopelman R. (1986) J. Stat. Phys. 42:185-200 (Fractal kinetics)
- Jung et al. (2023) Pharmaceutics 15:304 (Fractal PK models)
- NIH Malaria Dataset (Rajaraman et al., 2018)

## Status

✅ **HYPOTHESIS SUPPORTED** - Fractal dimension detects pathological changes
⚠️ **CONNECTION TO PK UNVALIDATED** - Requires paired image+PK data

