# Ablation Study Results: Do Fractal Features Help?

**Date:** November 2025  
**Conclusion:** NO SIGNIFICANT IMPROVEMENT DETECTED

---

## Executive Summary

We conducted rigorous ablation studies to answer:
> Do fractal features improve Vdss prediction compared to standard physicochemical features?

**Answer: We cannot demonstrate that they do.**

---

## Experimental Design

### Conditions
1. **Baseline:** 12 standard physicochemical features (MW, logP, logD, TPSA, HBD, HBA, RB, fup, etc.)
2. **Fractal:** 24 features (baseline + 12 fractal features including molecular fractal dimension, coupling coefficients, topological entropy)

### Protocol
- Dataset: Lombardo 1352 (862 valid compounds after filtering)
- Cross-validation: 5-fold
- Seeds: 10 independent random seeds
- Architecture: 3-layer MLP (64→32→1)
- Optimizer: Adam (lr=0.002, λ=0.0003)
- Epochs: 500
- Statistical test: Paired t-test across 50 fold-seed combinations

---

## Results

### Study 1: Simple SGD (Baseline)

| Condition | GMFE | % 2-fold |
|-----------|------|----------|
| Baseline | 2.651 ± 0.164 | 44.2% |
| Fractal | 2.653 ± 0.167 | 44.0% |
| **p-value** | **0.93** | **0.78** |

Both conditions performed poorly with SGD.

### Study 2: Adam Optimizer

| Condition | GMFE | Std | % 2-fold | Best Fold |
|-----------|------|-----|----------|-----------|
| Baseline | 2.35 | 1.43 | 56.0% | **1.001** |
| Fractal | 2.23 | 1.21 | 58.0% | 1.018 |
| **p-value** | **0.37** | - | **0.77** | - |

### Interpretation

1. **No statistically significant difference** between baseline and fractal conditions
2. **High variance** (std > 1.0) dominates any potential signal
3. **Both conditions can achieve GMFE ~1.0** on individual folds
4. The optimization landscape, not the features, determines success

---

## What This Means

### The Fractal Features Are NOT Proven to Help

Despite theoretical justification:
- Alexander-Orbach spectral dimension
- Molecular-tissue fractal coupling
- Mittag-Leffler response functions

**The empirical data does not show improvement.**

Possible explanations:
1. **Features don't capture the right physics** - our formulation may be wrong
2. **Effect is too small** - real but below detection threshold
3. **Redundancy** - fractal features may be correlated with baseline features
4. **Optimization swamps signal** - high variance obscures any benefit

### What We Learned

The key finding is that **training instability is the dominant problem**:

```
Seed 1: GMFE = 1.48 (good)
Seed 2: GMFE = 3.84 (bad)
Seed 5: GMFE = 1.33 (good)
```

The same model architecture with same features varies by 3x depending on random seed. This is unacceptable for a scientific tool.

---

## Recommendations

### 1. Fix Optimization First

Before adding new features, we need stable training:
- Ensemble predictions (average multiple seeds)
- Learning rate warmup and decay
- Better initialization schemes
- Bayesian neural networks for uncertainty

### 2. Revisit Fractal Features

If stability is achieved and we still want to test fractal features:
- Try different formulations of molecular fractal dimension
- Use actual tissue-specific features (not averaged)
- Incorporate fractal features as physics constraints, not inputs

### 3. Consider Simpler Approaches

Given baseline features can achieve GMFE ~1.0 on good seeds:
- Random Forest with physicochemical features
- XGBoost ensemble
- Linear model with carefully selected features

These may be more stable than neural networks for this dataset size.

---

## Raw Data

### Per-Seed Results (Adam)

| Seed | Baseline GMFE | Fractal GMFE | Δ |
|------|---------------|--------------|---|
| 1 | 1.482 | 1.666 | -0.184 |
| 2 | 3.840 | 3.182 | +0.658 |
| 3 | 2.768 | 2.563 | +0.204 |
| 4 | 1.630 | 1.744 | -0.114 |
| 5 | 1.333 | 1.689 | -0.356 |
| 6 | 2.371 | 2.156 | +0.215 |
| 7 | 2.001 | 1.980 | +0.021 |
| 8 | 2.609 | 2.003 | +0.605 |
| 9 | 2.748 | 2.687 | +0.061 |
| 10 | 2.737 | 2.628 | +0.110 |

Notice: 6/10 seeds show fractal features slightly better, 4/10 show baseline better. This is consistent with random variation.

---

## Conclusion

**Honest science requires reporting negative results.**

Our fractal pharmacokinetics theory is mathematically elegant and biologically plausible, but we have not demonstrated empirical benefit. The theory may still be correct - we simply cannot prove it with current methods.

The path forward is:
1. Achieve stable, reproducible training
2. Re-evaluate features with proper statistical power
3. Consider external validation

---

*This document represents honest scientific reporting. We state what we found, not what we hoped to find.*
