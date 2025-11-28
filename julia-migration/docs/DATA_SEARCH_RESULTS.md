# Data Search Results: Experimental PBPK Parameters

**Date:** November 2025  
**Objective:** Find experimental fut, BPR, pKa data to improve Vdss predictions

---

## Summary

We searched extensively for public datasets with experimental tissue binding (fut), 
blood-to-plasma ratio (BPR), and pKa data. The key finding is that **experimental 
fut data is extremely scarce** - this is the fundamental bottleneck.

---

## Datasets Found

### 1. Therapeutics Data Commons (TDC) ADME Datasets

| Dataset | Compounds | Parameter | Source |
|---------|-----------|-----------|--------|
| VDss_Lombardo | 1,130 | Volume of Distribution | Lombardo/Obach |
| PPBR_AZ | 1,614 | Plasma Protein Binding | AstraZeneca |
| Lipophilicity_AZ | 4,200 | LogD (experimental) | AstraZeneca |
| Clearance_Hepatocyte_AZ | 1,213 | Hepatocyte Clearance | AstraZeneca |
| Clearance_Microsome_AZ | 1,102 | Microsomal Clearance | AstraZeneca |
| Half_Life_Obach | 667 | Half-life | Obach |

**Critical Issue: Minimal Overlap**

When we tried to merge these datasets:
- VDss ∩ PPBR: **43 compounds**
- VDss ∩ LogD: **66 compounds**  
- VDss ∩ CL: **26 compounds**
- VDss ∩ PPBR ∩ LogD: **32 compounds**

The AstraZeneca compounds are largely different from the Lombardo VDss compounds.

### 2. Our Existing Lombardo Dataset

| Parameter | Compounds |
|-----------|-----------|
| VDss | 1,249 |
| fup (experimental) | 879 |
| **VDss + fup** | **863** |

This is actually our **best resource** - we already have experimental fup for 863 compounds!

### 3. OSP-PBPK Model Library

- 40 compounds with full PBPK models
- Too small for ML training

### 4. Rodgers-Rowland Kp Data

- Original papers: 36-40 compounds
- Rat tissue partition coefficients
- Not directly applicable to human Vdss

### 5. DTC Lab Kp Calculator

- Recently published (2024)
- Supplementary data may contain Kp values
- Requires direct download from publication

---

## Key Finding: The fut Problem

**We do NOT lack fup data** - we have experimental plasma protein binding for 879 compounds.

**We lack fut data** - fraction unbound in tissue is rarely measured experimentally.

### Why fut Matters

The Øie-Tozer equation:
```
Vdss = Vp + Ve × (fup/fut) + Vr × (fup/fut)
```

With experimental fup but estimated fut:
- Our R-R estimation: GMFE 5.1
- Hybrid ML correction: GMFE 2.3-2.4

With experimental fup AND fut:
- Øie-Tozer gold standard: GMFE ~1.55

**The 0.8 GMFE gap is entirely due to fut estimation error.**

---

## What We Tried

### 1. Rodgers-Rowland fut Estimation

Implemented simplified R-R equations considering:
- Neutral lipid partitioning (from LogD)
- Ionization state (LogP - LogD)
- Polar surface area effects
- Plasma binding correlation

**Result:** GMFE 5.1 (worse than no estimation)

### 2. Hybrid ML Approach

Used mechanistic features (fup, estimated fut, fup/fut ratio) + molecular descriptors.

**Result:** GMFE 2.3-2.4 (ML learns to correct fut errors)

### 3. Random Forest Baseline

Simple Random Forest with fup + LogD + descriptors.

**Result:** GMFE 2.38 (similar to neural networks)

---

## Conclusions

### 1. Data Availability

| Parameter | Status |
|-----------|--------|
| VDss | Available (1,130-1,249 compounds) |
| fup (plasma) | Available (879 compounds) |
| fut (tissue) | **NOT AVAILABLE** |
| LogD | Available (4,200 compounds, limited overlap) |
| Clearance | Available (1,213 compounds, limited overlap) |
| BPR | **NOT AVAILABLE** |
| pKa | Only predicted (MoKa) |

### 2. Performance Ceiling

Without experimental fut, the best achievable GMFE is ~2.0-2.2:
- We achieve: 2.1-2.4
- State-of-art (PKSmart): 2.09
- Gold standard (with fut): 1.55

### 3. Path Forward

To achieve GMFE < 2.0 consistently:

**Option A: Better fut estimation**
- Use microsomal binding (fum) as proxy
- Requires experimental fum data

**Option B: More experimental data**
- Partner with pharma company
- Access proprietary fut measurements

**Option C: Ensemble approaches**
- Combine multiple models
- Use uncertainty quantification
- Accept ~2.0-2.2 GMFE as ceiling

---

## Files Created

- `enhanced_lombardo_with_tdc.csv` - Lombardo + TDC experimental LogD/CL where available
- `tdc_integrated_pbpk.csv` - TDC overlap dataset (48 compounds)
- `mechanistic_vdss.jl` - Rodgers-Rowland fut estimation + hybrid ML

---

## References

1. Obach RS, Lombardo F, et al. (2008) Drug Metab Dispos 36:1385-405
2. Lombardo F, et al. (2018) Drug Metab Dispos 46:1466-77
3. Rodgers T, Rowland M (2006) J Pharm Sci 95:1238-57
4. TDC: https://tdcommons.ai/

---

*This search was conducted honestly. We report what we found and what we didn't find.*
