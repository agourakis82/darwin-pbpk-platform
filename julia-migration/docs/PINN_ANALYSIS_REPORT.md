# PINN for PBPK: Deep Research Analysis

## Executive Summary

**Question:** Is PINN (Physics-Informed Neural Networks) appropriate for our Vdss prediction task?

**Answer:** NO - PINNs solve a fundamentally different problem than what we're trying to do.

## The Two Different Problems

### What We're Doing (QSPR/ML Approach)
```
Molecular Structure + Descriptors → Vdss (Direct Prediction)
```
- **Input:** SMILES, MW, LogP, TPSA, fup, etc.
- **Output:** Vdss (L/kg)
- **Nature:** Statistical/correlational mapping
- **No time component** - single point prediction

### What PINNs Solve (ODE-Constrained Estimation)
```
Concentration-Time Data → ODE Parameters → Concentration Profile
```
- **Input:** Drug concentration measurements over time C(t)
- **Output:** PK parameters (k_a, k_e, V, CL, etc.) OR concentration curves
- **Nature:** Inverse problem with physics constraints
- **Time-series data required**

## Why PINN Won't Help Us

1. **No ODE to Constrain Against**
   - PINNs embed differential equations as loss terms
   - Vdss prediction from molecular structure has no underlying ODE
   - The physics is in tissue partitioning, not temporal dynamics

2. **Different Data Requirements**
   - PINN: Needs concentration-time curves (C vs t)
   - Our task: Molecular structure → single Vdss value
   - We don't have temporal PK data for training

3. **PBPK-iPINNs Are For Parameter Estimation**
   - Recent papers (arXiv:2509.12666) use PINNs to estimate tissue-specific parameters
   - They START with concentration data, ESTIMATE Kp (partition coefficients)
   - We need to PREDICT Vdss without any concentration data

## What the Literature Shows About Best Vdss Prediction

### State-of-the-Art Performance (from research):

| Method | GMFE | % Within 2-fold | Reference |
|--------|------|-----------------|-----------|
| Øie-Tozer (preclinical) | 1.55 | 81% | Lombardo 2018 |
| PKSmart (2024) | 2.09 | ~60% | Seal et al. |
| AstraZeneca proprietary | ~2.0 | ~60% | Industry |
| Our best result | 1.985 | 57.5% | This work |

### The Gap: What Top Methods Have That We Don't

1. **Predicted Animal PK Data**
   - PKSmart uses **rat, dog, monkey** predicted PK as features
   - Two-stage model: Structure → Animal PK → Human PK
   - Rationale: Animal-human correlation captures biology we can't model

2. **Fraction Unbound in Tissue (fut)**
   - Øie-Tozer equation: `Vdss = Vp + Ve*fu/fut + Vr*fu/fur`
   - `fut` is the missing link between molecular properties and distribution
   - Requires tissue binding experiments or mechanistic Kp prediction

3. **Blood-to-Plasma Ratio (BPR)**
   - For lipophilic drugs, BPR correlates with tissue distribution
   - Not just LogP - actual red blood cell partitioning

4. **LogD at Multiple pH Values**
   - Not just LogD7.4 but tissue-relevant pH (LogD5.0, LogD6.0)
   - Lysosomotropism for basic drugs

## The Fundamental Issue

We're trying to predict a **mechanistic outcome** (how drug distributes into tissues) 
using **statistical correlates** (molecular descriptors) without the **mechanistic bridge** 
(tissue partition coefficients).

```
Current approach (limited):
  Structure → [ML Black Box] → Vdss
  
What works better:
  Structure → Predicted Animal PK → [ML] → Human Vdss
  Structure → Predicted Kp → Øie-Tozer → Vdss
  Structure + fup + fut → Mechanistic → Vdss
```

## Recommendations

### Option 1: Two-Stage Animal PK Prediction (Like PKSmart)
- Train models to predict rat/dog/monkey VDss from structure
- Use predicted animal VDss as features for human VDss
- Expected improvement: GMFE ~1.8-2.0, more robust

### Option 2: Mechanistic Feature Augmentation
- Add predicted Kp (tissue partition coefficients) using Rodgers-Rowland
- Add predicted BPR (blood-plasma ratio)
- Add ionization state at tissue pH

### Option 3: Hybrid Mechanistic-ML
- Use ML to predict fut from structure
- Apply Øie-Tozer equation mechanistically
- Constrain ML output to satisfy mechanistic bounds

### Why ChemBERTa Didn't Help
- Embeddings capture general chemical similarity
- Don't capture PK-relevant features (ionization, membrane permeability)
- Need PK-specific learned representations

## Conclusion

**PINN is NOT appropriate** because we're doing property prediction, not ODE solving.

**What IS appropriate:**
1. Adding mechanistic features (Kp, BPR, fut predictions)
2. Two-stage animal-to-human transfer learning
3. Hybrid ML-mechanistic models

**Our GMFE of 1.985 on best fold is actually competitive** with published models, but 
mean GMFE of 2.19 shows we need more robust features to consistently hit <2.0.

## References

- [Lombardo 2018 - Øie-Tozer Vdss](https://pubmed.ncbi.nlm.nih.gov/31578209/)
- [PKSmart 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12466039/)
- [PBPK-iPINNs](https://arxiv.org/html/2509.12666)
- [QSPR vs PBPK evaluation](https://www.sciencedirect.com/science/article/abs/pii/S002235492030784X)
- [Hybrid ML-mechanistic](https://www.nature.com/articles/s41598-021-90637-1)
