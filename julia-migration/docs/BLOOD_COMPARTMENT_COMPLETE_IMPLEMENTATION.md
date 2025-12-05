# Blood Compartment - Complete Implementation Documentation

## Darwin PBPK Platform v2.5.0
**Date:** 2025-12-05  
**Author:** Darwin PBPK Platform Team

---

## Executive Summary

This document describes the complete blood compartment implementation in the Darwin PBPK Platform, integrating:

1. **FractalBlood.jl** - CTRW dynamics and multi-phase blood modeling
2. **SAM3Integration.jl** - AI-powered cell segmentation and fractal analysis
3. **LeukocyteDiagnostics.jl** - Integrated morphology + dynamics classification

### Key Achievements

| Metric | Value |
|--------|-------|
| ML Classifier AUC-ROC | **0.997** |
| Leukemia Detection Sensitivity | **100%** |
| Leukemia Detection Specificity | **92.5%** |
| Total Cells Analyzed | 15,501 |
| Cell Types Covered | 7 (4 normal + 3 leukemia subtypes) |

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    BLOOD COMPARTMENT                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  FractalBlood.jl │    │ SAM3Integration  │                  │
│  │                  │    │      .jl         │                  │
│  │ • CTRW dynamics  │    │ • Mask loading   │                  │
│  │ • Multi-phase    │    │ • Box-counting   │                  │
│  │ • Murray's Law   │    │ • Fractal Df     │                  │
│  │ • Mittag-Leffler │    │ • Morphometrics  │                  │
│  └────────┬─────────┘    └────────┬─────────┘                  │
│           │                       │                             │
│           └───────────┬───────────┘                             │
│                       ▼                                         │
│           ┌──────────────────────┐                              │
│           │ LeukocyteDiagnostics │                              │
│           │        .jl           │                              │
│           │                      │                              │
│           │ • Profile creation   │                              │
│           │ • CTRW estimation    │                              │
│           │ • ML classification  │                              │
│           │ • Drug response      │                              │
│           │ • Clinical reports   │                              │
│           └──────────────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Descriptions

### 2.1 FractalBlood.jl

**Location:** `julia-migration/src/DarwinPBPK/fractal_blood.jl`

Implements the paradigm shift from "well-stirred tank" to "fractal network of PFRs".

#### Core Components

| Component | Description |
|-----------|-------------|
| `BloodPhase` | Plasma, RBC, protein-bound phases |
| `VesselSegment` | Single vessel with Murray's Law properties |
| `FractalBloodModel` | Complete network + CTRW parameters |

#### Key Functions

```julia
# Create fractal vascular network
model = create_fractal_blood_model(
    num_levels=15,      # Branching levels
    hematocrit=0.45,    # RBC fraction
    fu=0.1,             # Fraction unbound
    alpha=1.37,         # Transit time power-law
    beta=0.8            # Anomalous diffusion exponent
)

# Simulate CTRW transport
results = simulate_ctrw(model, n_particles=1000, t_max=24.0)

# Mittag-Leffler for fractional kinetics
E_α(z) = mittag_leffler(α, z)
```

#### Mathematical Foundations

**Power-law transit time distribution:**
$$p(\tau) = \frac{\alpha-1}{\tau_{min}} \left(\frac{\tau}{\tau_{min}}\right)^{-\alpha}$$

**Anomalous diffusion (MSD):**
$$\langle x^2(t) \rangle = \frac{2D_0 t^\beta}{\Gamma(1+\beta)}$$

**Fractal rate constant:**
$$k(t) = k_0 \cdot t^{-h}$$

---

### 2.2 SAM3Integration.jl

**Location:** `julia-migration/src/DarwinPBPK/image_analysis/sam3_integration.jl`

Loads SAM-3 segmentation masks and performs fractal dimension analysis.

#### Data Structures

```julia
struct SAM3MaskData
    masks::Array{UInt8, 3}        # (N, H, W) cell masks
    combined_mask::BitMatrix      # All cells combined
    edge_mask::BitMatrix          # Edge detection
    scores::Vector{Float64}       # Confidence scores
    boxes::Matrix{Float64}        # Bounding boxes
    n_cells::Int
    image_shape::Tuple{Int, Int}
    cell_type::String
    source_image::String
    prompt_used::String
    cell_properties::Vector{Dict}
end

struct CellFractalMetrics
    cell_id::Int
    df_edge::Float64              # Edge fractal dimension
    df_area::Float64              # Area fractal dimension
    r_squared_edge::Float64
    r_squared_area::Float64
    area::Int                     # Pixels
    perimeter::Float64
    circularity::Float64          # 4π×area/perimeter²
    score::Float64                # SAM-3 confidence
end
```

#### Box-Counting Algorithm

```julia
function box_counting_fractal_dimension(
    binary_image::BitMatrix;
    min_box_size::Int=2,
    max_box_size::Union{Int, Nothing}=nothing,
    n_sizes::Int=10
)::Tuple{Float64, Float64, Vector{Int}, Vector{Int}}
    # Returns: (Df, R², box_sizes, box_counts)
end
```

---

### 2.3 LeukocyteDiagnostics.jl

**Location:** `julia-migration/src/DarwinPBPK/image_analysis/leukocyte_diagnostics.jl`

Integrated module combining morphology + dynamics for clinical diagnosis.

#### Main Functions

```julia
# Create complete cell profile
profile = create_leukocyte_profile("masks.npz")

# Classify sample
result = classify_cells(profile)
# Returns: DiagnosticResult with class, confidence, recommendations

# Simulate CTRW dynamics
dynamics = simulate_cell_dynamics(profile, t_max=50.0)

# Predict drug response
response = predict_cell_behavior(profile, drug_params)
```

#### Classification Logic

The classifier uses calibrated thresholds from ML training:

| Feature | Normal Range | Leukemia Range | Weight |
|---------|--------------|----------------|--------|
| df_edges | ~1.31 | ~1.60 | 30% |
| n_cells | 8-22 | 70-80 | 30% |
| df_combined | ~1.69 | ~1.76 | 20% |
| df_distribution | ~0.50 | ~0.60 | 10% |
| mean_circularity | ~0.50 | ~0.56 | 10% |

---

## 3. ML Training Results

### 3.1 Dataset

| Cell Type | N Images | N Cells | Source |
|-----------|----------|---------|--------|
| Neutrophils | 50 | 439 | WBC Dataset |
| Lymphocytes | 50 | 1,112 | WBC Dataset |
| Monocytes | 50 | 828 | WBC Dataset |
| Eosinophils | 50 | 1,102 | WBC Dataset |
| Leukemia (Pre) | 50 | 3,733 | ALL Dataset |
| Leukemia (Early) | 50 | 3,500 | ALL Dataset |
| Leukemia (Pro) | 50 | 4,787 | ALL Dataset |
| **Total** | **350** | **15,501** | |

### 3.2 Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| **Random Forest** | **95.7%** | 90.9% | **100%** | 95.2% | **0.997** |
| Logistic Regression | 97.1% | 96.7% | 96.7% | 96.7% | 0.996 |
| SVM | 97.1% | 96.7% | 96.7% | 96.7% | 0.993 |
| Gradient Boosting | 87.1% | 81.8% | 90.0% | 85.7% | 0.895 |

### 3.3 Confusion Matrix (Random Forest)

```
                  Predicted
                  Normal   Leukemia
Actual Normal        37        3
       Leukemia       0       30
```

- **Sensitivity: 100%** (no missed leukemia cases)
- **Specificity: 92.5%**
- **NPV: 100%** (if test says normal, it's reliable)

### 3.4 Feature Importance

```
df_edges             ████████████ 24.7%
mean_area            █████████ 18.9%
n_cells              ████████ 16.6%
df_distribution      ████████ 16.6%
mean_df_edge         ████ 9.6%
mean_circularity     ███ 6.4%
df_combined          █ 3.8%
std_df_edge          █ 3.5%
```

---

## 4. Fractal Dimension Reference Values

From differential analysis of 1,405 cells:

| Cell Type | Df (mean) | Df (std) | Circularity |
|-----------|-----------|----------|-------------|
| Neutrophils | 1.660 | 0.122 | 0.509 |
| Lymphocytes | 1.722 | 0.036 | 0.510 |
| Monocytes | 1.676 | 0.077 | 0.497 |
| Eosinophils | 1.711 | 0.032 | 0.492 |
| **Leukemia** | **1.761** | 0.068 | 0.564 |

**Key Finding:** Leukemia cells have significantly higher Df (p=0.002).

---

## 5. CTRW Parameter Mapping

The system estimates CTRW parameters from morphology:

| Morphology | β (diffusion) | α (transit) | τ_scale |
|------------|---------------|-------------|---------|
| Normal | 0.85 | 1.37 | 1.0 |
| Activated | 0.75 | 1.25 | 1.5 |
| Leukemia | 0.65 | 1.15 | 2.5 |

**Interpretation:**
- Lower β → more subdiffusive (anomalous trapping)
- Lower α → heavier tail transit times
- Higher τ_scale → longer waiting times

---

## 6. Drug Response Predictions

### Example: Leukemia vs Normal

| Metric | Normal | Leukemia | Difference |
|--------|--------|----------|------------|
| AUC (traditional) | 914.8 | 914.8 | 0% |
| AUC (fractal) | 1145.1 | 1304.4 | +14% |
| Survival factor | 0.74 | 0.77 | +3.6% |

**Clinical Implication:** Leukemia cells have altered pharmacokinetics with increased drug retention but paradoxically higher survival (drug resistance).

---

## 7. API Usage Examples

### 7.1 Complete Diagnostic Pipeline

```julia
using LeukocyteDiagnostics

# Single sample diagnosis
result = diagnose_sample("/path/to/masks.npz")

println("Class: $(result.predicted_class)")
println("Confidence: $(result.confidence * 100)%")
println("Morphology: $(result.morphology_interpretation)")
println("Dynamics: $(result.dynamics_interpretation)")
println("Recommendation: $(result.clinical_recommendation)")
```

### 7.2 Batch Analysis

```julia
results = analyze_batch("/path/to/masks_dir/")
report = generate_diagnostic_report(results)

println("Total samples: $(report["summary"]["total_samples"])")
println("High-risk cases: $(report["summary"]["high_risk_count"])")
println("Class distribution: $(report["summary"]["class_distribution"])")
```

### 7.3 Drug Response Simulation

```julia
profile = create_leukocyte_profile("/path/to/masks.npz")

drug_params = Dict(
    "dose" => 100.0,    # mg
    "k_el" => 0.1,      # 1/h
)

response = predict_cell_behavior(profile, drug_params, t_max=24.0)

println("AUC ratio (fractal/traditional): $(response["AUC_ratio"])")
println("Predicted survival: $(response["predicted_survival_factor"])")
println("$(response["interpretation"])")
```

---

## 8. File Structure

```
julia-migration/
├── src/DarwinPBPK/
│   ├── fractal_blood.jl              # CTRW dynamics
│   └── image_analysis/
│       ├── sam3_integration.jl       # SAM-3 mask loading
│       └── leukocyte_diagnostics.jl  # Integrated module
├── test/
│   ├── test_sam3_integration_final.jl
│   └── test_leukocyte_diagnostics.jl
└── docs/
    └── BLOOD_COMPARTMENT_COMPLETE_IMPLEMENTATION.md

analysis/fractal_poc/
├── export_sam3_masks.py              # Python mask exporter
├── differential_fractal_analysis.py  # Statistical analysis
├── ml_leukemia_classifier.py         # ML training
└── results/
    ├── differential_analysis/
    └── ml_classifier/
        ├── ml_results.json
        └── best_model.pkl
```

---

## 9. Dependencies

### Julia
- NPZ.jl - NumPy file loading
- JSON3.jl - JSON parsing
- Statistics (stdlib)
- LinearAlgebra (stdlib)

### Python (for SAM-3)
- torch
- numpy
- scipy
- scikit-learn
- PIL

---

## 10. References

1. Goirand et al. (2021) "Network-driven anomalous transport" - Nature Communications
2. Macheras (1996) "Fractal pharmacokinetics"
3. Murray's Law (1926) - Vascular branching
4. SAM-3 (Segment Anything Model 3) - Meta AI

---

## 11. Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.5.0 | 2025-12-05 | LeukocyteDiagnostics integration |
| 2.4.0 | 2025-12-04 | SAM-3 integration, ML classifier |
| 2.3.0 | 2025-11-30 | FractalBlood.jl CTRW dynamics |

---

*Darwin PBPK Platform - Q1 Scientific Rigor*
