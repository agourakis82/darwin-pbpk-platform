# Brain Kp,uu Model v2.0 - Release Notes

## Version: 2.0.0
## Date: 2025-01-29
## Status: Validated, Ready for Integration

---

## Summary

Complete rewrite of brain unbound partition coefficient (Kp,uu) prediction model with mechanistic transporter terms and hybrid ML correction.

## Performance Metrics

### Training Validation (n=41 drugs)
| Metric | v2.0 | v1.0 | Original |
|--------|------|------|----------|
| Within 2-fold | **80.5%** | 70.7% | 47.2% |
| Within 3-fold | 100% | 87.8% | - |
| GMFE | 1.42 | 1.79 | - |
| R² (log) | 0.90 | 0.65 | 0.12 |

### Independent Validation (n=30 drugs, none in training)
| Metric | Value | Assessment |
|--------|-------|------------|
| Within 2-fold | 60.0% | Acceptable |
| Within 3-fold | 86.7% | Good |
| GMFE | 1.87 | Moderate |
| R² (log) | 0.38 | Needs improvement |

### Performance by Drug Class (Independent)
| Class | Within 2-fold | Notes |
|-------|---------------|-------|
| TCAs | 85.7% | Excellent |
| Anticonvulsants | 60.0% | Good |
| Antidepressants | 100% | Excellent |
| Atypical Antipsychotics | 40.0% | Poor - needs LAT1/OATP |
| SNRIs | 0% | Limited data |

---

## Key Improvements in v2.0

### 1. Quantitative P-gp Efflux
- Replaced binary P-gp substrate flag with continuous efflux ratio
- Efflux factor = 1 / ER (efflux ratio)

### 2. OCT3 Uptake Model (NEW)
```julia
struct OCTSubstrateScore
    score::Float64          # 0-1 likelihood
    uptake_factor::Float64  # 1.0-3.0 multiplier
    rationale::String
end
```
- Predicts active uptake for cationic drugs
- Explains Kp,uu > 1 (propranolol 3.08, methylphenidate 3.43)
- Hydrophilic cation filter (logP < 1 blocks uptake)

### 3. BCRP Efflux Model (NEW)
```julia
struct BCRPSubstrateScore
    score::Float64          # 0-1 likelihood
    efflux_factor::Float64  # 0.25-1.0 multiplier
    rationale::String
end
```
- Models second major BBB efflux pump
- Affects neutral drugs, imidazopyridines
- Explains zolpidem (0.24), thiopental (0.17)

### 4. Drug Class Corrections
- Beta-blockers: bifurcation (lipophilic=uptake, hydrophilic=excluded)
- TCAs: NET/OCT involvement (+30% factor)
- Antihistamines: high brain accumulation
- Opioids: PMAT/OCT contribution

### 5. Improved Neutral Drug Handling
- Benzodiazepines: 0.9 factor (diazepam ~1.0)
- Barbiturates: 0.25 factor (thiopental 0.17)
- Stimulants: 1.0 factor (caffeine equilibrates)

### 6. Hybrid ML Correction
- Local regression from training data
- Drug class matching improves neighbor selection
- Reduces systematic bias

---

## Known Limitations

### Model Does NOT Handle Well:
1. **Zwitterions** (gabapentin, pregabalin) - need LAT1 transporter
2. **Atypical antipsychotics** with high P-gp but moderate penetration
3. **Small polar neutrals** with ENT-mediated uptake

### Recommended Improvements (Future):
1. Add LAT1 (Large Amino Acid Transporter 1) for amino acid analogs
2. Add OATP1A2 for organic anion uptake at BBB
3. Sub-compartment modeling (grey/white matter, CSF)

---

## Files

### Core Model
- `src/DarwinPBPK/compartments/brain_kpuu_v2.jl` - Main v2.0 model

### Validation Scripts
- `scripts/validation/validate_kpuu_v2.jl` - Training set validation
- `scripts/validation/independent_kpuu_validation.jl` - Independent validation
- `scripts/validation/fetch_independent_kpuu_data.jl` - PubChem data fetcher

### API
- `src/DarwinPBPK/api/pubchem_client.jl` - PubChem REST API client

---

## Usage

```julia
using .BrainKpuuV2

# Predict Kp,uu for a drug
result = predict_kpuu_v2(
    logP = 3.5,
    fup = 0.10,
    MW = 259.3,
    pKa = 9.4,
    charge_type = :base,
    pgp_efflux_ratio = 2.0,
    drug_class = :beta_blocker,
    use_ml_correction = true
)

println("Kp,uu = $(result.kpuu)")  # 2.18 (propranolol-like)
println("OCT score = $(result.oct_score.score)")
println("BCRP score = $(result.bcrp_score.score)")
```

---

## Data Sources

### Training Data
- Ma et al. 2024 (Heliyon) - 36 marketed CNS drugs

### Independent Validation
- Fridén et al. 2011 (J Med Chem) - Anticonvulsants
- Summerfield et al. 2007 (J Pharmacol Exp Ther) - TCAs
- Liu et al. 2018 (Drug Metab Dispos) - Atypical antipsychotics

### Compound Properties
- PubChem REST API (real-time fetch)

---

## Author
Darwin PBPK Platform Team

## License
Proprietary - Research Use
