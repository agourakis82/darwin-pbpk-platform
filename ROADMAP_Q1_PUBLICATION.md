# Darwin PBPK + MedLang: Roadmap to Q1+ Publication

## Current Status (November 2025)

### Achieved
- PBPK model validated on **1,232 drugs** (Obach-Lombardo dataset)
- **GMFE: 1.64** (target < 2.0)
- **78.1% within 2-fold** (target > 70%)
- **R^2: 0.755**, Correlation: 0.879
- MedLang DSL v1.0 with QSP, ML, Track C operators
- Julia 100% migration complete

### Publication Readiness: ~50%

---

## CRITICAL GAPS (Must-Fix for Q1)

### 1. ML/AI Integration Gaps

| Feature | Current | SOTA Requirement | Priority |
|---------|---------|------------------|----------|
| **Multimodal Encoders** | Placeholder | ChemBERTa + D-MPNN + 3D | CRITICAL |
| **Bayesian UQ** | Evidential only | MCMC + Variational Inference | CRITICAL |
| **Neural ODE** | Not implemented | Alternative to GNN temporal | HIGH |
| **Attention Mechanism** | Basic Dense | Multi-head (8+ heads) | HIGH |

### 2. Validation Gaps

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| External blind validation | 0 compounds | 15-20 compounds | CRITICAL |
| Uncertainty calibration | Not assessed | Reliability diagrams | CRITICAL |
| AFE/AAFE metrics | Not implemented | Regulatory standard | HIGH |
| Commercial comparison | None | vs Simcyp/GastroPlus | MEDIUM |

### 3. Clinical Applicability Gaps

| Feature | Status | Priority |
|---------|--------|----------|
| Population variability | Missing | MEDIUM |
| Drug-Drug Interactions (DDI) | Missing | MEDIUM |
| Special populations (pediatric, renal) | Missing | MEDIUM |
| QT/Cardiac safety | Missing | LOW |

### 4. MedLang DSL Gaps

| Feature | Status | Priority |
|---------|--------|----------|
| Complete transpiler | Partial | HIGH |
| Integration with GNN | Missing | HIGH |
| Error handling/validation | Basic | MEDIUM |
| Examples & tutorials | Few | MEDIUM |

---

## TIER 1: MUST-DO (Non-negotiable for Q1)

### 1.1 Implement Multimodal Molecular Encoders (3-4 weeks)

```julia
# Target architecture
struct MultimodalEncoder
    chemberta::ChemBERTaEncoder      # 768d SMILES embeddings
    gnn::DMPNNEncoder                 # 256d molecular graph
    conformer::ConformerEncoder       # 50d 3D structure
    fusion::CrossAttentionFusion      # 8 heads, 512d output
end
```

**Expected Impact:** R^2 improvement from 0.18 -> 0.50+ for clearance prediction

### 1.2 Implement Bayesian Uncertainty Quantification (2-3 weeks)

```julia
# Dual-mode Bayesian framework
struct BayesianPBPK
    mcmc::MCMCInference       # Gold standard (PyMC/Turing.jl)
    vi::VariationalInference  # Fast mode (ADVI)
end

# Output: Point estimate + 95% credible interval
predict_with_uncertainty(model, drug) -> (mean, ci_lower, ci_upper)
```

**Expected Impact:** Essential for regulatory acceptance, unique differentiator

### 1.3 External Blind Validation (3-4 weeks)

- Collect 15-20 compounds from PK-DB (not in training)
- Strict blind protocol: no parameter tuning post-prediction
- Complete regulatory metrics:
  - AFE (Average Fold Error)
  - AAFE (Absolute Average Fold Error)
  - % within 1.25x, 1.5x, 2.0x
  - Uncertainty calibration curves

### 1.4 Scope Alignment (1 week)

- Audit claims vs actual implementation
- Remove or implement claimed features
- Update README, documentation

---

## TIER 2: STRONGLY RECOMMENDED (High Impact)

### 2.1 Neural ODE Architecture (2 weeks)

```julia
# Alternative temporal evolution
struct NeuralODEPBPK
    encoder::GNNEncoder
    dynamics::NeuralODE{Chain}  # Continuous-time dynamics
    decoder::MLPDecoder
end
```

**Expected Impact:** Architecture comparison paper, shows innovation

### 2.2 MedLang Complete Integration (3-4 weeks)

```
MedLang DSL -> Julia AST -> PBPKParams -> ODE Solver -> GNN Prediction
                                      |
                                      v
                              Bayesian UQ -> Credible Intervals
```

**Expected Impact:** First PBPK DSL - unique contribution

### 2.3 Reproducibility Package (2-3 weeks)

- [ ] Pre-trained model checkpoints (Zenodo)
- [ ] Docker container
- [ ] Tutorial Jupyter notebook
- [ ] GitHub Actions CI/CD

---

## TIER 3: OPTIONAL BUT VALUABLE

### 3.1 Clinical Case Studies

- Warfarin DDI prediction
- Pediatric dosing (age-based scaling)
- Renal impairment adjustments

### 3.2 Population Variability Module

```julia
struct PopulationPBPK
    base_params::PBPKParams
    covariates::Dict{Symbol, Distribution}  # Age, weight, sex
    iiv::Vector{IIVSpec}                     # Inter-individual variability
end

generate_virtual_population(pop, n=1000) -> Vector{PBPKParams}
```

### 3.3 Spatial PDE-PBPK (if claiming)

- Tumor microenvironment modeling
- EPR effect quantification
- 2D/3D concentration gradients

---

## PUBLICATION STRATEGY

### Target Journals (in order of preference)

1. **CPT: Pharmacometrics & Systems Pharmacology** (45-55% chance)
   - Focus: GNN + Bayesian UQ for PBPK
   - Validation: 1,232 drugs + 20 blind compounds
   - Differentiator: Dual-mode Bayesian (MCMC + VI)

2. **Journal of Pharmacokinetics and Pharmacodynamics** (45-55% chance)
   - Focus: Bayesian-PBPK methodology
   - Validation: Uncertainty calibration
   - Differentiator: Open-source alternative to Simcyp

3. **Nature Communications** (25-35% chance)
   - Focus: End-to-end AI-PBPK pipeline
   - Requirements: Clinical case studies, broad impact
   - Differentiator: MedLang DSL + Multimodal + Bayesian

---

## IMPLEMENTATION TIMELINE

```
December 2025 - January 2026: TIER 1 (Critical)
├── Week 1-2: Multimodal encoders (ChemBERTa + D-MPNN)
├── Week 2-3: Bayesian UQ integration (Turing.jl)
├── Week 3-4: External blind validation (15-20 compounds)
└── Week 4: Scope alignment & documentation

February 2026: TIER 2 (High Priority)
├── Week 5-6: Neural ODE architecture
├── Week 6-7: MedLang complete integration
└── Week 7-8: Reproducibility package

March 2026: Validation & Benchmarking
├── Week 9-10: Clinical case studies
├── Week 10-11: Comparative benchmarking
└── Week 11-12: Sensitivity analyses

April 2026: Manuscript
├── Week 13-14: Write Methods & Results
├── Week 14-15: Write Discussion & Conclusions
└── Week 15-16: Internal review & revision

May 2026: Submission
└── Target: CPT: Pharmacometrics & Systems Pharmacology
```

---

## SPECIFIC NEXT ACTIONS

### Immediate (This Week)

1. **Implement ChemBERTa encoder wrapper**
   ```julia
   # Using Transformers.jl or PyCall to HuggingFace
   struct ChemBERTaEncoder
       model::TransformerModel
       tokenizer::Tokenizer
   end
   
   encode(enc::ChemBERTaEncoder, smiles::String) -> Vector{Float64}  # 768d
   ```

2. **Implement Turing.jl Bayesian PBPK**
   ```julia
   @model function bayesian_pbpk(obs_conc, times, dose)
       # Priors
       cl ~ LogNormal(log(10.0), 0.5)
       vd ~ LogNormal(log(70.0), 0.5)
       
       # PBPK simulation
       params = PBPKParams(clearance_hepatic=cl*0.9, clearance_renal=cl*0.1, ...)
       pred = simulate(params, dose; t_max=maximum(times))
       
       # Likelihood
       for (i, t) in enumerate(times)
           obs_conc[i] ~ Normal(pred["blood"][findtime(t)], 0.1)
       end
   end
   ```

3. **Collect external validation compounds**
   - Download PK-DB dataset
   - Select 20 diverse compounds (different BCS classes)
   - Hold out from any training/tuning

4. **Implement missing validation metrics**
   ```julia
   # Add to validation.jl
   average_fold_error(pred, obs) = 10^mean(log10.(pred ./ obs))
   absolute_average_fold_error(pred, obs) = 10^mean(abs.(log10.(pred ./ obs)))
   ```

---

## KEY LITERATURE TO CITE

1. Wang et al. (2025) - "AI-PBPK Platform" - CPT
2. Lombardo et al. (2018) - "1352 Drug Database" - DMD
3. Walter et al. (2025) - "ML Pharmacokinetics" - CTS
4. Wu et al. (2024) - "AI-PBPK Pharmacodynamics" - Front Pharmacol

---

## SUCCESS METRICS

| Milestone | Target Date | Metric |
|-----------|-------------|--------|
| Multimodal encoders | Dec 15, 2025 | R^2 CL > 0.40 |
| Bayesian UQ | Dec 31, 2025 | 95% CI coverage > 90% |
| External validation | Jan 15, 2026 | GMFE < 2.0 on blind set |
| Neural ODE | Feb 15, 2026 | Architecture comparison |
| MedLang integration | Feb 28, 2026 | End-to-end workflow |
| Manuscript draft | Apr 15, 2026 | Complete first draft |
| Submission | May 1, 2026 | CPT or JoPPD |

---

## CONCLUSION

Darwin PBPK has a **strong foundation** but needs **6 months of focused work** to reach Q1 publication quality. The key differentiators are:

1. **Bayesian-GNN integration** (unique)
2. **MedLang DSL** (first of its kind)
3. **Open-source alternative** to commercial PBPK
4. **Validated on 1,232 drugs** (already achieved!)

With Tier 1 items complete, publication in CPT or JoPPD is achievable by mid-2026.
