# Changelog

All notable changes to Darwin PBPK Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.5.0] - 2025-11-30

### Added - Fractal Blood Dynamics Module

#### Core Implementation
- **FractalBlood module**: Multi-phase tubular reactor with fractal network dynamics
  - Paradigm shift from "well-stirred tank" to "fractal network of PFRs"
  - CTRW (Continuous Time Random Walk) framework implementation
  - Multi-phase dynamics (plasma, RBC, protein-bound)
  - Fractal vascular network topology based on Murray's Law
  - Power-law transit time distributions

#### New Components
- `fractal_blood.jl` - Core fractal blood dynamics implementation
- `patient_profile.jl` - Patient demographics & scaling module
- `compartment_models.jl` - Physiological compartment models
- Fractal POC experimental analysis in `analysis/fractal_poc/`
  - Box-counting algorithm for fractal dimension calculation
  - Statistical validation (p < 0.001) for pathological vs normal cells
  - Theoretical model connecting image-derived fractal dimension to PK parameters

#### Documentation
- Updated `FRACTAL_PBPK_DEEP_RESEARCH.md` with POC experimental results
- Theoretical framework for df → h → PK parameter mapping
- Proof-of-concept validation with real blood cell image datasets

#### Dependencies
- Added QuadGK dependency for numerical integration
- Reorganized Project.toml dependencies alphabetically

### Changed
- Updated DarwinPBPK.jl to include new fractal blood modules
- Enhanced benchmark suite with fractal dynamics tests

### Files
- `julia-migration/src/DarwinPBPK/fractal_blood.jl` - NEW (755 lines)
- `julia-migration/src/DarwinPBPK/patient_profile.jl` - NEW
- `julia-migration/src/DarwinPBPK/compartment_models.jl` - NEW
- `analysis/fractal_poc/` - NEW directory with experimental POC
- `docs/FRACTAL_PBPK_DEEP_RESEARCH.md` - UPDATED with POC results
- `julia-migration/Project.toml` - UPDATED version and dependencies
- `julia-migration/benchmarks/benchmark_complete.jl` - UPDATED

---

## [2.4.1] - 2025-01-30

### Added
- DDI prediction module validation exceeding FDA/EMA criteria (96.2% within 2-fold, AFE 0.94, AAFE 1.33 for 26 external clinical studies).
- MedLang v1.0 DSL specification with EBNF grammar, dimensional analysis, refinement types, compartmental/DDI semantics, and Demetrios compiler backend.
- Demetrios PBPK standard library with unit type system, algebraic effects (GPU/Prob/Mut), and linear types for resource safety.
- Comprehensive DDI test suite (reversible inhibition, MBI, induction, multi-mechanism, phenotype-dependent, risk classification, transporters).
- DDI-PBPK integration for steady-state inhibitor concentrations ([I]h estimation) and transporter-CYP interplay.
- Validation scripts, debug tools, and figure generation for DDI analysis (e.g., fluconazole, fluvoxamine underpredictions).
- Examples: midazolam_ddi.medlang and demo_demetrios_compiler.jl.

### Changed
- Updated FDA DDI classification database with risk categories and 26 validation pairs.
- Updated MBI parameters with calibrated kinact/KI for CYP3A4 (kdeg=0.00048 min⁻¹).
- Methods documentation for Q1 publication, including equations for reversible/MBI/induction/OATP1B1 DDIs.

### Files
- 4 new docs: DDI_VALIDATION_REPORT.md, DEMETRIOS_PBPK_STDLIB.d, MEDLANG_SPECIFICATION.md, METHODS_DDI_PREDICTION.md
- 15 new scripts/tests: analyze_remaining_errors.jl, test_comprehensive_ddi.jl, ddi_prediction.jl, etc.
- 3 new src: ddi_pbpk_integration.jl, medlang_demetrios_compiler.jl, etc.
- 2 modified: fda_ddi_classification.jl, mbi_parameters.jl

## [2.4.0] - 2025-01-29

### Added - Brain Kp,uu Model v2.0

#### New Transporter Models
- **OCT3 Uptake Scoring**: Predicts organic cation transporter-mediated brain uptake
  - Explains Kp,uu > 1 for propranolol (3.08), methylphenidate (3.43)
  - Hydrophilic cation filter prevents false uptake predictions (atenolol fix)
- **BCRP Efflux Model**: Second major efflux pump at BBB
  - Handles neutral drugs, imidazopyridines (zolpidem, thiopental)
  - Cooperative with P-gp for dual-substrate drugs

#### Model Improvements
- Quantitative P-gp efflux (replaces binary substrate flag)
- Drug class-specific corrections (beta-blockers, TCAs, SSRIs, opioids, antihistamines)
- Hybrid ML correction with class-aware neighbor selection
- Improved neutral drug handling by class

#### Validation
- Training set (n=41): 80.5% within 2-fold, R²=0.90
- **Independent validation (n=30)**: 60.0% within 2-fold, R²=0.38
- Performance by class: TCAs 86%, Anticonvulsants 60%, Antipsychotics 40%

#### New Files
- `src/DarwinPBPK/compartments/brain_kpuu_v2.jl` - Main v2.0 model
- `src/DarwinPBPK/api/pubchem_client.jl` - PubChem REST API client
- `scripts/validation/independent_kpuu_validation.jl` - Holdout validation
- `docs/deep_dive/BRAIN_KPUU_V2_RELEASE.md` - Release documentation

#### Known Limitations
- Zwitterions (gabapentin, pregabalin) need LAT1 transporter - not yet implemented
- Atypical antipsychotics under-predicted - need OATP1A2 uptake model
- Small polar neutrals need ENT (equilibrative nucleoside transporter) model

### Changed
- Brain compartment model updated with v2.0 Kp,uu integration
- Version bump to 2.4.0

---

## [2.3.0] - 2025-01-28

### Added
- Kidney compartment deep dive with SOTA transporters
- Liver compartment with CYP450 and phase II metabolism
- Muscle compartment with exercise physiology
- Clinical blood work integration module

---

## [2.2.0] - 2025-01-27

### Added
- Quantum molecular descriptors module
- ChemBERTa embedding integration
- Multimodal encoder for drug featurization

---

## [2.1.0] - 2025-01-26

### Added
- Dynamic GNN for PBPK modeling
- Obach-Lombardo 1352 drug training pipeline
- External dataset validation framework

---

## [2.0.0-julia] - 2025-01-25

### Changed
- Complete migration from Python to Julia
- 100% Python code removed
- ODE solver rewritten in Julia with DifferentialEquations.jl
- All compartment models ported to Julia

### Added
- MedLang DSL integration
- Zenodo metadata for data archival
- Breaking change notifications

---

## [1.x] - Legacy Python Version

See legacy Python repository for historical changes.
