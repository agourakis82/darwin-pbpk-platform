# Changelog

All notable changes to Darwin PBPK Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
