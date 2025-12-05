# Changelog

All notable changes to Darwin PBPK Platform will be documented in this file.

## [2.6.0] - 2025-12-05

### Added

#### Blood Compartment Module
- **Blood Binding** (`blood_binding.jl`): Mechanistic B:P ratio calculation using Rodgers-Rowland equations
  - RBC partitioning with pH-dependent ion trapping
  - Plasma protein binding (albumin, AAG)
  - Platelet and WBC partitioning
  - PK-Sim style methodology

- **Drug-Specific WBC Binding**: Database for drugs with significant leukocyte accumulation
  - Chloroquine/hydroxychloroquine lysosomotropic accumulation
  - Antiretrovirals (tenofovir, dolutegravir, darunavir, ritonavir, maraviroc)
  - Azithromycin extreme accumulation (100:1 ratio)
  - Reservoir effect calculations

- **Hemodynamics** (`hemodynamics.jl`): Blood flow mechanics
  - Wall shear stress calculation
  - Shear-Induced Platelet Activation (SIPA)
  - vWF unfolding dynamics
  - Carreau-Yasuda non-Newtonian viscosity

- **Coagulation Extended** (`coagulation_extended.jl`): Enhanced coagulation cascade
  - FXI feedback loop (fixes Hockin-Mann model at low TF)
  - Contact pathway (FXII, kallikrein)
  - Platelet surface enhancement (300,000x)
  - Antithrombin dynamics

- **TGA Validation** (`tga_validation.jl`): Clinical validation framework
  - Clinical datasets: healthy, hemophilia, warfarin, DOACs
  - Goodness-of-fit metrics (AAFE, GMFE, R2)
  - FDA/EMA-style acceptance criteria

- **Lattice Boltzmann CFD** (`lattice_boltzmann.jl`): Blood flow simulation
  - D2Q9 lattice configuration
  - Vessel geometries (straight, stenosis, bifurcation)
  - Wall shear stress extraction
  - Hematocrit-dependent viscosity

- **Sensitivity Analysis** (`sensitivity_analysis.jl`): Parameter sensitivity framework
  - One-at-a-Time (OAT) local sensitivity
  - Morris screening (global)
  - Sobol variance decomposition (global)
  - PRCC correlation analysis (global)
  - Latin Hypercube Sampling

### Tests
- 271 new tests across all blood compartment modules
- Standalone test files for CI/CD compatibility

### Documentation
- `docs/BLOOD_COMPARTMENT_V260.md`: Comprehensive module documentation
- API documentation for all new functions
- Usage examples and references

## [2.5.0] - 2025-11-XX

### Added
- FractalBlood module with CTRW dynamics
- Multi-phase blood modeling
- Experimental POC for fractal analysis

## [2.4.1] - 2025-11-XX

### Added
- MedLang DDI example
- Demetrios demo integration
- DDI/MedLang enhancements

---

For detailed documentation, see `julia-migration/docs/`
