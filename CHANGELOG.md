# Changelog

All notable changes to Darwin PBPK Platform will be documented in this file.

## [2.7.0] - 2025-12-05

### Added

#### Advanced Blood Binding & mAb PBPK
- **Lipoprotein Binding** (`lipoprotein_binding.jl`): Drug partitioning to plasma lipoproteins
  - HDL, LDL, VLDL binding with partition coefficients
  - 20+ drugs in database (statins, immunosuppressants, antiarrhythmics, fat-soluble vitamins)
  - Disease state profiles (hypercholesterolemia, diabetic dyslipidemia, nephrotic syndrome)
  - LogP-based prediction for novel compounds
  - Integration with fu_plasma calculations

- **RBC Transporters** (`rbc_transporters.jl`): Active transport in red blood cells
  - AE1/Band3: Anion exchanger (chloroquine, organic anions)
  - GLUT1: Glucose transporter
  - ENT1/ENT2: Nucleoside transporters (gemcitabine, nucleoside analogs)
  - MCT1: Monocarboxylate transporter (lactate, pyruvate)
  - Michaelis-Menten kinetics with transporter expression
  - Disease state profiles (sickle cell, malaria-infected RBCs)

- **Disease State Binding** (`disease_state_binding.jl`): Comprehensive PK adjustments
  - Renal: CKD stages 1-5, ESRD, dialysis, AKI
  - Hepatic: Cirrhosis Child A/B/C, hepatitis, NAFLD
  - Pregnancy: Trimesters 1/2/3, postpartum
  - Critical illness: Sepsis, burns, trauma
  - Metabolic: Diabetes T1/T2, obesity, thyroid disorders
  - Automatic fu, Vd, CL, t1/2 adjustments

- **mAb PBPK Scaffold** (`mab_pbpk.jl`): Complete therapeutic antibody modeling
  - IgG subclass support (IgG1, IgG2, IgG4, Fab)
  - FcRn-mediated recycling with saturation kinetics
  - Target-Mediated Drug Disposition (TMDD)
  - Immunogenicity (ADA) effects on clearance
  - 10+ mAbs: rituximab, trastuzumab, pembrolizumab, nivolumab, infliximab, bevacizumab
  - Target database: CD20, HER2, PD-1, PD-L1, TNF-alpha, VEGF-A

### Tests
- 185 new tests for v2.7.0 modules
- Total blood compartment: 456+ tests passing

### Documentation
- `docs/BLOOD_COMPARTMENT_V270.md`: Complete API documentation
- Usage examples for all new modules

### Blood Compartment Status
- **95%+ SOTA Complete** - All major gaps addressed

---

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
- Sounio demo integration
- DDI/MedLang enhancements

---

For detailed documentation, see `julia-migration/docs/`
