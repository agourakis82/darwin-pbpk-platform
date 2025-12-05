"""
Darwin PBPK Platform - Julia Implementation

SOTA + Disruptive + Nature-tier PBPK modeling platform.
First Real Implementation of MedLang DSL (github.com/agourakis82/medlang)

Author: Dr. Demetrios Agourakis
Date: November 2025
"""

module DarwinPBPK

# Core modules
include("DarwinPBPK/patient_profile.jl")  # Patient demographics & scaling
include("DarwinPBPK/compartment_models.jl")  # Physiological compartment models
include("DarwinPBPK/fractal_blood.jl")  # NEW: Fractal blood dynamics (CTRW, multi-phase)
include("DarwinPBPK/compartments/white_blood_cells.jl")  # NEW: Detailed WBC modeling with subpopulations
include("DarwinPBPK/compartments/platelets.jl")  # NEW: Platelet compartment with activation dynamics
include("DarwinPBPK/compartments/coagulation.jl")  # NEW: Coagulation cascade ODE model (Wajima/Hockin-Mann)
include("DarwinPBPK/compartments/fibrinolysis.jl")  # NEW: Fibrinolysis system (plasmin, tPA, D-dimer)
include("DarwinPBPK/compartments/blood_binding.jl")  # NEW: B:P ratio, RBC/WBC partitioning (PK-Sim style)
include("DarwinPBPK/compartments/hemodynamics.jl")  # NEW: Shear-dependent effects, vWF, SIPA
include("DarwinPBPK/compartments/coagulation_extended.jl")  # NEW: FXI feedback, contact pathway
include("DarwinPBPK/compartments/tga_validation.jl")  # NEW: Thrombin Generation Assay validation
include("DarwinPBPK/compartments/sensitivity_analysis.jl")  # NEW: Local and global sensitivity analysis
include("DarwinPBPK/compartments/lattice_boltzmann.jl")  # NEW: Lattice Boltzmann Method for blood flow simulation
include("DarwinPBPK/image_analysis/leukocyte_fractal_analysis.jl")  # NEW: Fractal analysis of leukocyte morphology
include("DarwinPBPK/ode_solver.jl")
include("DarwinPBPK/dataset_generation.jl")
include("DarwinPBPK/dynamic_gnn.jl")  # FASE 2 ✅
include("DarwinPBPK/training.jl")     # FASE 2 ✅

# ML modules
include("DarwinPBPK/ml/multimodal_encoder.jl")  # FASE 3 ✅ (Real implementation with MolecularGraph.jl)
include("DarwinPBPK/ml/evidential.jl")          # FASE 3 ✅
include("DarwinPBPK/ml/bayesian_uq.jl")         # Q1 2025 ✅ (Bayesian UQ)

# Validation (FASE 4)
include("DarwinPBPK/validation.jl")              # FASE 4 ✅

# API (FASE 5)
include("DarwinPBPK/api/rest_api.jl")           # FASE 5 ✅

# MedLang DSL (First Real Implementation)
include("DarwinPBPK/medlang/MedLang.jl")        # MedLang DSL ✅

# Semantic Web Layer (FAIR Data)
include("DarwinPBPK/semantic/SemanticCore.jl")  # JSON-LD + OBO Foundry ✅

# Re-export principais
using .PatientProfile
using .CompartmentModels
using .FractalBlood
using .WhiteBloodCells
using .Platelets
using .Coagulation
using .Fibrinolysis
using .BloodBinding
using .Hemodynamics
using .CoagulationExtended
using .TGAValidation
using .SensitivityAnalysis
using .LatticeBoltzmann
using .LeukocyteFractalAnalysis
using .ODEPBPKSolver
using .DatasetGeneration
using .DynamicGNN
using .Training
using .MultimodalEncoder
using .Evidential
using .BayesianUQ
using .Validation
using .RESTAPI
using .MedLang
using .SemanticCore

# Export MedLang DSL functions
export parse_medlang, compile_model, compile_file, simulate_medlang
export load_medlang, generate_julia_module, validate_model, describe_model

# Export Bayesian UQ functions
export BayesianPBPKModel, PosteriorResult, ParameterPrior
export sample_posterior, variational_inference, default_pbpk_priors
export credible_interval, posterior_predictive, uncertainty_calibration
export VariationalPosterior, sample_variational, create_clearance_model

# Export Semantic Web functions
export DARWIN_CONTEXT, DRUG_CONTEXT, DDI_CONTEXT, PARAMETER_CONTEXT
export QUDTUnit, SemanticQuantity, QUDT_UNITS, get_qudt_unit
export to_jsonld_drug, to_jsonld_ddi, to_jsonld_parameter, to_jsonld_simulation
export to_turtle, serialize_entity, export_to_jsonld, export_to_turtle
export ProvenanceRecord, create_prediction_provenance, create_simulation_provenance
export create_semantic_drug, create_semantic_ddi, create_semantic_parameter
export annotate_with_provenance, validate_jsonld

# Export DOID (Disease Ontology) functions
export DOIDTerm, DOIDIndex, load_doid, search_doid
export get_disease_by_id, get_disease_by_name, get_disease_xrefs
export get_disease_hierarchy, get_diseases_for_drug_class

# Export Platelet functions
export PlateletCompartment, PlateletGranules, PlateletActivation
export create_platelet_compartment, activate_platelets!, aggregate_platelets!
export apply_antiplatelet_drug!, calculate_bleeding_risk, get_platelet_state
export simulate_platelet_dynamics, NORMAL_PLATELET_COUNT, NORMAL_MPV

# Export Coagulation functions
export CoagulationFactors, CoagulationSystem, AnticoagulantState
export create_coagulation_system, simulate_coagulation!
export apply_warfarin!, apply_doac!, apply_heparin!
export calculate_pt_inr, calculate_aptt, calculate_anti_xa
export thrombin_generation_assay, get_coagulation_state
export NORMAL_FACTOR_CONCENTRATIONS, FACTOR_HALF_LIVES

# Export Fibrinolysis functions
export FibrinolyticSystem, PlasminogenState, FibrinDegradation
export create_fibrinolytic_system, simulate_fibrinolysis!
export apply_tpa_therapy!, apply_antifibrinolytic!
export calculate_d_dimer, calculate_lysis_time
export get_fibrinolysis_state, plasmin_generation_assay
export NORMAL_PLASMINOGEN, NORMAL_TPA, NORMAL_PAI1

# Export Blood Binding functions (PK-Sim style)
export BloodComposition, DrugProperties, BloodPartitioning
export calculate_blood_plasma_ratio, calculate_rbc_partition
export calculate_wbc_partition, calculate_platelet_partition
export calculate_fu_blood, calculate_erythrocyte_water_partition
export create_drug_properties, get_blood_composition
export STANDARD_HEMATOCRIT, PHYSIOLOGICAL_PH

# Export Hemodynamics functions
export ShearEnvironment, VesselGeometry, FlowConditions
export calculate_wall_shear_stress, calculate_shear_rate
export shear_induced_platelet_activation, vwf_unfolding_probability
export calculate_residence_time, calculate_transport_rate
export create_vessel, get_flow_regime
export BLOOD_VISCOSITY, CRITICAL_SHEAR_RATES

# Export Extended Coagulation functions (FXI feedback)
export ExtendedCoagulationSystem, ContactPathway, PlateletSurface
export create_extended_coagulation, simulate_extended_coagulation!
export add_fxi_feedback!, add_contact_activation!, add_shear_effects!
export calculate_surface_reactions, polyphosphate_enhancement
export EXTENDED_KINETIC_PARAMS, FXI_FEEDBACK_PARAMS

# Export TGA Validation functions
export TGAParameters, ClinicalTGADataset, ValidationMetrics
export extract_tga_parameters, compare_to_clinical
export calculate_prediction_error, validate_coagulation_model
export HEALTHY_TGA_1PM_TF, HEALTHY_TGA_5PM_TF
export HEMOPHILIA_A_TGA, HEMOPHILIA_B_TGA
export WARFARIN_INR2_TGA, WARFARIN_INR3_TGA
export DOAC_RIVAROXABAN_TGA, DOAC_APIXABAN_TGA
export FXI_DEFICIENCY_TGA
export calculate_goodness_of_fit, get_all_clinical_datasets, print_validation_summary

# Export Sensitivity Analysis functions
export ParameterRange, SensitivityResult, SensitivityConfig
export one_at_a_time_sensitivity, calculate_elasticity, normalized_sensitivity_coefficient
export sobol_sensitivity, morris_screening, prcc_analysis
export latin_hypercube_sample, sobol_sequence, morris_trajectories
export rank_parameters, identify_influential_parameters
export sensitivity_tornado_plot_data, sensitivity_heatmap_data
export coagulation_sensitivity_wrapper, default_coagulation_parameters

# Export Lattice Boltzmann functions
export LatticeConfig, D2Q9Lattice, D3Q19Lattice
export FluidProperties, BoundaryConditions, SimulationDomain, LBMSimulation
export create_lbm_simulation, equilibrium_distribution
export collision_step!, streaming_step!, apply_boundary_conditions!
export run_lbm_simulation!
export calculate_velocity_field, calculate_density_field
export extract_wall_shear_stress, calculate_reynolds_number, calculate_womersley_number
export create_straight_tube, create_stenosis_geometry, create_bifurcation_geometry
export create_curved_vessel
export carreau_yasuda_viscosity, hematocrit_viscosity_correction
export validate_poiseuille_flow

end # module
