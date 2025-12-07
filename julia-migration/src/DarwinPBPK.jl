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
include("DarwinPBPK/compartments/gi_detailed.jl")  # NEW: 7-segment GI tract model (PK-Sim standard)
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
include("DarwinPBPK/compartments/lipoprotein_binding.jl")  # NEW: HDL/LDL/VLDL drug binding
include("DarwinPBPK/compartments/rbc_transporters.jl")  # NEW: RBC membrane transporters (Band3, OAT, OCT, GLUT1)
include("DarwinPBPK/compartments/disease_state_binding.jl")  # NEW: Disease state PK adjustments
include("DarwinPBPK/compartments/mab_pbpk.jl")  # NEW: mAb PBPK scaffold (FcRn, TMDD)
include("DarwinPBPK/compartments/immunoglobulin_isotypes.jl")  # NEW: IgM, IgA, IgE isotypes + complement
include("DarwinPBPK/compartments/acute_phase_response.jl")  # NEW: IL-6, CRP, SAA acute phase dynamics
include("DarwinPBPK/compartments/rbc_aging.jl")  # NEW: RBC age distribution, RDW effects
include("DarwinPBPK/compartments/spleen_res_clearance.jl")  # NEW: Splenic macrophage clearance
include("DarwinPBPK/compartments/circadian_effects.jl")  # NEW: Chronopharmacokinetic rhythms
include("DarwinPBPK/compartments/disease_ontology_pk.jl")  # NEW: DOID + ICD-10/11 PK integration
include("DarwinPBPK/compartments/anemia_polycythemia.jl")  # NEW: Hematocrit-dependent PK adjustments
include("DarwinPBPK/compartments/plasma_viscosity.jl")  # NEW: Blood rheology and viscosity effects
include("DarwinPBPK/compartments/blood_compartment_integrated.jl")  # NEW: Integration layer for all blood modules
include("DarwinPBPK/compartments/transporter_ontogeny.jl")  # NEW: Pediatric transporter ontogeny (Hunt 2024)
include("DarwinPBPK/compartments/rbc_dynamics_integrated.jl")  # NEW: Closed-loop RBC dynamics (hematopoiesis, EPO feedback, organ clearance)
# include("DarwinPBPK/image_analysis/leukocyte_fractal_analysis.jl")  # Temporarily disabled - needs Images package
include("DarwinPBPK/ode_solver.jl")
include("DarwinPBPK/dataset_generation.jl")
include("DarwinPBPK/dynamic_gnn.jl")  # FASE 2 ✅
include("DarwinPBPK/training.jl")     # FASE 2 ✅

# ML modules
include("DarwinPBPK/ml/multimodal_encoder.jl")  # FASE 3 ✅ (Real implementation with MolecularGraph.jl)
include("DarwinPBPK/ml/evidential.jl")          # FASE 3 ✅
include("DarwinPBPK/ml/bayesian_uq.jl")         # Q1 2025 ✅ (Bayesian UQ)
include("DarwinPBPK/ml/turing_pbpk.jl")         # Q1 2025 ✅ (Turing.jl PBPK Models)
include("DarwinPBPK/ml/chemberta_bridge.jl")    # Q1 2025 ✅ (ChemBERTa via PyCall)
include("DarwinPBPK/ml/dmpnn.jl")               # Q1 2025 ✅ (D-MPNN encoder)
include("DarwinPBPK/ml/mc_dropout.jl")          # Q1 2025 ✅ (MC-Dropout for UQ)
include("DarwinPBPK/ml/calibration.jl")         # Q1 2025 ✅ (Calibration metrics)
include("DarwinPBPK/ml/neural_ode.jl")          # Q1 2025 ✅ (Neural ODE PBPK)

# Validation (FASE 4)
include("DarwinPBPK/validation.jl")              # FASE 4 ✅
include("DarwinPBPK/validation/external_validation.jl")  # Q1 2025 ✅ (External validation protocol)

# MedLang DSL (First Real Implementation)
include("DarwinPBPK/medlang/MedLang.jl")        # MedLang DSL ✅

# Semantic Web Layer (FAIR Data) - must be before REST API
include("DarwinPBPK/semantic/SemanticCore.jl")  # JSON-LD + OBO Foundry ✅

# API (FASE 5) - depends on SemanticCore
include("DarwinPBPK/api/rest_api.jl")           # FASE 5 ✅

# External Validation Datasets
include("DarwinPBPK/external_datasets.jl")      # OSP, Zenodo, PK-DB ✅

# Clinical Data Integration (Q1 2025)
include("DarwinPBPK/clinical/ClinicalDataIntegration.jl")  # Clinical trial data import/NCA ✅

# Allometric Scaling (Q1 2025)
include("DarwinPBPK/scaling/AllometricScaling.jl")  # Multi-species scaling/IVIVE ✅

# Regulatory Reports (Q1 2025)
include("DarwinPBPK/regulatory/RegulatoryReports.jl")  # FDA/EMA report generation ✅

# Re-export principais
using .PatientProfile
using .CompartmentModels
using .FractalBlood
using .GIDetailed
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
using .LipoproteinBinding
using .RBCTransporters
using .DiseaseStateBinding
using .mAbPBPK
using .ImmunoglobulinIsotypes
using .AcutePhaseResponse
using .RBCAging
using .SpleenRESClearance
using .CircadianEffects
using .DiseaseOntologyPK
using .AnemiaPolycythemia
using .PlasmaViscosity
using .BloodCompartmentIntegrated
using .TransporterOntogeny
using .RBCDynamicsIntegrated
# using .LeukocyteFractalAnalysis  # Temporarily disabled
using .ODEPBPKSolver
using .DatasetGeneration
using .DynamicGNN
using .Training
using .MultimodalEncoder
using .Evidential
using .BayesianUQ
using .TuringPBPK
using .ChemBERTaBridge
using .DMPNN
using .MCDropout
using .Calibration
using .NeuralODE
using .Validation
using .ExternalValidation
using .RESTAPI
using .MedLang
using .SemanticCore
using .ExternalDatasets
using .ClinicalDataIntegration
using .AllometricScaling
using .RegulatoryReports

# Export MedLang DSL functions
export parse_medlang, compile_model, compile_file, simulate_medlang
export load_medlang, generate_julia_module, validate_model, describe_model

# Export Bayesian UQ functions
export BayesianPBPKModel, PosteriorResult, ParameterPrior
export sample_posterior, variational_inference, default_pbpk_priors
export credible_interval, posterior_predictive, uncertainty_calibration
export VariationalPosterior, sample_variational, create_clearance_model

# Export Turing.jl PBPK functions (Q1 2025)
export bayesian_one_compartment, bayesian_two_compartment, bayesian_pbpk_5organ
export bayesian_population_pk, sample_nuts, sample_advi, sample_fast
export posterior_summary, check_diagnostics, trace_data
export quick_bayesian_pk, predict_with_uncertainty

# Export ChemBERTa functions (Q1 2025)
export ChemBERTaEncoder  # From ChemBERTaBridge module
# Note: ChemBERTaBridge.initialize!, encode, encode_batch available via module

# Export D-MPNN functions (Q1 2025)
export DMPNNEncoder, DMPNNConv, encode_molecule, encode_molecules
export AttentionReadout, StandardGNNEncoder

# Export MC-Dropout functions (Q1 2025)
export MCDropoutWrapper, UncertaintyResult
export predict_with_uncertainty, predict_batch_with_uncertainty
export decompose_uncertainty, calibration_metrics, prediction_intervals
export pbpk_predict_with_uncertainty, pk_parameter_uncertainty
export add_dropout, has_dropout

# Export Calibration functions (Q1 2025)
export CalibrationResult, CalibrationMetrics
export expected_calibration_error, maximum_calibration_error
export reliability_diagram, calibration_curve
export continuous_ranked_probability_score, crps_gaussian, sharpness
export recalibrate_predictions, isotonic_calibration
export full_calibration_analysis, pk_calibration_analysis, multi_parameter_calibration

# Export External Validation functions (Q1 2025)
export ExternalValidationDataset, BlindValidationResult, ValidationReport
export load_pkdb_dataset, create_validation_dataset, example_validation_dataset
export run_blind_validation, generate_validation_report
export check_fda_criteria, get_failed_compounds

# Export SOTA Multimodal Encoder V2 (Q1 2025)
export SOTAMultimodalEncoderV2, create_sota_encoder, quick_encode, encoder_info
export CHEMBERTA_DIM, DMPNN_DIM

# Export Bootstrap validation functions (Q1 2025)
export BootstrapResult, bootstrap_metric
export gmfe_with_ci, afe_with_ci, aafe_with_ci, r_squared_with_ci
export percent_within_fold_with_ci, regulatory_metrics_with_ci
export format_bootstrap_result, latex_metrics_row

# Export Neural ODE functions (Q1 2025)
export NeuralODEPBPK, NeuralDynamics, HybridDynamics
export PhysicsInformedNeuralODE, AugmentedNeuralODE
export predict_concentrations, predict_pk, extract_pk_parameters
export create_neural_ode_pbpk, quick_predict, train_neural_ode!
export NODE_HIDDEN_DIM, NODE_LATENT_DIM

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

# Export Transporter Ontogeny functions (Q1+ SOTA 2024 - Hunt et al.)
export OntogenyProfile, TransporterOntogenyData, PediatricAge, AgeCategory
export OntogenyFunction, SigmoidalMaturation, BiphasicMaturation, LinearMaturation
export calculate_maturation, get_transporter_ontogeny, apply_ontogeny_scaling
export get_renal_transporter_ontogeny, get_hepatic_transporter_ontogeny
export get_intestinal_transporter_ontogeny, get_bbb_transporter_ontogeny
export predict_pediatric_clearance, generate_ontogeny_curve, validate_ontogeny_prediction
export get_pediatric_disease_modifiers, get_doid_pediatric_conditions, get_icd_codes_for_condition
export RENAL_TRANSPORTER_ONTOGENY, HEPATIC_TRANSPORTER_ONTOGENY
export INTESTINAL_TRANSPORTER_ONTOGENY, BBB_TRANSPORTER_ONTOGENY
export OBO_FOUNDRY_PREFIXES, UBERON_TERMS, CL_TERMS, GO_TERMS, PR_TERMS
export ICD_CLASSIFICATIONS, DOID_PEDIATRIC_CONDITIONS
export export_transporter_to_jsonld

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

# Export Lipoprotein Binding functions
export LipoproteinProfile, DrugLipoproteinBinding, LipoproteinBindingResult
export calculate_lipoprotein_binding, calculate_fu_with_lipoproteins
export get_lipoprotein_drug_data, create_lipoprotein_profile
export apply_disease_lipoproteins, partition_drug_to_lipoproteins
export LIPOPROTEIN_DRUG_DATABASE, NORMAL_LIPOPROTEIN_LEVELS
export HYPERCHOLESTEROLEMIA_PROFILE, DIABETIC_DYSLIPIDEMIA_PROFILE

# Export RBC Transporter functions
export RBCTransporter, RBCTransporterProfile, DrugRBCTransport, RBCTransportResult
export calculate_rbc_transport, calculate_rbc_accumulation
export get_transporter_drug_data, create_rbc_transporter_profile
export apply_disease_transporters, simulate_rbc_uptake
export RBC_TRANSPORTER_DATABASE, RBC_DRUG_TRANSPORT_DATABASE
export NORMAL_RBC_TRANSPORTERS, SICKLE_CELL_TRANSPORTERS

# Export Disease State Binding functions
export DiseaseState, PlasmaProteinState, BindingAdjustments, DiseaseBindingResult
export calculate_binding_adjustments, calculate_adjusted_fu
export get_disease_state, create_disease_binding_model
export apply_disease_adjustments, get_clinical_examples
export DISEASE_BINDING_DATABASE, CKD_STAGES, CIRRHOSIS_STAGES
export PREGNANCY_TRIMESTERS, SEPSIS_SEVERITY

# Export mAb PBPK functions
export mAbProperties, TargetProperties, TMDDParameters, FcRnParameters
export mAbPKResult, mAbSimulationResult
export calculate_tmdd_clearance, calculate_fcrn_recycling
export simulate_mab_pk, calculate_target_occupancy
export get_mab_data, get_target_data, create_mab_model
export apply_ada_effect, calculate_immunogenicity_risk
export MAB_DATABASE, TARGET_DATABASE
export FCRN_PARAMETERS, TMDD_DEFAULT_PARAMS

# Export Immunoglobulin Isotype functions
export ImmunoglobulinProperties, ComplementSystem, ImmuneComplex
export create_igg_subclass, create_igm, create_iga, create_ige
export calculate_complement_activation, calculate_immune_complex_clearance
export calculate_isotype_clearance, calculate_fc_receptor_binding
export IMMUNOGLOBULIN_DATABASE, COMPLEMENT_PARAMETERS, FC_RECEPTOR_DATABASE

# Export Acute Phase Response functions
export AcutePhaseState, CytokineProfile, AcutePhaseProtein
export create_acute_phase_state, simulate_acute_phase!, calculate_protein_changes
export apply_acute_phase_binding, get_time_course, get_cytokine_profile
export predict_pk_changes, get_dosing_recommendation
export ACUTE_PHASE_PROTEINS, CYTOKINE_EFFECTS

# Export RBC Aging functions
export RBCPopulation, RBCAgeDistribution, ReticulocyteState
export create_normal_rbc_population, create_disease_population
export calculate_age_weighted_transport, calculate_rdw_effect
export simulate_rbc_turnover, get_age_distribution
export RBC_AGE_PARAMETERS, RETICULOCYTE_FACTORS

# Export Spleen RES Clearance functions
export SpleenState, RESCapacity, MacrophagePool
export create_normal_spleen, create_disease_spleen
export calculate_res_clearance, calculate_splenic_uptake
export apply_splenectomy, calculate_fcr_mediated_clearance
export SPLEEN_PARAMETERS, RES_TISSUE_WEIGHTS

# Export Circadian Effects functions
export CircadianState, CircadianParameter
export create_default_parameters, get_circadian_factor, simulate_circadian_variation
export calculate_optimal_dosing_time, get_chronotype_adjustment
export calculate_circadian_pk_effect, calculate_chronotherapy_benefit
export CIRCADIAN_PARAMETERS, CHRONOTYPE_SHIFTS

# Export Disease Ontology PK functions
export DiseaseCode, DiseasePKProfile, OntologyMapping
export get_pk_adjustments_by_doid, get_pk_adjustments_by_icd10
export get_pk_adjustments_by_icd11, search_disease_pk
export map_disease_hierarchy, get_pk_with_fallback
export combine_disease_profiles, list_supported_diseases
export get_disease_summary
export DOID_PK_DATABASE, ICD10_TO_DOID, ICD11_TO_DOID
export DISEASE_HIERARCHY

# Export Anemia/Polycythemia functions
export HematologicalState, AnemiaProfile, PolycythemiaProfile
export RBCIndices, ReticulocyteState, EPOState
export create_normal_hematology, create_anemia_state, create_polycythemia_state
export calculate_hematocrit_correction, calculate_blood_plasma_ratio
export calculate_vd_correction, calculate_clearance_correction
export apply_anemia_pk_adjustments, apply_polycythemia_pk_adjustments
export calculate_rbc_partitioning, estimate_reticulocyte_effect
export simulate_epo_therapy, calculate_transfusion_effect
export ANEMIA_PROFILES, POLYCYTHEMIA_PROFILES, RBC_PARTITION_DATABASE
export NORMAL_HEMATOLOGY, EPO_PARAMETERS

# Export RBC Dynamics Integrated functions (Closed-loop hematopoiesis)
export RBCDynamicsState, HematopoiesisParams, EPOFeedback, OrganRBCClearance
export IronMetabolism, BilirubinState, RBCTurnoverResult, RBCAgeDistribution
export initialize_rbc_dynamics, update_rbc_dynamics!, simulate_rbc_dynamics
export calculate_organ_rbc_clearance, calculate_epo_response
export calculate_reticulocyte_release, calculate_age_weighted_transport
export get_effective_hematocrit, get_rbc_mediated_clearance
export apply_rbc_dynamics_to_pk, create_rbc_ode_system
export NORMAL_RBC_PARAMS, DISEASE_RBC_DYNAMICS

# Export Plasma Viscosity functions
export ViscosityState, BloodRheology, PerfusionState
export CarreauYasudaParams, FahrauesLindqvistParams
export calculate_plasma_viscosity, calculate_blood_viscosity
export calculate_carreau_yasuda_viscosity, calculate_apparent_viscosity
export calculate_fahraeus_lindqvist_effect, calculate_microvascular_viscosity
export calculate_perfusion_effect, calculate_hepatic_flow
export calculate_renal_flow, calculate_tissue_perfusion
export apply_hyperviscosity_adjustments, apply_hemodilution_adjustments
export create_normal_rheology, create_hyperviscosity_state
export estimate_viscosity_from_hematocrit, estimate_viscosity_from_proteins
export NORMAL_VISCOSITY, SHEAR_RATE_RANGES, HYPERVISCOSITY_SYNDROMES
export CARREAU_YASUDA_NORMAL, FAHRAEUS_LINDQVIST_PARAMS

# Export Blood Compartment Integrated functions
export BloodCompartmentState, DrugBloodProperties, IntegratedPKAdjustments
export create_blood_state, create_blood_state_from_disease
export update_blood_state!, get_current_adjustments
export calculate_integrated_pk_parameters
export apply_time_step!, get_ode_parameters
export validate_blood_state, get_integration_summary
# Disease ontology bridge functions
export create_state_from_doid_profile, map_ontology_to_binding_state
export get_binding_adjustments_by_doid, calculate_fu_from_disease_code
export get_binding_adjustments_by_icd10, create_state_from_icd10

# Export ODE Blood State Integration functions
export DynamicPBPKParams, BloodStateODECallback
export solve_with_blood_state, simulate_with_blood_state
export create_blood_state_callback

# Export External Datasets functions
export ExternalDataSource, PKDBStudy, OSPRecord, ZenodoRecord
export list_available_datasets, load_osp_ddi, load_osp_pediatrics
export load_zenodo_betalactam, query_pkdb_api, list_pkdb_studies
export get_ddi_auc_ratios, get_pediatric_clearance
export validate_against_external_data, summarize_external_datasets
export AVAILABLE_DATASETS

# Export Clinical Data Integration functions (Q1 2025)
export ClinicalStudy, PKObservation, Subject, DosingRecord
export NCAResult, BioequivalenceResult, ValidationMetrics
export load_clinical_study, load_sdtm_pc, load_adam_adpc
export compute_nca, compute_nca_population
export validate_model_against_clinical, bioequivalence_assessment
export literature_pk_entry, aggregate_literature_data, LiteraturePKEntry
export ClinicalPKDatabase, add_study!, query_studies

# Export Allometric Scaling functions (Q1 2025)
export Species, SpeciesData, AllometricModel, ScalingResult
export MOUSE, RAT, RABBIT, DOG, CYNOMOLGUS_MONKEY, RHESUS_MONKEY, MINIPIG, HUMAN
export get_species_data, scale_clearance, scale_volume, scale_half_life
export simple_allometry, rule_of_exponents, brain_weight_correction
export mlp_correction, dedrick_plot, predict_human_pk, predict_first_in_human_dose
export ivive_clearance, hepatocyte_scaling, microsomal_scaling
export calculate_allometric_exponent, fit_allometric_model
export SPECIES_DATABASE, STANDARD_EXPONENTS

# Export Regulatory Reports functions (Q1 2025)
export RegulatoryReport, ModelDocumentation, ValidationSummary
export DDIAssessment, SensitivitySummary, PopulationCovariates
export generate_fda_report, generate_ema_report, generate_ich_m12_report
export generate_validation_summary, generate_ddi_assessment
export export_report_markdown, export_report_json
export FDA_ACCEPTANCE_CRITERIA, EMA_ACCEPTANCE_CRITERIA

# Export 7-Segment GI Tract functions
export GISegment, GITract, DrugGIProperties
export create_gi_tract, calculate_gi_absorption, simulate_gi_transit
export calculate_ionization_fraction, calculate_permeability
export calculate_bcs_class, calculate_fg
export example_drug_metoprolol, example_drug_atorvastatin, example_drug_lisinopril
export GI_SEGMENTS, SEGMENT_NAMES
export gi7_ode_system!, solve_gi7, simulate_gi7, compare_gi_models
export GI_STOMACH_IDX, GI_DUODENUM_IDX, GI_JEJUNUM_UPPER_IDX, GI_JEJUNUM_LOWER_IDX
export GI_ILEUM_UPPER_IDX, GI_ILEUM_LOWER_IDX, GI_COLON_IDX, GI_SEGMENT_INDICES

# Export FractalBlood + ODE integration functions
export FractalBloodParams, PBPKParamsWithFractal
export integrate_fractal_blood!, create_fractal_pbpk_params
export fractal_transit_time_distribution, apply_fractal_dispersion

# Export core ODE solver functions
export PBPKParams, solve, simulate, validate_mass_conservation
export PBPK_ORGANS, NUM_ORGANS

# Export Oral absorption functions
export OralParams, effective_f, solve_oral, simulate_oral

# Export PK-Sim Database functions
export load_compartment_database, create_liver_compartment, create_kidney_compartment
export create_brain_compartment, create_adipose_compartment, create_blood_compartment
export create_all_compartments, validate_compartment_parameters

end # module
