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

# Re-export principais
using .PatientProfile
using .CompartmentModels
using .FractalBlood
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

# Export MedLang DSL functions
export parse_medlang, compile_model, compile_file, simulate_medlang
export load_medlang, generate_julia_module, validate_model, describe_model

# Export Bayesian UQ functions
export BayesianPBPKModel, PosteriorResult, ParameterPrior
export sample_posterior, variational_inference, default_pbpk_priors
export credible_interval, posterior_predictive, uncertainty_calibration
export VariationalPosterior, sample_variational, create_clearance_model

end # module
