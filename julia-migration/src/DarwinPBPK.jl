"""
Darwin PBPK Platform - Julia Implementation

SOTA + Disruptive + Nature-tier PBPK modeling platform.

Author: Dr. Demetrios Agourakis
Date: November 2025
"""

module DarwinPBPK

# Core modules
include("DarwinPBPK/ode_solver.jl")
include("DarwinPBPK/dataset_generation.jl")
include("DarwinPBPK/dynamic_gnn.jl")  # FASE 2 ✅
include("DarwinPBPK/training.jl")     # FASE 2 ✅

# ML modules
# include("DarwinPBPK/ml/multimodal_encoder.jl")  # FASE 3 ✅ (requer Transformers.jl)
include("DarwinPBPK/ml/evidential.jl")          # FASE 3 ✅

# Validation (FASE 4)
include("DarwinPBPK/validation.jl")              # FASE 4 ✅

# API (FASE 5)
include("DarwinPBPK/api/rest_api.jl")           # FASE 5 ✅

# Re-export principais
using .ODEPBPKSolver
using .DatasetGeneration
using .DynamicGNN
using .Training
# using .MultimodalEncoder  # Opcional (requer Transformers.jl)
using .Evidential
using .Validation
using .RESTAPI

end # module

