# ===========================================================================
# ML-BASED TRANSPORTER SUBSTRATE PREDICTION
# ===========================================================================
# Predicts transporter substrates from molecular structure using:
# 1. Multimodal encoder (SMILES + GNN + Quantum)
# 2. Multi-label classification heads for each transporter
#
# This bridges the ML representations with mechanistic PBPK models:
# - Predicted transporter affinities → GI absorption model
# - Predicted P-gp substrate probability → Efflux ratio estimation
# - Predicted Km values → Saturation kinetics
#
# Training data: DrugBank, ChEMBL transporter annotations
# ===========================================================================

module TransporterPredictor

using Flux
using Functors: @functor
using Statistics

# Include parent encoder
include("multimodal_encoder.jl")
using .MultimodalEncoder

# ===========================================================================
# TRANSPORTER DEFINITIONS
# ===========================================================================

# Intestinal transporters relevant for oral absorption
const INTESTINAL_TRANSPORTERS = [
    # Uptake transporters
    :PEPT1,    # SLC15A1 - Peptides, β-lactams, ACE inhibitors
    :OCT1,     # SLC22A1 - Cationic drugs (metformin)
    :OCT3,     # SLC22A3 - Organic cations
    :OATP2B1,  # SLCO2B1 - Organic anions, statins
    :ENT1,     # SLC29A1 - Nucleosides
    :ENT2,     # SLC29A2 - Nucleosides
    :MCT1,     # SLC16A1 - Monocarboxylates
    :LAT1,     # SLC7A5 - Large neutral amino acids
    :LAT2,     # SLC7A8 - Amino acids, gabapentin
    :ASBT,     # SLC10A2 - Bile acids (ileum-specific)

    # Efflux transporters
    :PGP,      # ABCB1 - P-glycoprotein
    :BCRP,     # ABCG2 - Breast cancer resistance protein
    :MRP2,     # ABCC2 - Multidrug resistance protein 2
]

const N_TRANSPORTERS = length(INTESTINAL_TRANSPORTERS)

# Transporter to index mapping
const TRANSPORTER_IDX = Dict(t => i for (i, t) in enumerate(INTESTINAL_TRANSPORTERS))

# ===========================================================================
# TRANSPORTER PREDICTION HEAD
# ===========================================================================

"""
Multi-label classification head for transporter substrate prediction.

For each transporter, predicts:
- is_substrate: Binary probability (0-1)
- affinity_class: Low/Medium/High (for Km estimation)
- is_inhibitor: Whether drug inhibits the transporter
"""
struct TransporterHead
    shared::Chain
    substrate_heads::Vector{Dense}  # One per transporter
    affinity_heads::Vector{Dense}   # Km class prediction
    inhibitor_heads::Vector{Dense}  # Inhibitor prediction
end

@functor TransporterHead

function TransporterHead(input_dim::Int; hidden_dim::Int = 256)
    # Shared layers
    shared = Chain(
        Dense(input_dim, hidden_dim, relu),
        Dropout(0.3),
        Dense(hidden_dim, hidden_dim, relu),
        Dropout(0.2),
    )

    # Per-transporter heads
    substrate_heads = [Dense(hidden_dim, 1, σ) for _ in 1:N_TRANSPORTERS]
    affinity_heads = [Dense(hidden_dim, 3) for _ in 1:N_TRANSPORTERS]  # Low/Med/High
    inhibitor_heads = [Dense(hidden_dim, 1, σ) for _ in 1:N_TRANSPORTERS]

    return TransporterHead(shared, substrate_heads, affinity_heads, inhibitor_heads)
end

"""
Predict transporter interactions for a molecule embedding.

Returns:
- substrate_probs: Vector of substrate probabilities [N_TRANSPORTERS]
- affinity_classes: Matrix of affinity class logits [3, N_TRANSPORTERS]
- inhibitor_probs: Vector of inhibitor probabilities [N_TRANSPORTERS]
"""
function (head::TransporterHead)(x::Vector{Float32})
    # Shared representation
    h = head.shared(x)

    # Per-transporter predictions
    substrate_probs = Float32[]
    affinity_logits = zeros(Float32, 3, N_TRANSPORTERS)
    inhibitor_probs = Float32[]

    for (i, (sub_head, aff_head, inh_head)) in enumerate(zip(
        head.substrate_heads, head.affinity_heads, head.inhibitor_heads
    ))
        push!(substrate_probs, sub_head(h)[1])
        affinity_logits[:, i] = aff_head(h)
        push!(inhibitor_probs, inh_head(h)[1])
    end

    return (
        substrate_probs = substrate_probs,
        affinity_logits = affinity_logits,
        inhibitor_probs = inhibitor_probs
    )
end

# ===========================================================================
# COMPLETE TRANSPORTER PREDICTOR MODEL
# ===========================================================================

"""
Complete transporter substrate prediction model.

Combines:
- EnhancedMultimodalEncoder: SMILES + GNN + Quantum → 512d embedding
- TransporterHead: Multi-label classification for all transporters
"""
struct TransporterPredictorModel
    encoder::EnhancedMultimodalEncoder
    head::TransporterHead
end

@functor TransporterPredictorModel

function TransporterPredictorModel(; use_gnn::Bool = true, use_quantum::Bool = true)
    encoder = EnhancedMultimodalEncoder(use_gnn = use_gnn, use_quantum = use_quantum)
    head = TransporterHead(FUSION_DIM)
    return TransporterPredictorModel(encoder, head)
end

"""
Predict transporter substrates from SMILES.
"""
function (model::TransporterPredictorModel)(smiles::String)
    # Encode molecule
    embedding = model.encoder(smiles)

    # Predict transporter interactions
    predictions = model.head(embedding)

    return predictions
end

# ===========================================================================
# TRANSPORTER PREDICTION RESULT
# ===========================================================================

"""
Structured result for transporter predictions with interpretation.
"""
struct TransporterPrediction
    transporter::Symbol
    gene::String
    is_substrate::Bool
    substrate_probability::Float64
    affinity_class::Symbol  # :low, :medium, :high
    estimated_km_uM::Float64
    is_inhibitor::Bool
    inhibitor_probability::Float64
end

# Gene symbols for transporters
const TRANSPORTER_GENES = Dict(
    :PEPT1 => "SLC15A1",
    :OCT1 => "SLC22A1",
    :OCT3 => "SLC22A3",
    :OATP2B1 => "SLCO2B1",
    :ENT1 => "SLC29A1",
    :ENT2 => "SLC29A2",
    :MCT1 => "SLC16A1",
    :LAT1 => "SLC7A5",
    :LAT2 => "SLC7A8",
    :ASBT => "SLC10A2",
    :PGP => "ABCB1",
    :BCRP => "ABCG2",
    :MRP2 => "ABCC2",
)

# Typical Km ranges (μM) for each affinity class
const KM_RANGES = Dict(
    :low => (500.0, 2000.0),    # Low affinity
    :medium => (50.0, 500.0),   # Medium affinity
    :high => (1.0, 50.0),       # High affinity
)

"""
Convert raw predictions to structured TransporterPrediction results.
"""
function interpret_predictions(
    predictions::NamedTuple;
    substrate_threshold::Float64 = 0.5,
    inhibitor_threshold::Float64 = 0.5
)::Vector{TransporterPrediction}
    results = TransporterPrediction[]

    for (i, transporter) in enumerate(INTESTINAL_TRANSPORTERS)
        # Substrate probability
        sub_prob = Float64(predictions.substrate_probs[i])
        is_substrate = sub_prob >= substrate_threshold

        # Affinity class (argmax of logits)
        aff_logits = predictions.affinity_logits[:, i]
        aff_class_idx = argmax(aff_logits)
        aff_class = [:low, :medium, :high][aff_class_idx]

        # Estimate Km from affinity class
        km_range = KM_RANGES[aff_class]
        estimated_km = sqrt(km_range[1] * km_range[2])  # Geometric mean

        # Inhibitor probability
        inh_prob = Float64(predictions.inhibitor_probs[i])
        is_inhibitor = inh_prob >= inhibitor_threshold

        push!(results, TransporterPrediction(
            transporter,
            TRANSPORTER_GENES[transporter],
            is_substrate,
            sub_prob,
            aff_class,
            estimated_km,
            is_inhibitor,
            inh_prob
        ))
    end

    return results
end

"""
Get predicted substrates as a simple list.
"""
function get_substrate_transporters(
    predictions::Vector{TransporterPrediction};
    min_probability::Float64 = 0.5
)::Vector{Symbol}
    return [p.transporter for p in predictions
            if p.is_substrate && p.substrate_probability >= min_probability]
end

"""
Get P-gp efflux ratio estimate from prediction.

Uses the P-gp substrate probability and affinity to estimate
an in vitro efflux ratio for the GI absorption model.
"""
function estimate_pgp_efflux_ratio(predictions::Vector{TransporterPrediction})::Float64
    pgp_pred = findfirst(p -> p.transporter == :PGP, predictions)

    if pgp_pred === nothing
        return 1.0  # No P-gp prediction
    end

    pred = predictions[pgp_pred]

    if !pred.is_substrate
        return 1.0  # Not a P-gp substrate
    end

    # Estimate ER based on affinity class and probability
    base_er = Dict(:low => 3.0, :medium => 10.0, :high => 30.0)[pred.affinity_class]

    # Scale by probability
    er = 1.0 + (base_er - 1.0) * pred.substrate_probability

    return er
end

# ===========================================================================
# INTEGRATION WITH GI ABSORPTION MODEL
# ===========================================================================

"""
Convert ML predictions to GI model parameters.

Returns a NamedTuple compatible with simulate_oral_absorption_enhanced:
- transporters: List of predicted uptake transporters
- is_pgp_substrate: Boolean
- pgp_efflux_ratio: Estimated ER
- carrier_km_values: Dict of transporter → Km estimates
"""
function predictions_to_gi_params(predictions::Vector{TransporterPrediction})
    # Get uptake transporters (exclude efflux)
    efflux_transporters = [:PGP, :BCRP, :MRP2]
    uptake_transporters = [p.transporter for p in predictions
                          if p.is_substrate && !(p.transporter in efflux_transporters)]

    # P-gp prediction
    pgp_pred = findfirst(p -> p.transporter == :PGP, predictions)
    is_pgp = pgp_pred !== nothing && predictions[pgp_pred].is_substrate
    pgp_er = estimate_pgp_efflux_ratio(predictions)

    # Km values for each transporter
    km_values = Dict{Symbol, Float64}()
    for p in predictions
        if p.is_substrate
            km_values[p.transporter] = p.estimated_km_uM
        end
    end

    return (
        uptake_transporters = uptake_transporters,
        is_pgp_substrate = is_pgp,
        pgp_efflux_ratio = pgp_er,
        carrier_km_values = km_values,
        predictions = predictions  # Full predictions for advanced use
    )
end

# ===========================================================================
# HIGH-LEVEL API
# ===========================================================================

"""
Predict transporter interactions for a drug from SMILES.

This is the main entry point for GI absorption predictions.

Returns structured predictions that can be used directly with
the mechanistic GI absorption model.

Example:
```julia
result = predict_transporters("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # Ibuprofen
println("P-gp substrate: ", result.is_pgp_substrate)
println("Uptake transporters: ", result.uptake_transporters)
println("Efflux ratio: ", result.pgp_efflux_ratio)
```
"""
function predict_transporters(smiles::String; model::Union{TransporterPredictorModel, Nothing} = nothing)
    # Use default model if not provided
    if model === nothing
        model = TransporterPredictorModel()
    end

    # Get raw predictions
    raw_preds = model(smiles)

    # Interpret predictions
    predictions = interpret_predictions(raw_preds)

    # Convert to GI model parameters
    gi_params = predictions_to_gi_params(predictions)

    return gi_params
end

"""
Batch predict for multiple drugs.
"""
function predict_transporters_batch(
    smiles_list::Vector{String};
    model::Union{TransporterPredictorModel, Nothing} = nothing
)
    return [predict_transporters(s; model = model) for s in smiles_list]
end

# ===========================================================================
# PRETRAINED MODEL LOADING
# ===========================================================================

using BSON

"""
Save trained model to BSON file.
"""
function save_model(model::TransporterPredictorModel, path::String)
    BSON.@save path model = model
    @info "Model saved to $path"
end

"""
Load pretrained model from BSON file.
"""
function load_model(path::String)::TransporterPredictorModel
    BSON.@load path model
    @info "Model loaded from $path"
    return model
end

# Export public API
export TransporterPredictorModel, TransporterPrediction, TransporterHead
export predict_transporters, predict_transporters_batch
export interpret_predictions, get_substrate_transporters, estimate_pgp_efflux_ratio
export predictions_to_gi_params
export save_model, load_model
export INTESTINAL_TRANSPORTERS, N_TRANSPORTERS, TRANSPORTER_GENES

end # module
