# ===========================================================================
# EPISTEMIC TYPES MODULE
# ===========================================================================
# Julia implementation of Sounio's epistemic computing types.
#
# This module provides:
# - Knowledge{T} type for uncertainty-aware values
# - Confidence propagation through computations
# - Serialization to/from Sounio EpistemicValue format
# - Integration with PBPK parameter uncertainty
#
# Based on Sounio v0.97.0 epistemic computing specification.
#
# Author: Dr. Sounio Agourakis
# Date: January 2026
# Version: 1.0.0
# ===========================================================================

module EpistemicTypes

using Dates
using Statistics
using JSON3

export Knowledge, ConfidenceLevel, Provenance
export propagate_confidence, combine_knowledge, validate_confidence
export to_epistemic_value, from_epistemic_value
export with_confidence, with_provenance
export EpistemicPBPKParams, EpistemicDrugData
export ConfidenceGate, check_confidence_gate
export ProvenanceChain, add_provenance

# ===========================================================================
# Core Types
# ===========================================================================

"""
    ConfidenceLevel

Enumeration of confidence levels for epistemic values.
Maps to Sounio's ConfidenceLevel enum.
"""
@enum ConfidenceLevel begin
    VERY_LOW = 1       # ε < 0.3 - Highly uncertain, experimental
    LOW = 2            # 0.3 ≤ ε < 0.5 - Limited confidence
    MODERATE = 3       # 0.5 ≤ ε < 0.7 - Acceptable for exploratory
    HIGH = 4           # 0.7 ≤ ε < 0.9 - Good confidence
    VERY_HIGH = 5      # ε ≥ 0.9 - High confidence, validated
end

"""
    confidence_level(ε::Float64) -> ConfidenceLevel

Convert numeric confidence to ConfidenceLevel enum.
"""
function confidence_level(ε::Float64)::ConfidenceLevel
    @assert 0.0 ≤ ε ≤ 1.0 "Confidence must be in [0, 1]"

    if ε < 0.3
        return VERY_LOW
    elseif ε < 0.5
        return LOW
    elseif ε < 0.7
        return MODERATE
    elseif ε < 0.9
        return HIGH
    else
        return VERY_HIGH
    end
end

export confidence_level

"""
    Provenance

Origin and history of an epistemic value.
"""
@enum Provenance begin
    MEASURED      # Direct measurement (experimental data)
    COMPUTED      # Computed from other values
    ESTIMATED     # Estimated (model prediction, ML inference)
    LITERATURE    # From literature/database
    ASSUMED       # Assumed/default value
    INTERPOLATED  # Interpolated between known values
    EXTRAPOLATED  # Extrapolated beyond known range
    CONSENSUS     # Expert consensus/guideline value
end

"""
    ProvenanceChain

Chain of provenance tracking transformation history.
"""
struct ProvenanceChain
    entries::Vector{NamedTuple{(:provenance, :source, :timestamp), Tuple{Provenance, String, DateTime}}}
end

ProvenanceChain() = ProvenanceChain([])

function add_provenance(chain::ProvenanceChain, prov::Provenance, source::String)
    new_entries = copy(chain.entries)
    push!(new_entries, (provenance=prov, source=source, timestamp=now()))
    ProvenanceChain(new_entries)
end

function Base.show(io::IO, chain::ProvenanceChain)
    print(io, "ProvenanceChain[$(length(chain.entries)) entries]")
end

# ===========================================================================
# Knowledge{T} Type
# ===========================================================================

"""
    Knowledge{T}

Epistemic value type matching Sounio's Knowledge<T>.

A Knowledge value wraps a base value with:
- `value::T` - The underlying value
- `confidence::Float64` - Confidence in [0, 1] (ε in epistemic theory)
- `provenance::Provenance` - Origin of the value
- `timestamp::DateTime` - When the value was created/measured
- `metadata::Dict{String, Any}` - Additional metadata

# Confidence Semantics
- ε = 1.0: Perfect knowledge (deterministic)
- ε = 0.9+: Very high confidence (validated experimental)
- ε = 0.7-0.9: High confidence (good quality data)
- ε = 0.5-0.7: Moderate confidence (acceptable)
- ε < 0.5: Low confidence (uncertain)

# Examples
```julia
# Create epistemic value
cl = Knowledge(10.5, 0.85, MEASURED)

# Access value and confidence
value(cl)        # 10.5
confidence(cl)   # 0.85
is_confident(cl) # true (ε ≥ 0.7)

# Propagate through computation
vd = Knowledge(70.0, 0.90, LITERATURE)
ke = cl / vd     # Confidence: min(0.85, 0.90) = 0.85
```
"""
struct Knowledge{T}
    value::T
    confidence::Float64
    provenance::Provenance
    timestamp::DateTime
    metadata::Dict{String, Any}

    function Knowledge{T}(
        value::T,
        confidence::Float64,
        provenance::Provenance,
        timestamp::DateTime,
        metadata::Dict{String, Any}
    ) where T
        @assert 0.0 ≤ confidence ≤ 1.0 "Confidence must be in [0, 1], got $confidence"
        new{T}(value, confidence, provenance, timestamp, metadata)
    end
end

# Convenience constructors
function Knowledge(value::T, confidence::Float64, provenance::Provenance) where T
    Knowledge{T}(value, confidence, provenance, now(), Dict{String, Any}())
end

function Knowledge(value::T, confidence::Float64) where T
    Knowledge{T}(value, confidence, COMPUTED, now(), Dict{String, Any}())
end

function Knowledge(value::T) where T
    Knowledge{T}(value, 1.0, ASSUMED, now(), Dict{String, Any}())
end

# Accessors
value(k::Knowledge) = k.value
confidence(k::Knowledge) = k.confidence
provenance(k::Knowledge) = k.provenance
timestamp(k::Knowledge) = k.timestamp
metadata(k::Knowledge) = k.metadata

export value, confidence, provenance, timestamp, metadata

# Predicates
is_confident(k::Knowledge; threshold::Float64=0.7) = k.confidence ≥ threshold
is_measured(k::Knowledge) = k.provenance == MEASURED
is_computed(k::Knowledge) = k.provenance == COMPUTED
is_estimated(k::Knowledge) = k.provenance == ESTIMATED

export is_confident, is_measured, is_computed, is_estimated

# ===========================================================================
# Knowledge Modifiers
# ===========================================================================

"""
    with_confidence(k::Knowledge, ε::Float64) -> Knowledge

Create new Knowledge with updated confidence.
"""
function with_confidence(k::Knowledge{T}, ε::Float64) where T
    Knowledge{T}(k.value, ε, k.provenance, k.timestamp, k.metadata)
end

"""
    with_provenance(k::Knowledge, prov::Provenance) -> Knowledge

Create new Knowledge with updated provenance.
"""
function with_provenance(k::Knowledge{T}, prov::Provenance) where T
    Knowledge{T}(k.value, k.confidence, prov, k.timestamp, k.metadata)
end

"""
    with_metadata(k::Knowledge, key::String, value) -> Knowledge

Create new Knowledge with added metadata.
"""
function with_metadata(k::Knowledge{T}, key::String, val) where T
    new_meta = copy(k.metadata)
    new_meta[key] = val
    Knowledge{T}(k.value, k.confidence, k.provenance, k.timestamp, new_meta)
end

export with_metadata

# ===========================================================================
# Confidence Propagation
# ===========================================================================

"""
    propagate_confidence(inputs::Vector{<:Knowledge}) -> Float64

Propagate confidence through computation using minimum rule.

In epistemic computing, the output confidence is bounded by the
minimum confidence of all inputs (weakest link principle).
"""
function propagate_confidence(inputs::Vector{<:Knowledge})::Float64
    if isempty(inputs)
        return 1.0
    end
    return minimum(k.confidence for k in inputs)
end

"""
    propagate_confidence(k1::Knowledge, k2::Knowledge) -> Float64

Propagate confidence for binary operation.
"""
function propagate_confidence(k1::Knowledge, k2::Knowledge)::Float64
    min(k1.confidence, k2.confidence)
end

"""
    combine_knowledge(values::Vector{Knowledge{T}}; method=:weighted_mean) -> Knowledge{T}

Combine multiple epistemic values into one.

Methods:
- `:mean` - Simple mean, confidence = mean of confidences
- `:weighted_mean` - Confidence-weighted mean
- `:conservative` - Use value with highest confidence
- `:minimum` - Use minimum value, min confidence
- `:maximum` - Use maximum value, min confidence
"""
function combine_knowledge(
    values::Vector{Knowledge{T}};
    method::Symbol = :weighted_mean
) where T <: Number
    if isempty(values)
        error("Cannot combine empty vector of Knowledge values")
    end

    if length(values) == 1
        return values[1]
    end

    if method == :mean
        v = mean(value(k) for k in values)
        ε = mean(confidence(k) for k in values)
        return Knowledge(v, ε, COMPUTED)

    elseif method == :weighted_mean
        weights = [confidence(k) for k in values]
        total_weight = sum(weights)
        v = sum(value(k) * confidence(k) for k in values) / total_weight
        ε = sum(w^2 for w in weights) / (total_weight * length(values))
        return Knowledge(v, sqrt(ε), COMPUTED)

    elseif method == :conservative
        best = argmax(k -> k.confidence, values)
        return with_provenance(values[best], COMPUTED)

    elseif method == :minimum
        min_val = minimum(value(k) for k in values)
        min_conf = minimum(confidence(k) for k in values)
        return Knowledge(min_val, min_conf, COMPUTED)

    elseif method == :maximum
        max_val = maximum(value(k) for k in values)
        min_conf = minimum(confidence(k) for k in values)
        return Knowledge(max_val, min_conf, COMPUTED)

    else
        error("Unknown combination method: $method")
    end
end

# ===========================================================================
# Arithmetic Operations with Confidence Propagation
# ===========================================================================

# Binary operations propagate confidence via minimum rule
function Base.:+(k1::Knowledge{T}, k2::Knowledge{T}) where T <: Number
    Knowledge(value(k1) + value(k2), propagate_confidence(k1, k2), COMPUTED)
end

function Base.:-(k1::Knowledge{T}, k2::Knowledge{T}) where T <: Number
    Knowledge(value(k1) - value(k2), propagate_confidence(k1, k2), COMPUTED)
end

function Base.:*(k1::Knowledge{T}, k2::Knowledge{T}) where T <: Number
    Knowledge(value(k1) * value(k2), propagate_confidence(k1, k2), COMPUTED)
end

function Base.:/(k1::Knowledge{T}, k2::Knowledge{T}) where T <: Number
    Knowledge(value(k1) / value(k2), propagate_confidence(k1, k2), COMPUTED)
end

function Base.:^(k::Knowledge{T}, n::Integer) where T <: Number
    Knowledge(value(k)^n, confidence(k), COMPUTED)
end

# Operations with scalars preserve confidence
function Base.:+(k::Knowledge{T}, x::Number) where T <: Number
    Knowledge(value(k) + x, confidence(k), COMPUTED)
end

function Base.:+(x::Number, k::Knowledge{T}) where T <: Number
    Knowledge(x + value(k), confidence(k), COMPUTED)
end

function Base.:*(k::Knowledge{T}, x::Number) where T <: Number
    Knowledge(value(k) * x, confidence(k), COMPUTED)
end

function Base.:*(x::Number, k::Knowledge{T}) where T <: Number
    Knowledge(x * value(k), confidence(k), COMPUTED)
end

function Base.:/(k::Knowledge{T}, x::Number) where T <: Number
    Knowledge(value(k) / x, confidence(k), COMPUTED)
end

# Unary operations
function Base.:-(k::Knowledge{T}) where T <: Number
    Knowledge(-value(k), confidence(k), COMPUTED)
end

function Base.abs(k::Knowledge{T}) where T <: Number
    Knowledge(abs(value(k)), confidence(k), COMPUTED)
end

function Base.sqrt(k::Knowledge{T}) where T <: Number
    Knowledge(sqrt(value(k)), confidence(k), COMPUTED)
end

function Base.exp(k::Knowledge{T}) where T <: Number
    Knowledge(exp(value(k)), confidence(k), COMPUTED)
end

function Base.log(k::Knowledge{T}) where T <: Number
    Knowledge(log(value(k)), confidence(k), COMPUTED)
end

# Comparison (compare values, ignore confidence)
Base.:<(k1::Knowledge, k2::Knowledge) = value(k1) < value(k2)
Base.:>(k1::Knowledge, k2::Knowledge) = value(k1) > value(k2)
Base.:(==)(k1::Knowledge, k2::Knowledge) = value(k1) == value(k2) && confidence(k1) == confidence(k2)
Base.isapprox(k1::Knowledge, k2::Knowledge; kwargs...) = isapprox(value(k1), value(k2); kwargs...)

# ===========================================================================
# Confidence Gating
# ===========================================================================

"""
    ConfidenceGate

Gate that blocks operations below minimum confidence threshold.
"""
struct ConfidenceGate
    min_confidence::Float64
    action::Symbol  # :error, :warn, :clamp, :pass

    function ConfidenceGate(min_confidence::Float64, action::Symbol=:error)
        @assert 0.0 ≤ min_confidence ≤ 1.0
        @assert action in (:error, :warn, :clamp, :pass)
        new(min_confidence, action)
    end
end

"""
    check_confidence_gate(gate::ConfidenceGate, k::Knowledge) -> Knowledge

Check Knowledge against confidence gate.

Actions:
- `:error` - Throw error if below threshold
- `:warn` - Log warning and continue
- `:clamp` - Clamp confidence to threshold
- `:pass` - Pass through unchanged
"""
function check_confidence_gate(gate::ConfidenceGate, k::Knowledge{T}) where T
    if confidence(k) >= gate.min_confidence
        return k
    end

    if gate.action == :error
        error("Confidence $(confidence(k)) below gate threshold $(gate.min_confidence)")
    elseif gate.action == :warn
        @warn "Knowledge confidence $(confidence(k)) below threshold $(gate.min_confidence)"
        return k
    elseif gate.action == :clamp
        return with_confidence(k, gate.min_confidence)
    else  # :pass
        return k
    end
end

"""
    validate_confidence(k::Knowledge; min_confidence=0.0) -> Bool

Validate that Knowledge meets minimum confidence requirement.
"""
function validate_confidence(k::Knowledge; min_confidence::Float64=0.0)::Bool
    confidence(k) >= min_confidence
end

# ===========================================================================
# Serialization to/from Sounio Format
# ===========================================================================

"""
    to_epistemic_value(k::Knowledge) -> Dict

Convert Knowledge to Sounio EpistemicValue JSON format.
"""
function to_epistemic_value(k::Knowledge)::Dict{String, Any}
    Dict{String, Any}(
        "value" => value(k),
        "confidence" => confidence(k),
        "confidence_level" => string(confidence_level(confidence(k))),
        "provenance" => string(provenance(k)),
        "timestamp" => Dates.format(timestamp(k), "yyyy-mm-ddTHH:MM:SS"),
        "metadata" => metadata(k)
    )
end

"""
    from_epistemic_value(::Type{T}, dict::Dict) -> Knowledge{T}

Parse Sounio EpistemicValue JSON to Knowledge.
"""
function from_epistemic_value(::Type{T}, dict::Dict) where T
    val = convert(T, dict["value"])
    conf = Float64(dict["confidence"])

    prov_str = get(dict, "provenance", "COMPUTED")
    prov = parse_provenance(prov_str)

    ts_str = get(dict, "timestamp", nothing)
    ts = ts_str !== nothing ? DateTime(ts_str, "yyyy-mm-ddTHH:MM:SS") : now()

    meta = get(dict, "metadata", Dict{String, Any}())

    Knowledge{T}(val, conf, prov, ts, meta)
end

function parse_provenance(s::String)::Provenance
    s_upper = uppercase(s)
    if s_upper == "MEASURED"
        return MEASURED
    elseif s_upper == "COMPUTED"
        return COMPUTED
    elseif s_upper == "ESTIMATED"
        return ESTIMATED
    elseif s_upper == "LITERATURE"
        return LITERATURE
    elseif s_upper == "ASSUMED"
        return ASSUMED
    elseif s_upper == "INTERPOLATED"
        return INTERPOLATED
    elseif s_upper == "EXTRAPOLATED"
        return EXTRAPOLATED
    elseif s_upper == "CONSENSUS"
        return CONSENSUS
    else
        return COMPUTED
    end
end

# ===========================================================================
# Pretty Printing
# ===========================================================================

function Base.show(io::IO, k::Knowledge{T}) where T
    level = confidence_level(confidence(k))
    print(io, "Knowledge{$T}($(value(k)), ε=$(confidence(k)), $(provenance(k)), $(level))")
end

function Base.show(io::IO, ::MIME"text/plain", k::Knowledge{T}) where T
    level = confidence_level(confidence(k))
    println(io, "Knowledge{$T}")
    println(io, "  Value:      $(value(k))")
    println(io, "  Confidence: $(confidence(k)) ($(level))")
    println(io, "  Provenance: $(provenance(k))")
    println(io, "  Timestamp:  $(timestamp(k))")
    if !isempty(metadata(k))
        println(io, "  Metadata:   $(length(metadata(k))) entries")
    end
end

# ===========================================================================
# Epistemic PBPK Types
# ===========================================================================

"""
    EpistemicPBPKParams

PBPK parameters with epistemic uncertainty tracking.
"""
struct EpistemicPBPKParams
    cl_hepatic::Knowledge{Float64}
    cl_renal::Knowledge{Float64}
    vd::Knowledge{Float64}
    ka::Knowledge{Float64}
    f_oral::Knowledge{Float64}
    kp_values::Dict{String, Knowledge{Float64}}

    # Aggregate confidence
    min_confidence::Float64
    mean_confidence::Float64
end

function EpistemicPBPKParams(;
    cl_hepatic::Knowledge{Float64},
    cl_renal::Knowledge{Float64},
    vd::Knowledge{Float64},
    ka::Knowledge{Float64},
    f_oral::Knowledge{Float64},
    kp_values::Dict{String, Knowledge{Float64}} = Dict{String, Knowledge{Float64}}()
)
    all_params = [cl_hepatic, cl_renal, vd, ka, f_oral, values(kp_values)...]
    min_conf = minimum(confidence(p) for p in all_params)
    mean_conf = mean(confidence(p) for p in all_params)

    EpistemicPBPKParams(cl_hepatic, cl_renal, vd, ka, f_oral, kp_values, min_conf, mean_conf)
end

"""
    EpistemicDrugData

Drug data with epistemic uncertainty tracking.
"""
struct EpistemicDrugData
    name::String
    mw::Knowledge{Float64}
    logp::Knowledge{Float64}
    fu_plasma::Knowledge{Float64}
    bp_ratio::Knowledge{Float64}
    pka_acidic::Union{Knowledge{Float64}, Nothing}
    pka_basic::Union{Knowledge{Float64}, Nothing}

    # Aggregate confidence
    min_confidence::Float64
end

function EpistemicDrugData(;
    name::String,
    mw::Knowledge{Float64},
    logp::Knowledge{Float64},
    fu_plasma::Knowledge{Float64},
    bp_ratio::Knowledge{Float64} = Knowledge(0.55, 0.5, ASSUMED),
    pka_acidic::Union{Knowledge{Float64}, Nothing} = nothing,
    pka_basic::Union{Knowledge{Float64}, Nothing} = nothing
)
    params = [mw, logp, fu_plasma, bp_ratio]
    pka_acidic !== nothing && push!(params, pka_acidic)
    pka_basic !== nothing && push!(params, pka_basic)

    min_conf = minimum(confidence(p) for p in params)

    EpistemicDrugData(name, mw, logp, fu_plasma, bp_ratio, pka_acidic, pka_basic, min_conf)
end

"""
    to_sounio_epistemic(params::EpistemicPBPKParams) -> Dict

Convert EpistemicPBPKParams to Sounio-compatible JSON.
"""
function to_sounio_epistemic(params::EpistemicPBPKParams)::Dict{String, Any}
    Dict{String, Any}(
        "cl_hepatic" => to_epistemic_value(params.cl_hepatic),
        "cl_renal" => to_epistemic_value(params.cl_renal),
        "vd" => to_epistemic_value(params.vd),
        "ka" => to_epistemic_value(params.ka),
        "f_oral" => to_epistemic_value(params.f_oral),
        "kp_values" => Dict(k => to_epistemic_value(v) for (k, v) in params.kp_values),
        "min_confidence" => params.min_confidence,
        "mean_confidence" => params.mean_confidence
    )
end

export to_sounio_epistemic

end  # module EpistemicTypes
