# Tier 1 Implementation Microtasks

## Deep Analysis Chain-of-Thought

### Architecture Assessment

After reviewing the codebase, I identified the following integration points:

1. **dynamic_gnn.jl:540-600** - `forward_with_smiles()` already calls `mol_encoder(smiles)` expecting a multimodal encoder
2. **multimodal_encoder.jl** - Has SMILESEncoder (GRU) but NOT real ChemBERTa
3. **bayesian_uq.jl** - Has manual MCMC but NOT Turing.jl NUTS
4. **validation.jl** - Has AFE/AAFE but missing bootstrap CIs and external validation protocol

### Dependencies Analysis
- Project.toml already includes: `Turing`, `PyCall`, `Flux`, `GraphNeuralNetworks`
- Missing: `Transformers.jl` (optional - can use PyCall to HuggingFace)
- Missing: `AdvancedMH.jl` (optional - Turing provides NUTS)

---

## 1. ChemBERTa Encoder Integration

### Goal
Replace placeholder SMILESEncoder with real ChemBERTa via PyCall to HuggingFace Transformers.

### Reasoning
- ChemBERTa provides 768d embeddings pre-trained on 77M SMILES
- Character-level GRU is ~50% less effective for molecular property prediction
- D-MPNN adds directed message passing for better reaction/metabolite prediction

---

### Microtask 1.1: ChemBERTa Python Bridge Setup
**File**: `julia-migration/src/DarwinPBPK/ml/chemberta_bridge.jl` (NEW)
**Effort**: 2-3 hours
**Dependencies**: PyCall, HuggingFace transformers

**Steps**:
1. Create Python environment initialization for transformers
2. Load `seyonec/ChemBERTa-zinc-base-v1` model
3. Implement tokenizer wrapper
4. Implement forward pass with caching
5. Handle batch encoding efficiently

**Code skeleton**:
```julia
module ChemBERTaBridge

using PyCall

const transformers = PyNULL()
const tokenizer = PyNULL()
const model = PyNULL()

function __init__()
    copy!(transformers, pyimport("transformers"))
    # Load model lazily on first use
end

function load_chemberta!(model_name::String="seyonec/ChemBERTa-zinc-base-v1")
    copy!(tokenizer, transformers.AutoTokenizer.from_pretrained(model_name))
    copy!(model, transformers.AutoModel.from_pretrained(model_name))
    model.eval()  # Inference mode
end

function encode(smiles::String)::Vector{Float32}
    inputs = tokenizer(smiles, return_tensors="pt", padding=true, truncation=true, max_length=512)
    outputs = model(inputs...)
    # CLS token embedding
    return Vector{Float32}(outputs.last_hidden_state[0, 0, :].detach().numpy())
end

function encode_batch(smiles_batch::Vector{String})::Matrix{Float32}
    inputs = tokenizer(smiles_batch, return_tensors="pt", padding=true, truncation=true, max_length=512)
    outputs = model(inputs...)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return Matrix{Float32}(cls_embeddings')
end

export load_chemberta!, encode, encode_batch

end
```

**Verification**:
```julia
using DarwinPBPK.ChemBERTaBridge
load_chemberta!()
emb = encode("CCO")  # Ethanol
@assert length(emb) == 768
```

---

### Microtask 1.2: D-MPNN Implementation
**File**: `julia-migration/src/DarwinPBPK/ml/dmpnn.jl` (NEW)
**Effort**: 4-5 hours
**Dependencies**: GraphNeuralNetworks.jl, MolecularGraph.jl

**Background**:
D-MPNN (Directed Message Passing) from Yang et al. 2019 is SOTA for molecular property prediction.
Key difference from GAT: messages pass along directed bonds, preventing "self-messaging".

**Steps**:
1. Create directed edge construction from SMILES
2. Implement DMPNNConv layer (bond-to-atom, atom-to-bond messages)
3. Implement atom-level and bond-level aggregation
4. Add readout layer with attention pooling

**Code skeleton**:
```julia
module DMPNN

using Flux
using GraphNeuralNetworks
using MolecularGraph

struct DMPNNConv
    W_message::Dense
    W_atom::Dense
end

@functor DMPNNConv

function DMPNNConv(node_dim::Int, edge_dim::Int, hidden_dim::Int)
    W_message = Dense(node_dim + edge_dim => hidden_dim, relu)
    W_atom = Dense(hidden_dim + node_dim => hidden_dim)
    return DMPNNConv(W_message, W_atom)
end

# Directed message passing: bond → atom → bond
function (layer::DMPNNConv)(g::GNNGraph, x::AbstractMatrix, e::AbstractMatrix)
    # ... implementation
end

struct DMPNNEncoder
    conv_layers::Vector{DMPNNConv}
    readout::Dense
    depth::Int
end

# ... full implementation

export DMPNNEncoder, DMPNNConv

end
```

**Verification**:
```julia
encoder = DMPNNEncoder()
emb = encoder("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
@assert length(emb) == 256
```

---

### Microtask 1.3: Update MultimodalMolecularEncoder
**File**: `julia-migration/src/DarwinPBPK/ml/multimodal_encoder.jl` (MODIFY)
**Effort**: 2 hours
**Dependencies**: ChemBERTaBridge, DMPNN

**Steps**:
1. Add ChemBERTa as primary SMILES encoder (768d)
2. Replace GAT with D-MPNN (256d)
3. Keep QuantumEncoder (128d)
4. Update CrossAttentionFusion for new dimensions
5. Add fallback to GRU if PyCall fails

**Changes**:
```julia
struct MultimodalMolecularEncoderV2
    chemberta::ChemBERTaEncoder  # NEW: 768d
    dmpnn::DMPNNEncoder          # NEW: 256d (replaces GAT)
    quantum::QuantumEncoder      # KEEP: 128d
    fusion::CrossAttentionFusion # UPDATE: 1152d input → 512d output
    fallback_smiles::SMILESEncoder  # Fallback if PyCall unavailable
end
```

---

### Microtask 1.4: Integration Tests
**File**: `julia-migration/test/test_multimodal_encoder.jl` (NEW/MODIFY)
**Effort**: 1-2 hours

**Test cases**:
1. Single SMILES encoding (valid)
2. Batch encoding (10 molecules)
3. Invalid SMILES handling (graceful degradation)
4. Dimension consistency checks
5. GPU compatibility (if CUDA available)
6. Performance benchmark (target: <50ms per molecule)

---

## 2. Bayesian UQ with Turing.jl

### Goal
Replace manual MCMC with Turing.jl NUTS for proper posterior inference.

### Reasoning
- NUTS (No-U-Turn Sampler) is ~10× more efficient than Metropolis-Hastings
- Turing.jl provides automatic differentiation for gradient-based sampling
- Essential for regulatory acceptance (credible intervals)

---

### Microtask 2.1: Turing.jl PBPK Model
**File**: `julia-migration/src/DarwinPBPK/ml/turing_pbpk.jl` (NEW)
**Effort**: 3-4 hours
**Dependencies**: Turing, DifferentialEquations

**Steps**:
1. Define `@model` macro for one-compartment PK
2. Define `@model` macro for multi-compartment PBPK
3. Implement prior selection from `default_pbpk_priors()`
4. Connect ODE solver to likelihood
5. Implement posterior predictive sampling

**Code skeleton**:
```julia
module TuringPBPK

using Turing
using DifferentialEquations
using ..ODEPBPKSolver: solve_pbpk, PBPKParams

@model function bayesian_one_compartment(
    obs_conc::Vector{Float64},
    times::Vector{Float64},
    dose::Float64;
    volume::Float64 = 50.0
)
    # Priors
    CL ~ LogNormal(log(10.0), 0.5)
    sigma ~ truncated(Normal(0.0, 0.3), 0.0, Inf)
    
    # Model prediction
    pred_conc = dose ./ volume .* exp.(-CL ./ volume .* times)
    
    # Likelihood (log-normal errors common in PK)
    for i in eachindex(obs_conc)
        obs_conc[i] ~ LogNormal(log(pred_conc[i] + 1e-10), sigma)
    end
end

@model function bayesian_pbpk_full(
    obs_conc::Matrix{Float64},  # [n_organs, n_times]
    times::Vector{Float64},
    dose::Float64;
    fixed_params::PBPKParams
)
    # Priors for key parameters
    CL_hepatic ~ LogNormal(log(10.0), 0.5)
    CL_renal ~ LogNormal(log(5.0), 0.5)
    Kp_liver ~ LogNormal(log(2.0), 0.5)
    Kp_kidney ~ LogNormal(log(1.5), 0.5)
    sigma ~ truncated(Normal(0.0, 0.2), 0.0, Inf)
    
    # Construct PBPKParams
    params = PBPKParams(
        clearance_hepatic = CL_hepatic,
        clearance_renal = CL_renal,
        partition_coeffs = Dict(
            "liver" => Kp_liver,
            "kidney" => Kp_kidney,
            # ... other organs from fixed_params
        ),
        # ... rest from fixed_params
    )
    
    # Solve ODE
    sol = solve_pbpk(params, dose; t_span=(0.0, maximum(times)))
    
    # Extract predictions at observation times
    pred = # interpolate sol at times
    
    # Likelihood
    for i in axes(obs_conc, 1)
        for j in axes(obs_conc, 2)
            obs_conc[i, j] ~ LogNormal(log(pred[i, j] + 1e-10), sigma)
        end
    end
end

function sample_nuts(model; n_samples=2000, n_warmup=1000, n_chains=4)
    sampler = NUTS(0.65)  # Target acceptance rate
    chain = sample(model, sampler, MCMCThreads(), n_samples, n_chains; 
                   init_params=nothing, discard_initial=n_warmup)
    return chain
end

function sample_vi(model; n_iterations=10000)
    q = vi(model, ADVI(10, n_iterations))
    return q
end

export bayesian_one_compartment, bayesian_pbpk_full
export sample_nuts, sample_vi

end
```

**Verification**:
```julia
# Generate synthetic data
times = collect(0.0:0.5:24.0)
true_CL = 12.0
dose = 100.0
obs_conc = dose ./ 50.0 .* exp.(-true_CL ./ 50.0 .* times) .+ 0.1 .* randn(length(times))

# Sample posterior
model = bayesian_one_compartment(obs_conc, times, dose)
chain = sample_nuts(model; n_samples=1000, n_warmup=500, n_chains=2)

# Check recovery
@assert abs(mean(chain[:CL]) - true_CL) / true_CL < 0.1  # Within 10%
```

---

### Microtask 2.2: MC-Dropout for GNN
**File**: `julia-migration/src/DarwinPBPK/ml/mc_dropout.jl` (NEW)
**Effort**: 2-3 hours

**Steps**:
1. Create `MCDropoutGNN` wrapper for DynamicPBPKGNN
2. Implement `forward_with_dropout()` that keeps dropout active during inference
3. Implement `predict_with_uncertainty()` that runs N forward passes
4. Compute mean prediction and epistemic uncertainty

**Code skeleton**:
```julia
module MCDropout

using Flux
using Statistics

struct MCDropoutWrapper{M}
    model::M
    dropout_rate::Float64
end

function forward_with_dropout(wrapper::MCDropoutWrapper, x; training::Bool=true)
    # Force dropout to be active even during inference
    Flux.trainmode!(wrapper.model, training)
    return wrapper.model(x)
end

function predict_with_uncertainty(
    wrapper::MCDropoutWrapper,
    inputs;
    n_samples::Int = 50
)
    predictions = []
    for _ in 1:n_samples
        pred = forward_with_dropout(wrapper, inputs; training=true)
        push!(predictions, pred)
    end
    
    pred_stack = cat(predictions..., dims=ndims(predictions[1]) + 1)
    
    mean_pred = mean(pred_stack, dims=ndims(pred_stack))
    std_pred = std(pred_stack, dims=ndims(pred_stack))
    
    return (
        mean = dropdims(mean_pred, dims=ndims(pred_stack)),
        std = dropdims(std_pred, dims=ndims(pred_stack)),
        samples = pred_stack
    )
end

export MCDropoutWrapper, predict_with_uncertainty

end
```

---

### Microtask 2.3: Deep Ensembles
**File**: `julia-migration/src/DarwinPBPK/ml/deep_ensembles.jl` (NEW)
**Effort**: 2-3 hours

**Steps**:
1. Create `EnsemblePBPKGNN` that holds N independently trained models
2. Implement ensemble prediction with uncertainty
3. Add diversity-promoting training objective (optional)

---

### Microtask 2.4: Update BayesianUQ Module
**File**: `julia-migration/src/DarwinPBPK/ml/bayesian_uq.jl` (MODIFY)
**Effort**: 2 hours

**Changes**:
1. Import and re-export TuringPBPK functions
2. Add `BayesianInference` unified interface
3. Add `DualModeUQ` that combines MCMC + VI
4. Update calibration metrics (ECE with proper binning)

---

### Microtask 2.5: Calibration Metrics
**File**: `julia-migration/src/DarwinPBPK/ml/calibration.jl` (NEW)
**Effort**: 2 hours

**Steps**:
1. Implement Expected Calibration Error (ECE) with M bins
2. Implement reliability diagram data generation
3. Implement sharpness metrics
4. Implement CRPS (Continuous Ranked Probability Score)

**Code skeleton**:
```julia
function expected_calibration_error(
    predicted_means::Vector{Float64},
    predicted_stds::Vector{Float64},
    observed::Vector{Float64};
    n_bins::Int = 10
)
    # Compute z-scores
    z_scores = (observed .- predicted_means) ./ predicted_stds
    
    # Expected coverage at each bin
    coverage_levels = range(0.1, 0.95, length=n_bins)
    expected = Float64[]
    observed_coverage = Float64[]
    
    for level in coverage_levels
        z_crit = quantile(Normal(), (1 + level) / 2)
        push!(expected, level)
        push!(observed_coverage, mean(abs.(z_scores) .< z_crit))
    end
    
    # ECE = weighted average of |expected - observed|
    ece = mean(abs.(expected .- observed_coverage))
    
    return ece
end
```

---

## 3. Validation Metrics Enhancement

### Goal
Complete the regulatory metrics suite with bootstrap CIs and external validation protocol.

---

### Microtask 3.1: Bootstrap Confidence Intervals
**File**: `julia-migration/src/DarwinPBPK/validation.jl` (MODIFY)
**Effort**: 2 hours

**Steps**:
1. Implement `bootstrap_metric()` generic function
2. Add bootstrap CIs to AFE, AAFE, GMFE
3. Add `regulatory_metrics_with_ci()` function

**Code addition**:
```julia
function bootstrap_metric(
    metric_fn::Function,
    pred::Vector{Float64},
    obs::Vector{Float64};
    n_bootstrap::Int = 1000,
    ci_level::Float64 = 0.95
)
    n = length(pred)
    bootstrap_values = Float64[]
    
    for _ in 1:n_bootstrap
        indices = rand(1:n, n)  # Sample with replacement
        val = metric_fn(pred[indices], obs[indices])
        if isfinite(val)
            push!(bootstrap_values, val)
        end
    end
    
    α = (1 - ci_level) / 2
    ci_lower = quantile(bootstrap_values, α)
    ci_upper = quantile(bootstrap_values, 1 - α)
    
    return (
        estimate = metric_fn(pred, obs),
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        se = std(bootstrap_values)
    )
end

function regulatory_metrics_with_ci(
    pred::Vector{Float64},
    obs::Vector{Float64};
    n_bootstrap::Int = 1000
)
    gmfe_ci = bootstrap_metric(geometric_mean_fold_error, pred, obs; n_bootstrap)
    afe_ci = bootstrap_metric(average_fold_error, pred, obs; n_bootstrap)
    aafe_ci = bootstrap_metric(absolute_average_fold_error, pred, obs; n_bootstrap)
    
    return Dict(
        "GMFE" => gmfe_ci,
        "AFE" => afe_ci,
        "AAFE" => aafe_ci,
        # ... other metrics
    )
end
```

---

### Microtask 3.2: External Validation Protocol
**File**: `julia-migration/src/DarwinPBPK/validation/external_validation.jl` (NEW)
**Effort**: 3-4 hours

**Steps**:
1. Define `ExternalValidationDataset` struct
2. Implement data loader for PK-DB format
3. Implement blind prediction protocol (no parameter tuning)
4. Implement comprehensive validation report

**Code skeleton**:
```julia
module ExternalValidation

using CSV
using DataFrames
using JSON
using Dates

struct ExternalValidationDataset
    name::String
    source::String  # "PK-DB", "DrugBank", etc.
    compounds::Vector{String}
    smiles::Vector{String}
    observed_pk::Dict{String, Any}  # CL, Vd, F, etc.
    metadata::Dict{String, Any}
end

struct BlindValidationResult
    dataset_name::String
    n_compounds::Int
    predictions::Dict{String, Vector{Float64}}
    observed::Dict{String, Vector{Float64}}
    metrics::Dict{String, Any}
    timestamp::DateTime
    model_version::String
end

function load_pkdb_dataset(filepath::String)::ExternalValidationDataset
    # Parse PK-DB JSON/CSV format
    # ...
end

function run_blind_validation(
    model,
    dataset::ExternalValidationDataset;
    parameters::Vector{Symbol} = [:CL, :Vd, :F]
)::BlindValidationResult
    # Predict without any parameter tuning
    predictions = Dict{String, Vector{Float64}}()
    
    for param in parameters
        preds = Float64[]
        for (smiles, _) in zip(dataset.smiles, dataset.compounds)
            pred = predict_parameter(model, smiles, param)
            push!(preds, pred)
        end
        predictions[string(param)] = preds
    end
    
    # Compute metrics
    metrics = Dict{String, Any}()
    for param in parameters
        pred = predictions[string(param)]
        obs = dataset.observed_pk[string(param)]
        
        metrics[string(param)] = regulatory_metrics_with_ci(pred, obs)
    end
    
    return BlindValidationResult(
        dataset.name,
        length(dataset.compounds),
        predictions,
        dataset.observed_pk,
        metrics,
        now(),
        "DarwinPBPK v2.10.0"
    )
end

function generate_validation_report(result::BlindValidationResult; output_path::String)
    # Generate comprehensive markdown/HTML report
    # Include:
    # - Summary statistics table
    # - Scatter plots (pred vs obs)
    # - Fold error distributions
    # - Calibration plots (if UQ available)
    # - Individual compound results
end

export ExternalValidationDataset, BlindValidationResult
export load_pkdb_dataset, run_blind_validation, generate_validation_report

end
```

---

### Microtask 3.3: Uncertainty-Aware Regulatory Metrics
**File**: `julia-migration/src/DarwinPBPK/validation.jl` (MODIFY)
**Effort**: 1-2 hours

**Steps**:
1. Add `gmfe_with_uncertainty()` that accounts for prediction uncertainty
2. Add `aafe_with_uncertainty()` 
3. Add probability of meeting FDA criteria

**Code addition**:
```julia
"""
GMFE with prediction uncertainty (Bayesian version).

If predictions have uncertainty (credible intervals), compute
the probability that the GMFE meets regulatory criteria.
"""
function gmfe_with_uncertainty(
    pred_samples::Matrix{Float64},  # [n_mc_samples, n_compounds]
    obs::Vector{Float64};
    threshold::Float64 = 2.0
)
    n_mc = size(pred_samples, 1)
    gmfe_samples = Float64[]
    
    for i in 1:n_mc
        pred = pred_samples[i, :]
        gmfe = geometric_mean_fold_error(pred, obs)
        if isfinite(gmfe)
            push!(gmfe_samples, gmfe)
        end
    end
    
    prob_acceptable = mean(gmfe_samples .< threshold)
    
    return (
        mean = mean(gmfe_samples),
        std = std(gmfe_samples),
        ci_lower = quantile(gmfe_samples, 0.025),
        ci_upper = quantile(gmfe_samples, 0.975),
        prob_acceptable = prob_acceptable
    )
end
```

---

## Summary: Priority Order

### Week 1 (High Priority)
| # | Microtask | Effort | Impact |
|---|-----------|--------|--------|
| 2.1 | Turing.jl PBPK Model | 4h | Critical for UQ |
| 3.1 | Bootstrap CIs | 2h | Quick win |
| 1.1 | ChemBERTa Bridge | 3h | R² improvement |

### Week 2 (Medium Priority)
| # | Microtask | Effort | Impact |
|---|-----------|--------|--------|
| 1.2 | D-MPNN | 5h | SOTA architecture |
| 2.2 | MC-Dropout | 3h | Epistemic UQ |
| 3.2 | External Validation | 4h | Publication-critical |

### Week 3 (Completion)
| # | Microtask | Effort | Impact |
|---|-----------|--------|--------|
| 1.3 | Update Multimodal | 2h | Integration |
| 2.3 | Deep Ensembles | 3h | Robust UQ |
| 2.5 | Calibration | 2h | Reliability |
| 1.4 | Integration Tests | 2h | Quality |

---

## Expected Outcomes

After implementing all microtasks:

| Metric | Current | Expected |
|--------|---------|----------|
| CL R² | 0.18 | 0.45-0.55 |
| 95% CI coverage | N/A | >90% |
| GMFE (blind) | 1.64 | <1.8 |
| ECE | N/A | <0.05 |

---

## File Checklist

### New Files
- [ ] `ml/chemberta_bridge.jl`
- [ ] `ml/dmpnn.jl`
- [ ] `ml/turing_pbpk.jl`
- [ ] `ml/mc_dropout.jl`
- [ ] `ml/deep_ensembles.jl`
- [ ] `ml/calibration.jl`
- [ ] `validation/external_validation.jl`

### Modified Files
- [ ] `ml/multimodal_encoder.jl`
- [ ] `ml/bayesian_uq.jl`
- [ ] `validation.jl`
- [ ] `DarwinPBPK.jl` (exports)

### Test Files
- [ ] `test/test_multimodal_encoder.jl`
- [ ] `test/test_turing_pbpk.jl`
- [ ] `test/test_external_validation.jl`
