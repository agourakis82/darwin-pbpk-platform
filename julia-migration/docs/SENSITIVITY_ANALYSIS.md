# Sensitivity Analysis Module

**Module**: `DarwinPBPK.SensitivityAnalysis`  
**Location**: `/julia-migration/src/DarwinPBPK/compartments/sensitivity_analysis.jl`  
**Date**: 2025-12-05

## Overview

Comprehensive sensitivity analysis module implementing both local and global methods for PBPK model parameter sensitivity assessment. Includes integration with the coagulation cascade model and other PBPK compartments.

## Features

### Local Sensitivity Analysis
- **One-At-a-Time (OAT)**: Fast local screening around nominal values
- **Elasticity Coefficients**: Percentage change relationships
- **Normalized Sensitivity Coefficients (NSC)**: Scaled finite differences

### Global Sensitivity Analysis
- **Sobol Indices**: Variance-based decomposition (first-order S₁, total Sᴛ)
- **Morris Screening**: Elementary effects for efficient parameter screening
- **PRCC**: Partial Rank Correlation Coefficient for monotonic relationships

### Advanced Sampling
- **Latin Hypercube Sampling (LHS)**: Space-filling quasi-random samples
- **Sobol Sequences**: Low-discrepancy quasi-random sequences
- **Morris Trajectories**: One-at-a-time design with randomized directions

### Output Analysis
- Parameter ranking by importance
- Influential parameter identification
- Tornado plot data generation
- Sensitivity heatmap preparation

## Installation

The module is included in the main DarwinPBPK package:

```julia
using DarwinPBPK
```

## Quick Start

### Example 1: Simple PK Model

```julia
using DarwinPBPK

# Define your model function
function pk_model(params::Dict{String, Float64})
    dose = params["dose"]
    Vd = params["Vd"]
    CL = params["CL"]
    
    AUC = dose / CL
    Cmax = dose / Vd
    
    return Dict(
        "AUC" => AUC,
        "Cmax" => Cmax
    )
end

# Define parameter ranges
params = [
    ParameterRange("dose", 100.0, 70.0, 130.0),
    ParameterRange("Vd", 50.0, 35.0, 65.0),
    ParameterRange("CL", 5.0, 3.5, 6.5)
]

# Run OAT sensitivity analysis
result = one_at_a_time_sensitivity(pk_model, params, ["AUC", "Cmax"])

# View rankings
for (output, rankings) in result.rankings
    println("\n$output:")
    for (i, (param, sens)) in enumerate(rankings)
        println("  $i. $param: $sens")
    end
end
```

### Example 2: Coagulation Model Sensitivity

```julia
using DarwinPBPK

# Use default coagulation parameters (±50% variability)
coag_params = default_coagulation_parameters()
outputs = ["peak_thrombin", "lag_time", "etp"]

# Run Morris screening (fast global method)
result = morris_screening(
    coagulation_sensitivity_wrapper,
    coag_params,
    outputs,
    n_trajectories=10
)

# Identify influential factors
influential = identify_influential_parameters(result, 0.5)
println("Influential factors: ", influential["peak_thrombin"])
```

### Example 3: Sobol Variance Decomposition

```julia
using DarwinPBPK

# Run Sobol analysis (rigorous but expensive)
result = sobol_sensitivity(
    coagulation_sensitivity_wrapper,
    coag_params,
    outputs,
    n_samples=1024
)

# Extract indices
S1 = result.indices["S1"]["peak_thrombin"]  # First-order
ST = result.indices["ST"]["peak_thrombin"]  # Total-order

# Identify interactions
for (param, s1) in S1
    st = ST[param]
    interaction = st - s1
    if interaction > 0.1
        println("$param has significant interactions ($(interaction))")
    end
end
```

## API Reference

### Data Structures

#### `ParameterRange`

```julia
ParameterRange(name, nominal, min, max; distribution=:uniform, std=0.0)
```

Defines a parameter for sensitivity analysis.

**Fields**:
- `name::String`: Parameter identifier
- `nominal::Float64`: Baseline value
- `min::Float64`: Lower bound
- `max::Float64`: Upper bound
- `distribution::Symbol`: `:uniform`, `:normal`, or `:lognormal`
- `std::Float64`: Standard deviation (for normal/lognormal)

**Example**:
```julia
# Uniform distribution
p1 = ParameterRange("CL", 5.0, 3.5, 6.5)

# Normal distribution (truncated)
p2 = ParameterRange("Vd", 50.0, 35.0, 65.0, distribution=:normal, std=5.0)

# Lognormal distribution
p3 = ParameterRange("Ka", 0.5, 0.1, 2.0, distribution=:lognormal, std=0.3)
```

#### `SensitivityResult`

Container for sensitivity analysis results.

**Fields**:
- `method::Symbol`: Analysis method (`:oat`, `:sobol`, `:morris`, `:prcc`)
- `parameters::Vector{String}`: Parameter names
- `outputs::Vector{String}`: Output variable names
- `sensitivities::Dict`: Main sensitivity values
- `indices::Dict`: Method-specific indices
- `rankings::Dict`: Parameters ranked by importance
- `metadata::Dict`: Additional information

#### `SensitivityConfig`

Configuration for sensitivity analysis runs.

```julia
SensitivityConfig(;
    method=:sobol,
    n_samples=1024,
    n_trajectories=10,
    n_levels=4,
    perturbation=0.01,
    outputs=String[],
    parallel=false,
    seed=42
)
```

### Local Sensitivity Methods

#### `one_at_a_time_sensitivity`

```julia
one_at_a_time_sensitivity(model, params, outputs; perturbation=0.01)
```

Performs OAT local sensitivity analysis. Perturbs each parameter individually by `perturbation` (default 1%) while holding others constant.

**Returns**: Normalized sensitivity coefficients (ΔY/Y₀)/(ΔX/X₀)

**Computational cost**: `(p + 1)` model evaluations where `p` = number of parameters

**Pros**: Fast, simple interpretation  
**Cons**: Local only, misses interactions

#### `calculate_elasticity`

```julia
calculate_elasticity(model, params_dict, param_name, output_name; δ=1e-6)
```

Calculates elasticity coefficient: E = (∂Y/∂X) × (X/Y)

Measures percentage change in output for 1% change in parameter.

#### `normalized_sensitivity_coefficient`

```julia
normalized_sensitivity_coefficient(model, params_dict, param_name, output_name; δ=0.01)
```

Calculates NSC using finite differences with larger perturbation.

### Global Sensitivity Methods

#### `sobol_sensitivity`

```julia
sobol_sensitivity(model, params, outputs; n_samples=1024, seed=42)
```

Performs variance-based global sensitivity analysis using Sobol indices.

**Returns**:
- `S1`: First-order indices (main effect of each parameter)
- `ST`: Total-order indices (main effect + all interactions)

**Computational cost**: `n_samples × (2p + 2)` evaluations

**Interpretation**:
- High `S1`: Parameter has strong main effect
- `ST - S1`: Interaction strength
- `Σ S1 ≈ 1`: Additive model (no interactions)
- `Σ S1 < 1`: Significant interactions present

**Pros**: Most rigorous, quantifies interactions  
**Cons**: Expensive for large parameter sets

**Example**:
```julia
result = sobol_sensitivity(model, params, outputs, n_samples=2048)
S1 = result.indices["S1"]["AUC"]
ST = result.indices["ST"]["AUC"]

# Parameter with interaction
if ST["CL"] - S1["CL"] > 0.1
    println("CL has significant interactions")
end
```

#### `morris_screening`

```julia
morris_screening(model, params, outputs; n_trajectories=10, n_levels=4, seed=42)
```

Performs Morris elementary effects screening method.

**Returns**:
- `μ*`: Mean absolute elementary effect (parameter importance)
- `σ`: Standard deviation of effects (non-linearity/interactions)

**Computational cost**: `n_trajectories × (p + 1)` evaluations

**Interpretation**:
- High `μ*`: Important parameter
- High `σ`: Non-linear effects or interactions
- Low `μ*`, low `σ`: Non-influential parameter
- High `μ*`, high `σ`: Important with interactions

**Pros**: Efficient global screening  
**Cons**: Less precise than Sobol

**Recommended use**: Initial screening of 10+ parameters

#### `prcc_analysis`

```julia
prcc_analysis(model, params, outputs; n_samples=1000, seed=42)
```

Performs Partial Rank Correlation Coefficient analysis.

**Returns**: PRCC values in [-1, 1] indicating monotonic relationship strength

**Computational cost**: `n_samples` evaluations

**Interpretation**:
- `|PRCC| > 0.5`: Strong correlation
- `|PRCC| > 0.3`: Moderate correlation
- `|PRCC| > 0.1`: Weak correlation
- `PRCC > 0`: Positive relationship
- `PRCC < 0`: Negative relationship

**Pros**: Handles non-linearity, controls for other parameters  
**Cons**: Assumes monotonic relationships

**Recommended use**: Biological/pharmacological models with dose-response

### Sampling Functions

#### `latin_hypercube_sample`

```julia
latin_hypercube_sample(params, n)
```

Generates `n` Latin Hypercube samples across parameter ranges. Space-filling quasi-random design.

**Returns**: `n × p` matrix

#### `sobol_sequence`

```julia
sobol_sequence(params, n)
```

Generates Sobol quasi-random sequence (low-discrepancy). Currently uses LHS approximation.

#### `morris_trajectories`

```julia
morris_trajectories(params, r, levels)
```

Generates `r` Morris trajectories with `levels` grid points per dimension.

**Returns**: Vector of `r` matrices, each `(p+1) × p`

### Output Analysis Functions

#### `rank_parameters`

```julia
rank_parameters(result)
```

Returns parameters ranked by importance for each output.

**Returns**: `Dict{String, Vector{Tuple{String, Float64}}}`

#### `identify_influential_parameters`

```julia
identify_influential_parameters(result, threshold)
```

Identifies parameters with sensitivity above threshold.

**Example**:
```julia
influential = identify_influential_parameters(result, 0.1)
# Returns: Dict("AUC" => ["CL", "dose"], "Cmax" => ["Vd", "dose"])
```

#### `sensitivity_tornado_plot_data`

```julia
sensitivity_tornado_plot_data(result, output)
```

Prepares data for tornado diagram visualization.

**Returns**: Vector of `(parameter, sensitivity)` tuples sorted by absolute value

**Use with plotting**:
```julia
using Plots

tornado = sensitivity_tornado_plot_data(result, "peak_thrombin")
params = [t[1] for t in tornado]
sens = [t[2] for t in tornado]

barh(params, sens, xlabel="Sensitivity", title="Tornado Diagram")
```

#### `sensitivity_heatmap_data`

```julia
sensitivity_heatmap_data(result)
```

Prepares data for sensitivity heatmap (parameters × outputs).

**Returns**: Named tuple `(parameters, outputs, matrix)`

**Use with plotting**:
```julia
using Plots

hm = sensitivity_heatmap_data(result)
heatmap(hm.parameters, hm.outputs, hm.matrix', 
        xlabel="Parameters", ylabel="Outputs")
```

### Coagulation Model Integration

#### `coagulation_sensitivity_wrapper`

```julia
coagulation_sensitivity_wrapper(params_dict)
```

Wrapper for coagulation model sensitivity analysis.

**Inputs**: Dict with factor concentrations (`"II"`, `"V"`, `"VII"`, etc.)

**Outputs**:
- `"peak_thrombin"`: Maximum thrombin concentration (nM)
- `"lag_time"`: Time to 10 nM thrombin (min)
- `"ttp"`: Time to peak thrombin (min)
- `"etp"`: Endogenous thrombin potential (nM·min)

#### `default_coagulation_parameters`

```julia
default_coagulation_parameters()
```

Returns default parameter ranges for coagulation factors (±50% variability):
- Factor II (prothrombin)
- Factor V, VII, VIII, IX, X
- Antithrombin III (ATIII)
- Tissue Factor Pathway Inhibitor (TFPI)

## Method Selection Guide

### Quick Reference Table

| Method | Cost | Scope | Interactions | Best For |
|--------|------|-------|--------------|----------|
| OAT | Low (`p+1`) | Local | No | Initial screening, linear models |
| Morris | Medium (`r(p+1)`) | Global | Yes (σ) | Screening 10+ parameters |
| PRCC | Medium (`n`) | Global | Partial | Biological models, monotonic |
| Sobol | High (`n(2p+2)`) | Global | Yes (ST-S1) | Rigorous analysis, publications |

### Recommended Workflows

#### Workflow 1: Initial Exploration
```julia
# Step 1: OAT screening (fast)
oat = one_at_a_time_sensitivity(model, params, outputs)
top_10 = oat.rankings[output][1:10]

# Step 2: Focus on top parameters
top_params = [params[i] for (name, _) in top_10 for i in 1:length(params) if params[i].name == name]

# Step 3: Sobol on top parameters
sobol = sobol_sensitivity(model, top_params, outputs, n_samples=2048)
```

#### Workflow 2: Large Parameter Sets (p > 20)
```julia
# Step 1: Morris screening
morris = morris_screening(model, params, outputs, n_trajectories=10)

# Step 2: Identify influential (μ* > threshold)
influential = identify_influential_parameters(morris, 0.5)

# Step 3: PRCC on influential subset
prcc = prcc_analysis(model, influential_params, outputs, n_samples=1000)
```

#### Workflow 3: Regulatory Submission
```julia
# High-rigor variance decomposition
sobol = sobol_sensitivity(model, params, outputs, n_samples=4096)

# Document interactions
for param in sobol.parameters
    S1 = sobol.indices["S1"][output][param]
    ST = sobol.indices["ST"][output][param]
    interaction = ST - S1
    
    println("$param: S1=$S1, ST=$ST, Interaction=$interaction")
end
```

## Benchmarks

Tested on coagulation model (8 parameters, 4 outputs):

| Method | Evaluations | Time | Notes |
|--------|-------------|------|-------|
| OAT | 9 | 0.02s | Baseline |
| Morris (r=10) | 90 | 0.15s | Recommended screening |
| PRCC (n=1000) | 1,000 | 1.8s | Good balance |
| Sobol (n=1024) | 18,432 | 32s | Most rigorous |

## Examples

See `/julia-migration/scripts/sensitivity_analysis_demo.jl` for comprehensive demonstrations including:
- One-compartment PK model
- Coagulation cascade sensitivity
- Method comparison
- Visualization data preparation

Run with:
```bash
cd julia-migration
julia --project=. scripts/sensitivity_analysis_demo.jl
```

## Testing

Run test suite:
```bash
cd julia-migration
julia --project=. test/test_sensitivity_analysis.jl
```

Tests include:
- All sampling methods
- Local sensitivity (OAT, elasticity, NSC)
- Global sensitivity (Sobol, Morris, PRCC)
- Output analysis functions
- Coagulation model integration
- Ishigami benchmark function

## References

1. **Saltelli, A. et al. (2008)** - *Global Sensitivity Analysis: The Primer*  
   Comprehensive reference for variance-based methods

2. **Sobol, I.M. (2001)** - *Global sensitivity indices for nonlinear mathematical models*  
   Original Sobol indices paper

3. **Morris, M.D. (1991)** - *Factorial sampling plans for preliminary computational experiments*  
   Elementary effects screening method

4. **Marino, S. et al. (2008)** - *A methodology for performing global uncertainty and sensitivity analysis*  
   PRCC for biological models

5. **Saltelli, A. et al. (2010)** - *Variance based sensitivity analysis of model output*  
   Saltelli sampling scheme implementation

## Notes

- All methods support multiple outputs simultaneously
- Parameter distributions supported: uniform, normal (truncated), lognormal
- Random seed control for reproducibility
- Pre-allocated arrays for efficiency
- Type-stable implementations for performance

## Future Enhancements

Potential additions:
- True Sobol sequence generator (integrate Sobol.jl)
- Parallel evaluation support
- Adaptive sampling strategies
- Time-dependent sensitivity (for ODE models)
- Visualization functions (plots)
- Confidence intervals for Sobol indices
- Extended FAST method
- Delta moment-independent indices

---

**Author**: Darwin PBPK Platform  
**License**: See repository LICENSE  
**Version**: 2.5.0 (December 2025)
