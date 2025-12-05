# Sensitivity Analysis Module - Quick Reference

**File**: `sensitivity_analysis.jl`  
**Module**: `DarwinPBPK.SensitivityAnalysis`

## What's Included

### Parameter Structures
- `ParameterRange` - Define parameter name, nominal, min, max, distribution
- `SensitivityResult` - Results container with sensitivities, indices, rankings
- `SensitivityConfig` - Configuration for analysis runs

### Local Sensitivity Analysis
- `one_at_a_time_sensitivity()` - OAT method (fast, local)
- `calculate_elasticity()` - Elasticity coefficient E = (∂Y/∂X)×(X/Y)
- `normalized_sensitivity_coefficient()` - NSC = (ΔY/Y)/(ΔX/X)

### Global Sensitivity Analysis
- `sobol_sensitivity()` - Variance-based (S₁, Sᴛ indices)
- `morris_screening()` - Elementary effects (μ*, σ)
- `prcc_analysis()` - Partial rank correlation coefficient

### Sampling Functions
- `latin_hypercube_sample()` - LHS sampling
- `sobol_sequence()` - Sobol quasi-random sequence
- `morris_trajectories()` - Morris OAT design

### Output Analysis
- `rank_parameters()` - Rank by importance
- `identify_influential_parameters()` - Filter by threshold
- `sensitivity_tornado_plot_data()` - Data for tornado charts
- `sensitivity_heatmap_data()` - Data for heatmaps

### Coagulation Integration
- `coagulation_sensitivity_wrapper()` - Wrapper for coag model
- `default_coagulation_parameters()` - Standard factor ranges

## Quick Start

```julia
using DarwinPBPK

# Define model
function my_model(params::Dict{String, Float64})
    return Dict("output1" => params["x1"] * 2.0 + params["x2"])
end

# Define parameters
params = [
    ParameterRange("x1", 1.0, 0.5, 1.5),
    ParameterRange("x2", 2.0, 1.0, 3.0)
]

# Run sensitivity analysis
result = sobol_sensitivity(my_model, params, ["output1"], n_samples=1024)

# View results
println(result.rankings["output1"])
```

## Method Selection

| Method | When to Use | Cost |
|--------|-------------|------|
| **OAT** | Initial screening, linear models | Low (p+1) |
| **Morris** | Screen 10+ parameters globally | Medium (r×(p+1)) |
| **PRCC** | Biological models, monotonic | Medium (n) |
| **Sobol** | Rigorous analysis, publications | High (n×(2p+2)) |

## Typical Workflow

1. **OAT** for quick initial ranking
2. **Morris** for global screening of top 10-15 parameters  
3. **Sobol** for rigorous variance decomposition of key parameters
4. **PRCC** for monotonic relationship quantification

## Coagulation Example

```julia
# Get default parameters (8 coagulation factors)
coag_params = default_coagulation_parameters()

# Run Morris screening
result = morris_screening(
    coagulation_sensitivity_wrapper,
    coag_params,
    ["peak_thrombin", "lag_time", "etp"],
    n_trajectories=10
)

# View most influential factors
println(result.rankings["peak_thrombin"][1:3])
```

## Full Documentation

See `/julia-migration/docs/SENSITIVITY_ANALYSIS.md` for:
- Complete API reference
- Detailed examples
- Interpretation guidelines
- Benchmark results
- Literature references

## Demo Script

Run comprehensive demo:
```bash
julia --project=. scripts/sensitivity_analysis_demo.jl
```

## Tests

Run test suite:
```bash
julia --project=. test/test_sensitivity_analysis.jl
```

---

**Status**: ✓ Fully implemented and tested  
**Integration**: ✓ Exported from main DarwinPBPK module  
**Documentation**: ✓ Complete  
**Date**: 2025-12-05
