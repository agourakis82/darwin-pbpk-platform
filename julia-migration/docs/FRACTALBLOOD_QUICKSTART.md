# FractalBlood Integration - Quick Start Guide

**5-Minute Guide to Using FractalBlood with PBPK**

---

## Installation

```julia
cd julia-migration
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

---

## Basic Usage (Copy-Paste Ready)

### Method 1: From Fractal Network

```julia
using DarwinPBPK.ODEPBPKSolver
using DarwinPBPK.ODEPBPKSolver: FractalBlood

# Create fractal vascular network
fractal_model = FractalBlood.create_fractal_blood_model(
    num_levels = 15,
    alpha = 1.37,
    beta = 0.8
)

# Standard PBPK parameters
pbpk = PBPKParams(
    clearance_hepatic = 10.0,
    clearance_renal = 2.0
)

# Integrate!
combined = integrate_fractal_blood!(pbpk, fractal_model)

# Access parameters
println("Alpha: ", combined.fractal.alpha)
println("Mean transit time: ", combined.fractal.tau_mean, " seconds")
```

### Method 2: Direct Parameters

```julia
using DarwinPBPK.ODEPBPKSolver

# Create fractal-enhanced PBPK directly
params = create_fractal_pbpk_params(
    alpha = 1.37,           # Power-law exponent
    tau_min = 0.1,          # Min transit time (seconds)
    tau_mean = 20.0,        # Mean transit time (seconds)
    beta = 0.8,             # Anomalous diffusion
    clearance_hepatic = 10.0,
    clearance_renal = 2.0,
    use_convolution = true  # Enable full convolution
)
```

---

## Quick Examples

### Example 1: Transit Time Distribution

```julia
fractal_params = FractalBloodParams(
    enabled = true,
    alpha = 1.37,
    tau_min = 0.1
)

# Evaluate at t = 1.0 second
E_t = fractal_transit_time_distribution(1.0, fractal_params)
println("E(1.0s) = $E_t")
```

### Example 2: Network Validation

```julia
# Create network
vessels = FractalBlood.create_fractal_network(15)

# Validate topology
validation = FractalBlood.validate_network_topology(vessels)

println("Murray's Law compliance: ", 
        validation["murray_law_compliance"] * 100, "%")
```

### Example 3: Run Demo

```julia
include("examples/fractalblood_integration_demo.jl")
```

---

## Key Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `alpha` | 1.37 | Power-law exponent (from fractal D≈2.7) |
| `tau_min` | 0.1 s | Minimum transit time (aorta only) |
| `tau_mean` | 20 s | Mean circulation time |
| `beta` | 0.8 | Anomalous diffusion exponent |
| `use_convolution` | false/true | Full convolution (slower, accurate) |

---

## API Cheat Sheet

```julia
# Structs
FractalBloodParams(enabled, alpha, tau_min, tau_mean, beta, ...)
PBPKParamsWithFractal(pbpk, fractal)

# Integration
integrate_fractal_blood!(pbpk_params, fractal_model)
create_fractal_pbpk_params(; alpha=1.37, tau_min=0.1, ...)

# Functions
fractal_transit_time_distribution(t, fractal_params)
apply_fractal_dispersion(C_input, t, history, fractal_params)
```

---

## Testing

```bash
# Run tests
julia --project=. test/test_fractalblood_integration.jl

# Run demo
julia --project=. examples/fractalblood_integration_demo.jl
```

---

## Full Documentation

- **Complete API**: `docs/FRACTALBLOOD_INTEGRATION.md`
- **Summary**: `docs/FRACTALBLOOD_INTEGRATION_SUMMARY.md`
- **This guide**: `docs/FRACTALBLOOD_QUICKSTART.md`

---

## Common Issues

### Error: "FractalBlood alpha must be > 1.0"
**Fix**: Set `alpha > 1.0` (typical: 1.37)

### Error: Module not found
**Fix**: Run from `julia-migration/` directory with `--project=.`

### Slow performance
**Fix**: Set `use_convolution = false` for faster approximation

---

**Need help?** See full documentation in `docs/FRACTALBLOOD_INTEGRATION.md`
