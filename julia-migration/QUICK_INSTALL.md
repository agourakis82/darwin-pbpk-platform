# 🚀 Darwin PBPK Platform - Quick Install Guide

**Production-Ready Julia Implementation**

---

## ⚡ **One-Command Install**

```bash
# Clone and setup
git clone https://github.com/agourakis82/darwin-pbpk-platform.git
cd darwin-pbpk-platform/julia-migration
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

---

## 🧪 **Quick Test**

```julia
# Start Julia in the project
julia --project=.

# Test the platform
using DarwinPBPK

# Create PBPK parameters
p = DarwinPBPK.ODEPBPKSolver.PBPKParams(
    clearance_hepatic=10.0,
    clearance_renal=5.0,
    partition_coeffs=Dict("liver" => 2.0, "kidney" => 1.5)
)

# Run simulation (100mg dose, 24 hours)
sol = DarwinPBPK.ODEPBPKSolver.solve(p, 100.0, (0.0, 24.0))

println("✅ Darwin PBPK Platform working!")
println("   Simulated $(length(sol.t)) time points")
println("   Final blood concentration: $(sol[end][1]) mg/L")
```

---

## 📊 **Benchmark Performance**

```julia
using BenchmarkTools

# Benchmark single simulation
@btime DarwinPBPK.ODEPBPKSolver.solve($p, 100.0, (0.0, 24.0))

# Expected: ~7ms (2.5× faster than Python)
```

---

## 🔬 **Validate Regulatory Metrics**

```julia
# Test validation functions
pred = [1.0, 2.0, 3.0, 4.0, 5.0]
obs = [1.1, 2.1, 2.9, 4.2, 4.8]

gmfe = DarwinPBPK.Validation.geometric_mean_fold_error(pred, obs)
r2 = DarwinPBPK.Validation.r_squared(pred, obs)

println("GMFE: $gmfe (target: <2.0)")
println("R²: $r2 (target: >0.90)")
```

---

## 🎯 **System Requirements**

- **Julia:** 1.9+ (tested on 1.10.0)
- **RAM:** 4GB minimum, 8GB recommended
- **CPU:** Any modern processor
- **GPU:** Optional (CUDA support available)
- **OS:** Linux, macOS, Windows

---

## 📚 **Documentation**

- `README.md` - Project overview
- `PRODUCTION_READY_SUMMARY.md` - Performance results
- `docs/` - Detailed documentation
- `examples/` - Usage examples

---

## 🆘 **Support**

- **Issues:** GitHub Issues
- **Email:** [your-email]
- **Documentation:** Full API docs available

---

**🚀 Ready to revolutionize PBPK modeling with 2.5× performance gains!**
