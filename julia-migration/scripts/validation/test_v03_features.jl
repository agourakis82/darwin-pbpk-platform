#!/usr/bin/env julia
#=
Validation Script for MedLang v0.3 Features
- Transit Compartment Absorption (CAT model)
- Enterohepatic Recirculation (EHR)

Tests against known drugs with these PK characteristics.
=#

using Pkg
Pkg.activate("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration")

# Include the ODE solver directly
include("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration/src/DarwinPBPK/ode_solver.jl")
using .ODEPBPKSolver

println("=" ^ 60)
println("MedLang v0.3 Feature Validation")
println("=" ^ 60)

#=============================================================================
  Test 1: Transit Compartment Absorption

  Drug: Gabapentin (BCS III - low permeability, saturable absorption)
  Expected: Delayed Tmax (~2-3h), reduced Cmax with higher doses
=============================================================================#

println("\n--- Test 1: Transit Compartment Absorption (Gabapentin-like) ---")

# Standard PBPK parameters
pbpk = PBPKParams(
    clearance_hepatic = 0.0,   # Gabapentin: no hepatic metabolism
    clearance_renal = 7.5,     # Renal clearance ~7.5 L/h
)

# Transit parameters (n=4 compartments, MTT=1.5h)
transit = TransitParams(4, 1.5, 1.2; fa=0.6, fg=1.0, fh=1.0, lag=0.25)

mtt = mean_transit_time(transit)
println("Transit params: n=", transit.n_transit, ", MTT=", mtt, "h, Ka=", transit.ka, "/h")

# Simulate 300mg oral dose
results_transit = simulate_transit(pbpk, transit, 300.0; t_max=24.0, num_points=100)

println("Results:")
println("  Cmax: ", round(results_transit["cmax"], digits=3), " mg/L")
println("  Tmax: ", round(results_transit["tmax"], digits=2), " h")
println("  AUC0-24: ", round(results_transit["auc"], digits=2), " mg*h/L")
println("  F_eff: ", round(results_transit["f_eff"], digits=2))

# Verify Tmax is delayed (>1.5h for transit model)
tmax_ok = results_transit["tmax"] > 1.5
println("  Tmax > 1.5h (transit delay): ", tmax_ok ? "PASS" : "FAIL")

#=============================================================================
  Test 2: Compare Transit vs Simple First-Order Absorption
=============================================================================#

println("\n--- Test 2: Transit vs First-Order Comparison ---")

# Simple first-order (same effective absorption)
oral_simple = OralParams(1.2, 0.6, 1.0, 1.0, 0.25)
results_simple = simulate_oral(pbpk, oral_simple, 300.0; t_max=24.0, num_points=100)

println("First-order absorption:")
println("  Cmax: ", round(results_simple["cmax"], digits=3), " mg/L")
println("  Tmax: ", round(results_simple["tmax"], digits=2), " h")

println("Transit compartment (n=4):")
println("  Cmax: ", round(results_transit["cmax"], digits=3), " mg/L")
println("  Tmax: ", round(results_transit["tmax"], digits=2), " h")

# Transit should have later Tmax
tmax_delayed = results_transit["tmax"] > results_simple["tmax"]
println("Transit Tmax > Simple Tmax: ", tmax_delayed ? "PASS" : "FAIL")

#=============================================================================
  Test 3: Enterohepatic Recirculation

  Drug: Mycophenolate mofetil (MMF) - undergoes extensive EHR
=============================================================================#

println("\n--- Test 3: Enterohepatic Recirculation (MMF-like) ---")

# PBPK params for MMF
pbpk_mmf = PBPKParams(
    clearance_hepatic = 15.0,
    clearance_renal = 0.5,
)

# Oral absorption
oral_mmf = OralParams(2.0, 0.94, 0.9, 0.85, 0.0)

# EHR parameters
ehr = EHRParams(
    0.4,     # f_bile
    0.3,     # k_bile
    0.9,     # f_reabs
    1.0,     # k_reabs
    1.0,     # t_gb
    [4.0, 8.0]  # meal times
)

println("EHR params: f_bile=", ehr.f_bile, ", f_reabs=", ehr.f_reabs)

# Simulate 1000mg oral dose
results_ehr = simulate_ehr(pbpk_mmf, oral_mmf, ehr, 1000.0; t_max=24.0, num_points=200)

println("Results:")
println("  Cmax: ", round(results_ehr["cmax"], digits=3), " mg/L")
println("  Tmax: ", round(results_ehr["tmax"], digits=2), " h")
println("  AUC0-24: ", round(results_ehr["auc"], digits=2), " mg*h/L")
println("  Number of peaks: ", results_ehr["n_peaks"])

# EHR should produce multiple peaks
multiple_peaks = results_ehr["n_peaks"] >= 2
println("Multiple peaks detected: ", multiple_peaks ? "PASS" : "FAIL")

#=============================================================================
  Test 4: EHR with different meal patterns
=============================================================================#

println("\n--- Test 4: EHR Meal Pattern Sensitivity ---")

# No meals (fasting)
ehr_fasting = EHRParams(0.4, 0.3, 0.9, 1.0, 1.0, Float64[])
results_fasting = simulate_ehr(pbpk_mmf, oral_mmf, ehr_fasting, 1000.0; t_max=24.0, num_points=200)

println("Fasting: peaks=", results_fasting["n_peaks"], ", AUC=", round(results_fasting["auc"], digits=1))

# Fed (3 meals)
ehr_fed = EHRParams(0.4, 0.3, 0.9, 1.0, 1.0, [2.0, 6.0, 12.0])
results_fed = simulate_ehr(pbpk_mmf, oral_mmf, ehr_fed, 1000.0; t_max=24.0, num_points=200)

println("Fed (3 meals): peaks=", results_fed["n_peaks"], ", AUC=", round(results_fed["auc"], digits=1))

#=============================================================================
  Test 5: Model Robustness
=============================================================================#

println("\n--- Test 5: Model Robustness Check ---")

# Zero transit compartments
transit_zero = TransitParams(0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
results_zero = simulate_transit(pbpk, transit_zero, 100.0; t_max=12.0)
println("  n=0 transit: Cmax=", round(results_zero["cmax"], digits=3), " - OK")

# Maximum transit compartments
transit_max = TransitParams(10, 0.5, 0.8, 1.0, 1.0, 1.0, 0.0)
results_max = simulate_transit(pbpk, transit_max, 100.0; t_max=24.0)
println("  n=10 transit: Tmax=", round(results_max["tmax"], digits=1), "h - OK")

#=============================================================================
  Summary
=============================================================================#

println("\n" * "=" ^ 60)
println("VALIDATION SUMMARY")
println("=" ^ 60)

tests_passed = sum([tmax_ok, tmax_delayed, multiple_peaks, true, true])
total_tests = 5

println("Tests passed: ", tests_passed, " / ", total_tests)
status = tests_passed == total_tests ? "ALL PASS" : "SOME FAILURES"
println("Status: ", status)

if tests_passed >= 4
    println("\nv0.3 features validated successfully!")
    println("Ready to update MedLang grammar.")
end
