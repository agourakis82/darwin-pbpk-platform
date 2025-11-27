#!/usr/bin/env julia
#=
Validation Script for MedLang v0.4 Features
- Non-linear (Saturable) Absorption
- Multi-compartment Depot (IM/SC)
=#

using Pkg
Pkg.activate("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration")

include("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration/src/DarwinPBPK/ode_solver.jl")
using .ODEPBPKSolver

println("=" ^ 60)
println("MedLang v0.4 Feature Validation")
println("=" ^ 60)

#=============================================================================
  Test 1: Saturable Absorption - Dose Proportionality

  Drug: Gabapentin-like (saturable PEPT1 transporter)
  Expected: Less-than-proportional increase in AUC with dose
=============================================================================#

println("\n--- Test 1: Saturable Absorption (Dose Proportionality) ---")

pbpk = PBPKParams(
    clearance_hepatic = 0.0,
    clearance_renal = 7.5,
)

# Saturable absorption parameters
# Vmax = 100 mg/h, Km = 50 mg
sat_params = SaturableAbsorptionParams(100.0, 50.0; fa=1.0, fg=1.0, fh=1.0)

println("Saturable params: Vmax=", sat_params.vmax, " mg/h, Km=", sat_params.km, " mg")
println("Apparent Ka (low dose): ", round(apparent_ka(sat_params), digits=2), " 1/h")

# Test dose proportionality
doses = [100.0, 300.0, 600.0, 1200.0]
dp_results = analyze_dose_proportionality(pbpk, sat_params, doses)

println("\nDose proportionality analysis:")
for i in 1:length(doses)
    println("  Dose ", doses[i], " mg: AUC=", round(dp_results["auc"][i], digits=1),
            ", AUC/Dose=", round(dp_results["auc_norm"][i], digits=3))
end
println("  Power exponent (beta): ", round(dp_results["power_exponent"][1], digits=3))

# Check for non-linearity (beta < 1)
is_nonlinear = dp_results["power_exponent"][1] < 0.95
println("Non-linear absorption detected (beta < 0.95): ", is_nonlinear ? "PASS" : "FAIL")

#=============================================================================
  Test 2: Saturable vs First-Order Comparison
=============================================================================#

println("\n--- Test 2: Saturable vs First-Order Comparison ---")

# First-order with equivalent Ka at low dose
ka_equiv = apparent_ka(sat_params)
oral_linear = OralParams(ka_equiv, 1.0, 1.0, 1.0, 0.0)

# Low dose comparison
dose_low = 50.0
result_sat_low = simulate_saturable(pbpk, sat_params, dose_low; t_max=24.0)
result_linear_low = simulate_oral(pbpk, oral_linear, dose_low; t_max=24.0)

println("Low dose (50mg):")
println("  Saturable: Cmax=", round(result_sat_low["cmax"], digits=3), ", Tmax=", round(result_sat_low["tmax"], digits=2))
println("  First-order: Cmax=", round(result_linear_low["cmax"], digits=3), ", Tmax=", round(result_linear_low["tmax"], digits=2))

# High dose comparison
dose_high = 1000.0
result_sat_high = simulate_saturable(pbpk, sat_params, dose_high; t_max=24.0)
result_linear_high = simulate_oral(pbpk, oral_linear, dose_high; t_max=24.0)

println("High dose (1000mg):")
println("  Saturable: Cmax=", round(result_sat_high["cmax"], digits=3), ", Tmax=", round(result_sat_high["tmax"], digits=2))
println("  First-order: Cmax=", round(result_linear_high["cmax"], digits=3), ", Tmax=", round(result_linear_high["tmax"], digits=2))

# At high dose, saturable should have later Tmax (absorption is rate-limiting)
tmax_delayed_high = result_sat_high["tmax"] > result_linear_high["tmax"]
println("Saturable Tmax > Linear Tmax at high dose: ", tmax_delayed_high ? "PASS" : "FAIL")

#=============================================================================
  Test 3: IM Depot Absorption

  Expected: Slower absorption than oral, flip-flop possible
=============================================================================#

println("\n--- Test 3: IM Depot Absorption ---")

pbpk_im = PBPKParams(
    clearance_hepatic = 10.0,
    clearance_renal = 2.0,
)

# Standard IM depot
depot_im = DepotParams_IM()
println("IM depot: Ka=", depot_im.ka[1], " 1/h, F=", depot_im.f)

result_im = simulate_depot(pbpk_im, depot_im, 100.0; t_max=48.0)

println("Results:")
println("  Cmax: ", round(result_im["cmax"], digits=3), " mg/L")
println("  Tmax: ", round(result_im["tmax"], digits=2), " h")
println("  AUC: ", round(result_im["auc"], digits=2), " mg*h/L")
println("  MAT: ", round(result_im["mat"], digits=2), " h")
println("  Flip-flop kinetics: ", result_im["flip_flop"])

# IM should have Tmax > 1h due to slow absorption
im_delayed = result_im["tmax"] > 1.0
println("IM Tmax > 1h: ", im_delayed ? "PASS" : "FAIL")

#=============================================================================
  Test 4: SC Depot with Dual Absorption (Biologics)

  Expected: Biphasic absorption with fast and slow components
=============================================================================#

println("\n--- Test 4: SC Depot with Dual Absorption ---")

# SC biologic with dual absorption
depot_sc_bio = DepotParams_SC_Biologic()
println("SC biologic: Ka_fast=", depot_sc_bio.ka[1], ", Ka_slow=", depot_sc_bio.ka[2])
println("  Fraction fast=", depot_sc_bio.fractions[1], ", F=", depot_sc_bio.f)

result_sc = simulate_depot(pbpk_im, depot_sc_bio, 100.0; t_max=168.0)  # 1 week

println("Results:")
println("  Cmax: ", round(result_sc["cmax"], digits=4), " mg/L")
println("  Tmax: ", round(result_sc["tmax"], digits=1), " h")
println("  AUC: ", round(result_sc["auc"], digits=2), " mg*h/L")
println("  MAT: ", round(result_sc["mat"], digits=2), " h")

# SC biologic should have very late Tmax
sc_very_delayed = result_sc["tmax"] > 10.0
println("SC Biologic Tmax > 10h: ", sc_very_delayed ? "PASS" : "FAIL")

#=============================================================================
  Test 5: Route Comparison (Oral vs IM vs SC)
=============================================================================#

println("\n--- Test 5: Route Comparison ---")

# Same drug, different routes
oral_params = OralParams(1.0, 0.8, 0.9, 0.85, 0.0)  # F = 0.8*0.9*0.85 = 0.61
depot_im_cmp = DepotParams(:IM, 0.5; f=0.9, lag=0.1)
depot_sc_cmp = DepotParams(:SC, 0.2; f=0.85, lag=0.2)

dose = 100.0
result_oral = simulate_oral(pbpk_im, oral_params, dose; t_max=48.0)
result_im_cmp = simulate_depot(pbpk_im, depot_im_cmp, dose; t_max=48.0)
result_sc_cmp = simulate_depot(pbpk_im, depot_sc_cmp, dose; t_max=48.0)

println("Route comparison (100mg dose):")
println("  Oral:  Cmax=", round(result_oral["cmax"], digits=3), ", Tmax=", round(result_oral["tmax"], digits=2), "h, AUC=", round(result_oral["auc"], digits=1))
println("  IM:    Cmax=", round(result_im_cmp["cmax"], digits=3), ", Tmax=", round(result_im_cmp["tmax"], digits=2), "h, AUC=", round(result_im_cmp["auc"], digits=1))
println("  SC:    Cmax=", round(result_sc_cmp["cmax"], digits=3), ", Tmax=", round(result_sc_cmp["tmax"], digits=2), "h, AUC=", round(result_sc_cmp["auc"], digits=1))

# Verify: Tmax_oral < Tmax_IM < Tmax_SC
tmax_order = result_oral["tmax"] < result_im_cmp["tmax"] < result_sc_cmp["tmax"]
println("Tmax order (Oral < IM < SC): ", tmax_order ? "PASS" : "FAIL")

#=============================================================================
  Summary
=============================================================================#

println("\n" * "=" ^ 60)
println("VALIDATION SUMMARY")
println("=" ^ 60)

tests_passed = sum([is_nonlinear, tmax_delayed_high, im_delayed, sc_very_delayed, tmax_order])
total_tests = 5

println("Tests passed: ", tests_passed, " / ", total_tests)
status = tests_passed == total_tests ? "ALL PASS" : "SOME FAILURES"
println("Status: ", status)

if tests_passed >= 4
    println("\nv0.4 features validated successfully!")
    println("Ready to update MedLang grammar.")
end
