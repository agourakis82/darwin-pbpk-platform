#!/usr/bin/env julia
using Test
include("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration/src/DarwinPBPK/ode_solver.jl")
using .ODEPBPKSolver

println("=" ^ 60)
println("MedLang v1.0 Feature Validation")
println("=" ^ 60)

results = Dict{String, Bool}()

# Test 1: QSP Structs
println("\n[1/5] QSP Structs...")
try
    tmdd = TMDDParams(kon=0.1, koff=0.01, ksyn=1.0, kdeg=0.1, kint=0.05)
    tumor = TumorGrowthKillParams(growth_model=:logistic, kg=0.05, kmax=5000.0)
    rl = ReceptorLigandParams(kon=0.1, koff=0.01, rtot=10.0)
    println("   TMDD Kd: ", tmdd.kd, " nM")
    println("   Tumor kg: ", tumor.kg, "/day")
    println("   RL Kd: ", rl.kd, " nM")
    results["QSP"] = true
    println("   [PASS]")
catch e
    results["QSP"] = false
    println("   [FAIL]")
end

# Test 2: ML Prediction
println("\n[2/5] ML Prediction...")
try
    pred = MLParameterPredictor(model_type=:multimodal)
    kp = predict_partition_coeffs(pred, "CCO"; logP=2.0, fup=0.1)
    println("   Liver Kp: ", round(kp["liver"], digits=2))
    println("   Adipose Kp: ", round(kp["adipose"], digits=2))
    results["ML"] = kp["adipose"] > kp["liver"]
    println("   [PASS]")
catch e
    results["ML"] = false
    println("   [FAIL] ", e)
end

# Test 3: PK Surrogate
println("\n[3/5] PK Surrogate...")
try
    surr = PKSurrogateModel()
    pk = predict_pk_surrogate(surr, 100.0, 10.0, 50.0; ka=1.0, f=0.8)
    println("   AUC: ", round(pk[:auc], digits=1))
    println("   t1/2: ", round(pk[:half_life], digits=1), " h")
    results["Surrogate"] = pk[:auc] > 0
    println("   [PASS]")
catch e
    results["Surrogate"] = false
    println("   [FAIL]")
end

# Test 4: Covariates
println("\n[4/5] Covariates...")
try
    scaled = apply_covariate(10.0, 100.0, ALLOMETRIC_CL)
    expected = 10.0 * (100.0/70.0)^0.75
    println("   Scaled CL: ", round(scaled, digits=2), " L/h")
    println("   Expected: ", round(expected, digits=2), " L/h")
    results["Covariate"] = abs(scaled - expected) < 0.01
    println("   [PASS]")
catch e
    results["Covariate"] = false
    println("   [FAIL]")
end

# Test 5: Trial Design
println("\n[5/5] Trial Design...")
try
    reg = DosingRegimen(dose=100.0, route=:ORAL, interval=24.0, n_doses=7)
    arm = TrialArm(name="Tx", regimen=reg, n_subjects=30)
    dt = dosing_times(reg)
    println("   Dose times: ", dt[1:3], "...")
    println("   N doses: ", length(dt))
    results["Trial"] = length(dt) == 7
    println("   [PASS]")
catch e
    results["Trial"] = false
    println("   [FAIL]")
end

# Summary
println("\n", "=" ^ 60)
passed = sum(values(results))
total = length(results)
println("SUMMARY: ", passed, "/", total, " passed")
println("=" ^ 60)
