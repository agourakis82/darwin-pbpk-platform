# ===========================================================================
# TEST: ML-MEDLANG INTEGRATION
# ===========================================================================
# Demonstrates the full pipeline:
# 1. ML predicts transporter substrates from SMILES
# 2. TransporterPredictor → NamedTuple with transporters, Km, P-gp
# 3. MLMedLangIntegration generates MedLang DSL code
# 4. MedLang.compile_model() → PBPKParams
# 5. MedLang.simulate_oral() → concentration-time profiles
#
# This is the CORRECT architecture: ML feeds MedLang DSL
# ===========================================================================

using Test

# Add project to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using DarwinPBPK
using DarwinPBPK.MedLang
using DarwinPBPK.MedLang.MLMedLangIntegration

println("="^70)
println("ML → MEDLANG INTEGRATION TEST")
println("="^70)

# ===========================================================================
# Test 1: Parameter Estimation Functions
# ===========================================================================

@testset "Parameter Estimation" begin
    # Test Ka estimation from transporters
    transporters = [:PEPT1, :OCT1]
    km_values = Dict(:PEPT1 => 50.0, :OCT1 => 200.0)

    ka = estimate_ka_from_transporters(transporters, km_values; baseline_ka=1.0)
    @test ka > 1.0  # Uptake transporters should increase Ka
    @test ka < 5.0  # But not unreasonably high
    println("✓ Ka from PEPT1+OCT1: $(round(ka, digits=2)) /h")

    # Test Fg estimation
    fg_no_efflux = estimate_fg_from_transporters([:PEPT1])
    fg_with_pgp = estimate_fg_from_transporters([:PEPT1, :PGP])
    @test fg_no_efflux > fg_with_pgp  # P-gp should reduce Fg
    println("✓ Fg without P-gp: $(round(fg_no_efflux, digits=2))")
    println("✓ Fg with P-gp: $(round(fg_with_pgp, digits=2))")

    # Test Fh estimation
    fh_low_cl = estimate_fh_from_clearance(5.0)   # Low clearance
    fh_high_cl = estimate_fh_from_clearance(50.0) # High clearance
    @test fh_low_cl > fh_high_cl
    @test fh_low_cl > 0.9
    @test fh_high_cl < 0.5
    println("✓ Fh (CLh=5 L/h): $(round(fh_low_cl, digits=2))")
    println("✓ Fh (CLh=50 L/h): $(round(fh_high_cl, digits=2))")
end

# ===========================================================================
# Test 2: MedLang Code Generation
# ===========================================================================

@testset "MedLang Code Generation" begin
    # Simulate ML prediction result (what TransporterPredictor would return)
    ml_result = (
        uptake_transporters = [:PEPT1, :OATP2B1],
        is_pgp_substrate = true,
        pgp_efflux_ratio = 8.5,
        carrier_km_values = Dict(:PEPT1 => 80.0, :OATP2B1 => 150.0, :PGP => 50.0),
        predictions = []  # Full predictions would be here
    )

    # Generate MedLang absorption block
    absorption_block = generate_medlang_absorption(
        ml_result;
        drug_name = "Cephalexin",
        cl_hepatic = 0.5,  # Renally cleared
        cyp_substrate = :none
    )

    @test occursin("route: oral", absorption_block)
    @test occursin("absorption {", absorption_block)
    @test occursin("Ka:", absorption_block)
    @test occursin("firstpass {", absorption_block)
    @test occursin("Fg:", absorption_block)
    @test occursin("Fh:", absorption_block)

    println("\n--- Generated Absorption Block ---")
    println(absorption_block)

    # Generate full MedLang model
    full_model = generate_full_medlang_model(
        ml_result;
        drug_name = "Cephalexin",
        mw = 347.4,
        cl_hepatic = 0.5,
        cl_renal = 15.0,
        cyp_substrate = :none
    )

    @test occursin("model Cephalexin_PBPK", full_model)
    @test occursin("organ liver", full_model)
    @test occursin("clearance hepatic", full_model)
    @test occursin("clearance renal", full_model)

    println("\n--- Generated Full MedLang Model ---")
    println(full_model)
end

# ===========================================================================
# Test 3: End-to-End Simulation with MedLang DSL
# ===========================================================================

@testset "End-to-End MedLang Simulation" begin
    # ML prediction for a P-gp substrate with CYP3A4 metabolism
    ml_result_midazolam = (
        uptake_transporters = Symbol[],  # Passive diffusion
        is_pgp_substrate = true,
        pgp_efflux_ratio = 5.0,
        carrier_km_values = Dict{Symbol, Float64}(:PGP => 30.0),
        predictions = []
    )

    # Generate MedLang model
    medlang_source = generate_full_medlang_model(
        ml_result_midazolam;
        drug_name = "Midazolam",
        mw = 325.8,
        cl_hepatic = 27.0,  # High CYP3A4 clearance
        cl_renal = 0.5,
        cyp_substrate = :CYP3A4,
        fu = 0.04
    )

    println("\n--- Midazolam MedLang Model ---")
    println(medlang_source)

    # Validate model
    issues = validate_model(medlang_source)
    println("\nValidation issues: ", isempty(issues) ? "None" : join(issues, "\n"))

    # Compile to PBPKParams
    params = compile_model(medlang_source)
    @test params isa DarwinPBPK.ODEPBPKSolver.PBPKParams
    @test params.hepatic_clearance ≈ 27.0
    @test params.renal_clearance ≈ 0.5
    println("✓ Model compiled to PBPKParams")

    # Simulate oral dose
    results = simulate_oral(medlang_source, 15.0; t_max=24.0, num_points=100)

    @test haskey(results, "plasma")
    @test haskey(results, "time")
    @test haskey(results, "cmax")
    @test haskey(results, "tmax")
    @test haskey(results, "auc")

    println("\n--- Simulation Results ---")
    println("Cmax: $(round(results["cmax"], digits=3)) mg/L")
    println("Tmax: $(round(results["tmax"], digits=2)) h")
    println("AUC: $(round(results["auc"], digits=2)) mg·h/L")

    # Check that first-pass reduced bioavailability
    # Midazolam F_oral ≈ 0.30-0.50 (CYP3A4 substrate)
    # Expected: Fg ≈ 0.44, Fh ≈ 0.57 → F_eff ≈ 0.25
    @test results["cmax"] < 5.0  # Should be low due to first-pass
    @test results["tmax"] > 0.2  # Should have lag + absorption time
end

# ===========================================================================
# Test 4: Drug Examples with Known Transporter Profiles
# ===========================================================================

@testset "Known Drug Transporter Profiles" begin

    # --- Metformin (OCT substrate, no metabolism) ---
    ml_metformin = (
        uptake_transporters = [:OCT1, :OCT3],
        is_pgp_substrate = false,
        pgp_efflux_ratio = 1.0,
        carrier_km_values = Dict(:OCT1 => 1000.0, :OCT3 => 500.0),  # Low affinity
        predictions = []
    )

    model_metformin = generate_full_medlang_model(
        ml_metformin;
        drug_name = "Metformin",
        mw = 129.2,
        cl_hepatic = 0.0,  # No hepatic metabolism
        cl_renal = 30.0,   # Renally cleared
        cyp_substrate = :none
    )

    results_metformin = simulate_oral(model_metformin, 500.0; t_max=24.0)

    println("\n--- Metformin (500 mg oral) ---")
    println("Cmax: $(round(results_metformin["cmax"], digits=2)) mg/L")
    println("Tmax: $(round(results_metformin["tmax"], digits=2)) h")
    # Metformin F ≈ 50-60% (saturable absorption)
    # Fg ≈ 1.0 (no gut metabolism), Fh ≈ 1.0 (no hepatic metabolism)

    # --- Digoxin (P-gp substrate, saturable) ---
    ml_digoxin = (
        uptake_transporters = [:OATP2B1],
        is_pgp_substrate = true,
        pgp_efflux_ratio = 15.0,  # Strong P-gp
        carrier_km_values = Dict(:OATP2B1 => 200.0, :PGP => 20.0),
        predictions = []
    )

    model_digoxin = generate_full_medlang_model(
        ml_digoxin;
        drug_name = "Digoxin",
        mw = 780.9,
        cl_hepatic = 2.0,
        cl_renal = 6.0,
        cyp_substrate = :none,
        fu = 0.25
    )

    results_digoxin = simulate_oral(model_digoxin, 0.25; t_max=48.0)

    println("\n--- Digoxin (0.25 mg oral) ---")
    println("Cmax: $(round(results_digoxin["cmax"], digits=4)) mg/L")
    println("Tmax: $(round(results_digoxin["tmax"], digits=2)) h")
    # Digoxin F ≈ 70% (P-gp saturation at therapeutic doses)

    # --- Atorvastatin (OATP substrate, CYP3A4) ---
    ml_atorvastatin = (
        uptake_transporters = [:OATP2B1],
        is_pgp_substrate = true,
        pgp_efflux_ratio = 3.0,
        carrier_km_values = Dict(:OATP2B1 => 50.0, :PGP => 100.0),
        predictions = []
    )

    model_atorvastatin = generate_full_medlang_model(
        ml_atorvastatin;
        drug_name = "Atorvastatin",
        mw = 558.6,
        cl_hepatic = 40.0,  # High hepatic extraction
        cl_renal = 0.1,
        cyp_substrate = :CYP3A4,
        fu = 0.02
    )

    results_atorvastatin = simulate_oral(model_atorvastatin, 40.0; t_max=24.0)

    println("\n--- Atorvastatin (40 mg oral) ---")
    println("Cmax: $(round(results_atorvastatin["cmax"], digits=4)) mg/L")
    println("Tmax: $(round(results_atorvastatin["tmax"], digits=2)) h")
    # Atorvastatin F ≈ 12% (extensive first-pass)

    @test results_atorvastatin["cmax"] < results_metformin["cmax"]  # Much lower due to first-pass
end

# ===========================================================================
# Test 5: Transporter Annotation Generation
# ===========================================================================

@testset "Transporter Annotations" begin
    ml_result = (
        uptake_transporters = [:PEPT1, :MCT1, :LAT2],
        is_pgp_substrate = true,
        pgp_efflux_ratio = 6.0,
        carrier_km_values = Dict(:PEPT1 => 45.0, :MCT1 => 200.0, :LAT2 => 150.0, :PGP => 35.0),
        predictions = []
    )

    annotation = generate_transporter_annotation(ml_result)

    @test occursin("ML Transporter Predictions", annotation)
    @test occursin("PEPT1", annotation)
    @test occursin("MCT1", annotation)
    @test occursin("P-gp substrate", annotation)

    println("\n--- Transporter Annotation ---")
    println(annotation)
end

println("\n" * "="^70)
println("ALL TESTS PASSED - ML → MEDLANG INTEGRATION COMPLETE")
println("="^70)
println("""

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│  SMILES → TransporterPredictor (ML) → NamedTuple                │
│                                    ↓                            │
│                        MLMedLangIntegration                     │
│                                    ↓                            │
│                     MedLang DSL Model Code                      │
│                                    ↓                            │
│              MedLang.compile_model() → PBPKParams               │
│                                    ↓                            │
│             MedLang.simulate_oral() → C(t) profiles             │
└─────────────────────────────────────────────────────────────────┘

This uses FULL MEDLANG DSL:
- parse_medlang() for AST generation
- transpile_to_pbpk_params() for compilation
- simulate_oral_pbpk() for 15-compartment ODE
- validate_model() for model checking
- All MedLang Track D features available!
""")
