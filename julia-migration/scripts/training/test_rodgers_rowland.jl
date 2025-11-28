# Test Rodgers-Rowland mechanistic tissue partition prediction
# Compare with our current best model (GMFE 1.90)

using Pkg
Pkg.activate("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration")

# Load the tissue partition module
include("../../src/DarwinPBPK/tissue_partition.jl")
using .TissuePartition

println("=" ^ 60)
println("RODGERS-ROWLAND TISSUE PARTITION TEST")
println("=" ^ 60)

# Test with some example drugs
println("\n1. Testing with known drugs...")

# Diazepam: logP=2.8, pKa=3.4 (weak base), fup=0.02, Vdss≈1.1 L/kg
diazepam = DrugProperties(2.8, [3.4], 0.02, 1.0, MONOPROTIC_BASE)
kp_diazepam = calculate_all_kp(diazepam)
vdss_diazepam = calculate_vdss_mechanistic(diazepam)
fut_diazepam = calculate_fut(diazepam)

println("\nDiazepam:")
println("  fup = $(diazepam.fup)")
println("  Predicted fut = $(round(fut_diazepam, digits=4))")
println("  fup/fut ratio = $(round(diazepam.fup/fut_diazepam, digits=4))")
println("  Predicted Vdss = $(round(vdss_diazepam, digits=2)) L")
println("  Vdss/70kg = $(round(vdss_diazepam/70, digits=2)) L/kg")
println("  Literature Vdss ≈ 1.1 L/kg")
println("  Key Kp values:")
println("    Adipose: $(round(kp_diazepam["adipose"], digits=2))")
println("    Muscle: $(round(kp_diazepam["muscle"], digits=2))")
println("    Brain: $(round(kp_diazepam["brain"], digits=2))")
println("    Liver: $(round(kp_diazepam["liver"], digits=2))")

# Propranolol: logP=3.5, pKa=9.4 (strong base), fup=0.13, Vdss≈4 L/kg
propranolol = DrugProperties(3.5, [9.4], 0.13, 1.0, MONOPROTIC_BASE)
kp_propranolol = calculate_all_kp(propranolol)
vdss_propranolol = calculate_vdss_mechanistic(propranolol)
fut_propranolol = calculate_fut(propranolol)

println("\nPropranolol:")
println("  fup = $(propranolol.fup)")
println("  Predicted fut = $(round(fut_propranolol, digits=4))")
println("  fup/fut ratio = $(round(propranolol.fup/fut_propranolol, digits=4))")
println("  Predicted Vdss = $(round(vdss_propranolol, digits=2)) L")
println("  Vdss/70kg = $(round(vdss_propranolol/70, digits=2)) L/kg")
println("  Literature Vdss ≈ 4 L/kg")

# Warfarin: logP=2.6, pKa=5.0 (weak acid), fup=0.01, Vdss≈0.14 L/kg
warfarin = DrugProperties(2.6, [5.0], 0.01, 1.0, MONOPROTIC_ACID)
kp_warfarin = calculate_all_kp(warfarin)
vdss_warfarin = calculate_vdss_mechanistic(warfarin)
fut_warfarin = calculate_fut(warfarin)

println("\nWarfarin:")
println("  fup = $(warfarin.fup)")
println("  Predicted fut = $(round(fut_warfarin, digits=4))")
println("  fup/fut ratio = $(round(warfarin.fup/fut_warfarin, digits=4))")
println("  Predicted Vdss = $(round(vdss_warfarin, digits=2)) L")
println("  Vdss/70kg = $(round(vdss_warfarin/70, digits=3)) L/kg")
println("  Literature Vdss ≈ 0.14 L/kg")

# Midazolam: logP=3.9, pKa=6.2 (weak base), fup=0.04, Vdss≈1.1 L/kg
midazolam = DrugProperties(3.9, [6.2], 0.04, 1.0, MONOPROTIC_BASE)
vdss_midazolam = calculate_vdss_mechanistic(midazolam)
fut_midazolam = calculate_fut(midazolam)

println("\nMidazolam:")
println("  fup = $(midazolam.fup)")
println("  Predicted fut = $(round(fut_midazolam, digits=4))")
println("  Predicted Vdss = $(round(vdss_midazolam, digits=2)) L")
println("  Vdss/70kg = $(round(vdss_midazolam/70, digits=2)) L/kg")
println("  Literature Vdss ≈ 1.1 L/kg")

# Caffeine: logP=-0.1, neutral, fup=0.65, Vdss≈0.6 L/kg
caffeine = DrugProperties(-0.1, Float64[], 0.65, 1.0, NEUTRAL)
vdss_caffeine = calculate_vdss_mechanistic(caffeine)
fut_caffeine = calculate_fut(caffeine)

println("\nCaffeine:")
println("  fup = $(caffeine.fup)")
println("  Predicted fut = $(round(fut_caffeine, digits=4))")
println("  Predicted Vdss = $(round(vdss_caffeine, digits=2)) L")
println("  Vdss/70kg = $(round(vdss_caffeine/70, digits=2)) L/kg")
println("  Literature Vdss ≈ 0.6 L/kg")

println("\n" * "=" ^ 60)
println("2. Testing on Lombardo dataset...")
println("=" ^ 60)

# Load Lombardo dataset
using CSV, DataFrames

data_path = "/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/Lombardo"
lombardo_file = joinpath(data_path, "Lombardo_Vdss.csv")

if isfile(lombardo_file)
    df = CSV.read(lombardo_file, DataFrame)
    println("\nLoaded $(nrow(df)) compounds from Lombardo dataset")

    # Check available columns
    println("Columns: $(names(df))")

    # We need: SMILES, Vdss, fup, logP, and ideally pKa
    # Let's see what we have
    if "fup" in names(df) && "Vdss" in names(df) && "logP" in names(df)

        # Filter for compounds with required data
        df_valid = dropmissing(df, [:fup, :Vdss, :logP])
        println("Compounds with fup, Vdss, logP: $(nrow(df_valid))")

        # If we have pKa
        has_pka = "pKa" in names(df) || "pKa_basic" in names(df) || "pka_basic" in names(df)

        if has_pka
            println("pKa data available!")
        else
            println("No pKa data - will treat all as neutral (conservative)")
        end

        # Test prediction
        predictions = Float64[]
        observations = Float64[]

        for row in eachrow(df_valid)
            # Get drug properties
            logP = row.logP
            fup = row.fup
            vdss_obs = row.Vdss  # L/kg

            # Skip invalid values
            if fup <= 0 || fup >= 1 || logP < -5 || logP > 8 || vdss_obs <= 0
                continue
            end

            # Create drug (assume neutral without pKa info)
            drug = DrugProperties(logP, Float64[], fup, 1.0, NEUTRAL)

            # Predict Vdss
            vdss_pred_L = calculate_vdss_mechanistic(drug)
            vdss_pred_Lkg = vdss_pred_L / 70.0

            push!(predictions, vdss_pred_Lkg)
            push!(observations, vdss_obs)
        end

        println("\nEvaluated $(length(predictions)) compounds")

        # Calculate GMFE
        log_errors = abs.(log10.(predictions) .- log10.(observations))
        gmfe = 10^mean(log_errors)

        println("\nRodgers-Rowland (all neutral) GMFE: $(round(gmfe, digits=3))")

        # Calculate fold error distribution
        fold_errors = max.(predictions ./ observations, observations ./ predictions)
        within_2fold = sum(fold_errors .<= 2.0) / length(fold_errors) * 100
        within_3fold = sum(fold_errors .<= 3.0) / length(fold_errors) * 100

        println("Within 2-fold: $(round(within_2fold, digits=1))%")
        println("Within 3-fold: $(round(within_3fold, digits=1))%")

        # Compare with simple fup/fut approach
        println("\n" * "-" ^ 40)
        println("Comparison with simple fut estimation:")
        println("-" ^ 40)

        simple_predictions = Float64[]
        for row in eachrow(df_valid)
            logP = row.logP
            fup = row.fup
            vdss_obs = row.Vdss

            if fup <= 0 || fup >= 1 || logP < -5 || logP > 8 || vdss_obs <= 0
                continue
            end

            # Simple fut estimation (from our best model)
            P = 10^logP
            fut_simple = 1 / (1 + 0.05 * clamp(P, 0.001, 1e6))
            fut_simple = clamp(fut_simple, 0.01, 0.99)

            # Øie-Tozer prediction
            Vp = 0.043  # L/kg plasma
            Ve = 0.15   # L/kg extracellular
            Vr = 0.45   # L/kg rest
            vdss_simple = Vp + Ve * (fup/fut_simple) + Vr * (fup/fut_simple)

            push!(simple_predictions, vdss_simple)
        end

        log_errors_simple = abs.(log10.(simple_predictions) .- log10.(observations))
        gmfe_simple = 10^mean(log_errors_simple)
        println("Simple fut estimation GMFE: $(round(gmfe_simple, digits=3))")

    else
        println("Missing required columns (fup, Vdss, logP)")
    end
else
    println("Lombardo dataset not found at: $lombardo_file")

    # Try to find it
    println("\nSearching for dataset...")
    for (root, dirs, files) in walkdir("/home/agourakis82/workspace/darwin-pbpk-platform/data")
        for f in files
            if occursin("Lombardo", f) && endswith(f, ".csv")
                println("Found: $(joinpath(root, f))")
            end
        end
    end
end

println("\n" * "=" ^ 60)
println("KEY INSIGHTS")
println("=" ^ 60)
println("""
The Rodgers-Rowland equations provide MECHANISTIC understanding:

1. Tissue Kp depends on:
   - Neutral lipid content (adipose: 85%, muscle: 1%)
   - Phospholipid content (kidney: 2.4%, brain: 0.15%)
   - Acidic phospholipids (kidney: 5×10⁻³, muscle: 1.5×10⁻³)
   - Water content (intracellular vs extracellular)
   - pH-dependent ionization (bases accumulate in acidic lysosomes)

2. Drug ionization matters:
   - Bases (pKa > 7): bind to acidic phospholipids → higher Kp
   - Acids: albumin binding dominates
   - Neutrals: lipid partitioning only

3. The key insight for fut estimation:
   - fut is NOT uniform across tissues
   - It depends on tissue composition and drug properties
   - Volume-weighted average gives effective fut for Vdss

4. Missing data problem:
   - Without pKa, we can't properly classify drugs
   - Treating all as neutral is conservative but inaccurate for bases
   - Need pKa data to unlock full potential of this approach
""")
