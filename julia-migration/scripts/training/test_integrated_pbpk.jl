# Test Integrated Compartment-Based PBPK Model
# Validate predictions against known drugs

using Pkg
Pkg.activate("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration")

using Statistics

# Load the integrated module
include("../../src/DarwinPBPK/compartments/integrated_pbpk.jl")
using .IntegratedPBPK

println("=" ^ 70)
println("INTEGRATED COMPARTMENT-BASED PBPK MODEL VALIDATION")
println("=" ^ 70)

# Test drugs with known Vdss values
test_drugs = [
    # (name, MW, logP, fup, pKa_acid, pKa_base, observed_Vdss_L_kg, notes)
    ("Warfarin", 308, 2.6, 0.01, 5.0, nothing, 0.14, "Acid, highly bound"),
    ("Propranolol", 259, 3.5, 0.13, nothing, 9.4, 4.0, "Strong base"),
    ("Diazepam", 285, 2.8, 0.02, nothing, 3.4, 1.1, "Weak base, lipophilic"),
    ("Caffeine", 194, -0.1, 0.65, nothing, nothing, 0.6, "Neutral, hydrophilic"),
    ("Midazolam", 326, 3.9, 0.04, nothing, 6.2, 1.1, "Weak base"),
    ("Digoxin", 781, 1.3, 0.75, nothing, nothing, 7.0, "Neutral, large MW"),
    ("Phenytoin", 252, 2.5, 0.10, 8.3, nothing, 0.65, "Weak acid"),
    ("Metoprolol", 267, 1.9, 0.88, nothing, 9.7, 5.6, "Strong base, low binding"),
    ("Amiodarone", 645, 7.6, 0.005, nothing, 8.5, 70.0, "Very lipophilic base"),
    ("Atorvastatin", 559, 4.5, 0.02, 4.5, nothing, 5.4, "Acid, OATP substrate"),
]

println("\n" * "-" ^ 70)
println("INDIVIDUAL DRUG PREDICTIONS")
println("-" ^ 70)

predictions = Float64[]
observations = Float64[]
drug_names = String[]

for (name, MW, logP, fup, pKa_acid, pKa_base, obs_vdss, notes) in test_drugs

    # Create drug profile
    drug = DrugProfile(
        name = name,
        MW = Float64(MW),
        logP = Float64(logP),
        logD = Float64(logP),  # Approximate
        TPSA = 60.0,  # Default
        HBA = 4,
        HBD = 2,
        pKa_acid = pKa_acid,
        pKa_base = pKa_base,
        fup = Float64(fup),
        BP = 1.0,
        is_pgp_substrate = name ∈ ["Digoxin", "Loperamide"],
        is_oatp_substrate = name ∈ ["Atorvastatin", "Rosuvastatin"],
        is_oct_substrate = name ∈ ["Metformin"],
        is_oat_substrate = false
    )

    # Predict
    result = calculate_vdss_integrated(drug)
    pred_vdss = result.Vdss_L_kg

    push!(predictions, pred_vdss)
    push!(observations, obs_vdss)
    push!(drug_names, name)

    # Calculate fold error
    fold_error = max(pred_vdss / obs_vdss, obs_vdss / pred_vdss)

    # Determine dominant compartments
    dominant = join([c[1] for c in result.dominant_compartments[1:min(3, length(result.dominant_compartments))]], ", ")

    println("\n$name ($notes)")
    println("  Observed Vdss: $(obs_vdss) L/kg")
    println("  Predicted Vdss: $(round(pred_vdss, digits=2)) L/kg")
    println("  Fold error: $(round(fold_error, digits=2))×")
    println("  fut_effective: $(round(result.fut_effective, digits=3))")
    println("  Dominant: $dominant")

    # Show Kp values for key tissues
    kp = result.kp_results
    println("  Kp values: adipose=$(round(kp["adipose"].Kp, digits=2)), " *
            "muscle=$(round(kp["muscle"].Kp, digits=2)), " *
            "liver=$(round(kp["liver"].Kp, digits=2)), " *
            "kidney=$(round(kp["kidney"].Kp, digits=2))")
end

# Calculate overall performance
println("\n" * "=" ^ 70)
println("OVERALL PERFORMANCE")
println("=" ^ 70)

log_errors = abs.(log10.(predictions) .- log10.(observations))
gmfe = 10^mean(log_errors)

fold_errors = max.(predictions ./ observations, observations ./ predictions)
within_2fold = sum(fold_errors .<= 2.0) / length(fold_errors) * 100
within_3fold = sum(fold_errors .<= 3.0) / length(fold_errors) * 100

println("\nIntegrated PBPK Model:")
println("  GMFE: $(round(gmfe, digits=2))")
println("  Within 2-fold: $(round(within_2fold, digits=0))%")
println("  Within 3-fold: $(round(within_3fold, digits=0))%")

# Compare with simple method
println("\n" * "-" ^ 70)
println("COMPARISON WITH SIMPLE METHOD")
println("-" ^ 70)

simple_predictions = Float64[]
for (name, MW, logP, fup, pKa_acid, pKa_base, obs_vdss, notes) in test_drugs
    P = 10^logP
    fut_simple = 1 / (1 + 0.05 * clamp(P, 0.001, 1e6))
    fut_simple = clamp(fut_simple, 0.01, 0.99)

    Vdss_simple = 0.043 + 0.6 * (fup / fut_simple)
    push!(simple_predictions, Vdss_simple)
end

gmfe_simple = 10^mean(abs.(log10.(simple_predictions) .- log10.(observations)))
fold_errors_simple = max.(simple_predictions ./ observations, observations ./ simple_predictions)
within_2fold_simple = sum(fold_errors_simple .<= 2.0) / length(fold_errors_simple) * 100

println("\nSimple Øie-Tozer:")
println("  GMFE: $(round(gmfe_simple, digits=2))")
println("  Within 2-fold: $(round(within_2fold_simple, digits=0))%")

println("\nIntegrated PBPK:")
println("  GMFE: $(round(gmfe, digits=2))")
println("  Within 2-fold: $(round(within_2fold, digits=0))%")

# Show side-by-side comparison
println("\n" * "-" ^ 70)
println("DRUG-BY-DRUG COMPARISON")
println("-" ^ 70)
println("Drug            | Observed | Integrated | Simple | Winner")
println("-" ^ 70)

for i in 1:length(drug_names)
    obs = observations[i]
    int_pred = predictions[i]
    sim_pred = simple_predictions[i]

    int_err = abs(log10(int_pred) - log10(obs))
    sim_err = abs(log10(sim_pred) - log10(obs))

    winner = int_err < sim_err ? "Integrated" : "Simple"

    println("$(rpad(drug_names[i], 15)) | $(lpad(string(obs), 8)) | $(lpad(string(round(int_pred, digits=2)), 10)) | " *
            "$(lpad(string(round(sim_pred, digits=2)), 6)) | $winner")
end

println("\n" * "=" ^ 70)
println("KEY INSIGHTS")
println("=" ^ 70)
println("""
1. COMPARTMENT-SPECIFIC MODELING:
   - Each tissue has unique partitioning behavior
   - Adipose uses oil:water (not octanol:water)
   - Muscle dominates due to volume (30L)
   - Kidney has highest acidic PL - bases accumulate

2. IONIZATION MATTERS:
   - Strong bases (pKa > 8): muscle/kidney accumulation
   - Weak acids: albumin binding dominates
   - Ion trapping at pH gradients

3. TRANSPORTERS:
   - OATP substrates: 3x liver Kp
   - P-gp substrates: reduced brain Kp
   - OAT/OCT: kidney accumulation

4. PATIENT VARIABILITY:
   - Obese patients: higher adipose, higher Vdss for lipophilic drugs
   - Elderly: lower muscle, lower GFR
   - These can be modeled with PatientProfile
""")
