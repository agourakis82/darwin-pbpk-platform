#!/usr/bin/env julia
#
# KIDNEY MODEL VALIDATION
# ========================
# Tests the enhanced kidney compartment model against literature Kp values
# and renal clearance data.
#

using Printf
using Statistics

# Add project to load path
push!(LOAD_PATH, joinpath(@__DIR__, "../../src"))

include("../../src/DarwinPBPK/compartments/kidney.jl")
using .KidneyCompartment

println("="^70)
println("KIDNEY COMPARTMENT MODEL VALIDATION")
println("="^70)
println()

# Test drugs with known kidney Kp values
test_drugs = [
    # AMINOGLYCOSIDES - massive kidney accumulation
    (name="Gentamicin", logP=-3.1, pKa=8.2, fup=0.90, is_base=true, is_acid=false,
     is_aminoglycoside=true, n_charges=5, observed_Kp=15.0,
     note="Aminoglycoside, nephrotoxic"),

    # OCT2 + MATE substrates
    (name="Metformin", logP=-1.5, pKa=11.5, fup=0.99, is_base=true, is_acid=false,
     is_oct2=true, is_mate=true, observed_Kp=5.0,
     note="OCT2+MATE, CLrenal 4x GFR"),

    # OAT substrates
    (name="Furosemide", logP=2.0, pKa=3.9, fup=0.01, is_base=false, is_acid=true,
     is_oat3=true, observed_Kp=2.5,
     note="OAT3 substrate, loop diuretic"),

    (name="Tenofovir", logP=-1.6, pKa=3.8, fup=0.93, is_base=false, is_acid=true,
     is_oat1=true, observed_Kp=4.0,
     note="OAT1 substrate, antiviral"),

    # Lipophilic bases
    (name="Propranolol", logP=3.5, pKa=9.5, fup=0.10, is_base=true, is_acid=false,
     observed_Kp=3.5,
     note="Lipophilic base, APL + lysosomal"),

    (name="Imipramine", logP=4.8, pKa=9.4, fup=0.10, is_base=true, is_acid=false,
     observed_Kp=4.0,
     note="TCA, strong lysosomal trapping"),

    # Neutral / P-gp
    (name="Digoxin", logP=1.3, pKa=nothing, fup=0.75, is_base=false, is_acid=false,
     is_pgp=true, observed_Kp=0.5,
     note="P-gp substrate, cardiac glycoside"),

    # Highly bound acid
    (name="Warfarin", logP=2.6, pKa=5.1, fup=0.01, is_base=false, is_acid=true,
     observed_Kp=0.3,
     note="Highly bound, minimal kidney distribution"),
]

println("--- Kp Predictions vs Literature ---")
println()

errors = Float64[]
within_2fold = 0
within_3fold = 0

for drug in test_drugs
    # Build kwargs
    kwargs = Dict{Symbol, Any}(
        :logP => drug.logP,
        :fup => drug.fup,
        :is_base => drug.is_base,
        :is_acid => drug.is_acid,
    )

    if !isnothing(drug.pKa)
        kwargs[:pKa] = drug.pKa
    end

    # Handle optional parameters using haskey on NamedTuple
    if haskey(drug, :is_aminoglycoside) && drug.is_aminoglycoside
        kwargs[:is_aminoglycoside] = true
        kwargs[:n_positive_charges] = drug.n_charges
    end

    if haskey(drug, :is_oct2) && drug.is_oct2
        kwargs[:is_oct2_substrate] = true
    end
    if haskey(drug, :is_mate) && drug.is_mate
        kwargs[:is_mate_substrate] = true
    end
    if haskey(drug, :is_oat1) && drug.is_oat1
        kwargs[:is_oat1_substrate] = true
    end
    if haskey(drug, :is_oat3) && drug.is_oat3
        kwargs[:is_oat3_substrate] = true
    end
    if haskey(drug, :is_pgp) && drug.is_pgp
        kwargs[:is_pgp_substrate] = true
    end

    # Calculate Kp
    predicted_Kp = calculate_kp_kidney(; kwargs...)
    observed_Kp = drug.observed_Kp

    # Calculate fold error
    fold_error = predicted_Kp > observed_Kp ?
                 predicted_Kp / observed_Kp :
                 observed_Kp / predicted_Kp

    push!(errors, fold_error)

    if fold_error <= 2.0
        global within_2fold += 1
    end
    if fold_error <= 3.0
        global within_3fold += 1
    end

    # Status
    status = fold_error <= 2.0 ? "✓" : (fold_error <= 3.0 ? "~" : "✗")

    @printf("%s %s:\n", status, drug.name)
    @printf("  Predicted: %.2f, Observed: %.2f (%.1fx error)\n",
            predicted_Kp, observed_Kp, fold_error)
    @printf("  %s\n\n", drug.note)
end

# Summary statistics
n = length(test_drugs)
gmfe = exp(mean(log.(errors)))

println("-"^50)
@printf("SUMMARY (n=%d drugs):\n", n)
@printf("  GMFE: %.2f\n", gmfe)
@printf("  Within 2-fold: %d/%d (%.0f%%)\n", within_2fold, n, 100*within_2fold/n)
@printf("  Within 3-fold: %d/%d (%.0f%%)\n", within_3fold, n, 100*within_3fold/n)
println()

# ============================================================================
# RENAL CLEARANCE VALIDATION
# ============================================================================

println("="^70)
println("RENAL CLEARANCE PREDICTIONS")
println("="^70)
println()

cl_test_drugs = [
    (name="Metformin", fup=0.99, logP=-1.5, pKa=11.5, is_base=true, is_acid=false,
     is_oct2=true, is_mate=true, observed_CLr=500.0,
     note="OCT2+MATE: CLrenal 4-5x GFR"),

    (name="Penicillin G", fup=0.45, logP=1.8, pKa=2.8, is_base=false, is_acid=true,
     is_oat1=true, is_oat3=true, observed_CLr=400.0,
     note="OAT1/3: major secretion"),

    (name="Furosemide", fup=0.01, logP=2.0, pKa=3.9, is_base=false, is_acid=true,
     is_oat3=true, observed_CLr=120.0,
     note="OAT3: secretion overcomes binding"),

    (name="Digoxin", fup=0.75, logP=1.3, pKa=nothing, is_base=false, is_acid=false,
     observed_CLr=90.0,
     note="Mainly filtration: ~fup × GFR"),

    (name="Propranolol", fup=0.10, logP=3.5, pKa=9.5, is_base=true, is_acid=false,
     observed_CLr=1.0,
     note="Lipophilic: extensive reabsorption"),
]

println("--- Renal Clearance Predictions ---")
println()

for drug in cl_test_drugs
    kwargs = Dict{Symbol, Any}(
        :fup => drug.fup,
        :logP => drug.logP,
        :is_base => drug.is_base,
        :is_acid => drug.is_acid,
    )

    if !isnothing(drug.pKa)
        kwargs[:pKa] = drug.pKa
    end
    if haskey(drug, :is_oct2) && drug.is_oct2
        kwargs[:is_oct2_substrate] = true
    end
    if haskey(drug, :is_mate) && drug.is_mate
        kwargs[:is_mate_substrate] = true
    end
    if haskey(drug, :is_oat1) && drug.is_oat1
        kwargs[:is_oat1_substrate] = true
    end
    if haskey(drug, :is_oat3) && drug.is_oat3
        kwargs[:is_oat3_substrate] = true
    end

    result = estimate_renal_clearance_contribution(; kwargs...)

    @printf("%s:\n", drug.name)
    @printf("  CL_filtration: %.1f mL/min\n", result.CL_filtration)
    @printf("  Secretion ratio: %.1f\n", result.secretion_ratio)
    @printf("  Reabsorption: %.0f%%\n", result.reabsorption * 100)
    @printf("  CL_renal predicted: %.1f mL/min (observed: %.1f)\n",
            result.CL_renal, drug.observed_CLr)
    @printf("  %s\n\n", drug.note)
end

# ============================================================================
# NEPHROTOXICITY RISK ASSESSMENT
# ============================================================================

println("="^70)
println("NEPHROTOXICITY RISK ASSESSMENT")
println("="^70)
println()

nephrotox_drugs = [
    (name="Gentamicin", Kp=15.0, is_aminoglycoside=true, dose=5.0,
     expected="Very High"),
    (name="Cisplatin", Kp=8.0, is_oct2=true, is_base=false, logP=-2.2,
     expected="High"),
    (name="Metformin", Kp=5.0, is_oct2=true, is_base=true, pKa=11.5, logP=-1.5,
     expected="Moderate"),
    (name="Propranolol", Kp=3.5, is_base=true, pKa=9.5, logP=3.5,
     expected="Low"),
]

for drug in nephrotox_drugs
    kwargs = Dict{Symbol, Any}(
        :Kp_kidney => drug.Kp,
    )

    if haskey(drug, :is_aminoglycoside) && drug.is_aminoglycoside
        kwargs[:is_aminoglycoside] = true
        kwargs[:dose_mg_kg] = drug.dose
    end
    if haskey(drug, :is_oct2) && drug.is_oct2
        kwargs[:is_oct2_substrate] = true
    end
    if haskey(drug, :is_base)
        kwargs[:is_base] = drug.is_base
    end
    if haskey(drug, :pKa)
        kwargs[:pKa] = drug.pKa
    end
    if haskey(drug, :logP)
        kwargs[:logP] = drug.logP
    end

    result = estimate_nephrotoxicity_risk(; kwargs...)

    status = result.risk_level == drug.expected ? "✓" : "~"

    @printf("%s %s: Risk %.1f/10 (%s, expected %s)\n",
            status, drug.name, result.risk_score, result.risk_level, drug.expected)
    for mech in result.mechanisms
        @printf("    → %s\n", mech)
    end
    println()
end

# ============================================================================
# URINE PH EFFECTS
# ============================================================================

println("="^70)
println("URINE pH EFFECTS ON REABSORPTION")
println("="^70)
println()

println("Weak Acid (pKa 4.5, logP 2.0) - e.g., Salicylic acid")
for pH in [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    reabs = calculate_reabsorption_fraction(
        logP=2.0, pKa=4.5, is_acid=true, urine_pH=pH
    )
    bar = "█"^Int(round(reabs * 50))
    @printf("  pH %.1f: %.0f%% %s\n", pH, reabs*100, bar)
end
println("  → Alkalinize urine to increase excretion of weak acids")
println()

println("Weak Base (pKa 9.0, logP 2.0) - e.g., Amphetamine")
for pH in [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    reabs = calculate_reabsorption_fraction(
        logP=2.0, pKa=9.0, is_base=true, urine_pH=pH
    )
    bar = "█"^Int(round(reabs * 50))
    @printf("  pH %.1f: %.0f%% %s\n", pH, reabs*100, bar)
end
println("  → Acidify urine to increase excretion of weak bases")
println()

println("="^70)
println("ALL KIDNEY MODEL TESTS COMPLETED")
println("="^70)
