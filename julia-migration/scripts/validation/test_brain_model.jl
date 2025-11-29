#!/usr/bin/env julia
#
# BRAIN MODEL VALIDATION
# ======================
# Tests the enhanced brain compartment model against literature Kp values
# and Kp,uu data for CNS drugs.
#

using Printf
using Statistics

# Add project to load path
push!(LOAD_PATH, joinpath(@__DIR__, "../../src"))

include("../../src/DarwinPBPK/compartments/brain.jl")
using .BrainCompartment

println("="^70)
println("BRAIN COMPARTMENT MODEL VALIDATION")
println("="^70)
println()

# Test drugs with known brain Kp and Kp,uu values
test_drugs = [
    # Good CNS penetrators
    (name="Diazepam", logP=2.8, MW=285.0, TPSA=33.0, HBD=0, fup=0.02,
     is_base=false, is_acid=false, is_pgp=false,
     observed_Kp=0.9, observed_Kpuu=0.8,
     note="Benzodiazepine: optimal CNS"),

    (name="Caffeine", logP=-0.1, MW=194.0, TPSA=58.0, HBD=0, fup=0.65,
     is_base=true, is_acid=false, pKa=0.6, is_pgp=false,
     observed_Kp=0.8, observed_Kpuu=1.0,
     note="Freely equilibrates"),

    (name="Haloperidol", logP=4.3, MW=376.0, TPSA=40.0, HBD=1, fup=0.08,
     is_base=true, is_acid=false, pKa=8.3, is_pgp=false,
     observed_Kp=15.0, observed_Kpuu=3.0,
     note="Antipsychotic: brain accumulator"),

    # P-gp affected
    (name="Risperidone", logP=3.0, MW=410.0, TPSA=62.0, HBD=0, fup=0.10,
     is_base=true, is_acid=false, pKa=8.2, is_pgp=true,
     observed_Kp=10.0, observed_Kpuu=0.3,
     note="P-gp substrate: high Kp, low Kpuu"),

    (name="Loperamide", logP=4.8, MW=477.0, TPSA=44.0, HBD=1, fup=0.03,
     is_base=true, is_acid=false, pKa=8.6, is_pgp=true,
     observed_Kp=0.05, observed_Kpuu=0.02,
     note="Strong P-gp: no CNS effect"),

    (name="Morphine", logP=0.9, MW=285.0, TPSA=52.0, HBD=2, fup=0.65,
     is_base=true, is_acid=false, pKa=8.0, is_pgp=true,
     observed_Kp=0.3, observed_Kpuu=0.4,
     note="P-gp substrate: moderate BBB"),

    # Poor BBB penetrators
    (name="Atenolol", logP=-0.1, MW=266.0, TPSA=85.0, HBD=4, fup=0.95,
     is_base=true, is_acid=false, pKa=9.6, is_pgp=false,
     observed_Kp=0.04, observed_Kpuu=0.1,
     note="Too polar: minimal CNS"),
]

println("--- Brain Kp Predictions vs Literature ---")
println()

kp_errors = Float64[]
kpuu_errors = Float64[]
kp_within_2fold = 0
kp_within_3fold = 0
kpuu_within_2fold = 0
kpuu_within_3fold = 0

for drug in test_drugs
    # Build kwargs
    kwargs = Dict{Symbol, Any}(
        :logP => drug.logP,
        :fup => drug.fup,
        :MW => drug.MW,
        :TPSA => drug.TPSA,
        :HBD => drug.HBD,
        :is_base => drug.is_base,
        :is_acid => drug.is_acid,
        :is_pgp_substrate => drug.is_pgp,
    )

    if haskey(drug, :pKa)
        kwargs[:pKa] = drug.pKa
    end

    # Calculate Kp and Kpuu
    result = calculate_kpuu_brain(; kwargs...)
    predicted_Kp = result.Kp
    predicted_Kpuu = result.Kpuu

    observed_Kp = drug.observed_Kp
    observed_Kpuu = drug.observed_Kpuu

    # Calculate fold errors
    kp_fold_error = predicted_Kp > observed_Kp ?
                    predicted_Kp / observed_Kp :
                    observed_Kp / predicted_Kp

    kpuu_fold_error = predicted_Kpuu > observed_Kpuu ?
                      predicted_Kpuu / observed_Kpuu :
                      observed_Kpuu / predicted_Kpuu

    push!(kp_errors, kp_fold_error)
    push!(kpuu_errors, kpuu_fold_error)

    if kp_fold_error <= 2.0
        global kp_within_2fold += 1
    end
    if kp_fold_error <= 3.0
        global kp_within_3fold += 1
    end
    if kpuu_fold_error <= 2.0
        global kpuu_within_2fold += 1
    end
    if kpuu_fold_error <= 3.0
        global kpuu_within_3fold += 1
    end

    # Status
    kp_status = kp_fold_error <= 2.0 ? "✓" : (kp_fold_error <= 3.0 ? "~" : "✗")
    kpuu_status = kpuu_fold_error <= 2.0 ? "✓" : (kpuu_fold_error <= 3.0 ? "~" : "✗")

    @printf("%s %s:\n", kp_status, drug.name)
    @printf("  Kp:   Predicted %.3f, Observed %.3f (%.1fx error)\n",
            predicted_Kp, observed_Kp, kp_fold_error)
    @printf("  %s Kpuu: Predicted %.3f, Observed %.3f (%.1fx error)\n",
            kpuu_status, predicted_Kpuu, observed_Kpuu, kpuu_fold_error)
    @printf("  fub: %.3f | %s\n", result.fub, drug.note)
    println()
end

# Summary statistics
n = length(test_drugs)
kp_gmfe = exp(mean(log.(kp_errors)))
kpuu_gmfe = exp(mean(log.(kpuu_errors)))

println("-"^50)
@printf("SUMMARY (n=%d drugs):\n", n)
println()
@printf("Kp,brain:\n")
@printf("  GMFE: %.2f\n", kp_gmfe)
@printf("  Within 2-fold: %d/%d (%.0f%%)\n", kp_within_2fold, n, 100*kp_within_2fold/n)
@printf("  Within 3-fold: %d/%d (%.0f%%)\n", kp_within_3fold, n, 100*kp_within_3fold/n)
println()
@printf("Kp,uu (pharmacologically relevant):\n")
@printf("  GMFE: %.2f\n", kpuu_gmfe)
@printf("  Within 2-fold: %d/%d (%.0f%%)\n", kpuu_within_2fold, n, 100*kpuu_within_2fold/n)
@printf("  Within 3-fold: %d/%d (%.0f%%)\n", kpuu_within_3fold, n, 100*kpuu_within_3fold/n)
println()

# ============================================================================
# BBB PERMEABILITY ASSESSMENT
# ============================================================================

println("="^70)
println("BBB PERMEABILITY ASSESSMENT")
println("="^70)
println()

bbb_test_drugs = [
    (name="Diazepam", MW=285.0, logP=2.8, TPSA=33.0, HBD=0,
     expected="High", note="Optimal CNS drug"),
    (name="Caffeine", MW=194.0, logP=-0.1, TPSA=58.0, HBD=0,
     expected="Moderate", note="Small, polar but no HBDs"),
    (name="Haloperidol", MW=376.0, logP=4.3, TPSA=40.0, HBD=1,
     expected="Moderate", note="Lipophilic but possible P-gp"),
    (name="Loperamide", MW=477.0, logP=4.8, TPSA=44.0, HBD=1, is_pgp=true,
     expected="Low", note="P-gp substrate"),
    (name="Atenolol", MW=266.0, logP=-0.1, TPSA=85.0, HBD=4,
     expected="Low", note="Too many HBDs"),
    (name="Methotrexate", MW=454.0, logP=-1.8, TPSA=210.0, HBD=5,
     expected="Very Low", note="Very polar acid"),
]

for drug in bbb_test_drugs
    kwargs = Dict{Symbol, Any}(
        :MW => drug.MW,
        :logP => drug.logP,
        :TPSA => drug.TPSA,
        :HBD => drug.HBD,
    )
    if haskey(drug, :is_pgp)
        kwargs[:is_pgp_substrate] = drug.is_pgp
    end

    result = estimate_bbb_permeability(; kwargs...)

    match = result.category == drug.expected ? "✓" : "~"

    @printf("%s %s: Score %.1f → %s (expected %s)\n",
            match, drug.name, result.score, result.category, drug.expected)

    if length(result.bonuses) > 0
        @printf("  + %s\n", join(result.bonuses, ", "))
    end
    if length(result.penalties) > 0
        @printf("  - %s\n", join(result.penalties, ", "))
    end
    println()
end

# ============================================================================
# CNS DRUG CLASSIFICATION
# ============================================================================

println("="^70)
println("CNS DRUG CLASSIFICATION")
println("="^70)
println()

cns_drugs = [
    (name="Diazepam", logP=2.8, MW=285.0, TPSA=33.0, HBD=0, fup=0.02,
     expected_class="CNS+"),
    (name="Haloperidol", logP=4.3, MW=376.0, TPSA=40.0, HBD=1, fup=0.08,
     is_base=true, pKa=8.3, expected_class="CNS+"),
    (name="Risperidone", logP=3.0, MW=410.0, TPSA=62.0, HBD=0, fup=0.10,
     is_base=true, pKa=8.2, is_pgp=true, expected_class="CNS±"),
    (name="Loperamide", logP=4.8, MW=477.0, TPSA=44.0, HBD=1, fup=0.03,
     is_base=true, pKa=8.6, is_pgp=true, expected_class="CNS-"),
    (name="Atenolol", logP=-0.1, MW=266.0, TPSA=85.0, HBD=4, fup=0.95,
     is_base=true, pKa=9.6, expected_class="CNS-"),
]

for drug in cns_drugs
    kwargs = Dict{Symbol, Any}(
        :logP => drug.logP,
        :MW => drug.MW,
        :TPSA => drug.TPSA,
        :HBD => drug.HBD,
        :fup => drug.fup,
    )
    if haskey(drug, :is_base)
        kwargs[:is_base] = drug.is_base
    end
    if haskey(drug, :pKa)
        kwargs[:pKa] = drug.pKa
    end
    if haskey(drug, :is_pgp)
        kwargs[:is_pgp_substrate] = drug.is_pgp
    end

    result = predict_cns_penetration_class(; kwargs...)

    match = result.cns_class == drug.expected_class ? "✓" : "~"

    @printf("%s %s: %s (expected %s)\n",
            match, drug.name, result.cns_class, drug.expected_class)
    @printf("  Kp,uu: %.3f | BBB Score: %.1f | %s\n",
            result.Kpuu, result.bbb_score, result.recommendation)
    println()
end

# ============================================================================
# P-gp EFFLUX DEMONSTRATION
# ============================================================================

println("="^70)
println("P-gp EFFLUX EFFECT DEMONSTRATION")
println("="^70)
println()

println("Effect of P-gp on brain exposure (same physicochemistry):")
println()

# Compare with and without P-gp substrate status
test_drug = (logP=3.5, MW=400.0, TPSA=50.0, HBD=1, fup=0.10)

# Without P-gp
result_no_pgp = calculate_kpuu_brain(
    logP=test_drug.logP, fup=test_drug.fup,
    MW=test_drug.MW, TPSA=test_drug.TPSA, HBD=test_drug.HBD,
    is_base=true, pKa=8.5, is_pgp_substrate=false
)

# With P-gp
result_with_pgp = calculate_kpuu_brain(
    logP=test_drug.logP, fup=test_drug.fup,
    MW=test_drug.MW, TPSA=test_drug.TPSA, HBD=test_drug.HBD,
    is_base=true, pKa=8.5, is_pgp_substrate=true
)

@printf("Without P-gp substrate:\n")
@printf("  Kp,brain: %.2f | Kpuu: %.3f | %s\n",
        result_no_pgp.Kp, result_no_pgp.Kpuu, result_no_pgp.interpretation)
println()
@printf("With P-gp substrate:\n")
@printf("  Kp,brain: %.2f | Kpuu: %.3f | %s\n",
        result_with_pgp.Kp, result_with_pgp.Kpuu, result_with_pgp.interpretation)
println()
@printf("P-gp efflux ratio: %.1fx reduction in Kp,brain\n",
        result_no_pgp.Kp / result_with_pgp.Kp)
@printf("                   %.1fx reduction in Kp,uu\n",
        result_no_pgp.Kpuu / result_with_pgp.Kpuu)
println()

println("="^70)
println("ALL BRAIN MODEL TESTS COMPLETED")
println("="^70)
