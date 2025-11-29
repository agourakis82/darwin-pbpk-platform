#!/usr/bin/env julia
# ===========================================================================
# VALIDATION: IMPROVED Kp,uu MODEL
# ===========================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Printf
using Statistics

include(joinpath(@__DIR__, "..", "..", "src", "DarwinPBPK", "compartments", "brain_kpuu_improved.jl"))
using .BrainKpuuImproved

println("=" ^ 80)
println("IMPROVED Kp,uu MODEL VALIDATION")
println("=" ^ 80)
println()

# ===========================================================================
# VALIDATION DATASET (held out from training)
# ===========================================================================

# These 36 drugs from Ma et al. 2024 - use as validation
validation_set = [
    # (name, logP, fup, MW, pKa, charge_type, pgp_efflux_ratio, Kpuu_obs)
    ("Buspirone", 2.4, 0.05, 385.5, 7.3, :base, 1.0, 1.29),
    ("Carisoprodol", 2.4, 0.40, 260.3, nothing, :neutral, 1.0, 0.34),
    ("Carbamazepine", 2.5, 0.24, 236.3, nothing, :neutral, 1.5, 0.27),
    ("Chlorpromazine", 5.2, 0.05, 318.9, 9.3, :base, 2.0, 0.65),
    ("Citalopram", 3.5, 0.20, 324.4, 9.5, :base, 2.0, 0.68),
    ("Clozapine", 3.2, 0.05, 326.8, 7.5, :base, 1.0, 1.01),
    ("Cyclobenzaprine", 5.0, 0.07, 275.4, 8.5, :base, 1.0, 1.62),
    ("Diazepam", 2.8, 0.02, 284.7, nothing, :neutral, 1.0, 1.02),
    ("Fluvoxamine", 2.8, 0.23, 318.3, 9.4, :base, 1.0, 1.32),
    ("Fluoxetine", 4.0, 0.06, 309.3, 9.8, :base, 2.0, 0.89),
    ("Haloperidol", 4.3, 0.08, 375.9, 8.3, :base, 1.0, 1.06),
    ("Hydrocodone", 1.2, 0.55, 299.4, 8.9, :base, 2.5, 1.96),
    ("Hydroxyzine", 2.4, 0.07, 374.9, 7.1, :base, 3.0, 1.51),
    ("Lamotrigine", 1.9, 0.45, 256.1, 5.7, :base, 1.0, 0.64),
    ("Meprobamate", 0.7, 0.80, 218.3, nothing, :neutral, 1.0, 0.42),
    ("Metoclopramide", 2.6, 0.60, 299.8, 9.3, :base, 3.0, 0.52),
    ("Methylphenidate", 2.0, 0.85, 233.3, 8.8, :base, 1.0, 3.43),
    ("Midazolam", 3.9, 0.03, 325.8, 6.0, :base, 8.0, 0.14),
    ("Morphine", 0.9, 0.65, 285.3, 8.0, :base, 3.0, 0.72),
    ("Nortriptyline", 4.7, 0.07, 263.4, 9.7, :base, 2.0, 1.63),
    ("9-OH-Risperidone", 2.3, 0.23, 426.5, 8.2, :base, 25.0, 0.02),
    ("Paroxetine", 3.6, 0.05, 329.4, 9.9, :base, 2.5, 0.86),
    ("Phenacetin", 1.6, 0.70, 179.2, nothing, :neutral, 1.0, 0.55),
    ("Phenytoin", 2.5, 0.10, 252.3, nothing, :neutral, 1.5, 0.28),
    ("Propranolol", 3.5, 0.10, 259.3, 9.4, :base, 2.0, 3.08),
    ("Propoxyphene", 4.2, 0.22, 339.5, 9.0, :base, 2.0, 0.85),
    ("Quinidine", 3.4, 0.13, 324.4, 8.5, :base, 15.0, 0.05),
    ("Risperidone", 3.0, 0.10, 410.5, 8.2, :base, 5.0, 0.26),
    ("Selegiline", 2.7, 0.06, 187.3, 7.5, :base, 1.0, 1.30),
    ("Sertraline", 5.1, 0.02, 306.2, 9.5, :base, 2.5, 1.44),
    ("Sulpiride", -0.6, 0.60, 341.4, 9.1, :base, 20.0, 0.06),
    ("Thiopental", 2.9, 0.15, 242.3, nothing, :neutral, 1.0, 0.17),
    ("Trazodone", 2.8, 0.11, 371.9, 7.1, :base, 2.0, 0.56),
    ("Venlafaxine", 2.7, 0.73, 277.4, 9.4, :base, 2.5, 0.98),
    ("Warfarin", 2.7, 0.01, 308.3, 5.1, :acid, 1.0, 0.19),
    ("Zolpidem", 3.0, 0.08, 307.4, 6.2, :base, 1.0, 0.24),
]

# ===========================================================================
# RUN VALIDATION: MECHANISTIC ONLY
# ===========================================================================

println("TEST 1: Mechanistic Model Only")
println("-" ^ 60)

predictions_mech = Float64[]
observations = Float64[]

println()
@printf("%-20s %8s %8s %8s %6s\n", "Drug", "Obs", "Pred", "Ratio", "Status")
println("-" ^ 60)

for drug in validation_set
    name, logP, fup, MW, pKa, charge, pgp_er, kpuu_obs = drug

    result = calculate_kpuu_improved(
        logP=logP, fup=fup, MW=MW, pKa=pKa,
        charge_type=charge,
        pgp_efflux_ratio=pgp_er
    )

    push!(predictions_mech, result.kpuu)
    push!(observations, kpuu_obs)

    ratio = result.kpuu / kpuu_obs
    within_2fold = 0.5 <= ratio <= 2.0
    status = within_2fold ? "OK" : (0.33 <= ratio <= 3.0 ? "~" : "X")

    @printf("%-20s %8.2f %8.2f %8.2f %6s\n", name, kpuu_obs, result.kpuu, ratio, status)
end

println("-" ^ 60)

metrics_mech = calculate_validation_metrics(predictions_mech, observations)

println()
println("MECHANISTIC MODEL METRICS:")
@printf("  N:                  %d\n", metrics_mech.n)
@printf("  Within 2-fold:      %.1f%%\n", metrics_mech.pct_within_2fold)
@printf("  Within 3-fold:      %.1f%%\n", metrics_mech.pct_within_3fold)
@printf("  GMFE:               %.2f\n", metrics_mech.gmfe)
@printf("  AFE:                %.2f\n", metrics_mech.afe)
@printf("  RMSE (log):         %.2f\n", metrics_mech.rmse_log)
@printf("  R²:                 %.3f\n", metrics_mech.r_squared)
println()

# ===========================================================================
# RUN VALIDATION: HYBRID MODEL (with ML correction)
# ===========================================================================

println("=" ^ 60)
println("TEST 2: Hybrid Model (Mechanistic + ML Correction)")
println("-" ^ 60)

predictions_hybrid = Float64[]

println()
@printf("%-20s %8s %8s %8s %6s\n", "Drug", "Obs", "Pred", "Ratio", "Status")
println("-" ^ 60)

for (i, drug) in enumerate(validation_set)
    name, logP, fup, MW, pKa, charge, pgp_er, kpuu_obs = drug

    result = predict_kpuu_hybrid(
        logP=logP, fup=fup, MW=MW, pKa=pKa,
        charge_type=charge,
        pgp_efflux_ratio=pgp_er,
        use_ml_correction=true
    )

    push!(predictions_hybrid, result.kpuu)

    ratio = result.kpuu / kpuu_obs
    within_2fold = 0.5 <= ratio <= 2.0
    status = within_2fold ? "OK" : (0.33 <= ratio <= 3.0 ? "~" : "X")

    @printf("%-20s %8.2f %8.2f %8.2f %6s\n", name, kpuu_obs, result.kpuu, ratio, status)
end

println("-" ^ 60)

metrics_hybrid = calculate_validation_metrics(predictions_hybrid, observations)

println()
println("HYBRID MODEL METRICS:")
@printf("  N:                  %d\n", metrics_hybrid.n)
@printf("  Within 2-fold:      %.1f%%\n", metrics_hybrid.pct_within_2fold)
@printf("  Within 3-fold:      %.1f%%\n", metrics_hybrid.pct_within_3fold)
@printf("  GMFE:               %.2f\n", metrics_hybrid.gmfe)
@printf("  AFE:                %.2f\n", metrics_hybrid.afe)
@printf("  RMSE (log):         %.2f\n", metrics_hybrid.rmse_log)
@printf("  R²:                 %.3f\n", metrics_hybrid.r_squared)
println()

# ===========================================================================
# COMPARISON
# ===========================================================================

println("=" ^ 80)
println("COMPARISON: OLD vs NEW MODEL")
println("=" ^ 80)
println()

println("                          | Old Model | Mechanistic | Hybrid  | Target")
println("--------------------------|-----------|-------------|---------|--------")
@printf("Within 2-fold             |   47.2%%   |    %.1f%%    |  %.1f%%  |  >70%%\n",
        metrics_mech.pct_within_2fold, metrics_hybrid.pct_within_2fold)
@printf("RMSE (log)                |    0.53   |     %.2f    |   %.2f  |  <0.45\n",
        metrics_mech.rmse_log, metrics_hybrid.rmse_log)
@printf("R²                        |    0.12   |     %.2f    |   %.2f  |  >0.50\n",
        metrics_mech.r_squared, metrics_hybrid.r_squared)
@printf("GMFE                      |    2.70   |     %.2f    |   %.2f  |  <2.0\n",
        metrics_mech.gmfe, metrics_hybrid.gmfe)
println()

# ===========================================================================
# IDENTIFY REMAINING PROBLEM DRUGS
# ===========================================================================

println("=" ^ 60)
println("REMAINING OUTLIERS (outside 2-fold)")
println("-" ^ 60)

outliers = []
for (i, drug) in enumerate(validation_set)
    name = drug[1]
    kpuu_obs = drug[8]
    kpuu_pred = predictions_hybrid[i]
    ratio = kpuu_pred / kpuu_obs

    if !(0.5 <= ratio <= 2.0)
        push!(outliers, (name=name, obs=kpuu_obs, pred=kpuu_pred, ratio=ratio))
    end
end

if isempty(outliers)
    println("  No outliers! All predictions within 2-fold.")
else
    @printf("%-20s %8s %8s %8s %s\n", "Drug", "Obs", "Pred", "Ratio", "Issue")
    println("-" ^ 60)
    for o in sort(outliers, by=x->abs(log10(x.ratio)), rev=true)
        issue = o.ratio > 1 ? "OVER" : "UNDER"
        @printf("%-20s %8.2f %8.2f %8.2fx %s\n", o.name, o.obs, o.pred, o.ratio, issue)
    end
end
println()

# ===========================================================================
# FINAL ASSESSMENT
# ===========================================================================

println("=" ^ 80)
println("FINAL ASSESSMENT")
println("=" ^ 80)
println()

target_met = metrics_hybrid.pct_within_2fold >= 70.0

if target_met
    println("STATUS: TARGET MET!")
    println("  The improved model achieves ≥70% within 2-fold accuracy.")
    println("  This is comparable to published QSAR models and industry standards.")
else
    println("STATUS: IMPROVEMENT NEEDED")
    @printf("  Current: %.1f%% within 2-fold\n", metrics_hybrid.pct_within_2fold)
    println("  Target:  70% within 2-fold")
    println()
    println("  Remaining issues to address:")
    for o in outliers[1:min(5, length(outliers))]
        println("    - $(o.name): $(round(o.ratio, digits=1))x error")
    end
end

println()
println("=" ^ 80)
