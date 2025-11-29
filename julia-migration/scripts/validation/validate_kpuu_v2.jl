# ===========================================================================
# VALIDATION SCRIPT FOR BRAIN Kp,uu MODEL v2.0
# ===========================================================================
# Target: >80% within 2-fold accuracy
# Comparison with v1 (72.2% achieved)
# ===========================================================================

using Printf

# Include the v2 model
include("../../src/DarwinPBPK/compartments/brain_kpuu_v2.jl")
using .BrainKpuuV2

# Include v1 for comparison
include("../../src/DarwinPBPK/compartments/brain_kpuu_improved.jl")
using .BrainKpuuImproved

# ===========================================================================
# EXTERNAL VALIDATION DATASET (Ma et al. 2024 - 36 drugs)
# ===========================================================================

const EXTERNAL_VALIDATION = [
    # (name, logP, fup, MW, pKa, charge, pgp_er, drug_class, Kpuu_obs)

    # === HIGH Kp,uu (>1.0) - Active uptake dominant ===
    ("Propranolol", 3.5, 0.10, 259.3, 9.4, :base, 2.0, :beta_blocker, 3.08),
    ("Methylphenidate", 2.0, 0.85, 233.3, 8.8, :base, 1.0, :stimulant, 3.43),
    ("Hydrocodone", 1.2, 0.55, 299.4, 8.9, :base, 2.5, :opioid, 1.96),
    ("Nortriptyline", 4.7, 0.07, 263.4, 9.7, :base, 2.0, :tca, 1.63),
    ("Cyclobenzaprine", 5.0, 0.07, 275.4, 8.5, :base, 1.0, :muscle_relaxant, 1.62),
    ("Hydroxyzine", 2.4, 0.07, 374.9, 7.1, :base, 3.0, :antihistamine, 1.51),
    ("Sertraline", 5.1, 0.02, 306.2, 9.5, :base, 2.5, :ssri, 1.44),
    ("Fluvoxamine", 2.8, 0.23, 318.3, 9.4, :base, 1.0, :ssri, 1.32),
    ("Selegiline", 2.7, 0.06, 187.3, 7.5, :base, 1.0, :maoi, 1.30),
    ("Buspirone", 2.4, 0.05, 385.5, 7.3, :base, 1.0, :anxiolytic, 1.29),
    ("Haloperidol", 4.3, 0.08, 375.9, 8.3, :base, 1.0, :antipsychotic, 1.06),
    ("Diazepam", 2.8, 0.02, 284.7, nothing, :neutral, 1.0, :benzodiazepine, 1.02),
    ("Clozapine", 3.2, 0.05, 326.8, 7.5, :base, 1.0, :antipsychotic, 1.01),
    ("Caffeine", -0.1, 0.65, 194.0, 0.6, :neutral, 1.0, :stimulant, 1.00),

    # === MODERATE Kp,uu (0.5-1.0) ===
    ("Venlafaxine", 2.7, 0.73, 277.4, 9.4, :base, 2.5, :snri, 0.98),
    ("Fluoxetine", 4.0, 0.06, 309.3, 9.8, :base, 2.0, :ssri, 0.89),
    ("Paroxetine", 3.6, 0.05, 329.4, 9.9, :base, 2.5, :ssri, 0.86),
    ("Propoxyphene", 4.2, 0.22, 339.5, 9.0, :base, 2.0, :opioid, 0.85),
    ("Morphine", 0.9, 0.65, 285.3, 8.0, :base, 3.0, :opioid, 0.72),
    ("Citalopram", 3.5, 0.20, 324.4, 9.5, :base, 2.0, :ssri, 0.68),
    ("Chlorpromazine", 5.2, 0.05, 318.9, 9.3, :base, 2.0, :antipsychotic, 0.65),
    ("Lamotrigine", 1.9, 0.45, 256.1, 5.7, :base, 1.0, :anticonvulsant, 0.64),
    ("Trazodone", 2.8, 0.11, 371.9, 7.1, :base, 2.0, :antidepressant, 0.56),
    ("Phenacetin", 1.6, 0.70, 179.2, nothing, :neutral, 1.0, :analgesic, 0.55),
    ("Metoclopramide", 2.6, 0.60, 299.8, 9.3, :base, 3.0, :prokinetic, 0.52),

    # === LOW Kp,uu (0.1-0.5) - Efflux dominant ===
    ("Meprobamate", 0.7, 0.80, 218.3, nothing, :neutral, 1.0, :anxiolytic, 0.42),
    ("Carisoprodol", 2.4, 0.40, 260.3, nothing, :neutral, 1.0, :muscle_relaxant, 0.34),
    ("Phenytoin", 2.5, 0.10, 252.3, nothing, :neutral, 1.5, :anticonvulsant, 0.28),
    ("Carbamazepine", 2.5, 0.24, 236.3, nothing, :neutral, 1.5, :anticonvulsant, 0.27),
    ("Risperidone", 3.0, 0.10, 410.5, 8.2, :base, 5.0, :antipsychotic, 0.26),
    ("Zolpidem", 3.0, 0.08, 307.4, 6.2, :neutral, 1.0, :imidazopyridine, 0.24),
    ("Warfarin", 2.7, 0.01, 308.3, 5.1, :acid, 1.0, :anticoagulant, 0.19),
    ("Thiopental", 2.9, 0.15, 242.3, nothing, :neutral, 1.0, :barbiturate, 0.17),
    ("Midazolam", 3.9, 0.03, 325.8, 6.0, :base, 8.0, :benzodiazepine, 0.14),

    # === VERY LOW Kp,uu (<0.1) - Strong efflux ===
    ("Sulpiride", -0.6, 0.60, 341.4, 9.1, :base, 20.0, :antipsychotic, 0.06),
    ("Quinidine", 3.4, 0.13, 324.4, 8.5, :base, 15.0, :antiarrhythmic, 0.05),
    ("Atenolol", -0.1, 0.95, 266.0, 9.6, :base, 1.0, :beta_blocker, 0.04),
    ("Digoxin", 1.3, 0.75, 780.9, nothing, :neutral, 30.0, :cardiac, 0.03),
    ("Loperamide", 4.8, 0.03, 477.0, 8.6, :base, 50.0, :opioid, 0.02),
    ("9-OH-Risperidone", 2.3, 0.23, 426.5, 8.2, :base, 25.0, :antipsychotic, 0.02),
    ("Methotrexate", -1.8, 0.50, 454.0, 4.7, :acid, 1.0, :antimetabolite, 0.02),
]

# ===========================================================================
# VALIDATION FUNCTIONS
# ===========================================================================

function validate_external_v2()
    println("=" ^ 70)
    println("BRAIN Kp,uu MODEL v2.0 - EXTERNAL VALIDATION")
    println("Target: >80% within 2-fold")
    println("=" ^ 70)
    println()

    results = []

    for drug in EXTERNAL_VALIDATION
        name, logP, fup, MW, pKa, charge, pgp, drug_class, kpuu_obs = drug

        # V2 prediction
        pred_v2 = predict_kpuu_v2(
            logP=logP, fup=fup, MW=MW, pKa=pKa,
            charge_type=charge, pgp_efflux_ratio=pgp, drug_class=drug_class,
            use_ml_correction=true
        )

        # V1 prediction for comparison
        pred_v1 = BrainKpuuImproved.predict_kpuu_hybrid(
            logP=logP, fup=fup, MW=MW, pKa=pKa,
            charge_type=charge, pgp_efflux_ratio=pgp
        )

        ratio_v2 = pred_v2.kpuu / kpuu_obs
        ratio_v1 = pred_v1.kpuu / kpuu_obs

        push!(results, (
            name = name,
            observed = kpuu_obs,
            pred_v2 = pred_v2.kpuu,
            pred_v1 = pred_v1.kpuu,
            ratio_v2 = ratio_v2,
            ratio_v1 = ratio_v1,
            within_2fold_v2 = 0.5 <= ratio_v2 <= 2.0,
            within_2fold_v1 = 0.5 <= ratio_v1 <= 2.0,
            drug_class = drug_class,
            oct_score = pred_v2.oct_score.score,
            bcrp_score = pred_v2.bcrp_score.score
        ))
    end

    # Sort by observed Kp,uu
    sort!(results, by=r -> r.observed, rev=true)

    # Print detailed results
    println("DETAILED PREDICTIONS:")
    println("-" ^ 90)
    @printf("%-20s %8s %8s %8s %8s %8s %6s %6s\n",
            "Drug", "Obs", "v2", "v1", "R_v2", "R_v1", "OCT", "BCRP")
    println("-" ^ 90)

    for r in results
        v2_flag = r.within_2fold_v2 ? "✓" : "✗"
        v1_flag = r.within_2fold_v1 ? "✓" : "✗"
        improved = !r.within_2fold_v1 && r.within_2fold_v2 ? " ↑" :
                   r.within_2fold_v1 && !r.within_2fold_v2 ? " ↓" : ""

        @printf("%-20s %8.2f %7.2f%s %7.2f%s %8.2f %8.2f %6.2f %6.2f%s\n",
                r.name[1:min(20, length(r.name))],
                r.observed, r.pred_v2, v2_flag, r.pred_v1, v1_flag,
                r.ratio_v2, r.ratio_v1, r.oct_score, r.bcrp_score, improved)
    end

    println("-" ^ 90)

    # Calculate metrics
    n = length(results)

    n_2fold_v2 = count(r -> r.within_2fold_v2, results)
    n_2fold_v1 = count(r -> r.within_2fold_v1, results)

    n_3fold_v2 = count(r -> 0.33 <= r.ratio_v2 <= 3.0, results)
    n_3fold_v1 = count(r -> 0.33 <= r.ratio_v1 <= 3.0, results)

    pct_2fold_v2 = 100.0 * n_2fold_v2 / n
    pct_2fold_v1 = 100.0 * n_2fold_v1 / n
    pct_3fold_v2 = 100.0 * n_3fold_v2 / n
    pct_3fold_v1 = 100.0 * n_3fold_v1 / n

    # GMFE
    gmfe_v2 = 10^(sum(abs(log10(r.ratio_v2)) for r in results) / n)
    gmfe_v1 = 10^(sum(abs(log10(r.ratio_v1)) for r in results) / n)

    # AFE
    afe_v2 = 10^(sum(log10(r.ratio_v2) for r in results) / n)
    afe_v1 = 10^(sum(log10(r.ratio_v1) for r in results) / n)

    # R² in log space
    log_obs = [log10(r.observed) for r in results]
    log_pred_v2 = [log10(r.pred_v2) for r in results]
    log_pred_v1 = [log10(r.pred_v1) for r in results]

    ss_tot = sum((log_obs .- mean(log_obs)).^2)
    ss_res_v2 = sum((log_obs .- log_pred_v2).^2)
    ss_res_v1 = sum((log_obs .- log_pred_v1).^2)

    r2_v2 = 1 - ss_res_v2 / ss_tot
    r2_v1 = 1 - ss_res_v1 / ss_tot

    # Print comparison
    println()
    println("=" ^ 70)
    println("MODEL COMPARISON: v2.0 vs v1.0")
    println("=" ^ 70)
    println()

    println("METRIC              v2.0         v1.0        Target    Status")
    println("-" ^ 70)

    target_2fold = pct_2fold_v2 >= 80.0 ? "✓ MET" : "✗ MISS"
    @printf("Within 2-fold:      %5.1f%%       %5.1f%%       >80%%     %s\n",
            pct_2fold_v2, pct_2fold_v1, target_2fold)

    @printf("Within 3-fold:      %5.1f%%       %5.1f%%       >90%%\n",
            pct_3fold_v2, pct_3fold_v1)

    target_gmfe = gmfe_v2 <= 1.8 ? "✓ MET" : "✗ MISS"
    @printf("GMFE:               %5.2f        %5.2f        <1.80    %s\n",
            gmfe_v2, gmfe_v1, target_gmfe)

    target_afe = 0.8 <= afe_v2 <= 1.25 ? "✓ MET" : "✗ MISS"
    @printf("AFE:                %5.2f        %5.2f        0.8-1.25 %s\n",
            afe_v2, afe_v1, target_afe)

    target_r2 = r2_v2 >= 0.60 ? "✓ MET" : "✗ MISS"
    @printf("R² (log):           %5.2f        %5.2f        >0.60    %s\n",
            r2_v2, r2_v1, target_r2)

    println("-" ^ 70)

    # Improvement summary
    improved = count(r -> !r.within_2fold_v1 && r.within_2fold_v2, results)
    regressed = count(r -> r.within_2fold_v1 && !r.within_2fold_v2, results)

    println()
    println("CHANGE FROM v1 TO v2:")
    println("  Newly within 2-fold: $improved drugs")
    println("  Newly outside 2-fold: $regressed drugs")
    println("  Net improvement: $(improved - regressed) drugs")

    # Identify remaining outliers
    outliers = filter(r -> !r.within_2fold_v2, results)

    if !isempty(outliers)
        println()
        println("REMAINING OUTLIERS ($(length(outliers)) drugs):")
        println("-" ^ 60)
        for r in outliers
            dir = r.ratio_v2 > 2.0 ? "OVER-predicted" : "UNDER-predicted"
            @printf("  %-18s: Obs=%.2f, Pred=%.2f (%.1fx) - %s\n",
                    r.name, r.observed, r.pred_v2, r.ratio_v2, dir)
        end
    end

    # Return results
    return (
        n = n,
        pct_2fold_v2 = pct_2fold_v2,
        pct_2fold_v1 = pct_2fold_v1,
        pct_3fold_v2 = pct_3fold_v2,
        gmfe_v2 = gmfe_v2,
        gmfe_v1 = gmfe_v1,
        afe_v2 = afe_v2,
        r2_v2 = r2_v2,
        r2_v1 = r2_v1,
        improved = improved,
        regressed = regressed,
        outliers = outliers,
        results = results
    )
end

# Helper function
function mean(x)
    sum(x) / length(x)
end

# ===========================================================================
# RUN VALIDATION
# ===========================================================================

println("\n" * "=" ^ 70)
println("STARTING VALIDATION...")
println("=" ^ 70 * "\n")

result = validate_external_v2()

println("\n" * "=" ^ 70)
println("FINAL ASSESSMENT")
println("=" ^ 70)

if result.pct_2fold_v2 >= 80.0
    println("\n🎯 TARGET MET! v2.0 achieves $(round(result.pct_2fold_v2, digits=1))% within 2-fold")
    println("   (Improvement from $(round(result.pct_2fold_v1, digits=1))% in v1)")
elseif result.pct_2fold_v2 > result.pct_2fold_v1
    println("\n📈 IMPROVED but target not met")
    println("   v2.0: $(round(result.pct_2fold_v2, digits=1))% vs v1.0: $(round(result.pct_2fold_v1, digits=1))%")
    println("   Need to fix $(ceil(Int, (80 - result.pct_2fold_v2) * result.n / 100)) more drugs to reach 80%")
else
    println("\n⚠️ NO IMPROVEMENT - v2.0 did not outperform v1.0")
end
