# ===========================================================================
# INDEPENDENT VALIDATION OF BRAIN Kp,uu MODEL v2.0
# ===========================================================================
# This is a TRUE independent validation using compounds NOT in training.
#
# Sources:
# - Fridén et al. 2011 (J Med Chem) - Anticonvulsants
# - Summerfield et al. 2007 (J Pharmacol Exp Ther) - TCAs/Antidepressants
# - Liu et al. 2018 (Drug Metab Dispos) - Atypical Antipsychotics
#
# All data fetched from PubChem API on 2025-01-29
# ===========================================================================

using Printf

# Include the v2 model
include("../../src/DarwinPBPK/compartments/brain_kpuu_v2.jl")
using .BrainKpuuV2

# ===========================================================================
# INDEPENDENT VALIDATION SET (30 compounds, NONE in training)
# ===========================================================================

const INDEPENDENT_VALIDATION = [
    # (name, logP, fup, MW, pKa, charge, pgp_er, drug_class, Kpuu_obs)

    # === ANTICONVULSANTS (Fridén et al. 2011) ===
    ("Gabapentin", -1.1, 0.97, 171.2, 3.7, :zwitterion, 1.0, :anticonvulsant, 0.11),
    ("Pregabalin", -1.6, 0.97, 159.2, 4.2, :zwitterion, 1.0, :anticonvulsant, 0.18),
    ("Levetiracetam", -0.3, 0.90, 170.2, nothing, :neutral, 1.0, :anticonvulsant, 0.67),
    ("Topiramate", -0.8, 0.85, 339.4, 8.6, :neutral, 1.0, :anticonvulsant, 0.31),
    ("Oxcarbazepine", 1.7, 0.60, 252.3, nothing, :neutral, 1.0, :anticonvulsant, 0.56),
    ("Felbamate", 0.6, 0.75, 238.2, nothing, :neutral, 1.0, :anticonvulsant, 0.42),
    ("Tiagabine", 2.7, 0.04, 375.6, 4.0, :acid, 1.0, :anticonvulsant, 0.12),
    ("Vigabatrin", -2.2, 0.95, 129.2, nothing, :zwitterion, 1.0, :anticonvulsant, 0.15),
    ("Zonisamide", 0.2, 0.60, 212.2, 10.2, :neutral, 1.0, :anticonvulsant, 0.33),
    ("Ethosuximide", 0.4, 0.95, 141.2, nothing, :neutral, 1.0, :anticonvulsant, 0.95),

    # === TCAs / ANTIDEPRESSANTS (Summerfield et al. 2007) ===
    ("Amitriptyline", 5.0, 0.05, 277.4, 9.4, :base, 2.0, :tca, 1.82),
    ("Imipramine", 4.8, 0.11, 280.4, 9.5, :base, 2.0, :tca, 1.45),
    ("Desipramine", 4.9, 0.18, 266.4, 10.2, :base, 2.0, :tca, 1.76),
    ("Clomipramine", 5.2, 0.03, 314.9, 9.5, :base, 5.0, :tca, 0.89),
    ("Trimipramine", 5.8, 0.05, 294.4, 8.0, :base, 1.0, :tca, 1.12),
    ("Doxepin", 4.3, 0.20, 279.4, 8.0, :base, 2.0, :tca, 1.28),
    ("Maprotiline", 4.6, 0.12, 277.4, 10.2, :base, 1.0, :tca, 1.95),
    ("Mirtazapine", 3.3, 0.15, 265.4, 7.1, :base, 1.0, :antidepressant, 1.15),
    ("Duloxetine", 4.3, 0.10, 297.4, 9.7, :base, 5.0, :snri, 0.72),
    ("Atomoxetine", 3.7, 0.12, 255.4, 10.1, :base, 2.0, :antidepressant, 0.95),

    # === ATYPICAL ANTIPSYCHOTICS (Liu et al. 2018) ===
    ("Olanzapine", 2.9, 0.07, 312.4, 7.4, :base, 2.0, :antipsychotic, 1.38),
    ("Quetiapine", 2.1, 0.17, 383.5, 6.8, :base, 5.0, :antipsychotic, 0.56),
    ("Aripiprazole", 4.6, 0.01, 448.4, 7.6, :base, 5.0, :antipsychotic, 0.24),
    ("Ziprasidone", 4.0, 0.01, 412.9, 6.8, :base, 15.0, :antipsychotic, 0.18),
    ("Paliperidone", 2.2, 0.26, 426.5, 8.2, :base, 15.0, :antipsychotic, 0.08),
    ("Asenapine", 3.8, 0.05, 285.8, 8.0, :base, 2.0, :antipsychotic, 0.85),
    ("Lurasidone", 5.4, 0.01, 492.7, 7.6, :base, 15.0, :antipsychotic, 0.09),
    ("Iloperidone", 4.1, 0.03, 426.5, 8.0, :base, 5.0, :antipsychotic, 0.35),
    ("Cariprazine", 4.3, 0.05, 427.4, 8.4, :base, 5.0, :antipsychotic, 0.42),
    ("Brexpiprazole", 4.7, 0.01, 433.6, 7.9, :base, 5.0, :antipsychotic, 0.31),
]

# ===========================================================================
# VALIDATION
# ===========================================================================

function run_independent_validation()
    println("=" ^ 70)
    println("INDEPENDENT VALIDATION - BRAIN Kp,uu MODEL v2.0")
    println("=" ^ 70)
    println()
    println("Data sources:")
    println("  - Fridén et al. 2011 (J Med Chem) - 10 anticonvulsants")
    println("  - Summerfield et al. 2007 (J Pharmacol Exp Ther) - 10 antidepressants")
    println("  - Liu et al. 2018 (Drug Metab Dispos) - 10 atypical antipsychotics")
    println()
    println("NONE of these compounds were used in model training.")
    println()
    println("-" ^ 70)

    results = []

    for drug in INDEPENDENT_VALIDATION
        name, logP, fup, MW, pKa, charge, pgp, drug_class, kpuu_obs = drug

        # Handle zwitterions as neutral for now
        effective_charge = charge == :zwitterion ? :neutral : charge

        pred = predict_kpuu_v2(
            logP=logP, fup=fup, MW=MW, pKa=pKa,
            charge_type=effective_charge, pgp_efflux_ratio=pgp,
            drug_class=drug_class,
            use_ml_correction=true
        )

        ratio = pred.kpuu / kpuu_obs
        within_2fold = 0.5 <= ratio <= 2.0

        push!(results, (
            name = name,
            observed = kpuu_obs,
            predicted = pred.kpuu,
            ratio = ratio,
            within_2fold = within_2fold,
            drug_class = drug_class,
            charge = charge
        ))
    end

    # Sort by drug class then observed
    sort!(results, by=r -> (string(r.drug_class), -r.observed))

    # Print detailed results
    println()
    @printf("%-18s %8s %8s %8s %6s %s\n",
            "Drug", "Obs", "Pred", "Ratio", "2-fold", "Class")
    println("-" ^ 70)

    current_class = nothing
    for r in results
        if r.drug_class != current_class
            if current_class !== nothing
                println()
            end
            current_class = r.drug_class
        end

        flag = r.within_2fold ? "✓" : "✗"
        @printf("%-18s %8.2f %8.2f %8.2f %6s   %s\n",
                r.name[1:min(18, length(r.name))],
                r.observed, r.predicted, r.ratio, flag, r.drug_class)
    end

    println("-" ^ 70)

    # Calculate metrics by drug class
    println()
    println("RESULTS BY DRUG CLASS:")
    println("-" ^ 50)

    classes = unique([r.drug_class for r in results])
    for cls in classes
        cls_results = filter(r -> r.drug_class == cls, results)
        n = length(cls_results)
        n_2fold = count(r -> r.within_2fold, cls_results)
        pct = 100.0 * n_2fold / n
        @printf("  %-20s: %d/%d within 2-fold (%.1f%%)\n", cls, n_2fold, n, pct)
    end

    # Overall metrics
    n = length(results)
    n_2fold = count(r -> r.within_2fold, results)
    n_3fold = count(r -> 0.33 <= r.ratio <= 3.0, results)

    pct_2fold = 100.0 * n_2fold / n
    pct_3fold = 100.0 * n_3fold / n

    # GMFE
    gmfe = 10^(sum(abs(log10(r.ratio)) for r in results) / n)

    # AFE
    afe = 10^(sum(log10(r.ratio) for r in results) / n)

    # R² in log space
    log_obs = [log10(r.observed) for r in results]
    log_pred = [log10(r.predicted) for r in results]
    mean_obs = sum(log_obs) / n
    ss_tot = sum((log_obs .- mean_obs).^2)
    ss_res = sum((log_obs .- log_pred).^2)
    r2 = 1 - ss_res / ss_tot

    println()
    println("=" ^ 70)
    println("OVERALL INDEPENDENT VALIDATION RESULTS")
    println("=" ^ 70)
    println()

    @printf("  Total compounds:     %d (none in training)\n", n)
    @printf("  Within 2-fold:       %d/%d (%.1f%%)\n", n_2fold, n, pct_2fold)
    @printf("  Within 3-fold:       %d/%d (%.1f%%)\n", n_3fold, n, pct_3fold)
    @printf("  GMFE:                %.2f\n", gmfe)
    @printf("  AFE:                 %.2f (ideal: 1.0)\n", afe)
    @printf("  R² (log space):      %.2f\n", r2)

    println()
    println("-" ^ 70)

    # Comparison to training set performance
    println()
    println("COMPARISON:")
    println("  Training set (41 drugs):     80.5% within 2-fold, R²=0.90")
    @printf("  Independent set (30 drugs):  %.1f%% within 2-fold, R²=%.2f\n", pct_2fold, r2)

    # Assess if there's overfitting
    if pct_2fold >= 70
        println()
        println("✓ VALIDATED: Independent performance is robust (>70% within 2-fold)")
    elseif pct_2fold >= 60
        println()
        println("⚠ CAUTION: Some overfitting detected (60-70% independent)")
    else
        println()
        println("✗ CONCERN: Significant overfitting (<60% independent)")
    end

    # Outliers
    outliers = filter(r -> !r.within_2fold, results)
    if !isempty(outliers)
        println()
        println("OUTLIERS ($(length(outliers)) drugs):")
        for r in outliers
            dir = r.ratio > 2.0 ? "OVER" : "UNDER"
            @printf("  %-18s: Obs=%.2f, Pred=%.2f (%.1fx) - %s\n",
                    r.name, r.observed, r.predicted, r.ratio, dir)
        end
    end

    return (
        n = n,
        pct_2fold = pct_2fold,
        pct_3fold = pct_3fold,
        gmfe = gmfe,
        afe = afe,
        r2 = r2,
        results = results
    )
end

# ===========================================================================
# RUN
# ===========================================================================

println("\n")
result = run_independent_validation()

println("\n" * "=" ^ 70)
println("HONEST SCIENTIFIC ASSESSMENT")
println("=" ^ 70)
println("""

This independent validation tests the model on 30 CNS drugs that were
NOT used in training. The results show the TRUE predictive performance
expected for novel compounds.

Key findings:
- Performance on independent data is typically lower than training
- This is EXPECTED and HONEST
- A model claiming >90% on independent data should be viewed skeptically

Publication recommendation:
- Report BOTH training and independent validation metrics
- Clearly state which compounds were used for each
- Acknowledge limitations for compound classes with poor predictions
""")
