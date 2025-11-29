# LIVER COMPARTMENT MODEL VALIDATION
# ===================================
#
# Tests the enhanced liver Kp model against literature values
#
# Key improvements validated:
# 1. Lysosomal trapping (liver has 2.5% lysosomes, 5x muscle)
# 2. Effective K_tissue for basic drug binding
# 3. Transporter effects (OATPs, OCT1, P-gp)

using Printf

# Include the liver module
include("../../src/DarwinPBPK/compartments/liver.jl")
using .LiverCompartment

# Test drugs with literature Kp values
const VALIDATION_DRUGS = [
    # Drug Name | logP | pKa | fup | is_base | is_acid | is_oatp | is_oct | is_pgp | Observed Kp | Notes

    # Beta-blockers (basic drugs)
    ("Propranolol", 3.5, 9.5, 0.10, true, false, false, false, false, 4.0, "Beta-blocker, lysosomal trapping"),
    ("Metoprolol", 1.9, 9.7, 0.88, true, false, false, false, false, 2.5, "Hydrophilic beta-blocker"),
    ("Atenolol", -0.2, 9.6, 0.94, true, false, false, false, false, 0.8, "Very hydrophilic base"),

    # Tricyclic antidepressants (lipophilic bases)
    ("Imipramine", 4.8, 9.4, 0.10, true, false, false, false, false, 10.0, "TCA, strong lysosomal"),
    ("Amitriptyline", 4.9, 9.4, 0.05, true, false, false, false, false, 12.0, "TCA, very lipophilic"),

    # Antimalarials (basic, massive lysosomal trapping)
    ("Chloroquine", 4.6, 10.1, 0.40, true, false, false, false, false, 100.0, "Extreme lysosomal accumulation"),

    # Statins (OATP substrates - acidic drugs)
    ("Atorvastatin", 4.5, 4.5, 0.02, false, true, true, false, false, 25.0, "OATP1B1/1B3 substrate"),
    ("Rosuvastatin", -0.3, 4.6, 0.12, false, true, true, false, false, 10.0, "Hydrophilic, OATP + BCRP"),
    ("Pravastatin", -0.8, 4.2, 0.50, false, true, true, false, false, 15.0, "Very hydrophilic statin"),

    # OCT1 substrates
    ("Metformin", -1.5, 11.5, 0.99, true, false, false, true, false, 5.0, "OCT1 substrate"),

    # Neutral/weakly acidic drugs
    ("Diazepam", 2.8, 3.4, 0.02, false, false, false, false, false, 3.5, "Lipophilic neutral"),
    ("Warfarin", 2.6, 5.1, 0.01, false, true, false, false, false, 0.6, "Acidic, albumin bound"),

    # P-gp substrates
    ("Digoxin", 1.3, nothing, 0.75, false, false, false, false, true, 0.5, "P-gp efflux reduces Kp"),
    ("Quinidine", 3.4, 8.5, 0.13, true, false, false, false, true, 3.0, "Base + P-gp substrate"),
]

function validate_liver_model()
    println("=" ^ 80)
    println("LIVER COMPARTMENT MODEL VALIDATION")
    println("=" ^ 80)
    println()

    results = []

    for (name, logP, pKa, fup, is_base, is_acid, is_oatp, is_oct, is_pgp, observed_kp, notes) in VALIDATION_DRUGS
        # Calculate predicted Kp
        predicted_kp = calculate_kp_liver(
            logP=logP,
            fup=fup,
            pKa=pKa,
            is_base=is_base,
            is_acid=is_acid,
            is_oatp_substrate=is_oatp,
            is_oct_substrate=is_oct,
            is_pgp_substrate=is_pgp
        )

        # Calculate fold error
        fold_error = max(predicted_kp / observed_kp, observed_kp / predicted_kp)

        push!(results, (name, logP, pKa, fup, is_base, predicted_kp, observed_kp, fold_error, notes))
    end

    # Print results table
    println("Drug             | logP  | pKa   | fup  | Type  | Pred Kp | Obs Kp | Fold | Status  | Notes")
    println("-" ^ 110)

    for (name, logP, pKa, fup, is_base, pred, obs, fold, notes) in results
        drug_type = is_base ? "Base" : "Acid/N"
        pka_str = isnothing(pKa) ? "N/A  " : @sprintf("%.1f  ", pKa)

        status = if fold <= 2.0
            "PASS"
        elseif fold <= 3.0
            "OK"
        else
            "CHECK"
        end

        status_color = if fold <= 2.0
            "GREEN"
        elseif fold <= 3.0
            "YELLOW"
        else
            "RED"
        end

        @printf("%-16s | %5.1f | %s | %.2f | %-5s | %7.2f | %6.1f | %.2f | %-7s | %s\n",
                name, logP, pka_str, fup, drug_type, pred, obs, fold, status, notes[1:min(end, 30)])
    end

    # Calculate summary statistics
    fold_errors = [r[8] for r in results]
    gmfe = exp(sum(log.(fold_errors)) / length(fold_errors))
    within_2fold = count(x -> x <= 2.0, fold_errors) / length(fold_errors) * 100
    within_3fold = count(x -> x <= 3.0, fold_errors) / length(fold_errors) * 100

    println()
    println("=" ^ 80)
    println("SUMMARY STATISTICS")
    println("=" ^ 80)
    @printf("GMFE:              %.2f\n", gmfe)
    @printf("Within 2-fold:     %.0f%% (%d/%d)\n", within_2fold, count(x -> x <= 2.0, fold_errors), length(fold_errors))
    @printf("Within 3-fold:     %.0f%% (%d/%d)\n", within_3fold, count(x -> x <= 3.0, fold_errors), length(fold_errors))

    # Categorized analysis
    println()
    println("=" ^ 80)
    println("CATEGORIZED ANALYSIS")
    println("=" ^ 80)

    # Beta-blockers
    bb_results = filter(r -> occursin("beta", lowercase(r[9])), results)
    if !isempty(bb_results)
        bb_gmfe = exp(sum(log.([r[8] for r in bb_results])) / length(bb_results))
        @printf("\nBeta-blockers (n=%d): GMFE = %.2f\n", length(bb_results), bb_gmfe)
        for r in bb_results
            @printf("  %-16s: Pred %.2f vs Obs %.1f (%.2fx)\n", r[1], r[6], r[7], r[8])
        end
    end

    # Tricyclics
    tca_results = filter(r -> occursin("TCA", r[9]), results)
    if !isempty(tca_results)
        tca_gmfe = exp(sum(log.([r[8] for r in tca_results])) / length(tca_results))
        @printf("\nTricyclic Antidepressants (n=%d): GMFE = %.2f\n", length(tca_results), tca_gmfe)
        for r in tca_results
            @printf("  %-16s: Pred %.2f vs Obs %.1f (%.2fx)\n", r[1], r[6], r[7], r[8])
        end
    end

    # OATP substrates (statins)
    oatp_results = filter(r -> occursin("OATP", r[9]), results)
    if !isempty(oatp_results)
        oatp_gmfe = exp(sum(log.([r[8] for r in oatp_results])) / length(oatp_results))
        @printf("\nOATP Substrates (n=%d): GMFE = %.2f\n", length(oatp_results), oatp_gmfe)
        for r in oatp_results
            @printf("  %-16s: Pred %.2f vs Obs %.1f (%.2fx)\n", r[1], r[6], r[7], r[8])
        end
    end

    # Lysosomal trapping cases
    lyso_results = filter(r -> occursin("lysosom", lowercase(r[9])), results)
    if !isempty(lyso_results)
        lyso_gmfe = exp(sum(log.([r[8] for r in lyso_results])) / length(lyso_results))
        @printf("\nLysosomal Trapping Cases (n=%d): GMFE = %.2f\n", length(lyso_results), lyso_gmfe)
        for r in lyso_results
            @printf("  %-16s: Pred %.2f vs Obs %.1f (%.2fx)\n", r[1], r[6], r[7], r[8])
        end
    end

    println()
    println("=" ^ 80)
    println("INDIVIDUAL MECHANISM TESTS")
    println("=" ^ 80)

    # Test lysosomal trapping calculation
    println("\n--- Lysosomal Trapping (liver 2.5% vs muscle 0.5%) ---")
    for (name, pka, logP) in [("Propranolol", 9.5, 3.5), ("Imipramine", 9.4, 4.8), ("Chloroquine", 10.1, 4.6)]
        lyso = calculate_lysosomal_trapping_liver(pKa=pka, logP=logP)
        @printf("%-16s: pKa=%.1f, logP=%.1f → Lysosomal contribution = %.3f\n", name, pka, logP, lyso)
    end

    # Test effective K_tissue
    println("\n--- Effective K_tissue (liver 3x more APL than muscle) ---")
    for logP in [0.0, 1.5, 2.5, 3.5, 4.5, 5.5]
        kt = calculate_effective_K_tissue_liver(logP)
        @printf("logP = %.1f → K_tissue_liver = %.2f\n", logP, kt)
    end

    # Test transporter effects
    println("\n--- Transporter Effect Multipliers ---")
    println("OATP substrate only:     $(estimate_transporter_effect(is_oatp_substrate=true))x")
    println("OCT1 substrate only:     $(estimate_transporter_effect(is_oct_substrate=true))x")
    println("P-gp substrate only:     $(estimate_transporter_effect(is_pgp_substrate=true))x")
    println("OATP + P-gp:             $(estimate_transporter_effect(is_oatp_substrate=true, is_pgp_substrate=true))x")

    # Test first-pass calculations
    println("\n--- First-Pass Bioavailability (Well-Stirred Model) ---")
    for (name, fub, CLint) in [("High E (Propranolol)", 0.10, 5.0),
                               ("Med E (Codeine)", 0.70, 0.5),
                               ("Low E (Diazepam)", 0.02, 0.1)]
        result = calculate_first_pass_bioavailability(fub=fub, CLint_L_min=CLint)
        @printf("%-20s: E = %.2f, F_hepatic = %.1f%%\n", name, result.extraction_ratio, result.F_hepatic * 100)
    end

    return (gmfe=gmfe, within_2fold=within_2fold, within_3fold=within_3fold, results=results)
end

# Run validation
validate_liver_model()
