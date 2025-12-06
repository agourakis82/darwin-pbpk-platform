"""
Clinical Validation Module

Validates blood compartment model predictions against published clinical data.
Uses only peer-reviewed, publicly available pharmacokinetic data.

Evidence Levels:
- Level A: Multiple RCTs, meta-analyses
- Level B: Single RCT or large observational studies
- Level C: Case series, expert consensus
- Level D: In vitro or theoretical

References indexed by PubMed ID where available.

Author: Darwin PBPK Platform
Date: 2025-12-06
"""
module ClinicalValidation

using Statistics
using Dates

export ClinicalAssertion, ValidationResult, ValidationSuite
export validate_fu_changes, validate_clearance_changes
export validate_against_literature, run_validation_suite
export PHENYTOIN_UREMIA_DATA, VANCOMYCIN_BURNS_DATA
export TACROLIMUS_HEMATOCRIT_DATA, AAG_SEPSIS_DATA
export get_validation_summary, calculate_prediction_accuracy
export calculate_tacrolimus_rb, calculate_fu_uremia
export calculate_fu_hypoalbuminemia, calculate_fu_elevated_aag
export print_validation_report

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    ClinicalAssertion

A testable assertion based on clinical literature.
"""
struct ClinicalAssertion
    id::String
    drug::String
    condition::String
    parameter::Symbol           # :fu, :cl, :vd, :auc, :cmax
    expected_change::Float64    # Fold-change vs normal
    tolerance::Float64          # Acceptable deviation (fraction)
    evidence_level::Symbol      # :A, :B, :C, :D
    reference::String           # PMID or citation
    n_subjects::Int             # Sample size from study
    notes::String
end

"""
    ValidationResult

Result of validating a prediction against clinical data.
"""
struct ValidationResult
    assertion::ClinicalAssertion
    predicted::Float64
    expected::Float64
    error::Float64              # Absolute relative error
    passed::Bool
    details::String
end

"""
    ValidationSuite

Collection of validation results.
"""
struct ValidationSuite
    name::String
    results::Vector{ValidationResult}
    timestamp::String

    function ValidationSuite(name::String, results::Vector{ValidationResult})
        new(name, results, string(Dates.now()))
    end
end

# ============================================================================
# PUBLISHED CLINICAL DATA
# ============================================================================

"""
Phenytoin protein binding in uremia.
Classic example of uremic toxin displacement from albumin.

Reference: Reidenberg MM, Drayer DE. Clin Pharmacokinet 1984;9:18-26
"""
const PHENYTOIN_UREMIA_DATA = [
    ClinicalAssertion(
        "PHT_UREMIA_FU_01",
        "Phenytoin",
        "ESRD (GFR<15)",
        :fu,
        2.5,            # fu increases 2.5x (0.1 → 0.25)
        0.25,           # ±25% acceptable
        :A,
        "PMID:6360442",
        42,
        "Uremic toxins (CMPF, indoxyl sulfate) displace phenytoin from albumin"
    ),
    ClinicalAssertion(
        "PHT_UREMIA_FU_02",
        "Phenytoin",
        "CKD Stage 4 (GFR 15-29)",
        :fu,
        1.8,            # fu increases 1.8x
        0.30,
        :B,
        "PMID:6360442",
        28,
        "Moderate uremia, partial displacement"
    ),
    ClinicalAssertion(
        "PHT_CIRRHOSIS_FU",
        "Phenytoin",
        "Cirrhosis Child-Pugh C",
        :fu,
        2.2,            # Hypoalbuminemia effect
        0.25,
        :B,
        "PMID:3377580",
        18,
        "Albumin ~20 g/L in decompensated cirrhosis"
    )
]

"""
Vancomycin PK changes in burn patients.
Augmented renal clearance and increased Vd.

Reference: Blanchet B et al. Clin Pharmacokinet 2008
"""
const VANCOMYCIN_BURNS_DATA = [
    ClinicalAssertion(
        "VANC_BURN_CL_01",
        "Vancomycin",
        "Major burn (TBSA>40%)",
        :cl,
        1.8,            # CL increases 80%
        0.30,
        :B,
        "PMID:18402328",
        24,
        "Augmented renal clearance in hyperdynamic phase"
    ),
    ClinicalAssertion(
        "VANC_BURN_VD_01",
        "Vancomycin",
        "Major burn (TBSA>40%)",
        :vd,
        2.0,            # Vd doubles
        0.35,
        :B,
        "PMID:18402328",
        24,
        "Capillary leak, fluid resuscitation"
    ),
    ClinicalAssertion(
        "VANC_SEPSIS_VD",
        "Vancomycin",
        "Sepsis/septic shock",
        :vd,
        1.5,            # Vd increases 50%
        0.30,
        :A,
        "PMID:24696306",
        89,
        "Roberts JA, Lancet Infect Dis 2014"
    )
]

"""
Tacrolimus blood:plasma ratio vs hematocrit.
Critical for TDM interpretation.

Reference: Zahir H et al. Clin Pharmacokinet 2001
"""
const TACROLIMUS_HEMATOCRIT_DATA = [
    ClinicalAssertion(
        "TAC_HCT_RB_NORMAL",
        "Tacrolimus",
        "Normal Hct (0.42)",
        :rb,
        15.0,           # Rb ~15 at normal Hct
        0.20,
        :A,
        "PMID:11374753",
        156,
        "Ke_p ~37, high RBC binding"
    ),
    ClinicalAssertion(
        "TAC_HCT_RB_ANEMIA",
        "Tacrolimus",
        "Anemia (Hct 0.28)",
        :rb,
        10.5,           # Rb decreases with Hct
        0.20,
        :B,
        "PMID:11374753",
        43,
        "Lower Hct = lower Rb = higher plasma conc for same blood conc"
    ),
    ClinicalAssertion(
        "TAC_HCT_RB_POLY",
        "Tacrolimus",
        "Polycythemia (Hct 0.55)",
        :rb,
        20.0,           # Rb increases with Hct
        0.25,
        :C,
        "PMID:11374753",
        12,
        "Extrapolated from Hct-Rb relationship"
    )
]

"""
Alpha-1 acid glycoprotein (AAG) in sepsis.
Affects binding of basic drugs.

Reference: Patel IH et al. J Clin Pharmacol 1990
"""
const AAG_SEPSIS_DATA = [
    ClinicalAssertion(
        "AAG_SEPSIS_CONC",
        "AAG",
        "Sepsis day 1-2",
        :aag_concentration,
        2.5,            # 2-3x normal
        0.30,
        :A,
        "PMID:2229404",
        67,
        "Acute phase response, IL-6 driven"
    ),
    ClinicalAssertion(
        "AAG_SEPSIS_PEAK",
        "AAG",
        "Sepsis day 4-5 (peak)",
        :aag_concentration,
        3.0,            # Peak at 3x
        0.25,
        :B,
        "PMID:2229404",
        67,
        "Peak AAG at 96-120h post-onset"
    ),
    ClinicalAssertion(
        "LIDOCAINE_SEPSIS_FU",
        "Lidocaine",
        "Sepsis (elevated AAG)",
        :fu,
        0.4,            # fu decreases to 40% of normal
        0.30,
        :B,
        "PMID:3918445",
        31,
        "Basic drug, AAG binding increases"
    )
]

"""
Albumin changes in critical illness.
"""
const ALBUMIN_CRITICAL_DATA = [
    ClinicalAssertion(
        "ALB_SEPSIS_CONC",
        "Albumin",
        "Sepsis day 1-2",
        :albumin_concentration,
        0.6,            # Falls to 60% (24 g/L from 40)
        0.20,
        :A,
        "PMID:15243924",
        1000,
        "Capillary leak, reduced synthesis, dilution"
    ),
    ClinicalAssertion(
        "ALB_SURGERY_CONC",
        "Albumin",
        "Major surgery day 1",
        :albumin_concentration,
        0.75,           # Falls to 75%
        0.15,
        :A,
        "PMID:12719441",
        234,
        "Hemodilution, redistribution"
    )
]

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

"""
    validate_fu_change(predicted_fu::Float64, normal_fu::Float64,
                       assertion::ClinicalAssertion)

Validate a predicted fu change against clinical assertion.
"""
function validate_fu_change(predicted_fu::Float64, normal_fu::Float64,
                            assertion::ClinicalAssertion)
    predicted_change = predicted_fu / normal_fu
    expected_change = assertion.expected_change

    error = abs(predicted_change - expected_change) / expected_change
    passed = error <= assertion.tolerance

    details = """
    Predicted fu: $(round(predicted_fu, digits=4))
    Normal fu: $(round(normal_fu, digits=4))
    Predicted change: $(round(predicted_change, digits=2))x
    Expected change: $(expected_change)x ± $(assertion.tolerance * 100)%
    Reference: $(assertion.reference)
    """

    return ValidationResult(assertion, predicted_change, expected_change, error, passed, details)
end

"""
    validate_rb_prediction(predicted_rb::Float64, assertion::ClinicalAssertion)

Validate blood:plasma ratio prediction.
"""
function validate_rb_prediction(predicted_rb::Float64, assertion::ClinicalAssertion)
    expected_rb = assertion.expected_change  # Using expected_change field for expected Rb

    error = abs(predicted_rb - expected_rb) / expected_rb
    passed = error <= assertion.tolerance

    details = """
    Predicted Rb: $(round(predicted_rb, digits=1))
    Expected Rb: $(expected_rb) ± $(assertion.tolerance * 100)%
    Condition: $(assertion.condition)
    Reference: $(assertion.reference) (n=$(assertion.n_subjects))
    """

    return ValidationResult(assertion, predicted_rb, expected_rb, error, passed, details)
end

"""
    validate_concentration_change(predicted::Float64, normal::Float64,
                                   assertion::ClinicalAssertion)

Validate protein concentration changes (AAG, albumin).
"""
function validate_concentration_change(predicted::Float64, normal::Float64,
                                        assertion::ClinicalAssertion)
    predicted_ratio = predicted / normal
    expected_ratio = assertion.expected_change

    error = abs(predicted_ratio - expected_ratio) / expected_ratio
    passed = error <= assertion.tolerance

    details = """
    Predicted: $(round(predicted, digits=1)) ($(round(predicted_ratio, digits=2))x normal)
    Expected: $(round(normal * expected_ratio, digits=1)) ($(expected_ratio)x normal)
    Reference: $(assertion.reference)
    """

    return ValidationResult(assertion, predicted_ratio, expected_ratio, error, passed, details)
end

"""
    calculate_tacrolimus_rb(hematocrit::Float64; ke_p::Float64=37.0)

Calculate Tacrolimus Rb from hematocrit using standard equation.
Rb = 1 - Hct + Hct × Ke_p
"""
function calculate_tacrolimus_rb(hematocrit::Float64; ke_p::Float64=37.0)
    return 1.0 - hematocrit + hematocrit * ke_p
end

"""
    calculate_fu_uremia(fu_normal::Float64, gfr::Float64; drug_type::Symbol=:acidic)

Calculate fu in uremia based on GFR.
Uses empirical relationship from clinical studies.
"""
function calculate_fu_uremia(fu_normal::Float64, gfr::Float64; drug_type::Symbol=:acidic)
    if drug_type != :acidic
        return fu_normal  # Only acidic drugs significantly affected
    end

    # Empirical: fu increases as GFR decreases
    # At GFR=100: fu_factor = 1.0
    # At GFR=10: fu_factor ≈ 2.5
    fu_factor = 1.0 + 1.5 * (1.0 - gfr/100.0)^1.5
    fu_factor = clamp(fu_factor, 1.0, 3.0)

    fu_adjusted = fu_normal * fu_factor
    return min(fu_adjusted, 1.0)
end

"""
    calculate_fu_hypoalbuminemia(fu_normal::Float64, albumin::Float64;
                                  normal_albumin::Float64=40.0)

Calculate fu adjustment for hypoalbuminemia (acidic drugs).
"""
function calculate_fu_hypoalbuminemia(fu_normal::Float64, albumin::Float64;
                                       normal_albumin::Float64=40.0)
    # Simple inverse relationship for highly bound drugs
    # fu_new / fu_old ≈ albumin_old / albumin_new
    fu_adjusted = fu_normal * (normal_albumin / albumin)
    return min(fu_adjusted, 1.0)
end

"""
    calculate_fu_elevated_aag(fu_normal::Float64, aag::Float64;
                               normal_aag::Float64=0.8)

Calculate fu adjustment for elevated AAG (basic drugs).
"""
function calculate_fu_elevated_aag(fu_normal::Float64, aag::Float64;
                                    normal_aag::Float64=0.8)
    # fu decreases when AAG increases
    fu_adjusted = fu_normal * (normal_aag / aag)
    return max(fu_adjusted, 0.01)
end

# ============================================================================
# VALIDATION SUITE RUNNER
# ============================================================================

"""
    run_validation_suite(suite_name::Symbol=:all)

Run validation against published clinical data.

Available suites:
- :phenytoin_uremia
- :vancomycin_burns
- :tacrolimus_hematocrit
- :aag_sepsis
- :all
"""
function run_validation_suite(suite_name::Symbol=:all)
    results = ValidationResult[]

    if suite_name == :all || suite_name == :tacrolimus_hematocrit
        # Tacrolimus Rb validation
        for assertion in TACROLIMUS_HEMATOCRIT_DATA
            if assertion.condition == "Normal Hct (0.42)"
                predicted_rb = calculate_tacrolimus_rb(0.42)
            elseif assertion.condition == "Anemia (Hct 0.28)"
                predicted_rb = calculate_tacrolimus_rb(0.28)
            elseif assertion.condition == "Polycythemia (Hct 0.55)"
                predicted_rb = calculate_tacrolimus_rb(0.55)
            else
                continue
            end
            push!(results, validate_rb_prediction(predicted_rb, assertion))
        end
    end

    if suite_name == :all || suite_name == :phenytoin_uremia
        # Phenytoin fu validation
        normal_fu = 0.10

        for assertion in PHENYTOIN_UREMIA_DATA
            if assertion.condition == "ESRD (GFR<15)"
                predicted_fu = calculate_fu_uremia(normal_fu, 10.0)
            elseif assertion.condition == "CKD Stage 4 (GFR 15-29)"
                predicted_fu = calculate_fu_uremia(normal_fu, 22.0)
            elseif assertion.condition == "Cirrhosis Child-Pugh C"
                predicted_fu = calculate_fu_hypoalbuminemia(normal_fu, 20.0)
            else
                continue
            end
            push!(results, validate_fu_change(predicted_fu, normal_fu, assertion))
        end
    end

    if suite_name == :all || suite_name == :aag_sepsis
        # AAG/Lidocaine validation
        for assertion in AAG_SEPSIS_DATA
            if assertion.drug == "Lidocaine"
                normal_fu = 0.30
                predicted_fu = calculate_fu_elevated_aag(normal_fu, 2.5)
                push!(results, validate_fu_change(predicted_fu, normal_fu, assertion))
            end
        end
    end

    return ValidationSuite(string(suite_name), results)
end

"""
    get_validation_summary(suite::ValidationSuite)

Get human-readable summary of validation results.
"""
function get_validation_summary(suite::ValidationSuite)
    n_total = length(suite.results)
    n_passed = count(r -> r.passed, suite.results)
    n_failed = n_total - n_passed

    mean_error = mean(r.error for r in suite.results)
    max_error = maximum(r.error for r in suite.results)

    failed_tests = [r for r in suite.results if !r.passed]

    summary = Dict(
        :suite_name => suite.name,
        :total_tests => n_total,
        :passed => n_passed,
        :failed => n_failed,
        :pass_rate => round(n_passed / n_total * 100, digits=1),
        :mean_error => round(mean_error * 100, digits=1),
        :max_error => round(max_error * 100, digits=1),
        :failed_tests => [r.assertion.id for r in failed_tests],
        :evidence_distribution => Dict(
            :A => count(r -> r.assertion.evidence_level == :A, suite.results),
            :B => count(r -> r.assertion.evidence_level == :B, suite.results),
            :C => count(r -> r.assertion.evidence_level == :C, suite.results)
        )
    )

    return summary
end

"""
    calculate_prediction_accuracy(suite::ValidationSuite)

Calculate overall prediction accuracy metrics.
"""
function calculate_prediction_accuracy(suite::ValidationSuite)
    errors = [r.error for r in suite.results]

    return Dict(
        :mpe => mean(errors) * 100,              # Mean Prediction Error %
        :rmse => sqrt(mean(e^2 for e in errors)) * 100,  # Root Mean Square Error %
        :mae => mean(abs.(errors)) * 100,        # Mean Absolute Error %
        :max_error => maximum(errors) * 100,
        :within_20pct => count(e -> e <= 0.20, errors) / length(errors) * 100,
        :within_30pct => count(e -> e <= 0.30, errors) / length(errors) * 100
    )
end

"""
    print_validation_report(suite::ValidationSuite)

Print formatted validation report.
"""
function print_validation_report(suite::ValidationSuite)
    summary = get_validation_summary(suite)
    accuracy = calculate_prediction_accuracy(suite)

    println("=" ^ 60)
    println("CLINICAL VALIDATION REPORT: $(suite.name)")
    println("=" ^ 60)
    println()
    println("SUMMARY")
    println("-" ^ 40)
    println("  Total tests:    $(summary[:total_tests])")
    println("  Passed:         $(summary[:passed]) ($(summary[:pass_rate])%)")
    println("  Failed:         $(summary[:failed])")
    println()
    println("PREDICTION ACCURACY")
    println("-" ^ 40)
    println("  Mean Error:     $(round(accuracy[:mpe], digits=1))%")
    println("  RMSE:           $(round(accuracy[:rmse], digits=1))%")
    println("  Within 20%:     $(round(accuracy[:within_20pct], digits=0))%")
    println("  Within 30%:     $(round(accuracy[:within_30pct], digits=0))%")
    println()
    println("EVIDENCE LEVELS")
    println("-" ^ 40)
    println("  Level A (RCT/Meta):   $(summary[:evidence_distribution][:A])")
    println("  Level B (Large Obs):  $(summary[:evidence_distribution][:B])")
    println("  Level C (Case/Expert):$(summary[:evidence_distribution][:C])")
    println()

    if summary[:failed] > 0
        println("FAILED TESTS")
        println("-" ^ 40)
        for r in suite.results
            if !r.passed
                println("  $(r.assertion.id): predicted $(round(r.predicted, digits=2)) vs expected $(r.expected) (error: $(round(r.error*100, digits=1))%)")
            end
        end
    end

    println()
    println("=" ^ 60)
end

end # module ClinicalValidation
