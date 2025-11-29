#!/usr/bin/env julia
# ===========================================================================
# BRAIN COMPARTMENT SOTA VALIDATION
# ===========================================================================
# Validates the enhanced brain model against literature data
# Tests all novel features from our Socratic discussion
# ===========================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Printf

# Include brain module
include(joinpath(@__DIR__, "..", "..", "src", "DarwinPBPK", "compartments", "brain.jl"))
using .BrainCompartment

println("=" ^ 80)
println("DARWIN PBPK PLATFORM - BRAIN COMPARTMENT SOTA VALIDATION")
println("=" ^ 80)
println()

# ===========================================================================
# TEST 1: BASELINE Kp,uu VALIDATION WITH LITERATURE DATA
# ===========================================================================

println("TEST 1: Baseline Kp,uu vs Literature")
println("-" ^ 60)

# Literature data for CNS drugs (from our discussion and documentation)
# Format: (name, logP, fup, MW, TPSA, HBD, is_base, pKa, is_pgp, Kp_lit, Kpuu_lit)
literature_drugs = [
    ("Diazepam", 2.8, 0.02, 285.0, 33.0, 0, false, nothing, false, 0.9, 0.8),
    ("Caffeine", -0.1, 0.65, 194.0, 58.0, 0, true, 0.6, false, 0.8, 1.0),
    ("Haloperidol", 4.3, 0.08, 376.0, 40.0, 1, true, 8.3, false, 15.0, 3.0),
    ("Risperidone", 3.0, 0.10, 410.0, 62.0, 0, true, 8.2, true, 10.0, 0.3),
    ("Loperamide", 4.8, 0.03, 477.0, 44.0, 1, true, 8.6, true, 0.05, 0.02),
    ("Morphine", 0.9, 0.65, 285.0, 52.0, 2, true, 8.0, true, 0.3, 0.4),
    ("Atenolol", -0.1, 0.95, 266.0, 85.0, 4, true, 9.6, false, 0.04, 0.1),
]

n_within_2fold = 0
n_within_3fold = 0
total = length(literature_drugs)

for drug in literature_drugs
    name, logP, fup, MW, TPSA, HBD, is_base, pKa, is_pgp, Kp_lit, Kpuu_lit = drug

    result = calculate_kpuu_brain(
        logP=logP, fup=fup, MW=MW, TPSA=TPSA, HBD=HBD,
        is_base=is_base, pKa=pKa, is_pgp_substrate=is_pgp
    )

    ratio = result.Kpuu / Kpuu_lit
    within_2fold = 0.5 <= ratio <= 2.0
    within_3fold = 0.33 <= ratio <= 3.0

    if within_2fold
        global n_within_2fold += 1
    end
    if within_3fold
        global n_within_3fold += 1
    end

    status = within_2fold ? "OK" : (within_3fold ? "~" : "X")

    @printf("  %-15s: Pred=%.2f  Lit=%.2f  Ratio=%.2f  [%s]\n",
            name, result.Kpuu, Kpuu_lit, ratio, status)
end

pct_2fold = 100.0 * n_within_2fold / total
pct_3fold = 100.0 * n_within_3fold / total

println()
@printf("  Within 2-fold: %d/%d (%.1f%%)\n", n_within_2fold, total, pct_2fold)
@printf("  Within 3-fold: %d/%d (%.1f%%)\n", n_within_3fold, total, pct_3fold)

test1_pass = pct_2fold >= 60.0  # Target: 60% within 2-fold for brain
println("  TEST 1: ", test1_pass ? "PASS" : "NEEDS IMPROVEMENT")
println()

# ===========================================================================
# TEST 2: CIRCADIAN P-gp VARIATION
# ===========================================================================

println("TEST 2: Circadian P-gp Variation")
println("-" ^ 60)

# Expected: ~2x variation between morning peak and night nadir
morning_pgp = calculate_circadian_pgp_activity(MORNING_PEAK)
night_pgp = calculate_circadian_pgp_activity(NIGHT_NADIR)
circadian_ratio = morning_pgp / night_pgp

@printf("  Morning P-gp activity: %.2f (100%%)\n", morning_pgp)
@printf("  Night P-gp activity:   %.2f (%.0f%%)\n", night_pgp, night_pgp * 100)
@printf("  Circadian variation:   %.1fx\n", circadian_ratio)

test2_pass = 1.8 <= circadian_ratio <= 2.5
println("  Expected: ~2x variation")
println("  TEST 2: ", test2_pass ? "PASS" : "FAIL")
println()

# Test all phases
println("  P-gp activity by phase:")
for phase in instances(CircadianPhase)
    activity = calculate_circadian_pgp_activity(phase)
    @printf("    %-15s: %.0f%%\n", String(Symbol(phase)), activity * 100)
end
println()

# ===========================================================================
# TEST 3: INFLAMMATION EFFECT ON P-gp
# ===========================================================================

println("TEST 3: Inflammation Effect on P-gp")
println("-" ^ 60)

# Test IL-6 effect (literature: up to 84% reduction)
il6_folds = [1.0, 5.0, 10.0, 50.0, 100.0]
println("  IL-6 effect (literature: up to 84% reduction):")

for il6 in il6_folds
    pgp_func = calculate_inflammation_pgp_effect(il6_fold=il6)
    @printf("    IL-6 %.0fx: P-gp function = %.0f%%\n", il6, pgp_func * 100)
end

# Check that 100x IL-6 gives severe reduction
pgp_100x = calculate_inflammation_pgp_effect(il6_fold=100.0)
test3_pass = pgp_100x <= 0.30  # Should be <30% function at extreme inflammation

println()
@printf("  At 100x IL-6: %.0f%% P-gp function\n", pgp_100x * 100)
println("  Expected: <30% (severe inflammation)")
println("  TEST 3: ", test3_pass ? "PASS" : "FAIL")
println()

# ===========================================================================
# TEST 4: MENINGITIS BBB STAGING
# ===========================================================================

println("TEST 4: Meningitis BBB Staging")
println("-" ^ 60)

println("  BBB permeability by meningitis stage:")
for stage in instances(MeningitisStage)
    result = calculate_meningitis_bbb_state(stage)
    @printf("    %-20s: TJ=%.0f%%, P-gp=%.0f%%, Perm=%.1fx\n",
            String(Symbol(stage)),
            result.tj_integrity * 100,
            result.pgp_function * 100,
            result.penetration_multiplier)
end

# Test TB fibrotic stage paradox
fibrotic = calculate_meningitis_bbb_state(STAGE_IV_FIBROTIC)
severe = calculate_meningitis_bbb_state(STAGE_III_SEVERE)

println()
println("  TB Fibrotic Stage Paradox:")
@printf("    Stage III (severe):  permeability = %.1fx\n", severe.penetration_multiplier)
@printf("    Stage IV (fibrotic): permeability = %.1fx\n", fibrotic.penetration_multiplier)

# Fibrotic should have LOWER permeability than severe
test4_pass = fibrotic.penetration_multiplier < severe.penetration_multiplier
println("  Expected: Stage IV < Stage III (paradox)")
println("  TEST 4: ", test4_pass ? "PASS" : "FAIL")
println()

# ===========================================================================
# TEST 5: CSF DRUG PENETRATION
# ===========================================================================

println("TEST 5: CSF Drug Penetration in Meningitis")
println("-" ^ 60)

drugs_to_test = [:vancomycin, :linezolid, :rifampicin, :isoniazid, :fluconazole]

println("  Drug penetration (Stage II, inflamed):")
for drug in drugs_to_test
    result = estimate_csf_penetration_meningitis(
        drug=drug, meningitis_stage=STAGE_II_ESTABLISHED
    )
    if !isnothing(result.ratio)
        @printf("    %-15s: %.0f%% CSF/Plasma\n", String(drug), result.ratio * 100)
    end
end

# Test dexamethasone effect on vancomycin
vanco_no_dexa = estimate_csf_penetration_meningitis(
    drug=:vancomycin, meningitis_stage=STAGE_II_ESTABLISHED, on_dexamethasone=false
)
vanco_with_dexa = estimate_csf_penetration_meningitis(
    drug=:vancomycin, meningitis_stage=STAGE_II_ESTABLISHED, on_dexamethasone=true
)

println()
println("  Dexamethasone effect on vancomycin:")
@printf("    Without dexa: %.0f%%\n", vanco_no_dexa.ratio * 100)
@printf("    With dexa:    %.0f%% (%.0f%% reduction)\n",
        vanco_with_dexa.ratio * 100,
        (1 - vanco_with_dexa.ratio / vanco_no_dexa.ratio) * 100)

# Literature: 29% reduction
expected_reduction = 0.29
actual_reduction = 1 - vanco_with_dexa.ratio / vanco_no_dexa.ratio
test5_pass = abs(actual_reduction - expected_reduction) < 0.05  # Within 5%

println("  Expected reduction: 29%")
println("  TEST 5: ", test5_pass ? "PASS" : "FAIL")
println()

# ===========================================================================
# TEST 6: PEDIATRIC BBB MATURATION
# ===========================================================================

println("TEST 6: Pediatric BBB Maturation")
println("-" ^ 60)

println("  BBB maturity by age group:")
for age in instances(AgeGroup)
    result = calculate_pediatric_bbb_maturity(age)
    @printf("    %-18s: Maturity=%.0f%%, P-gp=%.0f%%, Perm=%.1fx\n",
            String(Symbol(age)),
            result.bbb_maturity * 100,
            result.pgp_expression * 100,
            result.permeability_factor)
end

# Preterm should have highest permeability
preterm = calculate_pediatric_bbb_maturity(PRETERM_NEONATE)
adult = calculate_pediatric_bbb_maturity(ADULT)

println()
@printf("  Preterm vs Adult permeability: %.1fx\n", preterm.permeability_factor)

test6_pass = preterm.permeability_factor >= 2.0 && adult.permeability_factor == 1.0
println("  Expected: Preterm >= 2x adult")
println("  TEST 6: ", test6_pass ? "PASS" : "FAIL")
println()

# ===========================================================================
# TEST 7: COVID BBB EFFECTS
# ===========================================================================

println("TEST 7: COVID-19 BBB Effects")
println("-" ^ 60)

phases = [:acute, :post_acute, :long_covid, :recovered]

println("  BBB status by COVID phase:")
for phase in phases
    result = calculate_covid_bbb_dysfunction(phase=phase, had_severe_covid=true)
    @printf("    %-12s: TJ=%.0f%%, P-gp=%.0f%%, Perm=%.1fx\n",
            String(phase),
            result.tj_integrity * 100,
            result.pgp_function * 100,
            result.permeability_factor)
end

# Acute should have highest disruption
acute = calculate_covid_bbb_dysfunction(phase=:acute)
long_covid = calculate_covid_bbb_dysfunction(phase=:long_covid, had_severe_covid=true, has_brain_fog=true)

println()
@printf("  Acute COVID permeability:     %.1fx\n", acute.permeability_factor)
@printf("  Long COVID permeability:      %.1fx\n", long_covid.permeability_factor)

test7_pass = acute.permeability_factor >= 3.0 && long_covid.permeability_factor > 1.0
println("  Expected: Acute >= 3x, Long COVID > 1x")
println("  TEST 7: ", test7_pass ? "PASS" : "FAIL")
println()

# ===========================================================================
# TEST 8: GLYMPHATIC/SLEEP EFFECTS
# ===========================================================================

println("TEST 8: Glymphatic Clearance (Sleep)")
println("-" ^ 60)

sleep_qualities = [100.0, 80.0, 60.0, 40.0, 20.0]

println("  Clearance by sleep quality:")
for sq in sleep_qualities
    result = calculate_glymphatic_clearance_factor(
        sleep_quality=sq, hours_of_sleep=7.0
    )
    @printf("    Sleep quality %.0f%%: Clearance = %.0f%% [%s]\n",
            sq, result.clearance_factor * 100, result.accumulation_risk)
end

poor_sleep = calculate_glymphatic_clearance_factor(sleep_quality=30.0, hours_of_sleep=4.0)
good_sleep = calculate_glymphatic_clearance_factor(sleep_quality=90.0, hours_of_sleep=8.0)

println()
@printf("  Good sleep clearance:  %.0f%%\n", good_sleep.clearance_factor * 100)
@printf("  Poor sleep clearance:  %.0f%%\n", poor_sleep.clearance_factor * 100)

test8_pass = good_sleep.clearance_factor > poor_sleep.clearance_factor * 1.5
println("  Expected: Good >> Poor sleep")
println("  TEST 8: ", test8_pass ? "PASS" : "FAIL")
println()

# ===========================================================================
# TEST 9: WHITE/GREY MATTER DISTRIBUTION
# ===========================================================================

println("TEST 9: White/Grey Matter Distribution")
println("-" ^ 60)

logP_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

println("  Equilibration time by lipophilicity:")
for logP in logP_values
    result = calculate_white_grey_matter_distribution(logP=logP)
    @printf("    logP %.1f: t½=%.0fh, steady-state=%.1f days [%s]\n",
            logP,
            result.equilibration_halflife_hours,
            result.time_to_steady_state_days,
            result.time_to_steady_state_days > 7 ? "SLOW" : "FAST")
end

# Test that lipophilic drugs take weeks
lipophilic = calculate_white_grey_matter_distribution(logP=4.0)
hydrophilic = calculate_white_grey_matter_distribution(logP=0.5)

println()
@printf("  Hydrophilic (logP 0.5): %.1f days to steady state\n", hydrophilic.time_to_steady_state_days)
@printf("  Lipophilic (logP 4.0):  %.1f days to steady state\n", lipophilic.time_to_steady_state_days)

test9_pass = lipophilic.time_to_steady_state_days > 10 && hydrophilic.time_to_steady_state_days < 3
println("  Expected: Lipophilic >> 1 week, Hydrophilic < 1 week")
println("  TEST 9: ", test9_pass ? "PASS" : "FAIL")
println()

# ===========================================================================
# TEST 10: INTRANASAL DELIVERY
# ===========================================================================

println("TEST 10: Intranasal BBB Bypass")
println("-" ^ 60)

# Test P-gp substrate vs non-substrate
pgp_sub = predict_intranasal_brain_bioavailability(MW=350.0, logP=2.5, is_pgp_substrate=true)
non_pgp = predict_intranasal_brain_bioavailability(MW=350.0, logP=2.5, is_pgp_substrate=false)

@printf("  P-gp substrate (IN):    %.0f%% brain bioavailability\n", pgp_sub.total_brain_bioavailability * 100)
@printf("  Non-P-gp substrate (IN): %.0f%% brain bioavailability\n", non_pgp.total_brain_bioavailability * 100)
@printf("  Direct pathway:         %.0f%% (bypasses P-gp!)\n", pgp_sub.direct_pathway_contribution * 100)
@printf("  Advantage vs oral:      %.1fx for P-gp substrates\n", pgp_sub.advantage_vs_oral)

# P-gp substrates should have advantage via intranasal
test10_pass = pgp_sub.advantage_vs_oral > 2.0
println()
println("  Expected: P-gp substrates > 2x advantage via IN")
println("  TEST 10: ", test10_pass ? "PASS" : "FAIL")
println()

# ===========================================================================
# TEST 11: DYNAMIC Kp,uu INTEGRATION
# ===========================================================================

println("TEST 11: Dynamic Kp,uu Integration")
println("-" ^ 60)

# Test a P-gp substrate in various conditions
println("  Risperidone (P-gp substrate) under different conditions:")

# Baseline (healthy adult, midday)
baseline = calculate_dynamic_kpuu(
    logP=3.0, fup=0.10, MW=410.0, TPSA=62.0, HBD=0,
    is_base=true, pKa=8.2, is_pgp_substrate=true,
    circadian_phase=MIDDAY,
    immune_status=IMMUNOCOMPETENT,
    age_group=ADULT
)

# Night dosing
night = calculate_dynamic_kpuu(
    logP=3.0, fup=0.10, MW=410.0, TPSA=62.0, HBD=0,
    is_base=true, pKa=8.2, is_pgp_substrate=true,
    circadian_phase=NIGHT_NADIR,
    immune_status=IMMUNOCOMPETENT,
    age_group=ADULT
)

# Sepsis
sepsis = calculate_dynamic_kpuu(
    logP=3.0, fup=0.10, MW=410.0, TPSA=62.0, HBD=0,
    is_base=true, pKa=8.2, is_pgp_substrate=true,
    circadian_phase=MIDDAY,
    immune_status=SEVERE_INFLAMMATION,
    il6_fold=50.0,
    age_group=ADULT
)

# Infant with meningitis
infant_meningitis = calculate_dynamic_kpuu(
    logP=3.0, fup=0.10, MW=410.0, TPSA=62.0, HBD=0,
    is_base=true, pKa=8.2, is_pgp_substrate=true,
    circadian_phase=MIDDAY,
    immune_status=MODERATE_INFLAMMATION,
    meningitis_stage=STAGE_II_ESTABLISHED,
    age_group=INFANT
)

@printf("    Baseline (adult, day):  Kp,uu = %.2f\n", baseline.Kpuu_dynamic)
@printf("    Night dosing:           Kp,uu = %.2f (%.1fx)\n",
        night.Kpuu_dynamic, night.fold_change)
@printf("    Sepsis (IL-6 50x):      Kp,uu = %.2f (%.1fx)\n",
        sepsis.Kpuu_dynamic, sepsis.fold_change)
@printf("    Infant + meningitis:    Kp,uu = %.2f (%.1fx)\n",
        infant_meningitis.Kpuu_dynamic, infant_meningitis.fold_change)

# All pathological conditions should increase Kp,uu
test11_pass = (night.Kpuu_dynamic > baseline.Kpuu_dynamic &&
               sepsis.Kpuu_dynamic > baseline.Kpuu_dynamic * 2 &&
               infant_meningitis.Kpuu_dynamic > baseline.Kpuu_dynamic * 3)

println()
println("  Expected: All pathological > baseline, sepsis > 2x, infant+meningitis > 3x")
println("  TEST 11: ", test11_pass ? "PASS" : "FAIL")
println()

# ===========================================================================
# TEST 12: LITHIUM SPECIAL CASE
# ===========================================================================

println("TEST 12: Lithium BBB Penetration")
println("-" ^ 60)

# Normal therapeutic level
normal = calculate_lithium_brain_penetration(plasma_level_mEq_L=0.8)

# Toxic level
toxic = calculate_lithium_brain_penetration(plasma_level_mEq_L=1.8)

# Dehydrated elderly on NSAIDs
risk_patient = calculate_lithium_brain_penetration(
    plasma_level_mEq_L=1.0,
    age_group=ELDERLY,
    has_dehydration=true,
    on_nsaids=true
)

@printf("  Normal (0.8 mEq/L):  brain = %.2f mEq/L [%s]\n",
        normal.brain_level, normal.toxicity_risk)
@printf("  Toxic (1.8 mEq/L):   brain = %.2f mEq/L [%s]\n",
        toxic.brain_level, toxic.toxicity_risk)
@printf("  High-risk patient:   effective plasma = %.2f mEq/L [%s]\n",
        risk_patient.plasma_level, risk_patient.toxicity_risk)

# Brain:plasma ratio should be ~0.5-0.8
test12_pass = (0.5 <= normal.brain_plasma_ratio <= 0.8 &&
               toxic.toxicity_risk == "TOXIC: Seizures, confusion. Consider dialysis." &&
               risk_patient.plasma_level > 1.0)  # Risk factors increased effective level

println()
@printf("  Brain:plasma ratio: %.2f (expected 0.5-0.8)\n", normal.brain_plasma_ratio)
println("  TEST 12: ", test12_pass ? "PASS" : "FAIL")
println()

# ===========================================================================
# SUMMARY
# ===========================================================================

println("=" ^ 80)
println("VALIDATION SUMMARY")
println("=" ^ 80)

tests = [
    ("Baseline Kp,uu Literature", test1_pass),
    ("Circadian P-gp Variation", test2_pass),
    ("Inflammation P-gp Effect", test3_pass),
    ("Meningitis BBB Staging", test4_pass),
    ("CSF Drug Penetration", test5_pass),
    ("Pediatric BBB Maturation", test6_pass),
    ("COVID BBB Effects", test7_pass),
    ("Glymphatic Sleep Effects", test8_pass),
    ("White/Grey Matter", test9_pass),
    ("Intranasal Delivery", test10_pass),
    ("Dynamic Kp,uu Integration", test11_pass),
    ("Lithium Special Case", test12_pass),
]

passed = sum(t[2] for t in tests)
total_tests = length(tests)

println()
for (name, result) in tests
    status = result ? "PASS" : "FAIL"
    marker = result ? "[OK]" : "[X]"
    @printf("  %s %-35s %s\n", marker, name, status)
end

println()
println("-" ^ 80)
@printf("TOTAL: %d/%d tests passed (%.0f%%)\n", passed, total_tests, 100 * passed / total_tests)

if passed == total_tests
    println()
    println("ALL TESTS PASSED!")
    println("Brain SOTA model validated successfully.")
else
    println()
    println("Some tests failed. Review results above.")
end

println("=" ^ 80)
