"""
Test Clinical Validation Module

Validates our model predictions against published clinical data.
This is the key test for establishing clinical credibility.
"""

using Test
using Dates

include("../src/DarwinPBPK/compartments/clinical_validation.jl")
using .ClinicalValidation

@testset "Clinical Validation" begin

    @testset "Tacrolimus Rb vs Hematocrit" begin
        # This is well-established pharmacokinetic relationship
        # Rb = 1 - Hct + Hct × Ke_p, where Ke_p ≈ 37 for tacrolimus

        rb_normal = calculate_tacrolimus_rb(0.42)
        rb_anemia = calculate_tacrolimus_rb(0.28)
        rb_poly = calculate_tacrolimus_rb(0.55)

        # Expected values from literature (Zahir et al., PMID:11374753)
        @test rb_normal ≈ 16.12 atol=2.0   # Literature: ~15
        @test rb_anemia ≈ 11.08 atol=2.0   # Should be lower
        @test rb_poly ≈ 20.85 atol=3.0     # Should be higher

        # Relationship must hold
        @test rb_anemia < rb_normal < rb_poly

        println("  Tacrolimus Rb:")
        println("    Anemia (Hct=0.28): $(round(rb_anemia, digits=1))")
        println("    Normal (Hct=0.42): $(round(rb_normal, digits=1))")
        println("    Polycythemia (Hct=0.55): $(round(rb_poly, digits=1))")
    end

    @testset "Phenytoin fu in Uremia" begin
        # Classic example: uremic toxins displace phenytoin from albumin
        # Normal fu ≈ 0.10, ESRD fu ≈ 0.25 (2.5x increase)

        normal_fu = 0.10

        fu_ckd3 = calculate_fu_uremia(normal_fu, 45.0)  # GFR 45
        fu_ckd4 = calculate_fu_uremia(normal_fu, 22.0)  # GFR 22
        fu_esrd = calculate_fu_uremia(normal_fu, 10.0)  # GFR 10

        # Expected from Reidenberg 1984 (PMID:6360442)
        @test fu_ckd3 > normal_fu                    # Should increase
        @test fu_ckd4 > fu_ckd3                      # Progressive increase
        @test fu_esrd > fu_ckd4                      # Highest in ESRD
        @test 0.20 < fu_esrd < 0.35                  # Literature: ~0.25

        println("  Phenytoin fu in renal impairment:")
        println("    Normal (GFR=100): $(round(normal_fu, digits=3))")
        println("    CKD3 (GFR=45): $(round(fu_ckd3, digits=3))")
        println("    CKD4 (GFR=22): $(round(fu_ckd4, digits=3))")
        println("    ESRD (GFR=10): $(round(fu_esrd, digits=3))")
        println("    Literature ESRD: 0.25")
    end

    @testset "fu in Hypoalbuminemia" begin
        # Acidic drugs: fu increases when albumin decreases
        normal_fu = 0.10
        normal_albumin = 40.0

        fu_mild = calculate_fu_hypoalbuminemia(normal_fu, 30.0)  # Mild
        fu_mod = calculate_fu_hypoalbuminemia(normal_fu, 25.0)   # Moderate
        fu_severe = calculate_fu_hypoalbuminemia(normal_fu, 20.0) # Severe (cirrhosis)

        # Should follow inverse relationship
        @test fu_mild ≈ normal_fu * (40/30) atol=0.01
        @test fu_mod ≈ normal_fu * (40/25) atol=0.01
        @test fu_severe ≈ normal_fu * (40/20) atol=0.01

        # Cirrhosis data (PMID:3377580): fu increases ~2.2x
        @test 1.8 < fu_severe/normal_fu < 2.5

        println("  fu in hypoalbuminemia (acidic drug, fu_normal=0.10):")
        println("    Albumin 30 g/L: fu=$(round(fu_mild, digits=3)) ($(round(fu_mild/normal_fu, digits=1))x)")
        println("    Albumin 25 g/L: fu=$(round(fu_mod, digits=3)) ($(round(fu_mod/normal_fu, digits=1))x)")
        println("    Albumin 20 g/L: fu=$(round(fu_severe, digits=3)) ($(round(fu_severe/normal_fu, digits=1))x)")
    end

    @testset "fu with Elevated AAG (Basic Drugs)" begin
        # Basic drugs: fu decreases when AAG increases (sepsis, inflammation)
        normal_fu = 0.30  # Lidocaine
        normal_aag = 0.8

        fu_mild = calculate_fu_elevated_aag(normal_fu, 1.2)   # 1.5x AAG
        fu_mod = calculate_fu_elevated_aag(normal_fu, 2.0)    # 2.5x AAG
        fu_severe = calculate_fu_elevated_aag(normal_fu, 2.5) # 3x AAG (sepsis)

        # Should decrease
        @test fu_mild < normal_fu
        @test fu_mod < fu_mild
        @test fu_severe < fu_mod

        # Sepsis data (PMID:3918445): fu decreases to ~40% of normal
        @test 0.3 < fu_severe/normal_fu < 0.5

        println("  fu with elevated AAG (basic drug, fu_normal=0.30):")
        println("    AAG 1.2 g/L: fu=$(round(fu_mild, digits=3)) ($(round(fu_mild/normal_fu*100))% of normal)")
        println("    AAG 2.0 g/L: fu=$(round(fu_mod, digits=3)) ($(round(fu_mod/normal_fu*100))% of normal)")
        println("    AAG 2.5 g/L: fu=$(round(fu_severe, digits=3)) ($(round(fu_severe/normal_fu*100))% of normal)")
    end

    @testset "Full Validation Suite" begin
        suite = run_validation_suite(:all)
        summary = get_validation_summary(suite)
        accuracy = calculate_prediction_accuracy(suite)

        println("\n  === VALIDATION SUITE RESULTS ===")
        println("  Total tests: $(summary[:total_tests])")
        println("  Passed: $(summary[:passed])/$(summary[:total_tests]) ($(summary[:pass_rate])%)")
        println("  Mean error: $(summary[:mean_error])%")
        println("  Within 30%: $(round(accuracy[:within_30pct], digits=0))%")

        # Key acceptance criteria
        @test summary[:pass_rate] >= 70.0  # At least 70% pass
        @test accuracy[:mpe] < 25.0        # Mean error < 25%
        @test accuracy[:within_30pct] >= 80.0  # 80% within 30%

        # Check we have good evidence coverage
        @test summary[:evidence_distribution][:A] >= 1  # At least 1 Level A
        @test summary[:evidence_distribution][:B] >= 2  # At least 2 Level B

        println("\n  Evidence levels:")
        println("    Level A (RCT): $(summary[:evidence_distribution][:A])")
        println("    Level B (Large obs): $(summary[:evidence_distribution][:B])")
        println("    Level C (Case): $(summary[:evidence_distribution][:C])")
    end

    @testset "Individual Assertion Details" begin
        suite = run_validation_suite(:all)

        println("\n  === DETAILED RESULTS ===")
        for result in suite.results
            status = result.passed ? "PASS" : "FAIL"
            println("  [$status] $(result.assertion.id)")
            println("      Drug: $(result.assertion.drug)")
            println("      Condition: $(result.assertion.condition)")
            println("      Predicted: $(round(result.predicted, digits=2)) vs Expected: $(result.expected)")
            println("      Error: $(round(result.error*100, digits=1))% (tolerance: $(result.assertion.tolerance*100)%)")
            println("      Evidence: Level $(result.assertion.evidence_level) - $(result.assertion.reference)")
            println()
        end
    end

end

println("\n=== Clinical Validation Tests Complete ===")
