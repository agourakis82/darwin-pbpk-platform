"""
Test Suite for TGA Validation Module

Tests thrombin generation assay validation against clinical datasets.

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

using Test
using DarwinPBPK
using Statistics

@testset "TGA Validation Module" begin

    @testset "TGAParameters Construction" begin
        params = TGAParameters(
            3.5,      # lag_time
            9.2,      # time_to_peak
            315.0,    # peak_thrombin
            1820.0,   # ETP
            98.0,     # velocity_index
            8.5,      # width_50
            22.0,     # start_tail
            1.0,      # TF concentration
            "test"    # patient condition
        )

        @test params.lag_time == 3.5
        @test params.peak_thrombin == 315.0
        @test params.tf_concentration == 1.0
        @test params.patient_condition == "test"
    end

    @testset "Extract TGA Parameters from Simulated Curve" begin
        # Simulate a realistic thrombin generation curve
        # Peak at ~10 min, max ~300 nM
        time_points = collect(0.0:0.1:30.0)  # 0-30 minutes

        # Realistic curve: slow rise, peak, decay
        thrombin_curve = [
            max(0.0, 300.0 * ((t - 3.0) / 7.0) * exp(-(t - 10.0)^2 / 40.0))
            for t in time_points
        ]

        params = extract_tga_parameters(thrombin_curve, time_points,
                                       tf_concentration=1.0,
                                       patient_condition="simulated_healthy")

        # Check that parameters are in reasonable ranges
        @test params.lag_time >= 0.0
        @test params.lag_time < 10.0  # Should be < 10 min for healthy
        @test params.time_to_peak >= params.lag_time
        @test params.peak_thrombin > 0.0
        @test params.etp > 0.0  # ETP should be positive
        @test params.velocity_index > 0.0

        println("\nExtracted TGA Parameters from Simulated Curve:")
        println("  Lag time: $(round(params.lag_time, digits=2)) min")
        println("  Time to peak: $(round(params.time_to_peak, digits=2)) min")
        println("  Peak thrombin: $(round(params.peak_thrombin, digits=1)) nM")
        println("  ETP: $(round(params.etp, digits=1)) nM·min")
        println("  Velocity Index: $(round(params.velocity_index, digits=1)) nM/min")
    end

    @testset "Clinical Reference Datasets" begin
        # Test that all datasets are accessible
        @test HEALTHY_TGA_1PM_TF.name == "Healthy adults (1pM TF)"
        @test HEALTHY_TGA_1PM_TF.n_subjects == 123
        @test HEALTHY_TGA_1PM_TF.lag_time_mean ≈ 3.7
        @test HEALTHY_TGA_1PM_TF.peak_thrombin_mean ≈ 312.0

        @test HEMOPHILIA_A_TGA.condition == "hemophilia_A_severe"
        @test HEMOPHILIA_A_TGA.peak_thrombin_mean < HEALTHY_TGA_1PM_TF.peak_thrombin_mean
        @test HEMOPHILIA_A_TGA.lag_time_mean > HEALTHY_TGA_1PM_TF.lag_time_mean

        @test WARFARIN_INR2_TGA.condition == "warfarin_INR2"
        @test WARFARIN_INR2_TGA.etp_mean < HEALTHY_TGA_5PM_TF.etp_mean

        @test DOAC_RIVAROXABAN_TGA.condition == "rivaroxaban_therapeutic"

        # Test all datasets are retrievable
        all_datasets = get_all_clinical_datasets()
        @test length(all_datasets) == 9
        @test all(d -> d isa ClinicalTGADataset, all_datasets)

        println("\nAvailable Clinical TGA Datasets: $(length(all_datasets))")
        for dataset in all_datasets
            println("  - $(dataset.name) (N=$(dataset.n_subjects))")
        end
    end

    @testset "Prediction Error Calculation" begin
        # Test fold error calculation
        pred = 100.0
        obs = 50.0

        fe, aafe = calculate_prediction_error(pred, obs)
        @test fe == 2.0  # 100/50 = 2
        @test aafe == 2.0  # max(2, 1/2) = 2

        # Test reverse
        pred2 = 50.0
        obs2 = 100.0
        fe2, aafe2 = calculate_prediction_error(pred2, obs2)
        @test fe2 == 0.5  # 50/100 = 0.5
        @test aafe2 == 2.0  # max(0.5, 2) = 2

        # Perfect prediction
        fe3, aafe3 = calculate_prediction_error(100.0, 100.0)
        @test fe3 == 1.0
        @test aafe3 == 1.0
    end

    @testset "Goodness of Fit Metrics" begin
        # Perfect predictions
        predicted = [100.0, 200.0, 300.0, 400.0, 500.0]
        observed = [100.0, 200.0, 300.0, 400.0, 500.0]

        metrics = calculate_goodness_of_fit(predicted, observed)

        @test metrics.afe ≈ 1.0 atol=1e-10
        @test metrics.aafe ≈ 1.0 atol=1e-10
        @test metrics.rmse ≈ 0.0 atol=1e-10
        @test metrics.mae ≈ 0.0 atol=1e-10
        @test metrics.r_squared ≈ 1.0 atol=1e-10
        @test metrics.within_2fold == 1.0
        @test metrics.within_3fold == 1.0

        # 2-fold predictions (should pass)
        predicted2 = [100.0, 200.0, 300.0, 400.0, 500.0]
        observed2 = [50.0, 100.0, 150.0, 200.0, 250.0]  # All 2-fold

        metrics2 = calculate_goodness_of_fit(predicted2, observed2)
        @test metrics2.aafe ≈ 2.0
        @test metrics2.within_2fold == 1.0  # 100% within 2-fold

        # Mixed predictions
        predicted3 = [100.0, 200.0, 300.0, 400.0, 600.0]
        observed3 = [100.0, 100.0, 100.0, 200.0, 100.0]  # Mix of 1×, 2×, 3×, 6×

        metrics3 = calculate_goodness_of_fit(predicted3, observed3)
        @test metrics3.within_2fold == 0.6  # 3/5 = 60%
        @test metrics3.within_3fold == 0.8  # 4/5 = 80%
    end

    @testset "Compare to Clinical Dataset" begin
        # Create simulated TGA matching healthy reference
        simulated = TGAParameters(
            3.8,      # lag_time (close to 3.7 ± 0.6)
            9.3,      # time_to_peak (close to 9.5 ± 1.2)
            320.0,    # peak_thrombin (close to 312 ± 58)
            1850.0,   # ETP (close to 1815 ± 294)
            97.0,     # velocity_index (close to 95 ± 18)
            8.0,      # width_50
            20.0,     # start_tail
            1.0,      # TF
            "simulated_healthy"
        )

        comparison = compare_to_clinical(simulated, HEALTHY_TGA_1PM_TF)

        # Check structure
        @test haskey(comparison, "reference_dataset")
        @test haskey(comparison, "parameter_comparisons")
        @test haskey(comparison, "overall_metrics")
        @test haskey(comparison, "acceptance_criteria")

        @test comparison["reference_dataset"] == "Healthy adults (1pM TF)"
        @test comparison["n_subjects"] == 123

        # Check metrics
        metrics = comparison["overall_metrics"]
        @test metrics.aafe < 2.0  # Should pass (good match)
        @test metrics.r_squared > 0.7  # Should pass

        # Check acceptance
        criteria = comparison["acceptance_criteria"]
        @test criteria["AAFE_pass"] == true
        @test criteria["R2_pass"] == true
        @test criteria["all_criteria_met"] == true

        # Check parameter comparisons
        param_comp = comparison["parameter_comparisons"]
        @test haskey(param_comp, "lag_time")
        @test haskey(param_comp, "peak_thrombin")
        @test haskey(param_comp, "etp")

        lag_comp = param_comp["lag_time"]
        @test abs(lag_comp["z_score"]) < 1.0  # Within 1 SD
        @test lag_comp["within_1sd"] == true

        println("\nComparison to Clinical (Healthy 1pM TF):")
        println("  AAFE: $(round(metrics.aafe, digits=3)) " *
                (criteria["AAFE_pass"] ? "✓" : "✗"))
        println("  R²: $(round(metrics.r_squared, digits=3)) " *
                (criteria["R2_pass"] ? "✓" : "✗"))
        println("  Within 2-fold: $(round(metrics.within_2fold * 100, digits=1))% " *
                (criteria["within_2fold_pass"] ? "✓" : "✗"))
        println("  Overall: " * (criteria["all_criteria_met"] ? "✓ PASS" : "✗ FAIL"))
    end

    @testset "Validate Multiple Simulations" begin
        # Create multiple simulated scenarios

        # 1. Healthy (1pM TF)
        healthy_sim = TGAParameters(
            3.6, 9.4, 315.0, 1800.0, 96.0, 8.0, 20.0, 1.0, "healthy"
        )

        # 2. Hemophilia A (very impaired)
        hemophilia_sim = TGAParameters(
            10.8, 21.5, 82.0, 490.0, 11.5, 15.0, 35.0, 1.0, "hemophilia_A"
        )

        # 3. Warfarin INR 2 (moderately impaired)
        warfarin_sim = TGAParameters(
            5.0, 10.5, 225.0, 1200.0, 56.0, 10.0, 25.0, 5.0, "warfarin_INR2"
        )

        simulations = Dict(
            "healthy" => healthy_sim,
            "hemophilia_A" => hemophilia_sim,
            "warfarin_INR2" => warfarin_sim
        )

        datasets = [
            HEALTHY_TGA_1PM_TF,
            HEMOPHILIA_A_TGA,
            WARFARIN_INR2_TGA
        ]

        results = validate_coagulation_model(simulations, datasets)

        # Check structure
        @test haskey(results, "validation_results")
        @test haskey(results, "summary")
        @test haskey(results, "timestamp")

        summary = results["summary"]
        @test summary["n_datasets_validated"] == 3
        @test haskey(summary, "mean_AAFE")
        @test haskey(summary, "mean_R2")
        @test haskey(summary, "pass_rate")

        # Should have good overall performance
        @test summary["mean_AAFE"] < 2.0
        @test summary["mean_R2"] > 0.7
        @test summary["pass_rate"] >= 0.8  # At least 80% pass

        println("\nMulti-Dataset Validation Summary:")
        println("  Datasets validated: $(summary["n_datasets_validated"])")
        println("  Datasets passed: $(summary["datasets_passed_all_criteria"])")
        println("  Pass rate: $(round(summary["pass_rate"] * 100, digits=1))%")
        println("  Mean AAFE: $(round(summary["mean_AAFE"], digits=3))")
        println("  Mean R²: $(round(summary["mean_R2"], digits=3))")
        println("  Overall: " *
                (summary["overall_model_acceptable"] ? "✓ ACCEPTABLE" : "✗ NOT ACCEPTABLE"))

        # Print detailed validation summary
        print_validation_summary(results)
    end

    @testset "Edge Cases" begin
        # Test with minimal curve
        time = [0.0, 1.0, 2.0]
        thrombin = [0.0, 100.0, 50.0]

        params = extract_tga_parameters(thrombin, time)
        @test params.peak_thrombin == 100.0
        @test params.time_to_peak == 1.0
        @test params.etp > 0.0

        # Test with flat curve (no thrombin generation)
        flat_thrombin = [0.0, 0.0, 0.0]
        params_flat = extract_tga_parameters(flat_thrombin, time)
        @test params_flat.peak_thrombin == 0.0
        @test params_flat.etp == 0.0
    end

    @testset "Clinical Dataset Properties" begin
        # Verify physiological relationships

        # 1. Higher TF → shorter lag, higher peak
        @test HEALTHY_TGA_5PM_TF.lag_time_mean < HEALTHY_TGA_1PM_TF.lag_time_mean
        @test HEALTHY_TGA_5PM_TF.peak_thrombin_mean > HEALTHY_TGA_1PM_TF.peak_thrombin_mean

        # 2. Hemophilia → much lower thrombin generation
        @test HEMOPHILIA_A_TGA.peak_thrombin_mean < 0.3 * HEALTHY_TGA_1PM_TF.peak_thrombin_mean
        @test HEMOPHILIA_A_TGA.etp_mean < 0.3 * HEALTHY_TGA_1PM_TF.etp_mean

        # 3. Anticoagulation → reduced thrombin
        @test WARFARIN_INR2_TGA.etp_mean < HEALTHY_TGA_5PM_TF.etp_mean
        @test DOAC_RIVAROXABAN_TGA.peak_thrombin_mean < HEALTHY_TGA_5PM_TF.peak_thrombin_mean

        # 4. Higher INR → more impairment
        @test WARFARIN_INR3_TGA.etp_mean < WARFARIN_INR2_TGA.etp_mean
        @test WARFARIN_INR3_TGA.peak_thrombin_mean < WARFARIN_INR2_TGA.peak_thrombin_mean

        # 5. Apixaban more potent than rivaroxaban
        @test DOAC_APIXABAN_TGA.etp_mean < DOAC_RIVAROXABAN_TGA.etp_mean
    end

end

println("\n" * "="^80)
println("TGA Validation Module - All Tests Passed ✓")
println("="^80)
