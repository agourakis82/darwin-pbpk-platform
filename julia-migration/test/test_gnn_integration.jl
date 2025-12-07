# ===========================================================================
# Tests for GNN-MedLang Integration Module
# ===========================================================================
# Standalone test file - can be run without full DarwinPBPK compilation
# ===========================================================================

using Test

# Include module directly for faster testing
include(joinpath(@__DIR__, "..", "src", "DarwinPBPK", "medlang", "gnn_integration.jl"))
using .GNNMedLangIntegration

@testset "GNN-MedLang Integration" begin

    @testset "PKPrediction struct" begin
        pred = PKPrediction(
            15.0,   # clearance L/h
            50.0,   # volume L
            0.8,    # bioavailability
            1.5,    # ka 1/h
            4.0,    # half_life h
            0.05    # fu
        )

        @test pred.clearance == 15.0
        @test pred.volume == 50.0
        @test pred.bioavailability == 0.8
        @test pred.ka == 1.5
        @test pred.half_life == 4.0
        @test pred.fu == 0.05
    end

    @testset "PKPredictionWithUQ struct" begin
        mean_pred = PKPrediction(15.0, 50.0, 0.8, 1.5, 4.0, 0.05)
        std_pred = PKPrediction(2.0, 5.0, 0.1, 0.2, 0.5, 0.01)
        ci_lower = PKPrediction(11.0, 40.0, 0.6, 1.1, 3.0, 0.03)
        ci_upper = PKPrediction(19.0, 60.0, 1.0, 1.9, 5.0, 0.07)

        pred_uq = PKPredictionWithUQ(mean_pred, std_pred, ci_lower, ci_upper, 100)

        @test pred_uq.mean.clearance == 15.0
        @test pred_uq.std.clearance == 2.0
        @test pred_uq.ci_lower.clearance == 11.0
        @test pred_uq.ci_upper.clearance == 19.0
        @test pred_uq.n_samples == 100
    end

    @testset "MedLang generation - oral" begin
        pred = PKPrediction(15.0, 50.0, 0.8, 1.5, 4.0, 0.05)

        medlang = generate_medlang_from_gnn(pred, "TestDrug"; route=:oral, dose=100.0)

        # Check structure
        @test occursin("model TESTDRUG_GNN", medlang)
        @test occursin("Route: Oral", medlang)
        @test occursin("CL : Clearance = 15.0", medlang)
        @test occursin("V : Volume = 50.0", medlang)
        @test occursin("ka : Rate = 1.5", medlang)
        @test occursin("F : Fraction = 0.8", medlang)
        @test occursin("fu : Fraction = 0.05", medlang)
        @test occursin("d(A_gut)/dt", medlang)
        @test occursin("d(C_plasma)/dt", medlang)
        @test occursin("dose 100.0 [mg]", medlang)
    end

    @testset "MedLang generation - IV" begin
        pred = PKPrediction(15.0, 50.0, 1.0, 0.0, 4.0, 0.05)

        medlang = generate_medlang_from_gnn(pred, "IVDrug"; route=:iv, dose=50.0)

        # Check structure
        @test occursin("model IVDRUG_GNN_IV", medlang)
        @test occursin("Route: IV Bolus", medlang)
        @test occursin("CL : Clearance = 15.0", medlang)
        @test occursin("V : Volume = 50.0", medlang)
        @test occursin("dose 50.0 [mg]", medlang)
        @test occursin("via IV", medlang)
        # IV model should not have gut compartment
        @test !occursin("A_gut", medlang)
    end

    @testset "MedLang with uncertainty" begin
        mean_pred = PKPrediction(15.0, 50.0, 0.8, 1.5, 4.0, 0.05)
        std_pred = PKPrediction(2.0, 5.0, 0.1, 0.2, 0.5, 0.01)
        ci_lower = PKPrediction(11.0, 40.0, 0.6, 1.1, 3.0, 0.03)
        ci_upper = PKPrediction(19.0, 60.0, 0.95, 1.9, 5.0, 0.07)

        pred_uq = PKPredictionWithUQ(mean_pred, std_pred, ci_lower, ci_upper, 100)

        medlang = generate_medlang_with_uncertainty(pred_uq, "UQDrug")

        # Check uncertainty block
        @test occursin("Uncertainty Estimates (95% CI)", medlang)
        @test occursin("CL: 11.0 - 19.0", medlang)
        @test occursin("V: 40.0 - 60.0", medlang)
        @test occursin("Based on 100 MC-Dropout samples", medlang)
    end

    @testset "Population MedLang generation" begin
        mean_pred = PKPrediction(15.0, 50.0, 0.8, 1.5, 4.0, 0.05)
        std_pred = PKPrediction(3.0, 10.0, 0.1, 0.3, 0.5, 0.01)
        ci_lower = PKPrediction(9.0, 30.0, 0.6, 0.9, 3.0, 0.03)
        ci_upper = PKPrediction(21.0, 70.0, 1.0, 2.1, 5.0, 0.07)

        pred_uq = PKPredictionWithUQ(mean_pred, std_pred, ci_lower, ci_upper, 100)

        pop_medlang = create_population_medlang(pred_uq, "PopDrug"; n_subjects=50, dose=200.0)

        # Check population structure
        @test occursin("population_model POPDRUG_POP", pop_medlang)
        @test occursin("TV_CL : Clearance = 15.0", pop_medlang)
        @test occursin("TV_V : Volume = 50.0", pop_medlang)
        @test occursin("omega_CL", pop_medlang)
        @test occursin("omega_V", pop_medlang)
        @test occursin("individual CL_i", pop_medlang)
        @test occursin("individual V_i", pop_medlang)
        @test occursin("population n = 50", pop_medlang)
        @test occursin("dose 200.0 [mg]", pop_medlang)
    end

    @testset "PK parameter clamping" begin
        # Test extreme values get clamped
        extreme_pred = PKPrediction(
            0.01,    # Very low clearance
            1000.0,  # Very high volume
            1.5,     # Invalid F > 1
            0.01,    # Very low ka
            100.0,   # Very long half-life
            0.0001   # Very low fu
        )

        clamped = GNNMedLangIntegration.clamp_pk_params(extreme_pred)

        @test clamped.clearance >= 0.1
        @test clamped.volume <= 500.0
        @test clamped.bioavailability <= 1.0
        @test clamped.ka >= 0.1
        @test clamped.half_life <= 72.0
        @test clamped.fu >= 0.001
    end

    @testset "GNNPKPredictor struct" begin
        # Create mock predictor (without actual model)
        predictor = GNNPKPredictor(
            nothing,  # encoder
            nothing,  # pk_head
            true,     # use_uncertainty
            50        # n_mc_samples
        )

        @test predictor.use_uncertainty == true
        @test predictor.n_mc_samples == 50
    end

    @testset "GNNPBPKPipeline struct" begin
        # Mock functions
        mock_compile = x -> x
        mock_simulate = (m; t_max=24.0) -> nothing

        predictor = GNNPKPredictor(nothing, nothing, false, 10)
        pipeline = GNNPBPKPipeline(predictor, mock_compile, mock_simulate)

        @test pipeline.predictor === predictor
        @test pipeline.compile_fn === mock_compile
        @test pipeline.simulate_fn === mock_simulate
    end

    @testset "Derived parameter calculations" begin
        # ke = CL/V
        pred = PKPrediction(15.0, 50.0, 0.8, 1.5, 4.0, 0.05)
        expected_ke = 15.0 / 50.0  # 0.3 1/h

        medlang = generate_medlang_from_gnn(pred, "TestDrug")

        # Check ke is calculated correctly
        @test occursin("ke : Rate = 0.3", medlang)
    end

    @testset "Module export consistency" begin
        # Verify exports are available at module level
        @test isdefined(GNNMedLangIntegration, :PKPrediction)
        @test isdefined(GNNMedLangIntegration, :PKPredictionWithUQ)
        @test isdefined(GNNMedLangIntegration, :GNNPKPredictor)
        @test isdefined(GNNMedLangIntegration, :generate_medlang_from_gnn)
        @test isdefined(GNNMedLangIntegration, :generate_medlang_with_uncertainty)
        @test isdefined(GNNMedLangIntegration, :create_population_medlang)
        @test isdefined(GNNMedLangIntegration, :GNNPBPKPipeline)
    end

end

println("GNN-MedLang Integration tests completed!")
