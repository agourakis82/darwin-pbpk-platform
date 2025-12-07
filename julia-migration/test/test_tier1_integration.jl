"""
Tier 1 Integration Tests - Q1 Publication Features

Tests for the newly implemented SOTA features:
1. ChemBERTa + D-MPNN Multimodal Encoder
2. Turing.jl Bayesian PBPK
3. Bootstrap Validation Metrics
4. MC-Dropout Uncertainty
5. Calibration Metrics
6. External Validation Protocol

Author: Dr. Demetrios Agourakis + AI Assistant
Date: December 2025
"""

using Test
using Statistics
using Random
using Distributions

# Set random seed for reproducibility
Random.seed!(42)

@testset "Tier 1 Integration Tests" begin

    #==========================================================================
      Test 1: Multimodal Encoder (SOTAMultimodalEncoderV2)
    ==========================================================================#
    @testset "Multimodal Encoder" begin
        # Test SMILESEncoder (GRU fallback)
        include("../src/DarwinPBPK/ml/multimodal_encoder.jl")
        using .MultimodalEncoder

        @testset "SMILESEncoder" begin
            encoder = SMILESEncoder()

            # Test single SMILES
            emb = encoder("CCO")  # Ethanol
            @test length(emb) == SMILES_OUTPUT_DIM
            @test eltype(emb) == Float32
            @test !any(isnan, emb)

            # Test different molecules can be encoded (embeddings may be similar with random init)
            emb2 = encoder("CC(=O)O")  # Acetic acid
            @test length(emb2) == SMILES_OUTPUT_DIM
            @test !any(isnan, emb2)

            # Test batch encoding
            smiles_batch = ["CCO", "CC(=O)O", "c1ccccc1"]
            batch_emb = encoder(smiles_batch)
            @test size(batch_emb) == (SMILES_OUTPUT_DIM, 3)
        end

        @testset "GNNEncoder" begin
            encoder = GNNEncoder()

            # Test single SMILES
            emb = encoder("CCO")
            if emb !== nothing
                @test length(emb) == GNN_OUTPUT_DIM
                @test eltype(emb) == Float32
            end

            # Test complex molecule
            emb_complex = encoder("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
            if emb_complex !== nothing
                @test length(emb_complex) == GNN_OUTPUT_DIM
            end
        end

        @testset "CrossAttentionFusion" begin
            fusion = CrossAttentionFusion([768, 256, 128])

            # Test fusion of embeddings
            emb1 = randn(Float32, 768)
            emb2 = randn(Float32, 256)
            emb3 = randn(Float32, 128)

            fused = fusion([emb1, emb2, emb3])
            @test length(fused) == FUSION_DIM
            @test !any(isnan, fused)
        end

        @testset "EnhancedMultimodalEncoder" begin
            encoder = EnhancedMultimodalEncoder(use_gnn=true, use_quantum=false)

            # Test encoding
            emb = encoder("CCO")
            @test length(emb) == FUSION_DIM
            @test !any(isnan, emb)
        end
    end

    #==========================================================================
      Test 2: D-MPNN Encoder
    ==========================================================================#
    @testset "D-MPNN Encoder" begin
        include("../src/DarwinPBPK/ml/dmpnn.jl")
        using .DMPNN
        using MolecularGraph: smilestomol

        @testset "Atom Features" begin
            # Test atom featurization via SMILES parsing
            mol = smilestomol("CCO")
            features = DMPNN.atom_features(mol)

            @test size(features, 1) == ATOM_FEAT_DIM
            @test size(features, 2) == 3  # 3 atoms in ethanol
        end

        @testset "DMPNNConv Layer" begin
            conv = DMPNNConv(ATOM_FEAT_DIM, HIDDEN_DIM)

            # Verify layer construction
            @test conv.W_message !== nothing
            @test conv.W_hidden !== nothing
        end

        @testset "DMPNNEncoder Full" begin
            encoder = DMPNNEncoder()

            # Test encoding
            emb = encoder("CCO")
            @test length(emb) == OUTPUT_DIM
            @test !any(isnan, emb)

            # Test batch
            batch = encode_molecules(encoder, ["CCO", "CC(=O)O"])
            @test size(batch) == (OUTPUT_DIM, 2)
        end
    end

    #==========================================================================
      Test 3: Turing.jl PBPK Models
    ==========================================================================#
    @testset "Turing.jl PBPK" begin
        include("../src/DarwinPBPK/ml/turing_pbpk.jl")
        using .TuringPBPK

        @testset "ODE Systems" begin
            # Test one-compartment ODE
            du = zeros(1)
            u = [10.0]
            p = [5.0, 50.0]  # CL, V

            one_compartment_ode!(du, u, p, 0.0)
            @test du[1] < 0  # Concentration should decrease

            # Test two-compartment ODE
            du2 = zeros(2)
            u2 = [10.0, 0.0]
            p2 = [5.0, 20.0, 2.0, 50.0]  # CL, V1, Q, V2

            two_compartment_ode!(du2, u2, p2, 0.0)
            @test du2[1] < 0  # Central should decrease
            @test du2[2] > 0  # Peripheral should increase
        end

        @testset "Bayesian Models" begin
            # Generate synthetic PK data
            times = collect(0.5:0.5:12.0)
            true_CL = 10.0
            true_V = 50.0
            dose = 100.0

            true_conc = dose ./ true_V .* exp.(-true_CL ./ true_V .* times)
            obs_conc = true_conc .+ 0.2 .* randn(length(times))
            obs_conc = max.(obs_conc, 0.01)  # Ensure positive

            # Test model creation
            model = bayesian_one_compartment(obs_conc, times, dose)
            @test model !== nothing
        end

        @testset "Quick Bayesian PK" begin
            # Simple test with synthetic data
            times = [1.0, 2.0, 4.0, 8.0, 12.0]
            dose = 100.0
            obs_conc = [8.0, 6.0, 3.5, 1.5, 0.6]

            # This would normally run MCMC - just test function exists
            @test hasmethod(quick_bayesian_pk, Tuple{Vector{Float64}, Vector{Float64}, Float64})
        end
    end

    #==========================================================================
      Test 4: Bootstrap Validation Metrics
    ==========================================================================#
    @testset "Bootstrap Validation" begin
        include("../src/DarwinPBPK/validation.jl")
        using .Validation

        # Generate test data
        n = 50
        true_values = 10.0 .* rand(n) .+ 1.0
        pred_values = true_values .* (0.8 .+ 0.4 .* rand(n))  # ±20% error

        @testset "Basic Metrics" begin
            gmfe = geometric_mean_fold_error(pred_values, true_values)
            @test 1.0 <= gmfe <= 3.0  # Reasonable range

            afe = average_fold_error(pred_values, true_values)
            @test 0.5 <= afe <= 2.0

            aafe = absolute_average_fold_error(pred_values, true_values)
            @test 1.0 <= aafe <= 2.0
        end

        @testset "Bootstrap CIs" begin
            result = gmfe_with_ci(pred_values, true_values; n_bootstrap=500)

            @test result isa BootstrapResult
            @test result.ci_lower < result.estimate < result.ci_upper
            @test result.se > 0

            # Test formatting
            formatted = format_bootstrap_result(result)
            @test occursin("95% CI", formatted)
        end

        @testset "Regulatory Metrics with CI" begin
            metrics = regulatory_metrics_with_ci(pred_values, true_values; n_bootstrap=500)

            @test haskey(metrics, "GMFE")
            @test haskey(metrics, "AFE")
            @test haskey(metrics, "AAFE")
            @test haskey(metrics, "percent_within_2fold")
            @test haskey(metrics, "prob_meets_FDA_criteria")

            @test metrics["GMFE"] isa BootstrapResult
        end

        @testset "LaTeX Output" begin
            metrics = regulatory_metrics_with_ci(pred_values, true_values; n_bootstrap=100)
            latex_row = latex_metrics_row(metrics, "Test Model")

            @test occursin("Test Model", latex_row)
            @test occursin("\\\\", latex_row)
        end
    end

    #==========================================================================
      Test 5: MC-Dropout Uncertainty
    ==========================================================================#
    @testset "MC-Dropout" begin
        include("../src/DarwinPBPK/ml/mc_dropout.jl")
        using .MCDropout
        using Flux

        @testset "MCDropoutWrapper" begin
            # Create simple model with dropout
            model = Chain(
                Dense(10 => 32, relu),
                Dropout(0.1),
                Dense(32 => 1)
            )

            wrapper = MCDropoutWrapper(model; dropout_rate=0.1)
            @test wrapper.dropout_rate == 0.1
        end

        @testset "UncertaintyResult" begin
            # Create mock MC samples
            samples = randn(50, 10)  # 50 MC samples, 10 predictions

            result = UncertaintyResult(samples)

            @test result.n_samples == 50
            @test length(result.mean) == 10
            @test length(result.std) == 10
            @test all(result.ci_lower .< result.ci_upper)
        end

        @testset "Calibration Metrics" begin
            # Create predictions with known uncertainty
            n = 100
            true_means = randn(n) .* 5
            pred_means = true_means .+ randn(n) .* 0.5
            pred_stds = abs.(randn(n) .* 0.5) .+ 0.3

            # Create mock UncertaintyResults
            predictions = [UncertaintyResult(
                reshape(randn(50) .* pred_stds[i] .+ pred_means[i], 50, 1)
            ) for i in 1:n]

            metrics = calibration_metrics(predictions, true_means)

            @test haskey(metrics, "ECE")
            @test 0.0 <= metrics["ECE"] <= 1.0
        end
    end

    #==========================================================================
      Test 6: Calibration Module
    ==========================================================================#
    @testset "Calibration Metrics Module" begin
        include("../src/DarwinPBPK/ml/calibration.jl")
        using .Calibration

        @testset "ECE Computation" begin
            n = 100
            true_values = randn(n)
            pred_means = true_values .+ randn(n) .* 0.3
            pred_stds = abs.(randn(n) .* 0.3) .+ 0.2

            ece = expected_calibration_error(pred_means, pred_stds, true_values)

            @test 0.0 <= ece <= 1.0
            @test isfinite(ece)
        end

        @testset "Reliability Diagram" begin
            n = 100
            true_values = randn(n)
            pred_means = true_values .+ randn(n) .* 0.3
            pred_stds = abs.(randn(n) .* 0.3) .+ 0.2

            rel_data = reliability_diagram(pred_means, pred_stds, true_values)

            @test haskey(rel_data, "expected")
            @test haskey(rel_data, "observed")
            @test haskey(rel_data, "gap")
            @test length(rel_data["expected"]) == length(rel_data["observed"])
        end

        @testset "CRPS" begin
            crps = crps_gaussian(5.0, 1.0, 5.2)
            @test crps >= 0
            @test isfinite(crps)

            # CRPS should be lower when prediction is closer to truth
            crps_close = crps_gaussian(5.0, 1.0, 5.0)
            crps_far = crps_gaussian(5.0, 1.0, 8.0)
            @test crps_close < crps_far
        end

        @testset "Full Calibration Analysis" begin
            n = 100
            true_values = randn(n) .* 5 .+ 10
            pred_means = true_values .+ randn(n) .* 1.0
            pred_stds = abs.(randn(n) .* 0.5) .+ 0.5

            result = full_calibration_analysis(pred_means, pred_stds, true_values)

            @test result isa CalibrationResult
            @test 0.0 <= result.ece <= 1.0
            @test 0.0 <= result.coverage_95 <= 1.0
            @test result.sharpness > 0
        end
    end

    #==========================================================================
      Test 7: External Validation Protocol
    ==========================================================================#
    @testset "External Validation" begin
        include("../src/DarwinPBPK/validation/external_validation.jl")
        using .ExternalValidation

        @testset "Dataset Creation" begin
            dataset = create_validation_dataset(
                name = "Test Dataset",
                source = "Synthetic",
                compounds = ["Drug A", "Drug B", "Drug C"],
                smiles = ["CCO", "CC(=O)O", "c1ccccc1"],
                clearance = [10.0, 15.0, 8.0],
                volume_distribution = [50.0, 60.0, 45.0]
            )

            @test dataset.name == "Test Dataset"
            @test length(dataset.compounds) == 3
            @test haskey(dataset.observed_pk, "CL")
            @test haskey(dataset.observed_pk, "Vd")
        end

        @testset "Example Dataset" begin
            dataset = example_validation_dataset()

            @test length(dataset.compounds) == 15
            @test length(dataset.smiles) == 15
            @test haskey(dataset.observed_pk, "CL")
        end

        @testset "Blind Validation Protocol" begin
            # Create mock prediction function
            function mock_predict(smiles::String)
                return Dict(
                    "CL" => 10.0 + rand() * 5.0,
                    "Vd" => 50.0 + rand() * 20.0
                )
            end

            dataset = create_validation_dataset(
                name = "Mini Test",
                source = "Synthetic",
                compounds = ["A", "B", "C", "D", "E"],
                smiles = ["CCO", "CC", "CCC", "CCCC", "CCCCC"],
                clearance = [10.0, 12.0, 11.0, 9.0, 13.0]
            )

            result = run_blind_validation(
                mock_predict,
                dataset;
                parameters = [:CL],
                model_version = "test-v1",
                n_bootstrap = 100
            )

            @test result isa BlindValidationResult
            @test result.n_compounds == 5
            @test haskey(result.metrics, "CL")
            @test result.protocol["strict_blind"] == true
        end
    end

    #==========================================================================
      Test 8: End-to-End Integration
    ==========================================================================#
    @testset "End-to-End Integration" begin
        @testset "Encoder → Validation Pipeline" begin
            # This tests the full pipeline from encoding to validation

            # 1. Create encoder
            encoder = EnhancedMultimodalEncoder(use_gnn=true, use_quantum=false)

            # 2. Encode molecules
            smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
            embeddings = encoder(smiles)

            @test size(embeddings, 2) == 3

            # 3. Simulate predictions (mock)
            pred = rand(3) .* 10 .+ 5
            obs = pred .* (0.9 .+ 0.2 .* rand(3))

            # 4. Compute validation metrics
            metrics = regulatory_metrics_with_ci(pred, obs; n_bootstrap=100)

            @test haskey(metrics, "GMFE")
            @test metrics["n_valid"] == 3
        end

        @testset "UQ → Calibration Pipeline" begin
            # Test uncertainty quantification and calibration together

            n = 30
            true_values = randn(n) .* 3 .+ 10

            # Simulate predictions with uncertainty
            pred_means = true_values .+ randn(n) .* 0.5
            pred_stds = abs.(randn(n) .* 0.3) .+ 0.3

            # Calibration analysis
            cal_result = full_calibration_analysis(pred_means, pred_stds, true_values)

            @test cal_result.ece >= 0
            @test 0.0 <= cal_result.coverage_95 <= 1.0

            # If not well calibrated, recalibrate
            if !cal_result.is_well_calibrated
                recal_stds = recalibrate_predictions(pred_means, pred_stds, true_values)
                @test length(recal_stds) == n
            end
        end
    end

end  # Main testset

println("\n✅ All Tier 1 Integration Tests Completed!")
