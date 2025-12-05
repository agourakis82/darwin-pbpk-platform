"""
Test Suite for Sensitivity Analysis Module

Tests all local and global sensitivity analysis methods with example models.

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

using Test
using DarwinPBPK
using Statistics
using Random

# ============================================================================
# TEST MODELS
# ============================================================================

"""
Simple linear model for testing: y = 2x₁ + 3x₂ + 0.1x₃
"""
function linear_model(params::Dict{String, Float64})
    x1 = params["x1"]
    x2 = params["x2"]
    x3 = params["x3"]

    return Dict(
        "y" => 2.0 * x1 + 3.0 * x2 + 0.1 * x3,
        "z" => x1^2 + x2
    )
end

"""
Non-linear model with interactions: y = x₁² + x₁·x₂ + x₃
"""
function nonlinear_model(params::Dict{String, Float64})
    x1 = params["x1"]
    x2 = params["x2"]
    x3 = params["x3"]

    return Dict(
        "y" => x1^2 + x1 * x2 + x3,
        "z" => sin(x1) + cos(x2)
    )
end

"""
Ishigami function - standard SA benchmark with known Sobol indices
y = sin(x1) + a·sin²(x2) + b·x3⁴·sin(x1)
Standard: a=7, b=0.1, x ∈ [-π, π]³
"""
function ishigami_model(params::Dict{String, Float64})
    x1 = params["x1"]
    x2 = params["x2"]
    x3 = params["x3"]
    a = 7.0
    b = 0.1

    y = sin(x1) + a * sin(x2)^2 + b * x3^4 * sin(x1)

    return Dict("y" => y)
end

# ============================================================================
# TESTS: DATA STRUCTURES
# ============================================================================

@testset "Data Structures" begin
    # ParameterRange
    @testset "ParameterRange" begin
        pr = ParameterRange("test", 1.0, 0.5, 1.5)
        @test pr.name == "test"
        @test pr.nominal == 1.0
        @test pr.min == 0.5
        @test pr.max == 1.5
        @test pr.distribution == :uniform

        # Test validation
        @test_throws AssertionError ParameterRange("test", 1.0, 1.5, 0.5)  # min > max
        @test_throws AssertionError ParameterRange("test", 2.0, 0.5, 1.5)  # nominal outside range
    end

    # SensitivityConfig
    @testset "SensitivityConfig" begin
        config = SensitivityConfig(method=:sobol, n_samples=512)
        @test config.method == :sobol
        @test config.n_samples == 512
        @test config.seed == 42
    end
end

# ============================================================================
# TESTS: SAMPLING FUNCTIONS
# ============================================================================

@testset "Sampling Functions" begin
    Random.seed!(42)

    params = [
        ParameterRange("x1", 1.0, 0.0, 2.0),
        ParameterRange("x2", 0.5, 0.0, 1.0),
        ParameterRange("x3", 2.0, 1.0, 3.0)
    ]

    @testset "Latin Hypercube Sampling" begin
        n = 100
        samples = latin_hypercube_sample(params, n)

        @test size(samples) == (n, 3)

        # Check bounds
        @test all(0.0 .<= samples[:, 1] .<= 2.0)
        @test all(0.0 .<= samples[:, 2] .<= 1.0)
        @test all(1.0 .<= samples[:, 3] .<= 3.0)

        # Check space-filling property (approximate)
        # Each parameter should have roughly uniform coverage
        for j in 1:3
            sorted_vals = sort(samples[:, j])
            gaps = diff(sorted_vals)
            # Max gap should be reasonable
            @test maximum(gaps) < 0.15 * (params[j].max - params[j].min)
        end
    end

    @testset "Morris Trajectories" begin
        r = 5
        levels = 4
        trajectories = morris_trajectories(params, r, levels)

        @test length(trajectories) == r

        for traj in trajectories
            @test size(traj) == (4, 3)  # (p+1) × p

            # Check bounds
            @test all(0.0 .<= traj[:, 1] .<= 2.0)
            @test all(0.0 .<= traj[:, 2] .<= 1.0)
            @test all(1.0 .<= traj[:, 3] .<= 3.0)

            # Check that one parameter changes at each step
            for i in 1:3
                changes = traj[i+1, :] .!= traj[i, :]
                @test sum(changes) >= 1  # At least one change
            end
        end
    end
end

# ============================================================================
# TESTS: LOCAL SENSITIVITY ANALYSIS
# ============================================================================

@testset "Local Sensitivity Analysis" begin
    params = [
        ParameterRange("x1", 1.0, 0.5, 1.5),
        ParameterRange("x2", 1.0, 0.5, 1.5),
        ParameterRange("x3", 1.0, 0.5, 1.5)
    ]
    outputs = ["y", "z"]

    @testset "One-At-a-Time (OAT)" begin
        result = one_at_a_time_sensitivity(linear_model, params, outputs, perturbation=0.01)

        @test result.method == :oat
        @test length(result.parameters) == 3
        @test length(result.outputs) == 2

        # For linear model y = 2x₁ + 3x₂ + 0.1x₃
        # Normalized sensitivity: (∂y/∂xi)·(xi/y) at xi=1, y=5.1
        # S1 ≈ 2·(1/5.1) ≈ 0.392
        # S2 ≈ 3·(1/5.1) ≈ 0.588
        # S3 ≈ 0.1·(1/5.1) ≈ 0.020
        sens_y = result.sensitivities["y"]
        @test haskey(sens_y, "x1")
        @test haskey(sens_y, "x2")
        @test haskey(sens_y, "x3")

        # x2 should be most sensitive (coefficient = 3)
        @test abs(sens_y["x2"]) > abs(sens_y["x1"])
        @test abs(sens_y["x1"]) > abs(sens_y["x3"])

        # Rankings
        @test length(result.rankings["y"]) == 3
        @test result.rankings["y"][1][1] == "x2"  # Most important
    end

    @testset "Elasticity Coefficient" begin
        params_dict = Dict("x1" => 1.0, "x2" => 1.0, "x3" => 1.0)

        elast = calculate_elasticity(linear_model, params_dict, "x1", "y")

        # For y = 2x₁ + 3x₂ + 0.1x₃ at (1,1,1): y=5.1
        # E = (∂y/∂x₁)·(x₁/y) = 2·(1/5.1) ≈ 0.392
        @test elast ≈ 2.0 / 5.1 atol=1e-3
    end

    @testset "Normalized Sensitivity Coefficient" begin
        params_dict = Dict("x1" => 1.0, "x2" => 1.0, "x3" => 1.0)

        nsc = normalized_sensitivity_coefficient(linear_model, params_dict, "x2", "y")

        # For linear model, NSC ≈ elasticity
        @test nsc ≈ 3.0 / 5.1 atol=1e-2
    end
end

# ============================================================================
# TESTS: GLOBAL SENSITIVITY ANALYSIS - SOBOL
# ============================================================================

@testset "Sobol Sensitivity Analysis" begin
    params = [
        ParameterRange("x1", 1.0, 0.5, 1.5),
        ParameterRange("x2", 1.0, 0.5, 1.5),
        ParameterRange("x3", 1.0, 0.5, 1.5)
    ]
    outputs = ["y"]

    @testset "Linear Model Sobol" begin
        result = sobol_sensitivity(linear_model, params, outputs, n_samples=256, seed=42)

        @test result.method == :sobol
        @test haskey(result.indices, "S1")
        @test haskey(result.indices, "ST")

        S1 = result.indices["S1"]["y"]
        ST = result.indices["ST"]["y"]

        @test haskey(S1, "x1")
        @test haskey(S1, "x2")
        @test haskey(S1, "x3")

        # For linear model without interactions: S1 ≈ ST
        @test S1["x1"] ≈ ST["x1"] atol=0.2
        @test S1["x2"] ≈ ST["x2"] atol=0.2

        # x2 should have highest sensitivity (largest coefficient)
        @test S1["x2"] > S1["x1"]
        @test S1["x1"] > S1["x3"]

        # Sum of S1 should be ≈ 1 for additive model
        @test sum(values(S1)) ≈ 1.0 atol=0.3
    end

    @testset "Nonlinear Model Sobol" begin
        result = sobol_sensitivity(nonlinear_model, params, outputs, n_samples=256, seed=42)

        S1 = result.indices["S1"]["y"]
        ST = result.indices["ST"]["y"]

        # With interactions: ST > S1 (especially for x1, x2 which interact)
        @test ST["x1"] >= S1["x1"]
        @test ST["x2"] >= S1["x2"]

        # Rankings should exist
        @test length(result.rankings["y"]) == 3
    end
end

# ============================================================================
# TESTS: GLOBAL SENSITIVITY ANALYSIS - MORRIS
# ============================================================================

@testset "Morris Screening" begin
    params = [
        ParameterRange("x1", 1.0, 0.5, 1.5),
        ParameterRange("x2", 1.0, 0.5, 1.5),
        ParameterRange("x3", 1.0, 0.5, 1.5)
    ]
    outputs = ["y"]

    @testset "Linear Model Morris" begin
        result = morris_screening(linear_model, params, outputs, n_trajectories=10, seed=42)

        @test result.method == :morris
        @test haskey(result.indices, "μ_star")
        @test haskey(result.indices, "σ")

        μ_star = result.indices["μ_star"]["y"]
        σ = result.indices["σ"]["y"]

        @test haskey(μ_star, "x1")
        @test haskey(μ_star, "x2")
        @test haskey(μ_star, "x3")

        # x2 should have highest μ*
        @test μ_star["x2"] > μ_star["x1"]
        @test μ_star["x1"] > μ_star["x3"]

        # For linear model, σ should be small (no interactions)
        @test σ["x1"] < μ_star["x1"]
        @test σ["x2"] < μ_star["x2"]
    end

    @testset "Nonlinear Model Morris" begin
        result = morris_screening(nonlinear_model, params, outputs, n_trajectories=10, seed=42)

        μ_star = result.indices["μ_star"]["y"]
        σ = result.indices["σ"]["y"]

        # With interactions/nonlinearity, σ may be larger
        @test all(values(σ) .>= 0)

        # Metadata check
        @test result.metadata["n_trajectories"] == 10
        @test result.metadata["n_evaluations"] == 10 * 4  # r * (p+1)
    end
end

# ============================================================================
# TESTS: GLOBAL SENSITIVITY ANALYSIS - PRCC
# ============================================================================

@testset "PRCC Analysis" begin
    params = [
        ParameterRange("x1", 1.0, 0.5, 1.5),
        ParameterRange("x2", 1.0, 0.5, 1.5),
        ParameterRange("x3", 1.0, 0.5, 1.5)
    ]
    outputs = ["y"]

    @testset "Linear Model PRCC" begin
        result = prcc_analysis(linear_model, params, outputs, n_samples=500, seed=42)

        @test result.method == :prcc
        @test haskey(result.indices, "PRCC")

        prcc = result.indices["PRCC"]["y"]

        @test haskey(prcc, "x1")
        @test haskey(prcc, "x2")
        @test haskey(prcc, "x3")

        # PRCC values should be in [-1, 1]
        @test all(-1.0 .<= values(prcc) .<= 1.0)

        # x2 should have highest PRCC (strongest positive correlation)
        @test abs(prcc["x2"]) > abs(prcc["x1"])
        @test abs(prcc["x1"]) > abs(prcc["x3"])

        # All should be positive for additive positive model
        @test prcc["x1"] > 0
        @test prcc["x2"] > 0
        @test prcc["x3"] > 0
    end
end

# ============================================================================
# TESTS: OUTPUT ANALYSIS FUNCTIONS
# ============================================================================

@testset "Output Analysis" begin
    params = [
        ParameterRange("x1", 1.0, 0.5, 1.5),
        ParameterRange("x2", 1.0, 0.5, 1.5),
        ParameterRange("x3", 1.0, 0.5, 1.5)
    ]
    outputs = ["y", "z"]

    result = one_at_a_time_sensitivity(linear_model, params, outputs)

    @testset "Rank Parameters" begin
        rankings = rank_parameters(result)

        @test haskey(rankings, "y")
        @test haskey(rankings, "z")
        @test length(rankings["y"]) == 3

        # First ranked should be x2 for y
        @test rankings["y"][1][1] == "x2"
    end

    @testset "Identify Influential Parameters" begin
        influential = identify_influential_parameters(result, 0.3)

        @test haskey(influential, "y")
        @test haskey(influential, "z")

        # x2 should be influential for y
        @test "x2" in influential["y"]
    end

    @testset "Tornado Plot Data" begin
        tornado_data = sensitivity_tornado_plot_data(result, "y")

        @test length(tornado_data) == 3
        @test tornado_data[1][1] == "x2"  # Most important first

        # Sorted by absolute value
        @test abs(tornado_data[1][2]) >= abs(tornado_data[2][2])
        @test abs(tornado_data[2][2]) >= abs(tornado_data[3][2])
    end

    @testset "Heatmap Data" begin
        heatmap = sensitivity_heatmap_data(result)

        @test length(heatmap.parameters) == 3
        @test length(heatmap.outputs) == 2
        @test size(heatmap.matrix) == (3, 2)

        # Check values match sensitivities
        for (i, param) in enumerate(heatmap.parameters)
            for (j, out) in enumerate(heatmap.outputs)
                @test heatmap.matrix[i, j] == result.sensitivities[out][param]
            end
        end
    end
end

# ============================================================================
# TESTS: COAGULATION MODEL INTEGRATION
# ============================================================================

@testset "Coagulation Model Integration" begin
    @testset "Default Coagulation Parameters" begin
        params = default_coagulation_parameters()

        @test length(params) == 8
        @test params[1].name == "II"
        @test params[1].nominal == 1400.0

        # Check ranges are ±50%
        for p in params
            @test p.min ≈ p.nominal * 0.5 atol=1e-6
            @test p.max ≈ p.nominal * 1.5 atol=1e-6
        end
    end

    @testset "Coagulation Sensitivity Wrapper" begin
        params_dict = Dict(
            "II" => 1400.0,
            "V" => 22.0,
            "VII" => 10.0,
            "VIII" => 0.7,
            "IX" => 90.0,
            "X" => 170.0,
            "ATIII" => 2400.0,
            "TFPI" => 2.5
        )

        result = coagulation_sensitivity_wrapper(params_dict)

        @test haskey(result, "peak_thrombin")
        @test haskey(result, "lag_time")
        @test haskey(result, "ttp")
        @test haskey(result, "etp")

        @test result["peak_thrombin"] > 0
        @test result["lag_time"] > 0
        @test result["ttp"] > result["lag_time"]
        @test result["etp"] > 0
    end

    @testset "Coagulation OAT Sensitivity" begin
        params = default_coagulation_parameters()
        outputs = ["peak_thrombin", "lag_time", "etp"]

        result = one_at_a_time_sensitivity(
            coagulation_sensitivity_wrapper,
            params,
            outputs,
            perturbation=0.05
        )

        @test result.method == :oat
        @test length(result.parameters) == 8
        @test length(result.outputs) == 3

        # Factor II (prothrombin) should be highly influential
        @test haskey(result.sensitivities["peak_thrombin"], "II")
        @test abs(result.sensitivities["peak_thrombin"]["II"]) > 0.1

        # Rankings should work
        @test length(result.rankings["peak_thrombin"]) == 8
    end
end

# ============================================================================
# TESTS: ISHIGAMI BENCHMARK
# ============================================================================

@testset "Ishigami Benchmark" begin
    # Ishigami function: known analytical Sobol indices
    # S1_x1 = 0.3139, S1_x2 = 0.4424, S1_x3 = 0
    # ST_x1 = 0.5576, ST_x2 = 0.4424, ST_x3 = 0.2437

    params = [
        ParameterRange("x1", 0.0, -π, π),
        ParameterRange("x2", 0.0, -π, π),
        ParameterRange("x3", 0.0, -π, π)
    ]
    outputs = ["y"]

    @testset "Ishigami Sobol" begin
        result = sobol_sensitivity(ishigami_model, params, outputs, n_samples=2048, seed=123)

        S1 = result.indices["S1"]["y"]
        ST = result.indices["ST"]["y"]

        # Check approximate agreement with analytical values
        # Allow generous tolerance for finite sample size
        @test S1["x1"] ≈ 0.3139 atol=0.15
        @test S1["x2"] ≈ 0.4424 atol=0.15
        @test S1["x3"] ≈ 0.0 atol=0.10  # Should be near zero

        @test ST["x1"] ≈ 0.5576 atol=0.20
        @test ST["x2"] ≈ 0.4424 atol=0.15
        @test ST["x3"] ≈ 0.2437 atol=0.15

        # x3 has no first-order effect but total effect due to interaction
        @test ST["x3"] > S1["x3"]
    end
end

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@testset "Performance" begin
    params = [
        ParameterRange("x$i", 1.0, 0.5, 1.5) for i in 1:10
    ]

    function complex_model(p::Dict{String, Float64})
        y = sum(p["x$i"]^2 for i in 1:10)
        return Dict("y" => y)
    end

    @testset "OAT Performance" begin
        @test (@timed one_at_a_time_sensitivity(
            complex_model, params, ["y"]
        )).time < 1.0  # Should be fast
    end

    @testset "Morris Performance" begin
        @test (@timed morris_screening(
            complex_model, params, ["y"], n_trajectories=5
        )).time < 5.0
    end
end

println("\n" * "="^70)
println("Sensitivity Analysis Test Suite Complete")
println("="^70)
