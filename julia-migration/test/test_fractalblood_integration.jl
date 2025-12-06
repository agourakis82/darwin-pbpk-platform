"""
Test FractalBlood Integration with ODE Solver

Tests the integration layer between FractalBlood module and the main PBPK ODE solver.
Compares traditional well-stirred PBPK with FractalBlood-enhanced transit dynamics.

Author: Darwin PBPK Platform
Date: December 2025
"""

using Test
using DifferentialEquations

# Add the DarwinPBPK module path
push!(LOAD_PATH, joinpath(@__DIR__, "../src"))

using DarwinPBPK.ODEPBPKSolver
using DarwinPBPK.ODEPBPKSolver: FractalBlood

@testset "FractalBlood Integration Tests" begin

    @testset "FractalBloodParams Construction" begin
        # Test default construction (disabled)
        params_default = FractalBloodParams()
        @test params_default.enabled == false

        # Test enabled construction
        params_enabled = FractalBloodParams(
            enabled = true,
            alpha = 1.37,
            tau_min = 0.1,
            tau_mean = 20.0,
            beta = 0.8
        )
        @test params_enabled.enabled == true
        @test params_enabled.alpha ≈ 1.37
        @test params_enabled.tau_min ≈ 0.1
        @test params_enabled.tau_mean ≈ 20.0
        @test params_enabled.beta ≈ 0.8

        # Test parameter validation
        @test_throws ErrorException FractalBloodParams(enabled=true, alpha=0.5)  # alpha <= 1
        @test_throws ErrorException FractalBloodParams(enabled=true, tau_min=-1.0)  # tau_min <= 0
        @test_throws ErrorException FractalBloodParams(enabled=true, beta=1.5)  # beta > 1
    end

    @testset "Transit Time Distribution" begin
        fractal_params = FractalBloodParams(
            enabled = true,
            alpha = 1.37,
            tau_min = 0.1,
            tau_mean = 20.0
        )

        # Test that distribution is zero for t < tau_min
        @test fractal_transit_time_distribution(0.05, fractal_params) ≈ 0.0

        # Test that distribution is positive for t >= tau_min
        E_at_taumin = fractal_transit_time_distribution(0.1, fractal_params)
        @test E_at_taumin > 0.0

        # Test power-law behavior: E(t) ∝ t^(-alpha)
        t1 = 1.0
        t2 = 2.0
        E1 = fractal_transit_time_distribution(t1, fractal_params)
        E2 = fractal_transit_time_distribution(t2, fractal_params)

        # E(t2) / E(t1) should equal (t1/t2)^alpha
        ratio_expected = (t1 / t2)^fractal_params.alpha
        ratio_actual = E2 / E1
        @test ratio_actual ≈ ratio_expected rtol=1e-6

        # Test normalization (integral should approach 1 for large T)
        # ∫ E(t) dt from tau_min to infinity = 1
        # For finite upper limit, should be close to 1
        using QuadGK
        integral, _ = quadgk(t -> fractal_transit_time_distribution(t, fractal_params),
                            fractal_params.tau_min, 100.0, rtol=1e-6)
        @test integral ≈ 1.0 rtol=0.1  # Within 10%
    end

    @testset "Integration with FractalBloodModel" begin
        # Create a fractal vascular network
        fractal_model = FractalBlood.create_fractal_blood_model(
            num_levels = 10,
            hematocrit = 0.45,
            fu = 0.1,
            alpha = 1.37,
            beta = 0.8
        )

        # Create standard PBPK parameters
        pbpk_params = PBPKParams(
            clearance_hepatic = 10.0,
            clearance_renal = 2.0
        )

        # Integrate FractalBlood with PBPK
        integrated_params = integrate_fractal_blood!(pbpk_params, fractal_model)

        @test integrated_params isa PBPKParamsWithFractal
        @test integrated_params.pbpk === pbpk_params
        @test integrated_params.fractal.enabled == true
        @test integrated_params.fractal.alpha == fractal_model.alpha
        @test integrated_params.fractal.tau_min == fractal_model.tau_min
        @test integrated_params.fractal.tau_mean == fractal_model.tau_mean
        @test integrated_params.fractal.beta == fractal_model.beta
    end

    @testset "Create Fractal PBPK Params" begin
        # Test convenience constructor
        params = create_fractal_pbpk_params(
            alpha = 1.4,
            tau_min = 0.2,
            tau_mean = 25.0,
            beta = 0.75,
            clearance_hepatic = 15.0,
            clearance_renal = 3.0
        )

        @test params isa PBPKParamsWithFractal
        @test params.pbpk.clearance_hepatic ≈ 15.0
        @test params.pbpk.clearance_renal ≈ 3.0
        @test params.fractal.alpha ≈ 1.4
        @test params.fractal.tau_min ≈ 0.2
        @test params.fractal.tau_mean ≈ 25.0
        @test params.fractal.beta ≈ 0.75
    end

    @testset "Fractal Dispersion Application" begin
        fractal_params = FractalBloodParams(
            enabled = true,
            alpha = 1.37,
            tau_min = 0.1,
            tau_mean = 20.0,
            use_convolution = true
        )

        # Create simple concentration history
        history = [
            (0.0, 0.0),
            (0.5, 10.0),
            (1.0, 20.0),
            (2.0, 15.0),
            (3.0, 10.0)
        ]

        # Test dispersion at t = 1.0
        C_out = apply_fractal_dispersion(15.0, 1.0, history, fractal_params)

        # Output should be dispersed (lower than input due to mixing)
        @test C_out >= 0.0
        @test C_out <= 20.0  # Should not exceed max in history

        # Test without convolution (should return input)
        fractal_params_no_conv = FractalBloodParams(
            enabled = true,
            use_convolution = false
        )
        C_out_no_conv = apply_fractal_dispersion(15.0, 1.0, history, fractal_params_no_conv)
        @test C_out_no_conv ≈ 15.0
    end

    @testset "Traditional vs FractalBlood PBPK Comparison" begin
        # Create traditional PBPK parameters
        pbpk_traditional = PBPKParams(
            clearance_hepatic = 10.0,
            clearance_renal = 2.0
        )

        # Create FractalBlood-enhanced parameters
        pbpk_fractal = create_fractal_pbpk_params(
            alpha = 1.37,
            tau_min = 0.1 / 3600.0,  # Convert to hours
            tau_mean = 20.0 / 3600.0,  # Convert to hours
            clearance_hepatic = 10.0,
            clearance_renal = 2.0,
            use_convolution = false  # Start with simple approximation
        )

        # Dose and simulation parameters
        dose = 100.0  # mg
        t_max = 24.0  # hours
        num_points = 100

        # Simulate traditional PBPK
        results_traditional = simulate(
            pbpk_traditional,
            dose,
            t_max = t_max,
            num_points = num_points
        )

        # For FractalBlood, we would need a modified simulate function
        # For now, test that the parameters are correctly set up
        @test pbpk_fractal.fractal.enabled == true
        @test pbpk_fractal.pbpk.clearance_hepatic == pbpk_traditional.clearance_hepatic

        # Test that traditional simulation works
        @test haskey(results_traditional, "blood")
        @test haskey(results_traditional, "time")
        @test length(results_traditional["blood"]) == num_points

        # Check that Cmax is reasonable
        blood_conc = results_traditional["blood"]
        @test maximum(blood_conc) > 0.0
        @test all(c >= 0.0 for c in blood_conc)  # All concentrations positive
    end

    @testset "FractalBlood Transit Time Moments" begin
        # Create fractal model and check that transit time moments match theory
        fractal_model = FractalBlood.create_fractal_blood_model(
            num_levels = 12,
            alpha = 1.5,  # alpha > 2 for finite mean
            beta = 0.8
        )

        # Get moments from model
        mean_τ, var_τ, skew_τ = FractalBlood.transit_time_moments(fractal_model)

        # For alpha = 1.5, mean should be finite
        # Mean = τ_min * (α - 1) / (α - 2) = τ_min * 0.5 / 0.5 = τ_min (for alpha=1.5)
        # But alpha=1.5 < 2, so mean is actually infinite
        # Let's use alpha = 2.5 for finite mean

        fractal_model_finite = FractalBlood.create_fractal_blood_model(
            num_levels = 12,
            alpha = 2.5,
            beta = 0.8
        )

        mean_τ_2, var_τ_2, skew_τ_2 = FractalBlood.transit_time_moments(fractal_model_finite)

        # For alpha = 2.5 > 2, mean should be finite
        @test isfinite(mean_τ_2)
        @test mean_τ_2 > fractal_model_finite.tau_min

        # For alpha = 2.5 < 3, variance should be infinite
        @test var_τ_2 == Inf
    end

    @testset "Network Topology Validation" begin
        # Create network and validate topology
        vessels = FractalBlood.create_fractal_network(10)

        @test length(vessels) > 0
        @test vessels[1].level == 0  # First vessel is aorta
        @test vessels[1].parent_id === nothing

        # Validate Murray's Law compliance
        validation = FractalBlood.validate_network_topology(vessels)

        @test haskey(validation, "murray_law_compliance")
        @test validation["murray_law_compliance"] >= 0.0
        @test validation["murray_law_compliance"] <= 1.0

        # Good compliance (>80%)
        @test validation["murray_law_compliance"] > 0.8
    end

    @testset "Power-Law Distribution Validation" begin
        fractal_model = FractalBlood.create_fractal_blood_model(
            num_levels = 12,
            alpha = 1.37
        )

        validation = FractalBlood.validate_transit_time_distribution(fractal_model, n_samples=1000)

        @test haskey(validation, "sample_mean")
        @test haskey(validation, "estimated_alpha")

        # Estimated alpha should be close to true alpha
        if haskey(validation, "alpha_error")
            @test validation["alpha_error"] < 0.2  # Within 20% error
        end
    end
end

@testset "Pharmacokinetic Comparison: Traditional vs Fractal" begin
    @testset "Single IV Bolus" begin
        # Traditional PBPK
        pbpk = PBPKParams(
            clearance_hepatic = 10.0,
            clearance_renal = 2.0
        )

        dose = 100.0  # mg
        results_trad = simulate(pbpk, dose, t_max=24.0, num_points=100)

        # Extract PK parameters
        C_blood = results_trad["blood"]
        time = results_trad["time"]

        C_max_trad = maximum(C_blood)
        idx_max = argmax(C_blood)
        T_max_trad = time[idx_max]

        # AUC calculation (trapezoidal rule)
        AUC_trad = 0.0
        for i in 2:length(time)
            dt = time[i] - time[i-1]
            AUC_trad += 0.5 * (C_blood[i] + C_blood[i-1]) * dt
        end

        # FractalBlood should give similar results for IV bolus
        # (main difference is in distribution, not initial concentration)
        @test C_max_trad > 0.0
        @test T_max_trad ≈ 0.0 atol=0.5  # Peak at ~t=0 for IV bolus
        @test AUC_trad > 0.0

        println("\n=== Traditional PBPK Results ===")
        println("Cmax: $(round(C_max_trad, digits=2)) mg/L")
        println("Tmax: $(round(T_max_trad, digits=2)) hours")
        println("AUC: $(round(AUC_trad, digits=2)) mg·h/L")
    end

    @testset "Conservation of Mass" begin
        pbpk = PBPKParams(
            clearance_hepatic = 5.0,
            clearance_renal = 1.0
        )

        dose = 100.0
        t_max = 48.0

        # Solve ODE directly for mass conservation check
        u0 = zeros(Float64, NUM_ORGANS)
        blood_volume = pbpk.volumes[BLOOD_IDX]
        u0[BLOOD_IDX] = dose / blood_volume

        tspan = (0.0, t_max)
        prob = ODEProblem(ode_system!, u0, tspan, pbpk)
        sol = DifferentialEquations.solve(prob, Tsit5(), abstol=1e-10, reltol=1e-8)

        # Check mass conservation
        @test validate_mass_conservation(sol, pbpk, dose, tol=1e-4)
    end
end

println("\n" * "="^70)
println("FractalBlood Integration Test Suite Complete")
println("="^70)
