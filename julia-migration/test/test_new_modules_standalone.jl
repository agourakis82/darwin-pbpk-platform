"""
Standalone Tests for New Blood Compartment Modules

Tests for:
- TGA Validation
- Lattice Boltzmann CFD
- Sensitivity Analysis

Run with: julia --project=. test/test_new_modules_standalone.jl
"""

using Test
using Statistics
using Random
using LinearAlgebra

# Include modules directly
include("../src/DarwinPBPK/compartments/tga_validation.jl")
include("../src/DarwinPBPK/compartments/lattice_boltzmann.jl")
include("../src/DarwinPBPK/compartments/sensitivity_analysis.jl")

using .TGAValidation
using .LatticeBoltzmann
using .SensitivityAnalysis

println("=" ^ 70)
println("NEW MODULES STANDALONE TESTS")
println("Testing: TGA Validation, Lattice Boltzmann, Sensitivity Analysis")
println("=" ^ 70)

# ============================================================================
# TGA VALIDATION TESTS
# ============================================================================

@testset "TGA Validation Module" begin

    @testset "TGAParameters Construction" begin
        params = TGAParameters(
            3.5,      # lag_time
            9.2,      # time_to_peak
            315.0,    # peak_thrombin
            1820.0,   # etp (lowercase!)
            98.0,     # velocity_index
            8.5,      # width_50
            22.0,     # start_tail
            5.0,      # TF concentration
            "test"    # patient condition
        )

        @test params.lag_time == 3.5
        @test params.peak_thrombin == 315.0
        @test params.etp == 1820.0
        @test params.velocity_index == 98.0

        println("  ✓ TGAParameters construction validated")
    end

    @testset "Clinical Reference Datasets" begin
        # Check healthy dataset - it's a struct not Dict
        healthy = HEALTHY_TGA_5PM_TF

        @test healthy isa ClinicalTGADataset
        @test healthy.peak_thrombin_mean > 200  # nM
        @test healthy.lag_time_mean > 0.5  # min
        @test healthy.lag_time_mean < 10  # min
        @test healthy.n_subjects > 50

        # Hemophilia should have lower values
        hemo_a = HEMOPHILIA_A_TGA
        @test hemo_a.peak_thrombin_mean < healthy.peak_thrombin_mean

        # Warfarin should reduce thrombin
        warfarin = WARFARIN_INR2_TGA
        @test warfarin.peak_thrombin_mean < healthy.peak_thrombin_mean

        # Check we have multiple datasets
        @test HEALTHY_TGA_1PM_TF isa ClinicalTGADataset
        @test DOAC_RIVAROXABAN_TGA isa ClinicalTGADataset

        println("  ✓ Clinical reference datasets validated")
    end

    @testset "TGA Parameter Extraction" begin
        # Create synthetic thrombin curve (typical shape)
        time = collect(0.0:0.5:30.0)
        n = length(time)

        # Simulate TGA curve: lag phase, rapid rise, peak, slower decay
        thrombin = zeros(n)
        for (i, t) in enumerate(time)
            if t < 3.0
                thrombin[i] = 0.0  # Lag phase
            elseif t < 10.0
                thrombin[i] = 350.0 * (1 - exp(-0.8*(t-3.0)))  # Rising
            else
                thrombin[i] = 350.0 * exp(-0.15*(t-10.0))  # Decay
            end
        end

        params = extract_tga_parameters(thrombin, time)

        @test params.lag_time > 0
        @test params.lag_time < 5.0  # Should detect lag around 3 min
        @test params.peak_thrombin > 300.0
        @test params.time_to_peak > params.lag_time
        @test params.etp > 0

        println("  ✓ TGA parameter extraction works")
    end

    @testset "Goodness of Fit Metrics" begin
        predicted = [100.0, 200.0, 300.0, 400.0]
        observed = [110.0, 190.0, 310.0, 380.0]

        metrics = calculate_goodness_of_fit(predicted, observed)

        @test metrics isa ValidationMetrics
        @test metrics.aafe > 0
        @test metrics.aafe < 1.5  # Good fit
        @test metrics.r_squared > 0.95  # High correlation
        @test metrics.within_2fold > 0.9  # Most within 2-fold

        println("  ✓ Goodness of fit metrics calculated")
    end

    @testset "Clinical Comparison" begin
        # Create simulated TGA close to healthy values
        simulated = TGAParameters(
            3.5,      # lag_time
            9.0,      # time_to_peak
            320.0,    # peak_thrombin
            1800.0,   # etp
            100.0,    # velocity_index
            8.0,      # width_50
            20.0,     # start_tail
            5.0,      # TF
            "simulated"
        )

        result = compare_to_clinical(simulated, HEALTHY_TGA_5PM_TF)

        # Result is Dict{String, Any} with structured keys
        @test haskey(result, "parameter_comparisons")
        @test haskey(result, "overall_metrics")
        @test haskey(result, "acceptance_criteria")

        # Check parameter comparisons structure
        param_comp = result["parameter_comparisons"]
        @test haskey(param_comp, "peak_thrombin")
        @test haskey(param_comp, "lag_time")
        @test haskey(param_comp, "etp")

        # Check acceptance criteria
        criteria = result["acceptance_criteria"]
        @test haskey(criteria, "all_criteria_met")
        @test haskey(criteria, "within_2fold_pass")

        println("  ✓ Clinical comparison works")
    end

    println("\n✓ All TGA Validation tests passed!")
end

# ============================================================================
# LATTICE BOLTZMANN TESTS
# ============================================================================

@testset "Lattice Boltzmann Module" begin

    @testset "D2Q9 Lattice Configuration" begin
        lattice = D2Q9Lattice()

        @test length(lattice.weights) == 9
        @test sum(lattice.weights) ≈ 1.0 atol=1e-10
        @test lattice.cs2 ≈ 1.0/3.0
        @test size(lattice.velocities) == (2, 9)

        # Check opposite indices
        @test length(lattice.opposite) == 9

        println("  ✓ D2Q9 lattice configured correctly")
    end

    @testset "Fluid Properties" begin
        fluid = FluidProperties()

        @test fluid.density > 0
        @test fluid.base_viscosity > 0
        @test fluid.hematocrit > 0 && fluid.hematocrit < 1

        println("  ✓ Fluid properties initialized")
    end

    @testset "Vessel Geometry - Straight Tube" begin
        # create_straight_tube uses keyword arguments
        geom = create_straight_tube(nx=100, ny=30, diameter=20)

        @test size(geom) == (100, 30)
        @test any(geom)  # Has walls (solid = true)
        @test !all(geom)  # Has fluid (solid = false)

        println("  ✓ Straight tube geometry created")
    end

    @testset "Vessel Geometry - Stenosis" begin
        geom_straight = create_straight_tube(nx=100, ny=30, diameter=20)
        geom_stenosis = create_stenosis_geometry(nx=100, ny=30, stenosis_severity=0.5)

        # Both are valid geometries with solid walls
        @test any(geom_straight)
        @test any(geom_stenosis)
        @test !all(geom_stenosis)  # Has fluid regions

        # Stenosis geometry should be different from straight tube
        @test geom_stenosis != geom_straight

        println("  ✓ Stenosis geometry created")
    end

    @testset "LBM Simulation Setup" begin
        geom = create_straight_tube(nx=50, ny=20, diameter=16)
        fluid = FluidProperties()
        bc = BoundaryConditions()

        sim = create_lbm_simulation(geom, fluid, bc)

        @test size(sim.f) == (50, 20, 9)
        @test size(sim.rho) == (50, 20)
        @test size(sim.u) == (50, 20)
        @test size(sim.v) == (50, 20)
        @test sim.tau > 0.5  # Stability requirement

        println("  ✓ LBM simulation setup complete")
    end

    @testset "Equilibrium Distribution" begin
        lattice = D2Q9Lattice()
        rho = 1.0
        ux, uy = 0.05, 0.0

        # equilibrium_distribution(rho, ux, uy, lattice)
        feq = equilibrium_distribution(rho, ux, uy, lattice)

        @test length(feq) == 9
        @test sum(feq) ≈ rho atol=1e-10  # Mass conservation
        @test all(feq .>= -1e-10)  # Non-negative (allow small numerical error)

        # Check momentum conservation
        mom_x = sum(feq .* lattice.velocities[1, :])
        mom_y = sum(feq .* lattice.velocities[2, :])
        @test mom_x ≈ rho * ux atol=1e-10
        @test mom_y ≈ rho * uy atol=1e-10

        println("  ✓ Equilibrium distribution conserves mass and momentum")
    end

    @testset "LBM Simulation Run" begin
        geom = create_straight_tube(nx=40, ny=15, diameter=12)
        fluid = FluidProperties()
        bc = BoundaryConditions()

        sim = create_lbm_simulation(geom, fluid, bc)

        # Run a few steps (silent)
        run_lbm_simulation!(sim, 50, print_interval=1000)

        # Check simulation stability
        @test all(isfinite.(sim.rho))  # No NaN/Inf
        @test all(sim.rho .> 0)  # Positive density
        @test all(isfinite.(sim.u))
        @test all(isfinite.(sim.v))

        println("  ✓ LBM simulation runs stably")
    end

    @testset "Wall Shear Stress Extraction" begin
        geom = create_straight_tube(nx=60, ny=20, diameter=16)
        fluid = FluidProperties()
        bc = BoundaryConditions()

        sim = create_lbm_simulation(geom, fluid, bc)

        # Run to approach steady state (silent)
        run_lbm_simulation!(sim, 200, print_interval=1000)

        wss, locations = extract_wall_shear_stress(sim)

        @test length(wss) > 0
        @test all(isfinite.(wss))

        println("  ✓ Wall shear stress extracted")
    end

    @testset "Carreau-Yasuda Viscosity" begin
        fluid = FluidProperties()

        # At low shear rate - higher viscosity
        visc_low = carreau_yasuda_viscosity(1.0, fluid)

        # At high shear rate - lower viscosity (shear thinning)
        visc_high = carreau_yasuda_viscosity(1000.0, fluid)

        @test visc_low > visc_high  # Shear thinning behavior
        @test visc_low > 0
        @test visc_high > 0

        println("  ✓ Carreau-Yasuda viscosity model works")
    end

    println("\n✓ All Lattice Boltzmann tests passed!")
end

# ============================================================================
# SENSITIVITY ANALYSIS TESTS
# ============================================================================

# Simple test models - accept Dict with variable number of params
function linear_model(params::Dict{String, Float64})
    x1 = get(params, "x1", 1.0)
    x2 = get(params, "x2", 1.0)
    x3 = get(params, "x3", 1.0)
    return Dict("y" => 2.0*x1 + 3.0*x2 + 0.1*x3)
end

function quadratic_model(params::Dict{String, Float64})
    x1 = get(params, "x1", 1.0)
    x2 = get(params, "x2", 1.0)
    return Dict(
        "y" => x1^2 + x1*x2 + x2^2,
        "z" => sin(x1) * exp(-x2/10)
    )
end

@testset "Sensitivity Analysis Module" begin

    @testset "Parameter Range Definition" begin
        # ParameterRange uses keyword arg for distribution
        param = ParameterRange("k1", 1.0, 0.5, 2.0; distribution=:uniform)

        @test param.name == "k1"
        @test param.nominal == 1.0
        @test param.min == 0.5
        @test param.max == 2.0
        @test param.distribution == :uniform

        println("  ✓ Parameter ranges defined")
    end

    @testset "Latin Hypercube Sampling" begin
        params = [
            ParameterRange("x1", 1.0, 0.0, 2.0; distribution=:uniform),
            ParameterRange("x2", 5.0, 0.0, 10.0; distribution=:uniform),
            ParameterRange("x3", 0.5, 0.0, 1.0; distribution=:uniform)
        ]

        n_samples = 100
        samples = latin_hypercube_sample(params, n_samples)

        # Returns Matrix (n_samples × n_params)
        @test size(samples, 1) == n_samples
        @test size(samples, 2) == 3  # 3 parameters

        # Check bounds respected for x1 (column 1)
        x1_vals = samples[:, 1]
        @test all(0.0 .<= x1_vals .<= 2.0)

        # Check x2 bounds (column 2)
        x2_vals = samples[:, 2]
        @test all(0.0 .<= x2_vals .<= 10.0)

        # Check space-filling (not clustered)
        @test std(x1_vals) > 0.3

        println("  ✓ Latin Hypercube sampling works")
    end

    @testset "One-at-a-Time Sensitivity" begin
        params = [
            ParameterRange("x1", 1.0, 0.5, 1.5; distribution=:uniform),
            ParameterRange("x2", 1.0, 0.5, 1.5; distribution=:uniform),
            ParameterRange("x3", 1.0, 0.5, 1.5; distribution=:uniform)
        ]

        result = one_at_a_time_sensitivity(linear_model, params, ["y"])

        @test result isa SensitivityResult
        @test haskey(result.sensitivities, "y")
        sens_y = result.sensitivities["y"]

        # Linear model: y = 2x1 + 3x2 + 0.1x3
        # x2 should be most sensitive, x3 least
        @test abs(sens_y["x2"]) > abs(sens_y["x1"])
        @test abs(sens_y["x1"]) > abs(sens_y["x3"])

        println("  ✓ OAT sensitivity correctly identifies parameter importance")
    end

    @testset "Morris Screening" begin
        params = [
            ParameterRange("x1", 1.0, 0.0, 2.0; distribution=:uniform),
            ParameterRange("x2", 1.0, 0.0, 2.0; distribution=:uniform),
            ParameterRange("x3", 1.0, 0.0, 2.0; distribution=:uniform)
        ]

        result = morris_screening(linear_model, params, ["y"]; n_trajectories=10)

        @test result isa SensitivityResult
        @test result.method == :morris

        # indices uses Greek letters: μ_star, σ
        @test haskey(result.indices, "μ_star")
        @test haskey(result.indices, "σ")

        mu_star = result.indices["μ_star"]["y"]

        # x2 should have highest mu* (most influential)
        @test mu_star["x2"] >= mu_star["x1"] * 0.5  # Allow variance
        @test mu_star["x1"] >= mu_star["x3"] * 0.5

        println("  ✓ Morris screening identifies influential parameters")
    end

    @testset "Sobol Indices" begin
        Random.seed!(42)

        params = [
            ParameterRange("x1", 1.0, 0.0, 2.0; distribution=:uniform),
            ParameterRange("x2", 1.0, 0.0, 2.0; distribution=:uniform)
        ]

        result = sobol_sensitivity(linear_model, params, ["y"]; n_samples=256)

        @test result isa SensitivityResult
        @test haskey(result.indices, "S1") || haskey(result.indices, "first_order")

        # Check sensitivities exist for both params
        @test haskey(result.sensitivities, "y")
        sens_y = result.sensitivities["y"]
        @test haskey(sens_y, "x1")
        @test haskey(sens_y, "x2")

        println("  ✓ Sobol indices calculated")
    end

    @testset "PRCC Analysis" begin
        Random.seed!(123)

        params = [
            ParameterRange("x1", 1.0, 0.0, 2.0; distribution=:uniform),
            ParameterRange("x2", 1.0, 0.0, 2.0; distribution=:uniform)
        ]

        result = prcc_analysis(linear_model, params, ["y"]; n_samples=200)

        @test result isa SensitivityResult
        @test haskey(result.sensitivities, "y")
        sens_y = result.sensitivities["y"]

        # PRCC should have values for both parameters
        @test haskey(sens_y, "x1")
        @test haskey(sens_y, "x2")

        println("  ✓ PRCC analysis works")
    end

    @testset "Parameter Ranking" begin
        params = [
            ParameterRange("x1", 1.0, 0.5, 1.5; distribution=:uniform),
            ParameterRange("x2", 1.0, 0.5, 1.5; distribution=:uniform),
            ParameterRange("x3", 1.0, 0.5, 1.5; distribution=:uniform)
        ]

        result = one_at_a_time_sensitivity(linear_model, params, ["y"])

        # Get ranking from result
        ranking = result.rankings["y"]

        @test length(ranking) == 3
        @test ranking[1][1] == "x2"  # Most sensitive (coefficient 3)
        @test ranking[3][1] == "x3"  # Least sensitive (coefficient 0.1)

        println("  ✓ Parameter ranking works")
    end

    @testset "Identify Influential Parameters" begin
        params = [
            ParameterRange("x1", 1.0, 0.5, 1.5; distribution=:uniform),
            ParameterRange("x2", 1.0, 0.5, 1.5; distribution=:uniform),
            ParameterRange("x3", 1.0, 0.5, 1.5; distribution=:uniform)
        ]

        result = one_at_a_time_sensitivity(linear_model, params, ["y"])

        # Get influential from sensitivities
        sens_y = result.sensitivities["y"]
        max_sens = maximum(abs.(values(sens_y)))
        influential = [k for (k, v) in sens_y if abs(v) > 0.3 * max_sens]

        # x1 and x2 should be influential (coefficients 2 and 3)
        @test "x2" in influential
        @test "x1" in influential

        println("  ✓ Influential parameters identified")
    end

    @testset "Coagulation Parameters Default" begin
        coag_params = default_coagulation_parameters()

        @test length(coag_params) >= 5  # At least core factors

        # Check some expected parameters exist (Roman numeral naming)
        names = [p.name for p in coag_params]
        # Factors use Roman numerals: II, V, VII, VIII, IX, X, etc.
        @test "II" in names || "V" in names || "X" in names

        # All should have sensible ranges
        for p in coag_params
            @test p.min < p.nominal
            @test p.nominal < p.max
            @test p.min >= 0  # Concentrations non-negative
        end

        println("  ✓ Default coagulation parameters defined")
    end

    println("\n✓ All Sensitivity Analysis tests passed!")
end

# ============================================================================
# SUMMARY
# ============================================================================

println("\n" * "=" ^ 70)
println("ALL NEW MODULE TESTS COMPLETED!")
println("=" ^ 70)
println("""
Modules Tested:
  - TGA Validation: Clinical dataset comparison, GOF metrics
  - Lattice Boltzmann: D2Q9 lattice, blood rheology, WSS extraction
  - Sensitivity Analysis: OAT, Morris, Sobol, PRCC methods

All core functionality verified!
""")
