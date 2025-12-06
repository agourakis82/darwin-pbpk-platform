"""
Test Blood Partitioning Integration in ODE Solver

Tests the Blood:Plasma ratio functionality integrated into the main PBPK ODE solver.

Author: Darwin PBPK Platform
Date: 2025-12-06
"""

using Test
using DarwinPBPK.ODEPBPKSolver

@testset "Blood Partitioning Helper Functions" begin
    # Test 1: calculate_blood_plasma_ratio with default params
    @testset "Blood:Plasma Ratio Calculation" begin
        params = PBPKParams(
            ke_p = 1.0,
            hematocrit = 0.45,
            enable_bp_ratio = true
        )

        Rb = calculate_blood_plasma_ratio(params)

        # Expected: Rb = 1 - 0.45 + 0.45 × 1.0 = 1.0
        @test Rb ≈ 1.0 atol=1e-6

        # Test with Ke_p = 0.5 (drug excluded from RBC)
        params2 = PBPKParams(
            ke_p = 0.5,
            hematocrit = 0.45,
            enable_bp_ratio = true
        )

        Rb2 = calculate_blood_plasma_ratio(params2)

        # Expected: Rb = 1 - 0.45 + 0.45 × 0.5 = 0.775
        @test Rb2 ≈ 0.775 atol=1e-6

        # Test with Ke_p = 2.0 (drug accumulates in RBC)
        params3 = PBPKParams(
            ke_p = 2.0,
            hematocrit = 0.45,
            enable_bp_ratio = true
        )

        Rb3 = calculate_blood_plasma_ratio(params3)

        # Expected: Rb = 1 - 0.45 + 0.45 × 2.0 = 1.45
        @test Rb3 ≈ 1.45 atol=1e-6
    end

    # Test 2: partition_blood_concentration
    @testset "Blood Concentration Partitioning" begin
        C_blood = 100.0  # ng/mL
        Rb = 0.775
        Hct = 0.45

        partitioned = partition_blood_concentration(C_blood, Rb, Hct)

        # C_plasma = C_blood / Rb = 100 / 0.775 ≈ 129.03
        @test partitioned.C_plasma ≈ 129.03 atol=0.1

        # Ke_p = (Rb - (1 - Hct)) / Hct = (0.775 - 0.55) / 0.45 = 0.5
        # C_rbc = Ke_p × C_plasma = 0.5 × 129.03 ≈ 64.52
        @test partitioned.C_rbc ≈ 64.52 atol=0.1
    end

    # Test 3: get_unbound_plasma_concentration
    @testset "Unbound Plasma Concentration" begin
        C_blood = 100.0
        Rb = 1.45
        fu = 0.1  # 10% unbound

        C_unbound = get_unbound_plasma_concentration(C_blood, Rb, fu)

        # C_plasma = 100 / 1.45 ≈ 68.97
        # C_unbound = 68.97 × 0.1 ≈ 6.90
        @test C_unbound ≈ 6.90 atol=0.1
    end

    # Test 4: calculate_fu_blood
    @testset "Fraction Unbound in Blood" begin
        fu_plasma = 0.1
        Rb = 1.45

        fu_blood = calculate_fu_blood(fu_plasma, Rb)

        # fu_blood = 0.1 / 1.45 ≈ 0.069
        @test fu_blood ≈ 0.069 atol=0.001
    end

    # Test 5: estimate_ke_p_from_logP
    @testset "Ke_p Estimation from LogP" begin
        # Lipophilic neutral drug
        ke_p1 = estimate_ke_p_from_logP(3.0, :neutral, Float64[])
        @test 0.8 <= ke_p1 <= 1.5

        # Basic drug (accumulates in RBC)
        ke_p2 = estimate_ke_p_from_logP(2.0, :basic, [8.5])
        @test ke_p2 > 0.8  # Should be higher due to pH trapping

        # Acidic drug (excluded from RBC)
        ke_p3 = estimate_ke_p_from_logP(2.0, :acidic, [4.0])
        @test ke_p3 < 0.7  # Should be lower

        # Very polar drug
        ke_p4 = estimate_ke_p_from_logP(-1.0, :neutral, Float64[])
        @test 0.4 <= ke_p4 <= 0.7
    end
end

@testset "PBPK ODE System with Blood Partitioning" begin
    @testset "Backward Compatibility (BP Ratio Disabled)" begin
        # With enable_bp_ratio = false, should behave exactly as before
        params = PBPKParams(
            clearance_hepatic = 10.0,
            clearance_renal = 5.0,
            enable_bp_ratio = false
        )

        dose = 100.0
        tspan = (0.0, 24.0)
        time_points = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

        sol = solve(params, dose, tspan; time_points=time_points)

        @test sol.retcode == :Success
        @test length(sol) == length(time_points)

        # Mass conservation
        @test validate_mass_conservation(sol, params, dose)
    end

    @testset "Blood Partitioning Enabled - Low Ke_p" begin
        # Drug excluded from RBC (Ke_p = 0.5)
        params = PBPKParams(
            clearance_hepatic = 10.0,
            clearance_renal = 5.0,
            ke_p = 0.5,
            hematocrit = 0.45,
            fu_plasma = 0.1,
            enable_bp_ratio = true
        )

        dose = 100.0
        tspan = (0.0, 24.0)
        time_points = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

        sol_bp = solve(params, dose, tspan; time_points=time_points)

        @test sol_bp.retcode == :Success
        @test length(sol_bp) == length(time_points)

        # Mass conservation with BP ratio
        @test validate_mass_conservation(sol_bp, params, dose)

        # Check that plasma concentrations are higher than blood
        # (because drug is excluded from RBC)
        C_blood_t1 = sol_bp[2][BLOOD_IDX]
        Rb = calculate_blood_plasma_ratio(params)
        @test Rb < 1.0  # Should be < 1 for excluded drug

        partitioned = partition_blood_concentration(C_blood_t1, Rb, params.hematocrit)
        @test partitioned.C_plasma > C_blood_t1  # Plasma > blood
    end

    @testset "Blood Partitioning Enabled - High Ke_p" begin
        # Drug accumulates in RBC (Ke_p = 2.0)
        params = PBPKParams(
            clearance_hepatic = 10.0,
            clearance_renal = 5.0,
            ke_p = 2.0,
            hematocrit = 0.45,
            fu_plasma = 0.1,
            enable_bp_ratio = true
        )

        dose = 100.0
        tspan = (0.0, 24.0)
        time_points = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

        sol_bp = solve(params, dose, tspan; time_points=time_points)

        @test sol_bp.retcode == :Success

        # Mass conservation
        @test validate_mass_conservation(sol_bp, params, dose)

        # Check that plasma concentrations are lower than blood
        # (because drug accumulates in RBC)
        C_blood_t1 = sol_bp[2][BLOOD_IDX]
        Rb = calculate_blood_plasma_ratio(params)
        @test Rb > 1.0  # Should be > 1 for accumulating drug

        partitioned = partition_blood_concentration(C_blood_t1, Rb, params.hematocrit)
        @test partitioned.C_plasma < C_blood_t1  # Plasma < blood
        @test partitioned.C_rbc > partitioned.C_plasma  # RBC > plasma
    end

    @testset "Anemia vs Normal Hematocrit" begin
        # Normal hematocrit
        params_normal = PBPKParams(
            clearance_hepatic = 10.0,
            ke_p = 1.5,
            hematocrit = 0.45,
            fu_plasma = 0.1,
            enable_bp_ratio = true
        )

        # Anemia (low hematocrit)
        params_anemia = PBPKParams(
            clearance_hepatic = 10.0,
            ke_p = 1.5,
            hematocrit = 0.30,  # Anemia
            fu_plasma = 0.1,
            enable_bp_ratio = true
        )

        # Calculate B:P ratios
        Rb_normal = calculate_blood_plasma_ratio(params_normal)
        Rb_anemia = calculate_blood_plasma_ratio(params_anemia)

        # With Ke_p > 1, anemia should decrease Rb
        # Rb = 1 - Hct + Hct × Ke_p
        # Normal: 1 - 0.45 + 0.45 × 1.5 = 1.225
        # Anemia: 1 - 0.30 + 0.30 × 1.5 = 1.150
        @test Rb_anemia < Rb_normal
        @test Rb_normal ≈ 1.225 atol=0.001
        @test Rb_anemia ≈ 1.150 atol=0.001
    end

    @testset "Polycythemia vs Normal Hematocrit" begin
        # Normal hematocrit
        params_normal = PBPKParams(
            ke_p = 1.5,
            hematocrit = 0.45,
            enable_bp_ratio = true
        )

        # Polycythemia (high hematocrit)
        params_polyc = PBPKParams(
            ke_p = 1.5,
            hematocrit = 0.60,  # Polycythemia
            enable_bp_ratio = true
        )

        Rb_normal = calculate_blood_plasma_ratio(params_normal)
        Rb_polyc = calculate_blood_plasma_ratio(params_polyc)

        # With Ke_p > 1, polycythemia should increase Rb
        # Normal: 1.225
        # Polyc: 1 - 0.60 + 0.60 × 1.5 = 1.300
        @test Rb_polyc > Rb_normal
        @test Rb_polyc ≈ 1.300 atol=0.001
    end
end

@testset "Simulation Results Comparison" begin
    @testset "Effect of BP Ratio on PK Parameters" begin
        # Common parameters
        dose = 100.0
        tspan = (0.0, 48.0)

        # Simulate without BP ratio (baseline)
        params_baseline = PBPKParams(
            clearance_hepatic = 10.0,
            clearance_renal = 5.0,
            enable_bp_ratio = false
        )

        results_baseline = simulate(params_baseline, dose; t_max=48.0, num_points=100)

        # Simulate with BP ratio enabled (Ke_p = 0.5, drug excluded from RBC)
        params_excluded = PBPKParams(
            clearance_hepatic = 10.0,
            clearance_renal = 5.0,
            ke_p = 0.5,
            hematocrit = 0.45,
            fu_plasma = 0.1,
            enable_bp_ratio = true
        )

        results_excluded = simulate(params_excluded, dose; t_max=48.0, num_points=100)

        # With drug excluded from RBC:
        # - Plasma concentrations should be higher
        # - Clearance is based on unbound plasma, which is higher
        # - Overall elimination should be faster

        blood_baseline = results_baseline["blood"]
        blood_excluded = results_excluded["blood"]

        # At t=1h, compare blood concentrations
        C_baseline_1h = blood_baseline[11]  # Index 11 ≈ 1h
        C_excluded_1h = blood_excluded[11]

        # Blood concentration should be lower when drug is excluded from RBC
        # (because effective clearance is higher)
        @test C_excluded_1h < C_baseline_1h

        println("Baseline C_blood(1h) = $(C_baseline_1h)")
        println("Excluded C_blood(1h) = $(C_excluded_1h)")
        println("Ratio = $(C_excluded_1h / C_baseline_1h)")
    end
end

println("\n" * "="^80)
println("Blood Partitioning Integration Tests Summary")
println("="^80)
println("✓ All helper functions tested")
println("✓ ODE system integration verified")
println("✓ Mass conservation validated with BP ratio")
println("✓ Anemia/polycythemia effects demonstrated")
println("✓ PK parameter effects quantified")
println("="^80)
