"""
Test suite for ODE Solver + Blood Compartment Integration

Tests dynamic PK parameter adjustments during ODE simulation.
"""

using Test

# Load the ODE solver module directly
include("../src/DarwinPBPK/ode_solver.jl")
using .ODEPBPKSolver

@testset "ODE Blood Compartment Integration" begin

    @testset "DynamicPBPKParams Creation" begin
        base_params = PBPKParams(
            clearance_hepatic = 20.0,
            clearance_renal = 5.0
        )

        dyn = DynamicPBPKParams(base_params)

        @test dyn.hepatic_cl_factor == 1.0
        @test dyn.renal_cl_factor == 1.0
        @test dyn.fu_factor == 1.0
        @test dyn.rb == 1.0

        @test effective_cl_hepatic(dyn) == 20.0
        @test effective_cl_renal(dyn) == 5.0

        # Modify factors
        dyn.hepatic_cl_factor = 0.5
        dyn.renal_cl_factor = 0.3

        @test effective_cl_hepatic(dyn) == 10.0
        @test effective_cl_renal(dyn) == 1.5

        println("  DynamicPBPKParams created and modified successfully")
    end

    @testset "BloodStateODECallback Creation" begin
        # Normal state
        blood_normal = BloodStateODECallback()

        @test blood_normal.hematocrit == 0.42
        @test blood_normal.albumin_g_L == 40.0
        @test blood_normal.aag_g_L == 0.8
        @test blood_normal.is_acute_phase == false

        println("  Normal blood callback: Hct=$(blood_normal.hematocrit), Alb=$(blood_normal.albumin_g_L)")

        # Sepsis state
        blood_sepsis = BloodStateODECallback(
            hematocrit = 0.30,
            albumin_g_L = 20.0,
            aag_g_L = 2.5,
            il6_pg_mL = 200.0,
            gfr = 40.0,
            hepatic_flow = 60.0,
            ke_p = 37.0,
            fu_reference = 0.01,
            charge_type = :basic,
            extraction_ratio = 0.3,
            is_acute_phase = true
        )

        @test blood_sepsis.is_acute_phase == true
        @test blood_sepsis.albumin_g_L == 20.0
        @test blood_sepsis.aag_g_L == 2.5

        println("  Sepsis blood callback: IL-6=$(blood_sepsis.il6_pg_mL), AAG=$(blood_sepsis.aag_g_L)")
    end

    @testset "Blood State Update (Acute Phase)" begin
        blood = BloodStateODECallback(
            albumin_g_L = 20.0,
            aag_g_L = 2.5,
            il6_pg_mL = 200.0,
            crp_mg_L = 50.0,
            is_acute_phase = true
        )

        initial_il6 = blood.il6_pg_mL
        initial_aag = blood.aag_g_L

        # Simulate 24 hours
        for _ in 1:24
            update_blood_state_ode!(blood, 1.0)
        end

        # IL-6 should decay
        @test blood.il6_pg_mL < initial_il6
        @test blood.time_since_onset == 24.0

        println("  24h acute phase: IL-6 $(initial_il6) -> $(round(blood.il6_pg_mL, digits=1))")
        println("  24h acute phase: AAG $(initial_aag) -> $(round(blood.aag_g_L, digits=2))")
    end

    @testset "Dynamic Adjustments Calculation" begin
        # Test acidic drug in hypoalbuminemia
        blood_acidic = BloodStateODECallback(
            albumin_g_L = 20.0,  # 50% of normal
            charge_type = :acidic,
            fu_reference = 0.1,
            ke_p = 1.0
        )

        adj = calculate_dynamic_adjustments(blood_acidic)

        @test adj.fu_factor == 2.0  # 40/20 = 2x
        @test adj.rb ≈ 1.0 atol=0.01  # ke_p = 1, so Rb ≈ 1

        println("  Acidic drug in hypoalbuminemia: fu_factor=$(adj.fu_factor)")

        # Test basic drug in elevated AAG
        blood_basic = BloodStateODECallback(
            aag_g_L = 2.4,  # 3x normal
            charge_type = :basic,
            fu_reference = 0.3,
            ke_p = 5.0
        )

        adj = calculate_dynamic_adjustments(blood_basic)

        @test adj.fu_factor ≈ 0.333 atol=0.01  # 0.8/2.4 = 0.33
        @test adj.rb > 1.0  # Rb > 1 due to RBC partitioning

        println("  Basic drug with elevated AAG: fu_factor=$(round(adj.fu_factor, digits=3)), Rb=$(round(adj.rb, digits=2))")

        # Test high extraction drug
        blood_high_e = BloodStateODECallback(
            hepatic_flow = 45.0,  # 50% of normal
            extraction_ratio = 0.9
        )

        adj = calculate_dynamic_adjustments(blood_high_e)

        @test adj.hepatic_cl_factor == 0.5  # Flow-limited

        println("  High extraction drug with reduced flow: hepatic_cl_factor=$(adj.hepatic_cl_factor)")
    end

    @testset "Solve with Blood State - Normal" begin
        base_params = PBPKParams(
            clearance_hepatic = 20.0,
            clearance_renal = 5.0
        )

        blood = BloodStateODECallback()  # Normal state

        sol = solve_with_blood_state(
            base_params, blood, 100.0, (0.0, 24.0);
            time_points = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        )

        @test length(sol) == 7
        @test sol[1][1] > 0  # Initial concentration > 0
        @test sol[end][1] < sol[1][1]  # Concentration decreases

        println("  Normal simulation: C(0)=$(round(sol[1][1], digits=2)), C(24)=$(round(sol[end][1], digits=2)) mg/L")
    end

    @testset "Solve with Blood State - Sepsis Acute Phase" begin
        base_params = PBPKParams(
            clearance_hepatic = 20.0,
            clearance_renal = 5.0
        )

        # Sepsis patient receiving a basic drug (e.g., tacrolimus-like)
        blood_sepsis = BloodStateODECallback(
            hematocrit = 0.30,
            albumin_g_L = 20.0,
            aag_g_L = 2.5,
            il6_pg_mL = 200.0,
            gfr = 40.0,
            hepatic_flow = 60.0,
            ke_p = 37.0,
            fu_reference = 0.01,
            charge_type = :basic,
            extraction_ratio = 0.3,
            is_acute_phase = true,
            update_interval = 4.0  # Update every 4 hours
        )

        sol = solve_with_blood_state(
            base_params, blood_sepsis, 5.0, (0.0, 72.0);
            time_points = collect(0.0:4.0:72.0)
        )

        @test length(sol) > 0
        @test sol[1][1] > 0

        println("  Sepsis simulation (72h):")
        println("    C(0)=$(round(sol[1][1], digits=3)) mg/L")
        println("    C(24)=$(round(sol[7][1], digits=3)) mg/L")
        println("    C(72)=$(round(sol[end][1], digits=3)) mg/L")
    end

    @testset "Simulate with Blood State - Full Results" begin
        base_params = PBPKParams(
            clearance_hepatic = 15.0,
            clearance_renal = 3.0
        )

        blood = BloodStateODECallback(
            albumin_g_L = 25.0,
            aag_g_L = 1.5,
            charge_type = :acidic,
            fu_reference = 0.1,
            is_acute_phase = false
        )

        results = simulate_with_blood_state(
            base_params, blood, 100.0;
            t_max = 24.0,
            num_points = 25
        )

        @test haskey(results, "time")
        @test haskey(results, "plasma")
        @test haskey(results, "blood")
        @test haskey(results, "liver")
        @test haskey(results, "blood_state")

        @test length(results["time"]) == 25
        @test results["blood_state"]["initial_albumin"] == 25.0
        @test results["blood_state"]["charge_type"] == :acidic

        # Check Cmax and decay
        cmax_idx = argmax(results["plasma"])
        @test cmax_idx == 1  # IV bolus, max at t=0
        @test results["plasma"][end] < results["plasma"][1]

        println("  Full simulation results:")
        println("    Time points: $(length(results["time"]))")
        println("    Cmax (plasma): $(round(results["plasma"][1], digits=2)) mg/L")
        println("    C(24h): $(round(results["plasma"][end], digits=3)) mg/L")
        println("    Blood state recorded: albumin=$(results["blood_state"]["initial_albumin"]) g/L")
    end

    @testset "Compare Normal vs Disease PK" begin
        base_params = PBPKParams(
            clearance_hepatic = 20.0,
            clearance_renal = 10.0
        )

        # Normal patient
        blood_normal = BloodStateODECallback(
            gfr = 100.0,
            charge_type = :acidic,
            fu_reference = 0.1
        )

        # CKD patient (reduced GFR)
        blood_ckd = BloodStateODECallback(
            gfr = 30.0,  # Stage 4 CKD
            albumin_g_L = 32.0,
            charge_type = :acidic,
            fu_reference = 0.1
        )

        sol_normal = solve_with_blood_state(
            base_params, blood_normal, 100.0, (0.0, 24.0);
            time_points = [0.0, 12.0, 24.0]
        )

        sol_ckd = solve_with_blood_state(
            base_params, blood_ckd, 100.0, (0.0, 24.0);
            time_points = [0.0, 12.0, 24.0]
        )

        # CKD should have higher concentrations at 24h due to reduced clearance
        c24_normal = sol_normal[3][1]
        c24_ckd = sol_ckd[3][1]

        @test c24_ckd > c24_normal

        ratio = c24_ckd / c24_normal

        println("  Normal vs CKD comparison:")
        println("    C(24h) normal: $(round(c24_normal, digits=3)) mg/L")
        println("    C(24h) CKD: $(round(c24_ckd, digits=3)) mg/L")
        println("    Ratio CKD/Normal: $(round(ratio, digits=2))x")
    end

end

println("\n=== ODE Blood Integration Tests Complete ===")
