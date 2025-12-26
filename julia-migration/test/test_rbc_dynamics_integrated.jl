# ===========================================================================
# RBC DYNAMICS INTEGRATED MODULE TESTS
# ===========================================================================
# Tests for closed-loop RBC dynamics with hematopoiesis, EPO feedback,
# organ clearance, and transporter expression.
#
# Author: Dr. Sounio Agourakis
# Date: December 2025
# ===========================================================================

using Test
using DarwinPBPK

@testset "RBC Dynamics Integrated" begin

    @testset "Initialization" begin
        # Test normal initialization
        state = initialize_rbc_dynamics()
        @test state.hematocrit ≈ 0.45
        @test state.hemoglobin ≈ 15.0
        @test state.disease == :normal
        @test state.epo.epo_level ≈ 10.0  # Normal baseline EPO
        @test state.age_distribution.total_count > 0
        @test length(state.age_distribution.counts) == 120

        # Test custom hematocrit
        state_low = initialize_rbc_dynamics(hematocrit=0.30)
        @test state_low.hematocrit ≈ 0.30
        @test state_low.age_distribution.total_count < state.age_distribution.total_count

        # Test disease state initialization
        state_sickle = initialize_rbc_dynamics(disease=:sickle_cell, hematocrit=0.25)
        @test state_sickle.disease == :sickle_cell
        @test state_sickle.disease_params["rbc_lifespan"] ≈ 17.0
        @test state_sickle.hematocrit ≈ 0.25
    end

    @testset "EPO Response" begin
        state = initialize_rbc_dynamics()

        # Normal Hct - should have baseline EPO
        epo_normal = calculate_epo_response(0.45, state.epo)
        @test epo_normal < 15.0  # Should be near baseline

        # Low Hct - should stimulate EPO production
        epo_low = calculate_epo_response(0.30, state.epo)
        @test epo_low > epo_normal
        @test epo_low > 50.0  # Should be significantly elevated

        # High Hct - should suppress EPO
        epo_high = calculate_epo_response(0.55, state.epo)
        @test epo_high < epo_normal

        # Very low Hct - EPO capped at 1000 after update
        epo_very_low = calculate_epo_response(0.15, state.epo)
        @test epo_very_low > 100.0  # High stimulation
    end

    @testset "Reticulocyte Release" begin
        state = initialize_rbc_dynamics()

        # Baseline EPO - baseline production
        release_baseline = calculate_reticulocyte_release(10.0, state.hematopoiesis)
        @test release_baseline > 0

        # Elevated EPO - increased production
        release_elevated = calculate_reticulocyte_release(100.0, state.hematopoiesis)
        @test release_elevated > release_baseline

        # Maximum EPO - near maximum production
        release_max = calculate_reticulocyte_release(500.0, state.hematopoiesis)
        @test release_max > release_elevated
        @test release_max < 5 * release_baseline  # Should plateau
    end

    @testset "Single Step Update" begin
        state = initialize_rbc_dynamics()
        initial_hct = state.hematocrit
        initial_total = state.age_distribution.total_count

        # Single day update
        result = update_rbc_dynamics!(state, 1.0)

        @test !isnan(result.new_hematocrit)
        @test !isnan(result.epo_level)
        @test result.cells_produced > 0
        @test result.cells_cleared > 0
        @test abs(result.new_hematocrit - initial_hct) < 0.05  # Should be stable
    end

    @testset "Simulation Stability - Normal" begin
        state = initialize_rbc_dynamics(disease=:normal, hematocrit=0.45)

        # 30-day simulation
        results = simulate_rbc_dynamics(state, 30.0)

        @test length(results) == 30

        # Hct should remain stable for normal
        final_hct = results[end].new_hematocrit
        @test !isnan(final_hct)
        @test final_hct > 0.40
        @test final_hct < 0.55

        # EPO should remain near baseline
        final_epo = results[end].epo_level
        @test final_epo > 1.0
        @test final_epo < 50.0  # Not elevated

        # Mean RBC age should be reasonable
        @test results[end].mean_rbc_age > 30.0
        @test results[end].mean_rbc_age < 70.0
    end

    @testset "Simulation Stability - Sickle Cell" begin
        state = initialize_rbc_dynamics(disease=:sickle_cell, hematocrit=0.25)

        # 30-day simulation
        results = simulate_rbc_dynamics(state, 30.0)

        # Values should not be NaN
        @test !any(isnan(r.new_hematocrit) for r in results)
        @test !any(isnan(r.epo_level) for r in results)

        # Hct should stabilize (may drop due to short lifespan)
        final_hct = results[end].new_hematocrit
        @test final_hct >= 0.10  # Above minimum
        @test final_hct <= 0.35  # Below normal

        # EPO should be elevated (maxed out)
        final_epo = results[end].epo_level
        @test final_epo >= 100.0

        # High reticulocyte fraction (compensatory)
        final_retic = results[end].reticulocyte_fraction
        @test final_retic > 0.02  # >2% reticulocytes
    end

    @testset "Simulation Stability - CKD Anemia" begin
        state = initialize_rbc_dynamics(disease=:ckd_anemia, hematocrit=0.30)

        results = simulate_rbc_dynamics(state, 30.0)

        # Values should not be NaN
        @test !any(isnan(r.new_hematocrit) for r in results)

        # CKD has poor EPO response
        @test state.hematopoiesis.epo_responsiveness ≈ 0.3
    end

    @testset "Disease Parameters" begin
        # Test all disease states load correctly
        for disease in [:normal, :sickle_cell, :hemolytic_anemia, :thalassemia_major,
                        :aplastic_anemia, :ckd_anemia, :polycythemia_vera,
                        :spherocytosis, :g6pd_deficiency]
            state = initialize_rbc_dynamics(disease=disease)
            @test state.disease == disease
            @test haskey(state.disease_params, "rbc_lifespan")
            @test state.disease_params["rbc_lifespan"] > 0
            @test state.disease_params["rbc_lifespan"] <= 120.0
        end
    end

    @testset "Age-Weighted Transporter Expression" begin
        state = initialize_rbc_dynamics()

        # Transporter expression should be between 0 and 1
        @test state.effective_band3 > 0.5
        @test state.effective_band3 <= 1.0
        @test state.effective_glut1 > 0.4
        @test state.effective_glut1 <= 1.0
        @test state.effective_ent1 > 0.3
        @test state.effective_ent1 <= 1.0

        # Young population (sickle cell) should have higher transporter expression
        state_young = initialize_rbc_dynamics(disease=:sickle_cell)
        @test state_young.effective_band3 >= state.effective_band3 * 0.9  # Similar or higher
    end

    @testset "PK Integration" begin
        state = initialize_rbc_dynamics()

        # Test apply_rbc_dynamics_to_pk
        pk_result = apply_rbc_dynamics_to_pk(
            state,
            10.0,  # CL base (L/h)
            100.0, # Vd base (L)
            0.8;   # Ke:p
            is_hepatically_cleared=true,
            transporter=:band3
        )

        @test haskey(pk_result, "cl_adjusted")
        @test haskey(pk_result, "vd_adjusted")
        @test haskey(pk_result, "rb")
        @test haskey(pk_result, "hematocrit")

        @test pk_result["cl_adjusted"] > 0
        @test pk_result["vd_adjusted"] > 0
        @test pk_result["rb"] > 0
        @test pk_result["rb"] < 2.0  # Reasonable B:P ratio
    end

    @testset "Effective Hematocrit" begin
        state = initialize_rbc_dynamics(hematocrit=0.45)
        @test get_effective_hematocrit(state) ≈ 0.45

        # After simulation
        simulate_rbc_dynamics(state, 5.0)
        hct = get_effective_hematocrit(state)
        @test hct > 0.10
        @test hct < 0.70
    end

    @testset "RBC-Mediated Clearance" begin
        state = initialize_rbc_dynamics()

        clearance_info = get_rbc_mediated_clearance(state)

        @test haskey(clearance_info, "spleen_rbc_clearance")
        @test haskey(clearance_info, "liver_rbc_clearance")
        @test haskey(clearance_info, "total_rbc_clearance")
        @test haskey(clearance_info, "rbc_turnover_days")

        # Spleen should clear most RBCs
        @test clearance_info["spleen_rbc_clearance"] > clearance_info["liver_rbc_clearance"]

        # Turnover should be ~120 days for normal
        @test clearance_info["rbc_turnover_days"] > 50
        @test clearance_info["rbc_turnover_days"] < 150
    end

    @testset "ODE System Generator" begin
        ode_func = create_rbc_ode_system(:normal)
        @test ode_func isa Function

        # Test ODE function call
        u = [5.0, 0.45, 10.0, 0.01, 1.0]  # [RBC, Hct, EPO, Retic, Bili]
        du = zeros(5)
        ode_func(du, u, nothing, 0.0)

        @test !any(isnan, du)
        @test !any(isinf, du)

        # Test disease-specific ODE
        ode_sickle = create_rbc_ode_system(:sickle_cell)
        du_sickle = zeros(5)
        ode_sickle(du_sickle, u, nothing, 0.0)

        @test !any(isnan, du_sickle)
    end

    @testset "EPO Therapy" begin
        # Test exogenous EPO
        state = initialize_rbc_dynamics(
            disease=:ckd_anemia,
            hematocrit=0.28,
            on_epo_therapy=true,
            epo_dose=10000.0  # 10,000 IU/week
        )

        @test state.epo.is_exogenous == true
        @test state.epo.exogenous_dose ≈ 10000.0

        # EPO level should include exogenous contribution
        epo_response = calculate_epo_response(0.28, state.epo)
        @test epo_response > state.epo.epo_level  # Higher due to exogenous
    end

    @testset "Iron and Bilirubin Metabolism" begin
        state = initialize_rbc_dynamics()

        # Iron
        @test state.iron.serum_iron > 0
        @test state.iron.ferritin > 0
        @test state.iron.transferrin_saturation > 0
        @test state.iron.transferrin_saturation < 1.0

        # Bilirubin
        @test state.bilirubin.total ≈ 1.0  # Normal ~1 mg/dL
        @test state.bilirubin.unconjugated > 0
        @test state.bilirubin.conjugated > 0

        # After simulation with high turnover
        state_hemolytic = initialize_rbc_dynamics(disease=:hemolytic_anemia)
        simulate_rbc_dynamics(state_hemolytic, 10.0)

        # Bilirubin may increase with hemolysis
        @test state_hemolytic.bilirubin.total >= 0.1
    end

    @testset "Organ Clearance Distribution" begin
        state = initialize_rbc_dynamics()

        clearance = calculate_organ_rbc_clearance(state)

        # Total should be sum of organs
        total = clearance.spleen_clearance + clearance.liver_clearance +
                clearance.other_clearance
        @test isapprox(total, clearance.total_clearance, rtol=0.01)

        # Spleen dominates (~90%)
        spleen_frac = clearance.spleen_clearance / clearance.total_clearance
        @test spleen_frac > 0.80
        @test spleen_frac < 0.98
    end

    @testset "Constants" begin
        # NORMAL_RBC_PARAMS
        @test haskey(NORMAL_RBC_PARAMS, "rbc_production_rate")
        @test haskey(NORMAL_RBC_PARAMS, "normal_rbc_lifespan")
        @test haskey(NORMAL_RBC_PARAMS, "epo_baseline")
        @test haskey(NORMAL_RBC_PARAMS, "hct_setpoint")

        @test NORMAL_RBC_PARAMS["normal_rbc_lifespan"] ≈ 120.0
        @test NORMAL_RBC_PARAMS["hct_setpoint"] ≈ 0.45

        # DISEASE_RBC_DYNAMICS
        @test haskey(DISEASE_RBC_DYNAMICS, :normal)
        @test haskey(DISEASE_RBC_DYNAMICS, :sickle_cell)
        @test haskey(DISEASE_RBC_DYNAMICS, :ckd_anemia)

        @test DISEASE_RBC_DYNAMICS[:sickle_cell]["rbc_lifespan"] < 30.0
        @test DISEASE_RBC_DYNAMICS[:aplastic_anemia]["production_rate_factor"] < 0.5
    end

end

println("\n✅ All RBC Dynamics Integrated tests passed!")
