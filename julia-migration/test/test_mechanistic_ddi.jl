# ===========================================================================
# MECHANISTIC DDI MODULE TESTS
# ===========================================================================
# Tests for IVIVE-DDI, multi-inhibitor saturation, MBI, transporter DDI,
# and Monte Carlo uncertainty quantification.
#
# Author: Dr. Sounio Agourakis
# Date: December 2025
# ===========================================================================

using Test
using DarwinPBPK

@testset "Mechanistic DDI Module" begin

    @testset "IVIVE Functions" begin
        # Test fu_mic calculation
        fu_mic = calculate_fu_mic(0.1, 3.5)
        @test fu_mic > 0.0
        @test fu_mic < 1.0
        @test fu_mic < 0.1  # Should be lower than fu_plasma for lipophilic

        # Test fu_mic with different methods
        fu_mic_austin = calculate_fu_mic(0.5, 2.0; method=:austin)
        fu_mic_hallifax = calculate_fu_mic(0.5, 2.0; method=:hallifax)
        @test fu_mic_austin > 0.0
        @test fu_mic_hallifax > 0.0

        # Test fu_inc
        fu_inc = calculate_fu_inc(0.1, 3.5)
        @test fu_inc > 0.0
        @test fu_inc < 1.0

        # Test MPPGL scaling
        mppgl_adult = mppgl_scaling(40.0)
        @test mppgl_adult ≈ 40.0

        mppgl_neonate = mppgl_scaling(0.5)
        @test mppgl_neonate ≈ 26.0

        mppgl_child = mppgl_scaling(10.0)
        @test mppgl_child > 35.0
        @test mppgl_child < 40.0

        mppgl_elderly = mppgl_scaling(75.0)
        @test mppgl_elderly < 40.0
        @test mppgl_elderly >= 32.0

        # Test HPGL scaling
        hpgl_adult = hpgl_scaling(40.0)
        @test hpgl_adult ≈ 120e6

        # Test ISEF
        isef_3a4 = isef_adjustment(:CYP3A4)
        @test isef_3a4 ≈ 1.0

        isef_2d6 = isef_adjustment(:CYP2D6)
        @test isef_2d6 ≈ 0.4
    end

    @testset "Competitive Inhibition" begin
        # [I] = Ki → 2-fold inhibition
        factor = competitive_inhibition(1.0, 1.0)
        @test factor ≈ 2.0

        # [I] = 10 × Ki → 11-fold inhibition
        factor_high = competitive_inhibition(10.0, 1.0)
        @test factor_high ≈ 11.0

        # [I] = 0 → no inhibition
        factor_zero = competitive_inhibition(0.0, 1.0)
        @test factor_zero ≈ 1.0
    end

    @testset "Noncompetitive Inhibition" begin
        factor = noncompetitive_inhibition(1.0, 1.0)
        @test factor ≈ 2.0
    end

    @testset "Uncompetitive Inhibition" begin
        factor = uncompetitive_inhibition(1.0, 1.0, 10.0, 10.0)
        @test factor > 1.0
        @test factor < 2.0  # Less than competitive at same [I]/Ki
    end

    @testset "Mixed Inhibition" begin
        # alpha = 1 should behave like noncompetitive
        mixed_nc = mixed_inhibition(1.0, 1.0; alpha=1.0)
        @test mixed_nc ≈ 1.0  # Net effect on CLint

        # alpha > 1 (competitive-like)
        mixed_comp = mixed_inhibition(1.0, 1.0; alpha=10.0)
        @test mixed_comp > 0.5
        @test mixed_comp < 1.0
    end

    @testset "Mechanism-Based Inhibition" begin
        # At t=0, 100% enzyme active
        mbi_t0 = mechanism_based_inhibition(0.05, 5.0, 1.0, 0.02; time=0.0)
        @test mbi_t0 ≈ 1.0

        # At steady state, enzyme is inactivated
        mbi_ss = mechanism_based_inhibition(0.05, 5.0, 1.0, 0.02; time=Inf)
        @test mbi_ss > 0.0
        @test mbi_ss < 0.5  # Significant inactivation

        # Time course: should approach steady state
        mbi_t1 = mechanism_based_inhibition(0.05, 5.0, 1.0, 0.02; time=1.0)
        mbi_t24 = mechanism_based_inhibition(0.05, 5.0, 1.0, 0.02; time=24.0)
        @test mbi_t1 > mbi_t24  # More enzyme remaining at earlier time
        @test mbi_t24 ≈ mbi_ss atol=0.01  # Near steady state by 24h

        # No inhibitor → 100% enzyme
        mbi_no_inh = mechanism_based_inhibition(0.05, 5.0, 0.0, 0.02; time=Inf)
        @test mbi_no_inh ≈ 1.0
    end

    @testset "Induction" begin
        # Baseline induction
        ind_baseline = calculate_induction_magnitude(10.0, 1.0, 0.0)
        @test ind_baseline ≈ 1.0

        # At EC50, should get ~half Emax
        ind_ec50 = calculate_induction_magnitude(10.0, 1.0, 1.0)
        @test ind_ec50 ≈ 6.0  # 1 + (10 × 1/(1+1)) = 1 + 5 = 6

        # High concentration → near Emax
        ind_high = calculate_induction_magnitude(10.0, 1.0, 100.0)
        @test ind_high > 10.0
        @test ind_high < 11.5

        # Net effect with both inhibition and induction
        net = net_effect_inhibition_induction(5.0, 3.0)
        @test net ≈ 0.6  # 3.0/5.0

        # Delayed onset
        delayed_t0 = delayed_induction_onset(10.0, 1.0, 5.0, 0.02, 0.0)
        @test delayed_t0 ≈ 1.0

        delayed_ss = delayed_induction_onset(10.0, 1.0, 5.0, 0.02, 200.0)
        @test delayed_ss > 5.0  # Near steady state (Emax~6x at EC50)
    end

    @testset "fm Calculation" begin
        fm_result = calculate_fmcyp_inhibited(0.9, 10.0)
        # fm=0.9, inhibited 10-fold → 0.9/10 + 0.1 = 0.19
        @test fm_result ≈ 0.19

        # No inhibition
        fm_no_inh = calculate_fmcyp_inhibited(0.9, 1.0)
        @test fm_no_inh ≈ 1.0
    end

    @testset "Substrate Depletion" begin
        factor = substrate_depletion_factor(10.0, 100.0, 1000.0, 60.0)
        @test factor > 0.0
        @test factor < 1.0

        # No depletion at t=0
        factor_t0 = substrate_depletion_factor(10.0, 100.0, 1000.0, 0.0)
        @test factor_t0 ≈ 1.0
    end

    @testset "Transporter DDI" begin
        # OATP inhibition
        oatp_ratio = oatp_inhibition(1.0, 0.05, 0.1; fm_oatp1b1=0.5, fm_oatp1b3=0.2)
        @test oatp_ratio > 1.0
        @test oatp_ratio < 50.0

        # P-gp inhibition
        pgp_effect = pgp_inhibition(10.0, 1.0; tissue=:intestine)
        @test pgp_effect > 1.0  # Increased absorption

        # BCRP inhibition
        bcrp_effect = bcrp_inhibition(5.0, 0.5)
        @test bcrp_effect > 1.0

        # Combined CYP + transporter
        combined = combined_cyp_transporter_ddi(5.0, 3.0)
        @test combined.total_auc_ratio ≈ 15.0
        @test combined.cyp_contribution ≈ 5.0
        @test combined.transporter_contribution ≈ 3.0
    end

    @testset "Multi-Inhibitor Saturation" begin
        inh1 = InhibitorParams(
            "Inhibitor1", :competitive, 0.1, 0.0, 0.0, 0.5,
            1.0, 0.5, 100.0, [:CYP3A4], Symbol[]
        )
        inh2 = InhibitorParams(
            "Inhibitor2", :competitive, 0.2, 0.0, 0.0, 0.3,
            2.0, 0.6, 200.0, [:CYP3A4], Symbol[]
        )

        multi = multi_inhibitor_saturation([inh1, inh2], :CYP3A4)

        @test multi.enzyme == :CYP3A4
        @test multi.total_inhibition > 0.0
        @test multi.total_inhibition < 1.0
        @test multi.saturation_fraction > 0.0
        @test length(multi.inhibitors) == 2
    end

    @testset "InhibitorParams Construction" begin
        inh = InhibitorParams(
            "Ketoconazole",
            :competitive,
            0.015,
            0.0,
            0.0,
            0.01,
            6.5,
            0.065,
            5300.0,
            [:CYP3A4],
            [:P_gp]
        )

        @test inh.name == "Ketoconazole"
        @test inh.inhibitor_type == :competitive
        @test inh.ki ≈ 0.015
        @test :CYP3A4 in inh.target_enzymes
        @test :P_gp in inh.target_transporters
    end

    @testset "IVIVEParams Construction" begin
        ivive = IVIVEParams(0.1, 0.1, 0.5, 0.5, 1.0, 3.0, 300.0, 7.0, false)

        @test ivive.fu_plasma ≈ 0.1
        @test ivive.logp ≈ 3.0
        @test ivive.mw ≈ 300.0
    end

    @testset "Monte Carlo DDI Prediction" begin
        inh = InhibitorParams(
            "TestInhibitor", :competitive, 0.1, 0.0, 0.0, 0.5,
            1.0, 0.5, 100.0, [:CYP3A4], Symbol[]
        )
        victim_fm = Dict(:CYP3A4 => 0.8)
        victim_km = Dict(:CYP3A4 => 10.0)

        uncertainty = monte_carlo_ddi_prediction(
            inh, victim_fm, victim_km;
            n_sim = 100
        )

        @test uncertainty.n_simulations == 100
        @test uncertainty.mean_ratio > 1.0
        @test uncertainty.sd_ratio > 0.0
        @test uncertainty.ci_90_lower < uncertainty.ci_90_upper
        @test uncertainty.ci_95_lower < uncertainty.ci_95_upper
        @test uncertainty.ci_95_lower <= uncertainty.ci_90_lower
        @test uncertainty.p_exceeds_2fold >= 0.0
        @test uncertainty.p_exceeds_2fold <= 1.0
    end

    @testset "DDI Severity Classification" begin
        @test classify_ddi_severity(1.1) == :no_interaction
        @test classify_ddi_severity(1.5) == :weak
        @test classify_ddi_severity(3.0) == :moderate
        @test classify_ddi_severity(8.0) == :strong
    end

    @testset "Full DDI Prediction" begin
        ketoconazole = InhibitorParams(
            "Ketoconazole", :competitive, 0.015, 0.0, 0.0, 0.01,
            6.5, 0.065, 5300.0, [:CYP3A4], [:P_gp]
        )

        midazolam_fm = Dict(:CYP3A4 => 0.95)
        midazolam_ivive = IVIVEParams(0.03, 0.03, 0.5, 0.5, 1.0, 3.0, 325.0, 6.2, false)

        prediction = predict_ddi_mechanistic(
            ketoconazole, midazolam_fm, midazolam_ivive;
            n_monte_carlo = 100
        )

        @test prediction.perpetrator == "Ketoconazole"
        @test prediction.auc_ratio > 1.0
        @test prediction.cmax_ratio > 1.0
        @test prediction.severity in [:weak, :moderate, :strong]
        @test !isempty(prediction.recommendations)
        @test haskey(prediction.cyp_contributions, :CYP3A4)
    end

    @testset "DDI Time Course Simulation" begin
        inh = InhibitorParams(
            "MBI_Inhibitor", :mbi, 0.1, 0.03, 0.5, 0.1,
            5.0, 0.5, 2000.0, [:CYP3A4], Symbol[]
        )

        victim_fm = Dict(:CYP3A4 => 0.8)

        time_course = simulate_ddi_time_course(
            inh, victim_fm;
            duration = 48.0,
            dt = 1.0
        )

        @test length(time_course) > 0
        @test haskey(time_course[1], "time")
        @test haskey(time_course[1], "auc_ratio")
        @test haskey(time_course[1], "enzyme_activity")

        # AUC ratio should increase over time with MBI
        @test time_course[end]["auc_ratio"] > time_course[1]["auc_ratio"]
    end

    @testset "DDI Report Generation" begin
        ketoconazole = InhibitorParams(
            "Ketoconazole", :competitive, 0.015, 0.0, 0.0, 0.01,
            6.5, 0.065, 5300.0, [:CYP3A4], [:P_gp]
        )
        midazolam_fm = Dict(:CYP3A4 => 0.95)
        midazolam_ivive = IVIVEParams(0.03, 0.03, 0.5, 0.5, 1.0, 3.0, 325.0, 6.2, false)

        prediction = predict_ddi_mechanistic(
            ketoconazole, midazolam_fm, midazolam_ivive;
            n_monte_carlo = 50
        )

        report = create_ddi_report(prediction)

        @test occursin("DRUG-DRUG INTERACTION", report)
        @test occursin("Ketoconazole", report)
        @test occursin("AUC Ratio", report)
        @test occursin("RECOMMENDATIONS", report)
    end

    @testset "Clinical Validation" begin
        passes, fold_error = validate_against_clinical(10.0, 12.0)
        @test passes == true
        @test fold_error ≈ 1.2

        fails, fold_error2 = validate_against_clinical(5.0, 15.0)
        @test fails == false
        @test fold_error2 == 3.0
    end

    @testset "Constants" begin
        # Physiological params
        @test haskey(PHYSIOLOGICAL_PARAMS, "liver_weight")
        @test haskey(PHYSIOLOGICAL_PARAMS, "mppgl")
        @test haskey(PHYSIOLOGICAL_PARAMS, "cyp3a4_halflife")

        @test PHYSIOLOGICAL_PARAMS["liver_weight"] ≈ 1800.0
        @test PHYSIOLOGICAL_PARAMS["mppgl"] ≈ 40.0

        # CYP abundance
        @test haskey(CYP_ABUNDANCE, :CYP3A4)
        @test haskey(CYP_ABUNDANCE, :CYP2D6)
        @test CYP_ABUNDANCE[:CYP3A4]["liver"] > CYP_ABUNDANCE[:CYP2D6]["liver"]

        # Transporter abundance
        @test haskey(TRANSPORTER_ABUNDANCE, :OATP1B1)
        @test haskey(TRANSPORTER_ABUNDANCE, :P_gp)
    end

end

println("\n✅ All Mechanistic DDI tests passed!")
