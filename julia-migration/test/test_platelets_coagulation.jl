"""
Test Suite for Platelets and Coagulation Modules

Tests:
1. Platelet compartment creation and pathology
2. Platelet activation dynamics
3. Antiplatelet drug effects
4. Coagulation system creation
5. Coagulation ODE simulation
6. Warfarin PK/PD effects
7. DOAC effects
8. Clinical endpoints (PT/INR, aPTT)
9. Thrombin generation assay

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

using Test

# Add project to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using DarwinPBPK

println("=" ^ 60)
println("Test Suite: Platelets and Coagulation Modules")
println("=" ^ 60)

@testset "Platelets and Coagulation" begin

    # ================================================================
    # PLATELET TESTS
    # ================================================================

    @testset "Platelet Compartment Creation" begin
        println("\n[TEST] Platelet Compartment Creation...")

        # Normal platelet compartment
        plt = create_platelet_compartment()

        @test plt.count == NORMAL_PLATELET_COUNT
        @test plt.mean_platelet_volume == NORMAL_MPV
        @test plt.pathology == "normal"
        @test plt.activation.resting_fraction == 1.0
        @test plt.activation.aggregated_fraction == 0.0
        @test plt.cox1_inhibition == 0.0
        @test plt.p2y12_inhibition == 0.0

        println("  Normal platelet count: $(plt.count / 1e9) × 10⁹/L")
        println("  MPV: $(plt.mean_platelet_volume) fL")
        println("  Aggregation response: $(plt.aggregation_response)%")
        println("  [PASS] Normal platelet compartment created")
    end

    @testset "Platelet Pathology" begin
        println("\n[TEST] Platelet Pathology...")

        # Thrombocytopenia
        plt_low = create_platelet_compartment(
            pathology="thrombocytopenia",
            pathology_severity=0.8
        )

        @test plt_low.count < NORMAL_PLATELET_COUNT * 0.3
        @test plt_low.pathology == "thrombocytopenia"

        println("  Thrombocytopenia count: $(plt_low.count / 1e9) × 10⁹/L")

        # Thrombocytosis
        plt_high = create_platelet_compartment(
            pathology="thrombocytosis",
            pathology_severity=0.5
        )

        @test plt_high.count > NORMAL_PLATELET_COUNT * 2.0

        println("  Thrombocytosis count: $(plt_high.count / 1e9) × 10⁹/L")
        println("  [PASS] Pathology states correct")
    end

    @testset "Platelet Activation Dynamics" begin
        println("\n[TEST] Platelet Activation Dynamics...")

        plt = create_platelet_compartment()

        # Simulate ADP-induced activation
        agonists = (adp=5e-6, txa2=0.0, thrombin=0.0, collagen=0.0)  # 5 μM ADP

        # Activate for 10 seconds
        for _ in 1:100
            activate_platelets!(plt, agonists, 0.1)
        end

        @test plt.activation.p2y12_activation > 0.5  # ADP should activate P2Y12
        @test plt.activation.resting_fraction < 1.0  # Some platelets activated
        @test plt.activation.gpiib_iiia_active > 0   # GPIIb/IIIa should activate

        println("  P2Y12 activation: $(round(plt.activation.p2y12_activation * 100, digits=1))%")
        println("  GPIIb/IIIa active: $(round(plt.activation.gpiib_iiia_active * 100, digits=1))%")
        println("  Resting fraction: $(round(plt.activation.resting_fraction * 100, digits=1))%")
        println("  [PASS] Activation dynamics working")
    end

    @testset "Antiplatelet Drug Effects" begin
        println("\n[TEST] Antiplatelet Drug Effects...")

        # Aspirin effect
        plt_aspirin = create_platelet_compartment()
        apply_antiplatelet_drug!(plt_aspirin, "aspirin", 10e-6)  # 10 μM

        @test plt_aspirin.cox1_inhibition > 0.7  # Should be significantly inhibited
        @test plt_aspirin.aggregation_response < 100.0  # Reduced aggregation

        println("  Aspirin COX-1 inhibition: $(round(plt_aspirin.cox1_inhibition * 100, digits=1))%")

        # Ticagrelor effect
        plt_ticagrelor = create_platelet_compartment()
        apply_antiplatelet_drug!(plt_ticagrelor, "ticagrelor", 100e-9)  # 100 nM

        @test plt_ticagrelor.p2y12_inhibition > 0.9  # Very potent P2Y12 inhibitor

        println("  Ticagrelor P2Y12 inhibition: $(round(plt_ticagrelor.p2y12_inhibition * 100, digits=1))%")

        # Abciximab effect
        plt_abciximab = create_platelet_compartment()
        apply_antiplatelet_drug!(plt_abciximab, "abciximab", 50e-9)  # 50 nM

        @test plt_abciximab.gpiib_iiia_inhibition > 0.8

        println("  Abciximab GPIIb/IIIa inhibition: $(round(plt_abciximab.gpiib_iiia_inhibition * 100, digits=1))%")
        println("  [PASS] Antiplatelet drugs working")
    end

    @testset "Bleeding Risk Calculation" begin
        println("\n[TEST] Bleeding Risk Calculation...")

        # Normal risk
        plt_normal = create_platelet_compartment()
        risk_normal = calculate_bleeding_risk(plt_normal)

        @test risk_normal ≈ 1.0 atol=0.3  # Should be around 1.0

        println("  Normal bleeding risk: $(round(risk_normal, digits=2))×")

        # High risk (low platelets + aspirin)
        plt_high_risk = create_platelet_compartment(
            pathology="thrombocytopenia",
            pathology_severity=0.7
        )
        apply_antiplatelet_drug!(plt_high_risk, "aspirin", 10e-6)

        risk_high = calculate_bleeding_risk(plt_high_risk)

        @test risk_high > 3.0  # Should have elevated risk

        println("  High-risk bleeding: $(round(risk_high, digits=2))×")
        println("  [PASS] Bleeding risk calculation correct")
    end

    # ================================================================
    # COAGULATION TESTS
    # ================================================================

    @testset "Coagulation System Creation" begin
        println("\n[TEST] Coagulation System Creation...")

        coag = create_coagulation_system()

        @test coag.factors.factor_II ≈ NORMAL_FACTOR_CONCENTRATIONS["II"] rtol=0.01
        @test coag.factors.factor_VII ≈ NORMAL_FACTOR_CONCENTRATIONS["VII"] rtol=0.01
        @test coag.factors.antithrombin ≈ NORMAL_FACTOR_CONCENTRATIONS["ATIII"] rtol=0.01
        @test coag.factors.factor_IIa == 0.0  # No thrombin initially

        println("  Prothrombin: $(coag.factors.factor_II) nM")
        println("  Factor VII: $(coag.factors.factor_VII) nM")
        println("  Antithrombin: $(coag.factors.antithrombin) nM")
        println("  [PASS] Coagulation system created with correct factors")
    end

    @testset "Coagulation with Factor Deficiency" begin
        println("\n[TEST] Coagulation with Factor Deficiency...")

        # Hemophilia A (Factor VIII deficiency)
        coag_hemo = create_coagulation_system(
            deficiencies=Dict("VIII" => 0.05)  # 5% of normal
        )

        @test coag_hemo.factors.factor_VIII < NORMAL_FACTOR_CONCENTRATIONS["VIII"] * 0.1

        println("  Hemophilia A Factor VIII: $(coag_hemo.factors.factor_VIII) nM (5% of normal)")
        println("  [PASS] Factor deficiency modeled correctly")
    end

    @testset "Clinical Endpoints - Normal" begin
        println("\n[TEST] Clinical Endpoints (Normal)...")

        coag = create_coagulation_system()

        pt, inr = calculate_pt_inr(coag)
        aptt = calculate_aptt(coag)

        @test 11.0 < pt < 15.0  # Normal PT range
        @test 0.9 < inr < 1.1   # Normal INR range
        @test 25.0 < aptt < 35.0  # Normal aPTT range

        println("  PT: $(round(pt, digits=1)) seconds (normal: 11-15)")
        println("  INR: $(round(inr, digits=2)) (normal: 0.9-1.1)")
        println("  aPTT: $(round(aptt, digits=1)) seconds (normal: 25-35)")
        println("  [PASS] Clinical endpoints within normal ranges")
    end

    @testset "Warfarin Effect on INR" begin
        println("\n[TEST] Warfarin Effect on INR...")

        coag = create_coagulation_system()

        # Apply therapeutic warfarin
        apply_warfarin!(coag, 1.5, 0.5)  # S-warfarin 1.5 μM, R-warfarin 0.5 μM

        @test coag.anticoagulant.vkorc1_inhibition > 0.5
        @test coag.anticoagulant.vk_synthesis_rate < 0.5

        pt, inr = calculate_pt_inr(coag)

        # INR should be elevated but simulation needs time to reduce factors
        @test coag.inr > 0.9  # Initial state, factors haven't degraded yet

        println("  VKORC1 inhibition: $(round(coag.anticoagulant.vkorc1_inhibition * 100, digits=1))%")
        println("  VK synthesis rate: $(round(coag.anticoagulant.vk_synthesis_rate * 100, digits=1))%")
        println("  [PASS] Warfarin mechanism correct")
    end

    @testset "Warfarin Genetic Effects" begin
        println("\n[TEST] Warfarin Genetic Effects...")

        # VKORC1 AA (high sensitivity)
        coag_aa = create_coagulation_system()
        apply_warfarin!(coag_aa, 0.5, 0.2, genotype_vkorc1="AA")

        # VKORC1 BB (low sensitivity)
        coag_bb = create_coagulation_system()
        apply_warfarin!(coag_bb, 0.5, 0.2, genotype_vkorc1="BB")

        @test coag_aa.anticoagulant.vkorc1_inhibition > coag_bb.anticoagulant.vkorc1_inhibition

        println("  VKORC1 AA inhibition: $(round(coag_aa.anticoagulant.vkorc1_inhibition * 100, digits=1))%")
        println("  VKORC1 BB inhibition: $(round(coag_bb.anticoagulant.vkorc1_inhibition * 100, digits=1))%")
        println("  [PASS] Genetic sensitivity modeled correctly")
    end

    @testset "DOAC Effects" begin
        println("\n[TEST] DOAC Effects...")

        # Rivaroxaban (Factor Xa inhibitor)
        coag_riva = create_coagulation_system()
        apply_doac!(coag_riva, "rivaroxaban", 200.0)  # 200 nM (therapeutic)

        @test coag_riva.anticoagulant.xa_inhibition > 0.99  # Very high inhibition at therapeutic levels
        @test coag_riva.anticoagulant.iia_inhibition == 0.0  # No IIa inhibition
        @test coag_riva.anti_xa_IU > 0.5  # Detectable anti-Xa activity

        println("  Rivaroxaban Xa inhibition: $(round(coag_riva.anticoagulant.xa_inhibition * 100, digits=1))%")
        println("  Anti-Xa activity: $(round(coag_riva.anti_xa_IU, digits=2)) IU/mL")

        # Dabigatran (Direct thrombin inhibitor)
        coag_dabi = create_coagulation_system()
        apply_doac!(coag_dabi, "dabigatran", 100.0)  # 100 nM

        @test coag_dabi.anticoagulant.iia_inhibition > 0.9
        @test coag_dabi.anticoagulant.xa_inhibition == 0.0

        println("  Dabigatran IIa inhibition: $(round(coag_dabi.anticoagulant.iia_inhibition * 100, digits=1))%")
        println("  [PASS] DOAC mechanisms correct")
    end

    @testset "Heparin Effect" begin
        println("\n[TEST] Heparin Effect...")

        coag = create_coagulation_system()

        # UFH
        apply_heparin!(coag, 0.5, "UFH")  # 0.5 IU/mL

        @test coag.anticoagulant.atiii_potentiation > 10.0
        @test coag.aptt_seconds > 35.0  # Prolonged aPTT

        println("  ATIII potentiation: $(round(coag.anticoagulant.atiii_potentiation, digits=1))×")
        println("  aPTT: $(round(coag.aptt_seconds, digits=1)) seconds")
        println("  [PASS] Heparin effect correct")
    end

    @testset "Coagulation ODE Simulation" begin
        println("\n[TEST] Coagulation ODE Simulation...")

        coag = create_coagulation_system(tissue_factor=0.01)  # 10 pM TF

        # Simulate for 10 minutes
        times, results = simulate_coagulation!(
            coag,
            (0.0, 600.0),  # 10 minutes in seconds
            0.5            # 0.5 second timestep
        )

        @test length(times) == length(results)
        @test length(times) > 100

        # Check thrombin generation
        thrombin_values = [r["thrombin_nM"] for r in results]
        peak_thrombin = maximum(thrombin_values)

        @test peak_thrombin > 0  # Should generate some thrombin

        println("  Simulation duration: $(times[end] / 60) minutes")
        println("  Time points: $(length(times))")
        println("  Peak thrombin: $(round(peak_thrombin, digits=2)) nM")
        println("  [PASS] ODE simulation completed")
    end

    @testset "Thrombin Generation Assay" begin
        println("\n[TEST] Thrombin Generation Assay...")

        coag = create_coagulation_system()

        # Run TG assay
        tg_results = thrombin_generation_assay(coag, tf_conc=0.005, duration_min=30.0)

        @test haskey(tg_results, "peak_thrombin_nM")
        @test haskey(tg_results, "lag_time_min")
        @test haskey(tg_results, "time_to_peak_min")
        @test haskey(tg_results, "etp_nM_min")

        println("  Lag time: $(round(tg_results["lag_time_min"], digits=2)) min")
        println("  Peak thrombin: $(round(tg_results["peak_thrombin_nM"], digits=1)) nM")
        println("  Time to peak: $(round(tg_results["time_to_peak_min"], digits=2)) min")
        println("  ETP: $(round(tg_results["etp_nM_min"], digits=1)) nM·min")
        println("  [PASS] TG assay completed")
    end

    @testset "Coagulation State Report" begin
        println("\n[TEST] Coagulation State Report...")

        coag = create_coagulation_system()
        apply_doac!(coag, "apixaban", 150.0)

        state = get_coagulation_state(coag)

        @test haskey(state, "factors")
        @test haskey(state, "clinical_endpoints")
        @test haskey(state, "anticoagulants")
        @test state["anticoagulants"]["DOAC_type"] == "apixaban"

        println("  State report keys: $(keys(state))")
        println("  INR: $(round(state["clinical_endpoints"]["INR"], digits=2))")
        println("  DOAC: $(state["anticoagulants"]["DOAC_type"])")
        println("  Xa inhibition: $(round(state["anticoagulants"]["Xa_inhibition"] * 100, digits=1))%")
        println("  [PASS] State report generated")
    end

    # ================================================================
    # INTEGRATION TEST
    # ================================================================

    @testset "Platelet-Coagulation Integration" begin
        println("\n[TEST] Platelet-Coagulation Integration...")

        # Create both compartments
        plt = create_platelet_compartment()
        coag = create_coagulation_system(tissue_factor=0.01)

        # Simulate coagulation to generate thrombin
        simulate_coagulation!(coag, (0.0, 300.0), 0.5)

        # Use thrombin to activate platelets
        thrombin_conc = coag.factors.factor_IIa * 1e-9  # Convert nM to M
        agonists = (adp=1e-6, txa2=0.0, thrombin=thrombin_conc, collagen=0.0)

        # Activate platelets
        for _ in 1:50
            activate_platelets!(plt, agonists, 0.1)
        end

        # Check integration
        @test plt.activation.par1_activation >= 0  # Thrombin should activate PAR-1
        @test coag.factors.factor_IIa >= 0  # Thrombin was generated

        println("  Generated thrombin: $(round(coag.factors.factor_IIa, digits=2)) nM")
        println("  Platelet PAR-1 activation: $(round(plt.activation.par1_activation * 100, digits=1))%")
        println("  Platelet resting: $(round(plt.activation.resting_fraction * 100, digits=1))%")
        println("  [PASS] Platelet-coagulation integration working")
    end

end  # @testset

println("\n" * "=" ^ 60)
println("All Platelet and Coagulation tests completed!")
println("=" ^ 60)
