"""
Standalone Test for Platelets and Coagulation Modules

Loads modules directly without full DarwinPBPK package.
"""

using Test

# Include modules directly
include(joinpath(@__DIR__, "..", "src", "DarwinPBPK", "compartments", "platelets.jl"))
include(joinpath(@__DIR__, "..", "src", "DarwinPBPK", "compartments", "coagulation.jl"))

using .Platelets
using .Coagulation

println("=" ^ 60)
println("Standalone Test: Platelets and Coagulation Modules")
println("=" ^ 60)

@testset "Platelets and Coagulation - Standalone" begin

    # ================================================================
    # PLATELET TESTS
    # ================================================================

    @testset "Platelet Compartment Creation" begin
        println("\n[TEST] Platelet Compartment Creation...")

        plt = create_platelet_compartment()

        @test plt.count == NORMAL_PLATELET_COUNT
        @test plt.mean_platelet_volume == NORMAL_MPV
        @test plt.pathology == "normal"
        @test plt.activation.resting_fraction == 1.0

        println("  Platelet count: $(plt.count / 1e9) × 10⁹/L")
        println("  MPV: $(plt.mean_platelet_volume) fL")
        println("  [PASS]")
    end

    @testset "Platelet Pathology" begin
        println("\n[TEST] Platelet Pathology...")

        # Thrombocytopenia
        plt_low = create_platelet_compartment(
            pathology="thrombocytopenia",
            pathology_severity=0.8
        )
        @test plt_low.count < NORMAL_PLATELET_COUNT * 0.3
        println("  Thrombocytopenia: $(plt_low.count / 1e9) × 10⁹/L")

        # Thrombocytosis
        plt_high = create_platelet_compartment(
            pathology="thrombocytosis",
            pathology_severity=0.5
        )
        @test plt_high.count > NORMAL_PLATELET_COUNT * 2.0
        println("  Thrombocytosis: $(plt_high.count / 1e9) × 10⁹/L")
        println("  [PASS]")
    end

    @testset "Platelet Activation" begin
        println("\n[TEST] Platelet Activation...")

        plt = create_platelet_compartment()
        agonists = (adp=5e-6, txa2=0.0, thrombin=0.0, collagen=0.0)

        for _ in 1:100
            activate_platelets!(plt, agonists, 0.1)
        end

        @test plt.activation.p2y12_activation > 0.5
        @test plt.activation.resting_fraction < 1.0

        println("  P2Y12 activation: $(round(plt.activation.p2y12_activation * 100, digits=1))%")
        println("  Resting: $(round(plt.activation.resting_fraction * 100, digits=1))%")
        println("  [PASS]")
    end

    @testset "Antiplatelet Drugs" begin
        println("\n[TEST] Antiplatelet Drugs...")

        # Aspirin
        plt_aspirin = create_platelet_compartment()
        apply_antiplatelet_drug!(plt_aspirin, "aspirin", 10e-6)
        @test plt_aspirin.cox1_inhibition > 0.7
        println("  Aspirin COX-1 inhibition: $(round(plt_aspirin.cox1_inhibition * 100, digits=1))%")

        # Ticagrelor
        plt_tica = create_platelet_compartment()
        apply_antiplatelet_drug!(plt_tica, "ticagrelor", 100e-9)
        @test plt_tica.p2y12_inhibition > 0.9
        println("  Ticagrelor P2Y12 inhibition: $(round(plt_tica.p2y12_inhibition * 100, digits=1))%")

        println("  [PASS]")
    end

    @testset "Bleeding Risk" begin
        println("\n[TEST] Bleeding Risk...")

        plt_normal = create_platelet_compartment()
        risk_normal = calculate_bleeding_risk(plt_normal)
        @test risk_normal < 2.0
        println("  Normal risk: $(round(risk_normal, digits=2))×")

        plt_high_risk = create_platelet_compartment(
            pathology="thrombocytopenia", pathology_severity=0.7
        )
        apply_antiplatelet_drug!(plt_high_risk, "aspirin", 10e-6)
        risk_high = calculate_bleeding_risk(plt_high_risk)
        @test risk_high > 3.0
        println("  High risk: $(round(risk_high, digits=2))×")

        println("  [PASS]")
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

        println("  Prothrombin: $(coag.factors.factor_II) nM")
        println("  Factor VII: $(coag.factors.factor_VII) nM")
        println("  ATIII: $(coag.factors.antithrombin) nM")
        println("  [PASS]")
    end

    @testset "Clinical Endpoints - Normal" begin
        println("\n[TEST] Clinical Endpoints (Normal)...")

        coag = create_coagulation_system()
        pt, inr = calculate_pt_inr(coag)
        aptt = calculate_aptt(coag)

        @test 11.0 < pt < 15.0
        @test 0.9 < inr < 1.2
        @test 25.0 < aptt < 40.0

        println("  PT: $(round(pt, digits=1)) s")
        println("  INR: $(round(inr, digits=2))")
        println("  aPTT: $(round(aptt, digits=1)) s")
        println("  [PASS]")
    end

    @testset "Warfarin Effect" begin
        println("\n[TEST] Warfarin Effect...")

        coag = create_coagulation_system()
        apply_warfarin!(coag, 1.5, 0.5)

        @test coag.anticoagulant.vkorc1_inhibition > 0.5
        @test coag.anticoagulant.vk_synthesis_rate < 0.5

        println("  VKORC1 inhibition: $(round(coag.anticoagulant.vkorc1_inhibition * 100, digits=1))%")
        println("  VK synthesis: $(round(coag.anticoagulant.vk_synthesis_rate * 100, digits=1))%")
        println("  [PASS]")
    end

    @testset "Warfarin Genetics" begin
        println("\n[TEST] Warfarin Genetics...")

        coag_aa = create_coagulation_system()
        apply_warfarin!(coag_aa, 0.5, 0.2, genotype_vkorc1="AA")

        coag_bb = create_coagulation_system()
        apply_warfarin!(coag_bb, 0.5, 0.2, genotype_vkorc1="BB")

        @test coag_aa.anticoagulant.vkorc1_inhibition > coag_bb.anticoagulant.vkorc1_inhibition

        println("  AA inhibition: $(round(coag_aa.anticoagulant.vkorc1_inhibition * 100, digits=1))%")
        println("  BB inhibition: $(round(coag_bb.anticoagulant.vkorc1_inhibition * 100, digits=1))%")
        println("  [PASS]")
    end

    @testset "DOAC Effects" begin
        println("\n[TEST] DOAC Effects...")

        # Rivaroxaban
        coag_riva = create_coagulation_system()
        apply_doac!(coag_riva, "rivaroxaban", 200.0)
        @test coag_riva.anticoagulant.xa_inhibition > 0.99
        @test coag_riva.anticoagulant.iia_inhibition == 0.0
        println("  Rivaroxaban Xa inhibition: $(round(coag_riva.anticoagulant.xa_inhibition * 100, digits=1))%")

        # Apixaban
        coag_apix = create_coagulation_system()
        apply_doac!(coag_apix, "apixaban", 100.0)
        @test coag_apix.anticoagulant.xa_inhibition > 0.99
        println("  Apixaban Xa inhibition: $(round(coag_apix.anticoagulant.xa_inhibition * 100, digits=1))%")

        # Dabigatran
        coag_dabi = create_coagulation_system()
        apply_doac!(coag_dabi, "dabigatran", 100.0)
        @test coag_dabi.anticoagulant.iia_inhibition > 0.9
        @test coag_dabi.anticoagulant.xa_inhibition == 0.0
        println("  Dabigatran IIa inhibition: $(round(coag_dabi.anticoagulant.iia_inhibition * 100, digits=1))%")

        println("  [PASS]")
    end

    @testset "Heparin Effect" begin
        println("\n[TEST] Heparin Effect...")

        coag = create_coagulation_system()
        apply_heparin!(coag, 0.5, "UFH")

        @test coag.anticoagulant.atiii_potentiation > 10.0
        @test coag.aptt_seconds > 35.0

        println("  ATIII potentiation: $(round(coag.anticoagulant.atiii_potentiation, digits=1))×")
        println("  aPTT: $(round(coag.aptt_seconds, digits=1)) s")
        println("  [PASS]")
    end

    @testset "Coagulation ODE Simulation" begin
        println("\n[TEST] Coagulation ODE Simulation...")

        coag = create_coagulation_system(tissue_factor=0.01)

        times, results = simulate_coagulation!(coag, (0.0, 600.0), 0.5)

        @test length(times) == length(results)
        @test length(times) > 100

        thrombin_values = [r["thrombin_nM"] for r in results]
        peak = maximum(thrombin_values)

        println("  Duration: $(times[end] / 60) min")
        println("  Time points: $(length(times))")
        println("  Peak thrombin: $(round(peak, digits=2)) nM")
        println("  [PASS]")
    end

    @testset "Thrombin Generation Assay" begin
        println("\n[TEST] Thrombin Generation Assay...")

        coag = create_coagulation_system()
        tg = thrombin_generation_assay(coag, tf_conc=0.005, duration_min=30.0)

        @test haskey(tg, "peak_thrombin_nM")
        @test haskey(tg, "lag_time_min")
        @test haskey(tg, "etp_nM_min")

        println("  Lag time: $(round(tg["lag_time_min"], digits=2)) min")
        println("  Peak: $(round(tg["peak_thrombin_nM"], digits=1)) nM")
        println("  ETP: $(round(tg["etp_nM_min"], digits=1)) nM·min")
        println("  [PASS]")
    end

    @testset "State Reports" begin
        println("\n[TEST] State Reports...")

        plt = create_platelet_compartment()
        plt_state = get_platelet_state(plt)
        @test haskey(plt_state, "count")
        @test haskey(plt_state, "activation")
        println("  Platelet state keys: $(length(keys(plt_state)))")

        coag = create_coagulation_system()
        coag_state = get_coagulation_state(coag)
        @test haskey(coag_state, "factors")
        @test haskey(coag_state, "clinical_endpoints")
        println("  Coagulation state keys: $(length(keys(coag_state)))")

        println("  [PASS]")
    end

    # ================================================================
    # INTEGRATION
    # ================================================================

    @testset "Platelet-Coagulation Integration" begin
        println("\n[TEST] Platelet-Coagulation Integration...")

        plt = create_platelet_compartment()
        coag = create_coagulation_system(tissue_factor=0.01)

        # Generate thrombin
        simulate_coagulation!(coag, (0.0, 300.0), 0.5)

        # Use thrombin to activate platelets
        thrombin_M = coag.factors.factor_IIa * 1e-9
        agonists = (adp=1e-6, txa2=0.0, thrombin=thrombin_M, collagen=0.0)

        for _ in 1:50
            activate_platelets!(plt, agonists, 0.1)
        end

        println("  Thrombin generated: $(round(coag.factors.factor_IIa, digits=2)) nM")
        println("  PAR-1 activation: $(round(plt.activation.par1_activation * 100, digits=1))%")
        println("  [PASS]")
    end

end

println("\n" * "=" ^ 60)
println("All Standalone Tests Completed Successfully!")
println("=" ^ 60)
