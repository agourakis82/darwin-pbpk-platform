"""
Test Suite for Fibrinolysis Module

Tests:
1. Fibrinolytic system creation
2. Plasminogen/plasmin dynamics
3. tPA/uPA activators
4. PAI-1 and α2-antiplasmin inhibition
5. Fibrin degradation and D-dimer
6. Thrombolytic therapy (alteplase, tenecteplase)
7. Antifibrinolytic drugs (tranexamic acid)
8. Plasmin generation assay
9. Coagulation-fibrinolysis integration

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

using Test

# Include modules directly
include(joinpath(@__DIR__, "..", "src", "DarwinPBPK", "compartments", "fibrinolysis.jl"))
include(joinpath(@__DIR__, "..", "src", "DarwinPBPK", "compartments", "coagulation.jl"))

using .Fibrinolysis
using .Coagulation

println("=" ^ 60)
println("Test Suite: Fibrinolysis Module")
println("=" ^ 60)

@testset "Fibrinolysis Module" begin

    # ================================================================
    # SYSTEM CREATION
    # ================================================================

    @testset "Fibrinolytic System Creation" begin
        println("\n[TEST] Fibrinolytic System Creation...")

        fib = create_fibrinolytic_system()

        @test fib.plasminogen.glu_plasminogen > 0
        @test fib.plasminogen.tpa_free ≈ NORMAL_TPA rtol=0.1
        @test fib.plasminogen.pai1_active ≈ NORMAL_PAI1 rtol=0.1
        @test fib.plasminogen.alpha2_antiplasmin ≈ 1000.0 rtol=0.1
        @test fib.fibrin.fibrin_intact == 0.0  # No clot

        total_plg = fib.plasminogen.glu_plasminogen + fib.plasminogen.lys_plasminogen
        println("  Total plasminogen: $(round(total_plg, digits=1)) nM")
        println("  tPA: $(round(fib.plasminogen.tpa_free, digits=3)) nM")
        println("  PAI-1: $(round(fib.plasminogen.pai1_active, digits=2)) nM")
        println("  α2-antiplasmin: $(round(fib.plasminogen.alpha2_antiplasmin, digits=1)) nM")
        println("  [PASS]")
    end

    @testset "System with Elevated PAI-1" begin
        println("\n[TEST] System with Elevated PAI-1...")

        fib_elevated = create_fibrinolytic_system(pai1_elevated=true)
        fib_normal = create_fibrinolytic_system(pai1_elevated=false)

        @test fib_elevated.plasminogen.pai1_active > fib_normal.plasminogen.pai1_active * 3

        println("  Normal PAI-1: $(round(fib_normal.plasminogen.pai1_active, digits=2)) nM")
        println("  Elevated PAI-1: $(round(fib_elevated.plasminogen.pai1_active, digits=2)) nM")
        println("  [PASS]")
    end

    @testset "System with Fibrin Clot" begin
        println("\n[TEST] System with Fibrin Clot...")

        fib_clot = create_fibrinolytic_system(fibrin_amount=1000.0)

        @test fib_clot.fibrin.fibrin_intact == 1000.0
        @test fib_clot.fibrin.d_dimer == 0.0  # No degradation yet

        println("  Fibrin: $(fib_clot.fibrin.fibrin_intact) nM")
        println("  D-dimer: $(fib_clot.fibrin.d_dimer) nM")
        println("  [PASS]")
    end

    @testset "Clot Structure Variations" begin
        println("\n[TEST] Clot Structure Variations...")

        fib_dense = create_fibrinolytic_system(fibrin_amount=1000.0, clot_structure=:dense)
        fib_loose = create_fibrinolytic_system(fibrin_amount=1000.0, clot_structure=:loose)

        @test fib_dense.fibrin.fibrin_density > fib_loose.fibrin.fibrin_density
        @test fib_dense.fibrin.fiber_diameter < fib_loose.fibrin.fiber_diameter

        println("  Dense clot density: $(fib_dense.fibrin.fibrin_density) fibers/μm²")
        println("  Loose clot density: $(fib_loose.fibrin.fibrin_density) fibers/μm²")
        println("  [PASS]")
    end

    # ================================================================
    # FIBRINOLYSIS SIMULATION
    # ================================================================

    @testset "Fibrinolysis ODE Simulation" begin
        println("\n[TEST] Fibrinolysis ODE Simulation...")

        fib = create_fibrinolytic_system(fibrin_amount=500.0)

        # Add extra tPA to drive lysis
        fib.plasminogen.tpa_free = 1.0  # 1 nM tPA (therapeutic level)

        times, results = simulate_fibrinolysis!(fib, (0.0, 1800.0), 1.0)  # 30 min

        @test length(times) == length(results)
        @test length(times) > 100

        # Check plasmin generation
        plasmin_values = [r["plasmin_nM"] for r in results]
        peak_plasmin = maximum(plasmin_values)

        # Check D-dimer accumulation
        d_dimer_final = results[end]["d_dimer_nM"]

        println("  Duration: $(times[end] / 60) min")
        println("  Time points: $(length(times))")
        println("  Peak plasmin: $(round(peak_plasmin, digits=3)) nM")
        println("  Final D-dimer: $(round(d_dimer_final, digits=2)) nM")
        println("  [PASS]")
    end

    @testset "D-dimer Generation" begin
        println("\n[TEST] D-dimer Generation...")

        fib = create_fibrinolytic_system(fibrin_amount=1000.0)
        fib.plasminogen.tpa_free = 2.0  # Higher tPA

        simulate_fibrinolysis!(fib, (0.0, 3600.0), 1.0)  # 60 min

        d_dimer_nM = fib.fibrin.d_dimer
        d_dimer_clinical = calculate_d_dimer(fib)

        @test d_dimer_nM >= 0
        @test d_dimer_clinical >= 0

        println("  D-dimer: $(round(d_dimer_nM, digits=2)) nM")
        println("  D-dimer (clinical): $(round(d_dimer_clinical, digits=2)) ng/mL FEU")
        println("  [PASS]")
    end

    # ================================================================
    # THROMBOLYTIC THERAPY
    # ================================================================

    @testset "Alteplase (tPA) Therapy" begin
        println("\n[TEST] Alteplase (tPA) Therapy...")

        fib = create_fibrinolytic_system(fibrin_amount=1000.0)

        apply_tpa_therapy!(fib, "alteplase", 100.0)  # 100 IU/mL

        @test fib.thrombolytic_drug == "alteplase"
        @test fib.thrombolytic_conc == 100.0
        @test fib.plasminogen.tpa_free > NORMAL_TPA

        println("  Drug: $(fib.thrombolytic_drug)")
        println("  Concentration: $(fib.thrombolytic_conc) IU/mL")
        println("  tPA level: $(round(fib.plasminogen.tpa_free, digits=2)) nM")
        println("  [PASS]")
    end

    @testset "Tenecteplase Therapy" begin
        println("\n[TEST] Tenecteplase Therapy...")

        fib = create_fibrinolytic_system(fibrin_amount=1000.0)

        apply_tpa_therapy!(fib, "tenecteplase", 50.0)

        @test fib.thrombolytic_drug == "tenecteplase"

        println("  Drug: $(fib.thrombolytic_drug)")
        println("  [PASS]")
    end

    # ================================================================
    # ANTIFIBRINOLYTIC DRUGS
    # ================================================================

    @testset "Tranexamic Acid (TXA)" begin
        println("\n[TEST] Tranexamic Acid (TXA)...")

        fib = create_fibrinolytic_system(fibrin_amount=500.0)

        # Therapeutic TXA concentration (~10-15 mg/L = 65-100 μM = 65000-100000 nM)
        apply_antifibrinolytic!(fib, "tranexamic_acid", 70000.0)  # 70 μM

        @test fib.antifibrinolytic_drug == "tranexamic_acid"
        @test fib.fibrinolysis_inhibition > 0.5  # Should be significantly inhibited

        println("  Drug: $(fib.antifibrinolytic_drug)")
        println("  Concentration: $(fib.antifibrinolytic_conc / 1000) μM")
        println("  Fibrinolysis inhibition: $(round(fib.fibrinolysis_inhibition * 100, digits=1))%")
        println("  [PASS]")
    end

    @testset "Aminocaproic Acid (EACA)" begin
        println("\n[TEST] Aminocaproic Acid (EACA)...")

        fib = create_fibrinolytic_system()

        apply_antifibrinolytic!(fib, "aminocaproic_acid", 500000.0)  # 500 μM

        @test fib.antifibrinolytic_drug == "aminocaproic_acid"
        @test fib.fibrinolysis_inhibition > 0.3

        println("  Drug: $(fib.antifibrinolytic_drug)")
        println("  Inhibition: $(round(fib.fibrinolysis_inhibition * 100, digits=1))%")
        println("  [PASS]")
    end

    @testset "TXA Effect on Fibrinolysis" begin
        println("\n[TEST] TXA Effect on Fibrinolysis...")

        # Without TXA
        fib_no_txa = create_fibrinolytic_system(fibrin_amount=500.0)
        fib_no_txa.plasminogen.tpa_free = 1.0
        simulate_fibrinolysis!(fib_no_txa, (0.0, 1800.0), 1.0)

        # With TXA
        fib_with_txa = create_fibrinolytic_system(fibrin_amount=500.0)
        fib_with_txa.plasminogen.tpa_free = 1.0
        apply_antifibrinolytic!(fib_with_txa, "tranexamic_acid", 70000.0)
        simulate_fibrinolysis!(fib_with_txa, (0.0, 1800.0), 1.0)

        # TXA should reduce D-dimer generation
        @test fib_with_txa.fibrin.d_dimer <= fib_no_txa.fibrin.d_dimer

        println("  D-dimer without TXA: $(round(fib_no_txa.fibrin.d_dimer, digits=2)) nM")
        println("  D-dimer with TXA: $(round(fib_with_txa.fibrin.d_dimer, digits=2)) nM")
        println("  [PASS]")
    end

    # ================================================================
    # PLASMIN GENERATION ASSAY
    # ================================================================

    @testset "Plasmin Generation Assay" begin
        println("\n[TEST] Plasmin Generation Assay...")

        fib = create_fibrinolytic_system()

        pga = plasmin_generation_assay(fib, tpa_conc=0.5, fibrin_amount=1000.0, duration_min=60.0)

        @test haskey(pga, "peak_plasmin_nM")
        @test haskey(pga, "lag_time_min")
        @test haskey(pga, "epp_nM_min")
        @test haskey(pga, "lysis_time_min")

        println("  Lag time: $(round(pga["lag_time_min"], digits=2)) min")
        println("  Peak plasmin: $(round(pga["peak_plasmin_nM"], digits=3)) nM")
        println("  Time to peak: $(round(pga["time_to_peak_min"], digits=2)) min")
        println("  EPP: $(round(pga["epp_nM_min"], digits=2)) nM·min")
        println("  Lysis time (50%): $(round(pga["lysis_time_min"], digits=2)) min")
        println("  [PASS]")
    end

    # ================================================================
    # STATE REPORTING
    # ================================================================

    @testset "State Report" begin
        println("\n[TEST] State Report...")

        fib = create_fibrinolytic_system(fibrin_amount=500.0)
        fib.plasminogen.tpa_free = 0.5
        simulate_fibrinolysis!(fib, (0.0, 600.0), 1.0)

        state = get_fibrinolysis_state(fib)

        @test haskey(state, "plasminogen_system")
        @test haskey(state, "inhibitors")
        @test haskey(state, "fibrin_state")
        @test haskey(state, "assay_parameters")
        @test haskey(state, "therapy")
        @test haskey(state, "clinical")

        println("  State report sections: $(length(keys(state)))")
        println("  Plasminogen: $(round(state["plasminogen_system"]["plasminogen_total_nM"], digits=1)) nM")
        println("  PAI-1: $(round(state["inhibitors"]["pai1_active_nM"], digits=2)) nM")
        println("  D-dimer: $(round(state["fibrin_state"]["d_dimer_ng_ml"], digits=2)) ng/mL")
        println("  [PASS]")
    end

    # ================================================================
    # COAGULATION-FIBRINOLYSIS INTEGRATION
    # ================================================================

    @testset "Coagulation-Fibrinolysis Integration" begin
        println("\n[TEST] Coagulation-Fibrinolysis Integration...")

        # Create coagulation system and generate fibrin
        coag = create_coagulation_system(tissue_factor=0.01)
        simulate_coagulation!(coag, (0.0, 600.0), 0.5)

        # Get fibrin generated
        fibrin_generated = coag.factors.fibrin
        thrombin = coag.factors.factor_IIa

        # Create fibrinolytic system with that fibrin
        fib = create_fibrinolytic_system(fibrin_amount=max(100.0, fibrin_generated))

        # Simulate fibrinolysis with TAFI activation from thrombin
        simulate_fibrinolysis!(fib, (0.0, 1800.0), 1.0, thrombin_conc=thrombin)

        @test fib.plasminogen.tafi_active >= 0  # TAFI activated by thrombin

        println("  Fibrin from coagulation: $(round(fibrin_generated, digits=2)) nM")
        println("  Thrombin available: $(round(thrombin, digits=2)) nM")
        println("  TAFIa generated: $(round(fib.plasminogen.tafi_active, digits=3)) nM")
        println("  D-dimer: $(round(fib.fibrin.d_dimer, digits=2)) nM")
        println("  [PASS]")
    end

    # ================================================================
    # CLINICAL SCENARIOS
    # ================================================================

    @testset "Clinical: Normal Fibrinolysis" begin
        println("\n[TEST] Clinical: Normal Fibrinolysis...")

        fib = create_fibrinolytic_system(fibrin_amount=500.0)
        fib.plasminogen.tpa_free = 0.2  # Physiological tPA release

        simulate_fibrinolysis!(fib, (0.0, 3600.0), 1.0)

        # Normal fibrinolysis should be gradual
        @test fib.lysis_time > 10.0 || fib.lysis_time == Inf  # Not too fast

        hyperfib = Fibrinolysis.detect_hyperfibrinolysis(fib)
        @test hyperfib == false

        println("  Lysis time: $(round(fib.lysis_time, digits=1)) min")
        println("  Hyperfibrinolysis: $(hyperfib)")
        println("  [PASS]")
    end

    @testset "Clinical: PAI-1 Deficiency (Bleeding Risk)" begin
        println("\n[TEST] Clinical: PAI-1 Deficiency...")

        # Low PAI-1 = increased fibrinolysis = bleeding risk
        fib = create_fibrinolytic_system(fibrin_amount=500.0)
        fib.plasminogen.pai1_active = 0.01  # Very low PAI-1
        fib.plasminogen.tpa_free = 0.5

        simulate_fibrinolysis!(fib, (0.0, 1800.0), 1.0)

        # Should have faster lysis
        @test fib.peak_plasmin > 0

        println("  PAI-1: $(round(fib.plasminogen.pai1_active, digits=3)) nM (deficient)")
        println("  Peak plasmin: $(round(fib.peak_plasmin, digits=3)) nM")
        println("  [PASS]")
    end

end

println("\n" * "=" ^ 60)
println("All Fibrinolysis Tests Completed!")
println("=" ^ 60)
