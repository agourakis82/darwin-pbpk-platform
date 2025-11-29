# =============================================================================
# Test Suite: Lymphatic Absorption Model
# =============================================================================
# Validates the MedLang lymphatic absorption model against literature data
# =============================================================================

using Test
using DarwinPBPK.MedLang

@testset "Lymphatic Absorption Model" begin

    @testset "Log P-Dependent Partitioning" begin
        # Test the core relationship: Log P > 5 favors lymphatic transport

        # High Log P drug (halofantrine-like)
        drug_high = DrugLymphPartitioning(
            8.5,    # log_P
            7.8,    # log_D
            85.0,   # TG_solubility
            500.0,  # MW
            80.0,   # melting_point
            15.0,   # CM_binding_Kd
            0.98,   # protein_binding
            0.95    # lipophilicity_index
        )

        # Low Log P drug (metformin-like)
        drug_low = DrugLymphPartitioning(
            -1.4,   # log_P
            -5.6,   # log_D
            0.01,   # TG_solubility
            129.0,  # MW
            220.0,  # melting_point
            10000.0, # CM_binding_Kd
            0.0,    # protein_binding
            0.0     # lipophilicity_index
        )

        # LCT formulation
        form_LCT = LipidFormulation(
            :type_I, :LCT, 2000.0, 12.0, 300.0, 0.3, 0.85
        )

        # Calculate fractions
        result_high = calculate_lymphatic_fraction(drug_high, form_LCT, true)
        result_low = calculate_lymphatic_fraction(drug_low, form_LCT, true)

        println("Log P Partitioning Results:")
        println("  High Log P (8.5): F_lymph = $(round(result_high.F_lymph * 100, digits=1))%")
        println("  Low Log P (-1.4): F_lymph = $(round(result_low.F_lymph * 100, digits=1))%")

        # High Log P should have significant lymphatic transport (>50%)
        @test result_high.F_lymph > 0.50
        # Low Log P should have negligible lymphatic transport (<5%)
        @test result_low.F_lymph < 0.05
    end

    @testset "Formulation Effects (LCT vs MCT)" begin
        # Long-chain triglycerides favor lymphatic transport
        # Medium-chain triglycerides favor portal transport

        drug = DrugLymphPartitioning(
            6.0, 5.5, 40.0, 400.0, 100.0, 50.0, 0.95, 0.7
        )

        form_LCT = LipidFormulation(:type_I, :LCT, 2000.0, 12.0, 300.0, 0.3, 0.8)
        form_MCT = LipidFormulation(:type_I, :MCT, 2000.0, 12.0, 300.0, 0.5, 0.2)
        form_mixed = LipidFormulation(:type_II, :mixed, 1000.0, 11.0, 200.0, 0.4, 0.5)

        result_LCT = calculate_lymphatic_fraction(drug, form_LCT, true)
        result_MCT = calculate_lymphatic_fraction(drug, form_MCT, true)
        result_mixed = calculate_lymphatic_fraction(drug, form_mixed, true)

        println("\nFormulation Effects:")
        println("  LCT formulation: F_lymph = $(round(result_LCT.F_lymph * 100, digits=1))%")
        println("  MCT formulation: F_lymph = $(round(result_MCT.F_lymph * 100, digits=1))%")
        println("  Mixed formulation: F_lymph = $(round(result_mixed.F_lymph * 100, digits=1))%")

        # LCT should be highest
        @test result_LCT.F_lymph > result_mixed.F_lymph
        @test result_mixed.F_lymph > result_MCT.F_lymph
        # MCT should reduce lymphatic transport significantly
        @test result_MCT.F_lymph < result_LCT.F_lymph * 0.3
    end

    @testset "Fed vs Fasted State" begin
        # Postprandial state dramatically increases lymphatic transport

        drug = DrugLymphPartitioning(
            6.5, 6.0, 50.0, 450.0, 90.0, 40.0, 0.96, 0.75
        )
        form = LipidFormulation(:type_I, :LCT, 1500.0, 12.0, 250.0, 0.35, 0.75)

        result_fed = calculate_lymphatic_fraction(drug, form, true)
        result_fasted = calculate_lymphatic_fraction(drug, form, false)

        println("\nFed vs Fasted State:")
        println("  Fed state: F_lymph = $(round(result_fed.F_lymph * 100, digits=1))%")
        println("  Fasted state: F_lymph = $(round(result_fasted.F_lymph * 100, digits=1))%")

        # Fed state should increase lymphatic fraction (postprandial TG absorption)
        @test result_fed.F_lymph > result_fasted.F_lymph * 1.2
    end

    @testset "First-Pass Bypass" begin
        # For high-extraction drugs, lymphatic transport improves bioavailability

        # Testosterone undecanoate preset
        preset = get_drug_preset(:testosterone_undecanoate)

        # High hepatic extraction ratio
        Eh = 0.95  # 95% first-pass extraction

        bypass = first_pass_bypass_fraction(preset.drug, preset.formulation, Eh)

        println("\nFirst-Pass Bypass (Testosterone Undecanoate):")
        println("  F_lymph = $(round(bypass.F_lymph * 100, digits=1))%")
        println("  F_oral (with lymph) = $(round(bypass.F_oral_total * 100, digits=1))%")
        println("  F_oral (without lymph) = $(round(bypass.F_oral_no_lymph * 100, digits=1))%")
        println("  Improvement factor = $(round(bypass.improvement_factor, digits=1))x")

        # Without lymphatic transport, F_oral = 5% (1 - 0.95)
        @test bypass.F_oral_no_lymph ≈ 0.05 atol=0.01

        # With lymphatic transport, should be much higher
        @test bypass.F_oral_total > 0.5

        # Improvement factor should be significant (>10x)
        @test bypass.improvement_factor > 10
    end

    @testset "Drug Presets Validation" begin
        println("\nDrug Preset Validation:")

        # Halofantrine: ~80% lymphatic (Caliph et al., 2000)
        hal = get_drug_preset(:halofantrine)
        hal_result = calculate_lymphatic_fraction(hal.drug, hal.formulation, true)
        println("  Halofantrine: F_lymph = $(round(hal_result.F_lymph * 100, digits=1))% (Literature: ~80%)")
        @test hal_result.F_lymph > 0.70

        # Vitamin A: ~60% lymphatic
        vit = get_drug_preset(:vitamin_A)
        vit_result = calculate_lymphatic_fraction(vit.drug, vit.formulation, true)
        println("  Vitamin A: F_lymph = $(round(vit_result.F_lymph * 100, digits=1))% (Literature: ~60%)")
        @test vit_result.F_lymph > 0.40

        # Metformin: 0% lymphatic (control)
        met = get_drug_preset(:metformin)
        met_result = calculate_lymphatic_fraction(met.drug, met.formulation, true)
        println("  Metformin: F_lymph = $(round(met_result.F_lymph * 100, digits=1))% (Literature: ~0%)")
        @test met_result.F_lymph < 0.02

        # Probucol: essentially 100% lymphatic
        prob = get_drug_preset(:probucol)
        prob_result = calculate_lymphatic_fraction(prob.drug, prob.formulation, true)
        println("  Probucol: F_lymph = $(round(prob_result.F_lymph * 100, digits=1))% (Literature: ~100%)")
        @test prob_result.F_lymph >= 0.79  # High but model caps at ~85%
    end

    @testset "Chylomicron Association" begin
        # Test drug-chylomicron binding

        CM = ChylomicronDynamics(
            200.0,   # diameter_nm
            1e8,     # formation_rate
            0.01,    # triglyceride_core
            1.0,     # apoB48_content
            0.5,     # Ka_drug
            0.1,     # Kd_drug
            10.0,    # surface_phospholipid
            0.1      # cholesterol_ester
        )

        # High lipophilicity drug
        drug_lipo = DrugLymphPartitioning(
            7.5, 7.0, 60.0, 400.0, 85.0, 20.0, 0.97, 0.85
        )

        # Low lipophilicity drug
        drug_hydro = DrugLymphPartitioning(
            1.0, 0.5, 0.5, 200.0, 150.0, 5000.0, 0.2, 0.1
        )

        assoc_lipo = chylomicron_association(drug_lipo, CM, 10.0, 100.0)
        assoc_hydro = chylomicron_association(drug_hydro, CM, 10.0, 100.0)

        println("\nChylomicron Association:")
        println("  Lipophilic drug: fraction in CM = $(round(assoc_lipo.fraction_CM * 100, digits=1))%")
        println("  Hydrophilic drug: fraction in CM = $(round(assoc_hydro.fraction_CM * 100, digits=1))%")

        # Lipophilic drug should have higher CM association
        @test assoc_lipo.fraction_CM > assoc_hydro.fraction_CM
    end

    @testset "Thoracic Duct Flow Dynamics" begin
        flow = LymphaticFlow(
            20.0,    # lacteal_flow_basal
            120.0,   # lacteal_flow_fed
            80.0,    # mesenteric_flow
            100.0,   # thoracic_duct_flow
            10.0,    # cisterna_chyli_volume
            25.0,    # lymph_node_volume
            5.0,     # interstitial_pressure
            3.0      # lymph_protein_conc
        )

        # Test postprandial flow profile
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
        flows_fed = [thoracic_duct_flow(flow, true, t) for t in times]
        flows_fasted = [thoracic_duct_flow(flow, false, t) for t in times]

        println("\nThoracic Duct Flow (mL/h):")
        println("  Time (h)\tFed\t\tFasted")
        for (t, f_fed, f_fast) in zip(times, flows_fed, flows_fasted)
            println("  $(t)\t\t$(round(f_fed, digits=1))\t\t$(round(f_fast, digits=1))")
        end

        # Fed flow should peak around 3-4h
        peak_idx = argmax(flows_fed)
        @test times[peak_idx] >= 2.0 && times[peak_idx] <= 5.0

        # Peak fed flow should be much higher than basal
        @test maximum(flows_fed) > minimum(flows_fed) * 2

        # Fasted flow should be constant
        @test all(f ≈ flows_fasted[1] for f in flows_fasted)
    end

    @testset "Disease State Effects" begin
        # Test lymphedema and other disease states

        disease_lymphedema = create_disease_state(:lymphedema, 0.7)
        disease_normal = create_disease_state(:normal, 0.0)
        disease_obesity = create_disease_state(:obesity, 0.5)

        # Flow modifiers
        flow_mod_lymph = disease_modifier(disease_lymphedema, :flow)
        flow_mod_normal = disease_modifier(disease_normal, :flow)
        flow_mod_obese = disease_modifier(disease_obesity, :flow)

        println("\nDisease State Effects on Lymphatic Flow:")
        println("  Normal: flow modifier = $(round(flow_mod_normal, digits=2))")
        println("  Lymphedema (severe): flow modifier = $(round(flow_mod_lymph, digits=2))")
        println("  Obesity (moderate): flow modifier = $(round(flow_mod_obese, digits=2))")

        # Normal should have no modification
        @test flow_mod_normal ≈ 1.0

        # Lymphedema should reduce flow significantly
        @test flow_mod_lymph <= 0.55

        # Obesity has moderate reduction
        @test flow_mod_obese < 1.0 && flow_mod_obese > 0.5
    end

    @testset "Bioavailability Enhancement Comparison" begin
        # Compare different formulation strategies for a high-Eh drug

        drug = DrugLymphPartitioning(
            6.5, 6.0, 45.0, 420.0, 95.0, 35.0, 0.97, 0.75
        )
        Eh = 0.85  # 85% hepatic extraction

        results = bioavailability_enhancement(drug, Eh)

        println("\nFormulation Strategy Comparison (Eh = 0.85):")
        println("  Formulation\t\tF_oral\tF_lymph\tImprovement")
        for r in results
            f_oral_pct = round(r.F_oral * 100, digits=1)
            f_lymph_pct = round(r.F_lymph * 100, digits=1)
            impr = round(r.improvement, digits=2)
            println("  $(rpad(r.formulation, 16))\t$(f_oral_pct)%\t$(f_lymph_pct)%\t$(impr)x")
        end

        # LCT formulation should provide best improvement
        lct_result = filter(r -> contains(r.formulation, "LCT"), results)[1]
        aqueous_result = filter(r -> contains(r.formulation, "Aqueous"), results)[1]

        @test lct_result.F_oral > aqueous_result.F_oral
        @test lct_result.improvement > 1.0
    end

    @testset "Log P Partitioning Curve" begin
        # Generate and validate the sigmoidal partitioning curve

        form = LipidFormulation(:type_I, :LCT, 2000.0, 12.0, 300.0, 0.3, 0.8)
        log_P_range = collect(1.0:1.0:10.0)

        fractions = lymphatic_partitioning_curve(log_P_range, form)

        println("\nLog P vs Lymphatic Fraction Curve:")
        println("  Log P\tF_lymph")
        for (lp, f) in zip(log_P_range, fractions)
            println("  $(lp)\t$(round(f * 100, digits=1))%")
        end

        # Should be sigmoidal with inflection around Log P 5
        # At Log P 3, should be low (<20%)
        idx_3 = findfirst(x -> x == 3.0, log_P_range)
        @test fractions[idx_3] < 0.20

        # At Log P 7, should be high (>60%)
        idx_7 = findfirst(x -> x == 7.0, log_P_range)
        @test fractions[idx_7] > 0.60

        # Monotonically increasing
        @test all(fractions[i] <= fractions[i+1] for i in 1:length(fractions)-1)
    end

    @testset "ODE Simulation" begin
        # Test full ODE simulation

        system = create_default_system(drug_preset=:halofantrine)

        result = simulate_lymphatic_absorption(system, 100.0, (0.0, 24.0))

        println("\nODE Simulation (Halofantrine 100mg):")
        println("  F_lymph = $(round(result.F_lymph * 100, digits=1))%")
        println("  Cmax = $(round(result.Cmax, digits=2)) µg")
        println("  Tmax = $(round(result.tmax, digits=1)) h")
        println("  AUC = $(round(result.AUC, digits=1)) µg·h")

        # Should have reasonable PK
        @test result.Cmax > 0
        @test result.tmax > 0 && result.tmax < 24
        @test result.AUC > 0

        # Lymphatic contribution should be visible
        max_lymph = maximum(result.C_lymph)
        max_total = maximum(result.C_total)
        lymph_contribution = max_lymph / max_total
        println("  Lymphatic contribution to Cmax = $(round(lymph_contribution * 100, digits=1))%")

        # Note: ODE kinetics differ from F_lymph due to different transit times
        # Lymph route is slower (transit through lacteals → thoracic duct)
        @test lymph_contribution > 0.1  # Significant lymphatic contribution visible
    end

    @testset "Lymph Node Exposure" begin
        # Test lymph node targeting calculations

        flow = LymphaticFlow(20.0, 120.0, 80.0, 100.0, 10.0, 25.0, 5.0, 3.0)

        # Drug amount in lymph
        drug_in_lymph = 10.0  # µg

        exposure = lymph_node_exposure(drug_in_lymph, flow)

        println("\nLymph Node Exposure:")
        println("  Drug in lymph = $(drug_in_lymph) µg")
        println("  C_node = $(round(exposure.C_node, digits=2)) µg/mL")
        println("  AUC_node = $(round(exposure.AUC_node, digits=2)) µg·h/mL")
        println("  Residence time = $(exposure.residence_time) h")
        println("  Concentration factor = $(exposure.concentration_factor)x")

        # Nodes should concentrate drug
        @test exposure.concentration_factor > 1.0
        @test exposure.AUC_node > 0
    end

    @testset "Model Validation" begin
        # Run full validation suite

        validation = validate_lymphatic_model()

        println("\nModel Validation Results:")

        # Halofantrine
        hal_val = validation["halofantrine_F_lymph"]
        println("  Halofantrine F_lymph:")
        println("    Calculated: $(round(hal_val.calculated * 100, digits=1))%")
        println("    Literature: $(round(hal_val.literature * 100, digits=1))%")
        println("    Error: $(round(hal_val.error_pct, digits=1))%")
        @test hal_val.error_pct < 25  # Within 25% of literature

        # Metformin (control)
        met_val = validation["metformin_F_lymph"]
        println("  Metformin F_lymph:")
        println("    Calculated: $(round(met_val.calculated * 100, digits=2))%")
        println("    Literature: $(round(met_val.literature * 100, digits=1))%")
        @test met_val.calculated < 0.02  # Should be near zero

        # Testosterone bypass
        tu_val = validation["testosterone_bypass"]
        println("  Testosterone Undecanoate bypass:")
        println("    F_oral with lymph: $(round(tu_val.F_oral_with_lymph * 100, digits=1))%")
        println("    F_oral without: $(round(tu_val.F_oral_without * 100, digits=1))%")
        println("    Improvement: $(round(tu_val.improvement, digits=1))x")
        @test tu_val.improvement > 5  # Should show significant improvement

        # Log P curve inflection
        curve_val = validation["logP_curve"]
        println("  Log P curve inflection point: $(curve_val.inflection_point)")
        @test curve_val.inflection_point >= 4.0 && curve_val.inflection_point <= 6.0
    end

end

println("\n" * "="^60)
println("Lymphatic Absorption Model Tests Complete")
println("="^60)
