"""
Test suite for Blood Compartment Integration Module

Tests the unified blood compartment system that connects:
- Anemia/Polycythemia -> Viscosity -> Perfusion
- Disease Ontology -> Binding Adjustments
- Acute Phase Response -> Time-dependent Proteins
- ODE Solver Interface
"""

using Test

# Load the integration module directly
include("../src/DarwinPBPK/compartments/blood_compartment_integrated.jl")
using .BloodCompartmentIntegrated

@testset "Blood Compartment Integration" begin

    @testset "Basic State Creation" begin
        # Create normal blood state
        state = create_blood_state()

        @test state.hematocrit == 0.42
        @test state.hemoglobin == 14.0
        @test state.albumin_g_L == 40.0
        @test state.aag_g_L == 0.8
        @test state.gfr == 100.0
        @test state.hepatic_flow == 90.0
        @test state.disease_doid == ""
        @test state.is_acute_phase == false

        println("  Normal state created successfully")
    end

    @testset "Disease State Initialization" begin
        # Sepsis
        sepsis_state = create_blood_state_from_disease("DOID:0080559", severity=:moderate)

        @test sepsis_state.disease_doid == "DOID:0080559"
        @test sepsis_state.disease_severity == :moderate
        @test sepsis_state.is_acute_phase == true
        @test sepsis_state.acute_phase_trigger == :sepsis
        @test sepsis_state.albumin_g_L < 30.0  # Hypoalbuminemia
        @test sepsis_state.aag_g_L > 1.5  # Elevated AAG
        @test sepsis_state.il6_pg_mL > 100.0  # Elevated IL-6

        println("  Sepsis state: albumin=$(sepsis_state.albumin_g_L)g/L, AAG=$(sepsis_state.aag_g_L)g/L, IL-6=$(sepsis_state.il6_pg_mL)pg/mL")

        # CKD
        ckd_state = create_blood_state_from_disease("DOID:784", severity=:severe)

        @test ckd_state.gfr < 30.0  # Severe CKD
        @test ckd_state.hematocrit == 0.28  # Anemia of CKD

        println("  CKD state: GFR=$(ckd_state.gfr)mL/min, Hct=$(ckd_state.hematocrit)")

        # Cirrhosis
        cirrhosis_state = create_blood_state_from_disease("DOID:5082", severity=:moderate)

        @test cirrhosis_state.hepatic_flow < 90.0  # Reduced hepatic flow (some reduction from normal)
        @test cirrhosis_state.albumin_g_L < 30.0  # Hypoalbuminemia

        println("  Cirrhosis state: hepatic_flow=$(cirrhosis_state.hepatic_flow)L/h, albumin=$(cirrhosis_state.albumin_g_L)g/L")
    end

    @testset "Drug Properties" begin
        # Tacrolimus - high RBC binding, basic, AAG binding
        tacrolimus = DrugBloodProperties(
            "Tacrolimus";
            ke_p = 37.0,  # High RBC partitioning
            fu_plasma_reference = 0.01,  # Highly bound
            charge_type = :basic,
            albumin_binding = false,
            aag_binding = true,
            extraction_ratio = 0.3,
            renal_fraction = 0.05
        )

        @test tacrolimus.ke_p == 37.0
        @test tacrolimus.fu_plasma_reference == 0.01
        @test tacrolimus.aag_binding == true

        # Phenytoin - acidic, albumin binding
        phenytoin = DrugBloodProperties(
            "Phenytoin";
            ke_p = 1.0,
            fu_plasma_reference = 0.1,
            charge_type = :acidic,
            albumin_binding = true,
            aag_binding = false,
            extraction_ratio = 0.03,
            renal_fraction = 0.05
        )

        @test phenytoin.charge_type == :acidic
        @test phenytoin.albumin_binding == true

        println("  Drug properties created: Tacrolimus (Ke_p=37), Phenytoin (acidic)")
    end

    @testset "Hematocrit -> Viscosity -> Perfusion Cascade" begin
        # Normal state
        normal = create_blood_state()

        # Polycythemia state
        pv = create_blood_state_from_disease("DOID:8997", severity=:moderate)

        # Polycythemia should have higher viscosity
        @test pv.hematocrit > normal.hematocrit
        @test pv.blood_viscosity > normal.blood_viscosity
        @test pv.viscosity_factor > 1.0

        # Higher viscosity should reduce hepatic flow (with autoregulation)
        @test pv.hepatic_flow < normal.hepatic_flow

        println("  Polycythemia cascade:")
        println("    Hct: $(normal.hematocrit) -> $(pv.hematocrit)")
        println("    Blood viscosity: $(round(normal.blood_viscosity, digits=2)) -> $(round(pv.blood_viscosity, digits=2)) mPa·s")
        println("    Hepatic flow: $(round(normal.hepatic_flow, digits=1)) -> $(round(pv.hepatic_flow, digits=1)) L/h")
    end

    @testset "Integrated PK Parameters" begin
        # Create sepsis state and tacrolimus
        state = create_blood_state_from_disease("DOID:0080559", severity=:moderate)

        tacrolimus = DrugBloodProperties(
            "Tacrolimus";
            ke_p = 37.0,
            fu_plasma_reference = 0.01,
            charge_type = :basic,
            aag_binding = true,
            extraction_ratio = 0.3,
            renal_fraction = 0.05
        )

        base_params = Dict(
            :vd => 100.0,  # L
            :cl_hepatic => 20.0,  # L/h
            :cl_renal => 0.5,  # L/h
            :bioavailability => 0.25
        )

        # Calculate integrated parameters
        pk = calculate_integrated_pk_parameters(state, tacrolimus, base_params)

        @test pk.rb > 10.0  # High Rb for tacrolimus
        @test pk.fu_adjusted <= 0.01  # fu should decrease or stay same due to elevated AAG
        @test pk.hepatic_flow_L_h < 90.0  # Reduced in sepsis
        @test length(pk.clinical_notes) > 0  # Should have clinical notes

        println("  Tacrolimus in sepsis:")
        println("    Rb = $(round(pk.rb, digits=1))")
        println("    fu adjusted = $(round(pk.fu_adjusted, digits=4))")
        println("    Hepatic flow = $(round(pk.hepatic_flow_L_h, digits=1)) L/h")
        println("    Clinical notes:")
        for note in pk.clinical_notes
            println("      - $note")
        end
    end

    @testset "Acute Phase Time Evolution" begin
        # Create sepsis state
        state = create_blood_state_from_disease("DOID:0080559", severity=:moderate)

        initial_il6 = state.il6_pg_mL
        initial_aag = state.aag_g_L
        initial_crp = state.crp_mg_L

        # Simulate 24 hours
        for _ in 1:24
            update_blood_state!(state, 1.0)
        end

        # IL-6 should decay
        @test state.il6_pg_mL < initial_il6

        # AAG might increase (approaching target)
        # CRP should increase (peaks at 48-72h)
        @test state.crp_mg_L > initial_crp

        println("  Acute phase evolution (24h):")
        println("    IL-6: $initial_il6 -> $(round(state.il6_pg_mL, digits=1)) pg/mL")
        println("    AAG: $initial_aag -> $(round(state.aag_g_L, digits=2)) g/L")
        println("    CRP: $initial_crp -> $(round(state.crp_mg_L, digits=1)) mg/L")

        # Continue to 72 hours (peak CRP)
        for _ in 1:48
            update_blood_state!(state, 1.0)
        end

        println("  Acute phase evolution (72h):")
        println("    IL-6: $(round(state.il6_pg_mL, digits=1)) pg/mL")
        println("    AAG: $(round(state.aag_g_L, digits=2)) g/L")
        println("    CRP: $(round(state.crp_mg_L, digits=1)) mg/L")
    end

    @testset "ODE Interface" begin
        state = create_blood_state_from_disease("DOID:784", severity=:moderate)

        tacrolimus = DrugBloodProperties(
            "Tacrolimus";
            ke_p = 37.0,
            fu_plasma_reference = 0.01,
            charge_type = :basic,
            aag_binding = true
        )

        # Get ODE parameters
        ode_params = get_ode_parameters(state, tacrolimus)

        @test haskey(ode_params, :hepatic_flow)
        @test haskey(ode_params, :fu_blood)
        @test haskey(ode_params, :rb)
        @test haskey(ode_params, :renal_cl_factor)
        @test ode_params[:renal_cl_factor] < 1.0  # CKD reduces renal clearance

        println("  ODE parameters for CKD:")
        println("    Hepatic flow: $(round(ode_params[:hepatic_flow], digits=1)) L/h")
        println("    Renal CL factor: $(round(ode_params[:renal_cl_factor], digits=2))")
        println("    Rb: $(round(ode_params[:rb], digits=1))")
        println("    fu_blood: $(round(ode_params[:fu_blood], digits=5))")

        # Test apply_time_step!
        new_params = apply_time_step!(state, 1.0, tacrolimus)
        @test state.time == 1.0
        @test new_params[:time] == 1.0

        println("  Time step applied, t=$(state.time)h")
    end

    @testset "State Validation" begin
        # Normal state should be valid
        normal = create_blood_state()
        is_valid, issues = validate_blood_state(normal)
        @test is_valid
        @test isempty(issues)

        # Create extreme state
        extreme = create_blood_state()
        extreme.hematocrit = 0.10  # Critically low
        extreme.albumin_g_L = 10.0  # Critically low
        extreme.gfr = 3.0  # Dialysis needed

        is_valid, issues = validate_blood_state(extreme)
        @test !is_valid
        @test length(issues) >= 3

        println("  Validation issues for extreme state:")
        for issue in issues
            println("    - $issue")
        end
    end

    @testset "Integration Summary" begin
        state = create_blood_state_from_disease("DOID:0080559", severity=:severe)
        update_blood_state!(state, 12.0)  # 12 hours into sepsis

        summary = get_integration_summary(state)

        @test summary[:disease] == "DOID:0080559"
        @test summary[:severity] == :severe
        @test haskey(summary, :hematology)
        @test haskey(summary, :proteins)
        @test haskey(summary, :perfusion)
        @test summary[:acute_phase] !== nothing

        println("  Integration summary (12h sepsis):")
        println("    Disease: $(summary[:disease]) ($(summary[:severity]))")
        println("    Hematology: Hct=$(summary[:hematology][:hematocrit])")
        println("    Proteins: Albumin=$(summary[:proteins][:albumin_g_L])g/L, AAG=$(summary[:proteins][:aag_g_L])g/L")
        println("    Perfusion: Hepatic=$(summary[:perfusion][:hepatic_flow_L_h])L/h, GFR=$(summary[:perfusion][:gfr_mL_min])mL/min")
        println("    Acute phase: IL-6=$(summary[:acute_phase][:il6_pg_mL])pg/mL at $(summary[:acute_phase][:hours_since_onset])h")
    end

    @testset "Clinical Scenario: Phenytoin in Hypoalbuminemia" begin
        # Cirrhosis with low albumin
        state = create_blood_state_from_disease("DOID:5082", severity=:moderate)

        phenytoin = DrugBloodProperties(
            "Phenytoin";
            ke_p = 1.0,
            fu_plasma_reference = 0.1,
            charge_type = :acidic,
            albumin_binding = true,
            extraction_ratio = 0.03,
            renal_fraction = 0.05
        )

        base_params = Dict(
            :vd => 0.7,  # L/kg
            :cl_hepatic => 0.04,  # L/h/kg
            :cl_renal => 0.0,
            :bioavailability => 1.0
        )

        pk = calculate_integrated_pk_parameters(state, phenytoin, base_params)

        # In hypoalbuminemia, fu should increase for acidic drugs
        @test pk.fu_adjusted > phenytoin.fu_plasma_reference

        println("  Phenytoin in cirrhosis:")
        println("    Albumin: $(state.albumin_g_L) g/L (normal ~40)")
        println("    fu: $(phenytoin.fu_plasma_reference) -> $(round(pk.fu_adjusted, digits=3))")
        println("    This means therapeutic levels appear lower but free drug is higher")
        for note in pk.clinical_notes
            println("    Note: $note")
        end
    end

    @testset "Disease Ontology Bridge" begin
        # Test DOID -> Binding State mapping
        binding_state = map_ontology_to_binding_state("DOID:0080559")  # Sepsis
        @test binding_state == :sepsis

        binding_state = map_ontology_to_binding_state("DOID:784", severity=:severe)  # CKD severe
        @test binding_state == :ckd_stage4

        binding_state = map_ontology_to_binding_state("DOID:5082", severity=:mild)  # Cirrhosis mild
        @test binding_state == :cirrhosis_child_a

        println("  DOID -> Binding state mapping works")

        # Test binding adjustments by DOID
        adj = get_binding_adjustments_by_doid("DOID:0080559", :acidic)  # Acidic drug in sepsis
        @test adj[:fu_factor] > 1.0  # Hypoalbuminemia increases fu
        @test adj[:albumin_g_L] < 30.0
        @test adj[:aag_g_L] > 2.0  # Elevated AAG

        println("  Sepsis acidic drug: fu_factor=$(round(adj[:fu_factor], digits=2)), albumin=$(adj[:albumin_g_L])g/L")

        adj = get_binding_adjustments_by_doid("DOID:0080559", :basic)  # Basic drug in sepsis
        @test adj[:fu_factor] < 1.0  # Elevated AAG decreases fu

        println("  Sepsis basic drug: fu_factor=$(round(adj[:fu_factor], digits=2)) (AAG effect)")

        # Test fu calculation from disease code
        fu_phenytoin_normal = 0.1
        fu_phenytoin_cirrhosis = calculate_fu_from_disease_code(fu_phenytoin_normal, :acidic, "DOID:5082")
        @test fu_phenytoin_cirrhosis > fu_phenytoin_normal  # Increases in cirrhosis

        println("  Phenytoin fu: normal=$(fu_phenytoin_normal) -> cirrhosis=$(round(fu_phenytoin_cirrhosis, digits=3))")

        fu_lidocaine_normal = 0.3
        fu_lidocaine_sepsis = calculate_fu_from_disease_code(fu_lidocaine_normal, :basic, "DOID:0080559")
        @test fu_lidocaine_sepsis < fu_lidocaine_normal  # Decreases due to AAG elevation

        println("  Lidocaine fu: normal=$(fu_lidocaine_normal) -> sepsis=$(round(fu_lidocaine_sepsis, digits=3))")
    end

    @testset "ICD-10 Code Bridge" begin
        # Test ICD-10 lookup
        adj = get_binding_adjustments_by_icd10("N18.4", :acidic)  # CKD stage 4
        @test adj[:binding_state] == :ckd_stage4
        @test adj[:renal_cl_factor] < 0.5  # Severely reduced GFR

        println("  ICD-10 N18.4 (CKD4): renal_cl_factor=$(round(adj[:renal_cl_factor], digits=2))")

        # Test state creation from ICD-10
        state = create_state_from_icd10("A41")  # Sepsis
        @test state.disease_doid == "DOID:0080559"
        @test state.is_acute_phase == true

        println("  ICD-10 A41 -> DOID=$(state.disease_doid), acute_phase=$(state.is_acute_phase)")

        # Test unknown ICD-10 returns normal
        adj_unknown = get_binding_adjustments_by_icd10("Z99.9", :acidic)  # Unknown code
        @test adj_unknown[:fu_factor] == 1.0
        @test adj_unknown[:binding_state] == :normal

        println("  Unknown ICD-10 returns normal state")
    end

    @testset "Clinical Workflow: EHR Integration Simulation" begin
        # Simulate an EHR integration workflow
        # Patient comes in with ICD-10 codes from their chart

        patient_icd10_codes = ["E11", "N18.3"]  # T2DM + CKD Stage 3

        # Get adjustments for each condition
        adjustments = []
        for icd10 in patient_icd10_codes
            adj = get_binding_adjustments_by_icd10(icd10, :acidic)
            push!(adjustments, adj)
        end

        # Combine (use worst case)
        combined_fu_factor = maximum(a[:fu_factor] for a in adjustments)
        combined_renal_factor = minimum(a[:renal_cl_factor] for a in adjustments)

        @test combined_fu_factor >= 1.0  # Some increase expected
        @test combined_renal_factor < 1.0  # Reduced renal function

        println("  Simulated EHR workflow for E11 + N18.3:")
        println("    Combined fu_factor: $(round(combined_fu_factor, digits=2))")
        println("    Combined renal_cl_factor: $(round(combined_renal_factor, digits=2))")

        # Create state for worst condition
        state = create_state_from_icd10("N18.3")
        @test state.gfr < 50.0  # CKD Stage 3 GFR

        println("    Primary state GFR: $(round(state.gfr, digits=1)) mL/min")
    end

end

println("\n=== Blood Compartment Integration Tests Complete ===")
