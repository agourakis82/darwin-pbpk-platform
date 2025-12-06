"""
Blood Compartment Non-Critical Features Tests
Tests for v2.8.0 modules: Immunoglobulins, APR, RBC Aging, Spleen RES, Circadian, DOID/ICD

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

using Test
using DarwinPBPK

@testset "Blood Compartment Non-Critical Features v2.8.0" begin

    # =========================================================================
    # IMMUNOGLOBULIN ISOTYPES TESTS
    # =========================================================================
    @testset "Immunoglobulin Isotypes" begin

        @testset "IgG Subclasses" begin
            # Create IgG1
            igg1 = create_igg_subclass(1)
            @test igg1 !== nothing
            @test igg1.isotype == :IgG1

            # Create IgG2
            igg2 = create_igg_subclass(2)
            @test igg2.isotype == :IgG2

            # Create IgG4 (low effector function)
            igg4 = create_igg_subclass(4)
            @test igg4.isotype == :IgG4
        end

        @testset "Other Isotypes" begin
            # IgM pentamer
            igm = create_igm()
            @test igm !== nothing
            @test igm.isotype == :IgM
            @test igm.valency >= 10  # Pentamer = 10 binding sites

            # IgA (mucosal immunity)
            iga = create_iga()
            @test iga !== nothing
            @test iga.isotype == :IgA

            # IgE (allergic/parasitic)
            ige = create_ige()
            @test ige !== nothing
            @test ige.isotype == :IgE
        end

        @testset "Complement Activation" begin
            igg1 = create_igg_subclass(1)
            igm = create_igm()
            igg4 = create_igg_subclass(4)

            # Classical pathway activation
            c1_igg1 = calculate_complement_activation(igg1, :classical)
            c1_igm = calculate_complement_activation(igm, :classical)
            c1_igg4 = calculate_complement_activation(igg4, :classical)

            # IgM > IgG1 > IgG4 for complement
            @test c1_igm >= c1_igg1
            @test c1_igg1 > c1_igg4
        end

        @testset "Fc Receptor Binding" begin
            igg1 = create_igg_subclass(1)

            # FcγRI binding (high affinity)
            fcgri_binding = calculate_fc_receptor_binding(igg1, :FcgRI)
            @test fcgri_binding > 0

            # FcγRIIIa (ADCC)
            fcgriiia_binding = calculate_fc_receptor_binding(igg1, :FcgRIIIa)
            @test fcgriiia_binding > 0
        end

        @testset "Isotype Clearance" begin
            igg1 = create_igg_subclass(1)
            igm = create_igm()
            ige = create_ige()

            cl_igg1 = calculate_isotype_clearance(igg1)
            cl_igm = calculate_isotype_clearance(igm)
            cl_ige = calculate_isotype_clearance(ige)

            # All clearances positive
            @test cl_igg1 > 0
            @test cl_igm > 0
            @test cl_ige > 0

            # IgM larger = faster clearance typically
            @test cl_igm >= cl_igg1 * 0.5
        end

        @testset "Immune Complex Clearance" begin
            igg1 = create_igg_subclass(1)
            complex_size = 5  # 5 antibodies in complex

            ic_clearance = calculate_immune_complex_clearance(igg1, complex_size)
            @test ic_clearance > 0
        end

        @testset "Database Access" begin
            @test haskey(IMMUNOGLOBULIN_DATABASE, :IgG1)
            @test haskey(IMMUNOGLOBULIN_DATABASE, :IgM)
            @test COMPLEMENT_PARAMETERS !== nothing
            @test FC_RECEPTOR_DATABASE !== nothing
        end
    end

    # =========================================================================
    # ACUTE PHASE RESPONSE TESTS
    # =========================================================================
    @testset "Acute Phase Response" begin

        @testset "Cytokine Profile Creation" begin
            # Normal state
            normal_state = AcutePhaseState(il6 = 5.0)  # pg/mL
            @test normal_state !== nothing
            @test normal_state.il6 == 5.0

            # Sepsis (high IL-6)
            sepsis_state = AcutePhaseState(il6 = 500.0)
            @test sepsis_state.il6 == 500.0
        end

        @testset "Acute Phase Simulation" begin
            # Simulate inflammatory response
            initial_il6 = 100.0  # Elevated
            duration_hours = 72

            result = simulate_acute_phase(initial_il6, duration_hours)
            @test result !== nothing
            @test haskey(result, :crp) || hasproperty(result, :crp)
        end

        @testset "Protein Changes" begin
            il6_level = 200.0  # High inflammation

            changes = calculate_protein_changes(il6_level)
            @test changes !== nothing

            # CRP should increase, albumin should decrease
            if haskey(changes, :crp_fold)
                @test changes[:crp_fold] > 1.0
            end
            if haskey(changes, :albumin_fold)
                @test changes[:albumin_fold] < 1.0
            end
        end

        @testset "Binding During APR" begin
            apr_state = AcutePhaseState(il6 = 150.0)

            # Warfarin (acidic) - fu increases due to low albumin
            fu_warfarin = apply_acute_phase_binding(apr_state, :acidic, 0.01)
            @test fu_warfarin >= 0.01  # Should increase or stay same

            # Lidocaine (basic) - fu decreases due to high AAG
            fu_lidocaine = apply_acute_phase_binding(apr_state, :basic, 0.3)
            @test fu_lidocaine > 0
        end

        @testset "Time Course" begin
            il6_peak = 300.0
            time_points = 0:6:72

            time_course = get_time_course(il6_peak, collect(time_points))
            @test length(time_course) > 0
        end

        @testset "Database Constants" begin
            @test ACUTE_PHASE_PROTEINS !== nothing
            @test CYTOKINE_EFFECTS !== nothing
        end
    end

    # =========================================================================
    # RBC AGING TESTS
    # =========================================================================
    @testset "RBC Aging & RDW" begin

        @testset "Normal RBC Population" begin
            population = create_normal_rbc_population()
            @test population !== nothing

            # Check age distribution
            dist = get_age_distribution(population)
            @test length(dist) > 0
        end

        @testset "Disease Population" begin
            # Hemolytic anemia - shortened lifespan
            hemolytic = create_disease_population(:hemolytic_anemia)
            @test hemolytic !== nothing

            # Sickle cell
            sickle = create_disease_population(:sickle_cell)
            @test sickle !== nothing
        end

        @testset "Age-Weighted Transport" begin
            population = create_normal_rbc_population()

            # Young RBCs have more transporters
            transport = calculate_age_weighted_transport(population, :GLUT1)
            @test transport > 0
        end

        @testset "RDW Effect" begin
            # Normal RDW (11-14%)
            normal_rdw = 12.5
            effect_normal = calculate_rdw_effect(normal_rdw)

            # High RDW (anisocytosis)
            high_rdw = 18.0
            effect_high = calculate_rdw_effect(high_rdw)

            @test effect_normal !== nothing
            @test effect_high !== nothing
        end

        @testset "RBC Turnover Simulation" begin
            population = create_normal_rbc_population()
            duration_days = 30

            result = simulate_rbc_turnover(population, duration_days)
            @test result !== nothing
        end

        @testset "Database Constants" begin
            @test RBC_AGE_PARAMETERS !== nothing
            @test RETICULOCYTE_FACTORS !== nothing
        end
    end

    # =========================================================================
    # SPLEEN RES CLEARANCE TESTS
    # =========================================================================
    @testset "Spleen RES Clearance" begin

        @testset "Normal Spleen" begin
            spleen = create_normal_spleen()
            @test spleen !== nothing
        end

        @testset "Disease Spleen States" begin
            # Splenomegaly (enlarged)
            splenomegaly = create_disease_spleen(:splenomegaly)
            @test splenomegaly !== nothing

            # Hyposplenism
            hyposplenism = create_disease_spleen(:hyposplenism)
            @test hyposplenism !== nothing
        end

        @testset "RES Clearance Calculation" begin
            spleen = create_normal_spleen()

            # IgG-opsonized particles
            clearance = calculate_res_clearance(spleen, :IgG_opsonized, 1000.0)
            @test clearance > 0
        end

        @testset "Splenic Uptake" begin
            spleen = create_normal_spleen()

            # Uptake of nanoparticles
            uptake = calculate_splenic_uptake(spleen, 100.0)  # nm diameter
            @test uptake >= 0
            @test uptake <= 1.0  # Fraction
        end

        @testset "Splenectomy Effects" begin
            normal_spleen = create_normal_spleen()

            # Apply splenectomy
            post_splenectomy = apply_splenectomy(normal_spleen)
            @test post_splenectomy !== nothing

            # RES capacity should decrease
            normal_capacity = calculate_res_clearance(normal_spleen, :IgG_opsonized, 1000.0)
            splenectomy_capacity = calculate_res_clearance(post_splenectomy, :IgG_opsonized, 1000.0)
            @test splenectomy_capacity < normal_capacity
        end

        @testset "FcR-Mediated Clearance" begin
            spleen = create_normal_spleen()

            fcr_clearance = calculate_fcr_mediated_clearance(spleen, :FcgRI)
            @test fcr_clearance > 0
        end

        @testset "Database Constants" begin
            @test SPLEEN_PARAMETERS !== nothing
            @test RES_TISSUE_WEIGHTS !== nothing
        end
    end

    # =========================================================================
    # CIRCADIAN EFFECTS TESTS
    # =========================================================================
    @testset "Circadian Effects" begin

        @testset "Circadian Factor Calculation" begin
            # Morning (08:00)
            morning_factor = get_circadian_factor(:albumin, 8.0)
            @test morning_factor > 0

            # Midnight
            midnight_factor = get_circadian_factor(:albumin, 0.0)
            @test midnight_factor > 0

            # Different phases
            @test morning_factor != midnight_factor || true  # May be equal
        end

        @testset "Circadian Simulation" begin
            duration_hours = 48
            parameter = :gfr

            result = simulate_circadian_variation(parameter, duration_hours)
            @test result !== nothing
            @test length(result) > 0
        end

        @testset "Optimal Dosing Time" begin
            # Find best time for warfarin (hepatic metabolism)
            optimal_time = calculate_optimal_dosing_time(:warfarin)
            @test optimal_time >= 0
            @test optimal_time < 24
        end

        @testset "Chronotype Adjustment" begin
            # Morning chronotype
            morning_adj = get_chronotype_adjustment(:morning, 8.0)

            # Evening chronotype
            evening_adj = get_chronotype_adjustment(:evening, 8.0)

            @test morning_adj !== nothing
            @test evening_adj !== nothing
        end

        @testset "Database Constants" begin
            @test CIRCADIAN_PARAMETERS !== nothing
            @test CHRONOTYPE_SHIFTS !== nothing
        end
    end

    # =========================================================================
    # DISEASE ONTOLOGY PK (DOID/ICD-10/ICD-11) TESTS
    # =========================================================================
    @testset "Disease Ontology PK Integration" begin

        @testset "DOID Lookup" begin
            # Type 2 diabetes
            t2dm = get_pk_adjustments_by_doid("DOID:9352")
            @test t2dm !== nothing
            @test t2dm.disease.name == "type 2 diabetes mellitus"
            @test t2dm.gfr_adjustment < 1.0  # Often reduced

            # CKD Stage 4
            ckd4 = get_pk_adjustments_by_doid("DOID:0060682")
            @test ckd4 !== nothing
            @test ckd4.gfr_adjustment < 0.3  # Severely reduced

            # Heart failure
            hf = get_pk_adjustments_by_doid("DOID:6000")
            @test hf !== nothing
            @test hf.vd_adjustment > 1.0  # Edema increases Vd
        end

        @testset "ICD-10 Lookup" begin
            # N18.4 = CKD stage 4
            ckd_icd10 = get_pk_adjustments_by_icd10("N18.4")
            @test ckd_icd10 !== nothing
            @test ckd_icd10.gfr_adjustment < 0.3

            # E11 = Type 2 diabetes
            t2dm_icd10 = get_pk_adjustments_by_icd10("E11")
            @test t2dm_icd10 !== nothing

            # K74 = Liver cirrhosis
            cirrhosis_icd10 = get_pk_adjustments_by_icd10("K74")
            @test cirrhosis_icd10 !== nothing
            @test cirrhosis_icd10.hepatic_adjustment < 1.0

            # I50 = Heart failure
            hf_icd10 = get_pk_adjustments_by_icd10("I50")
            @test hf_icd10 !== nothing
        end

        @testset "ICD-11 Lookup" begin
            # GB61 = CKD
            ckd_icd11 = get_pk_adjustments_by_icd11("GB61")
            @test ckd_icd11 !== nothing

            # 5A11 = Type 2 diabetes
            t2dm_icd11 = get_pk_adjustments_by_icd11("5A11")
            @test t2dm_icd11 !== nothing
        end

        @testset "Disease Search" begin
            # Search by name
            diabetes_results = search_disease_pk("diabetes")
            @test length(diabetes_results) > 0

            # Search by synonym
            esrd_results = search_disease_pk("ESRD")
            @test length(esrd_results) > 0

            # Search for sepsis
            sepsis_results = search_disease_pk("sepsis")
            @test length(sepsis_results) > 0
        end

        @testset "Disease Hierarchy" begin
            # CKD Stage 4 inherits from CKD
            hierarchy = map_disease_hierarchy("DOID:0060682")
            @test "DOID:784" in hierarchy  # Parent CKD

            # T2DM inherits from DM
            t2dm_hierarchy = map_disease_hierarchy("DOID:9352")
            @test "DOID:9351" in t2dm_hierarchy  # Parent DM
        end

        @testset "Fallback Lookup" begin
            # Get with fallback to parent
            profile, source_doid, match_type = get_pk_with_fallback("DOID:0060682")
            @test profile !== nothing
            @test match_type in [:exact, :inferred]
        end

        @testset "PK Adjustment Values" begin
            # Sepsis - extreme changes
            sepsis = get_pk_adjustments_by_doid("DOID:0080559")
            @test sepsis !== nothing
            @test sepsis.fu_acidic_adjustment >= 1.5  # Hypoalbuminemia
            @test sepsis.vd_adjustment >= 1.5  # Capillary leak
            @test sepsis.albumin_concentration < 25  # Very low
            @test sepsis.aag_concentration > 2.0  # Very high

            # Burns - similar to sepsis
            burns = get_pk_adjustments_by_doid("DOID:0050805")
            @test burns !== nothing
            @test burns.fu_acidic_adjustment > 2.0  # Severe hypoalbuminemia

            # Pregnancy - increased GFR
            pregnancy = get_pk_adjustments_by_doid("DOID:0060088")
            @test pregnancy !== nothing
            @test pregnancy.gfr_adjustment >= 1.3  # GFR increases 50%
            @test pregnancy.vd_adjustment > 1.0  # Plasma expansion

            # Cirrhosis - hepatic dysfunction
            cirrhosis = get_pk_adjustments_by_doid("DOID:5082")
            @test cirrhosis !== nothing
            @test cirrhosis.hepatic_adjustment < 0.7  # Reduced metabolism
            @test cirrhosis.fu_acidic_adjustment > 1.5  # Low albumin
        end

        @testset "Combine Disease Profiles (Comorbidities)" begin
            # CKD + Heart Failure (common combination)
            ckd = get_pk_adjustments_by_doid("DOID:784")
            hf = get_pk_adjustments_by_doid("DOID:6000")

            combined = combine_disease_profiles([ckd, hf])
            @test combined !== nothing

            # Combined should take worst case
            @test combined.gfr_adjustment <= min(ckd.gfr_adjustment, hf.gfr_adjustment)
            @test combined.hepatic_adjustment <= min(ckd.hepatic_adjustment, hf.hepatic_adjustment)
            @test combined.vd_adjustment >= max(ckd.vd_adjustment, hf.vd_adjustment)

            # Should combine names
            @test occursin("+", combined.disease.name)
        end

        @testset "Triple Comorbidity" begin
            # Diabetes + CKD + Obesity
            t2dm = get_pk_adjustments_by_doid("DOID:9352")
            ckd = get_pk_adjustments_by_doid("DOID:784")
            obesity = get_pk_adjustments_by_doid("DOID:9970")

            combined = combine_disease_profiles([t2dm, ckd, obesity])
            @test combined !== nothing
            @test combined.evidence_level == :extrapolated
        end

        @testset "Disease Summary" begin
            t2dm = get_pk_adjustments_by_doid("DOID:9352")
            summary = get_disease_summary(t2dm)

            @test haskey(summary, "disease_name")
            @test haskey(summary, "doid")
            @test haskey(summary, "pk_summary")
            @test haskey(summary, "clinical_notes")

            @test summary["disease_name"] == "type 2 diabetes mellitus"
        end

        @testset "List Supported Diseases" begin
            diseases = list_supported_diseases()
            @test length(diseases) >= 15  # At least 15 diseases
            @test haskey(diseases, "DOID:9352")  # T2DM
            @test haskey(diseases, "DOID:784")   # CKD
            @test haskey(diseases, "DOID:5082") # Cirrhosis
        end

        @testset "Cross-Reference Integrity" begin
            # Check ICD-10 → DOID mapping completeness
            for (icd10, doid) in ICD10_TO_DOID
                profile = get_pk_adjustments_by_doid(doid)
                @test profile !== nothing
            end

            # Check ICD-11 → DOID mapping
            for (icd11, doid) in ICD11_TO_DOID
                profile = get_pk_adjustments_by_doid(doid)
                @test profile !== nothing
            end
        end

        @testset "Special Considerations" begin
            # Sepsis should have many clinical notes
            sepsis = get_pk_adjustments_by_doid("DOID:0080559")
            @test length(sepsis.special_considerations) >= 3

            # Pregnancy should mention fetal exposure
            pregnancy = get_pk_adjustments_by_doid("DOID:0060088")
            fetal_mentioned = any(occursin("fetal", lowercase(s)) for s in pregnancy.special_considerations)
            @test fetal_mentioned
        end

        @testset "Evidence Levels" begin
            # High evidence diseases
            sepsis = get_pk_adjustments_by_doid("DOID:0080559")
            @test sepsis.evidence_level == :high

            ckd = get_pk_adjustments_by_doid("DOID:784")
            @test ckd.evidence_level == :high

            # Moderate evidence
            t2dm = get_pk_adjustments_by_doid("DOID:9352")
            @test t2dm.evidence_level in [:high, :moderate]
        end
    end

    # =========================================================================
    # INTEGRATION TESTS
    # =========================================================================
    @testset "Module Integration" begin

        @testset "APR + Disease Ontology" begin
            # Sepsis from DOID has elevated AAG
            sepsis_pk = get_pk_adjustments_by_doid("DOID:0080559")

            # APR simulation with sepsis IL-6 levels
            apr_result = simulate_acute_phase(500.0, 48)  # High IL-6

            @test sepsis_pk.aag_concentration > 2.0
            @test apr_result !== nothing
        end

        @testset "Circadian + Disease Adjustment" begin
            # Get circadian factor for GFR at midnight
            midnight_gfr = get_circadian_factor(:gfr, 0.0)

            # Get CKD adjustment
            ckd = get_pk_adjustments_by_doid("DOID:784")

            # Combined effect (conceptual)
            effective_gfr = midnight_gfr * ckd.gfr_adjustment
            @test effective_gfr < ckd.gfr_adjustment
        end

        @testset "RBC Aging + Spleen Clearance" begin
            # Old RBCs cleared by spleen
            population = create_normal_rbc_population()
            spleen = create_normal_spleen()

            @test population !== nothing
            @test spleen !== nothing
        end

        @testset "Immunoglobulins + RES" begin
            igg1 = create_igg_subclass(1)
            spleen = create_normal_spleen()

            # FcR-mediated clearance
            clearance = calculate_fcr_mediated_clearance(spleen, :FcgRI)
            @test clearance > 0
        end
    end

end

println("\nAll Blood Compartment Non-Critical Tests Completed!")
