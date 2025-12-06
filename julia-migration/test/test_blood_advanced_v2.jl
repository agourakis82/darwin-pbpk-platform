"""
Test suite for Blood Compartment v2.7.0 modules:
- Lipoprotein Binding (HDL/LDL/VLDL)
- RBC Transporters (Band3, OAT, OCT, GLUT1)
- Disease State Binding Adjustments
- mAb PBPK (FcRn, TMDD)

Author: Dr. Demetrios Agourakis
Date: December 2025
"""

using Test

# Include modules directly for standalone testing
include("../src/DarwinPBPK/compartments/lipoprotein_binding.jl")
include("../src/DarwinPBPK/compartments/rbc_transporters.jl")
include("../src/DarwinPBPK/compartments/disease_state_binding.jl")
include("../src/DarwinPBPK/compartments/mab_pbpk.jl")

using .LipoproteinBinding
using .RBCTransporters
using .DiseaseStateBinding
using .mAbPBPK

println("=" ^ 70)
println("Blood Compartment v2.7.0 - Advanced Modules Test Suite")
println("=" ^ 70)

@testset "Blood Advanced v2.7.0" begin

    # =========================================================================
    # LIPOPROTEIN BINDING TESTS
    # =========================================================================
    @testset "Lipoprotein Binding" begin

        @testset "Lipoprotein Profile Creation" begin
            # Normal profile
            profile = create_normal_lipoprotein_profile()
            @test profile.hdl_c > 0
            @test profile.ldl_c > 0
            @test profile.vldl_c > 0
            @test profile.total_cholesterol > 0
            @test profile.triglycerides > 0
            @test profile.condition == :normal

            # Dyslipidemia profiles
            hyper = create_dyslipidemia_profile(:hypercholesterolemia)
            @test hyper.ldl_c > profile.ldl_c
            @test hyper.condition == :hypercholesterolemia

            diabetic = create_dyslipidemia_profile(:diabetic_dyslipidemia)
            @test diabetic.triglycerides > profile.triglycerides
            @test diabetic.hdl_c < profile.hdl_c

            fh = create_dyslipidemia_profile(:familial_hypercholesterolemia)
            @test fh.ldl_c > 250  # Very high LDL
        end

        @testset "Drug Database" begin
            # Check database exists and has entries
            @test length(LIPOPROTEIN_DRUG_DATABASE) > 0

            # Known drugs should be in database
            @test haskey(LIPOPROTEIN_DRUG_DATABASE, "atorvastatin")
            @test haskey(LIPOPROTEIN_DRUG_DATABASE, "cyclosporine")
            @test haskey(LIPOPROTEIN_DRUG_DATABASE, "amiodarone")

            # Check data structure (DrugLipoproteinBinding struct)
            atorva = LIPOPROTEIN_DRUG_DATABASE["atorvastatin"]
            @test atorva isa DrugLipoproteinBinding
            @test atorva.name == "atorvastatin"
            @test atorva.kp_hdl > 0
            @test atorva.kp_ldl > 0
            @test atorva.kp_vldl > 0
            @test atorva.logP > 0
        end

        @testset "Lipoprotein Binding Calculation" begin
            profile = create_normal_lipoprotein_profile()
            drug = LIPOPROTEIN_DRUG_DATABASE["atorvastatin"]

            # Calculate binding
            result = calculate_lipoprotein_binding(drug, profile)
            @test result isa Dict
            @test haskey(result, "f_hdl")
            @test haskey(result, "f_ldl")
            @test haskey(result, "f_vldl")
            @test haskey(result, "f_free")
            @test haskey(result, "f_total_lipoprotein")

            # Fractions should be valid
            @test result["f_hdl"] >= 0
            @test result["f_ldl"] >= 0
            @test result["f_vldl"] >= 0
            @test result["f_free"] > 0
            @test result["f_free"] <= 1.0

            # Sum of fractions should be ~1
            total = result["f_hdl"] + result["f_ldl"] + result["f_vldl"] + result["f_free"]
            @test isapprox(total, 1.0, atol=0.01)

            # Cyclosporine - very lipophilic
            cyclo_drug = LIPOPROTEIN_DRUG_DATABASE["cyclosporine"]
            cyclo = calculate_lipoprotein_binding(cyclo_drug, profile)
            @test cyclo["f_total_lipoprotein"] > 0  # Has some LP binding
        end

        @testset "fu with Lipoproteins" begin
            profile = create_normal_lipoprotein_profile()
            drug = LIPOPROTEIN_DRUG_DATABASE["atorvastatin"]

            # Compare fu with lipoprotein consideration
            fu_base = 0.02  # Standard fu for atorvastatin
            fu_adjusted = calculate_fu_with_lipoproteins(fu_base, drug, profile)

            @test fu_adjusted > 0
            @test fu_adjusted <= fu_base  # Should be lower due to LP binding
            @test fu_adjusted > 1e-6  # Minimum physiological
        end

        @testset "Get Lipoprotein Partition" begin
            # Known drug
            data = get_lipoprotein_partition("atorvastatin")
            @test data !== nothing
            @test data.name == "atorvastatin"

            # Unknown drug
            unknown = get_lipoprotein_partition("unknown_drug_xyz")
            @test unknown === nothing
        end

        @testset "Total Plasma Binding" begin
            profile = create_normal_lipoprotein_profile()

            # Use the exported function - calculate binding for multiple drugs
            cyclo = LIPOPROTEIN_DRUG_DATABASE["cyclosporine"]
            binding = calculate_lipoprotein_binding(cyclo, profile)

            @test binding isa Dict
            @test haskey(binding, "f_total_lipoprotein")
            @test binding["f_total_lipoprotein"] >= 0
        end
    end

    # =========================================================================
    # RBC TRANSPORTER TESTS
    # =========================================================================
    @testset "RBC Transporters" begin

        @testset "Transporter Profile Creation" begin
            profile = create_normal_rbc_transporters()
            @test profile isa RBCTransporterProfile
            @test profile.ae1 == 1.0
            @test profile.glut1 == 1.0
            @test profile.ent1 == 1.0
            @test profile.mct1 == 1.0
            @test profile.condition == :normal
        end

        @testset "Drug Transport Database" begin
            @test length(RBC_TRANSPORTER_SUBSTRATES) > 0

            # Known substrates
            @test haskey(RBC_TRANSPORTER_SUBSTRATES, "chloroquine")
            @test haskey(RBC_TRANSPORTER_SUBSTRATES, "hydroxychloroquine")

            # Check structure
            cq = RBC_TRANSPORTER_SUBSTRATES["chloroquine"]
            @test cq isa DrugRBCTransport
            @test cq.primary_transporter == :ae1
            @test cq.km > 0
            @test cq.is_substrate == true
        end

        @testset "RBC Transport Calculation" begin
            profile = create_normal_rbc_transporters()
            drug = RBC_TRANSPORTER_SUBSTRATES["chloroquine"]
            plasma_conc = 1000.0  # ng/mL
            rbc_conc = 100.0      # Initial RBC conc

            # API: calculate_rbc_transport(drug, plasma_conc, rbc_conc, transporters)
            result = calculate_rbc_transport(drug, plasma_conc, rbc_conc, profile)
            @test result isa Dict
            @test haskey(result, "net_flux")
            @test haskey(result, "active_influx")
            @test haskey(result, "passive_flux")
            @test haskey(result, "transporter_saturation")

            # Net flux should be positive (influx)
            @test result["net_flux"] >= 0 || result["active_influx"] >= 0
        end

        @testset "RBC Accumulation" begin
            profile = create_normal_rbc_transporters()
            drug = RBC_TRANSPORTER_SUBSTRATES["chloroquine"]

            # API: calculate_rbc_accumulation(drug, plasma_conc, transporters; time_hours)
            accum = calculate_rbc_accumulation(drug, 500.0, profile; time_hours=24.0)
            @test accum isa Dict
            @test haskey(accum, "rbc_plasma_ratio")
            @test accum["rbc_plasma_ratio"] > 0
        end

        @testset "Disease State Transporters" begin
            # Create custom transporter profile for disease states
            sickle = RBCTransporterProfile(
                ae1=0.7,      # Reduced in sickle cell
                glut1=1.2,    # May be upregulated
                condition=:sickle_cell
            )
            @test sickle isa RBCTransporterProfile
            @test sickle.condition == :sickle_cell
            @test sickle.ae1 < 1.0

            normal = create_normal_rbc_transporters()
            @test normal.condition == :normal
        end

        @testset "Get Transport Data" begin
            data = get_rbc_transport_data("chloroquine")
            @test data !== nothing
            @test data.name == "chloroquine"

            unknown = get_rbc_transport_data("unknown_drug_xyz")
            @test unknown === nothing
        end

        @testset "Transporter Inhibition" begin
            profile = create_normal_rbc_transporters()
            inhibitor = RBC_TRANSPORTER_SUBSTRATES["chloroquine"]  # Is inhibitor

            # Apply inhibition
            inhibited = apply_transporter_inhibition(profile, inhibitor, 1000.0)
            @test inhibited isa RBCTransporterProfile
            # AE1 should be reduced (chloroquine inhibits)
            @test inhibited.ae1 <= profile.ae1
        end
    end

    # =========================================================================
    # DISEASE STATE BINDING TESTS
    # =========================================================================
    @testset "Disease State Binding" begin

        @testset "Plasma Protein State" begin
            # Normal state
            normal = PlasmaProteinState()
            @test normal.albumin == 40.0  # g/L
            @test normal.aag > 0
            @test normal.albumin_function == 1.0

            # Custom state
            uremic = PlasmaProteinState(
                albumin = 30.0,
                aag = 1.5,
                urea = 25.0,
                albumin_function = 0.7
            )
            @test uremic.albumin < normal.albumin
            @test uremic.urea > normal.urea
        end

        @testset "Disease State Creation" begin
            # CKD stages
            ckd1 = create_disease_state(:ckd_stage1)
            @test ckd1 isa DiseaseState
            @test ckd1.gfr >= 90.0

            ckd5 = create_disease_state(:ckd_stage5)
            @test ckd5.gfr < 15.0

            # Cirrhosis
            cirr_a = create_disease_state(:cirrhosis_child_a)
            @test cirr_a.hepatic_function < 1.0

            cirr_c = create_disease_state(:cirrhosis_child_c)
            @test cirr_c.hepatic_function < cirr_a.hepatic_function

            # Pregnancy
            preg_t3 = create_disease_state(:pregnancy_t3)
            @test preg_t3.volume_status == :hypervolemic

            # Sepsis
            sepsis = create_disease_state(:sepsis)
            @test sepsis.inflammatory_state == :severe || sepsis.inflammatory_state == :moderate
        end

        @testset "Binding Adjustments Calculation" begin
            disease = create_disease_state(:ckd_stage4)

            # calculate_binding_adjustments is internal, test via calculate_adjusted_fu
            fu_normal = 0.1
            fu_adj = calculate_adjusted_fu(fu_normal, :acidic, disease)
            @test fu_adj >= fu_normal  # Increased fu in uremia
            @test fu_adj <= 1.0
        end

        @testset "Adjusted fu Calculation" begin
            disease = create_disease_state(:ckd_stage4)

            fu_normal = 0.1  # 10% free
            # API: calculate_adjusted_fu(fu_normal, drug_type, disease)
            fu_acidic = calculate_adjusted_fu(fu_normal, :acidic, disease)

            @test fu_acidic > fu_normal  # More free in uremia
            @test fu_acidic <= 1.0

            # Basic drugs - AAG may be elevated
            fu_basic = calculate_adjusted_fu(fu_normal, :basic, disease)
            @test fu_basic > 0
            @test fu_basic <= 1.0
        end

        @testset "Apply Disease Adjustments" begin
            disease = create_disease_state(:cirrhosis_child_c)

            fu_base = 0.05
            vd_base = 50.0  # L
            cl_base = 10.0  # L/h

            # API: apply_disease_adjustments(fu, vd, cl, drug_type, disease)
            result = apply_disease_adjustments(fu_base, vd_base, cl_base, :acidic, disease)
            @test result isa Dict
            @test haskey(result, "fu")
            @test haskey(result, "vd")
            @test haskey(result, "clearance")

            # Cirrhosis: fu↑, Cl↓
            @test result["fu"] >= fu_base
            @test result["clearance"] <= cl_base
        end

        @testset "Disease Database Coverage" begin
            # Test various disease states can be created
            for disease in [:ckd_stage1, :ckd_stage2, :ckd_stage3, :ckd_stage4, :ckd_stage5,
                           :cirrhosis_child_a, :cirrhosis_child_b, :cirrhosis_child_c,
                           :pregnancy_t1, :pregnancy_t2, :pregnancy_t3,
                           :sepsis, :burn]
                ds = create_disease_state(disease)
                @test ds isa DiseaseState
                @test ds.name == disease
            end
        end
    end

    # =========================================================================
    # mAb PBPK TESTS
    # =========================================================================
    @testset "mAb PBPK" begin

        @testset "mAb Properties Creation" begin
            # Create IgG1
            igg1 = create_igg1("test_mab"; target="CD20")
            @test igg1 isa mAbProperties
            @test igg1.igg_subclass == :igg1
            @test igg1.molecular_weight == 150.0
            @test igg1.target == "CD20"

            # Create IgG4
            igg4 = create_igg4("test_mab_igg4"; target="PD-1")
            @test igg4.igg_subclass == :igg4
            @test igg4.effector_function == false  # IgG4 reduced effector

            # Create Fab
            fab = create_fab("test_fab")
            @test fab.molecular_weight == 50.0
            @test fab.half_life_days < 21.0  # Shorter half-life

            # Create IgG2
            igg2 = create_igg2("test_igg2")
            @test igg2.igg_subclass == :igg2
        end

        @testset "mAb Database" begin
            @test length(MAB_DATABASE) > 0

            # Known mAbs
            @test haskey(MAB_DATABASE, "rituximab")
            @test haskey(MAB_DATABASE, "trastuzumab")
            @test haskey(MAB_DATABASE, "pembrolizumab")

            ritux = MAB_DATABASE["rituximab"]
            @test ritux isa mAbProperties
            @test ritux.target == "CD20"
            @test ritux.igg_subclass == :igg1
        end

        @testset "Target Database" begin
            @test length(TARGET_DATABASE) > 0

            # Known targets
            @test haskey(TARGET_DATABASE, "CD20")
            @test haskey(TARGET_DATABASE, "HER2")

            cd20 = TARGET_DATABASE["CD20"]
            @test cd20 isa TargetProperties
            @test cd20.expression_level > 0
            @test cd20.turnover_rate > 0
        end

        @testset "TMDD Parameters" begin
            tmdd = TMDDParameters()
            @test tmdd.kint > 0
            @test tmdd.kdeg > 0
            @test tmdd.target_baseline > 0

            # Custom TMDD
            custom = TMDDParameters(kint=0.1, kdeg=0.02, target_baseline=5.0)
            @test custom.kint == 0.1
            @test custom.ksyn ≈ custom.kdeg * custom.target_baseline
        end

        @testset "FcRn Recycling" begin
            mab = MAB_DATABASE["rituximab"]
            fcrn = FcRnParameters()
            mab_conc = 100.0  # nM

            # API: calculate_fcrn_recycling(mab, fcrn, mab_conc)
            recycling = calculate_fcrn_recycling(mab, fcrn, mab_conc)
            @test recycling isa Dict
            @test haskey(recycling, "recycling_fraction")
            @test haskey(recycling, "catabolism_fraction")
            @test haskey(recycling, "half_life_effect")

            @test recycling["recycling_fraction"] >= 0
            @test recycling["recycling_fraction"] <= 1.0
            @test recycling["catabolism_fraction"] + recycling["recycling_fraction"] ≈ 1.0
        end

        @testset "TMDD Clearance" begin
            mab = MAB_DATABASE["trastuzumab"]
            target = TARGET_DATABASE["HER2"]
            tmdd = TMDDParameters()

            # API: calculate_tmdd_clearance(mab, target, mab_conc, tmdd)
            # Low concentration - more TMDD effect
            cl_low = calculate_tmdd_clearance(mab, target, 1.0, tmdd)
            @test cl_low isa Dict
            @test haskey(cl_low, "cl_total")
            @test haskey(cl_low, "cl_tmdd")
            @test haskey(cl_low, "target_occupancy")
            @test cl_low["cl_total"] > 0

            # High concentration - saturated
            cl_high = calculate_tmdd_clearance(mab, target, 1000.0, tmdd)
            @test cl_high["cl_total"] > 0
        end

        @testset "Target Occupancy" begin
            mab = MAB_DATABASE["rituximab"]

            # API: calculate_target_occupancy(mab, mab_conc, target_conc)
            # Low dose
            to_low = calculate_target_occupancy(mab, 10.0, 1.0)
            # High dose
            to_high = calculate_target_occupancy(mab, 1000.0, 1.0)

            @test 0 <= to_low <= 1.0
            @test 0 <= to_high <= 1.0
            @test to_high >= to_low  # Higher conc = higher occupancy
        end

        @testset "mAb PK Simulation" begin
            mab = MAB_DATABASE["rituximab"]
            dose = 375.0  # mg
            time_points = collect(0.0:24.0:672.0)  # 4 weeks in hours

            # API: simulate_mab_pk(mab, dose, time_points; ...)
            result = simulate_mab_pk(mab, dose, time_points;
                body_weight = 70.0
            )

            @test result isa Dict
            @test haskey(result, "time") || haskey(result, "times")
            @test haskey(result, "concentration") || haskey(result, "concentrations")
        end

        @testset "IgG Subclass Differences" begin
            igg1 = create_igg1("test1")
            igg2 = create_igg2("test2")
            igg4 = create_igg4("test4")

            # All should have appropriate FcRn affinity
            @test igg1.fcrn_affinity > 0
            @test igg2.fcrn_affinity > 0
            @test igg4.fcrn_affinity > 0

            # Effector function differences
            @test igg1.effector_function == true
            @test igg4.effector_function == false
        end

        @testset "Immunogenicity State" begin
            # Default (no ADA)
            imm = ImmunogenicityState()
            @test imm.ada_positive == false
            @test imm.clearance_multiplier == 1.0

            # With ADA
            ada = ImmunogenicityState(ada_positive=true, ada_titer=100.0)
            @test ada.ada_positive == true
            @test ada.clearance_multiplier > 1.0  # Increased clearance
        end

        @testset "Get mAb and Target from Database" begin
            # Access via database directly
            @test haskey(MAB_DATABASE, "rituximab")
            mab = MAB_DATABASE["rituximab"]
            @test mab.name == "rituximab"

            @test haskey(TARGET_DATABASE, "CD20")
            target = TARGET_DATABASE["CD20"]
            @test target.name == "CD20"

            # Unknown - database access
            @test !haskey(MAB_DATABASE, "unknown_xyz")
            @test !haskey(TARGET_DATABASE, "unknown_xyz")
        end
    end

end

println("\n" * "=" ^ 70)
println("All Blood Advanced v2.7.0 tests completed!")
println("=" ^ 70)
