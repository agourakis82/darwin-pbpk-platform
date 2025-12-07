"""
Test suite for Transporter Ontogeny module

Validates:
1. Age category classification
2. Sigmoidal maturation functions
3. Renal/Hepatic/Intestinal/BBB transporter ontogeny
4. OBO Foundry ontology integration
5. ICD-10/11 classification
6. Clinical prediction accuracy
"""

using Test

# Include the module directly for testing
include("../src/DarwinPBPK/compartments/transporter_ontogeny.jl")
using .TransporterOntogeny

# Import specific items needed for tests
import .TransporterOntogeny: UBERON_TERMS, CL_TERMS, GO_TERMS, PR_TERMS
import .TransporterOntogeny: OBO_FOUNDRY_PREFIXES, export_transporter_to_jsonld

@testset "TransporterOntogeny Module" begin

    @testset "PediatricAge Construction" begin
        # Test term neonate
        age_neonate = PediatricAge(7.0; gestational_weeks=40.0)
        @test age_neonate.category == TERM_NEONATE
        @test age_neonate.postnatal_days == 7.0
        @test age_neonate.postmenstrual_age_weeks ≈ 41.0

        # Test preterm neonate
        age_preterm = PediatricAge(7.0; gestational_weeks=32.0)
        @test age_preterm.category == PRETERM_NEONATE
        @test age_preterm.corrected_age_days == 0.0  # Still negative corrected age

        # Test infant (6 months)
        age_infant = PediatricAge(months=6.0)
        @test age_infant.category == INFANT
        @test age_infant.postnatal_days ≈ 182.64 atol=1.0

        # Test child (8 years)
        age_child = PediatricAge(years=8.0)
        @test age_child.category == CHILD

        # Test adolescent (15 years)
        age_adolescent = PediatricAge(years=15.0)
        @test age_adolescent.category == ADOLESCENT

        # Test adult
        age_adult = PediatricAge(years=30.0)
        @test age_adult.category == ADULT

        # Test elderly
        age_elderly = PediatricAge(years=70.0)
        @test age_elderly.category == ELDERLY
    end

    @testset "Sigmoidal Maturation Function" begin
        # Create test maturation function
        mat = SigmoidalMaturation(
            200.0,  # TM50 = 200 days
            2.0,    # gamma
            1.0,    # Fmax
            0.1,    # Fmin
            :days
        )

        # At birth (day 0): should be Fmin
        age_birth = PediatricAge(0.0)
        @test calculate_maturation(mat, age_birth) ≈ 0.1

        # At TM50 (200 days): should be ~0.55 (midpoint between Fmin and Fmax)
        age_tm50 = PediatricAge(200.0)
        factor_tm50 = calculate_maturation(mat, age_tm50)
        @test factor_tm50 ≈ 0.55 atol=0.05

        # At 2 years: should be close to Fmax
        age_2y = PediatricAge(years=2.0)
        factor_2y = calculate_maturation(mat, age_2y)
        @test factor_2y > 0.9

        # At 5 years: should be ~Fmax
        age_5y = PediatricAge(years=5.0)
        factor_5y = calculate_maturation(mat, age_5y)
        @test factor_5y > 0.98
    end

    @testset "Renal Transporter Ontogeny" begin
        # Test OAT1 at various ages
        age_birth = PediatricAge(0.0)
        age_6m = PediatricAge(months=6.0)
        age_1y = PediatricAge(years=1.0)
        age_adult = PediatricAge(years=25.0)

        # OAT1: TM50 ~7 months, Fmin ~5%
        oat1_birth = get_transporter_ontogeny(:OAT1, age_birth)
        oat1_6m = get_transporter_ontogeny(:OAT1, age_6m)
        oat1_1y = get_transporter_ontogeny(:OAT1, age_1y)
        oat1_adult = get_transporter_ontogeny(:OAT1, age_adult)

        @test oat1_birth < 0.15  # Very low at birth
        @test 0.35 < oat1_6m < 0.65  # Around 50% at 6 months
        @test oat1_1y > 0.7  # >70% at 1 year
        @test oat1_adult > 0.99  # ~100% in adult

        # Test OCT2 (faster maturation)
        oct2_birth = get_transporter_ontogeny(:OCT2, age_birth)
        oct2_6m = get_transporter_ontogeny(:OCT2, age_6m)

        @test oct2_birth > oat1_birth  # OCT2 has higher baseline
        @test oct2_6m > oat1_6m  # OCT2 matures faster

        # Test all renal transporters at 6 months
        renal_6m = get_renal_transporter_ontogeny(age_6m)
        @test length(renal_6m) >= 10  # At least 10 transporters
        @test haskey(renal_6m, :OAT1)
        @test haskey(renal_6m, :OAT3)
        @test haskey(renal_6m, :OCT2)
        @test haskey(renal_6m, :MATE1)
        @test haskey(renal_6m, :P_gp_renal)
    end

    @testset "Hepatic Transporter Ontogeny" begin
        age_birth = PediatricAge(0.0)
        age_1y = PediatricAge(years=1.0)
        age_adult = PediatricAge(years=25.0)

        # OATP1B1: slow maturation (TM50 ~1.5 years)
        oatp1b1_birth = get_transporter_ontogeny(:OATP1B1, age_birth)
        oatp1b1_1y = get_transporter_ontogeny(:OATP1B1, age_1y)
        oatp1b1_adult = get_transporter_ontogeny(:OATP1B1, age_adult)

        @test oatp1b1_birth < 0.15  # Very low at birth
        @test oatp1b1_1y < 0.5  # Still <50% at 1 year (slow!)
        @test oatp1b1_adult > 0.99

        # NTCP: very slow (TM50 ~2 years) - explains neonatal cholestasis
        ntcp_birth = get_transporter_ontogeny(:NTCP, age_birth)
        @test ntcp_birth < 0.1  # Critically low

        # BSEP: even slower (TM50 ~2.5 years)
        bsep_birth = get_transporter_ontogeny(:BSEP, age_birth)
        @test bsep_birth < 0.15

        # Get all hepatic at 1 year
        hepatic_1y = get_hepatic_transporter_ontogeny(age_1y)
        @test length(hepatic_1y) >= 8
        @test haskey(hepatic_1y, :OATP1B1)
        @test haskey(hepatic_1y, :OCT1)
        @test haskey(hepatic_1y, :BSEP)
    end

    @testset "Intestinal Transporter Ontogeny" begin
        age_3m = PediatricAge(months=3.0)
        age_1y = PediatricAge(years=1.0)

        # PEPT1: early maturation (TM50 ~3 months) - for amino acid/peptide nutrition
        pept1_3m = get_transporter_ontogeny(:PEPT1, age_3m)
        @test pept1_3m > 0.6  # >60% at 3 months

        # P-gp intestinal: moderate
        pgp_3m = get_transporter_ontogeny(:P_gp_intestinal, age_3m)
        @test pgp_3m > 0.35  # Higher baseline than hepatic

        # Get all intestinal
        intestinal_1y = get_intestinal_transporter_ontogeny(age_1y)
        @test length(intestinal_1y) >= 5
        @test haskey(intestinal_1y, :PEPT1)
        @test haskey(intestinal_1y, :P_gp_intestinal)
        @test haskey(intestinal_1y, :BCRP_intestinal)
    end

    @testset "BBB Transporter Ontogeny" begin
        age_birth = PediatricAge(0.0)
        age_6m = PediatricAge(months=6.0)

        # GLUT1: very early maturation (critical for brain glucose)
        glut1_birth = get_transporter_ontogeny(:GLUT1, age_birth)
        @test glut1_birth > 0.65  # 70% at birth

        # LAT1: early (for amino acids)
        lat1_birth = get_transporter_ontogeny(:LAT1, age_birth)
        @test lat1_birth > 0.50  # 55% at birth

        # MCT1: very early (ketone bodies crucial for neonatal brain)
        mct1_birth = get_transporter_ontogeny(:MCT1, age_birth)
        @test mct1_birth > 0.75  # High at birth

        # P-gp BBB: biphasic, lower at birth → CNS drug exposure
        pgp_bbb_birth = get_transporter_ontogeny(:P_gp_BBB, age_birth)
        @test pgp_bbb_birth < 0.5  # Lower than nutrient transporters

        bbb_6m = get_bbb_transporter_ontogeny(age_6m)
        @test length(bbb_6m) >= 5
    end

    @testset "OBO Foundry Integration" begin
        # Test UBERON terms
        @test haskey(UBERON_TERMS, :kidney)
        @test UBERON_TERMS[:kidney].id == "UBERON:0002113"
        @test UBERON_TERMS[:proximal_tubule].id == "UBERON:0004134"
        @test haskey(UBERON_TERMS, :liver)
        @test haskey(UBERON_TERMS, :blood_brain_barrier)

        # Test Cell Ontology terms
        @test haskey(CL_TERMS, :hepatocyte)
        @test CL_TERMS[:hepatocyte].id == "CL:0000182"
        @test haskey(CL_TERMS, :enterocyte)
        @test haskey(CL_TERMS, :brain_endothelial)

        # Test GO terms
        @test haskey(GO_TERMS, :drug_transmembrane_transport)
        @test GO_TERMS[:organic_anion_transport].id == "GO:0015711"
        @test haskey(GO_TERMS, :ABC_transporter)

        # Test Protein Ontology terms
        @test haskey(PR_TERMS, :OAT1)
        @test PR_TERMS[:OAT1].gene == "SLC22A6"
        @test PR_TERMS[:OATP1B1].gene == "SLCO1B1"
        @test haskey(PR_TERMS, :P_gp)

        # Test OBO prefixes
        @test haskey(OBO_FOUNDRY_PREFIXES, "UBERON")
        @test haskey(OBO_FOUNDRY_PREFIXES, "CHEBI")
        @test haskey(OBO_FOUNDRY_PREFIXES, "DOID")
        @test haskey(OBO_FOUNDRY_PREFIXES, "ICD10")
        @test haskey(OBO_FOUNDRY_PREFIXES, "ICD11")
    end

    @testset "ICD-10/11 Classification" begin
        # Test neonatal conditions
        prematurity = get_icd_codes_for_condition(:prematurity)
        @test "P07.0" in prematurity.icd10
        @test "KA21.0" in prematurity.icd11
        @test prematurity.doid == "DOID:0060673"

        # Test CKD
        ckd = get_icd_codes_for_condition(:pediatric_ckd)
        @test "N18.3" in ckd.icd10
        @test ckd.doid == "DOID:784"

        # Test biliary atresia
        ba = get_icd_codes_for_condition(:biliary_atresia)
        @test "Q44.2" in ba.icd10
        @test ba.doid == "DOID:8545"

        # Test oncology
        all_codes = get_icd_codes_for_condition(:pediatric_all)
        @test "C91.00" in all_codes.icd10
    end

    @testset "DOID Pediatric Conditions" begin
        # Test DOID integration
        conditions = get_doid_pediatric_conditions()
        @test length(conditions) >= 8
        @test "DOID:784" in conditions  # CKD
        @test "DOID:8545" in conditions  # Biliary atresia

        # Test disease modifiers
        ckd_mod = get_pediatric_disease_modifiers("DOID:784")
        @test ckd_mod.name == "chronic kidney disease"
        @test :OAT1_reduced in ckd_mod.transporter_effects

        pfic_mod = get_pediatric_disease_modifiers("DOID:0060643")
        @test :BSEP_deficient in pfic_mod.transporter_effects
    end

    @testset "Clinical Prediction" begin
        # Predict metformin clearance in infant
        age_infant = PediatricAge(months=6.0)

        result = predict_pediatric_clearance(
            drug = "metformin",
            adult_clearance = 500.0,  # mL/min
            age = age_infant,
            transporters = [:OCT2, :MATE1, :MATE2K],
            fraction_each = [0.5, 0.35, 0.15]
        )

        @test result.drug == "metformin"
        @test result.overall_factor < 0.7  # Significantly reduced
        @test result.pediatric_clearance < 350.0  # <350 mL/min
        @test result.confidence in [:low, :moderate, :good, :high]
        @test haskey(result.ontogeny_factors, :OCT2)
        @test haskey(result.ontogeny_factors, :MATE1)
    end

    @testset "Ontogeny Curve Generation" begin
        curve = generate_ontogeny_curve(:OAT1; age_range_days=(0.0, 3650.0), n_points=50)

        @test curve.transporter == :OAT1
        @test length(curve.ages_days) == 50
        @test length(curve.factors) == 50
        @test curve.factors[1] < 0.15  # Low at birth
        @test curve.factors[end] > 0.95  # High at 10 years
        @test !isnan(curve.TM50_days)
        @test 150 < curve.TM50_days < 300  # ~7 months
    end

    @testset "JSON-LD Export" begin
        jsonld = export_transporter_to_jsonld(:OAT1)

        @test occursin("@context", jsonld)
        @test occursin("PR:", jsonld)
        @test occursin("GO:", jsonld)
        @test occursin("SLC22A6", jsonld)
        @test occursin("Organic Anion Transporter 1", jsonld)
        @test occursin("darwin:ontogeny", jsonld)
    end

    @testset "Transporter Profiles Complete" begin
        # Verify all transporters have required fields
        for (name, profile) in RENAL_TRANSPORTER_ONTOGENY
            @test haskey(profile, :name)
            @test haskey(profile, :gene)
            @test haskey(profile, :protein)
            @test haskey(profile, :maturation)
            @test haskey(profile, :adult_expression)
            @test haskey(profile, :substrates)
            @test length(profile.substrates) > 0
        end

        for (name, profile) in HEPATIC_TRANSPORTER_ONTOGENY
            @test haskey(profile, :name)
            @test haskey(profile, :gene)
            @test haskey(profile, :zonal_distribution)  # Hepatic has zonation
        end
    end

    @testset "Apply Ontogeny Scaling" begin
        age_neonate = PediatricAge(days=7.0)

        # Adult furosemide renal clearance via OAT1
        adult_cl = 100.0  # mL/min
        pediatric_cl = apply_ontogeny_scaling(adult_cl, :OAT1, age_neonate)

        @test pediatric_cl < 20.0  # <20% of adult
        @test pediatric_cl > 0.0
    end
end

println("All TransporterOntogeny tests passed!")
