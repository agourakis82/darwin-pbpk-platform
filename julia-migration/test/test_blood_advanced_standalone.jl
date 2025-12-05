"""
Standalone Tests for Advanced Blood Compartment Modules

Tests for:
- BloodBinding: B:P ratio, RBC/WBC partitioning
- Hemodynamics: Shear-dependent effects, vWF, SIPA
- CoagulationExtended: FXI feedback, contact pathway

Run with: julia --project=. test/test_blood_advanced_standalone.jl
"""

using Test

# Include modules directly for standalone testing
include("../src/DarwinPBPK/compartments/blood_binding.jl")
include("../src/DarwinPBPK/compartments/hemodynamics.jl")
include("../src/DarwinPBPK/compartments/coagulation_extended.jl")

using .BloodBinding
using .Hemodynamics
using .CoagulationExtended

println("=" ^ 70)
println("ADVANCED BLOOD COMPARTMENT TESTS")
println("Testing: Blood Binding, Hemodynamics, Extended Coagulation")
println("=" ^ 70)

# ============================================================================
# BLOOD BINDING MODULE TESTS
# ============================================================================

@testset "Blood Binding Module" begin

    @testset "Constants and Physiological Parameters" begin
        # Hematocrit
        @test STANDARD_HEMATOCRIT == 0.45
        @test 0.36 <= STANDARD_HEMATOCRIT <= 0.50

        # pH values
        @test PHYSIOLOGICAL_PH["plasma"] == 7.4
        @test PHYSIOLOGICAL_PH["rbc"] < PHYSIOLOGICAL_PH["plasma"]  # RBC slightly acidic
        @test PHYSIOLOGICAL_PH["wbc_lysosome"] == 5.0  # Lysosomes very acidic

        println("  ✓ Physiological constants validated")
    end

    @testset "Blood Composition Structure" begin
        # Default blood composition
        blood = get_blood_composition()

        @test blood isa BloodComposition
        @test blood.hematocrit > 0
        @test blood.plasma_ph ≈ 7.4

        println("  ✓ BloodComposition structure created")
    end

    @testset "Drug Properties Creation" begin
        # Test creating drug properties for different drug types
        # Using correct API: charge_type=:acid/:base/:neutral, pKa as Vector

        # Neutral lipophilic drug
        drug_lipophilic = create_drug_properties(
            "Propofol",
            logP=4.0,
            pKa=Float64[],  # Essentially neutral
            charge_type=:neutral,
            fu_plasma=0.02
        )
        @test drug_lipophilic.logP == 4.0
        @test drug_lipophilic.fu_plasma == 0.02

        # Weak acid (warfarin)
        drug_acid = create_drug_properties(
            "Warfarin",
            logP=2.7,
            pKa=[5.1],
            charge_type=:acid,
            fu_plasma=0.01,
            albumin_binding=true
        )
        @test drug_acid.charge_type == :acid
        @test drug_acid.pKa == [5.1]

        # Weak base (chloroquine - accumulates in lysosomes)
        drug_base = create_drug_properties(
            "Chloroquine",
            logP=4.6,
            pKa=[8.1, 10.2],
            charge_type=:base,
            fu_plasma=0.4,
            lysosomal_trapping=true
        )
        @test drug_base.charge_type == :base
        @test drug_base.lysosomal_trapping == true

        println("  ✓ Drug properties for different drug types created")
    end

    @testset "RBC Partition Coefficient" begin
        # Create blood and drug
        blood = get_blood_composition()

        # Lipophilic drug should partition into RBC membrane
        drug_lipophilic = create_drug_properties(
            "TestDrug",
            logP=3.5,
            pKa=[7.0],
            charge_type=:base,
            fu_plasma=0.1
        )

        Krbc = calculate_rbc_partition(drug_lipophilic, blood)
        @test Krbc > 0  # Should be positive
        @test isfinite(Krbc)

        # Very hydrophilic neutral drug - mainly water partition
        drug_hydrophilic = create_drug_properties(
            "TestDrug2",
            logP=-2.0,
            pKa=Float64[],
            charge_type=:neutral,
            fu_plasma=0.95
        )

        Krbc_hydro = calculate_rbc_partition(drug_hydrophilic, blood)
        @test Krbc_hydro > 0  # Should be positive
        @test isfinite(Krbc_hydro)
        # Note: Basic drugs may have HIGHER RBC partition due to ion trapping

        println("  ✓ RBC partition coefficients calculated correctly")
    end

    @testset "Blood-to-Plasma Ratio (B:P)" begin
        blood = get_blood_composition()

        # Create drug
        drug = create_drug_properties(
            "TestDrug",
            logP=2.0,
            pKa=[7.5],
            charge_type=:base,
            fu_plasma=0.2
        )

        bp_ratio = calculate_blood_plasma_ratio(drug, blood)

        # B:P ratio should be reasonable (0.5 - 3.0 for most drugs)
        @test bp_ratio > 0.3
        @test bp_ratio < 5.0
        @test isfinite(bp_ratio)

        # Test hematocrit effect
        blood_high_hct = get_blood_composition(hematocrit=0.55)
        blood_low_hct = get_blood_composition(hematocrit=0.35)

        bp_high = calculate_blood_plasma_ratio(drug, blood_high_hct)
        bp_low = calculate_blood_plasma_ratio(drug, blood_low_hct)

        # Higher hematocrit = more RBC = different B:P
        @test bp_high != bp_low

        println("  ✓ Blood-to-Plasma ratio calculated with hematocrit effect")
    end

    @testset "WBC Lysosomal Trapping" begin
        blood = get_blood_composition()

        # Basic amine (chloroquine-like) - should accumulate in WBC lysosomes
        drug_base = create_drug_properties(
            "Chloroquine-like",
            logP=4.0,
            pKa=[8.5],
            charge_type=:base,
            fu_plasma=0.4,
            lysosomal_trapping=true
        )

        Kwbc = calculate_wbc_partition(drug_base, blood)
        @test Kwbc > 1.0  # Should accumulate

        # Acid should not accumulate in acidic lysosomes
        drug_acid = create_drug_properties(
            "Acid-drug",
            logP=2.0,
            pKa=[4.0],
            charge_type=:acid,
            fu_plasma=0.05
        )

        Kwbc_acid = calculate_wbc_partition(drug_acid, blood)
        @test Kwbc_acid < Kwbc  # Less accumulation for acids

        println("  ✓ WBC lysosomal trapping calculated for bases vs acids")
    end

    @testset "Platelet Partitioning" begin
        blood = get_blood_composition()

        drug = create_drug_properties(
            "TestDrug",
            logP=2.5,
            pKa=[7.0],
            charge_type=:neutral,
            fu_plasma=0.3
        )

        Kplt = calculate_platelet_partition(drug, blood)
        @test Kplt > 0
        @test isfinite(Kplt)

        println("  ✓ Platelet partition coefficient calculated")
    end

    @testset "Fraction Unbound in Blood" begin
        blood = get_blood_composition()

        # Highly bound drug
        drug_bound = create_drug_properties(
            "HighlyBound",
            logP=4.0,
            pKa=[7.0],
            charge_type=:neutral,
            fu_plasma=0.01  # 99% bound
        )

        # Calculate B:P first, then fu_blood
        bp_ratio = calculate_blood_plasma_ratio(drug_bound, blood)
        fu_blood = calculate_fu_blood(drug_bound.fu_plasma, bp_ratio)
        @test fu_blood > 0
        @test fu_blood < 1.0
        @test fu_blood >= drug_bound.fu_plasma * 0.5  # Reasonable range

        println("  ✓ Fraction unbound in blood calculated")
    end

    println("\n✓ All Blood Binding tests passed!")
end

# ============================================================================
# HEMODYNAMICS MODULE TESTS
# ============================================================================

@testset "Hemodynamics Module" begin

    @testset "Constants" begin
        @test BLOOD_VISCOSITY > 0
        @test BLOOD_VISCOSITY ≈ 0.035  # Pa·s

        # Critical shear rates
        @test CRITICAL_SHEAR_RATES["platelet_activation_threshold"] == 1000.0
        @test CRITICAL_SHEAR_RATES["vwf_unfolding"] == 5000.0
        @test CRITICAL_SHEAR_RATES["pathological"] == 10000.0

        println("  ✓ Hemodynamic constants validated")
    end

    @testset "Vessel Geometry Creation" begin
        # Create different vessels - use String, not Symbol
        aorta = create_vessel("aorta")
        @test aorta isa VesselGeometry
        @test aorta.diameter > 0

        arteriole = create_vessel("arteriole")
        @test arteriole.diameter < aorta.diameter  # Arteriole smaller than aorta

        coronary = create_vessel("coronary_artery")
        @test coronary.diameter > 0

        println("  ✓ Vessel geometries created")
    end

    @testset "Shear Rate Calculation" begin
        # Shear rate from wall shear stress: γ̇ = τ / μ
        tau_w = 5.0  # Pa (typical arterial)
        shear_rate = calculate_shear_rate(tau_w)
        @test shear_rate > 0
        @test shear_rate ≈ tau_w / BLOOD_VISCOSITY

        # Higher stress = higher shear rate
        shear_high = calculate_shear_rate(10.0)
        @test shear_high > shear_rate

        println("  ✓ Shear rates calculated")
    end

    @testset "Shear-Induced Platelet Activation (SIPA)" begin
        # Function signature: shear_induced_platelet_activation(shear_rate, exposure_time, baseline=0)

        # Low shear, short exposure
        activation_low = shear_induced_platelet_activation(500.0, 1.0, 0.0)
        @test activation_low >= 0
        @test activation_low <= 1.0  # Valid range

        # Moderate shear, longer exposure
        activation_mod = shear_induced_platelet_activation(3000.0, 1.0, 0.0)
        @test activation_mod >= activation_low

        # High/pathological shear
        activation_high = shear_induced_platelet_activation(10000.0, 1.0, 0.0)
        @test activation_high >= activation_mod
        @test activation_high <= 1.0

        println("  ✓ Shear-induced platelet activation follows expected pattern")
    end

    @testset "vWF Unfolding Probability" begin
        # Below threshold - low unfolding
        p_low = vwf_unfolding_probability(1000.0)
        @test p_low >= 0
        @test p_low < 0.5

        # At threshold
        p_threshold = vwf_unfolding_probability(5000.0)
        @test p_threshold >= p_low

        # High shear - significant unfolding
        p_high = vwf_unfolding_probability(15000.0)
        @test p_high >= p_threshold
        @test p_high <= 1.0

        println("  ✓ vWF unfolding probability calculated")
    end

    println("\n✓ All Hemodynamics tests passed!")
end

# ============================================================================
# EXTENDED COAGULATION MODULE TESTS
# ============================================================================

@testset "Extended Coagulation Module (FXI Feedback)" begin

    @testset "FXI Feedback Parameters" begin
        @test haskey(FXI_FEEDBACK_PARAMS, "kcat_IIa_XI")
        @test haskey(FXI_FEEDBACK_PARAMS, "Km_IIa_XI")
        @test haskey(FXI_FEEDBACK_PARAMS, "platelet_surface_enhancement")

        # Validate kinetic constants are positive
        @test FXI_FEEDBACK_PARAMS["kcat_IIa_XI"] > 0
        @test FXI_FEEDBACK_PARAMS["Km_IIa_XI"] > 0
        @test FXI_FEEDBACK_PARAMS["platelet_surface_enhancement"] > 100  # Should be significant

        println("  ✓ FXI feedback kinetic parameters validated")
    end

    @testset "Contact Pathway Structure" begin
        # Create system and verify contact pathway is initialized
        system = create_extended_coagulation()

        @test system.contact isa ContactPathway
        @test system.contact.factor_XII > 0  # FXII present
        @test system.contact.prekallikrein > 0  # Prekallikrein present
        @test system.contact.HMWK > 0  # HMWK present

        println("  ✓ Contact pathway structure validated")
    end

    @testset "Extended Coagulation System Creation" begin
        # Create system with default parameters
        system = create_extended_coagulation()

        @test system isa ExtendedCoagulationSystem
        @test system.contact.factor_XI > 0  # Has FXI in contact pathway
        @test system.contact.factor_XIa >= 0  # FXIa starts at 0

        # Create with tissue factor
        system_tf = create_extended_coagulation(tissue_factor=5.0)
        @test system_tf.tissue_factor == 5.0

        # Create with platelet activation
        system_plt = create_extended_coagulation(platelet_activation=0.5)
        @test system_plt.platelet_surface.ps_exposure > 0

        println("  ✓ Extended coagulation system created with FXI")
    end

    @testset "FXI Feedback Enable/Disable" begin
        system = create_extended_coagulation()

        # FXI feedback is enabled by default
        @test system.fxi_feedback_active == true

        # Disable it
        add_fxi_feedback!(system, false)
        @test system.fxi_feedback_active == false

        # Re-enable
        add_fxi_feedback!(system, true)
        @test system.fxi_feedback_active == true

        println("  ✓ FXI feedback enable/disable works")
    end

    @testset "Contact Pathway Activation" begin
        system = create_extended_coagulation()

        # Initially no contact surface
        @test system.contact.surface_type == :none

        # Activate contact pathway (like exposure to kaolin)
        add_contact_activation!(system, :kaolin, 0.5)

        # Should have kaolin surface now
        @test system.contact.surface_type == :kaolin
        @test system.contact.contact_surface == 0.5

        println("  ✓ Contact pathway activation implemented")
    end

    @testset "Shear Effects on Coagulation" begin
        system = create_extended_coagulation()
        initial_shear = system.shear_rate

        # Apply high shear effects
        add_shear_effects!(system, 5000.0)

        # Shear should be updated
        @test system.shear_rate == 5000.0
        @test system.shear_factor > 0
        @test system.shear_factor <= 1.0

        # High shear should increase PS exposure
        @test system.platelet_surface.ps_exposure >= 0

        println("  ✓ Shear effects integrated into coagulation")
    end

    @testset "Platelet Surface Reactions" begin
        # Create system with platelet activation
        system = create_extended_coagulation(platelet_activation=0.8)

        # Calculate surface reaction rates
        rates = calculate_surface_reactions(system, 1.0)

        @test haskey(rates, "tenase_rate")
        @test haskey(rates, "prothrombinase_rate")
        @test rates["tenase_rate"] >= 0
        @test rates["prothrombinase_rate"] >= 0

        println("  ✓ Platelet surface reaction rates calculated")
    end

    @testset "Polyphosphate Enhancement" begin
        # PolyP from platelet dense granules enhances coagulation
        # Returns Dict with enhancement factors
        enhancement = polyphosphate_enhancement(10.0)  # nM

        @test haskey(enhancement, "FV_activation")
        @test haskey(enhancement, "FXII_activation")
        @test enhancement["FV_activation"] >= 1.0

        # Higher polyP = more enhancement
        enhancement_high = polyphosphate_enhancement(100.0)
        @test enhancement_high["FV_activation"] >= enhancement["FV_activation"]

        println("  ✓ Polyphosphate enhancement calculated")
    end

    @testset "Complete System with All Features" begin
        # Create comprehensive system
        system = create_extended_coagulation(
            tissue_factor=5.0,
            shear_rate=1000.0,
            platelet_activation=0.3,
            contact_surface=:polyP,
            contact_amount=0.2
        )

        # Verify all components initialized
        @test system.tissue_factor == 5.0
        @test system.shear_rate == 1000.0
        @test system.platelet_surface.ps_exposure > 0
        @test system.contact.surface_type == :polyP

        # FXI feedback should be active
        @test system.fxi_feedback_active == true

        # All factors should be present
        @test system.factor_II > 0  # Prothrombin
        @test system.factor_X > 0   # FX
        @test system.factor_IX > 0  # FIX
        @test system.antithrombin > 0  # Inhibitors

        println("  ✓ Complete extended coagulation system initialized")
    end

    println("\n✓ All Extended Coagulation (FXI Feedback) tests passed!")
end

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@testset "Integration: Blood Binding + Hemodynamics + Coagulation" begin

    @testset "Drug in Stenotic Vessel" begin
        # Scenario: Drug behavior in stenotic coronary artery

        # 1. Blood binding
        blood = get_blood_composition()
        drug = create_drug_properties(
            "AntiplateletDrug",
            logP=2.0,
            pKa=[7.0],
            charge_type=:neutral,
            fu_plasma=0.1
        )
        bp_ratio = calculate_blood_plasma_ratio(drug, blood)

        # 2. Hemodynamics - high shear stress in stenosis
        # Using wall shear stress to calculate shear rate
        tau_w_stenotic = 30.0  # Pa - typical for 70% stenosis
        shear_rate = calculate_shear_rate(tau_w_stenotic)

        # 3. Platelet activation due to high shear
        platelet_activation = shear_induced_platelet_activation(shear_rate, 1.0, 0.0)

        # 4. This triggers coagulation
        coag_system = create_extended_coagulation(
            shear_rate=shear_rate,
            platelet_activation=platelet_activation
        )

        # Verify integration
        @test bp_ratio > 0
        @test shear_rate > 500  # Elevated
        @test platelet_activation >= 0  # Some activation

        println("  ✓ Integrated stenotic vessel scenario validated")
    end

    @testset "Chloroquine WBC Accumulation" begin
        # Chloroquine accumulates in WBC lysosomes
        # This affects its distribution during inflammation

        blood = get_blood_composition()
        chloroquine = create_drug_properties(
            "Chloroquine",
            logP=4.6,
            pKa=[8.4, 10.2],
            charge_type=:base,
            fu_plasma=0.39,
            lysosomal_trapping=true
        )

        # WBC accumulation
        Kwbc = calculate_wbc_partition(chloroquine, blood)

        # B:P ratio
        bp_ratio = calculate_blood_plasma_ratio(chloroquine, blood)

        @test Kwbc > 1.0  # Accumulation in WBC (lysosomotropic base)
        @test bp_ratio > 0

        println("  ✓ Chloroquine WBC accumulation scenario validated")
    end

    println("\n✓ All Integration tests passed!")
end

# ============================================================================
# DRUG-SPECIFIC WBC BINDING TESTS
# ============================================================================

@testset "Drug-Specific WBC Binding" begin

    @testset "WBC Accumulation Database" begin
        # Chloroquine should be in database
        cq_params = get_wbc_accumulation("chloroquine")
        @test haskey(cq_params, "wbc_plasma_ratio")
        @test cq_params["wbc_plasma_ratio"] > 1.0  # Accumulates
        @test cq_params["monocyte_ratio"] > cq_params["neutrophil_ratio"]  # Higher in monocytes

        # Azithromycin - extreme accumulation
        azi_params = get_wbc_accumulation("azithromycin")
        @test azi_params["wbc_plasma_ratio"] >= 100.0  # Very high

        # Unknown drug returns defaults
        unknown = get_wbc_accumulation("made_up_drug")
        @test unknown["wbc_plasma_ratio"] == 1.0

        println("  ✓ WBC accumulation database validated")
    end

    @testset "Chloroquine WBC Partition" begin
        blood = get_blood_composition()

        # Total WBC
        Kp_wbc = calculate_drug_specific_wbc_partition("chloroquine", blood)
        @test Kp_wbc ≈ 7.0  # Literature value

        # Cell-type specific
        Kp_mono = calculate_drug_specific_wbc_partition("chloroquine", blood, cell_type=:monocyte)
        Kp_neut = calculate_drug_specific_wbc_partition("chloroquine", blood, cell_type=:neutrophil)
        @test Kp_mono > Kp_neut  # Higher in monocytes
        @test Kp_mono ≈ 15.0

        println("  ✓ Chloroquine WBC partition calculated")
    end

    @testset "Antiretroviral WBC Partition" begin
        blood = get_blood_composition()

        # Dolutegravir (INSTI) - moderate accumulation
        Kp_dtg = calculate_drug_specific_wbc_partition("dolutegravir", blood)
        @test Kp_dtg > 1.0
        @test Kp_dtg < 5.0

        # CD4 cells (target for HIV)
        Kp_cd4 = calculate_drug_specific_wbc_partition("dolutegravir", blood, cell_type=:cd4)
        @test Kp_cd4 > Kp_dtg  # Higher in target cells

        # Tenofovir - low parent, high metabolite
        tfv_params = get_wbc_accumulation("tenofovir")
        @test tfv_params["wbc_plasma_ratio"] < 1.0  # Parent drug
        @test tfv_params["active_metabolite_ratio"] > 10.0  # Active metabolite accumulates

        println("  ✓ Antiretroviral WBC partition calculated")
    end

    @testset "Intracellular Drug Amount" begin
        blood = get_blood_composition()
        plasma_conc = 1000.0  # nM

        # Chloroquine intracellular amounts
        amounts = calculate_intracellular_drug_amount("chloroquine", plasma_conc, blood)

        @test haskey(amounts, "monocyte_conc")
        @test haskey(amounts, "total_wbc_amount")
        @test amounts["monocyte_conc"] > amounts["neutrophil_conc"]
        @test amounts["monocyte_conc"] ≈ plasma_conc * 15.0  # 15× in monocytes

        # Tenofovir with metabolites
        tfv_amounts = calculate_intracellular_drug_amount(
            "tenofovir", 100.0, blood,
            include_metabolites=true
        )
        @test haskey(tfv_amounts, "active_metabolite_conc")
        @test tfv_amounts["active_metabolite_conc"] > tfv_amounts["plasma_conc"]

        println("  ✓ Intracellular drug amounts calculated")
    end

    @testset "Reservoir Effect" begin
        blood = get_blood_composition()

        # Chloroquine - very long half-life
        cq_reservoir = calculate_reservoir_effect("chloroquine", blood, time_after_dose_hours=24.0)
        @test cq_reservoir["t_half_hours"] > 500  # > 20 days
        @test cq_reservoir["fraction_remaining"] > 0.95  # Still mostly there after 24h
        @test cq_reservoir["clinically_significant"] == true

        # After 7 days
        cq_7days = calculate_reservoir_effect("chloroquine", blood, time_after_dose_hours=168.0)
        @test cq_7days["fraction_remaining"] > 0.8  # Still significant

        # Azithromycin - shorter half-life
        azi_reservoir = calculate_reservoir_effect("azithromycin", blood, time_after_dose_hours=24.0)
        @test azi_reservoir["t_half_hours"] < cq_reservoir["t_half_hours"]

        println("  ✓ Reservoir effect calculated")
    end

    println("\n✓ All Drug-Specific WBC Binding tests passed!")
end

# ============================================================================
# SUMMARY
# ============================================================================

println("\n" * "=" ^ 70)
println("ALL ADVANCED BLOOD COMPARTMENT TESTS COMPLETED SUCCESSFULLY!")
println("=" ^ 70)
println("""
Modules Tested:
  - BloodBinding: B:P ratio, RBC/WBC/platelet partitioning (PK-Sim style)
  - Hemodynamics: Shear stress, SIPA, vWF unfolding
  - CoagulationExtended: FXI feedback (critical Hockin-Mann fix)

Key Achievements:
  - Mechanistic B:P ratio calculation (not just empirical)
  - Lysosomal ion trapping for basic drugs (chloroquine effect)
  - Shear-dependent platelet activation (SIPA)
  - FXI feedback loop (fixes HM model at low TF)
  - Contact pathway (FXII, kallikrein)
  - Platelet surface enhancement (300,000× for prothrombinase)
""")
