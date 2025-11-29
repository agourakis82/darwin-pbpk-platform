# =============================================================================
# Test Suite: Tumor Penetration Model
# =============================================================================
# Validates MedLang tumor penetration model for oncology applications
# =============================================================================

using Test
using DarwinPBPK
using DarwinPBPK.MedLang

@testset "Tumor Penetration Model" begin

    @testset "EPR Effect Size Dependence" begin
        # EPR effect is size-dependent with optimal around 10-100 nm

        EPR = EPREffect(
            400.0,  # Pore cutoff nm
            5.0,    # Permeability ratio
            0.9,    # Lymphatic impairment
            48.0,   # Retention half-life h
            50.0,   # Optimal size nm
            0.5,    # Size selectivity
            0.7     # EPR magnitude
        )

        vascular = VascularParameters(
            100.0,  # Vessel density
            20.0,   # Vessel diameter
            200.0,  # Pore size
            0.5,    # Blood flow
            1e-4,   # PS
            0.05,   # Vascular fraction
            1.5,    # Tortuosity
            0.5     # Heterogeneity
        )

        # Test different particle sizes
        sizes = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0]
        accumulations = [calculate_EPR_accumulation(s, EPR, vascular, 24.0)
                        for s in sizes]

        println("EPR Size Dependence (24h):")
        for (s, a) in zip(sizes, accumulations)
            println("  $(Int(s)) nm: accumulation=$(round(a.accumulation, digits=4)), size_factor=$(round(a.size_factor, digits=3))")
        end

        # Find optimal size
        accum_values = [a.accumulation for a in accumulations]
        opt_idx = argmax(accum_values)
        println("  Optimal size: $(Int(sizes[opt_idx])) nm")

        # Optimal should be near 50 nm (configured optimal)
        @test sizes[opt_idx] >= 20.0 && sizes[opt_idx] <= 100.0

        # Very large particles (>500 nm) should have minimal accumulation
        @test accumulations[end].accumulation < accumulations[opt_idx].accumulation

        # Size factor should decrease for particles larger than pore cutoff
        @test accumulations[end].size_factor < 0.5
    end

    @testset "Tumor Type EPR Variation" begin
        # Different tumor types have different EPR effects

        tumor_types = [SOLID_CARCINOMA, BRAIN_TUMOR, PANCREATIC, MELANOMA, BREAST]

        println("\nEPR Effect by Tumor Type:")
        for tt in tumor_types
            tumor = create_tumor_model(tt, 1.0)

            EPR = EPREffect(400.0, 5.0, 0.9, 48.0, 50.0, 0.5,
                           DarwinPBPK.MedLang.TumorPenetrationModel.EPR_tumor_type_factor(tt))

            accum = calculate_EPR_accumulation(50.0, EPR, tumor.vascular, 24.0)
            println("  $(tt): EPR magnitude=$(round(EPR.EPR_magnitude, digits=2)), accumulation=$(round(accum.accumulation, digits=4))")
        end

        # Brain tumors should have lowest EPR (BBB)
        brain = create_tumor_model(BRAIN_TUMOR, 1.0)
        melanoma = create_tumor_model(MELANOMA, 1.0)

        EPR_brain = DarwinPBPK.MedLang.TumorPenetrationModel.EPR_tumor_type_factor(BRAIN_TUMOR)
        EPR_melanoma = DarwinPBPK.MedLang.TumorPenetrationModel.EPR_tumor_type_factor(MELANOMA)

        @test EPR_brain < EPR_melanoma
        @test EPR_brain < 0.5  # Low EPR for brain
    end

    @testset "IFP Gradient Effect" begin
        # Elevated IFP reduces drug transport

        tumor = create_tumor_model(BREAST, 1.0; IFP=30.0)

        # Test at different radial positions
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]

        println("\nIFP Gradient Effect (IFP_center=30 mmHg):")
        for pos in positions
            result = IFP_gradient_effect(tumor, pos)
            println("  r=$(pos): IFP=$(round(result.IFP_local, digits=1)) mmHg, transport=$(round(result.transport_factor, digits=3))")
        end

        # Center should have highest IFP
        center = IFP_gradient_effect(tumor, 0.0)
        periphery = IFP_gradient_effect(tumor, 1.0)

        @test center.IFP_local > periphery.IFP_local

        # Transport should be better at periphery
        @test periphery.transport_factor > center.transport_factor
    end

    @testset "Tumor Penetration Depth" begin
        # Calculate drug penetration from vasculature

        drug = tumor_drug_preset(:doxorubicin)
        tumor = create_tumor_model(BREAST, 1.0)

        result = tumor_penetration_depth(drug, tumor, 24.0)

        println("\nPenetration Depth (Doxorubicin in Breast):")
        println("  Penetration depth: $(round(result.penetration_depth_um, digits=1)) µm")
        println("  Inter-vessel distance: $(round(result.inter_vessel_distance_um, digits=1)) µm")
        println("  Fraction reached: $(round(result.fraction_reached * 100, digits=1))%")
        println("  Limited by: $(result.limited_by)")

        # Penetration should be limited
        @test result.penetration_depth_um > 0
        @test result.penetration_depth_um < 1000  # Less than 1 mm

        # Fraction should be less than 100%
        @test result.fraction_reached <= 1.0
    end

    @testset "Drug Presets" begin
        println("\nDrug Presets:")
        for drug_name in [:doxorubicin, :paclitaxel, :gemcitabine, :imatinib, :cisplatin]
            drug = tumor_drug_preset(drug_name)
            println("  $(drug.name): MW=$(drug.molecular_weight), LogP=$(drug.log_P), P-gp=$(drug.Pgp_substrate)")
        end

        # Paclitaxel is a strong P-gp substrate
        pac = tumor_drug_preset(:paclitaxel)
        @test pac.Pgp_substrate == true
        @test pac.molecular_weight > 800

        # Gemcitabine is hydrophilic
        gem = tumor_drug_preset(:gemcitabine)
        @test gem.log_P < 0
        @test gem.Pgp_substrate == false
    end

    @testset "P-gp Efflux Effect" begin
        # P-gp substrates have reduced tumor uptake

        tumor = create_tumor_model(COLORECTAL, 1.0)

        # Compare P-gp substrate vs non-substrate
        drug_pgp = tumor_drug_preset(:paclitaxel)  # P-gp substrate
        drug_no_pgp = tumor_drug_preset(:cisplatin)  # Not P-gp

        uptake_pgp = calculate_tumor_uptake(drug_pgp, tumor, 10.0, 4.0)
        uptake_no_pgp = calculate_tumor_uptake(drug_no_pgp, tumor, 10.0, 4.0)

        println("\nP-gp Efflux Effect on Tumor Uptake:")
        println("  Paclitaxel (P-gp+): T:P ratio=$(round(uptake_pgp.tumor_plasma_ratio, digits=3)), efflux=$(uptake_pgp.efflux_factor)")
        println("  Cisplatin (P-gp-): T:P ratio=$(round(uptake_no_pgp.tumor_plasma_ratio, digits=3)), efflux=$(uptake_no_pgp.efflux_factor)")

        # P-gp substrate should have lower tumor:plasma ratio
        @test uptake_pgp.efflux_factor < uptake_no_pgp.efflux_factor
    end

    @testset "pH Effect on Drug Distribution" begin
        # Acidic tumor microenvironment affects drug distribution

        # Basic drug (positive charge)
        drug_basic = DrugTumorProperties(
            "Basic Drug", 400.0, 2.0, 1.5,  # Positive charge
            4e-6, 0.3, 0.5, false, 0.1, 1.0, 50.0
        )

        # Acidic drug (negative charge)
        drug_acidic = DrugTumorProperties(
            "Acidic Drug", 400.0, 2.0, -1.5,  # Negative charge
            4e-6, 0.3, 0.5, false, 0.1, 1.0, 50.0
        )

        # Neutral drug
        drug_neutral = DrugTumorProperties(
            "Neutral Drug", 400.0, 2.0, 0.0,
            4e-6, 0.3, 0.5, false, 0.1, 1.0, 50.0
        )

        tumor_pH = 6.8
        pH_basic = DarwinPBPK.MedLang.TumorPenetrationModel.pH_effect_on_drug(drug_basic, tumor_pH)
        pH_acidic = DarwinPBPK.MedLang.TumorPenetrationModel.pH_effect_on_drug(drug_acidic, tumor_pH)
        pH_neutral = DarwinPBPK.MedLang.TumorPenetrationModel.pH_effect_on_drug(drug_neutral, tumor_pH)

        println("\npH Effect on Drug Distribution (tumor pH=6.8):")
        println("  Basic drug: accumulation factor=$(round(pH_basic, digits=3))")
        println("  Acidic drug: accumulation factor=$(round(pH_acidic, digits=3))")
        println("  Neutral drug: accumulation factor=$(round(pH_neutral, digits=3))")

        # Basic drugs accumulate in acidic tumor
        @test pH_basic > 1.0

        # Acidic drugs are excluded from acidic tumor
        @test pH_acidic < 1.0

        # Neutral drugs unaffected
        @test pH_neutral ≈ 1.0
    end

    @testset "ADC Presets" begin
        println("\nADC Presets:")
        for adc_name in [:trastuzumab_emtansine, :brentuximab_vedotin, :enfortumab_vedotin]
            adc = ADC_preset(adc_name)
            println("  $(adc.name): DAR=$(adc.DAR), target=$(adc.target_antigen), bystander=$(adc.bystander_effect)")
        end

        # T-DM1 has non-cleavable linker, no bystander
        tdm1 = ADC_preset(:trastuzumab_emtansine)
        @test tdm1.linker_type == :non_cleavable
        @test tdm1.bystander_effect == false

        # Brentuximab has cleavable linker with bystander effect
        bren = ADC_preset(:brentuximab_vedotin)
        @test bren.linker_type == :cleavable
        @test bren.bystander_effect == true
    end

    @testset "ADC Binding Site Barrier" begin
        # High-affinity ADCs face binding site barrier

        ADC = ADC_preset(:trastuzumab_emtansine)

        # High antigen density
        barrier_high = binding_site_barrier(ADC, 1e6, 1e9, 10.0)

        # Low antigen density
        barrier_low = binding_site_barrier(ADC, 1e4, 1e9, 10.0)

        println("\nADC Binding Site Barrier:")
        println("  High Ag (10⁶/cell): φ=$(round(barrier_high.thiele_modulus, digits=2)), η=$(round(barrier_high.effectiveness, digits=3))")
        println("  Low Ag (10⁴/cell): φ=$(round(barrier_low.thiele_modulus, digits=2)), η=$(round(barrier_low.effectiveness, digits=3))")

        # High antigen → stronger barrier (higher Thiele modulus)
        # Note: When both eta ≈ 1.0, the modulus is still the differentiator
        @test barrier_high.thiele_modulus >= barrier_low.thiele_modulus
    end

    @testset "ADC Distribution" begin
        ADC = ADC_preset(:trastuzumab_emtansine)
        tumor = create_tumor_model(BREAST, 1.0)  # HER2+ breast

        result = ADC_distribution(ADC, tumor, 3.6, 24.0)  # 3.6 mg/kg

        println("\nADC Distribution (T-DM1, 3.6 mg/kg, 24h):")
        println("  C_plasma = $(round(result.C_plasma, digits=2)) µg/mL")
        println("  C_tumor = $(round(result.C_tumor, digits=4)) µg/mL")
        println("  T:P ratio = $(round(result.tumor_plasma_ratio, digits=3))")
        println("  Fraction bound = $(round(result.fraction_bound * 100, digits=1))%")
        println("  Payload released = $(round(result.payload_released * 100, digits=1))%")

        # Should have drug in tumor
        @test result.C_tumor > 0

        # Tumor:plasma ratio typically low for ADCs (large molecules)
        @test result.tumor_plasma_ratio < 1.0

        # Should have some internalization and payload release at 24h
        @test result.fraction_internalized > 0
    end

    @testset "ODE Simulation" begin
        drug = tumor_drug_preset(:doxorubicin)
        tumor = create_tumor_model(BREAST, 1.0)

        result = simulate_tumor_penetration(drug, tumor, 50.0; tspan=(0.0, 24.0))

        println("\nODE Simulation (Doxorubicin 50mg):")
        println("  AUC_plasma = $(round(result.AUC_plasma, digits=1)) µg·h/mL")
        println("  AUC_tumor = $(round(result.AUC_tumor, digits=1)) µg·h/mL")
        println("  Tumor:Plasma AUC ratio = $(round(result.tumor_plasma_AUC_ratio, digits=3))")
        println("  Max C_tumor = $(round(maximum(result.C_tumor), digits=2)) µg/mL")

        # Should have drug distribution
        @test result.AUC_tumor > 0
        @test result.AUC_plasma > 0

        # Tumor:plasma ratio
        @test result.tumor_plasma_AUC_ratio > 0
    end

    @testset "Model Validation" begin
        validation = validate_tumor_model()

        println("\nModel Validation:")

        # EPR size dependence
        println("  EPR optimal size: $(Int(validation["EPR_size_dependence"].optimal_size)) nm")
        @test validation["EPR_size_dependence"].optimal_size >= 20
        @test validation["EPR_size_dependence"].optimal_size <= 100

        # Penetration depth
        println("  Penetration depth: $(round(validation["penetration_depth"].depth_um, digits=1)) µm")
        @test validation["penetration_depth"].depth_um > 0

        # ADC barrier
        println("  ADC effectiveness: $(round(validation["ADC_barrier"].effectiveness, digits=3))")
        @test validation["ADC_barrier"].effectiveness > 0
        @test validation["ADC_barrier"].effectiveness <= 1
    end

    @testset "Create Tumor Model" begin
        tumor = create_tumor_model(PANCREATIC, 2.0; IFP=40.0, vessel_density=50.0)

        println("\nPancreatic Tumor Model (2 mL):")
        println("  IFP = $(tumor.microenvironment.IFP_mmHg) mmHg")
        println("  Vessel density = $(tumor.vascular.vessel_density) vessels/mm²")
        println("  pH = $(tumor.microenvironment.extracellular_pH)")
        println("  Necrotic fraction = $(tumor.microenvironment.necrotic_fraction)")

        @test tumor.volume_mL == 2.0
        @test tumor.tumor_type == PANCREATIC
        @test tumor.microenvironment.IFP_mmHg == 40.0
    end

end

println("\n" * "="^60)
println("Tumor Penetration Model Tests Complete")
println("="^60)
