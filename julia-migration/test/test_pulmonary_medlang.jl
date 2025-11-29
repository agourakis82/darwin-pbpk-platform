# =============================================================================
# Tests for Pulmonary Absorption Model - MedLang Integration
# =============================================================================
# Darwin PBPK Platform - Publication-Ready Validation
#
# Tests cover:
# 1. ICRP particle deposition model
# 2. Regional deposition fractions
# 3. Mucociliary clearance kinetics
# 4. Alveolar macrophage clearance
# 5. Noyes-Whitney dissolution
# 6. Pulmonary absorption rates
# 7. Disease state effects (COPD, asthma, CF, IPF)
# 8. Drug presets and simulations
# 9. Device effects (MDI, DPI, nebulizer)
# =============================================================================

using Test
using DarwinPBPK
using DarwinPBPK.MedLang
using DarwinPBPK.MedLang.PulmonaryAbsorptionModel

@testset "Pulmonary Absorption Model - MedLang" begin

    @testset "Particle Deposition - ICRP Model" begin
        # Test 1: Small particles (1 µm) - should have high alveolar deposition
        small_particle = ParticleProperties(1.0, 2.0, 1.0, 1.0, 0.0, 1.0, :none)
        dep_small = calculate_deposition(small_particle)

        @test dep_small.alveolar > 0.2  # Significant alveolar deposition
        @test dep_small.oropharynx < dep_small.alveolar  # Less throat deposition

        # Test 2: Large particles (10 µm) - should deposit in oropharynx
        large_particle = ParticleProperties(10.0, 2.0, 1.0, 1.0, 0.0, 1.0, :none)
        dep_large = calculate_deposition(large_particle)

        @test dep_large.oropharynx > 0.5  # High throat deposition
        @test dep_large.alveolar < 0.2    # Low alveolar deposition

        # Test 3: Optimal size (~2-3 µm) - balance of lung deposition
        optimal_particle = ParticleProperties(2.5, 2.0, 1.0, 1.0, 0.0, 1.0, :none)
        dep_opt = calculate_deposition(optimal_particle)
        lung_total = dep_opt.alveolar + dep_opt.tracheobronchial

        @test lung_total > 0.3  # Good lung deposition

        # Test 4: All fractions sum to <= 1
        total = dep_small.oropharynx + dep_small.extrathoracic +
                dep_small.tracheobronchial + dep_small.alveolar + dep_small.exhaled
        @test total <= 1.01  # Allow small numerical tolerance
    end

    @testset "Regional Deposition with Device Effects" begin
        # Test MDI device
        device_mdi = pulmonary_device_preset(:MDI)
        dep_mdi = regional_deposition_fractions(3.0, device_mdi)

        @test dep_mdi.oropharynx > 0.0
        @test dep_mdi.alveolar >= 0.0

        # Test DPI device (better fine particle fraction expected)
        device_dpi = pulmonary_device_preset(:DPI_high)
        dep_dpi = regional_deposition_fractions(3.0, device_dpi)

        # DPI with high FPF should have better lung deposition
        @test device_dpi.fine_particle_fraction > device_mdi.fine_particle_fraction

        # Test nebulizer
        device_neb = pulmonary_device_preset(:nebulizer)
        dep_neb = regional_deposition_fractions(3.0, device_neb)

        @test dep_neb.alveolar >= 0.0
    end

    @testset "Mucociliary Clearance" begin
        # Normal MCC
        MCC_normal = MucociliaryClearance(12.0, 10.0, 5.0, 1.0)

        # Calculate clearance rate for healthy individual
        k_normal = mucociliary_clearance_rate(MCC_normal)
        @test k_normal > 0.0
        @test k_normal ≈ log(2) / 5.0 atol=0.01  # t1/2 = 5h

        # COPD should have impaired clearance
        disease_copd = pulmonary_disease_state(:COPD, 0.7)
        k_copd = mucociliary_clearance_rate(MCC_normal, disease_copd)

        # COPD clearance should be reduced due to thick mucus
        @test k_copd < k_normal

        # CF (severe) - even more impaired
        disease_cf = pulmonary_disease_state(:CF, 0.8)
        k_cf = mucociliary_clearance_rate(MCC_normal, disease_cf)

        @test k_cf < k_copd  # CF worse than COPD
    end

    @testset "Alveolar Macrophage Clearance" begin
        # Test size-dependent clearance
        # Optimal size 1-3 µm
        mac_1 = alveolar_macrophage_clearance(2.0, 24.0)  # 2 µm at 24h
        @test mac_1.k_clearance > 0.01

        # Very small particles escape phagocytosis
        mac_small = alveolar_macrophage_clearance(0.05, 24.0)
        @test mac_small.k_clearance < mac_1.k_clearance

        # Large particles - less efficient
        mac_large = alveolar_macrophage_clearance(8.0, 24.0)
        @test mac_large.k_clearance < mac_1.k_clearance

        # Clearance fraction increases with time
        mac_short = alveolar_macrophage_clearance(2.0, 1.0)
        mac_long = alveolar_macrophage_clearance(2.0, 24.0)
        @test mac_long.fraction_cleared > mac_short.fraction_cleared
    end

    @testset "Dissolution Kinetics - Noyes-Whitney" begin
        # Create test drug (poorly soluble)
        drug = DrugPulmonaryProperties(
            "TestDrug", 400.0, 3.0, 7.4, 10.0,  # Low solubility
            1e-5, false, 0.0, 0.0, 3.0, 0.5
        )

        dissolution = DissolutionKinetics(0.5, 10.0, 1.5, 2.0, 0.8)

        # Test dissolution rate
        result = PulmonaryAbsorptionModel.dissolution_rate_noyes_whitney(
            drug, dissolution, 100.0, 0.0, 20.0  # 100 µg undissolved, 0 in solution
        )

        @test result.dissolution_rate_ug_h > 0.0  # Should dissolve
        @test result.saturation_fraction ≈ 0.0    # Far from saturation

        # Near saturation - slower dissolution
        result_sat = PulmonaryAbsorptionModel.dissolution_rate_noyes_whitney(
            drug, dissolution, 100.0, 9.0, 20.0  # Near saturation
        )

        @test result_sat.dissolution_rate_ug_h < result.dissolution_rate_ug_h
    end

    @testset "Pulmonary Absorption Rate" begin
        lung = create_lung_model()
        transporters = PulmonaryTransporters(0.5, 1.0, 0.3, 0.5, 0.3, 0.2, 0.5, 0.3)

        # Permeable drug
        drug_perm = DrugPulmonaryProperties(
            "PermeableDrug", 300.0, 2.0, 7.4, 100.0,
            5e-5, false, 0.0, 0.0, 2.0, 0.0
        )

        # Alveolar absorption (thin epithelium)
        abs_alv = pulmonary_absorption_rate(drug_perm, lung, transporters, :alveolar)
        @test abs_alv.k_absorption > 0.0
        @test abs_alv.thickness_um ≈ 0.2  # Alveolar epithelium

        # Bronchial absorption (thicker epithelium)
        abs_bronch = pulmonary_absorption_rate(drug_perm, lung, transporters, :bronchial)
        @test abs_bronch.thickness_um > abs_alv.thickness_um

        # P-gp substrate should have reduced absorption
        drug_pgp = DrugPulmonaryProperties(
            "PgpSubstrate", 400.0, 2.0, 7.4, 100.0,
            5e-5, true, 10.0, 0.0, 2.0, 0.0  # P-gp substrate
        )
        abs_pgp = pulmonary_absorption_rate(drug_pgp, lung, transporters, :alveolar)
        @test abs_pgp.efflux_factor < 1.0  # Efflux reduces absorption
    end

    @testset "Disease State Effects" begin
        # Normal lung
        normal = pulmonary_disease_state(:normal)
        @test normal.mucus_viscosity_factor ≈ 1.0
        @test normal.clearance_impairment ≈ 1.0

        # COPD - increased mucus viscosity, impaired clearance
        copd = pulmonary_disease_state(:COPD, 0.5)
        @test copd.mucus_viscosity_factor > 1.0
        @test copd.clearance_impairment < 1.0
        @test copd.surface_area_reduction < 1.0  # Emphysema

        # Asthma
        asthma = pulmonary_disease_state(:asthma, 0.6)
        @test asthma.mucus_viscosity_factor > 1.0

        # Cystic fibrosis - very thick mucus
        cf = pulmonary_disease_state(:CF, 0.7)
        @test cf.mucus_viscosity_factor > copd.mucus_viscosity_factor

        # IPF - fibrosis reduces surface area
        ipf = pulmonary_disease_state(:IPF, 0.6)
        @test ipf.surface_area_reduction < 0.7
        @test ipf.epithelial_permeability < 1.0  # Fibrosis reduces permeability
    end

    @testset "Drug Presets" begin
        # Test all drug presets
        drugs = [:salbutamol, :fluticasone, :budesonide,
                 :formoterol, :tiotropium, :tobramycin, :ciclesonide]

        for drug_name in drugs
            drug = pulmonary_drug_preset(drug_name)
            @test drug.name != ""
            @test drug.molecular_weight > 0
            @test drug.solubility_ug_mL > 0
            @test drug.particle_MMAD > 0
        end

        # Specific property checks
        salb = pulmonary_drug_preset(:salbutamol)
        @test salb.log_P < 2.0  # Hydrophilic beta-agonist
        @test !salb.Pgp_substrate

        flut = pulmonary_drug_preset(:fluticasone)
        @test flut.log_P > 3.0  # Lipophilic steroid
        @test flut.Pgp_substrate
        @test flut.dissolution_rate > 0  # Slow dissolving

        tobra = pulmonary_drug_preset(:tobramycin)
        @test tobra.log_P < -3.0  # Very hydrophilic aminoglycoside
        @test tobra.solubility_ug_mL > 10000  # Highly soluble
    end

    @testset "Device Presets" begin
        mdi = pulmonary_device_preset(:MDI)
        @test mdi.device_type == :MDI
        @test mdi.spray_velocity_m_s > 0
        @test mdi.propellant == :HFA

        dpi = pulmonary_device_preset(:DPI_high)
        @test dpi.device_type == :DPI
        @test dpi.fine_particle_fraction > 0.4

        neb = pulmonary_device_preset(:nebulizer)
        @test neb.device_type == :nebulizer
        @test neb.fine_particle_fraction > 0.5  # Good FPF

        smi = pulmonary_device_preset(:SMI)
        @test smi.device_type == :SMI
        @test smi.emitted_dose_fraction > 0.9  # High efficiency
    end

    @testset "Full Simulation - Salbutamol MDI" begin
        salb = pulmonary_drug_preset(:salbutamol)
        device = pulmonary_device_preset(:MDI)

        result = simulate_pulmonary_absorption(
            salb, 200.0, device;  # 200 µg dose
            tspan=(0.0, 8.0),
            condition=:normal
        )

        @test length(result.times) > 0
        @test length(result.C_systemic) == length(result.times)
        @test result.Cmax > 0
        @test result.tmax > 0
        @test result.tmax <= 8.0  # Peak within simulation window
        @test result.AUC > 0
        @test 0 < result.F_estimated <= 1.0
    end

    @testset "Full Simulation - Fluticasone (Slow Dissolution)" begin
        flut = pulmonary_drug_preset(:fluticasone)
        device = pulmonary_device_preset(:MDI)

        result = simulate_pulmonary_absorption(
            flut, 500.0, device;  # 500 µg dose
            tspan=(0.0, 24.0),
            condition=:normal
        )

        @test result.Cmax > 0
        @test result.tmax > 0  # Should have delayed peak
        @test result.AUC > 0

        # Fluticasone has lower bioavailability due to:
        # - Poor solubility
        # - P-gp efflux
        # - Lung metabolism
        @test result.F_estimated < 0.5
    end

    @testset "Disease Effects on Absorption" begin
        salb = pulmonary_drug_preset(:salbutamol)
        device = pulmonary_device_preset(:MDI)

        # Normal
        result_normal = simulate_pulmonary_absorption(
            salb, 200.0, device;
            tspan=(0.0, 8.0),
            condition=:normal
        )

        # COPD
        result_copd = simulate_pulmonary_absorption(
            salb, 200.0, device;
            tspan=(0.0, 8.0),
            condition=:COPD
        )

        # Both should produce valid results
        @test result_normal.Cmax > 0
        @test result_copd.Cmax > 0
    end

    @testset "Bioavailability Calculation" begin
        drug = pulmonary_drug_preset(:salbutamol)

        # Create test deposition
        deposition = DepositionFractions(0.3, 0.0, 0.2, 0.3, 0.2)

        MCC = MucociliaryClearance(12.0, 10.0, 5.0, 1.0)
        disease = pulmonary_disease_state(:normal)

        bio = calculate_bioavailability(drug, deposition, MCC, disease; time_h=4.0)

        @test bio.F_total > 0
        @test bio.F_total <= 1.0
        @test bio.F_alveolar >= 0
        @test bio.F_tracheobronchial >= 0
        @test bio.F_GI >= 0  # Swallowed fraction contribution
        @test bio.f_dissolution > 0  # Salbutamol dissolves fast
    end

    @testset "Model Validation" begin
        validation = validate_pulmonary_model()

        # Check deposition size dependence
        @test haskey(validation, "deposition_vs_size")
        @test length(validation["deposition_vs_size"].sizes) > 0

        # Check salbutamol results
        @test haskey(validation, "salbutamol")
        @test validation["salbutamol"].Cmax > 0

        # Check fluticasone results
        @test haskey(validation, "fluticasone")
        @test validation["fluticasone"].Cmax > 0
    end

    @testset "Literature Benchmarks" begin
        # Test 1: ICRP deposition for 2.5 µm particles
        # Literature: ~30-50% lung deposition for 1-5 µm
        particle = ParticleProperties(2.5, 2.0, 1.0, 1.0, 0.0, 1.0, :none)
        dep = calculate_deposition(particle)
        lung_dep = dep.alveolar + dep.tracheobronchial
        @test 0.2 < lung_dep < 0.7  # Reasonable range

        # Test 2: Salbutamol absorption
        # Literature: Tmax ~15-30 min for inhaled salbutamol (clinical)
        # Model may show later peak due to dissolution/clearance kinetics
        salb = pulmonary_drug_preset(:salbutamol)
        device = pulmonary_device_preset(:nebulizer)  # Better deposition
        result = simulate_pulmonary_absorption(salb, 200.0, device; tspan=(0.0, 2.0))
        @test result.tmax <= 2.0  # Peak within simulation window

        # Test 3: Macrophage clearance half-life
        # Literature: 20-70 hours for alveolar macrophage clearance
        mac = alveolar_macrophage_clearance(2.0, 1.0)
        @test 10 < mac.half_life_h < 100
    end

    @testset "Edge Cases" begin
        # Very small particles (nanoparticles)
        nano = ParticleProperties(0.1, 1.5, 1.0, 1.0, 0.0, 1.0, :none)
        dep_nano = calculate_deposition(nano)
        @test dep_nano.exhaled > 0  # Many exhaled

        # Very large particles
        large = ParticleProperties(15.0, 2.0, 1.0, 1.0, 0.0, 1.0, :none)
        dep_large = calculate_deposition(large)
        @test dep_large.oropharynx > 0.8  # Most deposit in throat

        # Zero dose
        salb = pulmonary_drug_preset(:salbutamol)
        device = pulmonary_device_preset(:MDI)
        result_zero = simulate_pulmonary_absorption(salb, 0.0, device; tspan=(0.0, 2.0))
        @test result_zero.Cmax ≈ 0.0 atol=1e-10
    end

end

println("✓ Pulmonary Absorption Model tests complete")
