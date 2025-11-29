# =============================================================================
# Tests for GLP-1 Agonist & Bariatric Surgery Model
# =============================================================================
# Darwin PBPK Platform - Publication-Ready Validation
#
# Tests cover:
# 1. GLP-1 agonist effects on gastric emptying
# 2. Time-dependent GLP-1 effects
# 3. Bariatric surgery physiology changes
# 4. Drug-specific bioavailability modifications
# 5. Combined GLP-1 + surgery scenarios
# 6. Simulation accuracy
# =============================================================================

using Test
using DarwinPBPK
using DarwinPBPK.MedLang
using DarwinPBPK.MedLang.GLP1BariatricModel

@testset "GLP-1 Agonist & Bariatric Surgery Model" begin

    @testset "GLP-1 Receptor and Agonist Structures" begin
        # Test default receptor
        receptor = GLP1BariatricModel.default_glp1_receptor()
        @test receptor.ec50_pM > 0
        @test receptor.ileum_expression > receptor.stomach_expression  # L-cells in ileum

        # Test GLP-1 agonist presets
        sema_sc = glp1_agonist_preset(:semaglutide_sc)
        @test sema_sc.name == "Semaglutide (SC)"
        @test sema_sc.half_life_h ≈ 168.0  # ~1 week
        @test sema_sc.dosing_frequency == :weekly

        lira = glp1_agonist_preset(:liraglutide)
        @test lira.name == "Liraglutide"
        @test lira.half_life_h ≈ 13.0
        @test lira.dosing_frequency == :daily

        tirz = glp1_agonist_preset(:tirzepatide)
        @test tirz.name == "Tirzepatide"
        @test tirz.relative_potency > sema_sc.relative_potency  # Dual agonist
    end

    @testset "GLP-1 Effect Calculations" begin
        sema = glp1_agonist_preset(:semaglutide_sc)
        receptor = GLP1BariatricModel.default_glp1_receptor()

        # Low concentration - moderate effect (GLP-1 agonists are potent)
        effect_low = calculate_glp1_effect(sema, 10.0, receptor)
        @test effect_low.gastric_emptying_t50_factor > 1.0
        @test effect_low.gastric_emptying_t50_factor < 2.5  # Even low conc has effect

        # High concentration - strong effect
        effect_high = calculate_glp1_effect(sema, 500.0, receptor)
        @test effect_high.gastric_emptying_t50_factor > effect_low.gastric_emptying_t50_factor
        @test effect_high.gastric_emptying_t50_factor > 2.0  # Significant delay

        # Ka reduction should mirror T50 increase
        @test effect_high.ka_reduction_factor < effect_low.ka_reduction_factor

        # Bile secretion reduced
        @test effect_high.bile_secretion_factor < 1.0
    end

    @testset "Gastric Emptying Delay - Time Course" begin
        sema = glp1_agonist_preset(:semaglutide_sc)

        # Before effect onset - no delay
        delay_0h = gastric_emptying_delay(sema, 1.0, 0.0)
        @test delay_0h ≈ 1.0  # No effect yet

        # At Tmax - near maximum effect
        delay_peak = gastric_emptying_delay(sema, 1.0, sema.tmax_h)
        @test delay_peak > 1.5

        # After elimination - reduced effect
        delay_late = gastric_emptying_delay(sema, 1.0, 200.0)
        @test delay_late < delay_peak
        @test delay_late > 1.0  # Some residual effect due to long t1/2

        # Dose dependence
        delay_low_dose = gastric_emptying_delay(sema, 0.5, 24.0)
        delay_high_dose = gastric_emptying_delay(sema, 2.4, 24.0)
        @test delay_high_dose > delay_low_dose
    end

    @testset "Transit Time Modifications" begin
        sema = glp1_agonist_preset(:semaglutide_sc)
        effect = calculate_glp1_effect(sema, 200.0)

        # GLP-1 alone
        transit_glp1 = transit_time_modification(effect, nothing)
        @test transit_glp1.gastric_t50 > 15.0  # Delayed from normal ~15 min
        @test transit_glp1.jejunum > 90.0  # Slowed from normal ~90 min

        # No modifications
        transit_normal = transit_time_modification(nothing, nothing)
        @test transit_normal.gastric_t50 ≈ 15.0
        @test transit_normal.jejunum ≈ 90.0
    end

    @testset "Bariatric Surgery Presets" begin
        # Sleeve gastrectomy
        vsg = surgery_preset(SLEEVE_GASTRECTOMY, 12.0)
        @test vsg.surgery_type == SLEEVE_GASTRECTOMY
        @test vsg.gastric_volume_mL < 250.0  # Reduced stomach
        @test !vsg.duodenum_bypassed  # Duodenum intact
        @test vsg.gastric_emptying_rate < 1.0  # Faster emptying

        # Roux-en-Y bypass
        rygb = surgery_preset(ROUX_EN_Y_BYPASS, 12.0)
        @test rygb.surgery_type == ROUX_EN_Y_BYPASS
        @test rygb.gastric_volume_mL < 50.0  # Tiny pouch
        @test rygb.duodenum_bypassed  # Key feature!
        @test rygb.gastric_emptying_rate < vsg.gastric_emptying_rate  # Even faster
        @test rygb.bile_mixing_delay_min > 0  # Delayed bile contact
        @test rygb.gut_hormone_amplification > vsg.gut_hormone_amplification  # Enhanced GLP-1

        # BPD
        bpd = surgery_preset(BILIOPANCREATIC_DIVERSION, 12.0)
        @test bpd.duodenum_bypassed
        @test bpd.common_channel_cm < rygb.common_channel_cm  # Shorter = more malabsorption

        # Gastric banding
        band = surgery_preset(GASTRIC_BANDING, 12.0)
        @test !band.duodenum_bypassed
        @test band.gastric_emptying_rate < 1.0  # Restricted
    end

    @testset "Post-Surgery Physiology Calculations" begin
        # RYGB at different time points
        rygb_acute = surgery_preset(ROUX_EN_Y_BYPASS, 1.0)  # 1 month
        physio_acute = GLP1BariatricModel.calculate_surgery_physiology(rygb_acute)

        rygb_chronic = surgery_preset(ROUX_EN_Y_BYPASS, 24.0)  # 2 years
        physio_chronic = GLP1BariatricModel.calculate_surgery_physiology(rygb_chronic)

        # Acute inflammation should be higher early
        @test physio_acute.acute_inflammation > physio_chronic.acute_inflammation

        # Adaptation should increase over time
        @test physio_chronic.gut_adaptation_factor > physio_acute.gut_adaptation_factor

        # Intestinal hypertrophy develops over time
        @test physio_chronic.intestinal_hypertrophy >= physio_acute.intestinal_hypertrophy

        # pH changes
        @test physio_acute.gastric_pH > 2.0  # Higher than normal
    end

    @testset "Dissolution Rate Modifications" begin
        # Acid drug (e.g., ibuprofen, pKa ~4.5)
        acid_normal = dissolution_rate_modifier(4.5, :acid, 2.0, 2.0)
        acid_high_ph = dissolution_rate_modifier(4.5, :acid, 5.0, 2.0)
        @test acid_high_ph > acid_normal  # Acids dissolve better at higher pH

        # Base drug (e.g., metformin, pKa ~12)
        base_normal = dissolution_rate_modifier(8.0, :base, 2.0, 2.0)
        base_high_ph = dissolution_rate_modifier(8.0, :base, 5.0, 2.0)
        @test base_high_ph < base_normal  # Bases dissolve better at lower pH

        # Neutral drug - no pH effect
        neutral = dissolution_rate_modifier(nothing, :neutral, 5.0, 2.0)
        @test neutral ≈ 1.0
    end

    @testset "Bioavailability Changes - Literature Validation" begin
        # Metformin increased after RYGB
        met_rygb = bioavailability_change(:metformin, ROUX_EN_Y_BYPASS)
        @test met_rygb.f_factor > 1.0
        @test met_rygb.evidence == "literature"

        # Levothyroxine decreased (pH-dependent)
        levo_rygb = bioavailability_change(:levothyroxine, ROUX_EN_Y_BYPASS)
        @test levo_rygb.f_factor < 1.0
        @test occursin("acid", levo_rygb.mechanism)

        # Tacrolimus highly increased
        tacro_rygb = bioavailability_change(:tacrolimus, ROUX_EN_Y_BYPASS)
        @test tacro_rygb.f_factor >= 1.5

        # Cyclosporine decreased (bile-dependent)
        cyclo_rygb = bioavailability_change(:cyclosporine, ROUX_EN_Y_BYPASS)
        @test cyclo_rygb.f_factor < 1.0
        @test occursin("bile", cyclo_rygb.mechanism)

        # Unknown drug
        unknown = bioavailability_change(:unknown_drug, ROUX_EN_Y_BYPASS)
        @test unknown.f_factor ≈ 1.0
        @test unknown.evidence == "assumed"
    end

    @testset "Fraction Absorbed - Bariatric" begin
        rygb = surgery_preset(ROUX_EN_Y_BYPASS, 12.0)
        physio = GLP1BariatricModel.calculate_surgery_physiology(rygb)

        # Highly soluble, permeable drug
        fa_good = calculate_fa_bariatric(
            10.0,      # High solubility
            nothing,   # No pKa effect
            :neutral,
            1e-4,      # Good permeability
            rygb,
            physio
        )
        @test fa_good.fa > 0.5  # Still reasonable absorption
        @test fa_good.bypass_loss > 0  # Some loss from duodenal bypass

        # Poorly soluble drug
        fa_poor = calculate_fa_bariatric(
            0.01,      # Very poor solubility
            nothing,
            :neutral,
            1e-4,
            rygb,
            physio
        )
        @test fa_poor.fa < fa_good.fa  # Reduced
        @test fa_poor.solubility_limitation < 1.0
    end

    @testset "Combined GLP-1 + Surgery Effects" begin
        sema = glp1_agonist_preset(:semaglutide_sc)
        rygb = surgery_preset(ROUX_EN_Y_BYPASS, 12.0)

        # Combined physiology
        gi_combined = create_modified_gi_params(
            glp1 = sema,
            glp1_dose_mg = 1.0,
            time_since_glp1_h = 24.0,
            surgery = rygb
        )

        # Duodenum should be bypassed (RYGB)
        @test !gi_combined.duodenum_functional

        # Gastric emptying: RYGB speeds up, but GLP-1 slows down
        # Net effect depends on relative magnitudes
        @test gi_combined.gastric_emptying_t50 > 0  # Valid value

        # Bile mixing delayed
        @test gi_combined.bile_delay_min > 0

        # GLP-1 alone
        gi_glp1_only = create_modified_gi_params(
            glp1 = sema,
            glp1_dose_mg = 1.0,
            time_since_glp1_h = 24.0
        )
        @test gi_glp1_only.duodenum_functional  # Normal anatomy
        @test gi_glp1_only.gastric_emptying_t50 > 15.0  # Delayed

        # Surgery alone
        gi_surgery_only = create_modified_gi_params(surgery = rygb)
        @test !gi_surgery_only.duodenum_functional
    end

    @testset "Oral Simulation with GLP-1" begin
        result = simulate_oral_with_glp1(
            0.05,      # ka (1/min)
            0.1,       # dissolution rate
            nothing,   # pKa
            :neutral,
            100.0,     # dose mg
            glp1_agonist_preset(:semaglutide_sc),
            1.0,       # GLP-1 dose
            24.0;      # time since GLP-1 dose
            tspan = (0.0, 12.0)
        )

        @test length(result.times) > 0
        @test length(result.A_systemic) == length(result.times)
        @test result.delay_factor > 1.0  # GLP-1 caused delay
        @test result.F > 0  # Some absorption occurred
        @test result.F <= 1.0
    end

    @testset "Oral Simulation Post-Bariatric" begin
        rygb = surgery_preset(ROUX_EN_Y_BYPASS, 12.0)

        result = simulate_oral_post_bariatric(
            0.05,      # ka
            0.1,       # dissolution
            nothing,   # pKa
            :neutral,
            100.0,     # dose
            rygb;
            tspan = (0.0, 12.0)
        )

        @test length(result.times) > 0
        @test result.surgery == ROUX_EN_Y_BYPASS
        @test result.F > 0
        @test result.tmax > 0  # Should have a Tmax

        # VSG simulation
        vsg = surgery_preset(SLEEVE_GASTRECTOMY, 12.0)
        result_vsg = simulate_oral_post_bariatric(
            0.05, 0.1, nothing, :neutral, 100.0, vsg;
            tspan = (0.0, 12.0)
        )

        @test result_vsg.surgery == SLEEVE_GASTRECTOMY
    end

    @testset "Model Validation" begin
        validation = validate_glp1_model()

        # Semaglutide delay validation
        @test haskey(validation, "semaglutide_delay")
        @test validation["semaglutide_delay"].at_4h > 1.0
        @test validation["semaglutide_delay"].at_24h > 1.0

        # RYGB physiology
        @test haskey(validation, "rygb_physiology")
        @test validation["rygb_physiology"].gastric_pH > 2.0
        @test validation["rygb_physiology"].effective_area < 1.0

        # Drug-specific changes
        @test haskey(validation, "drug_specific")
        @test validation["drug_specific"].metformin.f_factor > 1.0
        @test validation["drug_specific"].levothyroxine.f_factor < 1.0

        # Combined effects
        @test haskey(validation, "combined_effects")
    end

    @testset "All GLP-1 Agonist Presets" begin
        agonists = [:semaglutide_oral, :semaglutide_sc, :tirzepatide,
                    :liraglutide, :dulaglutide, :exenatide_er]

        for name in agonists
            agonist = glp1_agonist_preset(name)
            @test agonist.name != ""
            @test agonist.half_life_h > 0
            @test agonist.gastric_emptying_delay_factor > 1.0
            @test agonist.relative_potency > 0
        end

        # Tirzepatide should have highest potency (dual agonist)
        tirz = glp1_agonist_preset(:tirzepatide)
        sema = glp1_agonist_preset(:semaglutide_sc)
        @test tirz.relative_potency > sema.relative_potency
    end

    @testset "Edge Cases" begin
        # Zero GLP-1 dose
        gi_zero = create_modified_gi_params(
            glp1 = glp1_agonist_preset(:semaglutide_sc),
            glp1_dose_mg = 0.0,
            time_since_glp1_h = 24.0
        )
        @test gi_zero.gastric_emptying_t50 ≈ 15.0  # Normal

        # No surgery (control)
        no_surg = surgery_preset(NO_SURGERY, 0.0)
        @test no_surg.gastric_volume_mL ≈ 250.0
        @test no_surg.gastric_emptying_rate ≈ 1.0
        @test !no_surg.duodenum_bypassed

        # Very early post-surgery
        rygb_early = surgery_preset(ROUX_EN_Y_BYPASS, 0.5)  # 2 weeks
        physio_early = GLP1BariatricModel.calculate_surgery_physiology(rygb_early)
        @test physio_early.acute_inflammation > 0.5  # Still inflamed
    end

    @testset "Clinical Scenarios" begin
        # Scenario 1: Diabetic patient on semaglutide, needs acetaminophen
        sema = glp1_agonist_preset(:semaglutide_sc)
        delay = gastric_emptying_delay(sema, 2.4, 72.0)  # 3 days after weekly dose
        # Acetaminophen Tmax may be delayed
        @test delay > 1.0

        # Scenario 2: Post-RYGB patient on metformin
        met_change = bioavailability_change(:metformin, ROUX_EN_Y_BYPASS)
        # May need dose reduction due to increased absorption
        @test met_change.f_factor > 1.0

        # Scenario 3: Post-RYGB needing levothyroxine
        levo_change = bioavailability_change(:levothyroxine, ROUX_EN_Y_BYPASS)
        # May need dose increase
        @test levo_change.f_factor < 1.0

        # Scenario 4: Post-RYGB patient also on tirzepatide
        tirz = glp1_agonist_preset(:tirzepatide)
        rygb = surgery_preset(ROUX_EN_Y_BYPASS, 18.0)
        gi = create_modified_gi_params(
            glp1 = tirz,
            glp1_dose_mg = 15.0,
            time_since_glp1_h = 48.0,
            surgery = rygb
        )
        # GLP-1 may counteract rapid emptying post-RYGB
        @test gi.gastric_emptying_t50 > 0
    end

end

println("GLP-1 Agonist & Bariatric Surgery Model tests complete")
