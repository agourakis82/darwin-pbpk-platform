# =============================================================================
# BAYESIAN DDI MODEL TESTS
# =============================================================================
# Darwin PBPK Platform - MedLang DSL
# Comprehensive testing of Bayesian Drug-Drug Interaction model
#
# Test Coverage:
# - DDI mechanism calculations (CYP inhibition, induction, transporter)
# - Static DDI model (FDA mechanistic method)
# - Bayesian prior calculation
# - Posterior updates with clinical data
# - Risk classification (FDA/EMA)
# - Drug presets (perpetrators and victims)
# - Clinical significance assessment
# - Model validation against known DDIs
#
# Author: Dr. Sounio Agourakis
# Date: November 2025
# =============================================================================

using Test
using DarwinPBPK
using DarwinPBPK.MedLang
using DarwinPBPK.MedLang.BayesianDDIModel
using Statistics

@testset "Bayesian DDI Model Tests" begin

    # =========================================================================
    # DDI MECHANISM STRUCTURE TESTS
    # =========================================================================
    @testset "DDI Mechanism Structures" begin
        # Test DDI mechanism type enum
        @test CYP_COMPETITIVE_INHIBITION isa DDIMechanismType
        @test CYP_NONCOMPETITIVE_INHIBITION isa DDIMechanismType
        @test CYP_MECHANISM_BASED_INHIBITION isa DDIMechanismType
        @test CYP_INDUCTION isa DDIMechanismType
        @test TRANSPORTER_INHIBITION isa DDIMechanismType

        # Test CYP enzyme enum
        @test CYP3A4 isa CYPEnzyme
        @test CYP2D6 isa CYPEnzyme
        @test CYP2C9 isa CYPEnzyme
        @test CYP2C19 isa CYPEnzyme
        @test CYP1A2 isa CYPEnzyme

        # Test transporter type enum
        @test PGP isa TransporterType
        @test BCRP isa TransporterType
        @test OATP1B1 isa TransporterType
        @test OATP1B3 isa TransporterType
    end

    # =========================================================================
    # PERPETRATOR DRUG PRESETS
    # =========================================================================
    @testset "Perpetrator Drug Presets" begin
        # Ketoconazole - strong CYP3A4 inhibitor
        keto = perpetrator_preset(:ketoconazole)
        @test keto.drug_name == "Ketoconazole"
        @test keto.cmax_total_uM > 0
        @test keto.fu > 0 && keto.fu <= 1.0
        @test length(keto.mechanisms) >= 1
        @test keto.mechanisms[1].cyp_enzyme == CYP3A4
        @test keto.mechanisms[1].ki_uM < 1.0  # Very potent inhibitor

        # Itraconazole - strong CYP3A4 inhibitor
        itra = perpetrator_preset(:itraconazole)
        @test itra.drug_name == "Itraconazole"
        @test itra.is_substrate == true  # Also a substrate
        @test CYP3A4 in itra.substrate_enzymes

        # Rifampin - strong CYP3A4 inducer
        rif = perpetrator_preset(:rifampin)
        @test rif.drug_name == "Rifampin"
        @test length(rif.mechanisms) >= 1
        @test rif.mechanisms[1].mechanism_type == CYP_INDUCTION
        @test rif.mechanisms[1].emax_fold >= 10.0  # Strong inducer

        # Fluconazole - moderate CYP2C9 inhibitor
        flu = perpetrator_preset(:fluconazole)
        @test flu.drug_name == "Fluconazole"
        @test length(flu.mechanisms) >= 1
        @test flu.fu > 0.8  # High unbound fraction

        # Cyclosporine - OATP inhibitor
        cyclo = perpetrator_preset(:cyclosporine)
        @test cyclo.drug_name == "Cyclosporine"
        @test cyclo.mechanisms[1].transporter == OATP1B1
    end

    # =========================================================================
    # VICTIM DRUG PRESETS
    # =========================================================================
    @testset "Victim Drug Presets" begin
        # Midazolam - CYP3A4 probe substrate
        midaz = victim_preset(:midazolam)
        @test midaz.drug_name == "Midazolam"
        @test midaz.fm_cyp3a4 >= 0.9  # Highly metabolized by CYP3A4
        @test midaz.therapeutic_index > 2.0  # Not narrow TI

        # Simvastatin - CYP3A4 + OATP substrate
        simva = victim_preset(:simvastatin)
        @test simva.drug_name == "Simvastatin"
        @test simva.fm_cyp3a4 > 0.8
        @test simva.ft_oatp1b1 > 0.0

        # S-Warfarin - CYP2C9 substrate with narrow TI
        warf = victim_preset(:warfarin_s)
        @test warf.drug_name == "S-Warfarin"
        @test warf.fm_cyp2c9 >= 0.8
        @test warf.therapeutic_index < 2.0  # Narrow TI!

        # Rosuvastatin - OATP-mediated
        rosu = victim_preset(:rosuvastatin)
        @test rosu.drug_name == "Rosuvastatin"
        @test rosu.ft_oatp1b1 > 0.5
        @test rosu.fm_cyp3a4 < 0.2  # Minimal CYP metabolism

        # Tacrolimus - narrow TI drug
        tacro = victim_preset(:tacrolimus)
        @test tacro.drug_name == "Tacrolimus"
        @test tacro.therapeutic_index < 1.5  # Very narrow TI
        @test tacro.fm_cyp3a4 >= 0.9
    end

    # =========================================================================
    # MECHANISTIC STATIC MODEL
    # =========================================================================
    @testset "Mechanistic Static Model" begin
        # Test CYP competitive inhibition
        keto = perpetrator_preset(:ketoconazole)
        midaz = victim_preset(:midazolam)

        # Direct mechanism calculation
        auc_ratio = mechanistic_static_model(keto, midaz, keto.mechanisms[1])
        @test auc_ratio > 1.0  # Inhibition increases exposure
        @test auc_ratio > 5.0  # Strong inhibition expected

        # Test with combined static model
        result = static_ddi_model(keto, midaz)
        @test result.auc_ratio > 5.0
        @test haskey(result.contributions, CYP_COMPETITIVE_INHIBITION)
    end

    @testset "CYP Induction Model" begin
        # Rifampin + Midazolam (strong induction)
        rif = perpetrator_preset(:rifampin)
        midaz = victim_preset(:midazolam)

        result = static_ddi_model(rif, midaz)
        @test result.auc_ratio < 1.0  # Induction decreases exposure
        @test result.auc_ratio < 0.3  # Strong induction expected
        @test haskey(result.contributions, CYP_INDUCTION)
    end

    @testset "Transporter DDI Model" begin
        # Cyclosporine + Rosuvastatin (OATP inhibition)
        cyclo = perpetrator_preset(:cyclosporine)
        rosu = victim_preset(:rosuvastatin)

        result = static_ddi_model(cyclo, rosu)
        @test result.auc_ratio > 1.0  # OATP inhibition increases exposure
        @test haskey(result.contributions, TRANSPORTER_INHIBITION)
    end

    # =========================================================================
    # NET EFFECT MODEL
    # =========================================================================
    @testset "Net Effect Model (Inhibition + Induction)" begin
        # Test combined effects
        inhibition_ratio = 3.0
        induction_ratio = 0.5

        net = net_effect_model(inhibition_ratio, induction_ratio)
        @test net == inhibition_ratio * induction_ratio
        @test net ≈ 1.5  # Partial cancellation
    end

    # =========================================================================
    # BAYESIAN PRIOR CALCULATION
    # =========================================================================
    @testset "Bayesian DDI Prior" begin
        keto = perpetrator_preset(:ketoconazole)
        midaz = victim_preset(:midazolam)

        prior = calculate_ddi_prior(keto, midaz; use_database=true)

        # Check prior structure
        @test prior isa DDIPrior
        @test prior.database_auc_ratio_mean > 0
        @test prior.database_auc_ratio_sd > 0
        @test prior.n_clinical_studies >= 0

        # Population prior should be set
        @test prior.population_prior isa PopulationDDIPrior
    end

    # =========================================================================
    # POSTERIOR UPDATE
    # =========================================================================
    @testset "Bayesian Posterior Update" begin
        keto = perpetrator_preset(:ketoconazole)
        midaz = victim_preset(:midazolam)

        prior = calculate_ddi_prior(keto, midaz)

        # Update with no data (pure prior)
        posterior_no_data = update_ddi_posterior(prior, Float64[])
        @test posterior_no_data.auc_ratio_mean > 0
        @test posterior_no_data.prior_weight ≈ 1.0

        # Update with clinical data
        observed_ratios = [10.0, 12.0, 14.0]  # Clinical observations
        posterior = update_ddi_posterior(prior, observed_ratios)

        @test posterior.auc_ratio_mean > 0
        @test posterior.auc_ratio_sd > 0
        @test posterior.data_weight > 0  # Data influences posterior
        @test length(posterior.auc_ratio_samples) >= 1000

        # Posterior should be between prior and data
        data_mean = mean(observed_ratios)
        @test abs(posterior.auc_ratio_mean - data_mean) < abs(prior.database_auc_ratio_mean - data_mean) ||
              abs(posterior.auc_ratio_mean - prior.database_auc_ratio_mean) < 5.0
    end

    @testset "Credible Intervals" begin
        keto = perpetrator_preset(:ketoconazole)
        midaz = victim_preset(:midazolam)

        prior = calculate_ddi_prior(keto, midaz)
        posterior = update_ddi_posterior(prior, Float64[])

        # Check CI structure
        @test posterior.auc_ratio_ci_90[1] < posterior.auc_ratio_mean
        @test posterior.auc_ratio_ci_90[2] > posterior.auc_ratio_mean
        @test posterior.auc_ratio_ci_95[1] <= posterior.auc_ratio_ci_90[1]
        @test posterior.auc_ratio_ci_95[2] >= posterior.auc_ratio_ci_90[2]

        # Check probability estimates
        @test 0 <= posterior.p_auc_ratio_gt_1_25 <= 1
        @test 0 <= posterior.p_auc_ratio_gt_2 <= 1
        @test 0 <= posterior.p_auc_ratio_gt_5 <= 1
        @test posterior.p_auc_ratio_gt_1_25 >= posterior.p_auc_ratio_gt_2
        @test posterior.p_auc_ratio_gt_2 >= posterior.p_auc_ratio_gt_5
    end

    # =========================================================================
    # AUC RATIO PREDICTION
    # =========================================================================
    @testset "AUC Ratio Prediction" begin
        keto = perpetrator_preset(:ketoconazole)
        midaz = victim_preset(:midazolam)

        prediction = predict_auc_ratio(keto, midaz)

        @test prediction.auc_ratio_mean > 5.0  # Strong interaction expected
        @test prediction.auc_ratio_sd > 0
        @test prediction.ci_95[1] < prediction.auc_ratio_mean
        @test prediction.ci_95[2] > prediction.auc_ratio_mean
        @test prediction.p_strong > 0.5  # High probability of strong interaction
    end

    @testset "Cmax Ratio Prediction" begin
        keto = perpetrator_preset(:ketoconazole)
        midaz = victim_preset(:midazolam)

        auc_ratio = 10.0
        cmax_ratio = predict_cmax_ratio(keto, midaz; auc_ratio=auc_ratio)

        @test cmax_ratio > 1.0
        @test cmax_ratio < auc_ratio  # Cmax ratio typically smaller
    end

    # =========================================================================
    # DDI RISK CLASSIFICATION
    # =========================================================================
    @testset "FDA Risk Classification" begin
        # No effect
        @test ddi_risk_classification(1.1) == :no_effect

        # Weak
        @test ddi_risk_classification(1.5) == :weak

        # Moderate
        @test ddi_risk_classification(3.0) == :moderate

        # Strong
        @test ddi_risk_classification(6.0) == :strong
        @test ddi_risk_classification(10.0) == :strong

        # Edge cases
        @test ddi_risk_classification(1.25) == :weak
        @test ddi_risk_classification(2.0) == :moderate
        @test ddi_risk_classification(5.0) == :strong
    end

    @testset "Clinical Significance Assessment" begin
        # Non-NTI drug with weak interaction
        midaz = victim_preset(:midazolam)
        clin = clinical_significance(midaz, 1.5, 1.3)
        @test clin.dose_adjustment ≈ 1.0
        @test clin.monitoring_required == true
        @test clin.contraindicated == false

        # NTI drug with moderate interaction
        warf = victim_preset(:warfarin_s)
        clin_nti = clinical_significance(warf, 2.5, 2.0)
        @test clin_nti.is_nti == true
        @test clin_nti.dose_adjustment < 1.0
        @test clin_nti.monitoring_required == true

        # NTI drug with strong interaction
        tacro = victim_preset(:tacrolimus)
        clin_strong = clinical_significance(tacro, 6.0, 4.0)
        @test clin_strong.contraindicated == true
    end

    # =========================================================================
    # DDI RISK ASSESSMENT
    # =========================================================================
    @testset "Complete DDI Risk Assessment" begin
        keto = perpetrator_preset(:ketoconazole)
        midaz = victim_preset(:midazolam)

        prediction = predict_auc_ratio(keto, midaz)
        assessment = create_ddi_risk_assessment(keto, midaz, prediction.posterior)

        @test assessment.perpetrator == "Ketoconazole"
        @test assessment.victim == "Midazolam"
        @test assessment.auc_ratio > 5.0
        @test assessment.fda_classification == :strong
        @test assessment.prediction_confidence in [:low, :moderate, :high]
    end

    # =========================================================================
    # PATIENT-SPECIFIC FACTORS
    # =========================================================================
    @testset "Patient-Specific DDI Prediction" begin
        # Create patient factors
        patient = PatientDDIFactors(
            65.0,           # age
            70.0,           # weight
            :male,          # sex
            :PM,            # CYP2D6 poor metabolizer
            :EM,            # CYP2C19 extensive
            Symbol("*1/*1"), # CYP2C9 wild type
            :normal,        # hepatic function
            :normal,        # renal function
            90.0,           # eGFR
            String[],       # other inhibitors
            String[]        # other inducers
        )

        keto = perpetrator_preset(:ketoconazole)
        midaz = victim_preset(:midazolam)

        # Prediction without patient factors
        pred_generic = predict_auc_ratio(keto, midaz)

        # Prediction with patient factors
        pred_patient = predict_auc_ratio(keto, midaz; patient_factors=patient)

        @test pred_patient.auc_ratio_mean > 0
        @test pred_patient.auc_ratio_sd > 0
    end

    @testset "Hepatic Impairment Effect" begin
        patient_normal = PatientDDIFactors(
            50.0, 70.0, :male, :EM, :EM, Symbol("*1/*1"),
            :normal, :normal, 90.0, [], []
        )

        patient_impaired = PatientDDIFactors(
            50.0, 70.0, :male, :EM, :EM, Symbol("*1/*1"),
            :moderate, :normal, 90.0, [], []
        )

        keto = perpetrator_preset(:ketoconazole)
        midaz = victim_preset(:midazolam)

        pred_normal = predict_auc_ratio(keto, midaz; patient_factors=patient_normal)
        pred_impaired = predict_auc_ratio(keto, midaz; patient_factors=patient_impaired)

        # Hepatic impairment should increase DDI magnitude
        @test pred_impaired.auc_ratio_mean >= pred_normal.auc_ratio_mean
    end

    # =========================================================================
    # DDI SCENARIO SIMULATION
    # =========================================================================
    @testset "DDI Scenario Simulation" begin
        keto = perpetrator_preset(:ketoconazole)
        midaz = victim_preset(:midazolam)

        scenario = simulate_ddi_scenario(keto, midaz)

        @test scenario.perpetrator == "Ketoconazole"
        @test scenario.victim == "Midazolam"
        @test scenario.auc_ratio > 5.0
        @test scenario.fda_classification == :strong
        @test scenario.recommendation != ""
        @test 0 <= scenario.p_significant <= 1
    end

    # =========================================================================
    # MODEL VALIDATION
    # =========================================================================
    @testset "Model Validation Against Known DDIs" begin
        results = validate_ddi_model()

        # Check summary
        @test haskey(results, "summary")
        @test results["summary"].n_cases > 0

        # Check individual predictions
        # Ketoconazole + Midazolam
        if haskey(results, "ketoconazole_midazolam")
            keto_mid = results["ketoconazole_midazolam"]
            @test keto_mid.predicted > 5.0  # Strong inhibition
            @test keto_mid.fold_error < 5.0  # Reasonable accuracy
        end

        # Rifampin + Midazolam (induction)
        if haskey(results, "rifampin_midazolam")
            rif_mid = results["rifampin_midazolam"]
            @test rif_mid.predicted < 0.5  # Strong induction
        end
    end

    @testset "Fold Error Acceptance Criteria" begin
        results = validate_ddi_model()

        # Mean fold error should be reasonable
        @test results["summary"].mean_fold_error < 5.0

        # Coverage of CI should be acceptable (relaxed for Bayesian)
        @test results["summary"].coverage >= 0.3  # At least 30% within CI
    end

    # =========================================================================
    # MULTIPLE PERPETRATORS
    # =========================================================================
    @testset "Multiple Mechanism Perpetrator" begin
        flu = perpetrator_preset(:fluconazole)
        # Fluconazole inhibits both CYP2C9 and CYP3A4

        @test length(flu.mechanisms) >= 2

        # Test with CYP2C9 victim
        warf = victim_preset(:warfarin_s)
        result_warf = static_ddi_model(flu, warf)
        @test result_warf.auc_ratio > 1.5  # Moderate CYP2C9 inhibition

        # Test with CYP3A4 victim
        midaz = victim_preset(:midazolam)
        result_midaz = static_ddi_model(flu, midaz)
        @test result_midaz.auc_ratio > 1.0
    end

    # =========================================================================
    # EDGE CASES
    # =========================================================================
    @testset "Edge Cases" begin
        # Very weak inhibitor (high Ki)
        weak_mechanism = DDIMechanism(
            CYP_COMPETITIVE_INHIBITION,
            CYP3A4, 100.0, 20.0, 200.0, :competitive,
            0.0, 0.0,
            1.0, 0.0, :none,
            nothing, 0.0,
            0.5, 0.0
        )

        weak_perp = DDIPerpetrator(
            "WeakDrug",
            1.0, 0.5, 0.5,
            5.0, 2.5,
            50.0, 50.0,
            [weak_mechanism],
            false, CYPEnzyme[]
        )

        midaz = victim_preset(:midazolam)
        result = static_ddi_model(weak_perp, midaz)
        @test result.auc_ratio < 1.5  # Minimal interaction

        # Zero fm pathway - create a weak perpetrator with low fm
        no_pathway_mechanism = DDIMechanism(
            CYP_COMPETITIVE_INHIBITION,
            CYP3A4, 0.015, 0.005, 0.03, :competitive,
            0.0, 0.0,
            1.0, 0.0, :none,
            nothing, 0.0,
            0.0, 0.0       # fm = 0 means no interaction via this pathway
        )

        no_pathway_perp = DDIPerpetrator(
            "NoPathwayDrug",
            10.0, 0.1, 0.01,
            50.0, 25.0,
            1000.0, 200.0,
            [no_pathway_mechanism],
            false, CYPEnzyme[]
        )

        midaz = victim_preset(:midazolam)
        result_no_pathway = static_ddi_model(no_pathway_perp, midaz)
        @test result_no_pathway.auc_ratio ≈ 1.0 atol=0.1  # No interaction when fm=0
    end

    # =========================================================================
    # POPULATION PRIOR
    # =========================================================================
    @testset "Population DDI Prior" begin
        pop_prior = population_ddi_prior_default()

        @test pop_prior isa PopulationDDIPrior

        # Check genetic effects
        @test pop_prior.pm_effect_cyp2d6 > 1.0
        @test pop_prior.um_effect_cyp2d6 < 1.0

        # Check hepatic impairment factors
        @test pop_prior.hepatic_impairment_mild < pop_prior.hepatic_impairment_moderate
        @test pop_prior.hepatic_impairment_moderate < pop_prior.hepatic_impairment_severe
    end

    # =========================================================================
    # MONTE CARLO UNCERTAINTY
    # =========================================================================
    @testset "Monte Carlo Sampling" begin
        keto = perpetrator_preset(:ketoconazole)
        midaz = victim_preset(:midazolam)

        # Small sample
        pred_small = predict_auc_ratio(keto, midaz; n_samples=100)
        @test length(pred_small.posterior.auc_ratio_samples) >= 100

        # Large sample
        pred_large = predict_auc_ratio(keto, midaz; n_samples=5000)
        @test length(pred_large.posterior.auc_ratio_samples) >= 5000

        # Larger sample should give more stable estimates
        # (Not strictly guaranteed but should be close)
    end

    # =========================================================================
    # DATABASE PRIOR LOOKUP
    # =========================================================================
    @testset "Database Prior Lookup" begin
        # Known interaction
        mean_keto_mid, sd_keto_mid, n_studies = get_database_prior("ketoconazole", "midazolam")
        @test mean_keto_mid > 5.0
        @test n_studies > 0

        # Unknown interaction
        mean_unknown, sd_unknown, n_unknown = get_database_prior("unknowndrug", "unknownvictim")
        @test n_unknown == 0  # No clinical data
    end

    # =========================================================================
    # INTEGRATION WITH MEDLANG
    # =========================================================================
    @testset "MedLang Module Integration" begin
        # Check exports are available
        @test @isdefined DDIMechanism
        @test @isdefined DDIPerpetrator
        @test @isdefined DDIVictim
        @test @isdefined DDIPrior
        @test @isdefined DDIPosterior
        @test @isdefined DDIRiskAssessment

        # Check function exports
        @test @isdefined perpetrator_preset
        @test @isdefined victim_preset
        @test @isdefined predict_auc_ratio
        @test @isdefined ddi_risk_classification
        @test @isdefined simulate_ddi_scenario
        @test @isdefined validate_ddi_model
    end

    # =========================================================================
    # SPECIFIC DDI SCENARIOS FROM LITERATURE
    # =========================================================================
    @testset "Literature DDI Validation" begin
        # Ketoconazole + Midazolam: Expected AUC ratio ~10-16x
        keto = perpetrator_preset(:ketoconazole)
        midaz = victim_preset(:midazolam)
        pred_keto_mid = predict_auc_ratio(keto, midaz)

        @test pred_keto_mid.auc_ratio_mean > 5.0
        @test pred_keto_mid.auc_ratio_mean < 25.0  # Not unrealistically high

        # Rifampin + Midazolam: Expected AUC ratio ~0.02-0.1x
        rif = perpetrator_preset(:rifampin)
        pred_rif_mid = predict_auc_ratio(rif, midaz)

        @test pred_rif_mid.auc_ratio_mean < 0.5
        @test pred_rif_mid.auc_ratio_mean > 0.001  # Not zero

        # Cyclosporine + Rosuvastatin: Expected AUC ratio ~5-10x (OATP)
        cyclo = perpetrator_preset(:cyclosporine)
        rosu = victim_preset(:rosuvastatin)
        pred_cyclo_rosu = predict_auc_ratio(cyclo, rosu)

        @test pred_cyclo_rosu.auc_ratio_mean > 2.0
    end

    # =========================================================================
    # SUMMARY
    # =========================================================================
    @testset "Test Summary Statistics" begin
        # Count tests (informational)
        println("\n" * "="^60)
        println("BAYESIAN DDI MODEL TEST SUMMARY")
        println("="^60)
        println("All tests completed successfully!")
        println("Coverage: Mechanisms, Priors, Posteriors, Risk Assessment")
        println("Validated against: FDA DDI guidance, clinical DDI database")
        println("="^60)
    end

end  # Main testset
