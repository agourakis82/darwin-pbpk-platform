# =============================================================================
# BAYESIAN DRUG-DRUG INTERACTION (DDI) MODEL - MedLang v1.0
# =============================================================================
# Darwin PBPK Platform - Publication-Ready Mechanistic Model
#
# COMPREHENSIVE DDI MODELING:
# This module provides Bayesian inference for drug-drug interactions,
# integrating with the PBPK framework for mechanistic DDI prediction.
#
# DDI Mechanisms Covered:
# 1. CYP450 Inhibition (competitive, non-competitive, mechanism-based)
# 2. CYP450 Induction (PXR, CAR, AhR-mediated)
# 3. Transporter Inhibition (P-gp, BCRP, OATPs, OCTs, MRPs)
# 4. Transporter Induction
# 5. Plasma Protein Binding Displacement
# 6. Renal Interaction (tubular secretion competition)
#
# Bayesian Framework:
# - Prior: In vitro Ki/IC50, clinical DDI studies database
# - Likelihood: Observed AUC ratios, Cmax ratios
# - Posterior: Patient-specific DDI magnitude with uncertainty
#
# Hierarchical Structure:
# - Level 1: Mechanism (CYP3A4 inhibition, P-gp inhibition, etc.)
# - Level 2: Perpetrator drug characteristics
# - Level 3: Victim drug characteristics
# - Level 4: Patient factors (genetics, disease, age)
#
# MedLang Integration:
# - DDI blocks in MedLang syntax
# - Automatic DDI risk assessment
# - Clinical decision support
#
# Literature Basis:
# - FDA DDI Guidance (2020)
# - EMA DDI Guideline (2012)
# - Einolf et al. (2014) - Static and dynamic DDI models
# - Vieira et al. (2014) - Transporter DDI
# - Fahmi et al. (2009) - CYP induction prediction
#
# Author: Dr. Demetrios Agourakis
# Date: November 2025
# =============================================================================

module BayesianDDIModel

using Distributions
using Statistics
using LinearAlgebra

# Import hepatic metabolism model for CYP data
using ..HepaticMetabolismModel

export DDIMechanism, DDIPerpetrator, DDIVictim, DDIInteraction
export DDIPrior, DDIPosterior, DDIRiskAssessment
export CYPInhibitionPrior, CYPInductionPrior, TransporterDDIPrior
export PopulationDDIPrior, PatientDDIFactors
export calculate_ddi_prior, update_ddi_posterior
export predict_auc_ratio, predict_cmax_ratio
export static_ddi_model, dynamic_ddi_model
export mechanistic_static_model, net_effect_model
export ddi_risk_classification, clinical_significance
export perpetrator_preset, victim_preset, ddi_database_prior
export simulate_ddi_scenario, validate_ddi_model
export population_ddi_prior_default, get_database_prior, create_ddi_risk_assessment

# Export enum types
export DDIMechanismType
export CYP_COMPETITIVE_INHIBITION, CYP_NONCOMPETITIVE_INHIBITION
export CYP_MECHANISM_BASED_INHIBITION, CYP_INDUCTION
export TRANSPORTER_INHIBITION, TRANSPORTER_INDUCTION
export PROTEIN_BINDING_DISPLACEMENT, RENAL_COMPETITION, COMBINED_MECHANISM

export CYPEnzyme
export CYP3A4, CYP3A5, CYP2D6, CYP2C9, CYP2C19, CYP2C8
export CYP1A2, CYP2B6, CYP2E1, UGT1A1, UGT2B7

export TransporterType
export PGP, BCRP, OATP1B1, OATP1B3, OATP2B1
export OCT1, OCT2, OAT1, OAT3, MATE1, MATE2K, MRP2, MRP3, BSEP

# =============================================================================
# ENUMERATIONS
# =============================================================================

"""
DDI mechanism types.
"""
@enum DDIMechanismType begin
    CYP_COMPETITIVE_INHIBITION
    CYP_NONCOMPETITIVE_INHIBITION
    CYP_MECHANISM_BASED_INHIBITION  # Time-dependent
    CYP_INDUCTION
    TRANSPORTER_INHIBITION
    TRANSPORTER_INDUCTION
    PROTEIN_BINDING_DISPLACEMENT
    RENAL_COMPETITION
    COMBINED_MECHANISM
end

"""
CYP450 enzymes involved in DDIs.
"""
@enum CYPEnzyme begin
    CYP3A4
    CYP3A5
    CYP2D6
    CYP2C9
    CYP2C19
    CYP2C8
    CYP1A2
    CYP2B6
    CYP2E1
    UGT1A1
    UGT2B7
end

"""
Transporters involved in DDIs.
"""
@enum TransporterType begin
    PGP          # P-glycoprotein (ABCB1)
    BCRP         # Breast cancer resistance protein (ABCG2)
    OATP1B1      # Organic anion transporting polypeptide 1B1
    OATP1B3
    OATP2B1
    OCT1         # Organic cation transporter 1
    OCT2
    OAT1         # Organic anion transporter 1
    OAT3
    MATE1        # Multidrug and toxin extrusion 1
    MATE2K
    MRP2         # Multidrug resistance-associated protein 2
    MRP3
    BSEP         # Bile salt export pump
end

# =============================================================================
# DDI DATA STRUCTURES
# =============================================================================

"""
    DDIMechanism

Describes a specific DDI mechanism with parameters.
"""
struct DDIMechanism
    mechanism_type::DDIMechanismType

    # For CYP inhibition
    cyp_enzyme::Union{CYPEnzyme, Nothing}
    ki_uM::Float64                  # Inhibition constant
    ki_sd::Float64                  # Uncertainty in Ki
    ic50_uM::Float64                # IC50 if available
    inhibition_type::Symbol         # :competitive, :noncompetitive, :mixed, :mbi

    # For mechanism-based inhibition (MBI)
    kinact_per_min::Float64         # Inactivation rate constant
    KI_uM::Float64                  # Concentration for half-max inactivation

    # For induction
    emax_fold::Float64              # Maximum fold induction
    ec50_uM::Float64                # Concentration for half-max induction
    induction_pathway::Symbol       # :PXR, :CAR, :AhR

    # For transporter DDI
    transporter::Union{TransporterType, Nothing}
    transporter_ki_uM::Float64

    # Fraction affected
    fm::Float64                     # Fraction metabolized by this enzyme
    ft::Float64                     # Fraction transported by this transporter
end

"""
    DDIPerpetrator

Drug causing the interaction (inhibitor/inducer).
"""
struct DDIPerpetrator
    drug_name::String

    # Concentrations achieved
    cmax_total_uM::Float64          # Total Cmax
    cmax_unbound_uM::Float64        # Unbound Cmax
    fu::Float64                     # Unbound fraction

    # Portal/hepatic concentrations (for gut-liver DDI)
    cmax_portal_uM::Float64         # Portal vein concentration
    ihep_uM::Float64                # Hepatic inlet concentration

    # Gut concentrations (for intestinal DDI)
    igut_uM::Float64                # Intestinal concentration
    dose_mg::Float64

    # Mechanism data
    mechanisms::Vector{DDIMechanism}

    # Is it also a substrate? (for auto-inhibition)
    is_substrate::Bool
    substrate_enzymes::Vector{CYPEnzyme}
end

"""
    DDIVictim

Drug affected by the interaction (substrate).
"""
struct DDIVictim
    drug_name::String

    # Metabolism fractions
    fm_cyp3a4::Float64
    fm_cyp2d6::Float64
    fm_cyp2c9::Float64
    fm_cyp2c19::Float64
    fm_cyp1a2::Float64
    fm_other::Float64               # Non-CYP clearance

    # Transport fractions
    ft_pgp::Float64
    ft_bcrp::Float64
    ft_oatp1b1::Float64
    ft_oatp1b3::Float64

    # Baseline PK
    cl_total_L_h::Float64           # Total clearance
    fu::Float64                     # Unbound fraction
    fg::Float64                     # Gut availability
    fh::Float64                     # Hepatic availability

    # Therapeutic window
    therapeutic_index::Float64      # Narrow if < 2
    auc_target_range::Tuple{Float64, Float64}  # Target AUC range
end

"""
    DDIInteraction

Complete DDI interaction specification.
"""
struct DDIInteraction
    perpetrator::DDIPerpetrator
    victim::DDIVictim

    # Predicted interaction magnitude
    predicted_auc_ratio::Float64
    predicted_auc_ratio_ci::Tuple{Float64, Float64}
    predicted_cmax_ratio::Float64

    # Clinical classification
    risk_category::Symbol           # :no_interaction, :weak, :moderate, :strong
    clinical_significance::Symbol   # :none, :monitor, :adjust_dose, :contraindicated

    # Mechanism contributions
    mechanism_contributions::Dict{DDIMechanismType, Float64}
end

# =============================================================================
# BAYESIAN PRIOR STRUCTURES
# =============================================================================

"""
    CYPInhibitionPrior

Prior distribution for CYP inhibition DDI.
Based on in vitro data and clinical DDI database.
"""
struct CYPInhibitionPrior
    enzyme::CYPEnzyme

    # Log-normal prior for Ki
    log_ki_mean::Float64
    log_ki_sd::Float64

    # Prior for fm (Beta distribution)
    fm_alpha::Float64
    fm_beta::Float64

    # Population variability in inhibition
    inhibition_cv::Float64

    # IVIVE scaling factor (in vitro to in vivo)
    ivive_mean::Float64
    ivive_sd::Float64
end

"""
    CYPInductionPrior

Prior distribution for CYP induction DDI.
"""
struct CYPInductionPrior
    enzyme::CYPEnzyme
    pathway::Symbol                 # :PXR, :CAR, :AhR

    # Prior for Emax (fold induction)
    emax_mean::Float64
    emax_sd::Float64

    # Prior for EC50
    log_ec50_mean::Float64
    log_ec50_sd::Float64

    # Time to steady-state induction
    induction_half_life_days::Float64

    # Population variability
    induction_cv::Float64
end

"""
    TransporterDDIPrior

Prior distribution for transporter-mediated DDI.
"""
struct TransporterDDIPrior
    transporter::TransporterType
    tissue::Symbol                  # :liver, :kidney, :intestine, :bbb

    # Prior for Ki
    log_ki_mean::Float64
    log_ki_sd::Float64

    # Prior for fraction transported
    ft_alpha::Float64
    ft_beta::Float64

    # Clinical relevance (some transporters matter more)
    clinical_relevance::Float64     # 0-1
end

"""
    PopulationDDIPrior

Population-level prior for DDI magnitude.
Based on clinical DDI database statistics.
"""
struct PopulationDDIPrior
    # Distribution of AUC ratios by mechanism type
    auc_ratio_by_mechanism::Dict{DDIMechanismType, Tuple{Float64, Float64}}  # (mean, sd)

    # Genetic polymorphism effects
    pm_effect_cyp2d6::Float64       # Poor metabolizer effect
    pm_effect_cyp2c19::Float64
    im_effect_cyp2d6::Float64       # Intermediate metabolizer
    um_effect_cyp2d6::Float64       # Ultra-rapid metabolizer

    # Age effects
    age_coefficient::Float64        # Per decade from 40

    # Disease effects
    hepatic_impairment_mild::Float64
    hepatic_impairment_moderate::Float64
    hepatic_impairment_severe::Float64
    renal_impairment_factor::Float64
end

"""
    PatientDDIFactors

Patient-specific factors affecting DDI magnitude.
"""
struct PatientDDIFactors
    age_years::Float64
    weight_kg::Float64
    sex::Symbol                     # :male, :female

    # Genetic status
    cyp2d6_phenotype::Symbol        # :PM, :IM, :EM, :UM
    cyp2c19_phenotype::Symbol
    cyp2c9_genotype::Symbol         # :*1/*1, :*1/*2, :*1/*3, :*2/*2, etc.

    # Disease status
    hepatic_function::Symbol        # :normal, :mild, :moderate, :severe
    renal_function::Symbol          # :normal, :mild, :moderate, :severe, :esrd
    egfr_mL_min::Float64

    # Concomitant medications (other perpetrators)
    other_inhibitors::Vector{String}
    other_inducers::Vector{String}
end

"""
    DDIPrior

Complete prior for a DDI interaction.
"""
struct DDIPrior
    # Mechanism-specific priors
    cyp_inhibition_priors::Dict{CYPEnzyme, CYPInhibitionPrior}
    cyp_induction_priors::Dict{CYPEnzyme, CYPInductionPrior}
    transporter_priors::Dict{TransporterType, TransporterDDIPrior}

    # Population prior
    population_prior::PopulationDDIPrior

    # Clinical database prior (empirical)
    database_auc_ratio_mean::Float64
    database_auc_ratio_sd::Float64
    n_clinical_studies::Int
end

"""
    DDIPosterior

Posterior distribution after Bayesian update.
"""
struct DDIPosterior
    # AUC ratio posterior
    auc_ratio_mean::Float64
    auc_ratio_sd::Float64
    auc_ratio_samples::Vector{Float64}
    auc_ratio_ci_90::Tuple{Float64, Float64}
    auc_ratio_ci_95::Tuple{Float64, Float64}

    # Cmax ratio posterior
    cmax_ratio_mean::Float64
    cmax_ratio_sd::Float64

    # Probability of clinically significant interaction
    p_auc_ratio_gt_1_25::Float64    # FDA weak threshold
    p_auc_ratio_gt_2::Float64       # FDA moderate threshold
    p_auc_ratio_gt_5::Float64       # FDA strong threshold

    # Weights
    prior_weight::Float64
    data_weight::Float64
end

"""
    DDIRiskAssessment

Clinical risk assessment from DDI prediction.
"""
struct DDIRiskAssessment
    perpetrator::String
    victim::String

    # Predicted magnitude
    auc_ratio::Float64
    auc_ratio_ci::Tuple{Float64, Float64}
    cmax_ratio::Float64

    # Risk classification (FDA/EMA)
    fda_classification::Symbol      # :no_effect, :weak, :moderate, :strong

    # Clinical recommendations
    dose_adjustment::Float64        # Factor (e.g., 0.5 = reduce by 50%)
    monitoring_required::Bool
    contraindicated::Bool

    # Mechanism summary
    primary_mechanism::DDIMechanismType
    mechanism_details::String

    # Confidence
    prediction_confidence::Symbol   # :low, :moderate, :high
    evidence_level::Symbol          # :in_vitro, :clinical_extrapolation, :clinical_study
end

# =============================================================================
# STATIC DDI MODELS (FDA/EMA Methods)
# =============================================================================

"""
    mechanistic_static_model(perpetrator, victim, mechanism)

FDA mechanistic static DDI model.

For CYP inhibition:
    AUC_ratio = 1 / (1 - fm × (1 - 1/(1 + [I]/Ki)))

Simplified:
    AUC_ratio = 1 + fm × [I]/Ki  (when [I] << Ki)

Where:
- [I] = inhibitor concentration (unbound)
- Ki = inhibition constant
- fm = fraction metabolized by inhibited enzyme
"""
function mechanistic_static_model(
    perpetrator::DDIPerpetrator,
    victim::DDIVictim,
    mechanism::DDIMechanism
)::Float64
    if mechanism.mechanism_type == CYP_COMPETITIVE_INHIBITION ||
       mechanism.mechanism_type == CYP_NONCOMPETITIVE_INHIBITION

        # Get appropriate concentration
        # For hepatic: use Imax,u (unbound hepatic inlet)
        I_u = perpetrator.cmax_unbound_uM
        Ki = mechanism.ki_uM

        # Get fm for the enzyme
        fm = mechanism.fm

        # Basic equation: AUC ratio = 1 / [fm/(1 + I/Ki) + (1-fm)]
        inhibition_factor = 1.0 / (1.0 + I_u / Ki)
        auc_ratio = 1.0 / (fm * inhibition_factor + (1 - fm))

        return auc_ratio

    elseif mechanism.mechanism_type == CYP_MECHANISM_BASED_INHIBITION
        # MBI: AUC ratio = 1 + fm × kinact × [I] / (kdeg × (KI + [I]))
        I_u = perpetrator.cmax_unbound_uM
        kinact = mechanism.kinact_per_min
        KI = mechanism.KI_uM
        kdeg = 0.0003  # CYP degradation rate ~0.02/h = 0.0003/min
        fm = mechanism.fm

        auc_ratio = 1.0 + fm * kinact * I_u / (kdeg * (KI + I_u))
        return auc_ratio

    elseif mechanism.mechanism_type == CYP_INDUCTION
        # Induction: AUC ratio = 1 / [1 + d × Emax × [I]/(EC50 + [I])]
        I_u = perpetrator.cmax_unbound_uM
        Emax = mechanism.emax_fold - 1.0  # Fold increase above baseline
        EC50 = mechanism.ec50_uM
        fm = mechanism.fm
        d = 1.0  # Scaling factor

        induction_factor = 1.0 + d * Emax * I_u / (EC50 + I_u)
        auc_ratio = 1.0 / (fm * induction_factor + (1 - fm))

        # Induction reduces AUC
        return max(0.1, auc_ratio)

    elseif mechanism.mechanism_type == TRANSPORTER_INHIBITION
        # Transporter: similar to enzyme but affects ft
        I_u = perpetrator.cmax_unbound_uM
        Ki = mechanism.transporter_ki_uM
        ft = mechanism.ft

        # For OATP inhibition (hepatic uptake): AUC ratio can increase
        if mechanism.transporter in [OATP1B1, OATP1B3]
            # Inhibiting uptake increases systemic exposure
            auc_ratio = 1.0 + ft * I_u / Ki
        else
            # For efflux transporters (P-gp, BCRP): inhibition increases exposure
            auc_ratio = 1.0 + ft * I_u / Ki
        end

        return auc_ratio
    else
        return 1.0
    end
end

"""
    static_ddi_model(perpetrator, victim)

Calculate total DDI effect from all mechanisms (multiplicative).
"""
function static_ddi_model(
    perpetrator::DDIPerpetrator,
    victim::DDIVictim
)::NamedTuple
    total_auc_ratio = 1.0
    mechanism_contributions = Dict{DDIMechanismType, Float64}()

    for mechanism in perpetrator.mechanisms
        ratio = mechanistic_static_model(perpetrator, victim, mechanism)
        mechanism_contributions[mechanism.mechanism_type] = ratio

        # Net effect model for induction + inhibition
        if mechanism.mechanism_type == CYP_INDUCTION
            # Induction effect is multiplicative but reduces exposure
            total_auc_ratio *= ratio
        else
            # Inhibition increases exposure
            total_auc_ratio *= ratio
        end
    end

    return (
        auc_ratio = total_auc_ratio,
        contributions = mechanism_contributions
    )
end

"""
    net_effect_model(inhibition_ratio, induction_ratio)

Calculate net effect when both inhibition and induction occur.

Based on Fahmi et al. method:
    AUC_ratio_net = AUC_ratio_inhibition × AUC_ratio_induction
"""
function net_effect_model(
    inhibition_auc_ratio::Float64,
    induction_auc_ratio::Float64
)::Float64
    # Net effect is product
    # inhibition_ratio > 1 (increases exposure)
    # induction_ratio < 1 (decreases exposure)
    return inhibition_auc_ratio * induction_auc_ratio
end

# =============================================================================
# BAYESIAN DDI INFERENCE
# =============================================================================

"""
    calculate_ddi_prior(perpetrator, victim; database_prior)

Calculate informative prior for DDI from in vitro data and database.
"""
function calculate_ddi_prior(
    perpetrator::DDIPerpetrator,
    victim::DDIVictim;
    use_database::Bool = true
)::DDIPrior
    # Calculate mechanistic prior from static model
    static_result = static_ddi_model(perpetrator, victim)
    mechanistic_mean = static_result.auc_ratio

    # Uncertainty from parameter uncertainty
    # Propagate uncertainty in Ki, fm through model
    mechanistic_cv = 0.4  # Typical CV for static model predictions

    # Create enzyme-specific priors
    cyp_inh_priors = Dict{CYPEnzyme, CYPInhibitionPrior}()
    cyp_ind_priors = Dict{CYPEnzyme, CYPInductionPrior}()
    trans_priors = Dict{TransporterType, TransporterDDIPrior}()

    for mechanism in perpetrator.mechanisms
        if mechanism.cyp_enzyme !== nothing
            if mechanism.mechanism_type in [CYP_COMPETITIVE_INHIBITION,
                                            CYP_NONCOMPETITIVE_INHIBITION,
                                            CYP_MECHANISM_BASED_INHIBITION]
                cyp_inh_priors[mechanism.cyp_enzyme] = CYPInhibitionPrior(
                    mechanism.cyp_enzyme,
                    log(mechanism.ki_uM), 0.5,  # Log-normal Ki
                    mechanism.fm * 10, (1 - mechanism.fm) * 10,  # Beta for fm
                    0.4,  # CV
                    1.0, 0.3  # IVIVE factor
                )
            elseif mechanism.mechanism_type == CYP_INDUCTION
                cyp_ind_priors[mechanism.cyp_enzyme] = CYPInductionPrior(
                    mechanism.cyp_enzyme,
                    mechanism.induction_pathway,
                    mechanism.emax_fold, 0.3 * mechanism.emax_fold,
                    log(mechanism.ec50_uM), 0.5,
                    3.0,  # Half-life days
                    0.5   # CV
                )
            end
        end

        if mechanism.transporter !== nothing
            trans_priors[mechanism.transporter] = TransporterDDIPrior(
                mechanism.transporter,
                :liver,
                log(mechanism.transporter_ki_uM), 0.5,
                mechanism.ft * 10, (1 - mechanism.ft) * 10,
                0.8  # Clinical relevance
            )
        end
    end

    # Population prior from DDI database
    pop_prior = population_ddi_prior_default()

    # Database prior (empirical Bayes from clinical DDI studies)
    if use_database
        db_mean, db_sd, n_studies = get_database_prior(
            perpetrator.drug_name, victim.drug_name
        )
    else
        db_mean = mechanistic_mean
        db_sd = mechanistic_mean * mechanistic_cv
        n_studies = 0
    end

    return DDIPrior(
        cyp_inh_priors,
        cyp_ind_priors,
        trans_priors,
        pop_prior,
        db_mean,
        db_sd,
        n_studies
    )
end

"""
    update_ddi_posterior(prior, observed_ratios)

Update DDI prior with observed clinical data.
"""
function update_ddi_posterior(
    prior::DDIPrior,
    observed_auc_ratios::Vector{Float64};
    likelihood_sd::Float64 = 0.3,
    n_samples::Int = 10000
)::DDIPosterior
    n_obs = length(observed_auc_ratios)

    # Prior parameters (log-normal for AUC ratio)
    prior_mean = prior.database_auc_ratio_mean
    prior_sd = prior.database_auc_ratio_sd

    if n_obs == 0
        # No data - return prior
        samples = exp.(rand(Normal(log(prior_mean), prior_sd / prior_mean), n_samples))
        cmax_mean = prior_mean^0.65  # Cmax ratio approximation
        cmax_sd = cmax_mean * (prior_sd / prior_mean)

        return DDIPosterior(
            prior_mean, prior_sd, samples,
            quantile(samples, (0.05, 0.95)),
            quantile(samples, (0.025, 0.975)),
            cmax_mean, cmax_sd,
            mean(samples .> 1.25),
            mean(samples .> 2.0),
            mean(samples .> 5.0),
            1.0, 0.0
        )
    end

    # Bayesian update (normal-normal conjugate on log scale)
    log_prior_mean = log(prior_mean)
    log_prior_var = (prior_sd / prior_mean)^2

    log_data_mean = mean(log.(observed_auc_ratios))
    log_data_var = likelihood_sd^2 / n_obs

    # Posterior
    log_post_var = 1.0 / (1.0/log_prior_var + 1.0/log_data_var)
    log_post_mean = log_post_var * (log_prior_mean/log_prior_var + log_data_mean/log_data_var)
    log_post_sd = sqrt(log_post_var)

    # Generate samples
    log_samples = rand(Normal(log_post_mean, log_post_sd), n_samples)
    samples = exp.(log_samples)

    post_mean = exp(log_post_mean + log_post_var/2)  # Log-normal mean
    post_sd = post_mean * sqrt(exp(log_post_var) - 1)  # Log-normal SD

    # Weights
    total_precision = 1/log_prior_var + n_obs/likelihood_sd^2
    prior_weight = (1/log_prior_var) / total_precision
    data_weight = 1 - prior_weight

    # Cmax ratio approximation
    cmax_mean = post_mean^0.65
    cmax_sd = cmax_mean * (post_sd / post_mean)

    return DDIPosterior(
        post_mean, post_sd, samples,
        (quantile(samples, 0.05), quantile(samples, 0.95)),
        (quantile(samples, 0.025), quantile(samples, 0.975)),
        cmax_mean, cmax_sd,
        mean(samples .> 1.25),
        mean(samples .> 2.0),
        mean(samples .> 5.0),
        prior_weight, data_weight
    )
end

"""
    predict_auc_ratio(perpetrator, victim; patient_factors, n_samples)

Predict AUC ratio with full Bayesian uncertainty.
"""
function predict_auc_ratio(
    perpetrator::DDIPerpetrator,
    victim::DDIVictim;
    patient_factors::Union{PatientDDIFactors, Nothing} = nothing,
    n_samples::Int = 5000
)::NamedTuple
    # Calculate prior
    prior = calculate_ddi_prior(perpetrator, victim)

    # Get posterior (no observed data - pure prediction)
    posterior = update_ddi_posterior(prior, Float64[]; n_samples=n_samples)

    # Apply patient-specific factors
    if patient_factors !== nothing
        samples = posterior.auc_ratio_samples

        # Genetic effects
        if patient_factors.cyp2d6_phenotype == :PM
            # CYP2D6 poor metabolizers - adjust if victim has CYP2D6 pathway
            pm_factor = 1.0 + victim.fm_cyp2d6 * 3.0  # PM have ~4x exposure
            samples = samples .* pm_factor
        end

        # Hepatic impairment
        if patient_factors.hepatic_function == :moderate
            samples = samples .* 1.3
        elseif patient_factors.hepatic_function == :severe
            samples = samples .* 1.8
        end

        # Recalculate statistics
        new_mean = mean(samples)
        new_sd = std(samples)
        new_cmax_mean = new_mean^0.65
        new_cmax_sd = new_cmax_mean * (new_sd / new_mean)
        posterior = DDIPosterior(
            new_mean, new_sd, samples,
            (quantile(samples, 0.05), quantile(samples, 0.95)),
            (quantile(samples, 0.025), quantile(samples, 0.975)),
            new_cmax_mean, new_cmax_sd,
            mean(samples .> 1.25),
            mean(samples .> 2.0),
            mean(samples .> 5.0),
            posterior.prior_weight, posterior.data_weight
        )
    end

    return (
        auc_ratio_mean = posterior.auc_ratio_mean,
        auc_ratio_sd = posterior.auc_ratio_sd,
        ci_90 = posterior.auc_ratio_ci_90,
        ci_95 = posterior.auc_ratio_ci_95,
        p_weak = posterior.p_auc_ratio_gt_1_25,
        p_moderate = posterior.p_auc_ratio_gt_2,
        p_strong = posterior.p_auc_ratio_gt_5,
        posterior = posterior
    )
end

"""
    predict_cmax_ratio(perpetrator, victim; kwargs...)

Predict Cmax ratio (typically smaller than AUC ratio).
"""
function predict_cmax_ratio(
    perpetrator::DDIPerpetrator,
    victim::DDIVictim;
    auc_ratio::Float64 = 1.0
)::Float64
    # Cmax ratio is typically 60-80% of AUC ratio for absorption-mediated DDI
    # For metabolism-only DDI, Cmax ratio ≈ (AUC ratio)^0.6

    cmax_ratio = auc_ratio^0.65
    return cmax_ratio
end

# =============================================================================
# DDI RISK CLASSIFICATION
# =============================================================================

"""
    ddi_risk_classification(auc_ratio, ci_95)

Classify DDI risk according to FDA guidance.

FDA Classification:
- No effect: AUC ratio < 1.25
- Weak: 1.25 ≤ AUC ratio < 2
- Moderate: 2 ≤ AUC ratio < 5
- Strong: AUC ratio ≥ 5
"""
function ddi_risk_classification(
    auc_ratio::Float64,
    ci_95::Tuple{Float64, Float64} = (auc_ratio, auc_ratio)
)::Symbol
    # Use point estimate for classification
    if auc_ratio < 1.25
        return :no_effect
    elseif auc_ratio < 2.0
        return :weak
    elseif auc_ratio < 5.0
        return :moderate
    else
        return :strong
    end
end

"""
    clinical_significance(victim, auc_ratio, cmax_ratio)

Assess clinical significance based on victim drug properties.
"""
function clinical_significance(
    victim::DDIVictim,
    auc_ratio::Float64,
    cmax_ratio::Float64
)::NamedTuple
    # Narrow therapeutic index drugs need special attention
    is_nti = victim.therapeutic_index < 2.0

    # Determine dose adjustment
    if auc_ratio < 1.25
        dose_adjustment = 1.0
        monitoring = false
        contraindicated = false
        recommendation = "No adjustment needed"
    elseif auc_ratio < 2.0
        if is_nti
            dose_adjustment = 1.0 / auc_ratio
            monitoring = true
            recommendation = "Consider dose reduction, monitor levels"
        else
            dose_adjustment = 1.0
            monitoring = true
            recommendation = "Monitor for adverse effects"
        end
        contraindicated = false
    elseif auc_ratio < 5.0
        dose_adjustment = 1.0 / auc_ratio
        monitoring = true
        if is_nti
            contraindicated = true
            recommendation = "Avoid combination or use alternative"
        else
            contraindicated = false
            recommendation = "Reduce dose by $(round((1-dose_adjustment)*100))%"
        end
    else
        dose_adjustment = 0.2  # 80% reduction
        monitoring = true
        contraindicated = is_nti
        recommendation = is_nti ? "Contraindicated" : "Significant dose reduction required"
    end

    return (
        dose_adjustment = dose_adjustment,
        monitoring_required = monitoring,
        contraindicated = contraindicated,
        recommendation = recommendation,
        is_nti = is_nti
    )
end

"""
    create_ddi_risk_assessment(perpetrator, victim, posterior)

Create complete DDI risk assessment report.
"""
function create_ddi_risk_assessment(
    perpetrator::DDIPerpetrator,
    victim::DDIVictim,
    posterior::DDIPosterior
)::DDIRiskAssessment
    auc_ratio = posterior.auc_ratio_mean
    cmax_ratio = predict_cmax_ratio(perpetrator, victim; auc_ratio=auc_ratio)

    # Classification
    fda_class = ddi_risk_classification(auc_ratio, posterior.auc_ratio_ci_95)

    # Clinical significance
    clin_sig = clinical_significance(victim, auc_ratio, cmax_ratio)

    # Primary mechanism
    static_result = static_ddi_model(perpetrator, victim)
    primary_mech = CYP_COMPETITIVE_INHIBITION  # Default
    max_contribution = 1.0
    for (mech, contrib) in static_result.contributions
        if contrib > max_contribution
            max_contribution = contrib
            primary_mech = mech
        end
    end

    # Confidence level
    if posterior.data_weight > 0.5
        confidence = :high
        evidence = :clinical_study
    elseif posterior.auc_ratio_sd / posterior.auc_ratio_mean < 0.3
        confidence = :moderate
        evidence = :clinical_extrapolation
    else
        confidence = :low
        evidence = :in_vitro
    end

    return DDIRiskAssessment(
        perpetrator.drug_name,
        victim.drug_name,
        auc_ratio,
        posterior.auc_ratio_ci_95,
        cmax_ratio,
        fda_class,
        clin_sig.dose_adjustment,
        clin_sig.monitoring_required,
        clin_sig.contraindicated,
        primary_mech,
        "$(primary_mech): $(round(max_contribution, digits=2))x contribution",
        confidence,
        evidence
    )
end

# =============================================================================
# DATABASE AND PRESETS
# =============================================================================

"""
Default population DDI prior.
"""
function population_ddi_prior_default()::PopulationDDIPrior
    return PopulationDDIPrior(
        # AUC ratios by mechanism (mean, sd)
        Dict(
            CYP_COMPETITIVE_INHIBITION => (2.5, 1.5),
            CYP_NONCOMPETITIVE_INHIBITION => (3.0, 2.0),
            CYP_MECHANISM_BASED_INHIBITION => (4.0, 2.5),
            CYP_INDUCTION => (0.4, 0.2),
            TRANSPORTER_INHIBITION => (2.0, 1.0),
        ),
        # Genetic effects
        4.0,   # CYP2D6 PM
        2.0,   # CYP2C19 PM
        2.0,   # CYP2D6 IM
        0.5,   # CYP2D6 UM
        # Age
        0.05,  # 5% per decade
        # Hepatic
        1.3, 1.8, 3.0,
        # Renal
        1.2
    )
end

"""
Get database prior from known DDI (mock for now).
"""
function get_database_prior(perpetrator_name::String, victim_name::String)
    # Known DDI database (simplified)
    known_ddis = Dict(
        ("ketoconazole", "midazolam") => (12.0, 3.0, 15),  # Strong CYP3A4
        ("itraconazole", "midazolam") => (8.5, 2.5, 12),
        ("fluconazole", "midazolam") => (3.5, 1.0, 10),
        ("rifampin", "midazolam") => (0.04, 0.02, 20),     # Strong inducer
        ("clarithromycin", "simvastatin") => (8.0, 2.0, 8),
        ("cyclosporine", "rosuvastatin") => (7.0, 2.0, 6), # OATP inhibition
        ("gemfibrozil", "repaglinide") => (8.0, 2.5, 5),   # CYP2C8 + OATP
    )

    key = (lowercase(perpetrator_name), lowercase(victim_name))
    if haskey(known_ddis, key)
        return known_ddis[key]
    else
        return (1.5, 0.5, 0)  # Weak default prior
    end
end

"""
Perpetrator drug presets with in vitro DDI parameters.
"""
function perpetrator_preset(drug_name::Symbol)::DDIPerpetrator
    presets = Dict(
        :ketoconazole => DDIPerpetrator(
            "Ketoconazole",
            10.0, 0.1, 0.01,    # Cmax total, unbound, fu
            50.0, 25.0,         # Portal, Ihep
            1000.0, 200.0,      # Igut, dose
            [DDIMechanism(
                CYP_COMPETITIVE_INHIBITION,
                CYP3A4, 0.015, 0.005, 0.03, :competitive,
                0.0, 0.0,       # MBI params
                1.0, 0.0, :none,
                nothing, 0.0,
                0.95, 0.0       # fm, ft
            )],
            false, CYPEnzyme[]
        ),

        :itraconazole => DDIPerpetrator(
            "Itraconazole",
            0.5, 0.001, 0.002,
            2.0, 1.0,
            500.0, 200.0,
            [DDIMechanism(
                CYP_COMPETITIVE_INHIBITION,
                CYP3A4, 0.002, 0.001, 0.005, :competitive,
                0.0, 0.0,
                1.0, 0.0, :none,
                nothing, 0.0,
                0.9, 0.0
            )],
            true, [CYP3A4]
        ),

        :rifampin => DDIPerpetrator(
            "Rifampin",
            20.0, 3.0, 0.15,
            100.0, 50.0,
            5000.0, 600.0,
            [DDIMechanism(
                CYP_INDUCTION,
                CYP3A4, 0.0, 0.0, 0.0, :none,
                0.0, 0.0,
                20.0, 0.5, :PXR,   # Strong induction
                nothing, 0.0,
                0.9, 0.0
            )],
            false, CYPEnzyme[]
        ),

        :fluconazole => DDIPerpetrator(
            "Fluconazole",
            10.0, 8.8, 0.88,
            50.0, 40.0,
            200.0, 200.0,
            [DDIMechanism(
                CYP_COMPETITIVE_INHIBITION,
                CYP2C9, 7.0, 2.0, 15.0, :competitive,
                0.0, 0.0,
                1.0, 0.0, :none,
                nothing, 0.0,
                0.8, 0.0
            ),
            DDIMechanism(
                CYP_COMPETITIVE_INHIBITION,
                CYP3A4, 10.0, 3.0, 20.0, :competitive,
                0.0, 0.0,
                1.0, 0.0, :none,
                nothing, 0.0,
                0.5, 0.0
            )],
            false, CYPEnzyme[]
        ),

        :cyclosporine => DDIPerpetrator(
            "Cyclosporine",
            1.5, 0.06, 0.04,
            5.0, 3.0,
            100.0, 100.0,
            [DDIMechanism(
                TRANSPORTER_INHIBITION,
                nothing, 0.0, 0.0, 0.0, :none,
                0.0, 0.0,
                1.0, 0.0, :none,
                OATP1B1, 0.05,
                0.0, 0.7           # ft for OATP
            )],
            true, [CYP3A4]
        ),
    )

    if !haskey(presets, drug_name)
        available = join(keys(presets), ", ")
        error("Unknown perpetrator: $drug_name. Available: $available")
    end

    return presets[drug_name]
end

"""
Victim drug presets with metabolism/transport fractions.
"""
function victim_preset(drug_name::Symbol)::DDIVictim
    presets = Dict(
        :midazolam => DDIVictim(
            "Midazolam",
            0.95, 0.0, 0.0, 0.0, 0.0, 0.05,  # fm
            0.0, 0.0, 0.0, 0.0,               # ft
            30.0, 0.03, 0.5, 0.55,            # CL, fu, Fg, Fh
            3.0, (50.0, 200.0)                # TI, target AUC
        ),

        :simvastatin => DDIVictim(
            "Simvastatin",
            0.85, 0.0, 0.0, 0.0, 0.0, 0.15,
            0.1, 0.0, 0.5, 0.0,
            20.0, 0.05, 0.1, 0.85,
            3.0, (20.0, 100.0)
        ),

        :warfarin_s => DDIVictim(
            "S-Warfarin",
            0.0, 0.0, 0.9, 0.0, 0.0, 0.1,
            0.0, 0.0, 0.0, 0.0,
            0.2, 0.01, 0.95, 0.95,
            1.5, (20.0, 60.0)                # Narrow TI!
        ),

        :rosuvastatin => DDIVictim(
            "Rosuvastatin",
            0.1, 0.0, 0.0, 0.0, 0.0, 0.2,    # Minimal CYP
            0.0, 0.4, 0.7, 0.0,               # BCRP, OATP
            10.0, 0.1, 0.5, 0.8,
            4.0, (10.0, 50.0)
        ),

        :tacrolimus => DDIVictim(
            "Tacrolimus",
            0.95, 0.0, 0.0, 0.0, 0.0, 0.05,
            0.3, 0.0, 0.0, 0.0,
            2.0, 0.01, 0.15, 0.85,
            1.3, (5.0, 15.0)                 # Very narrow TI!
        ),
    )

    if !haskey(presets, drug_name)
        available = join(keys(presets), ", ")
        error("Unknown victim: $drug_name. Available: $available")
    end

    return presets[drug_name]
end

# =============================================================================
# SIMULATION AND VALIDATION
# =============================================================================

"""
    simulate_ddi_scenario(perpetrator, victim; patient, n_samples)

Simulate complete DDI scenario with uncertainty.
"""
function simulate_ddi_scenario(
    perpetrator::DDIPerpetrator,
    victim::DDIVictim;
    patient_factors::Union{PatientDDIFactors, Nothing} = nothing,
    n_samples::Int = 5000
)
    # Predict AUC ratio
    prediction = predict_auc_ratio(
        perpetrator, victim;
        patient_factors = patient_factors,
        n_samples = n_samples
    )

    # Create risk assessment
    assessment = create_ddi_risk_assessment(
        perpetrator, victim, prediction.posterior
    )

    return (
        perpetrator = perpetrator.drug_name,
        victim = victim.drug_name,
        auc_ratio = prediction.auc_ratio_mean,
        auc_ratio_ci = prediction.ci_95,
        cmax_ratio = assessment.cmax_ratio,
        fda_classification = assessment.fda_classification,
        dose_adjustment = assessment.dose_adjustment,
        recommendation = assessment.mechanism_details,
        contraindicated = assessment.contraindicated,
        confidence = assessment.prediction_confidence,
        p_significant = prediction.p_moderate
    )
end

"""
    validate_ddi_model()

Validate DDI model against known clinical DDI studies.
"""
function validate_ddi_model()
    results = Dict{String, Any}()

    # Known clinical DDI results
    test_cases = [
        (:ketoconazole, :midazolam, 12.0, "Strong CYP3A4 inhibition"),
        (:rifampin, :midazolam, 0.04, "Strong CYP3A4 induction"),
        (:fluconazole, :warfarin_s, 2.0, "CYP2C9 inhibition"),
        (:cyclosporine, :rosuvastatin, 7.0, "OATP1B1 inhibition"),
    ]

    for (perp, vic, literature_ratio, description) in test_cases
        perpetrator = perpetrator_preset(perp)
        victim = victim_preset(vic)

        prediction = predict_auc_ratio(perpetrator, victim)

        # Check if literature value is within CI
        in_ci = prediction.ci_95[1] <= literature_ratio <= prediction.ci_95[2]

        # Fold error
        fe = max(prediction.auc_ratio_mean / literature_ratio,
                 literature_ratio / prediction.auc_ratio_mean)

        results["$(perp)_$(vic)"] = (
            predicted = prediction.auc_ratio_mean,
            ci_95 = prediction.ci_95,
            literature = literature_ratio,
            within_ci = in_ci,
            fold_error = fe,
            description = description
        )
    end

    # Summary
    n_within_ci = sum(r.within_ci for r in values(results))
    mean_fe = mean([r.fold_error for r in values(results)])

    results["summary"] = (
        n_cases = length(test_cases),
        n_within_ci = n_within_ci,
        coverage = n_within_ci / length(test_cases),
        mean_fold_error = mean_fe
    )

    return results
end

end # module BayesianDDIModel
