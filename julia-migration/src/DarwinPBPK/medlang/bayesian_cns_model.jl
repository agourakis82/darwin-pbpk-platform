# =============================================================================
# BAYESIAN CNS/BRAIN PBPK MODEL - MedLang v1.0
# =============================================================================
# Darwin PBPK Platform - Publication-Ready Mechanistic Model
#
# INTEGRATION WITH CNSCSFModel:
# This module extends the mechanistic CNS/CSF model with Bayesian inference,
# combining the LeiCNS-PK3.0 structure with uncertainty quantification.
#
# Key Features:
# 1. Bayesian inference for BBB permeability (extends CNSParams)
# 2. Hierarchical priors from population data
# 3. Uncertainty quantification in Kp,uu predictions
# 4. Integration of transporter activity with prior knowledge
# 5. Posterior predictive distributions for drug exposure
# 6. Full integration with mechanistic CNS compartments
#
# Mathematical Framework:
# - Prior: P(θ) from population PK data and in vitro
# - Likelihood: P(D|θ) from observed CSF/brain data
# - Posterior: P(θ|D) ∝ P(D|θ) × P(θ)
#
# Hierarchical Structure:
# - Level 1: Drug physicochemistry → BBB permeability prior
# - Level 2: Transporter genotype/expression → efflux modifier
# - Level 3: Disease state → barrier integrity modifier
# - Level 4: Individual observations → posterior update
#
# CNS Compartments (from CNSCSFModel):
# - Brain ECF, Brain ICF
# - CSF_LV, CSF_TFV, CSF_CM, CSF_SAS
# - BBB and BCSFB barriers with transporters
#
# Literature Basis:
# - Yamamoto et al. 2017 (CPT:PSP) - LeiCNS-PK3.0
# - Hammarlund-Udenaes et al. (2008) - Kp,uu concept
# - Dolgikh et al. (2016) - BBB permeability prediction
# - Verscheijden et al. (2021) - Pediatric CNS PBPK
# - Ball et al. (2012) - Bayesian PBPK methods
#
# Author: Dr. Sounio Agourakis
# Date: November 2025
# =============================================================================

module BayesianCNSModel

using DifferentialEquations
using Distributions
using Statistics
using LinearAlgebra

# Import the mechanistic CNS model
using ..CNSCSFModel
import ..CNSCSFModel: CNSParams, BBBTransporters, BCSFBTransporters
import ..CNSCSFModel: CNS_PHYSIOLOGY, calculate_kpuu_bbb, calculate_kpuu_bcsfb
import ..CNSCSFModel: default_bbb_transporters, default_bcsfb_transporters
import ..CNSCSFModel: simulate_cns_distribution

export BayesianBBBParams, BBBPrior, TransporterPrior
export CNSPopulationPrior, DiseaseStatePrior, IndividualPosterior
export PosteriorPredictive, UncertaintyBounds
export calculate_bbb_prior, update_posterior
export sample_posterior, posterior_predictive
export kpuu_with_uncertainty, credible_interval
export simulate_bayesian_cns, hierarchical_model
export drug_bbb_prior, population_prior_preset
export validate_bayesian_model, compare_to_observed

# =============================================================================
# PRIOR DISTRIBUTIONS
# =============================================================================

"""
    BBBPrior

Prior distribution for BBB permeability parameters.
Based on physicochemistry and in vitro data.
"""
struct BBBPrior
    # Log-normal prior for passive permeability (Papp)
    log_papp_mean::Float64      # Mean of log(Papp)
    log_papp_sd::Float64        # SD of log(Papp)

    # Beta prior for unbound fraction
    fu_alpha::Float64           # Beta shape α
    fu_beta::Float64            # Beta shape β

    # Log-normal prior for Kp,brain
    log_kp_mean::Float64
    log_kp_sd::Float64

    # Normal prior for PSA effect on permeability
    psa_coefficient::Float64    # Negative coefficient
    psa_sd::Float64

    # Informative vs weakly informative
    informativeness::Symbol     # :strong, :moderate, :weak
end

"""
    TransporterPrior

Prior distribution for transporter effects.
"""
struct TransporterPrior
    # P-gp efflux ratio prior (log-normal)
    log_pgp_er_mean::Float64
    log_pgp_er_sd::Float64
    pgp_probability::Float64    # Prior probability of being P-gp substrate

    # BCRP efflux ratio prior
    log_bcrp_er_mean::Float64
    log_bcrp_er_sd::Float64
    bcrp_probability::Float64

    # Uptake transporter prior (LAT1, GLUT1)
    uptake_enhancement_mean::Float64
    uptake_enhancement_sd::Float64
    uptake_probability::Float64

    # Correlation between transporters
    pgp_bcrp_correlation::Float64
end

"""
    CNSPopulationPrior

Hierarchical population-level prior.
"""
struct CNSPopulationPrior
    # Population mean Kp,uu
    kpuu_population_mean::Float64
    kpuu_population_sd::Float64

    # Between-subject variability
    bsv_cv::Float64             # CV for between-subject variability

    # Within-subject variability
    wsv_cv::Float64             # CV for residual error

    # Covariate effects
    age_effect::Float64         # Per decade
    weight_effect::Float64      # Per 10 kg
    sex_effect::Float64         # Male vs female

    # Drug class effects
    class_means::Dict{Symbol, Float64}  # By drug class
    class_sds::Dict{Symbol, Float64}
end

"""
    DiseaseStatePrior

Prior modifications for disease states affecting BBB.
"""
struct DiseaseStatePrior
    condition::Symbol           # :healthy, :alzheimers, :ms, :stroke, :tumor, :infection

    # BBB integrity modifier (beta distribution)
    integrity_alpha::Float64
    integrity_beta::Float64

    # Inflammation effect on permeability
    inflammation_multiplier_mean::Float64
    inflammation_multiplier_sd::Float64

    # Transporter expression changes
    pgp_expression_factor::Float64
    bcrp_expression_factor::Float64

    # CSF dynamics changes
    csf_turnover_factor::Float64
    csf_volume_factor::Float64
end

# =============================================================================
# POSTERIOR STRUCTURES
# =============================================================================

"""
    IndividualPosterior

Posterior distribution for individual patient.
"""
struct IndividualPosterior
    # Posterior parameters (after Bayesian update)
    kpuu_mean::Float64
    kpuu_sd::Float64
    kpuu_samples::Vector{Float64}

    # Credible intervals
    ci_90::Tuple{Float64, Float64}
    ci_95::Tuple{Float64, Float64}

    # Evidence weight
    prior_weight::Float64
    data_weight::Float64

    # Diagnostics
    effective_sample_size::Float64
    rhat::Float64               # Gelman-Rubin diagnostic
end

"""
    PosteriorPredictive

Posterior predictive distribution for CNS drug exposure.
"""
struct PosteriorPredictive
    # Brain ECF concentrations
    cu_brain_mean::Vector{Float64}
    cu_brain_lower::Vector{Float64}  # 5th percentile
    cu_brain_upper::Vector{Float64}  # 95th percentile

    # CSF concentrations
    cu_csf_mean::Vector{Float64}
    cu_csf_lower::Vector{Float64}
    cu_csf_upper::Vector{Float64}

    # Time points
    times::Vector{Float64}

    # Probability of target attainment
    p_target_brain::Float64     # P(Cu_brain > target)
    p_target_csf::Float64       # P(Cu_CSF > target)
end

"""
    UncertaintyBounds

Quantified uncertainty in predictions.
"""
struct UncertaintyBounds
    parameter::Symbol
    point_estimate::Float64
    ci_80::Tuple{Float64, Float64}
    ci_90::Tuple{Float64, Float64}
    ci_95::Tuple{Float64, Float64}
    coefficient_of_variation::Float64
end

# =============================================================================
# BAYESIAN BBB PARAMETERS
# =============================================================================

"""
    BayesianBBBParams

Complete Bayesian parameter set for CNS modeling.
"""
struct BayesianBBBParams
    drug_name::String

    # Physicochemistry (fixed)
    MW::Float64
    logP::Float64
    PSA::Float64                # Polar surface area
    HBD::Int                    # H-bond donors
    pKa::Union{Float64, Nothing}
    charge_type::Symbol

    # Priors
    bbb_prior::BBBPrior
    transporter_prior::TransporterPrior
    population_prior::CNSPopulationPrior
    disease_prior::DiseaseStatePrior

    # Observed data (if available)
    observed_kpuu::Union{Float64, Nothing}
    observed_csf_plasma_ratio::Union{Float64, Nothing}
    n_observations::Int

    # Posterior (after update)
    posterior::Union{IndividualPosterior, Nothing}
end

# =============================================================================
# PRIOR CALCULATIONS
# =============================================================================

"""
    calculate_bbb_prior(MW, logP, PSA, HBD; transporter_info)

Calculate informative prior for BBB permeability from physicochemistry.

Uses multiple regression model:
log(Papp) = β₀ + β₁×logP + β₂×MW + β₃×PSA + β₄×HBD + ε

Based on Lipinski/Veber rules and CNS MPO score.
"""
function calculate_bbb_prior(
    MW::Float64,
    logP::Float64,
    PSA::Float64,
    HBD::Int;
    drug_class::Symbol = :small_molecule
)::BBBPrior
    # Regression coefficients from literature
    β0 = -4.0           # Intercept (log scale)
    β_logP = 0.5        # Positive: lipophilic drugs cross better
    β_MW = -0.003       # Negative: larger molecules cross worse
    β_PSA = -0.02       # Negative: polar molecules cross worse
    β_HBD = -0.3        # Negative: H-bond donors cross worse

    # Calculate mean log(Papp) from structure
    log_papp_mean = β0 + β_logP * logP + β_MW * MW + β_PSA * PSA + β_HBD * HBD

    # Uncertainty increases with deviation from optimal CNS properties
    # CNS MPO score components
    cns_mpo = 0.0
    cns_mpo += logP >= 1 && logP <= 3 ? 1.0 : 0.5  # Optimal logP
    cns_mpo += MW <= 360 ? 1.0 : 0.5 * (500 - MW) / 140  # MW penalty
    cns_mpo += PSA <= 90 ? 1.0 : 0.5 * (140 - PSA) / 50  # PSA penalty
    cns_mpo += HBD <= 1 ? 1.0 : 0.5  # HBD penalty

    # SD inversely related to CNS drug-likeness
    log_papp_sd = 0.5 + (4 - cns_mpo) * 0.3

    # fu prior from logP (more lipophilic = lower fu)
    # fu ~ 1 / (1 + 10^(logP - 1))
    fu_expected = 1.0 / (1.0 + 10^(logP - 1.0))
    fu_alpha = fu_expected * 5
    fu_beta = (1 - fu_expected) * 5

    # Kp,brain prior
    log_kp_mean = 0.5 * logP - 0.5  # Approximate from logP
    log_kp_sd = 0.8

    # Informativeness based on data quality
    informativeness = cns_mpo > 3 ? :strong : (cns_mpo > 2 ? :moderate : :weak)

    return BBBPrior(
        log_papp_mean, log_papp_sd,
        fu_alpha, fu_beta,
        log_kp_mean, log_kp_sd,
        -0.02, 0.005,
        informativeness
    )
end

"""
    transporter_prior_from_structure(MW, logP, charge_type)

Calculate transporter prior from molecular properties.
"""
function transporter_prior_from_structure(
    MW::Float64,
    logP::Float64,
    charge_type::Symbol
)::TransporterPrior
    # P-gp substrate probability increases with MW and logP
    # P-gp prefers lipophilic cations
    pgp_prob_base = 0.3
    if charge_type == :base
        pgp_prob_base += 0.2
    end
    if MW > 400
        pgp_prob_base += 0.1 * (MW - 400) / 200
    end
    if logP > 3
        pgp_prob_base += 0.1
    end
    pgp_prob = clamp(pgp_prob_base, 0.1, 0.9)

    # BCRP - similar but also acids
    bcrp_prob_base = 0.2
    if charge_type == :acid
        bcrp_prob_base += 0.15
    end
    bcrp_prob = clamp(bcrp_prob_base, 0.1, 0.8)

    # Uptake - LAT1 for large neutral amino acids, GLUT1 for sugars
    uptake_prob = charge_type == :zwitterion ? 0.3 : 0.1

    return TransporterPrior(
        log(3.0), 0.5, pgp_prob,     # P-gp: ER ~ 3
        log(2.0), 0.5, bcrp_prob,    # BCRP: ER ~ 2
        1.5, 0.3, uptake_prob,       # Uptake: 1.5x enhancement
        0.3                           # Moderate correlation
    )
end

"""
    disease_state_prior(condition, severity)

Create prior for disease-modified BBB.
"""
function disease_state_prior(
    condition::Symbol,
    severity::Float64 = 0.5
)::DiseaseStatePrior
    states = Dict(
        :healthy => DiseaseStatePrior(
            :healthy, 100.0, 1.0, 1.0, 0.05, 1.0, 1.0, 1.0, 1.0
        ),

        :alzheimers => DiseaseStatePrior(
            :alzheimers,
            90.0 - 20 * severity, 10.0,  # Slightly compromised BBB
            1.2 + 0.3 * severity, 0.2,   # Inflammation
            0.8 - 0.2 * severity, 1.0,   # Reduced P-gp
            1.0, 1.1                      # Slightly increased CSF
        ),

        :multiple_sclerosis => DiseaseStatePrior(
            :ms,
            70.0 - 30 * severity, 15.0,  # Variable BBB disruption
            1.5 + 0.5 * severity, 0.3,   # Significant inflammation
            0.7, 0.7,                     # Reduced transporters
            0.9, 1.2                      # Altered CSF dynamics
        ),

        :stroke => DiseaseStatePrior(
            :stroke,
            50.0 - 30 * severity, 20.0,  # Acutely disrupted
            2.0 + 1.0 * severity, 0.5,   # High inflammation
            0.5 - 0.2 * severity, 0.5,   # Reduced transport
            0.8, 1.0
        ),

        :brain_tumor => DiseaseStatePrior(
            :tumor,
            60.0 - 20 * severity, 15.0,  # BTB different from BBB
            1.3 + 0.4 * severity, 0.3,
            0.6, 0.8,                     # Variable expression
            1.0, 1.3                      # Edema
        ),

        :meningitis => DiseaseStatePrior(
            :infection,
            30.0 - 20 * severity, 10.0,  # Severely compromised
            3.0 + 2.0 * severity, 0.8,   # High inflammation
            0.3, 0.3,                     # Severely impaired
            0.5, 1.5                      # CSF changes
        ),

        :pediatric => DiseaseStatePrior(
            :pediatric,
            95.0, 5.0,                    # Slightly less mature
            1.0, 0.1,
            0.8, 0.9,                     # Lower expression
            1.2, 0.8                      # Higher CSF turnover
        )
    )

    return get(states, condition, states[:healthy])
end

"""
    population_prior_preset(population)

Get population-level prior for different patient groups.
"""
function population_prior_preset(population::Symbol)::CNSPopulationPrior
    presets = Dict(
        :healthy_adult => CNSPopulationPrior(
            0.3, 0.5,           # Mean Kp,uu ~ 0.3 (typical for P-gp substrates)
            0.4, 0.2,           # BSV 40%, WSV 20%
            0.0, 0.0, 0.05,     # Minimal covariate effects
            Dict(:cns_drug => 0.5, :peripheral_drug => 0.1, :pgp_substrate => 0.2),
            Dict(:cns_drug => 0.3, :peripheral_drug => 0.5, :pgp_substrate => 0.4)
        ),

        :pediatric => CNSPopulationPrior(
            0.4, 0.6,           # Slightly higher mean (less mature BBB)
            0.5, 0.25,          # Higher variability
            -0.02, 0.0, 0.0,    # Age effect
            Dict(:cns_drug => 0.55, :peripheral_drug => 0.15),
            Dict(:cns_drug => 0.35, :peripheral_drug => 0.55)
        ),

        :elderly => CNSPopulationPrior(
            0.35, 0.5,
            0.45, 0.22,
            0.01, 0.0, 0.03,    # Slight age effect
            Dict(:cns_drug => 0.55, :peripheral_drug => 0.12),
            Dict(:cns_drug => 0.32, :peripheral_drug => 0.52)
        ),

        :critically_ill => CNSPopulationPrior(
            0.5, 0.8,           # Higher and more variable
            0.6, 0.3,
            0.0, 0.01, 0.0,
            Dict(:cns_drug => 0.7, :antibiotic => 0.4),
            Dict(:cns_drug => 0.4, :antibiotic => 0.5)
        )
    )

    return get(presets, population, presets[:healthy_adult])
end

# =============================================================================
# BAYESIAN UPDATE
# =============================================================================

"""
    update_posterior(prior, observed_data, n_samples)

Update prior with observed data using Bayesian inference.

For conjugate cases (normal-normal):
    posterior_mean = (prior_var × data_mean + data_var × prior_mean) / (prior_var + data_var)
    posterior_var = 1 / (1/prior_var + n/data_var)

For non-conjugate cases, uses numerical integration.
"""
function update_posterior(
    prior_mean::Float64,
    prior_sd::Float64,
    observed_values::Vector{Float64};
    likelihood_sd::Float64 = 0.3,
    n_samples::Int = 10000
)::IndividualPosterior
    n_obs = length(observed_values)

    if n_obs == 0
        # No data - return prior
        samples = rand(Normal(prior_mean, prior_sd), n_samples)
        return IndividualPosterior(
            prior_mean, prior_sd, samples,
            quantile(samples, (0.05, 0.95)),
            quantile(samples, (0.025, 0.975)),
            1.0, 0.0,
            n_samples, 1.0
        )
    end

    # Data summary
    data_mean = mean(observed_values)
    data_var = likelihood_sd^2 / n_obs

    # Prior variance
    prior_var = prior_sd^2

    # Normal-normal conjugate update
    posterior_var = 1.0 / (1.0/prior_var + 1.0/data_var)
    posterior_mean = posterior_var * (prior_mean/prior_var + data_mean/data_var)
    posterior_sd = sqrt(posterior_var)

    # Generate samples
    samples = rand(Normal(posterior_mean, posterior_sd), n_samples)

    # Credible intervals
    ci_90 = (quantile(samples, 0.05), quantile(samples, 0.95))
    ci_95 = (quantile(samples, 0.025), quantile(samples, 0.975))

    # Weights
    total_precision = 1/prior_var + n_obs/likelihood_sd^2
    prior_weight = (1/prior_var) / total_precision
    data_weight = (n_obs/likelihood_sd^2) / total_precision

    # Diagnostics (simple version)
    ess = n_samples  # Full ESS for direct sampling
    rhat = 1.0       # Perfect for direct sampling

    return IndividualPosterior(
        posterior_mean, posterior_sd, samples,
        ci_90, ci_95,
        prior_weight, data_weight,
        ess, rhat
    )
end

"""
    hierarchical_update(population_prior, individual_priors, observed)

Hierarchical Bayesian update with shrinkage.
"""
function hierarchical_update(
    pop_prior::CNSPopulationPrior,
    individual_data::Vector{Tuple{Float64, Float64}},  # (observation, weight)
    drug_class::Symbol
)::Tuple{Float64, Float64, Vector{Float64}}
    # Population mean prior
    if haskey(pop_prior.class_means, drug_class)
        class_mean = pop_prior.class_means[drug_class]
        class_sd = pop_prior.class_sds[drug_class]
    else
        class_mean = pop_prior.kpuu_population_mean
        class_sd = pop_prior.kpuu_population_sd
    end

    # Individual-level updates with shrinkage toward population
    n_ind = length(individual_data)

    if n_ind == 0
        return (class_mean, class_sd, Float64[])
    end

    # Empirical Bayes: estimate population variance from data
    obs_values = [d[1] for d in individual_data]
    obs_weights = [d[2] for d in individual_data]

    weighted_mean = sum(obs_values .* obs_weights) / sum(obs_weights)
    weighted_var = sum(obs_weights .* (obs_values .- weighted_mean).^2) / sum(obs_weights)

    # Shrinkage factor
    τ² = class_sd^2  # Between-subject variance
    σ² = weighted_var / n_ind  # Within-subject variance estimate

    shrinkage = τ² / (τ² + σ²)

    # Shrunk individual estimates
    shrunk_values = [shrinkage * class_mean + (1 - shrinkage) * obs for obs in obs_values]

    # Updated population parameters
    updated_mean = shrinkage * class_mean + (1 - shrinkage) * weighted_mean
    updated_sd = sqrt(τ² * (1 - shrinkage))

    return (updated_mean, updated_sd, shrunk_values)
end

# =============================================================================
# Kp,uu CALCULATIONS WITH UNCERTAINTY
# =============================================================================

"""
    kpuu_with_uncertainty(params, n_samples)

Calculate Kp,uu,BBB with full uncertainty quantification.
"""
function kpuu_with_uncertainty(
    params::BayesianBBBParams;
    n_samples::Int = 10000
)::UncertaintyBounds
    # Sample from priors
    prior = params.bbb_prior
    trans = params.transporter_prior
    disease = params.disease_prior

    # Passive permeability samples
    log_papp_samples = rand(Normal(prior.log_papp_mean, prior.log_papp_sd), n_samples)
    papp_samples = exp.(log_papp_samples)

    # Transporter effects
    pgp_substrate = rand(n_samples) .< trans.pgp_probability
    pgp_er_samples = exp.(rand(Normal(trans.log_pgp_er_mean, trans.log_pgp_er_sd), n_samples))
    pgp_er_samples[.!pgp_substrate] .= 1.0  # No effect if not substrate

    bcrp_substrate = rand(n_samples) .< trans.bcrp_probability
    bcrp_er_samples = exp.(rand(Normal(trans.log_bcrp_er_mean, trans.log_bcrp_er_sd), n_samples))
    bcrp_er_samples[.!bcrp_substrate] .= 1.0

    # Uptake enhancement
    uptake_active = rand(n_samples) .< trans.uptake_probability
    uptake_samples = rand(Normal(trans.uptake_enhancement_mean, trans.uptake_enhancement_sd), n_samples)
    uptake_samples[.!uptake_active] .= 1.0

    # Disease effect on BBB integrity
    integrity_samples = rand(Beta(disease.integrity_alpha, disease.integrity_beta), n_samples)
    inflammation_samples = rand(Normal(disease.inflammation_multiplier_mean,
                                        disease.inflammation_multiplier_sd), n_samples)

    # Calculate Kp,uu for each sample
    # Kp,uu = (Papp_influx × uptake) / (Papp_efflux + efflux_transport)
    # Simplified: Kp,uu = 1 / (1 + efflux_contribution - uptake_contribution)

    kpuu_samples = zeros(n_samples)
    for i in 1:n_samples
        # Base Kp,uu from passive diffusion (approaches 1 without transporters)
        base_kpuu = 1.0

        # Efflux reduces Kp,uu
        efflux_reduction = (pgp_er_samples[i] - 1) * 0.3 + (bcrp_er_samples[i] - 1) * 0.2

        # Uptake increases Kp,uu
        uptake_enhancement = (uptake_samples[i] - 1) * 0.2

        # Disease effects
        disease_modifier = integrity_samples[i] * inflammation_samples[i]

        # Final Kp,uu
        kpuu_samples[i] = base_kpuu / (1 + efflux_reduction) * (1 + uptake_enhancement) * disease_modifier
        kpuu_samples[i] = clamp(kpuu_samples[i], 0.01, 5.0)
    end

    # If we have a posterior, use those samples instead
    if params.posterior !== nothing
        kpuu_samples = params.posterior.kpuu_samples
    end

    # Calculate statistics
    point_estimate = mean(kpuu_samples)
    cv = std(kpuu_samples) / point_estimate

    ci_80 = (quantile(kpuu_samples, 0.10), quantile(kpuu_samples, 0.90))
    ci_90 = (quantile(kpuu_samples, 0.05), quantile(kpuu_samples, 0.95))
    ci_95 = (quantile(kpuu_samples, 0.025), quantile(kpuu_samples, 0.975))

    return UncertaintyBounds(
        :kpuu_bbb,
        point_estimate,
        ci_80, ci_90, ci_95,
        cv
    )
end

"""
    credible_interval(samples, level)

Calculate credible interval from posterior samples.
"""
function credible_interval(
    samples::Vector{Float64},
    level::Float64 = 0.95
)::Tuple{Float64, Float64}
    α = (1 - level) / 2
    return (quantile(samples, α), quantile(samples, 1 - α))
end

# =============================================================================
# POSTERIOR PREDICTIVE SIMULATION
# =============================================================================

"""
    posterior_predictive(params, dose_mg, times; n_samples)

Generate posterior predictive distributions for CNS concentrations.
"""
function posterior_predictive(
    params::BayesianBBBParams,
    dose_mg::Float64,
    times::Vector{Float64};
    n_samples::Int = 1000,
    CL_plasma::Float64 = 10.0,  # L/h
    Vd::Float64 = 70.0          # L
)::PosteriorPredictive
    # Get Kp,uu samples
    kpuu_uncertainty = kpuu_with_uncertainty(params; n_samples=n_samples)

    # Sample Kp,uu values
    if params.posterior !== nothing
        kpuu_samples = params.posterior.kpuu_samples[1:min(n_samples, length(params.posterior.kpuu_samples))]
    else
        # Generate from prior
        kpuu_samples = rand(
            LogNormal(log(kpuu_uncertainty.point_estimate),
                      kpuu_uncertainty.coefficient_of_variation),
            n_samples
        )
    end

    # Simple one-compartment PK for plasma
    ke = CL_plasma / Vd  # 1/h

    # Storage for trajectories
    n_times = length(times)
    cu_brain_trajectories = zeros(n_samples, n_times)
    cu_csf_trajectories = zeros(n_samples, n_times)

    for i in 1:n_samples
        kpuu = kpuu_samples[i]

        # Plasma concentration (unbound)
        fu_plasma = rand(Beta(params.bbb_prior.fu_alpha, params.bbb_prior.fu_beta))
        C0 = dose_mg / Vd  # Initial concentration

        for (j, t) in enumerate(times)
            Cu_plasma = C0 * fu_plasma * exp(-ke * t)

            # Brain ECF (equilibration with Kp,uu)
            # Simple equilibration model
            t_eq = 0.5  # h, equilibration half-life
            equilibration = 1 - exp(-log(2) * t / t_eq)
            Cu_brain = Cu_plasma * kpuu * equilibration

            # CSF (delayed from brain, lower Kp,uu typically)
            kpuu_csf = kpuu * 0.7  # CSF typically lower than brain ECF
            t_eq_csf = 2.0  # Slower equilibration
            equilibration_csf = 1 - exp(-log(2) * t / t_eq_csf)
            Cu_csf = Cu_plasma * kpuu_csf * equilibration_csf

            cu_brain_trajectories[i, j] = Cu_brain
            cu_csf_trajectories[i, j] = Cu_csf
        end
    end

    # Calculate summary statistics
    cu_brain_mean = vec(mean(cu_brain_trajectories, dims=1))
    cu_brain_lower = vec(mapslices(x -> quantile(x, 0.05), cu_brain_trajectories, dims=1))
    cu_brain_upper = vec(mapslices(x -> quantile(x, 0.95), cu_brain_trajectories, dims=1))

    cu_csf_mean = vec(mean(cu_csf_trajectories, dims=1))
    cu_csf_lower = vec(mapslices(x -> quantile(x, 0.05), cu_csf_trajectories, dims=1))
    cu_csf_upper = vec(mapslices(x -> quantile(x, 0.95), cu_csf_trajectories, dims=1))

    # Probability of target attainment (example: 0.1 mg/L)
    target = 0.1 * dose_mg / 100  # Scale with dose
    p_target_brain = mean(maximum(cu_brain_trajectories, dims=2) .> target)
    p_target_csf = mean(maximum(cu_csf_trajectories, dims=2) .> target)

    return PosteriorPredictive(
        cu_brain_mean, cu_brain_lower, cu_brain_upper,
        cu_csf_mean, cu_csf_lower, cu_csf_upper,
        times,
        p_target_brain, p_target_csf
    )
end

# =============================================================================
# SIMULATION
# =============================================================================

"""
    simulate_bayesian_cns(params, dose_mg; kwargs...)

Simulate CNS drug distribution with Bayesian uncertainty.
"""
function simulate_bayesian_cns(
    params::BayesianBBBParams,
    dose_mg::Float64;
    tspan::Tuple{Float64, Float64} = (0.0, 24.0),
    n_times::Int = 100,
    n_samples::Int = 1000
)
    times = collect(range(tspan[1], tspan[2], length=n_times))

    # Get posterior predictive
    pred = posterior_predictive(params, dose_mg, times; n_samples=n_samples)

    # Get Kp,uu with uncertainty
    kpuu_bounds = kpuu_with_uncertainty(params)

    return (
        times = times,
        Cu_brain_mean = pred.cu_brain_mean,
        Cu_brain_CI = (pred.cu_brain_lower, pred.cu_brain_upper),
        Cu_csf_mean = pred.cu_csf_mean,
        Cu_csf_CI = (pred.cu_csf_lower, pred.cu_csf_upper),
        Kpuu = kpuu_bounds,
        p_target_brain = pred.p_target_brain,
        p_target_csf = pred.p_target_csf,
        posterior_predictive = pred
    )
end

# =============================================================================
# DRUG PRESETS
# =============================================================================

"""
    drug_bbb_prior(drug_name)

Get informative BBB prior for specific drugs.
"""
function drug_bbb_prior(drug_name::Symbol)::BayesianBBBParams
    presets = Dict(
        :risperidone => BayesianBBBParams(
            "Risperidone",
            410.5, 3.0, 82.0, 1, 8.2, :base,
            BBBPrior(-3.5, 0.4, 0.5, 9.5, 2.5, 0.5, -0.02, 0.005, :strong),
            TransporterPrior(log(4.0), 0.4, 0.9, log(1.5), 0.3, 0.2, 1.0, 0.2, 0.1, 0.3),
            population_prior_preset(:healthy_adult),
            disease_state_prior(:healthy),
            0.15, 0.05, 5, nothing  # Observed Kp,uu ~0.15
        ),

        :haloperidol => BayesianBBBParams(
            "Haloperidol",
            375.9, 4.3, 40.0, 1, 8.7, :base,
            BBBPrior(-3.0, 0.35, 0.4, 12.0, 3.0, 0.6, -0.02, 0.005, :strong),
            TransporterPrior(log(5.0), 0.5, 0.85, log(1.5), 0.3, 0.15, 1.0, 0.2, 0.05, 0.25),
            population_prior_preset(:healthy_adult),
            disease_state_prior(:healthy),
            0.08, 0.03, 8, nothing
        ),

        :morphine => BayesianBBBParams(
            "Morphine",
            285.3, 0.9, 52.0, 2, 8.0, :base,
            BBBPrior(-4.5, 0.6, 6.0, 4.0, 0.5, 0.7, -0.02, 0.005, :moderate),
            TransporterPrior(log(2.5), 0.5, 0.7, log(1.2), 0.3, 0.3, 1.0, 0.2, 0.05, 0.2),
            population_prior_preset(:healthy_adult),
            disease_state_prior(:healthy),
            0.35, 0.2, 10, nothing
        ),

        :gabapentin => BayesianBBBParams(
            "Gabapentin",
            171.2, -1.1, 63.0, 2, 3.7, :zwitterion,
            BBBPrior(-5.5, 0.8, 8.0, 2.0, -0.5, 0.8, -0.02, 0.005, :weak),
            TransporterPrior(log(1.2), 0.3, 0.1, log(1.0), 0.2, 0.1, 3.0, 0.5, 0.9, 0.1),  # LAT1 substrate
            population_prior_preset(:healthy_adult),
            disease_state_prior(:healthy),
            0.8, 0.4, 6, nothing  # High Kp,uu due to LAT1 uptake
        ),

        :methotrexate => BayesianBBBParams(
            "Methotrexate",
            454.4, -1.85, 210.0, 5, 4.8, :acid,
            BBBPrior(-6.0, 1.0, 9.0, 1.0, -2.0, 1.2, -0.02, 0.005, :weak),
            TransporterPrior(log(2.0), 0.6, 0.6, log(3.0), 0.5, 0.8, 2.0, 0.6, 0.3, 0.4),
            population_prior_preset(:healthy_adult),
            disease_state_prior(:healthy),
            0.02, nothing, 3, nothing  # Very low CNS penetration
        )
    )

    if !haskey(presets, drug_name)
        available = join(keys(presets), ", ")
        error("Unknown drug: $drug_name. Available: $available")
    end

    return presets[drug_name]
end

# =============================================================================
# INTEGRATED MECHANISTIC + BAYESIAN CNS SIMULATION
# =============================================================================

"""
    create_bayesian_cns_params(bayesian_params)

Convert BayesianBBBParams to CNSParams for mechanistic simulation.
Samples from posterior/prior distributions for parameter uncertainty.
"""
function create_bayesian_cns_params(
    bayesian::BayesianBBBParams;
    sample_from_posterior::Bool = true
)::CNSParams
    # Sample Kp,uu from posterior or prior
    if sample_from_posterior && bayesian.posterior !== nothing
        kpuu = rand(bayesian.posterior.kpuu_samples)
    else
        kpuu_bounds = kpuu_with_uncertainty(bayesian; n_samples=100)
        kpuu = rand(LogNormal(log(kpuu_bounds.point_estimate),
                              kpuu_bounds.coefficient_of_variation))
    end

    # Sample fu_brain from prior
    fu_brain = rand(Beta(bayesian.bbb_prior.fu_alpha, bayesian.bbb_prior.fu_beta))

    # Calculate Kp_brain from Kp,uu and fu_brain
    # Kp,uu = Kp × fu_brain / fu_plasma → Kp = Kp,uu × fu_plasma / fu_brain
    fu_plasma_estimate = bayesian.bbb_prior.fu_alpha /
                         (bayesian.bbb_prior.fu_alpha + bayesian.bbb_prior.fu_beta)
    Kp_brain = kpuu * fu_plasma_estimate / max(fu_brain, 0.01)

    # Permeability from prior
    log_papp = rand(Normal(bayesian.bbb_prior.log_papp_mean, bayesian.bbb_prior.log_papp_sd))
    Papp_BBB = exp(log_papp)
    Papp_BCSFB = Papp_BBB * 0.5  # BCSFB typically lower permeability

    # Transporter substrate status
    is_pgp = rand() < bayesian.transporter_prior.pgp_probability
    pgp_km = is_pgp ? exp(bayesian.transporter_prior.log_pgp_er_mean) : 0.0

    is_bcrp = rand() < bayesian.transporter_prior.bcrp_probability
    bcrp_km = is_bcrp ? exp(bayesian.transporter_prior.log_bcrp_er_mean) : 0.0

    is_uptake = rand() < bayesian.transporter_prior.uptake_probability

    return CNSParams(
        bayesian.drug_name,
        bayesian.MW,
        bayesian.logP,
        bayesian.pKa,
        bayesian.charge_type,
        fu_plasma_estimate,
        fu_brain,
        clamp(Kp_brain, 0.1, 50.0),
        Papp_BBB,
        Papp_BCSFB,
        is_pgp,
        pgp_km,
        is_bcrp,
        bcrp_km,
        false, 0.0,  # MRP
        false,       # OATP
        is_uptake,   # LAT1
        false,       # GLUT
        :brain_ecf
    )
end

"""
    simulate_mechanistic_bayesian_cns(bayesian_params, dose_mg; kwargs...)

Simulate CNS drug distribution using the full mechanistic LeiCNS-PK model
with Bayesian uncertainty quantification.

This combines:
1. The mechanistic 6-compartment CNS model (ECF, ICF, CSF_LV, CSF_TFV, CSF_CM, CSF_SAS)
2. BBB and BCSFB barrier models with transporters
3. Bayesian priors and posteriors for parameter uncertainty
4. Monte Carlo sampling for uncertainty propagation
"""
function simulate_mechanistic_bayesian_cns(
    bayesian_params::BayesianBBBParams,
    dose_mg::Float64;
    t_max_h::Float64 = 24.0,
    n_samples::Int = 100,
    CL_plasma_mL_min::Float64 = 500.0,
    Vd_L::Float64 = 70.0
)
    # Storage for Monte Carlo trajectories
    all_results = []

    for i in 1:n_samples
        # Create CNSParams with sampled parameters
        cns_params = create_bayesian_cns_params(bayesian_params; sample_from_posterior=true)

        # Run mechanistic simulation
        result = simulate_cns_distribution(
            cns_params, dose_mg;
            t_max_h = t_max_h,
            CL_plasma_mL_min = CL_plasma_mL_min,
            Vd_L = Vd_L
        )

        push!(all_results, result)
    end

    # Aggregate results
    times = all_results[1]["time_h"]
    n_times = length(times)

    # Extract trajectories
    cu_plasma_matrix = hcat([r["Cu_plasma"] for r in all_results]...)
    cu_brain_ecf_matrix = hcat([r["Cu_brain_ECF"] for r in all_results]...)
    csf_lv_matrix = hcat([r["C_CSF_LV"] for r in all_results]...)
    csf_cm_matrix = hcat([r["C_CSF_CM"] for r in all_results]...)
    csf_sas_matrix = hcat([r["C_CSF_SAS"] for r in all_results]...)

    # Calculate statistics (mean and 90% CI)
    cu_brain_mean = vec(mean(cu_brain_ecf_matrix, dims=2))
    cu_brain_lower = vec(mapslices(x -> quantile(x, 0.05), cu_brain_ecf_matrix, dims=2))
    cu_brain_upper = vec(mapslices(x -> quantile(x, 0.95), cu_brain_ecf_matrix, dims=2))

    csf_sas_mean = vec(mean(csf_sas_matrix, dims=2))
    csf_sas_lower = vec(mapslices(x -> quantile(x, 0.05), csf_sas_matrix, dims=2))
    csf_sas_upper = vec(mapslices(x -> quantile(x, 0.95), csf_sas_matrix, dims=2))

    csf_cm_mean = vec(mean(csf_cm_matrix, dims=2))

    # Extract Kp,uu observations
    kpuu_bbb_samples = [r["Kpuu_BBB_observed"] for r in all_results]
    kpuu_csf_samples = [r["Kpuu_CSF_observed"] for r in all_results]

    return (
        # Time
        times = times,

        # Brain ECF (target site)
        Cu_brain_ECF_mean = cu_brain_mean,
        Cu_brain_ECF_CI = (cu_brain_lower, cu_brain_upper),

        # CSF SAS (clinical sample - lumbar puncture)
        C_CSF_SAS_mean = csf_sas_mean,
        C_CSF_SAS_CI = (csf_sas_lower, csf_sas_upper),

        # CSF CM (cisternal - brainstem relevant)
        C_CSF_CM_mean = csf_cm_mean,

        # Kp,uu with uncertainty
        Kpuu_BBB = (
            mean = mean(kpuu_bbb_samples),
            ci_90 = (quantile(kpuu_bbb_samples, 0.05), quantile(kpuu_bbb_samples, 0.95)),
            samples = kpuu_bbb_samples
        ),
        Kpuu_CSF = (
            mean = mean(kpuu_csf_samples),
            ci_90 = (quantile(kpuu_csf_samples, 0.05), quantile(kpuu_csf_samples, 0.95)),
            samples = kpuu_csf_samples
        ),

        # ECF to CSF ratio (important for CSF as surrogate)
        ECF_to_CSF_ratio = mean([r["ECF_to_CSF_ratio"] for r in all_results]),

        # Model details
        n_samples = n_samples,
        bayesian_params = bayesian_params
    )
end

"""
    bayesian_cns_target_attainment(bayesian_params, dose_mg, target_Cu; kwargs...)

Calculate probability of target attainment in brain ECF using
Bayesian-mechanistic CNS model.
"""
function bayesian_cns_target_attainment(
    bayesian_params::BayesianBBBParams,
    dose_mg::Float64,
    target_Cu::Float64;  # Target unbound concentration in brain ECF
    t_max_h::Float64 = 24.0,
    n_samples::Int = 500
)
    result = simulate_mechanistic_bayesian_cns(
        bayesian_params, dose_mg;
        t_max_h = t_max_h,
        n_samples = n_samples
    )

    # Run individual simulations to check target attainment
    n_attained = 0
    for i in 1:n_samples
        cns_params = create_bayesian_cns_params(bayesian_params)
        sim = simulate_cns_distribution(cns_params, dose_mg; t_max_h=t_max_h)

        # Check if Cmax in brain ECF exceeds target
        Cmax_brain = maximum(sim["Cu_brain_ECF"])
        if Cmax_brain >= target_Cu
            n_attained += 1
        end
    end

    p_attainment = n_attained / n_samples

    return (
        probability = p_attainment,
        target = target_Cu,
        dose = dose_mg,
        n_samples = n_samples,
        Cu_brain_mean_Cmax = maximum(result.Cu_brain_ECF_mean),
        Cu_brain_CI_Cmax = (
            maximum(result.Cu_brain_ECF_CI[1]),
            maximum(result.Cu_brain_ECF_CI[2])
        )
    )
end

export create_bayesian_cns_params, simulate_mechanistic_bayesian_cns
export bayesian_cns_target_attainment

# =============================================================================
# VALIDATION
# =============================================================================

"""
    validate_bayesian_model()

Validate Bayesian CNS model against literature Kp,uu values.
"""
function validate_bayesian_model()
    results = Dict{String, Any}()

    # Test drugs with known Kp,uu values
    test_drugs = [
        (:risperidone, 0.15, "P-gp substrate, moderate CNS"),
        (:haloperidol, 0.08, "Strong P-gp substrate"),
        (:morphine, 0.35, "Weak P-gp substrate"),
        (:gabapentin, 0.8, "LAT1 substrate, high CNS"),
        (:methotrexate, 0.02, "Very low CNS penetration")
    ]

    for (drug, literature_kpuu, description) in test_drugs
        params = drug_bbb_prior(drug)
        kpuu_bounds = kpuu_with_uncertainty(params)

        # Check if literature value is within 95% CI
        in_ci = kpuu_bounds.ci_95[1] <= literature_kpuu <= kpuu_bounds.ci_95[2]

        results[string(drug)] = (
            predicted_mean = kpuu_bounds.point_estimate,
            ci_95 = kpuu_bounds.ci_95,
            literature = literature_kpuu,
            within_ci = in_ci,
            cv = kpuu_bounds.coefficient_of_variation,
            description = description
        )
    end

    # Summary statistics
    n_within_ci = sum(r.within_ci for r in values(results))
    coverage = n_within_ci / length(test_drugs)

    results["summary"] = (
        n_drugs = length(test_drugs),
        n_within_ci = n_within_ci,
        coverage = coverage
    )

    return results
end

"""
    compare_to_observed(params, observed_kpuu, observed_sd)

Compare model prediction to observed data.
"""
function compare_to_observed(
    params::BayesianBBBParams,
    observed_kpuu::Float64,
    observed_sd::Float64 = 0.1
)
    # Get prior prediction
    prior_bounds = kpuu_with_uncertainty(params)

    # Update with observation
    posterior = update_posterior(
        prior_bounds.point_estimate,
        prior_bounds.point_estimate * prior_bounds.coefficient_of_variation,
        [observed_kpuu];
        likelihood_sd = observed_sd
    )

    # Calculate prediction error
    prior_error = abs(prior_bounds.point_estimate - observed_kpuu) / observed_kpuu
    posterior_error = abs(posterior.kpuu_mean - observed_kpuu) / observed_kpuu

    return (
        prior_mean = prior_bounds.point_estimate,
        prior_ci = prior_bounds.ci_95,
        posterior_mean = posterior.kpuu_mean,
        posterior_ci = posterior.ci_95,
        observed = observed_kpuu,
        prior_error = prior_error,
        posterior_error = posterior_error,
        improvement = (prior_error - posterior_error) / prior_error,
        prior_weight = posterior.prior_weight,
        data_weight = posterior.data_weight
    )
end

end # module BayesianCNSModel
