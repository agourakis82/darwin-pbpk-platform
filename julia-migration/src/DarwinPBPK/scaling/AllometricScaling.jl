# ===========================================================================
# MULTI-SPECIES ALLOMETRIC SCALING MODULE
# ===========================================================================
# Comprehensive interspecies scaling for PK parameter prediction.
#
# Supports:
# - Simple allometry (power-law scaling)
# - Rule of exponents (Mahmood-Balian)
# - Brain weight correction (Boxenbaum)
# - Maximum lifespan potential (MLP)
# - Physiologically-based allometry (PBPK-informed)
# - IVIVE (In Vitro to In Vivo Extrapolation)
#
# Species: Mouse, Rat, Rabbit, Dog, Monkey, Minipig, Human
#
# References:
# - Mahmood & Balian (1996) J Pharm Sci 85:411-414
# - Boxenbaum (1982) J Pharmacokinet Biopharm 10:201-227
# - Hosea et al. (2009) JPKPD 36:1-19
# - Tang & Mayersohn (2006) Clin Pharmacokinet 45:1087-1107
#
# Author: Dr. Sounio Agourakis
# Date: December 2025
# Version: 1.0.0
# ===========================================================================

module AllometricScaling

using Statistics
using LinearAlgebra

export Species, SpeciesData, AllometricModel, ScalingResult
export MOUSE, RAT, RABBIT, DOG, CYNOMOLGUS_MONKEY, RHESUS_MONKEY, MINIPIG, HUMAN
export get_species_data, scale_clearance, scale_volume, scale_half_life
export simple_allometry, rule_of_exponents, brain_weight_correction
export mlp_correction, dedrick_plot, plot_allometry_data
export predict_human_pk, predict_first_in_human_dose
export ivive_clearance, hepatocyte_scaling, microsomal_scaling
export calculate_allometric_exponent, fit_allometric_model
export SPECIES_DATABASE, STANDARD_EXPONENTS

# ===========================================================================
# Species Data
# ===========================================================================

"""
Species identifier enum.
"""
@enum Species begin
    MOUSE
    RAT
    RABBIT
    DOG
    CYNOMOLGUS_MONKEY
    RHESUS_MONKEY
    MINIPIG
    HUMAN
end

"""
Physiological data for a species.
"""
struct SpeciesData
    name::String
    body_weight::Float64          # kg
    brain_weight::Float64         # g
    liver_weight::Float64         # g
    kidney_weight::Float64        # g
    heart_weight::Float64         # g
    blood_volume::Float64         # mL
    cardiac_output::Float64       # mL/min
    hepatic_blood_flow::Float64   # mL/min
    renal_blood_flow::Float64     # mL/min
    gfr::Float64                  # mL/min (glomerular filtration rate)
    max_lifespan::Float64         # years (MLP)
    hepatocytes_per_gram::Float64 # cells/g liver
    microsomal_protein::Float64   # mg/g liver
end

"""
Standard species physiological database.
Based on Brown et al. (1997), Davies & Morris (1993), and recent literature.
"""
const SPECIES_DATABASE = Dict{Species, SpeciesData}(
    MOUSE => SpeciesData(
        "Mouse", 0.025, 0.4, 1.5, 0.35, 0.12,
        1.7, 12.0, 1.8, 1.3, 0.28,
        4.0, 135e6, 45.0
    ),
    RAT => SpeciesData(
        "Rat", 0.25, 1.8, 10.0, 2.0, 0.8,
        16.0, 74.0, 13.8, 9.2, 1.5,
        5.0, 120e6, 45.0
    ),
    RABBIT => SpeciesData(
        "Rabbit", 3.5, 9.0, 95.0, 18.0, 8.0,
        165.0, 340.0, 77.0, 45.0, 7.8,
        13.0, 125e6, 35.0
    ),
    DOG => SpeciesData(
        "Dog", 10.0, 85.0, 320.0, 60.0, 120.0,
        850.0, 1500.0, 400.0, 260.0, 60.0,
        20.0, 120e6, 32.0
    ),
    CYNOMOLGUS_MONKEY => SpeciesData(
        "Cynomolgus Monkey", 5.0, 65.0, 120.0, 28.0, 25.0,
        300.0, 600.0, 150.0, 80.0, 15.0,
        35.0, 120e6, 35.0
    ),
    RHESUS_MONKEY => SpeciesData(
        "Rhesus Monkey", 8.0, 95.0, 180.0, 42.0, 35.0,
        450.0, 850.0, 200.0, 110.0, 22.0,
        40.0, 120e6, 35.0
    ),
    MINIPIG => SpeciesData(
        "Minipig", 25.0, 75.0, 550.0, 100.0, 100.0,
        1600.0, 2200.0, 450.0, 350.0, 75.0,
        25.0, 100e6, 30.0
    ),
    HUMAN => SpeciesData(
        "Human", 70.0, 1400.0, 1800.0, 310.0, 330.0,
        5000.0, 5600.0, 1450.0, 1200.0, 120.0,
        122.0, 120e6, 40.0
    )
)

"""
Standard allometric exponents for different PK parameters.
"""
const STANDARD_EXPONENTS = Dict{String, Float64}(
    "clearance" => 0.75,           # CL ~ BW^0.75
    "volume" => 1.0,               # Vd ~ BW^1.0
    "half_life" => 0.25,           # t1/2 ~ BW^0.25
    "renal_clearance" => 0.75,
    "hepatic_clearance" => 0.75,
    "cardiac_output" => 0.75,
    "blood_flow" => 0.75,
    "gfr" => 0.75,
    "absorption_rate" => -0.25,    # ka ~ BW^-0.25
)

"""
Get species data from database.
"""
function get_species_data(species::Species)::SpeciesData
    return SPECIES_DATABASE[species]
end

# ===========================================================================
# Allometric Models
# ===========================================================================

"""
Allometric model fit result.
"""
struct AllometricModel
    parameter::String
    coefficient::Float64          # 'a' in Y = a * BW^b
    exponent::Float64             # 'b' in Y = a * BW^b
    r_squared::Float64
    species_used::Vector{Species}
    values_used::Vector{Float64}
    weights_used::Vector{Float64}
end

"""
Scaling result with confidence interval.
"""
struct ScalingResult
    parameter::String
    predicted_value::Float64
    lower_95ci::Float64
    upper_95ci::Float64
    method::String
    species_source::Vector{Species}
end

# ===========================================================================
# Simple Allometry
# ===========================================================================

"""
    simple_allometry(param_values, body_weights) -> AllometricModel

Fit simple allometric equation: Y = a × BW^b

Uses log-linear regression: log(Y) = log(a) + b × log(BW)
"""
function simple_allometry(
    param_values::Vector{Float64},
    body_weights::Vector{Float64};
    parameter_name::String = "parameter",
    species::Vector{Species} = Species[]
)::AllometricModel
    n = length(param_values)
    @assert n == length(body_weights) "Values and weights must have same length"
    @assert n >= 2 "Need at least 2 species for allometric fit"

    # Log-transform
    log_y = log.(param_values)
    log_bw = log.(body_weights)

    # Linear regression
    X = hcat(ones(n), log_bw)
    beta = X \ log_y

    log_a = beta[1]
    b = beta[2]
    a = exp(log_a)

    # R-squared
    y_pred = X * beta
    ss_res = sum((log_y .- y_pred).^2)
    ss_tot = sum((log_y .- mean(log_y)).^2)
    r2 = 1 - ss_res / ss_tot

    return AllometricModel(
        parameter_name, a, b, r2,
        species, param_values, body_weights
    )
end

"""
    calculate_allometric_exponent(species_data, parameter) -> Float64

Calculate allometric exponent from multi-species data.
"""
function calculate_allometric_exponent(
    species_list::Vector{Species},
    param_values::Vector{Float64}
)::Float64
    body_weights = [SPECIES_DATABASE[s].body_weight for s in species_list]
    model = simple_allometry(param_values, body_weights)
    return model.exponent
end

# ===========================================================================
# Advanced Scaling Methods
# ===========================================================================

"""
    rule_of_exponents(model, target_bw; correction=:auto) -> Float64

Apply rule of exponents correction (Mahmood & Balian 1996).

- b ≤ 0.55: No correction needed
- 0.55 < b ≤ 0.70: Multiply by MLP
- 0.70 < b ≤ 1.0: Multiply by BrW
- b > 1.0: Vertical allometry (use fixed exponent)
"""
function rule_of_exponents(
    model::AllometricModel,
    target_bw::Float64;
    target_brain_weight::Float64 = 1400.0,  # Human default
    target_mlp::Float64 = 122.0,             # Human MLP years
    correction::Symbol = :auto
)::ScalingResult
    b = model.exponent
    a = model.coefficient

    # Basic prediction
    pred = a * target_bw^b

    # Apply correction based on exponent
    method = "simple_allometry"
    if correction == :auto
        if b <= 0.55
            # No correction needed
            method = "simple_allometry (b ≤ 0.55)"
        elseif b <= 0.70
            # MLP correction
            pred = pred * target_mlp
            method = "MLP_correction (0.55 < b ≤ 0.70)"
        elseif b <= 1.0
            # Brain weight correction
            pred = pred * target_brain_weight
            method = "brain_weight_correction (0.70 < b ≤ 1.0)"
        else
            # Vertical allometry - use fixed exponent 0.75
            pred = a * target_bw^0.75
            method = "vertical_allometry_fixed (b > 1.0)"
        end
    elseif correction == :mlp
        pred = pred * target_mlp
        method = "MLP_correction (forced)"
    elseif correction == :brain
        pred = pred * target_brain_weight
        method = "brain_weight_correction (forced)"
    end

    # Confidence interval (approximate using CV from fit)
    cv = 0.3  # Typical 30% CV for allometric predictions
    lower = pred * (1 - 1.96 * cv)
    upper = pred * (1 + 1.96 * cv)

    return ScalingResult(
        model.parameter, pred, lower, upper,
        method, model.species_used
    )
end

"""
    brain_weight_correction(cl_values, body_weights, brain_weights; target_bw=70.0) -> Float64

Boxenbaum brain weight correction method.
CL_human = a × BW_human^b × BrW_human
"""
function brain_weight_correction(
    cl_values::Vector{Float64},
    body_weights::Vector{Float64},
    brain_weights::Vector{Float64};
    target_bw::Float64 = 70.0,
    target_brw::Float64 = 1400.0
)::ScalingResult
    n = length(cl_values)

    # Fit: log(CL) = log(a) + b1*log(BW) + b2*log(BrW)
    log_cl = log.(cl_values)
    X = hcat(ones(n), log.(body_weights), log.(brain_weights))
    beta = X \ log_cl

    a = exp(beta[1])
    b1 = beta[2]
    b2 = beta[3]

    pred = a * target_bw^b1 * target_brw^b2

    # Confidence interval
    cv = 0.35
    lower = pred * (1 - 1.96 * cv)
    upper = pred * (1 + 1.96 * cv)

    return ScalingResult(
        "clearance", pred, lower, upper,
        "brain_weight_correction", Species[]
    )
end

"""
    mlp_correction(cl_values, body_weights, mlp_values; target_bw=70.0) -> Float64

Maximum lifespan potential correction.
CL_human = a × BW_human^b × MLP_human
"""
function mlp_correction(
    cl_values::Vector{Float64},
    body_weights::Vector{Float64},
    mlp_values::Vector{Float64};
    target_bw::Float64 = 70.0,
    target_mlp::Float64 = 122.0
)::ScalingResult
    n = length(cl_values)

    # Fit: log(CL) = log(a) + b1*log(BW) + b2*log(MLP)
    log_cl = log.(cl_values)
    X = hcat(ones(n), log.(body_weights), log.(mlp_values))
    beta = X \ log_cl

    a = exp(beta[1])
    b1 = beta[2]
    b2 = beta[3]

    pred = a * target_bw^b1 * target_mlp^b2

    # Confidence interval
    cv = 0.35
    lower = pred * (1 - 1.96 * cv)
    upper = pred * (1 + 1.96 * cv)

    return ScalingResult(
        "clearance", pred, lower, upper,
        "MLP_correction", Species[]
    )
end

# ===========================================================================
# Parameter Scaling Functions
# ===========================================================================

"""
    scale_clearance(cl_animal, species_from, species_to; method=:simple) -> Float64

Scale clearance between species.
"""
function scale_clearance(
    cl_animal::Float64,
    species_from::Species,
    species_to::Species;
    method::Symbol = :simple,
    exponent::Float64 = 0.75
)::Float64
    from_data = SPECIES_DATABASE[species_from]
    to_data = SPECIES_DATABASE[species_to]

    if method == :simple
        # Simple allometry: CL2 = CL1 × (BW2/BW1)^0.75
        return cl_animal * (to_data.body_weight / from_data.body_weight)^exponent
    elseif method == :hepatic_flow
        # Scale by hepatic blood flow ratio
        return cl_animal * (to_data.hepatic_blood_flow / from_data.hepatic_blood_flow)
    elseif method == :gfr
        # Scale by GFR ratio (for renally cleared drugs)
        return cl_animal * (to_data.gfr / from_data.gfr)
    else
        error("Unknown scaling method: $method")
    end
end

"""
    scale_volume(vd_animal, species_from, species_to; method=:simple) -> Float64

Scale volume of distribution between species.
"""
function scale_volume(
    vd_animal::Float64,
    species_from::Species,
    species_to::Species;
    method::Symbol = :simple,
    exponent::Float64 = 1.0
)::Float64
    from_data = SPECIES_DATABASE[species_from]
    to_data = SPECIES_DATABASE[species_to]

    if method == :simple
        # Simple allometry: Vd2 = Vd1 × (BW2/BW1)^1.0
        return vd_animal * (to_data.body_weight / from_data.body_weight)^exponent
    elseif method == :blood_volume
        # Scale by blood volume ratio
        return vd_animal * (to_data.blood_volume / from_data.blood_volume)
    else
        error("Unknown scaling method: $method")
    end
end

"""
    scale_half_life(t12_animal, species_from, species_to) -> Float64

Scale half-life between species.
"""
function scale_half_life(
    t12_animal::Float64,
    species_from::Species,
    species_to::Species;
    exponent::Float64 = 0.25
)::Float64
    from_data = SPECIES_DATABASE[species_from]
    to_data = SPECIES_DATABASE[species_to]

    # t1/2 scales as BW^0.25
    return t12_animal * (to_data.body_weight / from_data.body_weight)^exponent
end

# ===========================================================================
# Dedrick Plot Analysis
# ===========================================================================

"""
    dedrick_plot(species_list, cl_values, vd_values, t12_values) -> Dict

Generate Dedrick plot data for pharmacokinetic equivalence.

Transforms time by (BW)^(-0.25) and concentration by (BW/dose).
"""
function dedrick_plot(
    species_list::Vector{Species},
    cl_values::Vector{Float64},
    vd_values::Vector{Float64},
    t12_values::Vector{Float64}
)::Dict{String, Any}
    n = length(species_list)

    body_weights = [SPECIES_DATABASE[s].body_weight for s in species_list]

    # Kallynochrons (time scaling)
    kallynochrons = t12_values ./ (body_weights .^ 0.25)

    # Apolysichrons (CL/Vd normalized)
    apolysichrons = (cl_values ./ vd_values) .* (body_weights .^ 0.25)

    # Dienetichrons (volume normalized)
    dienetichrons = vd_values ./ body_weights

    return Dict(
        "species" => species_list,
        "body_weight" => body_weights,
        "kallynochrons" => kallynochrons,
        "apolysichrons" => apolysichrons,
        "dienetichrons" => dienetichrons,
        "t12_original" => t12_values,
        "cl_original" => cl_values,
        "vd_original" => vd_values
    )
end

# ===========================================================================
# Human PK Prediction
# ===========================================================================

"""
    predict_human_pk(animal_data; method=:rule_of_exponents) -> Dict

Predict human PK parameters from multi-species animal data.

# Arguments
- `animal_data`: Dict with species => Dict("cl" => val, "vd" => val, "t12" => val)
- `method`: Scaling method (:simple, :rule_of_exponents, :brain, :mlp)

# Returns
- Dict with predicted human CL, Vd, t1/2 and confidence intervals
"""
function predict_human_pk(
    animal_data::Dict{Species, Dict{String, Float64}};
    method::Symbol = :rule_of_exponents
)::Dict{String, ScalingResult}
    species_list = collect(keys(animal_data))

    results = Dict{String, ScalingResult}()

    # Extract data
    body_weights = [SPECIES_DATABASE[s].body_weight for s in species_list]
    brain_weights = [SPECIES_DATABASE[s].brain_weight for s in species_list]
    mlp_values = [SPECIES_DATABASE[s].max_lifespan for s in species_list]

    human_bw = SPECIES_DATABASE[HUMAN].body_weight
    human_brw = SPECIES_DATABASE[HUMAN].brain_weight
    human_mlp = SPECIES_DATABASE[HUMAN].max_lifespan

    # Clearance
    if all(haskey(animal_data[s], "cl") for s in species_list)
        cl_values = [animal_data[s]["cl"] for s in species_list]
        cl_model = simple_allometry(cl_values, body_weights;
                                     parameter_name="clearance", species=species_list)

        if method == :rule_of_exponents
            results["clearance"] = rule_of_exponents(cl_model, human_bw;
                                                      target_brain_weight=human_brw,
                                                      target_mlp=human_mlp)
        elseif method == :brain
            results["clearance"] = brain_weight_correction(cl_values, body_weights,
                                                           brain_weights;
                                                           target_bw=human_bw,
                                                           target_brw=human_brw)
        elseif method == :mlp
            results["clearance"] = mlp_correction(cl_values, body_weights, mlp_values;
                                                   target_bw=human_bw, target_mlp=human_mlp)
        else
            pred = cl_model.coefficient * human_bw^cl_model.exponent
            results["clearance"] = ScalingResult("clearance", pred, pred*0.5, pred*2.0,
                                                  "simple_allometry", species_list)
        end
    end

    # Volume of distribution
    if all(haskey(animal_data[s], "vd") for s in species_list)
        vd_values = [animal_data[s]["vd"] for s in species_list]
        vd_model = simple_allometry(vd_values, body_weights;
                                     parameter_name="volume", species=species_list)
        pred = vd_model.coefficient * human_bw^vd_model.exponent
        results["volume"] = ScalingResult("volume", pred, pred*0.5, pred*2.0,
                                           "simple_allometry", species_list)
    end

    # Half-life
    if all(haskey(animal_data[s], "t12") for s in species_list)
        t12_values = [animal_data[s]["t12"] for s in species_list]
        t12_model = simple_allometry(t12_values, body_weights;
                                      parameter_name="half_life", species=species_list)
        pred = t12_model.coefficient * human_bw^t12_model.exponent
        results["half_life"] = ScalingResult("half_life", pred, pred*0.5, pred*2.0,
                                              "simple_allometry", species_list)
    end

    return results
end

"""
    predict_first_in_human_dose(animal_noael, species; method=:hep) -> Float64

Predict first-in-human (FIH) dose from animal NOAEL.

Methods:
- :hep - Human Equivalent Dose based on BSA
- :allometric - Allometric scaling with safety factor
- :mabel - Minimum Anticipated Biological Effect Level
"""
function predict_first_in_human_dose(
    animal_noael::Float64,  # mg/kg
    species::Species;
    method::Symbol = :hep,
    safety_factor::Float64 = 10.0
)::Dict{String, Any}
    animal_data = SPECIES_DATABASE[species]
    human_data = SPECIES_DATABASE[HUMAN]

    # Body surface area (BSA) conversion factors (FDA Km values)
    km_factors = Dict(
        MOUSE => 3.0,
        RAT => 6.0,
        RABBIT => 12.0,
        DOG => 20.0,
        CYNOMOLGUS_MONKEY => 12.0,
        RHESUS_MONKEY => 12.0,
        MINIPIG => 27.0,
        HUMAN => 37.0
    )

    km_animal = km_factors[species]
    km_human = km_factors[HUMAN]

    if method == :hep
        # Human Equivalent Dose = Animal NOAEL × (Km_animal / Km_human)
        hed = animal_noael * (km_animal / km_human)
        fih = hed / safety_factor
    elseif method == :allometric
        # Allometric: Dose_human = Dose_animal × (BW_human/BW_animal)^0.75
        hed = animal_noael * (human_data.body_weight / animal_data.body_weight)^0.75
        fih = hed / safety_factor
    elseif method == :mabel
        # MABEL approach - return factor of pharmacologically active dose
        fih = animal_noael / (safety_factor * 10)  # Additional 10× for MABEL
        hed = fih * safety_factor
    else
        error("Unknown FIH method: $method")
    end

    return Dict{String, Any}(
        "hed_mg_kg" => hed,
        "fih_mg_kg" => fih,
        "fih_mg_70kg" => fih * 70.0,
        "safety_factor" => safety_factor,
        "method" => string(method)
    )
end

# ===========================================================================
# IVIVE (In Vitro to In Vivo Extrapolation)
# ===========================================================================

"""
    hepatocyte_scaling(cl_int_hep, species; fu_inc=1.0, fu_p=1.0) -> Float64

Scale intrinsic clearance from hepatocytes to in vivo clearance.

CL_int,invivo = CL_int,hep × HPGL × liver_weight / body_weight
CL_h = (Q_h × fu_b × CL_int) / (Q_h + fu_b × CL_int)  # Well-stirred model
"""
function hepatocyte_scaling(
    cl_int_hepatocyte::Float64,  # μL/min/10^6 cells
    species::Species;
    fu_inc::Float64 = 1.0,       # fraction unbound in incubation
    fu_plasma::Float64 = 1.0,    # fraction unbound in plasma
    rb::Float64 = 1.0            # blood:plasma ratio
)::Dict{String, Float64}
    data = SPECIES_DATABASE[species]

    # Hepatocellularity (cells per gram liver)
    hpgl = data.hepatocytes_per_gram  # cells/g

    # Scale intrinsic clearance to whole liver
    # CL_int (μL/min/10^6 cells) → (mL/min)
    cl_int_liver = cl_int_hepatocyte * (hpgl / 1e6) * (data.liver_weight / 1000)

    # Correct for binding
    cl_int_corrected = cl_int_liver * (fu_inc / fu_plasma)

    # Well-stirred model for hepatic clearance
    qh = data.hepatic_blood_flow  # mL/min
    fu_b = fu_plasma / rb

    cl_h = (qh * fu_b * cl_int_corrected) / (qh + fu_b * cl_int_corrected)

    # Extraction ratio
    eh = cl_h / qh

    return Dict(
        "cl_int_liver_ml_min" => cl_int_liver,
        "cl_int_corrected_ml_min" => cl_int_corrected,
        "cl_hepatic_ml_min" => cl_h,
        "cl_hepatic_L_h" => cl_h * 60 / 1000,
        "extraction_ratio" => eh,
        "hepatocellularity" => hpgl
    )
end

"""
    microsomal_scaling(cl_int_mic, species; fu_mic=1.0, fu_p=1.0) -> Float64

Scale intrinsic clearance from microsomes to in vivo clearance.
"""
function microsomal_scaling(
    cl_int_microsomal::Float64,  # μL/min/mg protein
    species::Species;
    fu_mic::Float64 = 1.0,
    fu_plasma::Float64 = 1.0,
    rb::Float64 = 1.0
)::Dict{String, Float64}
    data = SPECIES_DATABASE[species]

    # Microsomal protein per gram liver (MPPGL)
    mppgl = data.microsomal_protein  # mg/g

    # Scale to whole liver
    cl_int_liver = cl_int_microsomal * mppgl * (data.liver_weight / 1000)

    # Correct for binding
    cl_int_corrected = cl_int_liver * (fu_mic / fu_plasma)

    # Well-stirred model
    qh = data.hepatic_blood_flow
    fu_b = fu_plasma / rb

    cl_h = (qh * fu_b * cl_int_corrected) / (qh + fu_b * cl_int_corrected)
    eh = cl_h / qh

    return Dict(
        "cl_int_liver_ml_min" => cl_int_liver,
        "cl_int_corrected_ml_min" => cl_int_corrected,
        "cl_hepatic_ml_min" => cl_h,
        "cl_hepatic_L_h" => cl_h * 60 / 1000,
        "extraction_ratio" => eh,
        "mppgl" => mppgl
    )
end

"""
    ivive_clearance(cl_int, species_from, species_to; source=:hepatocyte) -> Dict

Perform complete IVIVE from in vitro data to predicted in vivo clearance.
"""
function ivive_clearance(
    cl_int::Float64,
    species_from::Species,
    species_to::Species;
    source::Symbol = :hepatocyte,
    fu_inc::Float64 = 1.0,
    fu_plasma_from::Float64 = 1.0,
    fu_plasma_to::Float64 = 1.0,
    rb_from::Float64 = 1.0,
    rb_to::Float64 = 1.0
)::Dict{String, Any}
    # Step 1: In vitro to in vivo for source species
    if source == :hepatocyte
        ivive_from = hepatocyte_scaling(cl_int, species_from;
                                         fu_inc=fu_inc, fu_plasma=fu_plasma_from, rb=rb_from)
    else
        ivive_from = microsomal_scaling(cl_int, species_from;
                                         fu_mic=fu_inc, fu_plasma=fu_plasma_from, rb=rb_from)
    end

    cl_invivo_from = ivive_from["cl_hepatic_L_h"]

    # Step 2: Allometric scaling to target species
    cl_invivo_to = scale_clearance(cl_invivo_from, species_from, species_to)

    # Adjust for binding differences
    cl_adjusted = cl_invivo_to * (fu_plasma_to / fu_plasma_from)

    return Dict(
        "source_species" => species_from,
        "target_species" => species_to,
        "cl_int_in_vitro" => cl_int,
        "cl_invivo_source_L_h" => cl_invivo_from,
        "cl_invivo_target_L_h" => cl_invivo_to,
        "cl_adjusted_L_h" => cl_adjusted,
        "ivive_details" => ivive_from
    )
end

# ===========================================================================
# Fit Allometric Model from Data
# ===========================================================================

"""
    fit_allometric_model(data_dict; parameter="clearance") -> AllometricModel

Fit allometric model from experimental data dictionary.

# Arguments
- `data_dict`: Dict{Species, Float64} mapping species to parameter values
- `parameter`: Name of parameter being fit
"""
function fit_allometric_model(
    data_dict::Dict{Species, Float64};
    parameter::String = "clearance"
)::AllometricModel
    species_list = collect(keys(data_dict))
    values = [data_dict[s] for s in species_list]
    weights = [SPECIES_DATABASE[s].body_weight for s in species_list]

    return simple_allometry(values, weights;
                            parameter_name=parameter,
                            species=species_list)
end

end # module
