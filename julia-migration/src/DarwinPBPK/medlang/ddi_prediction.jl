# =============================================================================
# DDI PREDICTION MODULE
# =============================================================================
# Static and dynamic DDI predictions using mechanistic equations
# Based on FDA/EMA guidance and published PBPK methodology
#
# Darwin PBPK Platform v2.10.0
# =============================================================================

module DDIPrediction

using Statistics

# Include the data files
include("databases/generated/fda_ddi_classification.jl")
include("databases/generated/mbi_parameters.jl")
include("databases/generated/clinical_ddi_validation.jl")

export predict_ddi, predict_inhibition_ddi, predict_mbi_ddi, predict_induction_ddi
export get_inhibitor_params, get_substrate_params, get_inducer_params
export validate_predictions, DDIResult
export predict_ddi_by_phenotype, CYP2D6_PHENOTYPES

# =============================================================================
# CYP2D6 PHENOTYPE VARIABILITY
# =============================================================================
# Activity scores and fm adjustments for different CYP2D6 phenotypes
# Based on CPIC guidelines and published literature

"""
CYP2D6 metabolizer phenotypes with activity scores.
- PM: Poor Metabolizer (activity score 0)
- IM: Intermediate Metabolizer (activity score 0.5-1)
- NM: Normal Metabolizer (activity score 1-2)
- UM: Ultrarapid Metabolizer (activity score >2)
"""
const CYP2D6_PHENOTYPES = Dict{Symbol, NamedTuple}(
    :PM => (
        name = "Poor Metabolizer",
        activity_score = 0.0,
        fm_multiplier = 0.0,  # No CYP2D6 activity
        frequency_caucasian = 0.07,  # 7% of Caucasians
        frequency_asian = 0.01,  # 1% of Asians
        frequency_african = 0.03,  # 3% of African Americans
        clinical_impact = :high,
        description = "No functional CYP2D6 - behaves as if fully inhibited"
    ),
    :IM => (
        name = "Intermediate Metabolizer",
        activity_score = 0.75,
        fm_multiplier = 0.5,  # 50% of normal CYP2D6 activity
        frequency_caucasian = 0.10,
        frequency_asian = 0.35,
        frequency_african = 0.10,
        clinical_impact = :moderate,
        description = "Reduced CYP2D6 activity"
    ),
    :NM => (
        name = "Normal Metabolizer",
        activity_score = 1.5,
        fm_multiplier = 1.0,  # Normal activity (reference)
        frequency_caucasian = 0.75,
        frequency_asian = 0.60,
        frequency_african = 0.80,
        clinical_impact = :none,
        description = "Normal CYP2D6 activity"
    ),
    :UM => (
        name = "Ultrarapid Metabolizer",
        activity_score = 3.0,
        fm_multiplier = 2.0,  # 2x normal CYP2D6 activity
        frequency_caucasian = 0.05,
        frequency_asian = 0.02,
        frequency_african = 0.05,
        clinical_impact = :high,
        description = "Increased CYP2D6 activity - may need higher doses"
    ),
)

# =============================================================================
# TYPICAL CLINICAL CONCENTRATIONS
# =============================================================================
# Cmax values (total, μM) at typical clinical doses
# Used when no concentration is provided
# Sources: FDA labels, published PK studies

const TYPICAL_CLINICAL_CMAX = Dict{Symbol, NamedTuple}(
    # CYP1A2 inhibitors
    :fluvoxamine => (cmax_um = 0.3, fu_p = 0.2, dose_mg = 100.0),  # 100mg
    :ciprofloxacin => (cmax_um = 8.0, fu_p = 0.6, dose_mg = 500.0),  # 500mg BID
    :enoxacin => (cmax_um = 10.0, fu_p = 0.6, dose_mg = 400.0),

    # CYP3A4 inhibitors
    :itraconazole => (cmax_um = 1.0, fu_p = 0.002, dose_mg = 200.0),  # Highly bound
    :ketoconazole => (cmax_um = 6.0, fu_p = 0.01, dose_mg = 400.0),
    :clarithromycin => (cmax_um = 3.0, fu_p = 0.3, dose_mg = 500.0),
    :erythromycin => (cmax_um = 4.0, fu_p = 0.2, dose_mg = 500.0),
    :ritonavir => (cmax_um = 15.0, fu_p = 0.02, dose_mg = 100.0),  # As booster
    :diltiazem => (cmax_um = 0.5, fu_p = 0.2, dose_mg = 120.0),
    :verapamil => (cmax_um = 0.4, fu_p = 0.1, dose_mg = 80.0),
    :fluconazole => (cmax_um = 30.0, fu_p = 0.9, dose_mg = 400.0),  # Low binding

    # CYP2D6 inhibitors
    :quinidine => (cmax_um = 10.0, fu_p = 0.1, dose_mg = 200.0),
    :paroxetine => (cmax_um = 0.15, fu_p = 0.05, dose_mg = 20.0),
    :fluoxetine => (cmax_um = 0.5, fu_p = 0.06, dose_mg = 20.0),
    # Bupropion: hydroxybupropion is the main CYP2D6 inhibitor (Cmax ~10x parent)
    :bupropion => (cmax_um = 5.0, fu_p = 0.2, dose_mg = 150.0),  # Effective [hydroxybupropion]

    # CYP2C8 inhibitors
    :gemfibrozil => (cmax_um = 100.0, fu_p = 0.01, dose_mg = 600.0),  # Highly bound

    # CYP2C9/2C19 inhibitors
    :amiodarone => (cmax_um = 2.0, fu_p = 0.04, dose_mg = 400.0),

    # Inducers
    :rifampin => (cmax_um = 20.0, fu_p = 0.2, dose_mg = 600.0),
    :carbamazepine => (cmax_um = 35.0, fu_p = 0.25, dose_mg = 400.0),
    :phenytoin => (cmax_um = 80.0, fu_p = 0.1, dose_mg = 300.0),
)

# =============================================================================
# DDI RESULT STRUCTURE
# =============================================================================

"""
Result of a DDI prediction.
"""
struct DDIResult
    perpetrator::Symbol
    victim::Symbol
    mechanism::Symbol          # :reversible, :mbi, :induction, :mixed
    enzyme::Symbol
    auc_ratio::Float64         # Predicted AUC ratio
    cmax_ratio::Float64        # Predicted Cmax ratio (approximate)
    confidence::Symbol         # :high, :medium, :low
    clinical_significance::Symbol  # :strong, :moderate, :weak, :none
    parameters_used::NamedTuple
    warnings::Vector{String}
end

# =============================================================================
# PARAMETER LOOKUP FUNCTIONS
# =============================================================================

"""
Get inhibitor parameters for a drug across all enzymes.
Returns Dict of enzyme => parameters.
"""
function get_inhibitor_params(drug::Symbol)
    results = Dict{Symbol, NamedTuple}()

    for (enzyme, inhibitors) in FDA_CYP_INHIBITORS
        if haskey(inhibitors, drug)
            results[enzyme] = inhibitors[drug]
        end
    end

    # Also check MBI parameters
    for (enzyme, inhibitors) in MBI_PARAMETERS
        if haskey(inhibitors, drug)
            if haskey(results, enzyme)
                # Merge with existing - MBI takes precedence
                results[enzyme] = merge(results[enzyme], (is_mbi=true, mbi_params=inhibitors[drug]))
            else
                results[enzyme] = (is_mbi=true, mbi_params=inhibitors[drug])
            end
        end
    end

    return results
end

"""
Get substrate fm values for a drug.
"""
function get_substrate_params(drug::Symbol)
    if haskey(QUANTITATIVE_FM_VALUES, drug)
        return QUANTITATIVE_FM_VALUES[drug]
    end

    # Check FDA substrates for enzyme assignment (but no quantitative fm)
    for (enzyme, substrates) in FDA_CYP_SUBSTRATES
        if haskey(substrates, drug)
            return substrates[drug]
        end
    end

    return nothing
end

"""
Get inducer parameters for a drug.
"""
function get_inducer_params(drug::Symbol)
    results = Dict{Symbol, NamedTuple}()

    for (enzyme, inducers) in FDA_CYP_INDUCERS
        if haskey(inducers, drug)
            results[enzyme] = inducers[drug]
        end
    end

    return results
end

# =============================================================================
# CONCENTRATION ESTIMATION
# =============================================================================

"""
Estimate hepatic inlet concentration [I]h for DDI prediction.

FDA recommends: [I]h = fu,p * (Cmax + Fa*Fg*ka*Dose/Qh)

Simplified here using typical values.
Returns concentration in μM.
"""
function estimate_hepatic_concentration(;
    dose_mg::Float64,
    cmax_um::Float64 = 0.0,
    fu_p::Float64 = 0.1,      # Fraction unbound in plasma
    fa::Float64 = 1.0,         # Fraction absorbed
    fg::Float64 = 1.0,         # Intestinal availability
    ka_per_hr::Float64 = 1.0,  # Absorption rate constant
    qh_l_hr::Float64 = 90.0,   # Hepatic blood flow
    mw::Float64 = 400.0        # Molecular weight
)
    if cmax_um > 0
        # If Cmax provided, use inlet concentration formula
        dose_umol = dose_mg / mw * 1000  # Convert to μmol
        inlet_term = fa * fg * ka_per_hr * dose_umol / qh_l_hr
        return fu_p * (cmax_um + inlet_term)
    else
        # Rough estimate from dose
        # Assume Cmax ~ dose/(Vd * BW) with typical Vd=1 L/kg, BW=70kg
        estimated_cmax = dose_mg / mw * 1000 / 70  # Very rough μM estimate
        return fu_p * estimated_cmax
    end
end

"""
Estimate intestinal concentration [I]g for gut-wall DDI.
FDA: [I]g = Fa*ka*Dose / (Qent * MW)
"""
function estimate_intestinal_concentration(;
    dose_mg::Float64,
    fa::Float64 = 1.0,
    ka_per_hr::Float64 = 1.0,
    qent_l_hr::Float64 = 18.0,  # Enterocyte blood flow
    mw::Float64 = 400.0
)
    dose_umol = dose_mg / mw * 1000
    return fa * ka_per_hr * dose_umol / qent_l_hr
end

# =============================================================================
# REVERSIBLE INHIBITION DDI PREDICTION
# =============================================================================

"""
Predict DDI from reversible (competitive) inhibition.

Basic equation:
    AUC_ratio = 1 / (fm/(1 + [I]/Ki) + (1-fm))

Where:
- fm: fraction metabolized by inhibited enzyme
- [I]: inhibitor concentration (unbound)
- Ki: inhibition constant
"""
function predict_reversible_inhibition(;
    fm::Float64,
    inhibitor_conc_um::Float64,
    ki_um::Float64
)
    # Calculate inhibition term
    R = 1.0 + inhibitor_conc_um / ki_um

    # AUC ratio
    auc_ratio = 1.0 / (fm / R + (1.0 - fm))

    return auc_ratio
end

"""
Predict DDI for a specific inhibitor-substrate pair.
"""
function predict_inhibition_ddi(
    inhibitor::Symbol,
    substrate::Symbol;
    inhibitor_dose_mg::Float64 = 0.0,
    inhibitor_cmax_um::Float64 = 0.0,
    fu_p::Float64 = 0.1
)
    warnings = String[]

    # Get inhibitor parameters
    inhib_params = get_inhibitor_params(inhibitor)
    if isempty(inhib_params)
        return DDIResult(
            inhibitor, substrate, :unknown, :unknown,
            1.0, 1.0, :low, :none,
            (;), ["No inhibitor parameters found for $inhibitor"]
        )
    end

    # Get substrate parameters
    sub_params = get_substrate_params(substrate)
    if isnothing(sub_params)
        push!(warnings, "No quantitative fm values for $substrate, using estimates")
    end

    # Find matching enzyme
    total_auc_ratio = 1.0
    enzymes_affected = Symbol[]
    last_fm = 0.5
    last_conc = 0.0
    last_ki = 1.0

    for (enzyme, inh_data) in inhib_params
        # Get fm for this enzyme
        fm_key = Symbol("fm_", lowercase(string(enzyme)[4:end]))  # e.g., :fm_3a4

        # Only consider enzymes where substrate has documented metabolism
        has_fm_data = !isnothing(sub_params) && hasproperty(sub_params, fm_key)

        fm = if has_fm_data
            getproperty(sub_params, fm_key)
        elseif !isnothing(sub_params) && hasproperty(sub_params, :sensitivity)
            # Check if this enzyme matches the substrate's primary enzyme
            # (sensitivity field implies it's the primary pathway)
            sub_params.sensitivity == :sensitive ? 0.8 : 0.5
        else
            # No fm data for this enzyme - skip unless it's the only option
            # This prevents e.g., fluconazole CYP3A4 affecting warfarin (CYP2C9 substrate)
            if length(inhib_params) > 1
                continue  # Skip this enzyme, try others
            end
            0.5  # Default only if no other options
        end

        # Skip if fm is negligible
        if fm < 0.1
            continue
        end

        # Get Ki
        ki = if hasproperty(inh_data, :ki_um)
            inh_data.ki_um
        elseif hasproperty(inh_data, :mbi_params)
            inh_data.mbi_params.ki_um
        else
            push!(warnings, "No Ki value for $inhibitor on $enzyme")
            continue
        end

        # Estimate inhibitor concentration
        # For CYP3A4, consider both hepatic and intestinal inhibition
        conc = if inhibitor_cmax_um > 0
            fu_p * inhibitor_cmax_um
        elseif inhibitor_dose_mg > 0
            estimate_hepatic_concentration(dose_mg=inhibitor_dose_mg, fu_p=fu_p)
        elseif haskey(TYPICAL_CLINICAL_CMAX, inhibitor)
            # Use literature values for known inhibitors
            pk = TYPICAL_CLINICAL_CMAX[inhibitor]
            pk.fu_p * pk.cmax_um
        else
            push!(warnings, "No inhibitor concentration provided, using default estimate")
            fu_p * 5.0  # Assume 5 μM total Cmax as default
        end

        # Calculate DDI using hybrid approach:
        # 1. Check if substrate has known clinical sensitivity (auc_fold_with_inhibitor)
        # 2. For potent inhibitors (Ki < 1 μM) with clinical data, use calibrated values
        # 3. For others, use mechanistic [I]/Ki model

        # First check substrate-specific clinical data from FDA_CYP_SUBSTRATES
        substrate_clinical_auc = 0.0
        if haskey(FDA_CYP_SUBSTRATES, enzyme) && haskey(FDA_CYP_SUBSTRATES[enzyme], substrate)
            sub_clinical = FDA_CYP_SUBSTRATES[enzyme][substrate]
            if hasproperty(sub_clinical, :auc_fold_with_inhibitor)
                substrate_clinical_auc = sub_clinical.auc_fold_with_inhibitor
            end
        end

        enzyme_auc_ratio = if substrate_clinical_auc > 1.0 && hasproperty(inh_data, :strength) && inh_data.strength == :strong
            # Use substrate-specific clinical data (most accurate)
            substrate_clinical_auc
        elseif hasproperty(inh_data, :auc_ratio) && inh_data.auc_ratio > 1.5 && ki < 1.0
            # Potent inhibitor with clinical data - use calibrated value
            # Clinical auc_ratio is for INDEX substrate (fm ~0.9)
            # Scale appropriately for actual substrate fm
            base_ratio = inh_data.auc_ratio
            index_fm = 0.9  # Assumed fm for index substrate in clinical study

            # Back-calculate the effective R (inhibition factor) from clinical data
            # AUC_ratio = 1 / (fm/R + (1-fm)), solve for R:
            # R = fm * AUC_ratio / (1 - (1-fm)*AUC_ratio)
            if base_ratio < 50.0  # Avoid numerical issues
                effective_R = index_fm * base_ratio / max(0.01, 1.0 - (1.0 - index_fm) * base_ratio)
                effective_R = min(effective_R, 100.0)  # Cap at 100x inhibition
            else
                effective_R = 100.0
            end

            # Apply to actual substrate fm
            1.0 / (fm / effective_R + (1.0 - fm))
        else
            # Calculate contribution from this enzyme using mechanistic model
            predict_reversible_inhibition(fm=fm, inhibitor_conc_um=conc, ki_um=ki)
        end

        # Store last values for reporting
        last_fm = fm
        last_conc = conc
        last_ki = ki

        # Combine (multiplicative for multiple pathways blocked)
        if enzyme_auc_ratio > 1.0
            total_auc_ratio *= enzyme_auc_ratio
            push!(enzymes_affected, enzyme)
        end
    end

    # Determine clinical significance
    significance = if total_auc_ratio >= 5.0
        :strong
    elseif total_auc_ratio >= 2.0
        :moderate
    elseif total_auc_ratio >= 1.25
        :weak
    else
        :none
    end

    # Confidence based on data quality
    confidence = if !isnothing(sub_params) && length(warnings) == 0
        :high
    elseif length(warnings) <= 1
        :medium
    else
        :low
    end

    primary_enzyme = isempty(enzymes_affected) ? :unknown : first(enzymes_affected)

    return DDIResult(
        inhibitor, substrate, :reversible, primary_enzyme,
        total_auc_ratio, sqrt(total_auc_ratio),  # Cmax ratio approximated
        confidence, significance,
        (fm=last_fm, conc=last_conc, ki=last_ki),
        warnings
    )
end

# =============================================================================
# MECHANISM-BASED INHIBITION DDI PREDICTION
# =============================================================================

"""
Predict DDI from mechanism-based (time-dependent) inhibition.

MBI equation:
    R = 1 + (kinact/kdeg) * [I]/(KI + [I])
    AUC_ratio = 1 / (fm/R + (1-fm))

Where:
- kinact: maximum inactivation rate (min⁻¹)
- KI: concentration for half-max inactivation (μM)
- kdeg: enzyme degradation rate constant (min⁻¹)
"""
function predict_mbi_inhibition(;
    fm::Float64,
    inhibitor_conc_um::Float64,
    kinact_per_min::Float64,
    ki_um::Float64,
    kdeg_per_min::Float64
)
    # Calculate steady-state inhibition ratio
    R = 1.0 + (kinact_per_min / kdeg_per_min) * inhibitor_conc_um / (ki_um + inhibitor_conc_um)

    # AUC ratio
    auc_ratio = 1.0 / (fm / R + (1.0 - fm))

    return auc_ratio
end

"""
Predict MBI DDI for a specific perpetrator-victim pair.
"""
function predict_mbi_ddi(
    inhibitor::Symbol,
    substrate::Symbol;
    inhibitor_dose_mg::Float64 = 0.0,
    inhibitor_cmax_um::Float64 = 0.0,
    fu_p::Float64 = 0.1
)
    warnings = String[]

    # Check MBI parameters
    mbi_found = false
    total_auc_ratio = 1.0
    primary_enzyme = :unknown
    params_used = (;)

    for (enzyme, inhibitors) in MBI_PARAMETERS
        if haskey(inhibitors, inhibitor)
            mbi_found = true
            mbi_data = inhibitors[inhibitor]

            # Get substrate fm
            sub_params = get_substrate_params(substrate)
            fm_key = Symbol("fm_", lowercase(string(enzyme)[4:end]))

            fm = if !isnothing(sub_params) && hasproperty(sub_params, fm_key)
                getproperty(sub_params, fm_key)
            else
                push!(warnings, "Using default fm=0.8 for $substrate via $enzyme")
                0.8
            end

            # Get kdeg
            kdeg = if haskey(ENZYME_KDEG, enzyme)
                ENZYME_KDEG[enzyme].kdeg_per_min
            else
                push!(warnings, "Using default kdeg for $enzyme")
                0.0005
            end

            # Estimate concentration
            conc = if inhibitor_cmax_um > 0
                fu_p * inhibitor_cmax_um
            elseif inhibitor_dose_mg > 0
                estimate_hepatic_concentration(dose_mg=inhibitor_dose_mg, fu_p=fu_p)
            elseif haskey(TYPICAL_CLINICAL_CMAX, inhibitor)
                pk = TYPICAL_CLINICAL_CMAX[inhibitor]
                pk.fu_p * pk.cmax_um
            else
                push!(warnings, "Using default clinical concentration")
                fu_p * 5.0
            end

            # Calculate MBI DDI
            # Use calibrated clinical value if available and fm is high
            auc_ratio = if hasproperty(mbi_data, :clinical_auc_ratio) && mbi_data.clinical_auc_ratio > 1.5 && fm >= 0.5
                # Use clinical calibration, scaled by fm
                clinical_ratio = mbi_data.clinical_auc_ratio
                index_fm = 0.9  # Assumed fm for index substrate

                # Back-calculate effective R and apply to actual fm
                if clinical_ratio < 50.0
                    effective_R = index_fm * clinical_ratio / max(0.01, 1.0 - (1.0 - index_fm) * clinical_ratio)
                    effective_R = min(effective_R, 100.0)
                else
                    effective_R = 100.0
                end
                1.0 / (fm / effective_R + (1.0 - fm))
            else
                # Use mechanistic model
                predict_mbi_inhibition(
                    fm = fm,
                    inhibitor_conc_um = conc,
                    kinact_per_min = mbi_data.kinact_per_min,
                    ki_um = mbi_data.ki_um,
                    kdeg_per_min = kdeg
                )
            end

            total_auc_ratio *= auc_ratio
            primary_enzyme = enzyme
            params_used = (
                fm = fm,
                conc = conc,
                kinact = mbi_data.kinact_per_min,
                ki = mbi_data.ki_um,
                kdeg = kdeg,
                clinical_observed = mbi_data.clinical_auc_ratio
            )
        end
    end

    if !mbi_found
        push!(warnings, "No MBI parameters for $inhibitor, trying reversible inhibition")
        return predict_inhibition_ddi(inhibitor, substrate;
            inhibitor_dose_mg=inhibitor_dose_mg,
            inhibitor_cmax_um=inhibitor_cmax_um,
            fu_p=fu_p)
    end

    # Determine significance
    significance = if total_auc_ratio >= 5.0
        :strong
    elseif total_auc_ratio >= 2.0
        :moderate
    elseif total_auc_ratio >= 1.25
        :weak
    else
        :none
    end

    confidence = length(warnings) == 0 ? :high : (length(warnings) <= 2 ? :medium : :low)

    return DDIResult(
        inhibitor, substrate, :mbi, primary_enzyme,
        total_auc_ratio, sqrt(total_auc_ratio),
        confidence, significance,
        params_used,
        warnings
    )
end

# =============================================================================
# INDUCTION DDI PREDICTION
# =============================================================================

"""
Predict DDI from enzyme induction.

FDA/EMA induction equation:
    Induction fold = 1 + d * Emax * [I]u / (EC50 + [I]u)
    AUC_ratio = 1 / (fm * Induction_fold + (1 - fm))

For strong inducers like rifampin:
- Clinical data shows ~90% decrease (AUC ratio ~0.04-0.1)
- Emax values from in vitro can overpredict if not calibrated

We use an empirical calibration based on observed clinical data:
- Rifampin + midazolam: observed AUC ratio ~0.04
"""
function predict_induction(;
    fm::Float64,
    inducer_conc_um::Float64,
    emax::Float64,
    ec50_um::Float64,
    use_empirical::Bool = false,
    observed_auc_decrease_pct::Float64 = 0.0
)
    # If EC50 is 0 or use_empirical, use calibrated empirical values
    if ec50_um <= 0 || use_empirical
        if observed_auc_decrease_pct > 0
            # Convert percentage decrease to AUC ratio
            # 90% decrease = AUC ratio of 0.10
            return (100.0 - observed_auc_decrease_pct) / 100.0
        else
            return fm * 0.2 + (1 - fm)  # Default moderate induction
        end
    end

    # Calculate induction factor with saturable kinetics
    # [I]u should be unbound intracellular concentration
    # For nuclear receptor activation, often use total hepatic concentration
    saturation = inducer_conc_um / (ec50_um + inducer_conc_um)
    induction_fold = 1.0 + emax * saturation

    # Cap the induction fold to prevent unrealistic values
    # Most strong inducers achieve ~10-20x induction in vivo
    induction_fold = min(induction_fold, 25.0)

    # AUC ratio: increased clearance = decreased AUC
    # AUC_ratio = CLbaseline / CLnew = 1 / induction_fold (for fm=1)
    # For partial metabolism: AUC_ratio = 1 / (fm * induction_fold + (1-fm))
    auc_ratio = 1.0 / (fm * induction_fold + (1.0 - fm))

    return auc_ratio
end

"""
Predict induction DDI for a specific inducer-substrate pair.

Uses empirical calibration when available (preferred for accuracy),
falls back to mechanistic prediction when needed.
"""
function predict_induction_ddi(
    inducer::Symbol,
    substrate::Symbol;
    inducer_dose_mg::Float64 = 0.0,
    inducer_cmax_um::Float64 = 0.0,
    fu_p::Float64 = 0.1,
    use_empirical::Bool = true  # Prefer calibrated values
)
    warnings = String[]

    # Get inducer parameters
    ind_params = get_inducer_params(inducer)
    if isempty(ind_params)
        return DDIResult(
            inducer, substrate, :induction, :unknown,
            1.0, 1.0, :low, :none,
            (;), ["No induction parameters found for $inducer"]
        )
    end

    # Get substrate parameters
    sub_params = get_substrate_params(substrate)

    total_auc_ratio = 1.0
    primary_enzyme = :unknown
    params_used = (;)

    for (enzyme, ind_data) in ind_params
        # Get fm for this enzyme
        fm_key = Symbol("fm_", lowercase(string(enzyme)[4:end]))

        fm = if !isnothing(sub_params) && hasproperty(sub_params, fm_key)
            getproperty(sub_params, fm_key)
        else
            push!(warnings, "Using default fm=0.9 for sensitive $substrate via $enzyme")
            0.9  # Assume sensitive substrate
        end

        # Estimate inducer concentration (use total, not unbound for nuclear receptor activation)
        conc = if inducer_cmax_um > 0
            inducer_cmax_um
        elseif inducer_dose_mg > 0
            estimate_hepatic_concentration(dose_mg=inducer_dose_mg, fu_p=1.0)  # Total concentration
        else
            push!(warnings, "Using typical rifampin concentration")
            10.0  # Typical for rifampin at 600mg
        end

        # Get expected decrease from clinical data if available
        observed_decrease = hasproperty(ind_data, :auc_decrease_pct) ? ind_data.auc_decrease_pct : 0.0

        # Calculate induction DDI
        # Use empirical calibration for known strong inducers
        auc_ratio = predict_induction(
            fm = fm,
            inducer_conc_um = conc,
            emax = ind_data.emax,
            ec50_um = ind_data.ec50_um,
            use_empirical = use_empirical && observed_decrease > 50.0,  # Use empirical for strong inducers
            observed_auc_decrease_pct = observed_decrease
        )

        # For multiple pathways, take the most affected one (not multiply)
        if auc_ratio < total_auc_ratio
            total_auc_ratio = auc_ratio
            primary_enzyme = enzyme
        end

        params_used = (
            fm = fm,
            conc = conc,
            emax = ind_data.emax,
            ec50_um = ind_data.ec50_um,
            expected_decrease = observed_decrease,
            method = (use_empirical && observed_decrease > 50.0) ? :empirical : :mechanistic
        )
    end

    # Determine significance (inverse for induction - lower ratio = stronger)
    significance = if total_auc_ratio <= 0.2
        :strong
    elseif total_auc_ratio <= 0.5
        :moderate
    elseif total_auc_ratio <= 0.8
        :weak
    else
        :none
    end

    confidence = if use_empirical && hasproperty(params_used, :expected_decrease) && params_used.expected_decrease > 0
        :high
    elseif length(warnings) == 0
        :medium
    else
        :low
    end

    return DDIResult(
        inducer, substrate, :induction, primary_enzyme,
        total_auc_ratio, sqrt(total_auc_ratio),
        confidence, significance,
        params_used,
        warnings
    )
end

# =============================================================================
# MAIN DDI PREDICTION FUNCTION
# =============================================================================

"""
    predict_ddi(perpetrator, victim; kwargs...)

Predict drug-drug interaction between perpetrator (inhibitor/inducer) and victim (substrate).

Automatically detects mechanism (inhibition, MBI, induction) and uses appropriate model.
Priority: MBI > reversible inhibition > induction (for drugs with multiple mechanisms)

# Arguments
- `perpetrator::Symbol`: Inhibitor or inducer drug
- `victim::Symbol`: Substrate drug
- `perpetrator_dose_mg::Float64`: Dose of perpetrator (optional)
- `perpetrator_cmax_um::Float64`: Cmax of perpetrator in μM (optional)
- `fu_p::Float64`: Fraction unbound in plasma (default 0.1)
- `mechanism::Symbol`: Force specific mechanism (:auto, :mbi, :reversible, :induction)

# Returns
- `DDIResult`: Struct with predicted AUC ratio and parameters

# Example
```julia
result = predict_ddi(:itraconazole, :midazolam)
println("Predicted AUC ratio: ", result.auc_ratio)
```
"""
function predict_ddi(
    perpetrator::Symbol,
    victim::Symbol;
    perpetrator_dose_mg::Float64 = 0.0,
    perpetrator_cmax_um::Float64 = 0.0,
    fu_p::Float64 = 0.1,
    mechanism::Symbol = :auto
)
    # Allow forcing specific mechanism
    if mechanism == :induction
        return predict_induction_ddi(perpetrator, victim;
            inducer_dose_mg = perpetrator_dose_mg,
            inducer_cmax_um = perpetrator_cmax_um,
            fu_p = fu_p)
    elseif mechanism == :mbi
        return predict_mbi_ddi(perpetrator, victim;
            inhibitor_dose_mg = perpetrator_dose_mg,
            inhibitor_cmax_um = perpetrator_cmax_um,
            fu_p = fu_p)
    elseif mechanism == :reversible
        return predict_inhibition_ddi(perpetrator, victim;
            inhibitor_dose_mg = perpetrator_dose_mg,
            inhibitor_cmax_um = perpetrator_cmax_um,
            fu_p = fu_p)
    end

    # Auto-detect: Check MBI first (most potent mechanism)
    is_mbi = false
    for (enzyme, inhibitors) in MBI_PARAMETERS
        if haskey(inhibitors, perpetrator)
            is_mbi = true
            break
        end
    end

    if is_mbi
        return predict_mbi_ddi(perpetrator, victim;
            inhibitor_dose_mg = perpetrator_dose_mg,
            inhibitor_cmax_um = perpetrator_cmax_um,
            fu_p = fu_p)
    end

    # Check reversible inhibition
    inhib_params = get_inhibitor_params(perpetrator)
    if !isempty(inhib_params)
        return predict_inhibition_ddi(perpetrator, victim;
            inhibitor_dose_mg = perpetrator_dose_mg,
            inhibitor_cmax_um = perpetrator_cmax_um,
            fu_p = fu_p)
    end

    # Check induction (lowest priority - typically causes decrease, not increase)
    ind_params = get_inducer_params(perpetrator)
    if !isempty(ind_params)
        return predict_induction_ddi(perpetrator, victim;
            inducer_dose_mg = perpetrator_dose_mg,
            inducer_cmax_um = perpetrator_cmax_um,
            fu_p = fu_p)
    end

    # No parameters found
    return DDIResult(
        perpetrator, victim, :unknown, :unknown,
        1.0, 1.0, :low, :none,
        (;), ["No DDI parameters found for $perpetrator"]
    )
end

"""
    predict_ddi_comprehensive(perpetrator, victim; kwargs...)

Comprehensive DDI prediction combining CYP metabolism AND transporter effects.
Use this for drugs with known dual mechanisms (e.g., repaglinide = CYP2C8 + OATP1B1).

Returns both individual contributions and combined effect.
"""
function predict_ddi_comprehensive(
    perpetrator::Symbol,
    victim::Symbol;
    perpetrator_dose_mg::Float64 = 0.0,
    perpetrator_cmax_um::Float64 = 0.0,
    fu_p::Float64 = 0.1
)
    # Get CYP-mediated DDI
    cyp_result = predict_ddi(perpetrator, victim;
        perpetrator_dose_mg = perpetrator_dose_mg,
        perpetrator_cmax_um = perpetrator_cmax_um,
        fu_p = fu_p)

    # Get transporter-mediated DDI
    transporter_result = predict_transporter_ddi(perpetrator, victim;
        inhibitor_cmax_um = perpetrator_cmax_um,
        fu_p = fu_p)

    # Combine effects (multiplicative for independent mechanisms)
    cyp_contribution = cyp_result.auc_ratio
    transporter_contribution = transporter_result.auc_ratio

    # Combined AUC ratio
    combined_auc_ratio = cyp_contribution * transporter_contribution

    # Determine primary mechanism
    primary_mechanism = if cyp_contribution >= transporter_contribution
        cyp_result.mechanism
    else
        :transporter
    end

    primary_enzyme = if cyp_contribution >= transporter_contribution
        cyp_result.enzyme
    else
        transporter_result.enzyme
    end

    # Combine warnings
    all_warnings = vcat(cyp_result.warnings, transporter_result.warnings)

    # Clinical significance
    significance = if combined_auc_ratio >= 5.0
        :strong
    elseif combined_auc_ratio >= 2.0
        :moderate
    elseif combined_auc_ratio >= 1.25
        :weak
    else
        :none
    end

    return (
        result = DDIResult(
            perpetrator, victim, primary_mechanism, primary_enzyme,
            combined_auc_ratio, sqrt(combined_auc_ratio),
            :medium, significance,
            (cyp = cyp_contribution, transporter = transporter_contribution),
            all_warnings
        ),
        cyp_result = cyp_result,
        transporter_result = transporter_result
    )
end

export predict_ddi_comprehensive

# =============================================================================
# PHENOTYPE-AWARE DDI PREDICTION
# =============================================================================

"""
    predict_ddi_by_phenotype(perpetrator, victim, phenotype; kwargs...)

Predict DDI accounting for CYP2D6 (or other CYP) metabolizer phenotype.

For CYP2D6 substrates:
- PM patients: Already have minimal CYP2D6 activity, so inhibitors have less additional effect
- UM patients: Have higher CYP2D6 activity, so inhibitors may have larger absolute effect

# Arguments
- `perpetrator::Symbol`: Inhibitor drug
- `victim::Symbol`: Substrate drug
- `phenotype::Symbol`: Metabolizer phenotype (:PM, :IM, :NM, :UM)
- `enzyme::Symbol`: Which enzyme's phenotype (default :CYP2D6)

# Returns
- Named tuple with DDI prediction and phenotype-specific information

# Example
```julia
# Paroxetine + codeine in a CYP2D6 PM vs NM
result_pm = predict_ddi_by_phenotype(:paroxetine, :codeine, :PM)
result_nm = predict_ddi_by_phenotype(:paroxetine, :codeine, :NM)
```
"""
function predict_ddi_by_phenotype(
    perpetrator::Symbol,
    victim::Symbol,
    phenotype::Symbol;
    enzyme::Symbol = :CYP2D6,
    perpetrator_dose_mg::Float64 = 0.0,
    perpetrator_cmax_um::Float64 = 0.0,
    fu_p::Float64 = 0.1
)
    # Get phenotype data
    if enzyme == :CYP2D6 && haskey(CYP2D6_PHENOTYPES, phenotype)
        pheno_data = CYP2D6_PHENOTYPES[phenotype]
    else
        return (
            error = "Unknown phenotype: $phenotype for $enzyme",
            auc_ratio = 1.0,
            phenotype = phenotype
        )
    end

    # Get baseline DDI prediction (for NM)
    baseline_result = predict_ddi(perpetrator, victim;
        perpetrator_dose_mg = perpetrator_dose_mg,
        perpetrator_cmax_um = perpetrator_cmax_um,
        fu_p = fu_p)

    # Get substrate fm for this enzyme
    sub_params = get_substrate_params(victim)
    fm_key = Symbol("fm_", lowercase(string(enzyme)[4:end]))  # e.g., :fm_2d6

    base_fm = if !isnothing(sub_params) && hasproperty(sub_params, fm_key)
        getproperty(sub_params, fm_key)
    else
        0.0  # Not a substrate for this enzyme
    end

    # If not a substrate for this enzyme, phenotype doesn't matter
    if base_fm < 0.1
        return (
            result = baseline_result,
            phenotype = phenotype,
            phenotype_name = pheno_data.name,
            fm_adjusted = base_fm,
            clinical_note = "$victim is not primarily metabolized by $enzyme - phenotype has minimal impact",
            ddi_in_phenotype = baseline_result.auc_ratio
        )
    end

    # Adjust fm based on phenotype
    fm_multiplier = pheno_data.fm_multiplier

    # For PM: fm_2d6 effectively becomes 0 (no activity)
    # The remaining clearance is through other pathways
    # AUC in PM = AUC_baseline / (1 - fm_2d6) [if fm_2d6 was the only pathway]

    if phenotype == :PM
        # PM already has no CYP2D6 activity - like being permanently inhibited
        # DDI effect is minimal because there's nothing to inhibit
        auc_ratio_in_phenotype = 1.0 + (baseline_result.auc_ratio - 1.0) * 0.1
        clinical_note = "PM patients have no CYP2D6 activity - inhibitor has minimal additional effect"

        # But baseline exposure is already elevated
        baseline_auc_multiplier = 1.0 / (1.0 - base_fm + 0.01)  # Avoid div by 0
        baseline_auc_multiplier = min(baseline_auc_multiplier, 10.0)  # Cap

    elseif phenotype == :IM
        # IM has reduced activity - partial effect
        auc_ratio_in_phenotype = 1.0 + (baseline_result.auc_ratio - 1.0) * 0.5
        clinical_note = "IM patients have reduced CYP2D6 - DDI effect is attenuated"
        baseline_auc_multiplier = 1.0 / (1.0 - base_fm * 0.5)

    elseif phenotype == :UM
        # UM has increased activity - inhibitor may have larger effect
        auc_ratio_in_phenotype = 1.0 + (baseline_result.auc_ratio - 1.0) * 1.2
        clinical_note = "UM patients have increased CYP2D6 - may need higher substrate doses"
        baseline_auc_multiplier = 1.0 / (1.0 + base_fm * 0.5)  # Lower baseline exposure

    else  # NM
        auc_ratio_in_phenotype = baseline_result.auc_ratio
        clinical_note = "Normal metabolizer - standard DDI prediction applies"
        baseline_auc_multiplier = 1.0
    end

    return (
        result = baseline_result,
        phenotype = phenotype,
        phenotype_name = pheno_data.name,
        activity_score = pheno_data.activity_score,
        fm_base = base_fm,
        baseline_auc_multiplier = round(baseline_auc_multiplier, digits=2),
        ddi_auc_ratio_nm = baseline_result.auc_ratio,
        ddi_auc_ratio_in_phenotype = round(auc_ratio_in_phenotype, digits=2),
        clinical_note = clinical_note,
        population_frequency = pheno_data.frequency_caucasian
    )
end

# =============================================================================
# TRANSPORTER DDI PREDICTION
# =============================================================================

"""
Predict DDI from transporter inhibition (P-gp, OATP1B1, BCRP, etc.)

Transporter DDI equation (simplified):
    AUC_ratio = 1 + [I]/Ki  (for hepatic uptake transporters like OATP1B1)
    AUC_ratio = 1 + fu*[I]/Ki  (for efflux transporters like P-gp, BCRP)

Where:
- [I]: inhibitor concentration at transporter site
- Ki: inhibition constant
- fu: fraction unbound
"""
function predict_transporter_ddi(
    inhibitor::Symbol,
    substrate::Symbol;
    inhibitor_cmax_um::Float64 = 0.0,
    fu_p::Float64 = 0.1
)
    warnings = String[]

    # Check if substrate is a transporter substrate
    substrate_transporters = Symbol[]
    for (transporter, substrates) in FDA_TRANSPORTER_SUBSTRATES
        if substrate in substrates
            push!(substrate_transporters, transporter)
        end
    end

    if isempty(substrate_transporters)
        return DDIResult(
            inhibitor, substrate, :transporter, :unknown,
            1.0, 1.0, :low, :none,
            (;), ["$substrate is not a known transporter substrate"]
        )
    end

    # Check inhibitor parameters
    total_auc_ratio = 1.0
    primary_transporter = :unknown
    params_used = (;)

    for transporter in substrate_transporters
        if !haskey(FDA_TRANSPORTER_INHIBITORS, transporter)
            continue
        end

        inhib_dict = FDA_TRANSPORTER_INHIBITORS[transporter]
        if !haskey(inhib_dict, inhibitor)
            continue
        end

        inhib_data = inhib_dict[inhibitor]
        ki = inhib_data.ki_um

        # Estimate inhibitor concentration
        conc = if inhibitor_cmax_um > 0
            inhibitor_cmax_um  # Use total for portal vein (OATP) or intestinal (P-gp)
        elseif haskey(TYPICAL_CLINICAL_CMAX, inhibitor)
            TYPICAL_CLINICAL_CMAX[inhibitor].cmax_um
        else
            push!(warnings, "Using default inhibitor concentration")
            5.0
        end

        # Use clinical AUC ratio if available
        if hasproperty(inhib_data, :auc_ratio) && inhib_data.auc_ratio > 1.0
            auc_ratio = inhib_data.auc_ratio
        else
            # Mechanistic calculation
            # For OATP1B1: use portal vein concentration (higher than systemic)
            # For P-gp/BCRP at gut: use intestinal concentration
            if transporter == :OATP1B1
                # Portal vein concentration ~10x systemic for oral drugs
                portal_conc = conc * 5.0
                auc_ratio = 1.0 + portal_conc / ki
            else
                # Efflux transporters - use unbound
                auc_ratio = 1.0 + fu_p * conc / ki
            end
        end

        # Cap at reasonable value
        auc_ratio = min(auc_ratio, 10.0)

        if auc_ratio > total_auc_ratio
            total_auc_ratio = auc_ratio
            primary_transporter = transporter
            params_used = (transporter=transporter, conc=conc, ki=ki)
        end
    end

    if total_auc_ratio <= 1.0
        push!(warnings, "No transporter inhibition found for $inhibitor on $substrate")
    end

    # Determine clinical significance
    significance = if total_auc_ratio >= 3.0
        :strong
    elseif total_auc_ratio >= 1.5
        :moderate
    elseif total_auc_ratio >= 1.25
        :weak
    else
        :none
    end

    confidence = length(warnings) == 0 ? :medium : :low

    return DDIResult(
        inhibitor, substrate, :transporter, primary_transporter,
        total_auc_ratio, sqrt(total_auc_ratio),
        confidence, significance,
        params_used,
        warnings
    )
end

export predict_transporter_ddi

# =============================================================================
# MULTI-PERPETRATOR DDI PREDICTION
# =============================================================================

"""
Result structure for multi-perpetrator DDI predictions.
"""
struct MultiDDIResult
    victim::Symbol
    perpetrators::Vector{Symbol}
    individual_results::Vector{DDIResult}
    combined_auc_ratio::Float64
    combined_cmax_ratio::Float64
    net_effect::Symbol  # :increase, :decrease, :mixed, :neutral
    clinical_significance::Symbol
    warnings::Vector{String}
end

"""
    predict_multi_ddi(perpetrators, victim)

Predict DDI when multiple perpetrators affect the same victim drug.
Handles:
- Multiple inhibitors (additive/synergistic effects)
- Inhibitor + inducer combinations (net effect)
- Same enzyme vs different enzyme interactions

# Arguments
- `perpetrators::Vector{Symbol}`: List of inhibitor/inducer drugs
- `victim::Symbol`: Substrate drug

# Returns
- `MultiDDIResult`: Combined prediction with individual contributions

# Example
```julia
# Patient on clarithromycin + diltiazem taking midazolam
result = predict_multi_ddi([:clarithromycin, :diltiazem], :midazolam)
println("Combined AUC ratio: ", result.combined_auc_ratio)
```
"""
function predict_multi_ddi(
    perpetrators::Vector{Symbol},
    victim::Symbol
)
    warnings = String[]
    individual_results = DDIResult[]

    # Track effects by enzyme
    enzyme_effects = Dict{Symbol, Vector{Float64}}()
    inhibition_ratios = Float64[]
    induction_ratios = Float64[]

    for perp in perpetrators
        result = predict_ddi(perp, victim)
        push!(individual_results, result)

        if result.mechanism == :induction
            push!(induction_ratios, result.auc_ratio)
        else
            push!(inhibition_ratios, result.auc_ratio)
        end

        # Track by enzyme
        if result.enzyme != :unknown
            if !haskey(enzyme_effects, result.enzyme)
                enzyme_effects[result.enzyme] = Float64[]
            end
            push!(enzyme_effects[result.enzyme], result.auc_ratio)
        end
    end

    # Calculate combined effect
    # For inhibitors on same enzyme: use maximum (competitive binding)
    # For inhibitors on different enzymes: multiply
    # For inhibitor + inducer: complex - use conservative estimate

    combined_auc_ratio = 1.0

    if !isempty(inhibition_ratios) && !isempty(induction_ratios)
        # Mixed inhibition/induction
        push!(warnings, "Mixed inhibition/induction - using net effect estimate")
        max_inhibition = maximum(inhibition_ratios)
        min_induction = minimum(induction_ratios)
        # Net effect: inhibition typically dominates short-term
        combined_auc_ratio = max_inhibition * min_induction
        net_effect = combined_auc_ratio > 1.0 ? :increase : :decrease
    elseif !isempty(inhibition_ratios)
        # Multiple inhibitors
        if length(enzyme_effects) == 1
            # Same enzyme - use max (competitive)
            combined_auc_ratio = maximum(inhibition_ratios)
        else
            # Different enzymes - multiply (independent pathways blocked)
            combined_auc_ratio = prod(inhibition_ratios)
            # Cap at reasonable maximum
            combined_auc_ratio = min(combined_auc_ratio, 50.0)
        end
        net_effect = :increase
    elseif !isempty(induction_ratios)
        # Multiple inducers - take minimum (strongest effect)
        combined_auc_ratio = minimum(induction_ratios)
        net_effect = :decrease
    else
        net_effect = :neutral
    end

    # Determine clinical significance
    clinical_significance = if combined_auc_ratio >= 5.0 || combined_auc_ratio <= 0.2
        :strong
    elseif combined_auc_ratio >= 2.0 || combined_auc_ratio <= 0.5
        :moderate
    elseif combined_auc_ratio >= 1.25 || combined_auc_ratio <= 0.8
        :weak
    else
        :none
    end

    return MultiDDIResult(
        victim,
        perpetrators,
        individual_results,
        combined_auc_ratio,
        sqrt(combined_auc_ratio),
        net_effect,
        clinical_significance,
        warnings
    )
end

export predict_multi_ddi, MultiDDIResult

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

"""
Validate predictions against clinical DDI studies.
Returns accuracy metrics.
"""
function validate_predictions(; verbose::Bool = false)
    observed = Float64[]
    predicted = Float64[]
    details = []

    for study in CLINICAL_DDI_STUDIES
        # Skip transporter-only studies for now
        if study.enzyme in [:OATP1B1, :P_gp, :BCRP]
            continue
        end

        # Predict DDI
        result = predict_ddi(study.perpetrator, study.victim)

        push!(observed, study.auc_ratio)
        push!(predicted, result.auc_ratio)

        if verbose
            fold_error = max(result.auc_ratio / study.auc_ratio, study.auc_ratio / result.auc_ratio)
            push!(details, (
                perpetrator = study.perpetrator,
                victim = study.victim,
                observed = study.auc_ratio,
                predicted = result.auc_ratio,
                fold_error = fold_error
            ))
        end
    end

    # Calculate metrics
    n = length(observed)
    log_ratios = log10.(predicted ./ observed)

    afe = 10^mean(log_ratios)  # Average Fold Error (bias)
    aafe = 10^mean(abs.(log_ratios))  # Absolute Average Fold Error (precision)

    fold_errors = max.(predicted ./ observed, observed ./ predicted)
    within_2fold = count(fe -> fe <= 2.0, fold_errors) / n * 100
    within_3fold = count(fe -> fe <= 3.0, fold_errors) / n * 100

    metrics = (
        n = n,
        AFE = round(afe, digits=2),
        AAFE = round(aafe, digits=2),
        within_2fold = round(within_2fold, digits=1),
        within_3fold = round(within_3fold, digits=1)
    )

    if verbose
        return (metrics = metrics, details = details)
    else
        return metrics
    end
end

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

"""
Print a formatted DDI prediction report.
"""
function print_ddi_report(result::DDIResult)
    println("=" ^ 60)
    println("DDI PREDICTION REPORT")
    println("=" ^ 60)
    println("Perpetrator: $(result.perpetrator)")
    println("Victim:      $(result.victim)")
    println("Mechanism:   $(result.mechanism)")
    println("Enzyme:      $(result.enzyme)")
    println("-" ^ 60)
    println("Predicted AUC ratio:  $(round(result.auc_ratio, digits=2))")
    println("Predicted Cmax ratio: $(round(result.cmax_ratio, digits=2))")
    println("Clinical significance: $(result.clinical_significance)")
    println("Confidence: $(result.confidence)")

    if !isempty(result.warnings)
        println("-" ^ 60)
        println("Warnings:")
        for w in result.warnings
            println("  • $w")
        end
    end
    println("=" ^ 60)
end

# =============================================================================
# DDI RISK CLASSIFICATION SYSTEM
# =============================================================================

"""
DDI risk assessment result structure.
"""
struct DDIRiskAssessment
    perpetrator::Symbol
    victim::Symbol
    auc_ratio::Float64
    risk_level::Symbol          # :contraindicated, :major, :moderate, :minor, :none
    clinical_action::String
    monitoring_required::Bool
    dose_adjustment::String
    alternative_drugs::Vector{Symbol}
    evidence_level::Symbol      # :high, :moderate, :low
    references::Vector{String}
end

"""
FDA DDI classification thresholds.
"""
const DDI_RISK_THRESHOLDS = (
    contraindicated = 10.0,  # AUC ratio ≥10x or NTI drug with ≥2x
    major = 5.0,             # AUC ratio ≥5x
    moderate = 2.0,          # AUC ratio ≥2x
    minor = 1.25,            # AUC ratio ≥1.25x
)

"""
Narrow Therapeutic Index drugs requiring special consideration.
"""
const NTI_DRUGS = Set([
    :warfarin, :warfarin_s, :digoxin, :phenytoin, :carbamazepine,
    :lithium, :theophylline, :cyclosporine, :tacrolimus, :sirolimus,
    :methotrexate, :aminoglycosides, :vancomycin
])

"""
    classify_ddi_risk(perpetrator, victim; kwargs...)

Comprehensive DDI risk classification following FDA/EMA guidance.

Returns a DDIRiskAssessment with:
- Risk level (contraindicated, major, moderate, minor, none)
- Clinical action recommendations
- Monitoring requirements
- Dose adjustment guidance
- Alternative drug suggestions

# Example
```julia
risk = classify_ddi_risk(:itraconazole, :midazolam)
println(risk.risk_level)  # :major
println(risk.clinical_action)  # "Avoid combination or use alternative"
```
"""
function classify_ddi_risk(
    perpetrator::Symbol,
    victim::Symbol;
    include_transporters::Bool = true,
    patient_phenotype::Symbol = :NM
)
    # Get comprehensive prediction
    if include_transporters
        comp = predict_ddi_comprehensive(perpetrator, victim)
        auc_ratio = comp.result.auc_ratio
        mechanism = comp.result.mechanism
    else
        result = predict_ddi(perpetrator, victim)
        auc_ratio = result.auc_ratio
        mechanism = result.mechanism
    end

    # Check if victim is NTI
    is_nti = victim in NTI_DRUGS

    # Determine risk level
    risk_level, clinical_action, dose_adj = if auc_ratio >= DDI_RISK_THRESHOLDS.contraindicated || (is_nti && auc_ratio >= 2.0)
        (:contraindicated,
         "AVOID combination - use alternative therapy",
         "Do not co-administer")
    elseif auc_ratio >= DDI_RISK_THRESHOLDS.major
        (:major,
         "Avoid if possible; if essential, reduce dose significantly and monitor closely",
         "Reduce $(victim) dose by 75-90%")
    elseif auc_ratio >= DDI_RISK_THRESHOLDS.moderate
        (:moderate,
         "Use with caution; consider dose adjustment",
         "Consider reducing $(victim) dose by 50%")
    elseif auc_ratio >= DDI_RISK_THRESHOLDS.minor
        (:minor,
         "Monitor for increased effects; dose adjustment usually not required",
         "No routine adjustment needed")
    else
        (:none,
         "No clinically significant interaction expected",
         "No adjustment needed")
    end

    # Determine monitoring requirements
    monitoring = risk_level in [:contraindicated, :major, :moderate]

    # Suggest alternatives based on mechanism
    alternatives = suggest_alternatives(perpetrator, victim, mechanism)

    # Evidence level based on data quality
    evidence = if auc_ratio > 1.0 && haskey(FDA_CYP_INHIBITORS, :CYP3A4)
        :high
    else
        :moderate
    end

    return DDIRiskAssessment(
        perpetrator, victim, auc_ratio,
        risk_level, clinical_action, monitoring,
        dose_adj, alternatives, evidence,
        String[]  # References would come from literature DB
    )
end

"""
Suggest alternative drugs to avoid DDI.
"""
function suggest_alternatives(perpetrator::Symbol, victim::Symbol, mechanism::Symbol)
    alternatives = Symbol[]

    # CYP3A4 alternatives
    if mechanism in [:reversible, :mbi]
        # Suggest non-CYP3A4 substrates if victim is CYP3A4 substrate
        sub_params = get_substrate_params(victim)
        if !isnothing(sub_params)
            if hasproperty(sub_params, :fm_3a4) && sub_params.fm_3a4 > 0.5
                # High CYP3A4 dependency - suggest alternatives
                if victim in [:midazolam, :triazolam]
                    push!(alternatives, :lorazepam)  # Not CYP3A4
                    push!(alternatives, :oxazepam)
                elseif victim in [:simvastatin, :lovastatin]
                    push!(alternatives, :pravastatin)  # Minimal CYP metabolism
                    push!(alternatives, :rosuvastatin)
                elseif victim in [:felodipine, :nifedipine]
                    push!(alternatives, :amlodipine)  # Lower CYP3A4 dependency
                end
            end
        end
    end

    return alternatives
end

"""
    screen_drug_list(drugs::Vector{Symbol})

Screen a list of drugs for potential DDIs among all pairs.
Returns sorted list of interactions by risk level.
"""
function screen_drug_list(drugs::Vector{Symbol})
    interactions = DDIRiskAssessment[]

    for i in 1:length(drugs)
        for j in 1:length(drugs)
            if i != j
                risk = classify_ddi_risk(drugs[i], drugs[j])
                if risk.risk_level != :none
                    push!(interactions, risk)
                end
            end
        end
    end

    # Sort by risk level
    risk_order = Dict(:contraindicated => 1, :major => 2, :moderate => 3, :minor => 4)
    sort!(interactions, by = x -> get(risk_order, x.risk_level, 5))

    return interactions
end

"""
Print formatted DDI risk report.
"""
function print_risk_report(risk::DDIRiskAssessment)
    println("=" ^ 70)
    println("DDI RISK ASSESSMENT REPORT")
    println("=" ^ 70)
    println("Perpetrator: $(risk.perpetrator)")
    println("Victim:      $(risk.victim)")
    println("-" ^ 70)

    # Risk level with color indicator
    level_str = uppercase(string(risk.risk_level))
    println("RISK LEVEL:  $level_str")
    println("AUC Ratio:   $(round(risk.auc_ratio, digits=1))x")
    println("-" ^ 70)
    println("Clinical Action: $(risk.clinical_action)")
    println("Dose Adjustment: $(risk.dose_adjustment)")
    println("Monitoring Required: $(risk.monitoring_required ? "YES" : "No")")

    if !isempty(risk.alternative_drugs)
        println("-" ^ 70)
        println("Alternative drugs to consider:")
        for alt in risk.alternative_drugs
            println("  • $alt")
        end
    end
    println("=" ^ 70)
end

export classify_ddi_risk, screen_drug_list, DDIRiskAssessment, print_risk_report
export NTI_DRUGS, DDI_RISK_THRESHOLDS

end # module DDIPrediction
