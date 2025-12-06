"""
PKSimParameters - Database interface for PK-Sim physiological parameters

This module provides:
1. Loading PK-Sim Human Reference Values from CSV
2. Organ-specific parameter structures
3. Scaling functions based on PK-Sim formulas
4. Validation against hardcoded values

Data source: PKSim_Human_Reference_Values.csv (1,127 parameters)
Reference: PK-Sim v11 (Bayer Technology Services)
"""

module PKSimParameters

using CSV
using DataFrames
using Statistics

export PKSimOrganParams, PKSimDatabase
export load_pksim_database, get_organ_params, get_physiological_value
export scale_organ_volume, scale_blood_flow, validate_parameters

"""
PKSimOrganParams - Physiological parameters for a specific organ

Based on PK-Sim human reference database structure
"""
struct PKSimOrganParams
    organ::String

    # Volume and flow
    volume_L::Union{Float64, Nothing}  # Organ volume (L) - calculated from scaling
    blood_flow_L_h::Union{Float64, Nothing}  # Blood flow (L/h)

    # Tissue fractions
    fraction_vascular::Float64  # Vascular fraction
    fraction_interstitial::Float64  # Interstitial fraction
    fraction_intracellular::Float64  # Intracellular fraction (calculated)

    # Tissue composition (from Vf parameters)
    vf_water::Float64  # Water fraction
    vf_protein::Float64  # Protein fraction
    vf_lipid::Float64  # Lipid fraction
    vf_neutral_lipid::Float64  # Neutral lipid fraction
    vf_phospholipid::Float64  # Phospholipid fraction

    # Physical properties
    density::Float64  # Tissue density (g/mL)
    ph::Union{Float64, Nothing}  # Tissue pH

    # Scaling
    allometric_scale_factor::Float64  # For volume scaling

    # Enzyme/transporter expression (organ-specific)
    enzyme_expression::Dict{String, Float64}
    transporter_expression::Dict{String, Float64}

    # Additional organ-specific parameters
    extra_params::Dict{String, Float64}
end

"""
PKSimDatabase - Container for all PK-Sim parameters
"""
struct PKSimDatabase
    raw_data::DataFrame
    organs::Vector{String}
    parameters::Dict{String, Dict{String, Float64}}
end

"""
    load_pksim_database(csv_path::String) -> PKSimDatabase

Load PK-Sim human reference values from CSV file
"""
function load_pksim_database(csv_path::String)
    # Load CSV
    df = CSV.read(csv_path, DataFrame)

    # Parse into structured format
    params_dict = Dict{String, Dict{String, Float64}}()

    for row in eachrow(df)
        container = row.container_name
        param_name = row.parameter_name
        value = row.default_value

        # Skip non-numeric values
        if ismissing(value) || !isa(value, Real)
            continue
        end

        # Initialize container dict if needed
        if !haskey(params_dict, container)
            params_dict[container] = Dict{String, Float64}()
        end

        # Store parameter
        params_dict[container][param_name] = Float64(value)
    end

    organs = unique(df.container_name)

    PKSimDatabase(df, organs, params_dict)
end

"""
    get_physiological_value(db::PKSimDatabase, container::String, parameter::String) -> Float64

Get a specific physiological parameter value from the database
"""
function get_physiological_value(db::PKSimDatabase, container::String, parameter::String)
    if !haskey(db.parameters, container)
        error("Container '$container' not found in PK-Sim database")
    end

    if !haskey(db.parameters[container], parameter)
        error("Parameter '$parameter' not found for container '$container'")
    end

    return db.parameters[container][parameter]
end

"""
    get_organ_params(db::PKSimDatabase, organ::String; weight::Float64=70.0,
                     cardiac_output::Float64=350.0) -> PKSimOrganParams

Extract organ-specific parameters from PK-Sim database with optional scaling

Arguments:
- db: PK-Sim database
- organ: Organ name (Liver, Kidney, Brain, Heart, Lung, Muscle, Fat, etc.)
- weight: Body weight (kg) for scaling (default 70 kg)
- cardiac_output: Cardiac output (L/h) for blood flow scaling (default 350 L/h)
"""
function get_organ_params(db::PKSimDatabase, organ::String;
                         weight::Float64=70.0, cardiac_output::Float64=350.0)

    if !haskey(db.parameters, organ)
        error("Organ '$organ' not found in PK-Sim database")
    end

    params = db.parameters[organ]

    # Extract tissue fractions
    frac_vascular = get(params, "Fraction vascular", 0.0)
    frac_interstitial = get(params, "Fraction interstitial", 0.0)
    frac_intracellular = 1.0 - frac_vascular - frac_interstitial

    # Extract tissue composition (Vf = volume fraction)
    vf_water = get(params, "Vf (water)", 0.7)
    vf_protein = get(params, "Vf (protein)", 0.2)
    vf_lipid = get(params, "Vf (lipid)", 0.1)
    vf_neutral_lipid = get(params, "Vf (neutral lipid)-RR", 0.0)
    vf_phospholipid = get(params, "Vf (phospholipid)-PT", 0.0)

    # Physical properties
    density = get(params, "Density (tissue)", 1.0)
    allometric_factor = get(params, "Allometric scale factor", 0.75)

    # pH (may not exist for all organs)
    ph = get(params, "pH", nothing)

    # Volume scaling (using allometric scaling)
    # V_organ = V_ref * (W / W_ref)^allometric_factor
    # Note: PK-Sim stores reference volumes for mouse, need human reference
    volume_L = scale_organ_volume(organ, weight, allometric_factor)

    # Blood flow scaling
    blood_flow_L_h = scale_blood_flow(organ, weight, cardiac_output)

    # Enzyme expression (liver-specific)
    enzyme_expr = Dict{String, Float64}()
    if organ == "Liver"
        # CYP expression would come from additional PK-Sim data
        # For now, use defaults that can be overridden
        enzyme_expr["CYP3A4"] = 1.0
        enzyme_expr["CYP2D6"] = 1.0
        enzyme_expr["CYP2C9"] = 1.0
        enzyme_expr["CYP2C19"] = 1.0
        enzyme_expr["CYP1A2"] = 1.0
    end

    # Transporter expression (liver and kidney)
    trans_expr = Dict{String, Float64}()
    if organ == "Liver"
        trans_expr["OATP1B1"] = 1.0
        trans_expr["OATP1B3"] = 1.0
        trans_expr["OCT1"] = 1.0
        trans_expr["MDR1"] = 1.0
    elseif organ == "Kidney"
        trans_expr["OAT1"] = 1.0
        trans_expr["OAT3"] = 1.0
        trans_expr["OCT2"] = 1.0
        trans_expr["MATE1"] = 1.0
    end

    # Extra parameters (store all remaining)
    extra = Dict{String, Float64}()
    for (key, val) in params
        if !(key in ["Fraction vascular", "Fraction interstitial", "Vf (water)",
                     "Vf (protein)", "Vf (lipid)", "Density (tissue)",
                     "Allometric scale factor", "pH"])
            extra[key] = val
        end
    end

    PKSimOrganParams(
        organ,
        volume_L,
        blood_flow_L_h,
        frac_vascular,
        frac_interstitial,
        frac_intracellular,
        vf_water,
        vf_protein,
        vf_lipid,
        vf_neutral_lipid,
        vf_phospholipid,
        density,
        ph,
        allometric_factor,
        enzyme_expr,
        trans_expr,
        extra
    )
end

"""
    scale_organ_volume(organ::String, weight::Float64, allometric_factor::Float64) -> Float64

Scale organ volume using PK-Sim allometric scaling formula

V_organ = V_ref * (BW / BW_ref)^allometric_factor

Reference values for 70 kg human (from PK-Sim documentation and literature)
"""
function scale_organ_volume(organ::String, weight::Float64, allometric_factor::Float64)
    # Reference volumes for 70 kg adult (L)
    # Source: ICRP Publication 89, PK-Sim v11 documentation
    reference_volumes = Dict(
        "Liver" => 1.8,
        "Kidney" => 0.31,  # Both kidneys
        "Brain" => 1.45,
        "Heart" => 0.33,
        "Lung" => 1.17,
        "Muscle" => 29.0,
        "Fat" => 18.2,
        "Bone" => 10.5,
        "Skin" => 3.3,
        "Spleen" => 0.19,
        "Pancreas" => 0.14,
        "SmallIntestine" => 1.07,
        "LargeIntestine" => 0.37,
        "Stomach" => 0.15,
        "Gonads" => 0.05
    )

    ref_weight = 70.0

    if !haskey(reference_volumes, organ)
        @warn "No reference volume for organ '$organ', returning nothing"
        return nothing
    end

    V_ref = reference_volumes[organ]
    V_scaled = V_ref * (weight / ref_weight)^allometric_factor

    return V_scaled
end

"""
    scale_blood_flow(organ::String, weight::Float64, cardiac_output::Float64) -> Float64

Scale organ blood flow using PK-Sim formulas

Q_organ = f_organ * CO

where f_organ is the organ blood flow fraction and CO is cardiac output
"""
function scale_blood_flow(organ::String, weight::Float64, cardiac_output::Float64)
    # Blood flow fractions (from PK-Sim human physiology)
    # These are fractions of cardiac output
    blood_flow_fractions = Dict(
        "Liver" => 0.255,  # Total hepatic blood flow (arterial + portal)
        "Kidney" => 0.175,
        "Brain" => 0.12,
        "Heart" => 0.04,
        "Lung" => 1.0,  # Receives full CO (pulmonary circulation)
        "Muscle" => 0.17,
        "Fat" => 0.05,
        "Bone" => 0.05,
        "Skin" => 0.05,
        "Spleen" => 0.03,
        "Pancreas" => 0.01,
        "SmallIntestine" => 0.10,
        "LargeIntestine" => 0.04,
        "Stomach" => 0.01,
        "Gonads" => 0.005
    )

    if !haskey(blood_flow_fractions, organ)
        @warn "No blood flow fraction for organ '$organ', returning nothing"
        return nothing
    end

    f_organ = blood_flow_fractions[organ]
    Q_organ = f_organ * cardiac_output

    return Q_organ
end

"""
    validate_parameters(db::PKSimDatabase, organ::String, hardcoded_params::Dict) -> Dict

Validate hardcoded parameters against PK-Sim database values

Returns a dictionary with validation results including:
- differences: Dict of parameter differences (%)
- warnings: Vector of parameters with >10% difference
- matches: Vector of parameters within 10%
"""
function validate_parameters(db::PKSimDatabase, organ::String, hardcoded_params::Dict)
    pksim_params = get_organ_params(db, organ)

    differences = Dict{String, Float64}()
    warnings = String[]
    matches = String[]

    # Compare volumes
    if haskey(hardcoded_params, "volume") && !isnothing(pksim_params.volume_L)
        hardcoded_vol = hardcoded_params["volume"]
        pksim_vol = pksim_params.volume_L
        diff_pct = abs(hardcoded_vol - pksim_vol) / pksim_vol * 100
        differences["volume"] = diff_pct

        if diff_pct > 10.0
            push!(warnings, "Volume differs by $(round(diff_pct, digits=1))%: hardcoded=$hardcoded_vol, PK-Sim=$pksim_vol")
        else
            push!(matches, "Volume matches within $(round(diff_pct, digits=1))%")
        end
    end

    # Compare blood flows
    if haskey(hardcoded_params, "blood_flow") && !isnothing(pksim_params.blood_flow_L_h)
        hardcoded_flow = hardcoded_params["blood_flow"]
        pksim_flow = pksim_params.blood_flow_L_h
        diff_pct = abs(hardcoded_flow - pksim_flow) / pksim_flow * 100
        differences["blood_flow"] = diff_pct

        if diff_pct > 10.0
            push!(warnings, "Blood flow differs by $(round(diff_pct, digits=1))%: hardcoded=$hardcoded_flow, PK-Sim=$pksim_flow")
        else
            push!(matches, "Blood flow matches within $(round(diff_pct, digits=1))%")
        end
    end

    # Compare tissue composition
    for (param_name, pksim_val) in [
        ("water", pksim_params.vf_water),
        ("protein", pksim_params.vf_protein),
        ("lipid", pksim_params.vf_lipid)
    ]
        if haskey(hardcoded_params, param_name)
            hardcoded_val = hardcoded_params[param_name]
            diff_pct = abs(hardcoded_val - pksim_val) / pksim_val * 100
            differences[param_name] = diff_pct

            if diff_pct > 10.0
                push!(warnings, "$param_name fraction differs by $(round(diff_pct, digits=1))%: hardcoded=$hardcoded_val, PK-Sim=$pksim_val")
            else
                push!(matches, "$param_name fraction matches within $(round(diff_pct, digits=1))%")
            end
        end
    end

    return Dict(
        "differences" => differences,
        "warnings" => warnings,
        "matches" => matches,
        "organ" => organ
    )
end

"""
    get_reference_cardiac_output(weight::Float64, age::Float64, sex::String) -> Float64

Calculate reference cardiac output based on physiological scaling

Uses PK-Sim formula: CO = CO_ref * (BW / BW_ref)^0.75

Default: 350 L/h for 70 kg adult
"""
function get_reference_cardiac_output(weight::Float64, age::Float64=30.0, sex::String="male")
    CO_ref = 350.0  # L/h for 70 kg adult
    BW_ref = 70.0

    # Allometric scaling
    CO = CO_ref * (weight / BW_ref)^0.75

    # Age correction (simplified, CO decreases with age)
    if age > 40
        age_factor = 1.0 - 0.005 * (age - 40)  # ~0.5% decrease per year after 40
        CO *= max(0.7, age_factor)
    end

    return CO
end

"""
    print_organ_summary(params::PKSimOrganParams)

Print a formatted summary of organ parameters
"""
function print_organ_summary(params::PKSimOrganParams)
    println("\n" * "="^60)
    println("PK-Sim Organ Parameters: $(params.organ)")
    println("="^60)

    if !isnothing(params.volume_L)
        println("Volume: $(round(params.volume_L, digits=2)) L")
    end

    if !isnothing(params.blood_flow_L_h)
        println("Blood flow: $(round(params.blood_flow_L_h, digits=1)) L/h")
    end

    println("\nTissue Fractions:")
    println("  Vascular: $(round(params.fraction_vascular * 100, digits=1))%")
    println("  Interstitial: $(round(params.fraction_interstitial * 100, digits=1))%")
    println("  Intracellular: $(round(params.fraction_intracellular * 100, digits=1))%")

    println("\nTissue Composition:")
    println("  Water: $(round(params.vf_water * 100, digits=1))%")
    println("  Protein: $(round(params.vf_protein * 100, digits=1))%")
    println("  Lipid: $(round(params.vf_lipid * 100, digits=1))%")

    if !isnothing(params.ph)
        println("\npH: $(round(params.ph, digits=2))")
    end

    println("\nDensity: $(params.density) g/mL")
    println("Allometric scale factor: $(params.allometric_scale_factor)")

    if !isempty(params.enzyme_expression)
        println("\nEnzyme Expression:")
        for (enz, expr) in params.enzyme_expression
            println("  $enz: $(round(expr, digits=2))")
        end
    end

    if !isempty(params.transporter_expression)
        println("\nTransporter Expression:")
        for (trans, expr) in params.transporter_expression
            println("  $trans: $(round(expr, digits=2))")
        end
    end

    println("="^60)
end

end # module PKSimParameters
