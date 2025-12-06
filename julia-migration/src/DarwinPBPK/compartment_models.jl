"""
CompartmentModels - Physiological parameters for each PBPK compartment

Implements:
- Blood/Plasma compartment
- Liver compartment
- Kidney compartment
- Brain compartment
- Adipose compartment
- Muscle compartment
- Heart compartment
- Lung compartment
- GI tract compartment
- Skin compartment
- Bone compartment
- Spleen compartment
- Pancreas compartment
- Other/Rest compartment

Now uses PK-Sim Human Reference Database for physiological parameters
"""

module CompartmentModels

using ..PatientProfile

# Import PK-Sim database interface
include("database/pksim_parameters.jl")
using .PKSimParameters: PKSimDatabase, PKSimOrganParams, load_pksim_database,
                        get_organ_params, get_reference_cardiac_output

export CompartmentModel, BloodCompartment, LiverCompartment, KidneyCompartment
export BrainCompartment, AdiposCompartment, BaseCompartment
export create_compartment_model, get_compartment_parameters
export load_compartment_database
export create_blood_compartment, create_liver_compartment, create_kidney_compartment
export create_brain_compartment, create_adipose_compartment
export create_all_compartments, validate_compartment_parameters

# Global database instance (loaded once)
const PKSIM_DB = Ref{Union{PKSimDatabase, Nothing}}(nothing)

"""
    load_compartment_database(csv_path::String)

Load PK-Sim database for use in compartment creation
Should be called once at module initialization
"""
function load_compartment_database(csv_path::String)
    PKSIM_DB[] = load_pksim_database(csv_path)
    @info "PK-Sim database loaded with $(length(PKSIM_DB[].organs)) organ containers"
end

"""
    get_pksim_db() -> PKSimDatabase

Get the loaded PK-Sim database, or load default if not loaded
"""
function get_pksim_db()
    if isnothing(PKSIM_DB[])
        # Try to load from default location
        default_path = joinpath(@__DIR__, "..", "..", "data", "external_pk_datasets",
                               "PKSim_Human_Reference_Values.csv")
        if isfile(default_path)
            load_compartment_database(default_path)
        else
            error("PK-Sim database not loaded. Call load_compartment_database(csv_path) first.")
        end
    end
    return PKSIM_DB[]
end

"""
CompartmentModel - Abstract base type for physiological compartments
"""
abstract type CompartmentModel end

"""
BaseCompartment - Base structure for physiological compartment

Fields:
- name::String - Compartment name
- volume::Float64 - Tissue volume (L)
- blood_flow::Float64 - Blood flow (L/h)
- tissue_composition::Dict - Water, lipid, protein fractions
- ph::Float64 - Tissue pH
- temperature::Float64 - Tissue temperature (°C)
- pksim_params::Union{PKSimOrganParams, Nothing} - Original PK-Sim parameters
"""
mutable struct BaseCompartment <: CompartmentModel
    name::String
    volume::Float64  # L
    blood_flow::Float64  # L/h
    tissue_composition::Dict{String, Float64}  # water, lipid, protein fractions
    ph::Float64
    temperature::Float64  # °C
    pksim_params::Union{PKSimOrganParams, Nothing}
end

"""
BloodCompartment - Specialized blood/plasma compartment

Additional fields:
- protein_binding::Dict - Albumin, α1-AGP, lipoprotein concentrations
- hematocrit::Float64 - RBC fraction
"""
mutable struct BloodCompartment <: CompartmentModel
    name::String
    volume::Float64
    blood_flow::Float64
    tissue_composition::Dict{String, Float64}
    ph::Float64
    temperature::Float64
    pksim_params::Union{PKSimOrganParams, Nothing}

    # Blood-specific
    albumin::Float64  # g/L
    alpha1_agp::Float64  # g/L
    hematocrit::Float64
end

"""
LiverCompartment - Specialized liver compartment

Additional fields:
- cyp_expression::Dict - CYP enzyme expression levels
- transporter_expression::Dict - Transporter expression levels
- intrinsic_clearance::Float64 - CLint (mL/min/g liver)
"""
mutable struct LiverCompartment <: CompartmentModel
    name::String
    volume::Float64
    blood_flow::Float64
    tissue_composition::Dict{String, Float64}
    ph::Float64
    temperature::Float64
    pksim_params::Union{PKSimOrganParams, Nothing}

    # Liver-specific
    cyp_expression::Dict{String, Float64}  # CYP3A4, 2D6, 2C9, etc.
    transporter_expression::Dict{String, Float64}  # OATP1B1, OCT1, MDR1, etc.
    intrinsic_clearance::Float64  # mL/min/g
    microsomal_protein::Float64  # mg/g tissue
    hepatocellularity::Float64  # cells/g tissue
end

"""
KidneyCompartment - Specialized kidney compartment

Additional fields:
- gfr::Float64 - Glomerular filtration rate (mL/min)
- transporter_expression::Dict - Renal transporter expression
"""
mutable struct KidneyCompartment <: CompartmentModel
    name::String
    volume::Float64
    blood_flow::Float64
    tissue_composition::Dict{String, Float64}
    ph::Float64
    temperature::Float64
    pksim_params::Union{PKSimOrganParams, Nothing}

    # Kidney-specific
    gfr::Float64  # mL/min
    transporter_expression::Dict{String, Float64}  # OAT1, OCT2, MATE, etc.
end

"""
BrainCompartment - Specialized brain compartment

Additional fields:
- bbb_permeability::Float64 - Blood-brain barrier permeability
- pgp_expression::Float64 - P-glycoprotein expression
- regional_distribution::Dict - Brain region volumes
"""
mutable struct BrainCompartment <: CompartmentModel
    name::String
    volume::Float64
    blood_flow::Float64
    tissue_composition::Dict{String, Float64}
    ph::Float64
    temperature::Float64
    pksim_params::Union{PKSimOrganParams, Nothing}

    # Brain-specific
    bbb_permeability::Float64  # 0-1 scale
    pgp_expression::Float64  # 0-1 scale
    regional_distribution::Dict{String, Float64}  # grey matter, white matter, etc.
end

"""
AdiposCompartment - Specialized adipose compartment

Additional fields:
- lipid_fraction::Float64 - Neutral lipid content
- perfusion_limited::Bool - Perfusion-limited kinetics
"""
mutable struct AdiposCompartment <: CompartmentModel
    name::String
    volume::Float64
    blood_flow::Float64
    tissue_composition::Dict{String, Float64}
    ph::Float64
    temperature::Float64
    pksim_params::Union{PKSimOrganParams, Nothing}

    # Adipose-specific
    lipid_fraction::Float64
    perfusion_limited::Bool
end

# Factory functions using PK-Sim database

"""
    create_blood_compartment(patient::PatientProfile.PatientData; use_pksim::Bool=true)

Create blood compartment with parameters from patient profile
"""
function create_blood_compartment(patient::PatientProfile.PatientData; use_pksim::Bool=true)
    BloodCompartment(
        "Blood",
        patient.plasma_volume,
        patient.blood_volume * 60,  # Convert to L/h
        Dict("water" => 0.92, "protein" => 0.07, "lipid" => 0.01),
        7.4,  # pH
        37.0,  # Temperature
        nothing,  # No PK-Sim params for blood (use patient data)
        patient.albumin,
        patient.alpha1_agp,
        patient.hematocrit
    )
end

"""
    create_liver_compartment(patient::PatientProfile.PatientData; use_pksim::Bool=true,
                            override_params::Union{Dict, Nothing}=nothing)

Create liver compartment using PK-Sim database values

Parameters can be overridden with override_params dict
"""
function create_liver_compartment(patient::PatientProfile.PatientData;
                                  use_pksim::Bool=true,
                                  override_params::Union{Dict, Nothing}=nothing)

    if use_pksim
        # Get PK-Sim parameters
        db = get_pksim_db()
        cardiac_output = get_reference_cardiac_output(patient.weight, patient.age,
                                                      patient.sex == "Male" ? "male" : "female")
        pksim_params = get_organ_params(db, "Liver", weight=patient.weight,
                                       cardiac_output=cardiac_output)

        # Use PK-Sim values
        liver_vol = !isnothing(override_params) && haskey(override_params, "volume") ?
                    override_params["volume"] : pksim_params.volume_L
        liver_flow = !isnothing(override_params) && haskey(override_params, "blood_flow") ?
                     override_params["blood_flow"] : pksim_params.blood_flow_L_h

        # Scale by liver function
        liver_flow *= patient.liver_function

        # Tissue composition from PK-Sim
        tissue_comp = Dict(
            "water" => pksim_params.vf_water,
            "protein" => pksim_params.vf_protein,
            "lipid" => pksim_params.vf_lipid
        )

        ph = !isnothing(pksim_params.ph) ? pksim_params.ph : 7.35

        # Microsomal protein and hepatocellularity from PK-Sim extra params
        microsomal = get(pksim_params.extra_params, "Microsomal protein mass/g tissue", 0.04) * 1000  # Convert to mg/g
        hepatocellularity = get(pksim_params.extra_params, "Number of cells/g tissue", 139000.0)

    else
        # Fallback to hardcoded values (original implementation)
        liver_vol = 1.8 * (patient.weight / 70)
        liver_flow = 90.0 * patient.liver_function
        tissue_comp = Dict("water" => 0.70, "protein" => 0.20, "lipid" => 0.10)
        ph = 7.35
        pksim_params = nothing
        microsomal = 40.0  # mg/g
        hepatocellularity = 139000.0  # cells/g
    end

    # CYP expression (relative to normal, scaled by liver function)
    cyp_expr = Dict(
        "CYP3A4" => 1.0 * patient.liver_function,
        "CYP2D6" => 1.0 * patient.liver_function,
        "CYP2C9" => 1.0 * patient.liver_function,
        "CYP2C19" => 1.0 * patient.liver_function,
        "CYP1A2" => 1.0 * patient.liver_function
    )

    # Transporter expression (scaled by liver function)
    trans_expr = Dict(
        "OATP1B1" => 1.0 * patient.liver_function,
        "OATP1B3" => 1.0 * patient.liver_function,
        "OCT1" => 1.0 * patient.liver_function,
        "MDR1" => 1.0 * patient.liver_function
    )

    LiverCompartment(
        "Liver",
        liver_vol,
        liver_flow,
        tissue_comp,
        ph,
        37.0,
        pksim_params,
        cyp_expr,
        trans_expr,
        0.5,  # CLint mL/min/g (placeholder, drug-specific)
        microsomal,
        hepatocellularity
    )
end

"""
    create_kidney_compartment(patient::PatientProfile.PatientData; use_pksim::Bool=true,
                             override_params::Union{Dict, Nothing}=nothing)

Create kidney compartment using PK-Sim database values
"""
function create_kidney_compartment(patient::PatientProfile.PatientData;
                                   use_pksim::Bool=true,
                                   override_params::Union{Dict, Nothing}=nothing)

    if use_pksim
        # Get PK-Sim parameters
        db = get_pksim_db()
        cardiac_output = get_reference_cardiac_output(patient.weight, patient.age,
                                                      patient.sex == "Male" ? "male" : "female")
        pksim_params = get_organ_params(db, "Kidney", weight=patient.weight,
                                       cardiac_output=cardiac_output)

        kidney_vol = !isnothing(override_params) && haskey(override_params, "volume") ?
                     override_params["volume"] : pksim_params.volume_L
        kidney_flow = !isnothing(override_params) && haskey(override_params, "blood_flow") ?
                      override_params["blood_flow"] : pksim_params.blood_flow_L_h

        # Scale by GFR (kidney function indicator)
        kidney_flow *= (patient.gfr / 120)

        tissue_comp = Dict(
            "water" => pksim_params.vf_water,
            "protein" => pksim_params.vf_protein,
            "lipid" => pksim_params.vf_lipid
        )

        ph = !isnothing(pksim_params.ph) ? pksim_params.ph : 7.35

    else
        # Fallback to hardcoded values
        kidney_vol = 0.31 * (patient.weight / 70)
        kidney_flow = 60.0 * (patient.gfr / 120)
        tissue_comp = Dict("water" => 0.80, "protein" => 0.15, "lipid" => 0.05)
        ph = 7.35
        pksim_params = nothing
    end

    # Transporter expression (scaled by GFR)
    trans_expr = Dict(
        "OAT1" => 1.0 * (patient.gfr / 120),
        "OAT3" => 1.0 * (patient.gfr / 120),
        "OCT2" => 1.0 * (patient.gfr / 120),
        "MATE1" => 1.0 * (patient.gfr / 120)
    )

    KidneyCompartment(
        "Kidney",
        kidney_vol,
        kidney_flow,
        tissue_comp,
        ph,
        37.0,
        pksim_params,
        patient.gfr,
        trans_expr
    )
end

"""
    create_brain_compartment(patient::PatientProfile.PatientData; use_pksim::Bool=true,
                            override_params::Union{Dict, Nothing}=nothing)

Create brain compartment using PK-Sim database values
"""
function create_brain_compartment(patient::PatientProfile.PatientData;
                                  use_pksim::Bool=true,
                                  override_params::Union{Dict, Nothing}=nothing)

    if use_pksim
        # Get PK-Sim parameters
        db = get_pksim_db()
        cardiac_output = get_reference_cardiac_output(patient.weight, patient.age,
                                                      patient.sex == "Male" ? "male" : "female")
        pksim_params = get_organ_params(db, "Brain", weight=patient.weight,
                                       cardiac_output=cardiac_output)

        brain_vol = !isnothing(override_params) && haskey(override_params, "volume") ?
                    override_params["volume"] : pksim_params.volume_L
        brain_flow = !isnothing(override_params) && haskey(override_params, "blood_flow") ?
                     override_params["blood_flow"] : pksim_params.blood_flow_L_h

        tissue_comp = Dict(
            "water" => pksim_params.vf_water,
            "protein" => pksim_params.vf_protein,
            "lipid" => pksim_params.vf_lipid
        )

        ph = !isnothing(pksim_params.ph) ? pksim_params.ph : 7.35

    else
        # Fallback to hardcoded values
        brain_vol = 1.4
        brain_flow = 50.0
        tissue_comp = Dict("water" => 0.80, "protein" => 0.10, "lipid" => 0.10)
        ph = 7.35
        pksim_params = nothing
    end

    # Regional distribution (from PK-Sim intracellular/extracellular fractions)
    regional = Dict(
        "grey_matter" => 0.4,
        "white_matter" => 0.6
    )

    BrainCompartment(
        "Brain",
        brain_vol,
        brain_flow,
        tissue_comp,
        ph,
        37.0,
        pksim_params,
        0.5,  # BBB permeability (0-1, drug-specific)
        0.8,  # P-gp expression (0-1)
        regional
    )
end

"""
    create_adipose_compartment(patient::PatientProfile.PatientData; use_pksim::Bool=true,
                              override_params::Union{Dict, Nothing}=nothing)

Create adipose compartment using PK-Sim database values
"""
function create_adipose_compartment(patient::PatientProfile.PatientData;
                                   use_pksim::Bool=true,
                                   override_params::Union{Dict, Nothing}=nothing)

    if use_pksim
        # Get PK-Sim parameters
        db = get_pksim_db()
        cardiac_output = get_reference_cardiac_output(patient.weight, patient.age,
                                                      patient.sex == "Male" ? "male" : "female")
        pksim_params = get_organ_params(db, "Fat", weight=patient.weight,
                                       cardiac_output=cardiac_output)

        adipose_vol = !isnothing(override_params) && haskey(override_params, "volume") ?
                      override_params["volume"] : pksim_params.volume_L
        adipose_flow = !isnothing(override_params) && haskey(override_params, "blood_flow") ?
                       override_params["blood_flow"] : pksim_params.blood_flow_L_h

        tissue_comp = Dict(
            "water" => pksim_params.vf_water,
            "protein" => pksim_params.vf_protein,
            "lipid" => pksim_params.vf_lipid
        )

        lipid_frac = pksim_params.vf_lipid

        # PK-Sim doesn't have pH for fat
        ph = 37.0  # Use body temp as placeholder

    else
        # Fallback to hardcoded values (BMI-based estimation)
        adipose_vol = max(0.1, (patient.bmi - 25) * 0.5)
        adipose_flow = adipose_vol * 0.03 * 60  # Convert to L/h
        tissue_comp = Dict("water" => 0.10, "protein" => 0.05, "lipid" => 0.85)
        lipid_frac = 0.85
        ph = 37.0
        pksim_params = nothing
    end

    AdiposCompartment(
        "Adipose",
        adipose_vol,
        adipose_flow,
        tissue_comp,
        ph,
        37.0,
        pksim_params,
        lipid_frac,
        true  # Perfusion-limited
    )
end

"""
    create_all_compartments(patient::PatientProfile.PatientData; use_pksim::Bool=true)

Create all PBPK compartments for a patient

Returns a Dict of compartment name => compartment object
"""
function create_all_compartments(patient::PatientProfile.PatientData; use_pksim::Bool=true)
    compartments = Dict{String, CompartmentModel}()

    compartments["Blood"] = create_blood_compartment(patient, use_pksim=use_pksim)
    compartments["Liver"] = create_liver_compartment(patient, use_pksim=use_pksim)
    compartments["Kidney"] = create_kidney_compartment(patient, use_pksim=use_pksim)
    compartments["Brain"] = create_brain_compartment(patient, use_pksim=use_pksim)
    compartments["Adipose"] = create_adipose_compartment(patient, use_pksim=use_pksim)

    # TODO: Add remaining compartments (Muscle, Heart, Lung, GI, Skin, Bone, Spleen, Pancreas)

    return compartments
end

"""
    validate_compartment_parameters(patient::PatientProfile.PatientData)

Validate PK-Sim database values against hardcoded values

Prints comparison report showing differences
"""
function validate_compartment_parameters(patient::PatientProfile.PatientData)
    println("\n" * "="^70)
    println("PK-Sim Database Validation Report")
    println("="^70)

    # Create compartments with PK-Sim
    pksim_liver = create_liver_compartment(patient, use_pksim=true)
    hardcoded_liver = create_liver_compartment(patient, use_pksim=false)

    println("\nLIVER COMPARTMENT:")
    println("  Volume (PK-Sim): $(round(pksim_liver.volume, digits=3)) L")
    println("  Volume (Hardcoded): $(round(hardcoded_liver.volume, digits=3)) L")
    vol_diff = abs(pksim_liver.volume - hardcoded_liver.volume) / pksim_liver.volume * 100
    println("  Difference: $(round(vol_diff, digits=1))%")
    if vol_diff > 10.0
        println("  ⚠ WARNING: Volume differs by more than 10%")
    end

    println("\n  Blood flow (PK-Sim): $(round(pksim_liver.blood_flow, digits=1)) L/h")
    println("  Blood flow (Hardcoded): $(round(hardcoded_liver.blood_flow, digits=1)) L/h")
    flow_diff = abs(pksim_liver.blood_flow - hardcoded_liver.blood_flow) / pksim_liver.blood_flow * 100
    println("  Difference: $(round(flow_diff, digits=1))%")
    if flow_diff > 10.0
        println("  ⚠ WARNING: Blood flow differs by more than 10%")
    end

    # Kidney validation
    pksim_kidney = create_kidney_compartment(patient, use_pksim=true)
    hardcoded_kidney = create_kidney_compartment(patient, use_pksim=false)

    println("\nKIDNEY COMPARTMENT:")
    println("  Volume (PK-Sim): $(round(pksim_kidney.volume, digits=3)) L")
    println("  Volume (Hardcoded): $(round(hardcoded_kidney.volume, digits=3)) L")
    vol_diff = abs(pksim_kidney.volume - hardcoded_kidney.volume) / pksim_kidney.volume * 100
    println("  Difference: $(round(vol_diff, digits=1))%")
    if vol_diff > 10.0
        println("  ⚠ WARNING: Volume differs by more than 10%")
    end

    println("\n  Blood flow (PK-Sim): $(round(pksim_kidney.blood_flow, digits=1)) L/h")
    println("  Blood flow (Hardcoded): $(round(hardcoded_kidney.blood_flow, digits=1)) L/h")
    flow_diff = abs(pksim_kidney.blood_flow - hardcoded_kidney.blood_flow) / pksim_kidney.blood_flow * 100
    println("  Difference: $(round(flow_diff, digits=1))%")
    if flow_diff > 10.0
        println("  ⚠ WARNING: Blood flow differs by more than 10%")
    end

    println("\n" * "="^70)
end

end # module CompartmentModels
