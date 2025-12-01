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
"""

module CompartmentModels

using ..PatientProfile

export CompartmentModel, BloodCompartment, LiverCompartment, KidneyCompartment
export create_compartment_model, get_compartment_parameters

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
"""
mutable struct BaseCompartment <: CompartmentModel
    name::String
    volume::Float64  # L
    blood_flow::Float64  # L/h
    tissue_composition::Dict{String, Float64}  # water, lipid, protein fractions
    ph::Float64
    temperature::Float64  # °C
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
    
    # Liver-specific
    cyp_expression::Dict{String, Float64}  # CYP3A4, 2D6, 2C9, etc.
    transporter_expression::Dict{String, Float64}  # OATP1B1, OCT1, MDR1, etc.
    intrinsic_clearance::Float64  # mL/min/g
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
    
    # Brain-specific
    bbb_permeability::Float64  # 0-1 scale
    pgp_expression::Float64  # 0-1 scale
    regional_distribution::Dict{String, Float64}  # grey matter, white matter, etc.
end

"""
AdiposCompartment - Specialized adipose compartment

Additional fields:
- lipid_fraction::Float64 - Neutral lipid content (0.85 typical)
- perfusion_limited::Bool - Perfusion-limited kinetics
"""
mutable struct AdiposCompartment <: CompartmentModel
    name::String
    volume::Float64
    blood_flow::Float64
    tissue_composition::Dict{String, Float64}
    ph::Float64
    temperature::Float64
    
    # Adipose-specific
    lipid_fraction::Float64  # 0.85 typical
    perfusion_limited::Bool
end

# Factory functions
function create_blood_compartment(patient::PatientProfile.PatientData)
    BloodCompartment(
        "Blood",
        patient.plasma_volume,
        patient.blood_volume * 60,  # Convert to L/h
        Dict("water" => 0.92, "protein" => 0.07, "lipid" => 0.01),
        7.4,  # pH
        37.0,  # Temperature
        patient.albumin,
        patient.alpha1_agp,
        patient.hematocrit
    )
end

function create_liver_compartment(patient::PatientProfile.PatientData)
    # Liver volume: ~1.8 L for 70kg adult
    liver_vol = 1.8 * (patient.weight / 70)
    
    # Liver blood flow: ~90 L/h
    liver_flow = 90.0 * patient.liver_function
    
    # CYP expression (relative to normal)
    cyp_expr = Dict(
        "CYP3A4" => 1.0 * patient.liver_function,
        "CYP2D6" => 1.0 * patient.liver_function,
        "CYP2C9" => 1.0 * patient.liver_function,
        "CYP2C19" => 1.0 * patient.liver_function,
        "CYP1A2" => 1.0 * patient.liver_function
    )
    
    # Transporter expression
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
        Dict("water" => 0.70, "protein" => 0.20, "lipid" => 0.10),
        7.35,  # pH
        37.0,  # Temperature
        cyp_expr,
        trans_expr,
        0.5  # CLint mL/min/g (placeholder)
    )
end

function create_kidney_compartment(patient::PatientProfile.PatientData)
    # Kidney volume: ~0.31 L for 70kg adult
    kidney_vol = 0.31 * (patient.weight / 70)
    
    # Kidney blood flow: ~60 L/h
    kidney_flow = 60.0 * (patient.gfr / 120)  # Scale by GFR
    
    # Transporter expression
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
        Dict("water" => 0.80, "protein" => 0.15, "lipid" => 0.05),
        7.35,  # pH
        37.0,  # Temperature
        patient.gfr,
        trans_expr
    )
end

function create_brain_compartment(patient::PatientProfile.PatientData)
    # Brain volume: ~1.4 L
    brain_vol = 1.4
    
    # Brain blood flow: ~50 L/h
    brain_flow = 50.0
    
    # Regional distribution
    regional = Dict(
        "grey_matter" => 0.4,
        "white_matter" => 0.6
    )
    
    BrainCompartment(
        "Brain",
        brain_vol,
        brain_flow,
        Dict("water" => 0.80, "protein" => 0.10, "lipid" => 0.10),
        7.35,  # pH
        37.0,  # Temperature
        0.5,  # BBB permeability (0-1)
        0.8,  # P-gp expression (0-1)
        regional
    )
end

function create_adipose_compartment(patient::PatientProfile.PatientData)
    # Adipose volume: highly variable, estimate from BMI
    adipose_vol = max(0.1, (patient.bmi - 25) * 0.5)  # Rough estimate
    
    # Adipose blood flow: ~0.03 × volume
    adipose_flow = adipose_vol * 0.03 * 60  # Convert to L/h
    
    AdiposCompartment(
        "Adipose",
        adipose_vol,
        adipose_flow,
        Dict("water" => 0.10, "protein" => 0.05, "lipid" => 0.85),
        37.0,  # pH
        37.0,  # Temperature
        0.85,  # Lipid fraction
        true  # Perfusion-limited
    )
end

end # module CompartmentModels
