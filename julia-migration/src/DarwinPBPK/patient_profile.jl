"""
PatientProfile - Comprehensive patient demographics and physiological scaling

Handles:
- Age (neonates → elderly)
- Sex (male/female)
- Weight, height, BMI
- Disease states (liver, kidney, sepsis, pregnancy, obesity)
- Physiological parameter scaling
"""

module PatientProfile

using Unitful

export PatientData, create_patient, scale_parameters

# Disease state enumeration
@enum DiseaseState begin
    HEALTHY = 0
    LIVER_CIRRHOSIS = 1
    LIVER_HEPATITIS = 2
    LIVER_NAFLD = 3
    KIDNEY_CKD_G1 = 4
    KIDNEY_CKD_G2 = 5
    KIDNEY_CKD_G3A = 6
    KIDNEY_CKD_G3B = 7
    KIDNEY_CKD_G4 = 8
    KIDNEY_CKD_G5 = 9
    KIDNEY_AKI = 10
    SEPSIS = 11
    PREGNANCY = 12
    OBESITY = 13
    ANEMIA = 14
end

"""
PatientData - Complete patient demographics and physiological state

Fields:
- age::Float64 - Age in years
- sex::String - "M" or "F"
- weight::Float64 - Body weight in kg
- height::Float64 - Height in cm
- bmi::Float64 - Body mass index (calculated)
- disease_states::Vector{DiseaseState} - Active disease states
- hematocrit::Float64 - Hematocrit (0.35-0.55)
- albumin::Float64 - Plasma albumin (g/L)
- alpha1_agp::Float64 - α1-acid glycoprotein (g/L)
- gfr::Float64 - Glomerular filtration rate (mL/min)
- liver_function::Float64 - Liver function score (0-1)
"""
mutable struct PatientData
    age::Float64  # years
    sex::String  # "M" or "F"
    weight::Float64  # kg
    height::Float64  # cm
    bmi::Float64  # kg/m²
    disease_states::Vector{DiseaseState}
    
    # Physiological parameters
    hematocrit::Float64  # 0.35-0.55
    albumin::Float64  # g/L
    alpha1_agp::Float64  # g/L
    gfr::Float64  # mL/min
    liver_function::Float64  # 0-1
    
    # Calculated volumes
    plasma_volume::Float64  # L
    blood_volume::Float64  # L
end

"""
create_patient(age, sex, weight, height; disease_states=[], ...)

Create a patient profile with automatic physiological scaling
"""
function create_patient(
    age::Float64,
    sex::String,
    weight::Float64,
    height::Float64;
    disease_states::Vector{DiseaseState} = DiseaseState[],
    hematocrit::Union{Float64, Nothing} = nothing,
    albumin::Union{Float64, Nothing} = nothing,
    gfr::Union{Float64, Nothing} = nothing
)

    # Validate inputs
    @assert age >= 0 "Age must be >= 0"
    @assert sex in ["M", "F"] "Sex must be 'M' or 'F'"
    @assert weight > 0 "Weight must be > 0"
    @assert height > 0 "Height must be > 0"

    # Calculate BMI
    bmi = weight / ((height / 100)^2)

    # Scale physiological parameters based on age, sex, disease
    plasma_vol = scale_plasma_volume(age, sex, weight, disease_states)
    blood_vol = plasma_vol / 0.55  # Assuming hematocrit ~0.45

    # Default physiological values
    hct = hematocrit !== nothing ? hematocrit : default_hematocrit(age, sex, disease_states)
    alb = albumin !== nothing ? albumin : default_albumin(age, disease_states)
    agp = default_alpha1_agp(age, disease_states)
    gfr_val = gfr !== nothing ? gfr : default_gfr(age, weight, sex, disease_states)
    liver_fn = default_liver_function(age, disease_states)

    PatientData(
        age, sex, weight, height, bmi, disease_states,
        hct, alb, agp, gfr_val, liver_fn,
        plasma_vol, blood_vol
    )
end

# Scaling functions
function scale_plasma_volume(age::Float64, sex::String, weight::Float64, disease_states::Vector{DiseaseState})
    # Base plasma volume (5% of body weight for males, 4.3% for females)
    base_pv = sex == "M" ? 0.05 * weight : 0.043 * weight
    
    # Age adjustments (pediatric)
    if age < 1
        base_pv *= 0.8  # Neonates have lower plasma volume
    elseif age < 5
        base_pv *= 0.9
    end
    
    # Disease adjustments
    for disease in disease_states
        if disease == PREGNANCY
            base_pv *= 1.5  # Plasma volume ↑ 50% in pregnancy
        elseif disease == OBESITY
            base_pv *= 1.2  # Slight increase in obesity
        end
    end
    
    return base_pv
end

function default_hematocrit(age::Float64, sex::String, disease_states::Vector{DiseaseState})
    # Normal hematocrit
    hct = sex == "M" ? 0.45 : 0.40
    
    # Age adjustments
    if age < 1
        hct = 0.35
    elseif age < 5
        hct = 0.38
    end
    
    # Disease adjustments
    for disease in disease_states
        if disease == ANEMIA
            hct *= 0.75  # Anemia reduces hematocrit
        end
    end
    
    return hct
end

function default_albumin(age::Float64, disease_states::Vector{DiseaseState})
    # Normal albumin: 35-50 g/L
    alb = 40.0
    
    # Disease adjustments
    for disease in disease_states
        if disease == LIVER_CIRRHOSIS
            alb *= 0.5  # Albumin ↓ 50% in cirrhosis
        elseif disease == LIVER_HEPATITIS
            alb *= 0.7  # Albumin ↓ 30% in hepatitis
        elseif disease == SEPSIS
            alb *= 0.6  # Albumin ↓ 40% in sepsis
        elseif disease == PREGNANCY
            alb *= 0.8  # Albumin ↓ 20% in pregnancy
        end
    end
    
    return alb
end

function default_alpha1_agp(age::Float64, disease_states::Vector{DiseaseState})
    # Normal α1-AGP: 0.4-1.0 g/L
    agp = 0.7
    
    # Disease adjustments (acute phase reactant)
    for disease in disease_states
        if disease == SEPSIS
            agp *= 3.0  # α1-AGP ↑ 300% in sepsis
        elseif disease == LIVER_HEPATITIS
            agp *= 2.0  # α1-AGP ↑ 200% in hepatitis
        end
    end
    
    return agp
end

function default_gfr(age::Float64, weight::Float64, sex::String, disease_states::Vector{DiseaseState})
    # Cockcroft-Gault for adults
    if age >= 18
        gfr = sex == "M" ? 140 : 120
    else
        # Pediatric: Schwartz formula
        gfr = 0.41 * 170 / 90  # Simplified
    end
    
    # Disease adjustments
    for disease in disease_states
        if disease == KIDNEY_CKD_G1
            gfr = 90.0
        elseif disease == KIDNEY_CKD_G2
            gfr = 60.0
        elseif disease == KIDNEY_CKD_G3A
            gfr = 45.0
        elseif disease == KIDNEY_CKD_G3B
            gfr = 30.0
        elseif disease == KIDNEY_CKD_G4
            gfr = 15.0
        elseif disease == KIDNEY_CKD_G5
            gfr = 5.0
        elseif disease == KIDNEY_AKI
            gfr *= 0.5
        end
    end
    
    return gfr
end

function default_liver_function(age::Float64, disease_states::Vector{DiseaseState})
    # Normal liver function: 1.0
    liver_fn = 1.0
    
    # Disease adjustments
    for disease in disease_states
        if disease == LIVER_CIRRHOSIS
            liver_fn *= 0.3  # Liver function ↓ 70%
        elseif disease == LIVER_HEPATITIS
            liver_fn *= 0.6  # Liver function ↓ 40%
        elseif disease == LIVER_NAFLD
            liver_fn *= 0.8  # Liver function ↓ 20%
        end
    end
    
    return liver_fn
end

end # module PatientProfile
