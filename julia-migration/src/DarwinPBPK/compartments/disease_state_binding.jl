"""
Disease State Binding Adjustments Module

Comprehensive adjustments to plasma protein binding and drug distribution
based on pathophysiological conditions.

Disease States Covered:
- Renal impairment (uremia, CKD stages)
- Hepatic impairment (cirrhosis, hepatitis)
- Pregnancy (trimester-specific)
- Pediatric (age-specific)
- Geriatric (age-related changes)
- Critical illness (sepsis, burns, trauma)
- Inflammatory conditions (RA, IBD, infections)
- Metabolic disorders (diabetes, obesity)
- Oncology (cancer cachexia, hypoalbuminemia)

References:
- Benet LZ (2002) - Changes in binding in disease states
- Roberts JA (2014) - PK in critically ill
- Abduljalil K (2012) - Pregnancy PBPK
- Edginton AN (2006) - Pediatric PBPK

Author: Darwin PBPK Platform
Date: 2025-12-05
"""
module DiseaseStateBinding

using Statistics

export DiseaseState, PlasmaProteinState, BindingAdjustments
export apply_disease_adjustments, calculate_adjusted_fu
export create_disease_state, get_disease_binding_factors
export DISEASE_BINDING_DATABASE

# ============================================================================
# CONSTANTS - Normal Reference Values
# ============================================================================

# Normal plasma protein concentrations
const NORMAL_ALBUMIN = 40.0          # g/L (4.0 g/dL)
const NORMAL_AAG = 0.8               # g/L (80 mg/dL)
const NORMAL_GLOBULINS = 25.0        # g/L
const NORMAL_TOTAL_PROTEIN = 70.0    # g/L

# Normal physiological parameters
const NORMAL_GFR = 100.0             # mL/min/1.73m²
const NORMAL_BILIRUBIN = 10.0        # μmol/L (0.6 mg/dL)
const NORMAL_CREATININE = 80.0       # μmol/L (0.9 mg/dL)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    PlasmaProteinState

Current state of plasma proteins.

# Fields
- `albumin::Float64`: Albumin concentration (g/L)
- `aag::Float64`: α1-acid glycoprotein concentration (g/L)
- `globulins::Float64`: Total globulins (g/L)
- `bilirubin::Float64`: Total bilirubin (μmol/L)
- `urea::Float64`: Blood urea nitrogen (mmol/L)
- `creatinine::Float64`: Serum creatinine (μmol/L)
- `albumin_function::Float64`: Functional albumin (0-1, accounts for modifications)
- `aag_function::Float64`: Functional AAG (0-1)
"""
struct PlasmaProteinState
    albumin::Float64
    aag::Float64
    globulins::Float64
    bilirubin::Float64
    urea::Float64
    creatinine::Float64
    albumin_function::Float64
    aag_function::Float64

    function PlasmaProteinState(;
        albumin = NORMAL_ALBUMIN,
        aag = NORMAL_AAG,
        globulins = NORMAL_GLOBULINS,
        bilirubin = NORMAL_BILIRUBIN,
        urea = 5.0,
        creatinine = NORMAL_CREATININE,
        albumin_function = 1.0,
        aag_function = 1.0
    )
        new(albumin, aag, globulins, bilirubin, urea, creatinine,
            albumin_function, aag_function)
    end
end

"""
    DiseaseState

Comprehensive disease state definition.

# Fields
- `name::Symbol`: Disease identifier
- `severity::Symbol`: :mild, :moderate, :severe
- `protein_state::PlasmaProteinState`: Plasma protein changes
- `gfr::Float64`: Glomerular filtration rate (mL/min)
- `hepatic_function::Float64`: Hepatic function (0-1)
- `cardiac_output::Float64`: Cardiac output (L/min, normal ~5)
- `hematocrit::Float64`: Hematocrit (fraction, normal 0.42)
- `volume_status::Symbol`: :euvolemic, :hypovolemic, :hypervolemic
- `inflammatory_state::Symbol`: :none, :mild, :moderate, :severe
- `notes::String`: Clinical notes
"""
struct DiseaseState
    name::Symbol
    severity::Symbol
    protein_state::PlasmaProteinState
    gfr::Float64
    hepatic_function::Float64
    cardiac_output::Float64
    hematocrit::Float64
    volume_status::Symbol
    inflammatory_state::Symbol
    notes::String

    function DiseaseState(name::Symbol;
        severity = :moderate,
        protein_state = PlasmaProteinState(),
        gfr = NORMAL_GFR,
        hepatic_function = 1.0,
        cardiac_output = 5.0,
        hematocrit = 0.42,
        volume_status = :euvolemic,
        inflammatory_state = :none,
        notes = ""
    )
        new(name, severity, protein_state, gfr, hepatic_function,
            cardiac_output, hematocrit, volume_status, inflammatory_state, notes)
    end
end

"""
    BindingAdjustments

Calculated adjustments to apply to drug binding.

# Fields
- `fu_acidic_factor::Float64`: Multiplier for acidic drug fu
- `fu_basic_factor::Float64`: Multiplier for basic drug fu
- `fu_neutral_factor::Float64`: Multiplier for neutral drug fu
- `vd_factor::Float64`: Volume of distribution adjustment
- `clearance_factor::Float64`: Clearance adjustment
- `half_life_factor::Float64`: Half-life adjustment
"""
struct BindingAdjustments
    fu_acidic_factor::Float64
    fu_basic_factor::Float64
    fu_neutral_factor::Float64
    vd_factor::Float64
    clearance_factor::Float64
    half_life_factor::Float64
end

# ============================================================================
# DISEASE STATE FACTORIES
# ============================================================================

"""
    create_disease_state(disease::Symbol; severity::Symbol = :moderate)

Create a predefined disease state.

Supported diseases:
- Renal: :ckd_stage1-5, :esrd, :dialysis, :aki
- Hepatic: :cirrhosis_child_a/b/c, :hepatitis, :nafld
- Pregnancy: :pregnancy_t1/t2/t3, :postpartum
- Age: :neonate, :infant, :child, :adolescent, :elderly
- Critical: :sepsis, :burn, :trauma, :covid_severe
- Metabolic: :diabetes_t1/t2, :obesity, :hypothyroid, :hyperthyroid
- Inflammatory: :rheumatoid_arthritis, :ibd, :sle
- Oncology: :cancer_cachexia, :chemotherapy
"""
function create_disease_state(disease::Symbol; severity::Symbol = :moderate)

    # =========================================
    # RENAL DISEASES
    # =========================================
    if disease == :ckd_stage1
        return DiseaseState(disease;
            severity = :mild,
            gfr = 90.0,
            protein_state = PlasmaProteinState(albumin = 38.0),
            notes = "CKD Stage 1: GFR ≥90, minimal protein changes"
        )
    elseif disease == :ckd_stage2
        return DiseaseState(disease;
            severity = :mild,
            gfr = 75.0,
            protein_state = PlasmaProteinState(
                albumin = 36.0,
                urea = 8.0,
                creatinine = 120.0,
                albumin_function = 0.95
            ),
            notes = "CKD Stage 2: GFR 60-89"
        )
    elseif disease == :ckd_stage3
        return DiseaseState(disease;
            severity = :moderate,
            gfr = 45.0,
            protein_state = PlasmaProteinState(
                albumin = 34.0,
                urea = 12.0,
                creatinine = 180.0,
                albumin_function = 0.85  # Uremic toxins affect binding
            ),
            notes = "CKD Stage 3: GFR 30-59, uremic toxin accumulation"
        )
    elseif disease == :ckd_stage4
        return DiseaseState(disease;
            severity = :severe,
            gfr = 22.0,
            protein_state = PlasmaProteinState(
                albumin = 32.0,
                urea = 20.0,
                creatinine = 300.0,
                albumin_function = 0.70  # Significant uremic displacement
            ),
            hematocrit = 0.32,  # Anemia of CKD
            notes = "CKD Stage 4: GFR 15-29, significant uremic effects"
        )
    elseif disease == :ckd_stage5 || disease == :esrd
        return DiseaseState(disease;
            severity = :severe,
            gfr = 8.0,
            protein_state = PlasmaProteinState(
                albumin = 30.0,
                urea = 35.0,
                creatinine = 600.0,
                albumin_function = 0.50  # Major uremic displacement
            ),
            hematocrit = 0.28,
            volume_status = :hypervolemic,
            notes = "ESRD: GFR <15, major binding displacement by uremic toxins"
        )
    elseif disease == :dialysis
        return DiseaseState(disease;
            severity = :severe,
            gfr = 5.0,
            protein_state = PlasmaProteinState(
                albumin = 32.0,  # Post-dialysis
                urea = 10.0,     # Cleared by dialysis
                creatinine = 400.0,
                albumin_function = 0.75  # Partially restored post-dialysis
            ),
            hematocrit = 0.30,
            notes = "Hemodialysis: intermittent correction of uremia"
        )
    elseif disease == :aki
        return DiseaseState(disease;
            severity = severity,
            gfr = severity == :mild ? 40.0 : (severity == :moderate ? 20.0 : 10.0),
            protein_state = PlasmaProteinState(
                albumin = 30.0,
                urea = 25.0,
                creatinine = 350.0,
                albumin_function = 0.65
            ),
            inflammatory_state = :moderate,
            notes = "AKI: rapid onset, often with inflammation"
        )
    elseif disease == :nephrotic_syndrome
        return DiseaseState(disease;
            severity = :severe,
            gfr = 60.0,  # May be preserved
            protein_state = PlasmaProteinState(
                albumin = 15.0,  # Severe hypoalbuminemia!
                aag = 1.2,       # May increase
                albumin_function = 0.90
            ),
            volume_status = :hypervolemic,
            notes = "Nephrotic: massive albumin loss, severe hypoalbuminemia"
        )

    # =========================================
    # HEPATIC DISEASES
    # =========================================
    elseif disease == :cirrhosis_child_a
        return DiseaseState(disease;
            severity = :mild,
            hepatic_function = 0.75,
            protein_state = PlasmaProteinState(
                albumin = 35.0,
                bilirubin = 30.0,
                albumin_function = 0.85
            ),
            notes = "Child-Pugh A: compensated cirrhosis"
        )
    elseif disease == :cirrhosis_child_b
        return DiseaseState(disease;
            severity = :moderate,
            hepatic_function = 0.50,
            protein_state = PlasmaProteinState(
                albumin = 28.0,
                bilirubin = 50.0,
                albumin_function = 0.70
            ),
            volume_status = :hypervolemic,  # Ascites
            notes = "Child-Pugh B: moderate cirrhosis with ascites"
        )
    elseif disease == :cirrhosis_child_c
        return DiseaseState(disease;
            severity = :severe,
            hepatic_function = 0.25,
            protein_state = PlasmaProteinState(
                albumin = 20.0,
                aag = 0.5,  # Also reduced synthesis
                bilirubin = 100.0,
                albumin_function = 0.50
            ),
            volume_status = :hypervolemic,
            cardiac_output = 7.0,  # Hyperdynamic circulation
            notes = "Child-Pugh C: decompensated cirrhosis"
        )
    elseif disease == :hepatitis_acute
        return DiseaseState(disease;
            severity = :moderate,
            hepatic_function = 0.60,
            protein_state = PlasmaProteinState(
                albumin = 32.0,
                aag = 1.2,  # Acute phase response
                bilirubin = 80.0,
                albumin_function = 0.80
            ),
            inflammatory_state = :moderate,
            notes = "Acute hepatitis: transient dysfunction"
        )
    elseif disease == :nafld
        return DiseaseState(disease;
            severity = :mild,
            hepatic_function = 0.85,
            protein_state = PlasmaProteinState(
                albumin = 38.0,
                albumin_function = 0.95
            ),
            notes = "NAFLD: minimal PK impact in early stages"
        )

    # =========================================
    # PREGNANCY
    # =========================================
    elseif disease == :pregnancy_t1
        return DiseaseState(disease;
            severity = :mild,
            protein_state = PlasmaProteinState(
                albumin = 36.0,  # Starting to decrease
                aag = 0.7,       # Decreases in pregnancy
                albumin_function = 1.0
            ),
            gfr = 120.0,  # GFR increases
            cardiac_output = 5.5,
            hematocrit = 0.38,
            notes = "Pregnancy T1: early physiological changes"
        )
    elseif disease == :pregnancy_t2
        return DiseaseState(disease;
            severity = :moderate,
            protein_state = PlasmaProteinState(
                albumin = 32.0,  # Dilutional hypoalbuminemia
                aag = 0.6,
                albumin_function = 1.0
            ),
            gfr = 140.0,
            cardiac_output = 6.5,
            hematocrit = 0.34,
            volume_status = :hypervolemic,
            notes = "Pregnancy T2: significant hemodilution"
        )
    elseif disease == :pregnancy_t3
        return DiseaseState(disease;
            severity = :moderate,
            protein_state = PlasmaProteinState(
                albumin = 28.0,  # Lowest point
                aag = 0.5,
                albumin_function = 1.0
            ),
            gfr = 150.0,
            cardiac_output = 7.0,
            hematocrit = 0.33,
            volume_status = :hypervolemic,
            notes = "Pregnancy T3: maximum physiological changes"
        )
    elseif disease == :preeclampsia
        return DiseaseState(disease;
            severity = :severe,
            protein_state = PlasmaProteinState(
                albumin = 25.0,
                aag = 1.0,  # Inflammation
                albumin_function = 0.80
            ),
            gfr = 80.0,  # Reduced in preeclampsia
            cardiac_output = 5.5,
            volume_status = :hypervolemic,
            inflammatory_state = :moderate,
            notes = "Preeclampsia: endothelial dysfunction"
        )

    # =========================================
    # PEDIATRIC / GERIATRIC
    # =========================================
    elseif disease == :neonate
        return DiseaseState(disease;
            severity = :mild,
            protein_state = PlasmaProteinState(
                albumin = 35.0,
                aag = 0.3,  # Very low in neonates
                bilirubin = 100.0,  # Physiological jaundice
                albumin_function = 0.70  # Fetal albumin has lower affinity
            ),
            gfr = 30.0,  # Immature renal function
            hepatic_function = 0.50,
            notes = "Neonate: immature organ function, fetal albumin"
        )
    elseif disease == :infant
        return DiseaseState(disease;
            severity = :mild,
            protein_state = PlasmaProteinState(
                albumin = 38.0,
                aag = 0.5,
                albumin_function = 0.85
            ),
            gfr = 70.0,
            hepatic_function = 0.75,
            notes = "Infant: maturing organ function"
        )
    elseif disease == :child
        return DiseaseState(disease;
            severity = :mild,
            protein_state = PlasmaProteinState(
                albumin = 42.0,  # May be slightly higher
                aag = 0.7,
                albumin_function = 0.95
            ),
            gfr = 120.0,  # Higher than adult (per BSA)
            hepatic_function = 0.90,
            notes = "Child: near-adult physiology"
        )
    elseif disease == :elderly
        return DiseaseState(disease;
            severity = :mild,
            protein_state = PlasmaProteinState(
                albumin = 35.0,  # Age-related decrease
                aag = 1.0,       # May increase
                albumin_function = 0.85  # Structural changes
            ),
            gfr = 60.0,  # Age-related GFR decline
            hepatic_function = 0.80,
            cardiac_output = 4.0,
            notes = "Elderly: age-related organ decline"
        )

    # =========================================
    # CRITICAL ILLNESS
    # =========================================
    elseif disease == :sepsis
        return DiseaseState(disease;
            severity = severity,
            protein_state = PlasmaProteinState(
                albumin = severity == :severe ? 18.0 : 25.0,
                aag = 2.5,  # Major acute phase response
                albumin_function = 0.50  # Capillary leak + modifications
            ),
            gfr = severity == :severe ? 30.0 : 60.0,
            hepatic_function = severity == :severe ? 0.40 : 0.70,
            cardiac_output = severity == :severe ? 8.0 : 6.5,
            volume_status = :hypovolemic,  # Despite edema (third spacing)
            inflammatory_state = :severe,
            notes = "Sepsis: major PK changes, AAG↑↑, albumin↓↓"
        )
    elseif disease == :burn
        return DiseaseState(disease;
            severity = severity,
            protein_state = PlasmaProteinState(
                albumin = severity == :severe ? 15.0 : 25.0,
                aag = 3.0,  # Extreme acute phase
                albumin_function = 0.40
            ),
            gfr = 150.0,  # Hyperdynamic early phase
            cardiac_output = 10.0,
            hematocrit = 0.50,  # Hemoconcentration
            volume_status = :hypovolemic,
            inflammatory_state = :severe,
            notes = "Burns: massive protein loss, hyperdynamic state"
        )
    elseif disease == :trauma
        return DiseaseState(disease;
            severity = severity,
            protein_state = PlasmaProteinState(
                albumin = 28.0,
                aag = 2.0,
                albumin_function = 0.70
            ),
            inflammatory_state = :moderate,
            notes = "Trauma: acute phase response"
        )
    elseif disease == :covid_severe
        return DiseaseState(disease;
            severity = :severe,
            protein_state = PlasmaProteinState(
                albumin = 25.0,
                aag = 2.0,
                albumin_function = 0.60
            ),
            gfr = 50.0,  # AKI common
            hepatic_function = 0.70,
            inflammatory_state = :severe,
            notes = "Severe COVID-19: cytokine storm, multi-organ"
        )

    # =========================================
    # METABOLIC DISORDERS
    # =========================================
    elseif disease == :diabetes_t1 || disease == :diabetes_t2
        return DiseaseState(disease;
            severity = :mild,
            protein_state = PlasmaProteinState(
                albumin = 38.0,
                albumin_function = 0.85  # Glycation reduces function
            ),
            notes = "Diabetes: glycated albumin has reduced binding"
        )
    elseif disease == :obesity
        return DiseaseState(disease;
            severity = :mild,
            protein_state = PlasmaProteinState(
                albumin = 36.0,
                aag = 1.0
            ),
            cardiac_output = 6.0,
            inflammatory_state = :mild,
            notes = "Obesity: chronic low-grade inflammation"
        )
    elseif disease == :hypothyroid
        return DiseaseState(disease;
            severity = :moderate,
            protein_state = PlasmaProteinState(
                albumin = 38.0,
                albumin_function = 0.90
            ),
            cardiac_output = 4.0,
            notes = "Hypothyroidism: reduced metabolism and clearance"
        )
    elseif disease == :hyperthyroid
        return DiseaseState(disease;
            severity = :moderate,
            protein_state = PlasmaProteinState(
                albumin = 35.0,
                albumin_function = 1.0
            ),
            cardiac_output = 8.0,
            notes = "Hyperthyroidism: increased metabolism and clearance"
        )

    # =========================================
    # INFLAMMATORY CONDITIONS
    # =========================================
    elseif disease == :rheumatoid_arthritis
        return DiseaseState(disease;
            severity = :moderate,
            protein_state = PlasmaProteinState(
                albumin = 34.0,
                aag = 1.5,
                albumin_function = 0.85
            ),
            inflammatory_state = :moderate,
            notes = "RA: chronic inflammation affects binding"
        )
    elseif disease == :ibd
        return DiseaseState(disease;
            severity = severity,
            protein_state = PlasmaProteinState(
                albumin = severity == :severe ? 25.0 : 32.0,
                aag = 1.5,
                albumin_function = 0.80
            ),
            inflammatory_state = severity == :severe ? :severe : :moderate,
            notes = "IBD: protein-losing enteropathy in severe cases"
        )
    elseif disease == :sle
        return DiseaseState(disease;
            severity = :moderate,
            protein_state = PlasmaProteinState(
                albumin = 32.0,
                aag = 1.3,
                albumin_function = 0.85
            ),
            gfr = 70.0,  # Lupus nephritis
            inflammatory_state = :moderate,
            notes = "SLE: multi-organ involvement"
        )

    # =========================================
    # ONCOLOGY
    # =========================================
    elseif disease == :cancer_cachexia
        return DiseaseState(disease;
            severity = :severe,
            protein_state = PlasmaProteinState(
                albumin = 22.0,  # Severe hypoalbuminemia
                aag = 1.8,
                albumin_function = 0.70
            ),
            hepatic_function = 0.70,
            inflammatory_state = :moderate,
            notes = "Cancer cachexia: profound metabolic changes"
        )
    elseif disease == :chemotherapy
        return DiseaseState(disease;
            severity = :moderate,
            protein_state = PlasmaProteinState(
                albumin = 30.0,
                aag = 1.5,
                albumin_function = 0.80
            ),
            hepatic_function = 0.70,
            gfr = 70.0,
            hematocrit = 0.30,
            inflammatory_state = :mild,
            notes = "Chemotherapy: multi-organ toxicity"
        )
    else
        # Default normal
        return DiseaseState(:normal;
            severity = :none,
            notes = "Normal healthy adult"
        )
    end
end

# ============================================================================
# BINDING ADJUSTMENT CALCULATIONS
# ============================================================================

"""
    calculate_binding_adjustments(disease::DiseaseState)

Calculate binding adjustment factors for a disease state.

# Returns
BindingAdjustments with multipliers for fu of acidic, basic, and neutral drugs.
"""
function calculate_binding_adjustments(disease::DiseaseState)
    ps = disease.protein_state

    # =========================================
    # Albumin effect (acidic drugs)
    # fu↑ when albumin↓ or function↓
    # =========================================
    albumin_ratio = NORMAL_ALBUMIN / ps.albumin
    albumin_effect = albumin_ratio / ps.albumin_function

    # Saturability correction: fu increases less at high fu
    fu_acidic_factor = 1.0 + (albumin_effect - 1.0) * 0.8

    # =========================================
    # AAG effect (basic drugs)
    # AAG↑ in inflammation → fu↓
    # =========================================
    aag_ratio = ps.aag / NORMAL_AAG
    aag_effect = aag_ratio * ps.aag_function

    # Basic drugs: inverse relationship
    fu_basic_factor = 1.0 / aag_effect

    # Inflammation amplifies AAG effect
    if disease.inflammatory_state == :severe
        fu_basic_factor *= 0.7  # More binding
    elseif disease.inflammatory_state == :moderate
        fu_basic_factor *= 0.85
    end

    # =========================================
    # Neutral drugs
    # Affected by both albumin and lipoproteins
    # =========================================
    fu_neutral_factor = (fu_acidic_factor + fu_basic_factor) / 2.0

    # =========================================
    # Volume of distribution
    # =========================================
    # Vd increases with fu (for most drugs)
    vd_factor = (fu_acidic_factor + fu_basic_factor) / 2.0

    # Edema increases Vd for hydrophilic drugs
    if disease.volume_status == :hypervolemic
        vd_factor *= 1.3
    elseif disease.volume_status == :hypovolemic
        vd_factor *= 0.85
    end

    # =========================================
    # Clearance adjustment
    # =========================================
    # Hepatic clearance: limited by fu for high ER drugs
    hepatic_cl_factor = disease.hepatic_function

    # Renal clearance
    renal_cl_factor = disease.gfr / NORMAL_GFR

    # Combined clearance (assume 50/50 hepatic/renal)
    clearance_factor = (hepatic_cl_factor + renal_cl_factor) / 2.0

    # =========================================
    # Half-life
    # t½ = 0.693 × Vd / CL
    # =========================================
    half_life_factor = vd_factor / clearance_factor

    return BindingAdjustments(
        fu_acidic_factor,
        fu_basic_factor,
        fu_neutral_factor,
        vd_factor,
        clearance_factor,
        half_life_factor
    )
end

"""
    calculate_adjusted_fu(fu_normal::Float64,
                          drug_type::Symbol,
                          disease::DiseaseState)

Calculate adjusted fraction unbound for a drug in a disease state.

# Arguments
- `fu_normal`: Normal fu value
- `drug_type`: :acidic, :basic, :neutral, :zwitterion
- `disease`: Disease state

# Returns
Adjusted fu value
"""
function calculate_adjusted_fu(fu_normal::Float64,
                                drug_type::Symbol,
                                disease::DiseaseState)
    adj = calculate_binding_adjustments(disease)

    factor = if drug_type == :acidic
        adj.fu_acidic_factor
    elseif drug_type == :basic
        adj.fu_basic_factor
    elseif drug_type == :neutral
        adj.fu_neutral_factor
    elseif drug_type == :zwitterion
        (adj.fu_acidic_factor + adj.fu_basic_factor) / 2.0
    else
        1.0
    end

    # Calculate adjusted fu
    fu_adjusted = fu_normal * factor

    # Constraints: 0 < fu ≤ 1
    fu_adjusted = max(fu_adjusted, 0.001)
    fu_adjusted = min(fu_adjusted, 1.0)

    return fu_adjusted
end

"""
    apply_disease_adjustments(fu_normal::Float64,
                               vd_normal::Float64,
                               cl_normal::Float64,
                               drug_type::Symbol,
                               disease::DiseaseState)

Apply disease-state adjustments to PK parameters.

# Returns
Dict with adjusted PK parameters
"""
function apply_disease_adjustments(fu_normal::Float64,
                                    vd_normal::Float64,
                                    cl_normal::Float64,
                                    drug_type::Symbol,
                                    disease::DiseaseState)
    adj = calculate_binding_adjustments(disease)

    fu_adjusted = calculate_adjusted_fu(fu_normal, drug_type, disease)

    # Vd adjustment
    vd_adjusted = vd_normal * adj.vd_factor

    # Clearance adjustment
    cl_adjusted = cl_normal * adj.clearance_factor

    # Half-life
    t_half_normal = 0.693 * vd_normal / cl_normal
    t_half_adjusted = 0.693 * vd_adjusted / cl_adjusted

    return Dict(
        "fu" => fu_adjusted,
        "fu_ratio" => fu_adjusted / fu_normal,
        "vd" => vd_adjusted,
        "vd_ratio" => vd_adjusted / vd_normal,
        "clearance" => cl_adjusted,
        "cl_ratio" => cl_adjusted / cl_normal,
        "half_life" => t_half_adjusted,
        "t_half_ratio" => t_half_adjusted / t_half_normal,
        "disease" => disease.name,
        "severity" => disease.severity
    )
end

# ============================================================================
# DISEASE-SPECIFIC BINDING DATABASE
# ============================================================================

"""
Pre-calculated binding factors for common disease-drug combinations.

Based on clinical pharmacokinetic studies.
"""
const DISEASE_BINDING_DATABASE = Dict{Tuple{Symbol, String}, Float64}(
    # (Disease, Drug) => fu_ratio

    # Uremia effects
    (:esrd, "phenytoin") => 2.5,      # Classic example: fu 0.1 → 0.25
    (:esrd, "valproic_acid") => 2.0,
    (:esrd, "diazepam") => 2.0,
    (:esrd, "warfarin") => 1.5,
    (:esrd, "furosemide") => 1.8,

    # Hepatic disease
    (:cirrhosis_child_c, "diazepam") => 2.5,
    (:cirrhosis_child_c, "propranolol") => 0.6,  # Basic drug, AAG may increase
    (:cirrhosis_child_c, "lidocaine") => 0.7,

    # Pregnancy
    (:pregnancy_t3, "phenytoin") => 1.5,
    (:pregnancy_t3, "valproic_acid") => 1.4,
    (:pregnancy_t3, "carbamazepine") => 1.2,

    # Sepsis/Critical illness
    (:sepsis, "ceftriaxone") => 1.8,
    (:sepsis, "vancomycin") => 1.5,
    (:sepsis, "meropenem") => 1.4,
    (:sepsis, "propranolol") => 0.5,  # AAG↑↑

    # Neonates
    (:neonate, "phenobarbital") => 2.0,
    (:neonate, "ampicillin") => 1.5,
    (:neonate, "gentamicin") => 1.3
)

"""
    get_disease_binding_factors(disease::Symbol, drug_name::String)

Get empirical binding factor from database if available.

Returns nothing if no specific data exists.
"""
function get_disease_binding_factors(disease::Symbol, drug_name::String)
    key = (disease, lowercase(drug_name))
    if haskey(DISEASE_BINDING_DATABASE, key)
        return DISEASE_BINDING_DATABASE[key]
    end
    return nothing
end

# ============================================================================
# CLINICAL UTILITIES
# ============================================================================

"""
    recommend_dose_adjustment(disease::DiseaseState,
                               drug_type::Symbol;
                               route::Symbol = :oral)

Provide general dosing recommendations based on disease state.

# Returns
Dict with:
- `dose_factor`: Recommended dose multiplier
- `interval_factor`: Dosing interval multiplier
- `loading_dose`: Whether loading dose recommended
- `monitoring`: Monitoring recommendations
"""
function recommend_dose_adjustment(disease::DiseaseState,
                                    drug_type::Symbol;
                                    route::Symbol = :oral)
    adj = calculate_binding_adjustments(disease)

    # Base recommendations on clearance change
    cl_ratio = adj.clearance_factor

    dose_factor = cl_ratio  # Reduce dose proportional to CL reduction
    interval_factor = 1.0 / adj.half_life_factor  # May need less frequent dosing

    # Loading dose considerations
    loading_dose = adj.vd_factor > 1.3  # If Vd increased, may need loading

    # Monitoring
    monitoring = String[]

    if disease.gfr < 30
        push!(monitoring, "Monitor renal function (creatinine, GFR)")
        push!(monitoring, "Consider TDM for narrow therapeutic index drugs")
    end

    if disease.hepatic_function < 0.5
        push!(monitoring, "Monitor hepatic function (LFTs)")
        push!(monitoring, "Watch for accumulation")
    end

    if disease.inflammatory_state == :severe
        push!(monitoring, "AAG levels may affect basic drug binding")
        push!(monitoring, "Monitor frequently during acute phase")
    end

    if disease.name in [:pregnancy_t1, :pregnancy_t2, :pregnancy_t3]
        push!(monitoring, "Consider fetal exposure")
        push!(monitoring, "May need TDM, especially for anticonvulsants")
    end

    return Dict(
        "dose_factor" => dose_factor,
        "interval_factor" => interval_factor,
        "loading_dose_recommended" => loading_dose,
        "monitoring" => monitoring,
        "fu_change" => drug_type == :acidic ? adj.fu_acidic_factor : adj.fu_basic_factor,
        "notes" => "Individualize based on clinical response and TDM"
    )
end

end # module DiseaseStateBinding
