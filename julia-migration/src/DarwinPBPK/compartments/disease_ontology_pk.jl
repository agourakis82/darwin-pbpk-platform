"""
Disease Ontology PK Integration Module

Links DOID (Disease Ontology) and ICD-10/ICD-11 codes to pharmacokinetic
adjustments, enabling ontology-driven disease state modeling.

Key Features:
- DOID-based disease identification
- ICD-10/ICD-11 code mapping
- Automatic PK adjustment lookup by disease code
- Hierarchical disease relationships for PK inference
- SNOMED-CT, MESH, OMIM cross-references

Clinical Relevance:
- Standardized disease coding for PBPK
- Regulatory-compliant disease identification
- Enables EHR integration for PBPK dosing
- Supports population PBPK with disease covariates

References:
- Schriml LM et al. (2019) Human Disease Ontology 2018 update
- ICD-11 Reference Guide (WHO, 2022)
- FDA Guidance on Disease Classification in Drug Development

Author: Darwin PBPK Platform
Date: 2025-12-05
"""
module DiseaseOntologyPK

using Statistics

export DiseaseCode, DiseasePKProfile, OntologyMapping
export get_pk_adjustments_by_doid, get_pk_adjustments_by_icd10
export get_pk_adjustments_by_icd11, search_disease_pk
export map_disease_hierarchy, get_pk_with_fallback
export combine_disease_profiles, list_supported_diseases
export get_disease_summary
export DOID_PK_DATABASE, ICD10_TO_DOID, ICD11_TO_DOID
export DISEASE_HIERARCHY

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    DiseaseCode

Standardized disease identifier with cross-references.

# Fields
- `doid::String`: DOID identifier (e.g., "DOID:9352")
- `icd10::Vector{String}`: ICD-10 codes
- `icd11::Vector{String}`: ICD-11 codes
- `snomed::Vector{String}`: SNOMED-CT codes
- `mesh::Vector{String}`: MeSH terms
- `omim::Vector{String}`: OMIM identifiers
- `name::String`: Canonical disease name
- `synonyms::Vector{String}`: Alternative names
"""
struct DiseaseCode
    doid::String
    icd10::Vector{String}
    icd11::Vector{String}
    snomed::Vector{String}
    mesh::Vector{String}
    omim::Vector{String}
    name::String
    synonyms::Vector{String}

    function DiseaseCode(;
        doid::String = "",
        icd10::Vector{String} = String[],
        icd11::Vector{String} = String[],
        snomed::Vector{String} = String[],
        mesh::Vector{String} = String[],
        omim::Vector{String} = String[],
        name::String = "",
        synonyms::Vector{String} = String[]
    )
        new(doid, icd10, icd11, snomed, mesh, omim, name, synonyms)
    end
end

"""
    DiseasePKProfile

PK adjustment profile for a disease state.

# Fields
- `disease::DiseaseCode`: Disease identifiers
- `gfr_adjustment::Float64`: GFR multiplier (1.0 = normal)
- `hepatic_adjustment::Float64`: Hepatic function multiplier
- `fu_acidic_adjustment::Float64`: fu adjustment for acidic drugs
- `fu_basic_adjustment::Float64`: fu adjustment for basic drugs
- `vd_adjustment::Float64`: Volume of distribution multiplier
- `absorption_adjustment::Float64`: Oral absorption multiplier
- `albumin_concentration::Float64`: Expected albumin (g/L)
- `aag_concentration::Float64`: Expected AAG (g/L)
- `special_considerations::Vector{String}`: Clinical notes
- `evidence_level::Symbol`: :high, :moderate, :low, :extrapolated
- `references::Vector{String}`: Literature references
"""
struct DiseasePKProfile
    disease::DiseaseCode
    gfr_adjustment::Float64
    hepatic_adjustment::Float64
    fu_acidic_adjustment::Float64
    fu_basic_adjustment::Float64
    vd_adjustment::Float64
    absorption_adjustment::Float64
    albumin_concentration::Float64
    aag_concentration::Float64
    special_considerations::Vector{String}
    evidence_level::Symbol
    references::Vector{String}

    function DiseasePKProfile(disease::DiseaseCode;
        gfr_adjustment::Float64 = 1.0,
        hepatic_adjustment::Float64 = 1.0,
        fu_acidic_adjustment::Float64 = 1.0,
        fu_basic_adjustment::Float64 = 1.0,
        vd_adjustment::Float64 = 1.0,
        absorption_adjustment::Float64 = 1.0,
        albumin_concentration::Float64 = 40.0,
        aag_concentration::Float64 = 0.8,
        special_considerations::Vector{String} = String[],
        evidence_level::Symbol = :moderate,
        references::Vector{String} = String[]
    )
        new(disease, gfr_adjustment, hepatic_adjustment,
            fu_acidic_adjustment, fu_basic_adjustment,
            vd_adjustment, absorption_adjustment,
            albumin_concentration, aag_concentration,
            special_considerations, evidence_level, references)
    end
end

"""
    OntologyMapping

Cross-reference mapping between ontologies.
"""
struct OntologyMapping
    source_system::Symbol
    source_code::String
    target_system::Symbol
    target_code::String
    relationship::Symbol  # :exact, :broader, :narrower, :related
end

# ============================================================================
# DOID → PK DATABASE
# ============================================================================

"""
Comprehensive DOID to PK adjustments database.
Based on clinical pharmacokinetic literature.
"""
const DOID_PK_DATABASE = Dict{String, DiseasePKProfile}(
    # =========================================================================
    # RENAL DISEASES
    # =========================================================================

    # Chronic kidney disease
    "DOID:784" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:784",
            icd10 = ["N18", "N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6"],
            icd11 = ["GB61"],
            snomed = ["709044004"],
            mesh = ["D051436"],
            name = "chronic kidney disease",
            synonyms = ["CKD", "chronic renal failure", "chronic renal insufficiency"]
        );
        gfr_adjustment = 0.5,  # Average across stages
        hepatic_adjustment = 0.95,
        fu_acidic_adjustment = 1.5,  # Uremic toxins displace
        fu_basic_adjustment = 1.2,
        vd_adjustment = 1.1,
        albumin_concentration = 35.0,
        aag_concentration = 1.2,
        special_considerations = [
            "GFR-dependent dosing required",
            "Uremic toxins compete for albumin binding",
            "AAG often elevated (inflammation)",
            "Consider dialyzability for ESRD"
        ],
        evidence_level = :high,
        references = ["Matzke GR et al. Kidney Int 2011"]
    ),

    # CKD Stage 3
    "DOID:0060681" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:0060681",
            icd10 = ["N18.3"],
            icd11 = ["GB61.2"],
            name = "chronic kidney disease stage 3",
            synonyms = ["CKD stage 3", "moderate CKD"]
        );
        gfr_adjustment = 0.45,  # GFR 30-59
        hepatic_adjustment = 1.0,
        fu_acidic_adjustment = 1.3,
        fu_basic_adjustment = 1.1,
        vd_adjustment = 1.05,
        albumin_concentration = 38.0,
        aag_concentration = 1.0,
        special_considerations = ["Dose reduction for renally cleared drugs"],
        evidence_level = :high
    ),

    # CKD Stage 4
    "DOID:0060682" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:0060682",
            icd10 = ["N18.4"],
            icd11 = ["GB61.3"],
            name = "chronic kidney disease stage 4",
            synonyms = ["CKD stage 4", "severe CKD"]
        );
        gfr_adjustment = 0.22,  # GFR 15-29
        hepatic_adjustment = 0.95,
        fu_acidic_adjustment = 1.8,
        fu_basic_adjustment = 1.3,
        vd_adjustment = 1.15,
        albumin_concentration = 34.0,
        aag_concentration = 1.3,
        special_considerations = [
            "Major dose reductions required",
            "Prepare for dialysis transition",
            "Monitor for uremic encephalopathy"
        ],
        evidence_level = :high
    ),

    # End-stage renal disease
    "DOID:783" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:783",
            icd10 = ["N18.5", "N18.6"],
            icd11 = ["GB61.4", "GB61.5"],
            snomed = ["46177005"],
            mesh = ["D007676"],
            name = "end stage renal disease",
            synonyms = ["ESRD", "end-stage kidney disease", "kidney failure"]
        );
        gfr_adjustment = 0.08,  # GFR < 15
        hepatic_adjustment = 0.90,
        fu_acidic_adjustment = 2.5,
        fu_basic_adjustment = 1.5,
        vd_adjustment = 1.3,
        albumin_concentration = 30.0,
        aag_concentration = 1.5,
        special_considerations = [
            "Dialysis clearance must be considered",
            "Post-dialysis supplemental dosing often needed",
            "High uremic toxin levels",
            "Anemia common"
        ],
        evidence_level = :high,
        references = ["Nolin TD et al. JASN 2016"]
    ),

    # Acute kidney injury
    "DOID:1074" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:1074",
            icd10 = ["N17", "N17.0", "N17.1", "N17.2", "N17.8", "N17.9"],
            icd11 = ["GB60"],
            snomed = ["14669001"],
            mesh = ["D058186"],
            name = "acute kidney injury",
            synonyms = ["AKI", "acute renal failure", "acute kidney failure"]
        );
        gfr_adjustment = 0.3,  # Variable, use KDIGO staging
        hepatic_adjustment = 0.95,
        fu_acidic_adjustment = 1.5,
        fu_basic_adjustment = 1.2,
        vd_adjustment = 1.2,
        albumin_concentration = 32.0,
        aag_concentration = 1.5,
        special_considerations = [
            "Rapidly changing GFR - frequent monitoring",
            "Often accompanies critical illness",
            "Consider augmented renal clearance in recovery phase"
        ],
        evidence_level = :moderate
    ),

    # =========================================================================
    # HEPATIC DISEASES
    # =========================================================================

    # Liver cirrhosis
    "DOID:5082" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:5082",
            icd10 = ["K74", "K74.0", "K74.1", "K74.2", "K74.6"],
            icd11 = ["DB93.1"],
            snomed = ["19943007"],
            mesh = ["D008103"],
            name = "liver cirrhosis",
            synonyms = ["cirrhosis", "hepatic cirrhosis"]
        );
        gfr_adjustment = 0.85,  # Hepatorenal syndrome risk
        hepatic_adjustment = 0.5,  # Variable by Child-Pugh
        fu_acidic_adjustment = 2.0,
        fu_basic_adjustment = 0.8,  # AAG may be low
        vd_adjustment = 1.5,  # Ascites
        absorption_adjustment = 0.8,
        albumin_concentration = 28.0,
        aag_concentration = 0.5,
        special_considerations = [
            "Child-Pugh score determines adjustment",
            "Ascites affects Vd of hydrophilic drugs",
            "Reduced first-pass metabolism",
            "Portal hypertension affects absorption",
            "Avoid hepatotoxic drugs"
        ],
        evidence_level = :high,
        references = ["Verbeeck RK. Eur J Clin Pharmacol 2008"]
    ),

    # Alcoholic liver disease
    "DOID:9452" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:9452",
            icd10 = ["K70", "K70.0", "K70.1", "K70.2", "K70.3", "K70.4"],
            icd11 = ["DB92"],
            snomed = ["41309000"],
            mesh = ["D008108"],
            name = "alcoholic liver disease",
            synonyms = ["ALD", "alcohol-related liver disease"]
        );
        gfr_adjustment = 0.9,
        hepatic_adjustment = 0.6,
        fu_acidic_adjustment = 1.8,
        fu_basic_adjustment = 0.9,
        vd_adjustment = 1.3,
        albumin_concentration = 30.0,
        aag_concentration = 0.6,
        special_considerations = [
            "Enzyme induction in early disease",
            "Enzyme inhibition in advanced disease",
            "CYP2E1 induced by alcohol",
            "Consider alcohol withdrawal effects"
        ],
        evidence_level = :moderate
    ),

    # Hepatic steatosis (NAFLD)
    "DOID:9452" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:0080208",
            icd10 = ["K76.0"],
            icd11 = ["DB92.0"],
            snomed = ["197321007"],
            name = "non-alcoholic fatty liver disease",
            synonyms = ["NAFLD", "fatty liver", "hepatic steatosis"]
        );
        gfr_adjustment = 1.0,
        hepatic_adjustment = 0.9,
        fu_acidic_adjustment = 1.1,
        fu_basic_adjustment = 1.0,
        vd_adjustment = 1.0,
        albumin_concentration = 38.0,
        aag_concentration = 0.9,
        special_considerations = [
            "CYP3A4 may be reduced",
            "UGT activity often preserved",
            "Often coexists with metabolic syndrome"
        ],
        evidence_level = :moderate
    ),

    # =========================================================================
    # DIABETES
    # =========================================================================

    # Diabetes mellitus
    "DOID:9351" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:9351",
            icd10 = ["E10", "E11", "E13", "E14"],
            icd11 = ["5A10", "5A11"],
            snomed = ["73211009"],
            mesh = ["D003920"],
            name = "diabetes mellitus",
            synonyms = ["diabetes", "DM"]
        );
        gfr_adjustment = 0.9,  # Hyperfiltration early, decline later
        hepatic_adjustment = 0.95,
        fu_acidic_adjustment = 1.1,
        fu_basic_adjustment = 1.0,
        vd_adjustment = 1.0,
        albumin_concentration = 38.0,
        aag_concentration = 1.0,
        special_considerations = [
            "Glycation affects albumin binding",
            "Gastroparesis affects oral absorption",
            "Nephropathy progression affects PK",
            "Hyperglycemia may affect tissue distribution"
        ],
        evidence_level = :moderate
    ),

    # Type 1 diabetes
    "DOID:9744" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:9744",
            icd10 = ["E10"],
            icd11 = ["5A10"],
            snomed = ["46635009"],
            mesh = ["D003922"],
            name = "type 1 diabetes mellitus",
            synonyms = ["T1DM", "insulin-dependent diabetes", "juvenile diabetes"]
        );
        gfr_adjustment = 0.95,
        hepatic_adjustment = 1.0,
        fu_acidic_adjustment = 1.05,
        fu_basic_adjustment = 1.0,
        vd_adjustment = 1.0,
        albumin_concentration = 40.0,
        aag_concentration = 0.9,
        special_considerations = [
            "Young patients - age-related PK important",
            "DKA affects drug distribution acutely"
        ],
        evidence_level = :moderate
    ),

    # Type 2 diabetes
    "DOID:9352" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:9352",
            icd10 = ["E11"],
            icd11 = ["5A11"],
            snomed = ["44054006"],
            mesh = ["D003924"],
            name = "type 2 diabetes mellitus",
            synonyms = ["T2DM", "non-insulin dependent diabetes", "adult-onset diabetes"]
        );
        gfr_adjustment = 0.85,
        hepatic_adjustment = 0.9,  # Often with NAFLD
        fu_acidic_adjustment = 1.15,
        fu_basic_adjustment = 1.0,
        vd_adjustment = 1.1,
        albumin_concentration = 36.0,
        aag_concentration = 1.1,
        special_considerations = [
            "Often comorbid with obesity, NAFLD, CKD",
            "Metformin contraindicated if GFR<30",
            "SGLT2i need GFR consideration"
        ],
        evidence_level = :high
    ),

    # =========================================================================
    # CARDIOVASCULAR
    # =========================================================================

    # Heart failure
    "DOID:6000" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:6000",
            icd10 = ["I50", "I50.0", "I50.1", "I50.9"],
            icd11 = ["BD1Z"],
            snomed = ["84114007"],
            mesh = ["D006333"],
            name = "heart failure",
            synonyms = ["CHF", "congestive heart failure", "cardiac failure"]
        );
        gfr_adjustment = 0.7,  # Cardiorenal syndrome
        hepatic_adjustment = 0.8,  # Hepatic congestion
        fu_acidic_adjustment = 1.2,
        fu_basic_adjustment = 0.9,
        vd_adjustment = 1.3,  # Edema
        absorption_adjustment = 0.8,  # GI congestion
        albumin_concentration = 34.0,
        aag_concentration = 1.2,
        special_considerations = [
            "Reduced cardiac output affects organ perfusion",
            "Hepatic congestion reduces metabolism",
            "Edema affects Vd of hydrophilic drugs",
            "Renal hypoperfusion reduces clearance"
        ],
        evidence_level = :high,
        references = ["Ogawa R et al. Clin Pharmacokinet 2014"]
    ),

    # =========================================================================
    # INFLAMMATORY/AUTOIMMUNE
    # =========================================================================

    # Rheumatoid arthritis
    "DOID:7148" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:7148",
            icd10 = ["M05", "M06"],
            icd11 = ["FA20"],
            snomed = ["69896004"],
            mesh = ["D001172"],
            name = "rheumatoid arthritis",
            synonyms = ["RA"]
        );
        gfr_adjustment = 0.95,
        hepatic_adjustment = 0.95,
        fu_acidic_adjustment = 1.0,
        fu_basic_adjustment = 0.7,  # AAG elevated
        vd_adjustment = 1.0,
        albumin_concentration = 36.0,
        aag_concentration = 1.8,
        special_considerations = [
            "Chronic inflammation elevates AAG",
            "Disease activity affects CYP expression",
            "Biologics have TMDD",
            "Monitor for MTX toxicity in renal impairment"
        ],
        evidence_level = :moderate
    ),

    # Systemic lupus erythematosus
    "DOID:9074" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:9074",
            icd10 = ["M32"],
            icd11 = ["4A40"],
            snomed = ["55464009"],
            mesh = ["D008180"],
            name = "systemic lupus erythematosus",
            synonyms = ["SLE", "lupus"]
        );
        gfr_adjustment = 0.8,  # Lupus nephritis common
        hepatic_adjustment = 0.95,
        fu_acidic_adjustment = 1.3,
        fu_basic_adjustment = 0.8,
        vd_adjustment = 1.1,
        albumin_concentration = 32.0,
        aag_concentration = 1.5,
        special_considerations = [
            "Lupus nephritis affects renal clearance",
            "Anti-drug antibodies common with biologics",
            "Flares affect protein binding acutely"
        ],
        evidence_level = :moderate
    ),

    # Inflammatory bowel disease
    "DOID:0050589" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:0050589",
            icd10 = ["K50", "K51"],
            icd11 = ["DD70", "DD71"],
            snomed = ["24526004"],
            mesh = ["D015212"],
            name = "inflammatory bowel disease",
            synonyms = ["IBD", "Crohn's disease", "ulcerative colitis"]
        );
        gfr_adjustment = 1.0,
        hepatic_adjustment = 1.0,
        fu_acidic_adjustment = 1.1,
        fu_basic_adjustment = 0.8,
        vd_adjustment = 1.0,
        absorption_adjustment = 0.7,  # Malabsorption
        albumin_concentration = 34.0,
        aag_concentration = 1.4,
        special_considerations = [
            "Malabsorption affects oral drugs",
            "Fistulas/resection alter gut transit",
            "Protein-losing enteropathy affects binding",
            "Anti-TNF clearance varies with disease activity"
        ],
        evidence_level = :moderate
    ),

    # =========================================================================
    # CRITICAL ILLNESS
    # =========================================================================

    # Sepsis
    "DOID:0080559" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:0080559",
            icd10 = ["A41", "R65.2"],
            icd11 = ["1G40"],
            snomed = ["91302008"],
            mesh = ["D018805"],
            name = "sepsis",
            synonyms = ["septicemia", "septic shock"]
        );
        gfr_adjustment = 0.6,  # AKI common
        hepatic_adjustment = 0.7,
        fu_acidic_adjustment = 2.0,  # Hypoalbuminemia
        fu_basic_adjustment = 0.5,  # AAG very elevated
        vd_adjustment = 1.8,  # Capillary leak
        absorption_adjustment = 0.3,  # Ileus common
        albumin_concentration = 20.0,
        aag_concentration = 2.5,
        special_considerations = [
            "Augmented renal clearance possible in hyperdynamic phase",
            "Capillary leak increases Vd dramatically",
            "Acute phase response changes protein binding",
            "Organ dysfunction affects clearance",
            "Consider loading doses for hydrophilic drugs"
        ],
        evidence_level = :high,
        references = ["Roberts JA et al. Lancet Infect Dis 2014"]
    ),

    # Burns
    "DOID:0050805" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:0050805",
            icd10 = ["T30", "T31", "T32"],
            icd11 = ["NE80"],
            snomed = ["125666000"],
            name = "burn injury",
            synonyms = ["thermal injury", "burn"]
        );
        gfr_adjustment = 1.2,  # Hyperdynamic, then AKI
        hepatic_adjustment = 1.3,  # Initially increased
        fu_acidic_adjustment = 2.5,  # Severe hypoalbuminemia
        fu_basic_adjustment = 0.6,  # AAG very elevated
        vd_adjustment = 2.0,  # Massive fluid shifts
        absorption_adjustment = 0.4,
        albumin_concentration = 18.0,
        aag_concentration = 3.0,
        special_considerations = [
            "Hyperdynamic phase increases clearance 50-100%",
            "Massive protein loss through wounds",
            "Augmented renal clearance in recovery",
            "Time-varying PK over weeks",
            "Consider TBSA for dose adjustments"
        ],
        evidence_level = :high,
        references = ["Blanchet B et al. Clin Pharmacokinet 2008"]
    ),

    # =========================================================================
    # PREGNANCY
    # =========================================================================

    # Pregnancy
    "DOID:0060088" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:0060088",
            icd10 = ["O00-O99", "Z33"],
            icd11 = ["JA00-JB0Z"],
            snomed = ["77386006"],
            mesh = ["D011247"],
            name = "pregnancy",
            synonyms = ["gravidity", "gestation"]
        );
        gfr_adjustment = 1.5,  # GFR increases 50%
        hepatic_adjustment = 1.1,
        fu_acidic_adjustment = 1.3,  # Dilutional
        fu_basic_adjustment = 1.0,
        vd_adjustment = 1.4,  # Plasma volume expansion
        absorption_adjustment = 0.9,
        albumin_concentration = 32.0,  # Dilutional decrease
        aag_concentration = 0.7,
        special_considerations = [
            "GFR increases 50% by second trimester",
            "Plasma volume expands 40-50%",
            "CYP3A4 and CYP2D6 induced",
            "CYP1A2 and CYP2C19 inhibited",
            "Consider fetal exposure for all drugs"
        ],
        evidence_level = :high,
        references = ["Abduljalil K et al. Clin Pharmacokinet 2012"]
    ),

    # =========================================================================
    # OBESITY
    # =========================================================================

    # Obesity
    "DOID:9970" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:9970",
            icd10 = ["E66"],
            icd11 = ["5B81"],
            snomed = ["414916001"],
            mesh = ["D009765"],
            name = "obesity",
            synonyms = ["morbid obesity", "overweight"]
        );
        gfr_adjustment = 1.2,  # Often hyperfiltration
        hepatic_adjustment = 0.9,  # NAFLD common
        fu_acidic_adjustment = 0.9,
        fu_basic_adjustment = 1.0,
        vd_adjustment = 1.5,  # Lipophilic drugs
        absorption_adjustment = 1.0,
        albumin_concentration = 38.0,
        aag_concentration = 1.1,
        special_considerations = [
            "Lipophilic drugs: Vd increases with fat mass",
            "Hydrophilic drugs: dose on lean body weight",
            "Consider adjusted body weight for dosing",
            "CYP2E1 activity increased",
            "Bariatric surgery changes absorption dramatically"
        ],
        evidence_level = :moderate,
        references = ["Hanley MJ et al. Clin Pharmacokinet 2010"]
    ),

    # =========================================================================
    # ONCOLOGY
    # =========================================================================

    # Cancer (general)
    "DOID:162" => DiseasePKProfile(
        DiseaseCode(
            doid = "DOID:162",
            icd10 = ["C00-C97"],
            icd11 = ["2A-2F"],
            snomed = ["363346000"],
            mesh = ["D009369"],
            name = "cancer",
            synonyms = ["malignancy", "neoplasm", "tumor"]
        );
        gfr_adjustment = 0.9,
        hepatic_adjustment = 0.9,
        fu_acidic_adjustment = 1.3,  # Cancer cachexia
        fu_basic_adjustment = 0.9,
        vd_adjustment = 1.0,
        albumin_concentration = 32.0,
        aag_concentration = 1.3,
        special_considerations = [
            "Cancer cachexia reduces albumin",
            "Hepatic metastases affect metabolism",
            "Prior chemotherapy affects organ function",
            "Drug interactions with chemotherapy",
            "Consider tumor lysis effects"
        ],
        evidence_level = :moderate
    )
)

# ============================================================================
# ICD-10 LOOKUP DATABASE
# ============================================================================

"""
ICD-10 code to DOID mapping for quick lookup.
"""
const ICD10_TO_DOID = Dict{String, String}(
    # Renal
    "N18" => "DOID:784",
    "N18.1" => "DOID:784",
    "N18.2" => "DOID:784",
    "N18.3" => "DOID:0060681",
    "N18.4" => "DOID:0060682",
    "N18.5" => "DOID:783",
    "N18.6" => "DOID:783",
    "N17" => "DOID:1074",

    # Hepatic
    "K74" => "DOID:5082",
    "K70" => "DOID:9452",
    "K76.0" => "DOID:0080208",

    # Diabetes
    "E10" => "DOID:9744",
    "E11" => "DOID:9352",

    # Cardiac
    "I50" => "DOID:6000",

    # Inflammatory
    "M05" => "DOID:7148",
    "M06" => "DOID:7148",
    "M32" => "DOID:9074",
    "K50" => "DOID:0050589",
    "K51" => "DOID:0050589",

    # Critical
    "A41" => "DOID:0080559",
    "R65.2" => "DOID:0080559",
    "T30" => "DOID:0050805",

    # Other
    "E66" => "DOID:9970",
    "Z33" => "DOID:0060088"
)

"""
ICD-11 code to DOID mapping.
"""
const ICD11_TO_DOID = Dict{String, String}(
    "GB61" => "DOID:784",
    "GB61.2" => "DOID:0060681",
    "GB61.3" => "DOID:0060682",
    "GB61.4" => "DOID:783",
    "GB60" => "DOID:1074",
    "DB93.1" => "DOID:5082",
    "DB92" => "DOID:9452",
    "5A10" => "DOID:9744",
    "5A11" => "DOID:9352",
    "BD1Z" => "DOID:6000",
    "FA20" => "DOID:7148",
    "4A40" => "DOID:9074",
    "1G40" => "DOID:0080559",
    "5B81" => "DOID:9970"
)

# ============================================================================
# LOOKUP FUNCTIONS
# ============================================================================

"""
    get_pk_adjustments_by_doid(doid::String)

Get PK adjustments by DOID identifier.

# Example
```julia
profile = get_pk_adjustments_by_doid("DOID:9352")  # Type 2 diabetes
```
"""
function get_pk_adjustments_by_doid(doid::String)
    # Normalize DOID format
    if !startswith(doid, "DOID:")
        doid = "DOID:" * doid
    end

    if haskey(DOID_PK_DATABASE, doid)
        return DOID_PK_DATABASE[doid]
    end

    return nothing
end

"""
    get_pk_adjustments_by_icd10(icd10::String)

Get PK adjustments by ICD-10 code.

# Example
```julia
profile = get_pk_adjustments_by_icd10("N18.4")  # CKD stage 4
profile = get_pk_adjustments_by_icd10("E11")    # Type 2 diabetes
```
"""
function get_pk_adjustments_by_icd10(icd10::String)
    # Remove dots for matching
    icd10_clean = uppercase(icd10)

    # Try exact match
    if haskey(ICD10_TO_DOID, icd10_clean)
        return get_pk_adjustments_by_doid(ICD10_TO_DOID[icd10_clean])
    end

    # Try prefix match (e.g., "N18.3" matches "N18")
    prefix = split(icd10_clean, ".")[1]
    if haskey(ICD10_TO_DOID, prefix)
        return get_pk_adjustments_by_doid(ICD10_TO_DOID[prefix])
    end

    return nothing
end

"""
    get_pk_adjustments_by_icd11(icd11::String)

Get PK adjustments by ICD-11 code.
"""
function get_pk_adjustments_by_icd11(icd11::String)
    icd11_clean = uppercase(icd11)

    if haskey(ICD11_TO_DOID, icd11_clean)
        return get_pk_adjustments_by_doid(ICD11_TO_DOID[icd11_clean])
    end

    # Try prefix match
    for (code, doid) in ICD11_TO_DOID
        if startswith(icd11_clean, code) || startswith(code, icd11_clean)
            return get_pk_adjustments_by_doid(doid)
        end
    end

    return nothing
end

"""
    search_disease_pk(query::String)

Search for disease PK profile by name or synonym.
"""
function search_disease_pk(query::String)
    query_lower = lowercase(query)
    results = DiseasePKProfile[]

    for (doid, profile) in DOID_PK_DATABASE
        disease = profile.disease

        # Check name
        if occursin(query_lower, lowercase(disease.name))
            push!(results, profile)
            continue
        end

        # Check synonyms
        for syn in disease.synonyms
            if occursin(query_lower, lowercase(syn))
                push!(results, profile)
                break
            end
        end
    end

    return results
end

# ============================================================================
# HIERARCHICAL INFERENCE
# ============================================================================

"""
Disease hierarchy for PK inference.
If specific disease not found, can use parent disease parameters.
"""
const DISEASE_HIERARCHY = Dict{String, String}(
    # CKD stages inherit from CKD
    "DOID:0060681" => "DOID:784",  # CKD3 → CKD
    "DOID:0060682" => "DOID:784",  # CKD4 → CKD
    "DOID:783" => "DOID:784",       # ESRD → CKD

    # Diabetes subtypes
    "DOID:9744" => "DOID:9351",     # T1DM → DM
    "DOID:9352" => "DOID:9351",     # T2DM → DM

    # IBD subtypes
    "DOID:8577" => "DOID:0050589",  # Crohn's → IBD
    "DOID:8778" => "DOID:0050589"   # UC → IBD
)

"""
    map_disease_hierarchy(doid::String)

Get hierarchical disease mapping for PK inference.
Returns list of DOIDs from specific to general.
"""
function map_disease_hierarchy(doid::String)
    hierarchy = [doid]

    current = doid
    while haskey(DISEASE_HIERARCHY, current)
        parent = DISEASE_HIERARCHY[current]
        push!(hierarchy, parent)
        current = parent
    end

    return hierarchy
end

"""
    get_pk_with_fallback(doid::String)

Get PK profile, falling back to parent disease if specific not found.
"""
function get_pk_with_fallback(doid::String)
    # Try direct lookup
    profile = get_pk_adjustments_by_doid(doid)
    if profile !== nothing
        return (profile, doid, :exact)
    end

    # Try hierarchy
    hierarchy = map_disease_hierarchy(doid)
    for parent_doid in hierarchy[2:end]
        profile = get_pk_adjustments_by_doid(parent_doid)
        if profile !== nothing
            return (profile, parent_doid, :inferred)
        end
    end

    return (nothing, "", :not_found)
end

# ============================================================================
# COMBINATION DISEASES
# ============================================================================

"""
    combine_disease_profiles(profiles::Vector{DiseasePKProfile})

Combine multiple disease profiles for comorbid patients.

Uses conservative approach: most impaired value for each parameter.
"""
function combine_disease_profiles(profiles::Vector{DiseasePKProfile})
    if isempty(profiles)
        return nothing
    end

    if length(profiles) == 1
        return profiles[1]
    end

    # Combine by taking worst case for each parameter
    combined_gfr = minimum(p.gfr_adjustment for p in profiles)
    combined_hepatic = minimum(p.hepatic_adjustment for p in profiles)
    combined_fu_acidic = maximum(p.fu_acidic_adjustment for p in profiles)
    combined_fu_basic = minimum(p.fu_basic_adjustment for p in profiles)  # AAG elevation
    combined_vd = maximum(p.vd_adjustment for p in profiles)
    combined_absorption = minimum(p.absorption_adjustment for p in profiles)
    combined_albumin = minimum(p.albumin_concentration for p in profiles)
    combined_aag = maximum(p.aag_concentration for p in profiles)

    # Combine disease codes
    combined_icd10 = String[]
    combined_names = String[]
    for p in profiles
        append!(combined_icd10, p.disease.icd10)
        push!(combined_names, p.disease.name)
    end

    combined_disease = DiseaseCode(
        doid = "COMBINED",
        icd10 = unique(combined_icd10),
        name = join(combined_names, " + ")
    )

    return DiseasePKProfile(combined_disease;
        gfr_adjustment = combined_gfr,
        hepatic_adjustment = combined_hepatic,
        fu_acidic_adjustment = combined_fu_acidic,
        fu_basic_adjustment = combined_fu_basic,
        vd_adjustment = combined_vd,
        absorption_adjustment = combined_absorption,
        albumin_concentration = combined_albumin,
        aag_concentration = combined_aag,
        special_considerations = vcat([p.special_considerations for p in profiles]...),
        evidence_level = :extrapolated
    )
end

# ============================================================================
# UTILITIES
# ============================================================================

"""
    list_supported_diseases()

List all diseases with PK profiles.
"""
function list_supported_diseases()
    diseases = Dict{String, String}()
    for (doid, profile) in DOID_PK_DATABASE
        diseases[doid] = profile.disease.name
    end
    return diseases
end

"""
    get_disease_summary(profile::DiseasePKProfile)

Get human-readable summary of disease PK effects.
"""
function get_disease_summary(profile::DiseasePKProfile)
    disease = profile.disease

    return Dict(
        "disease_name" => disease.name,
        "doid" => disease.doid,
        "icd10_codes" => disease.icd10,
        "icd11_codes" => disease.icd11,
        "pk_summary" => Dict(
            "gfr" => "$(round(profile.gfr_adjustment * 100))% of normal",
            "hepatic" => "$(round(profile.hepatic_adjustment * 100))% of normal",
            "fu_acidic_drugs" => "$(round(profile.fu_acidic_adjustment, digits=2))× normal",
            "fu_basic_drugs" => "$(round(profile.fu_basic_adjustment, digits=2))× normal",
            "vd" => "$(round(profile.vd_adjustment, digits=2))× normal",
            "albumin" => "$(profile.albumin_concentration) g/L",
            "aag" => "$(profile.aag_concentration) g/L"
        ),
        "clinical_notes" => profile.special_considerations,
        "evidence" => profile.evidence_level
    )
end

end # module DiseaseOntologyPK
