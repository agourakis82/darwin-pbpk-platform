# =============================================================================
# TRANSPORTER ONTOGENY MODULE - Q1+ SOTA 2024
# =============================================================================
# Darwin PBPK Platform - Pediatric Drug Development Support
#
# Scientific Foundation:
# - Hunt et al. (2024) CPT Pharmacometrics - ML estimation of RT ontogeny
# - Cheung et al. (2019) Clin Pharmacol Ther - Developmental changes in transporters
# - Prasad et al. (2016) Drug Metab Dispos - Transporter proteomics
# - van Groen et al. (2021) Clin Pharmacokinet - Pediatric PBPK transporters
# - Upreti & Wahlstrom (2016) Expert Opin Drug Metab Toxicol - Transporter ontogeny
#
# Key Features:
# 1. Sigmoidal maturation functions for 10+ renal transporters
# 2. Hepatic transporter ontogeny (OATP1B1/1B3, OCT1, MRP2, BSEP)
# 3. Intestinal transporter development (P-gp, BCRP, PEPT1)
# 4. Brain transporter BBB maturation
# 5. Full OBO Foundry integration (UBERON, CL, GO, ChEBI, PR)
# 6. DOID disease mappings for pediatric conditions
# 7. ICD-10/ICD-11 classification codes
#
# This module enables:
# - Pediatric dose extrapolation from adult data
# - Neonatal/infant PK prediction
# - Age-appropriate DDI assessment
# - Transporter-mediated disease modeling
# =============================================================================

module TransporterOntogeny

using Statistics
using Dates

export OntogenyProfile, TransporterOntogenyData, PediatricAge
export AgeCategory, OrganTransporters, OntogenyFunction
export PRETERM_NEONATE, TERM_NEONATE, INFANT, TODDLER, CHILD, ADOLESCENT, ADULT, ELDERLY
export OBOFoundryTerms, DOIDPediatric, ICDClassification

# Maturation functions and types
export SigmoidalMaturation, BiphasicMaturation, LinearMaturation
export calculate_maturation, calculate_ontogeny_factor, get_maturation_half_age
export sigmoidal_maturation, hill_maturation, biphasic_maturation
export get_transporter_ontogeny, apply_ontogeny_scaling

# Organ-specific
export get_renal_transporter_ontogeny, get_hepatic_transporter_ontogeny
export get_intestinal_transporter_ontogeny, get_bbb_transporter_ontogeny

# Disease integration
export get_pediatric_disease_modifiers, get_doid_pediatric_conditions
export get_icd_codes_for_condition, map_condition_to_transporter_effect

# Presets and validation
export RENAL_TRANSPORTER_ONTOGENY, HEPATIC_TRANSPORTER_ONTOGENY
export INTESTINAL_TRANSPORTER_ONTOGENY, BBB_TRANSPORTER_ONTOGENY
export validate_ontogeny_prediction, compare_with_clinical_data
export predict_pediatric_clearance, generate_ontogeny_curve

# =============================================================================
# OBO FOUNDRY ONTOLOGY INTEGRATION
# =============================================================================

"""
    OBOFoundryTerms

Integration with OBO Foundry ontologies for semantic annotation.

Supported Ontologies:
- UBERON: Anatomical structures (kidney, liver, intestine, brain)
- CL: Cell types (hepatocyte, enterocyte, proximal tubule cell)
- GO: Gene Ontology (transporter activity, drug transport)
- ChEBI: Chemical entities (drug substrates)
- PR: Protein Ontology (transporter proteins)
- DOID: Disease Ontology
- HP: Human Phenotype Ontology
"""
const OBO_FOUNDRY_PREFIXES = Dict{String, String}(
    # Anatomy & Cell
    "UBERON" => "http://purl.obolibrary.org/obo/UBERON_",
    "CL" => "http://purl.obolibrary.org/obo/CL_",
    "FMA" => "http://purl.org/sig/ont/fma/fma",

    # Function & Process
    "GO" => "http://purl.obolibrary.org/obo/GO_",
    "PR" => "http://purl.obolibrary.org/obo/PR_",

    # Chemistry
    "CHEBI" => "http://purl.obolibrary.org/obo/CHEBI_",
    "DRUGBANK" => "http://identifiers.org/drugbank/",

    # Disease & Phenotype
    "DOID" => "http://purl.obolibrary.org/obo/DOID_",
    "HP" => "http://purl.obolibrary.org/obo/HP_",
    "MONDO" => "http://purl.obolibrary.org/obo/MONDO_",
    "ORDO" => "http://www.orpha.net/ORDO/Orphanet_",

    # Clinical
    "ICD10" => "http://purl.bioontology.org/ontology/ICD10CM/",
    "ICD11" => "http://id.who.int/icd/entity/",
    "SNOMED" => "http://snomed.info/id/",
    "LOINC" => "http://loinc.org/",

    # Gene & Variant
    "HGNC" => "http://identifiers.org/hgnc/",
    "NCBI_GENE" => "http://identifiers.org/ncbigene/",
    "UNIPROT" => "http://identifiers.org/uniprot/",
    "PHARMGKB" => "http://identifiers.org/pharmgkb.gene/"
)

"""
    UBERONTerms

Anatomical terms from UBERON ontology for transporter localization.
"""
const UBERON_TERMS = Dict{Symbol, NamedTuple}(
    # Kidney
    :kidney => (id = "UBERON:0002113", name = "kidney", parent = "UBERON:0000062"),
    :renal_cortex => (id = "UBERON:0001225", name = "renal cortex", parent = "UBERON:0002113"),
    :proximal_tubule => (id = "UBERON:0004134", name = "proximal tubule", parent = "UBERON:0001225"),
    :distal_tubule => (id = "UBERON:0004135", name = "distal tubule", parent = "UBERON:0001225"),
    :collecting_duct => (id = "UBERON:0001232", name = "collecting duct", parent = "UBERON:0001225"),
    :glomerulus => (id = "UBERON:0000074", name = "glomerulus", parent = "UBERON:0001225"),

    # Liver
    :liver => (id = "UBERON:0002107", name = "liver", parent = "UBERON:0000062"),
    :hepatocyte => (id = "UBERON:0001153", name = "hepatocyte", parent = "UBERON:0002107"),
    :bile_canaliculus => (id = "UBERON:0001154", name = "bile canaliculus", parent = "UBERON:0002107"),
    :sinusoid => (id = "UBERON:0001281", name = "liver sinusoid", parent = "UBERON:0002107"),
    :periportal_zone => (id = "UBERON:0001279", name = "periportal region", parent = "UBERON:0002107"),
    :centrilobular_zone => (id = "UBERON:0001280", name = "centrilobular region", parent = "UBERON:0002107"),

    # Intestine
    :small_intestine => (id = "UBERON:0002108", name = "small intestine", parent = "UBERON:0000160"),
    :duodenum => (id = "UBERON:0002114", name = "duodenum", parent = "UBERON:0002108"),
    :jejunum => (id = "UBERON:0002115", name = "jejunum", parent = "UBERON:0002108"),
    :ileum => (id = "UBERON:0002116", name = "ileum", parent = "UBERON:0002108"),
    :enterocyte => (id = "UBERON:0000066", name = "enterocyte", parent = "UBERON:0002108"),
    :colon => (id = "UBERON:0001155", name = "colon", parent = "UBERON:0000160"),

    # Brain/BBB
    :brain => (id = "UBERON:0000955", name = "brain", parent = "UBERON:0001016"),
    :blood_brain_barrier => (id = "UBERON:0000120", name = "blood-brain barrier", parent = "UBERON:0000955"),
    :choroid_plexus => (id = "UBERON:0001886", name = "choroid plexus", parent = "UBERON:0000955"),
    :brain_capillary => (id = "UBERON:0003528", name = "brain capillary", parent = "UBERON:0000955"),

    # Placenta
    :placenta => (id = "UBERON:0001987", name = "placenta", parent = "UBERON:0000062"),
    :syncytiotrophoblast => (id = "UBERON:0000371", name = "syncytiotrophoblast", parent = "UBERON:0001987")
)

"""
    CLTerms

Cell type terms from Cell Ontology for transporter expression.
"""
const CL_TERMS = Dict{Symbol, NamedTuple}(
    # Kidney cells
    :proximal_tubule_cell => (id = "CL:0002306", name = "proximal tubule epithelial cell", uberon = "UBERON:0004134"),
    :distal_tubule_cell => (id = "CL:0002305", name = "distal tubule epithelial cell", uberon = "UBERON:0004135"),
    :podocyte => (id = "CL:0000653", name = "podocyte", uberon = "UBERON:0000074"),

    # Liver cells
    :hepatocyte => (id = "CL:0000182", name = "hepatocyte", uberon = "UBERON:0002107"),
    :cholangiocyte => (id = "CL:0000356", name = "cholangiocyte", uberon = "UBERON:0002107"),
    :kupffer_cell => (id = "CL:0000091", name = "Kupffer cell", uberon = "UBERON:0001281"),

    # Intestinal cells
    :enterocyte => (id = "CL:0000584", name = "enterocyte", uberon = "UBERON:0002108"),
    :goblet_cell => (id = "CL:0000160", name = "goblet cell", uberon = "UBERON:0002108"),

    # Brain/BBB cells
    :brain_endothelial => (id = "CL:0000653", name = "brain endothelial cell", uberon = "UBERON:0000120"),
    :astrocyte => (id = "CL:0000127", name = "astrocyte", uberon = "UBERON:0000955"),
    :pericyte => (id = "CL:0000669", name = "pericyte", uberon = "UBERON:0000955")
)

"""
    GOTerms

Gene Ontology terms for transporter molecular functions.
"""
const GO_TERMS = Dict{Symbol, NamedTuple}(
    # Molecular Function - Transport
    :drug_transmembrane_transport => (id = "GO:0006855", name = "drug transmembrane transport", aspect = "BP"),
    :organic_anion_transport => (id = "GO:0015711", name = "organic anion transport", aspect = "BP"),
    :organic_cation_transport => (id = "GO:0015695", name = "organic cation transport", aspect = "BP"),
    :bile_acid_transport => (id = "GO:0015721", name = "bile acid transport", aspect = "BP"),
    :xenobiotic_transport => (id = "GO:0042908", name = "xenobiotic transport", aspect = "BP"),

    # Transporter Activities
    :ABC_transporter => (id = "GO:0042626", name = "ATPase-coupled transmembrane transporter activity", aspect = "MF"),
    :SLC_transporter => (id = "GO:0022857", name = "transmembrane transporter activity", aspect = "MF"),
    :efflux_pump => (id = "GO:0015562", name = "efflux transmembrane transporter activity", aspect = "MF"),
    :uptake_transporter => (id = "GO:0015293", name = "symporter activity", aspect = "MF"),

    # Cellular Component
    :apical_membrane => (id = "GO:0016324", name = "apical plasma membrane", aspect = "CC"),
    :basolateral_membrane => (id = "GO:0016323", name = "basolateral plasma membrane", aspect = "CC"),
    :canalicular_membrane => (id = "GO:0046580", name = "canalicular membrane", aspect = "CC")
)

"""
    PRTerms

Protein Ontology terms for transporter proteins.
"""
const PR_TERMS = Dict{Symbol, NamedTuple}(
    # Renal Transporters
    :OAT1 => (id = "PR:Q4U2R8", uniprot = "Q4U2R8", gene = "SLC22A6", hgnc = "10968"),
    :OAT3 => (id = "PR:Q8TCC7", uniprot = "Q8TCC7", gene = "SLC22A8", hgnc = "10970"),
    :OCT2 => (id = "PR:O15244", uniprot = "O15244", gene = "SLC22A2", hgnc = "10963"),
    :MATE1 => (id = "PR:Q96FL8", uniprot = "Q96FL8", gene = "SLC47A1", hgnc = "29601"),
    :MATE2K => (id = "PR:Q86VL8", uniprot = "Q86VL8", gene = "SLC47A2", hgnc = "29602"),
    :URAT1 => (id = "PR:Q96S37", uniprot = "Q96S37", gene = "SLC22A12", hgnc = "17989"),

    # Hepatic Transporters
    :OATP1B1 => (id = "PR:Q9Y6L6", uniprot = "Q9Y6L6", gene = "SLCO1B1", hgnc = "10959"),
    :OATP1B3 => (id = "PR:Q9NPD5", uniprot = "Q9NPD5", gene = "SLCO1B3", hgnc = "10960"),
    :OCT1 => (id = "PR:O15245", uniprot = "O15245", gene = "SLC22A1", hgnc = "10962"),
    :NTCP => (id = "PR:Q14973", uniprot = "Q14973", gene = "SLC10A1", hgnc = "10905"),
    :BSEP => (id = "PR:O95342", uniprot = "O95342", gene = "ABCB11", hgnc = "42"),
    :MRP2 => (id = "PR:Q92887", uniprot = "Q92887", gene = "ABCC2", hgnc = "53"),
    :MRP3 => (id = "PR:O15438", uniprot = "O15438", gene = "ABCC3", hgnc = "54"),

    # Efflux Transporters (ubiquitous)
    :P_gp => (id = "PR:P08183", uniprot = "P08183", gene = "ABCB1", hgnc = "40"),
    :BCRP => (id = "PR:Q9UNQ0", uniprot = "Q9UNQ0", gene = "ABCG2", hgnc = "74"),
    :MRP4 => (id = "PR:O15439", uniprot = "O15439", gene = "ABCC4", hgnc = "55"),

    # Intestinal Transporters
    :PEPT1 => (id = "PR:P46059", uniprot = "P46059", gene = "SLC15A1", hgnc = "10920"),
    :PEPT2 => (id = "PR:Q16348", uniprot = "Q16348", gene = "SLC15A2", hgnc = "10921"),
    :OATP2B1 => (id = "PR:O94956", uniprot = "O94956", gene = "SLCO2B1", hgnc = "10961"),

    # Nutrient Transporters (BBB)
    :GLUT1 => (id = "PR:P11166", uniprot = "P11166", gene = "SLC2A1", hgnc = "11005"),
    :LAT1 => (id = "PR:Q01650", uniprot = "Q01650", gene = "SLC7A5", hgnc = "11063"),
    :MCT1 => (id = "PR:P53985", uniprot = "P53985", gene = "SLC16A1", hgnc = "10922")
)

# =============================================================================
# ICD-10 AND ICD-11 CLASSIFICATION
# =============================================================================

"""
    ICDClassification

ICD-10-CM and ICD-11 codes for pediatric conditions affecting transporters.
"""
const ICD_CLASSIFICATIONS = Dict{Symbol, NamedTuple}(
    # Neonatal conditions
    :prematurity => (
        icd10 = ["P07.0", "P07.1", "P07.2", "P07.3"],
        icd11 = ["KA21.0", "KA21.1", "KA21.2"],
        doid = "DOID:0060673",
        description = "Disorders related to short gestation and low birth weight"
    ),
    :neonatal_jaundice => (
        icd10 = ["P59.0", "P59.9", "P58.0"],
        icd11 = ["KB60", "KB61"],
        doid = "DOID:0080032",
        description = "Neonatal jaundice (affects OATP1B1/MRP2)"
    ),
    :neonatal_sepsis => (
        icd10 = ["P36.0", "P36.9"],
        icd11 = ["KA65.0", "KA65.9"],
        doid = "DOID:0080810",
        description = "Bacterial sepsis of newborn"
    ),

    # Pediatric kidney diseases
    :pediatric_aki => (
        icd10 = ["N17.0", "N17.9"],
        icd11 = ["GB60", "GB61"],
        doid = "DOID:0050630",
        description = "Acute kidney injury in children"
    ),
    :pediatric_ckd => (
        icd10 = ["N18.1", "N18.2", "N18.3", "N18.4", "N18.5"],
        icd11 = ["GB61.0", "GB61.1", "GB61.2", "GB61.3", "GB61.4"],
        doid = "DOID:784",
        description = "Chronic kidney disease stages 1-5"
    ),
    :nephrotic_syndrome => (
        icd10 = ["N04.0", "N04.9"],
        icd11 = ["GB40.0"],
        doid = "DOID:1184",
        description = "Nephrotic syndrome (affects protein binding)"
    ),
    :renal_tubular_disorder => (
        icd10 = ["N25.0", "N25.9"],
        icd11 = ["GB90.0"],
        doid = "DOID:447",
        description = "Disorders of renal tubular function"
    ),

    # Pediatric liver diseases
    :biliary_atresia => (
        icd10 = ["Q44.2"],
        icd11 = ["LB10.0"],
        doid = "DOID:8545",
        description = "Biliary atresia (affects BSEP/MRP2)"
    ),
    :neonatal_cholestasis => (
        icd10 = ["P59.1", "K71.0"],
        icd11 = ["KB62", "DB96.0"],
        doid = "DOID:13580",
        description = "Neonatal cholestatic liver disease"
    ),
    :progressive_familial_cholestasis => (
        icd10 = ["K76.89"],
        icd11 = ["DB99.Y"],
        doid = "DOID:0060643",
        description = "PFIC types 1-3 (BSEP mutations)"
    ),
    :pediatric_nafld => (
        icd10 = ["K76.0"],
        icd11 = ["DB92.0"],
        doid = "DOID:0080208",
        description = "Non-alcoholic fatty liver disease in children"
    ),

    # Metabolic/Genetic
    :gilberts_syndrome => (
        icd10 = ["E80.4"],
        icd11 = ["5C58.20"],
        doid = "DOID:2739",
        description = "Gilbert syndrome (UGT1A1 polymorphism)"
    ),
    :cystic_fibrosis => (
        icd10 = ["E84.0", "E84.9"],
        icd11 = ["CA25.0"],
        doid = "DOID:1485",
        description = "Cystic fibrosis (affects intestinal transporters)"
    ),
    :phenylketonuria => (
        icd10 = ["E70.0"],
        icd11 = ["5C50.00"],
        doid = "DOID:9281",
        description = "PKU (affects LAT1 substrates)"
    ),

    # Infectious/Inflammatory
    :pediatric_sepsis => (
        icd10 = ["A41.9", "R65.20"],
        icd11 = ["1G40", "MG40"],
        doid = "DOID:0060166",
        description = "Sepsis in children (CYP/transporter downregulation)"
    ),
    :kawasaki_disease => (
        icd10 = ["M30.3"],
        icd11 = ["4A44.1"],
        doid = "DOID:13378",
        description = "Kawasaki disease (acute inflammation)"
    ),
    :inflammatory_bowel_disease => (
        icd10 = ["K50.0", "K51.0"],
        icd11 = ["DD70", "DD71"],
        doid = "DOID:0050589",
        description = "IBD (affects intestinal P-gp/BCRP)"
    ),

    # Oncology
    :pediatric_all => (
        icd10 = ["C91.00"],
        icd11 = ["2A80.1"],
        doid = "DOID:9952",
        description = "Acute lymphoblastic leukemia (MTX transport)"
    ),
    :neuroblastoma => (
        icd10 = ["C74.90"],
        icd11 = ["2D10.0"],
        doid = "DOID:769",
        description = "Neuroblastoma (OCT2 substrate chemotherapy)"
    ),
    :wilms_tumor => (
        icd10 = ["C64.9"],
        icd11 = ["2C90.0"],
        doid = "DOID:2154",
        description = "Wilms tumor (renal function impact)"
    )
)

"""
    DOIDPediatricConditions

DOID terms specifically relevant to pediatric pharmacology.
"""
const DOID_PEDIATRIC_CONDITIONS = Dict{String, NamedTuple}(
    "DOID:0060673" => (name = "prematurity", transporter_effects = [:reduced_all], severity = :major),
    "DOID:0080032" => (name = "neonatal jaundice", transporter_effects = [:OATP1B1_reduced, :MRP2_reduced], severity = :moderate),
    "DOID:0080810" => (name = "neonatal sepsis", transporter_effects = [:cyp_downregulation, :transporter_downregulation], severity = :major),
    "DOID:784" => (name = "chronic kidney disease", transporter_effects = [:OAT1_reduced, :OAT3_reduced, :OCT2_reduced], severity = :variable),
    "DOID:8545" => (name = "biliary atresia", transporter_effects = [:BSEP_deficient, :MRP2_deficient], severity = :major),
    "DOID:0060643" => (name = "PFIC", transporter_effects = [:BSEP_deficient], severity = :major),
    "DOID:9281" => (name = "phenylketonuria", transporter_effects = [:LAT1_competition], severity = :moderate),
    "DOID:1485" => (name = "cystic fibrosis", transporter_effects = [:intestinal_altered], severity = :moderate),
    "DOID:0050589" => (name = "IBD", transporter_effects = [:Pgp_reduced, :BCRP_reduced], severity = :moderate)
)

# =============================================================================
# AGE CATEGORIES AND DEVELOPMENTAL STAGES
# =============================================================================

"""
    AgeCategory

FDA/ICH pediatric age categories with developmental characteristics.
"""
@enum AgeCategory begin
    PRETERM_NEONATE      # < 37 weeks gestational age
    TERM_NEONATE         # 0-27 days
    INFANT               # 28 days - 23 months
    TODDLER              # 2-5 years
    CHILD                # 6-11 years
    ADOLESCENT           # 12-17 years
    ADULT                # 18-65 years
    ELDERLY              # > 65 years
end

"""
    PediatricAge

Comprehensive pediatric age representation.
"""
struct PediatricAge
    postnatal_days::Float64         # Days after birth
    gestational_weeks::Float64      # Weeks at birth (37-42 term)
    postmenstrual_age_weeks::Float64  # GA + postnatal weeks
    category::AgeCategory
    corrected_age_days::Float64     # For premature: postnatal - (40-GA)*7
end

function PediatricAge(postnatal_days::Float64; gestational_weeks::Float64 = 40.0)
    postnatal_weeks = postnatal_days / 7.0
    pma = gestational_weeks + postnatal_weeks

    # Corrected age for prematurity
    corrected = postnatal_days - (40.0 - gestational_weeks) * 7.0
    corrected = max(0.0, corrected)

    # Determine category
    category = if gestational_weeks < 37
        PRETERM_NEONATE
    elseif postnatal_days <= 27
        TERM_NEONATE
    elseif postnatal_days <= 730  # 2 years
        INFANT
    elseif postnatal_days <= 2190  # 6 years
        TODDLER
    elseif postnatal_days <= 4380  # 12 years
        CHILD
    elseif postnatal_days <= 6570  # 18 years
        ADOLESCENT
    elseif postnatal_days <= 23725  # 65 years
        ADULT
    else
        ELDERLY
    end

    PediatricAge(postnatal_days, gestational_weeks, pma, category, corrected)
end

# Convenience constructors - keyword argument version
function PediatricAge(; years::Float64 = 0.0, months::Float64 = 0.0, days::Float64 = 0.0,
                       gestational_weeks::Float64 = 40.0)
    total_days = years * 365.25 + months * 30.44 + days
    PediatricAge(total_days; gestational_weeks = gestational_weeks)
end

# =============================================================================
# ONTOGENY FUNCTION TYPES
# =============================================================================

"""
    OntogenyFunction

Abstract type for maturation functions.
"""
abstract type OntogenyFunction end

"""
    SigmoidalMaturation

Classic sigmoidal (Hill) maturation function:

    f(age) = age^γ / (TM50^γ + age^γ)

Where:
- TM50 = age at 50% of adult activity
- γ = Hill coefficient (steepness)
"""
struct SigmoidalMaturation <: OntogenyFunction
    TM50::Float64       # Age at 50% maturation (days or PMA weeks)
    gamma::Float64      # Hill coefficient
    Fmax::Float64       # Maximum fraction of adult (usually 1.0)
    Fmin::Float64       # Minimum fraction at birth
    age_unit::Symbol    # :days, :weeks_pma, :months, :years
end

"""
    BiphasicMaturation

Two-phase maturation (rapid early + slow late):

    f(age) = w1 × sigmoid1(age) + w2 × sigmoid2(age)
"""
struct BiphasicMaturation <: OntogenyFunction
    TM50_early::Float64
    gamma_early::Float64
    weight_early::Float64
    TM50_late::Float64
    gamma_late::Float64
    weight_late::Float64
    age_unit::Symbol
end

"""
    LinearMaturation

Simple linear maturation (rare, for some transporters):

    f(age) = Fmin + (Fmax - Fmin) × min(1, age/Tmax)
"""
struct LinearMaturation <: OntogenyFunction
    Tmax::Float64       # Age at full maturation
    Fmax::Float64
    Fmin::Float64
    age_unit::Symbol
end

# =============================================================================
# MATURATION CALCULATION FUNCTIONS
# =============================================================================

"""
    calculate_maturation(func::SigmoidalMaturation, age::PediatricAge) -> Float64

Calculate fraction of adult activity using sigmoidal maturation.
"""
function calculate_maturation(func::SigmoidalMaturation, age::PediatricAge)::Float64
    # Convert age to appropriate unit
    age_val = if func.age_unit == :days
        age.postnatal_days
    elseif func.age_unit == :weeks_pma
        age.postmenstrual_age_weeks
    elseif func.age_unit == :months
        age.postnatal_days / 30.44
    elseif func.age_unit == :years
        age.postnatal_days / 365.25
    else
        age.postnatal_days
    end

    # Hill equation
    if age_val <= 0
        return func.Fmin
    end

    sigmoid = age_val^func.gamma / (func.TM50^func.gamma + age_val^func.gamma)
    fraction = func.Fmin + (func.Fmax - func.Fmin) * sigmoid

    return clamp(fraction, func.Fmin, func.Fmax)
end

function calculate_maturation(func::BiphasicMaturation, age::PediatricAge)::Float64
    age_val = if func.age_unit == :days
        age.postnatal_days
    elseif func.age_unit == :weeks_pma
        age.postmenstrual_age_weeks
    else
        age.postnatal_days / 365.25
    end

    if age_val <= 0
        return 0.0
    end

    # Two sigmoids
    sig1 = age_val^2 / (func.TM50_early^2 + age_val^2)
    sig2 = age_val^func.gamma_late / (func.TM50_late^func.gamma_late + age_val^func.gamma_late)

    fraction = func.weight_early * sig1 + func.weight_late * sig2
    return clamp(fraction, 0.0, 1.0)
end

function calculate_maturation(func::LinearMaturation, age::PediatricAge)::Float64
    age_val = if func.age_unit == :years
        age.postnatal_days / 365.25
    else
        age.postnatal_days
    end

    if age_val >= func.Tmax
        return func.Fmax
    end

    return func.Fmin + (func.Fmax - func.Fmin) * (age_val / func.Tmax)
end

# =============================================================================
# RENAL TRANSPORTER ONTOGENY PROFILES
# =============================================================================
# Source: Hunt et al. (2024) CPT Pharmacometrics & Systems Pharmacology
# Maximum likelihood estimation from human kidney cortex samples

"""
    RENAL_TRANSPORTER_ONTOGENY

Validated ontogeny profiles for renal transporters.
Parameters from Hunt et al. (2024) and Cheung et al. (2019).
"""
const RENAL_TRANSPORTER_ONTOGENY = Dict{Symbol, NamedTuple}(
    # === BASOLATERAL UPTAKE ===
    :OAT1 => (
        name = "Organic Anion Transporter 1",
        gene = "SLC22A6",
        protein = PR_TERMS[:OAT1],
        location = :basolateral,
        cell_type = CL_TERMS[:proximal_tubule_cell],
        go_function = GO_TERMS[:organic_anion_transport],
        maturation = SigmoidalMaturation(
            213.0,    # TM50 = 7.1 months postnatal
            1.86,     # gamma
            1.0,      # Fmax
            0.05,     # Fmin (5% at birth)
            :days
        ),
        adult_expression = 4.0,  # pmol/mg protein
        cv_adult = 0.38,
        substrates = ["PAH", "tenofovir", "adefovir", "cidofovir", "furosemide", "methotrexate"],
        clinical_impact = "Major pathway for antiviral nephrotoxicity; immature in neonates",
        references = ["Hunt 2024", "Cheung 2019", "Prasad 2016"]
    ),

    :OAT3 => (
        name = "Organic Anion Transporter 3",
        gene = "SLC22A8",
        protein = PR_TERMS[:OAT3],
        location = :basolateral,
        cell_type = CL_TERMS[:proximal_tubule_cell],
        go_function = GO_TERMS[:organic_anion_transport],
        maturation = SigmoidalMaturation(
            198.0,    # TM50 = 6.6 months
            1.72,     # gamma
            1.0,
            0.08,     # 8% at birth
            :days
        ),
        adult_expression = 2.5,
        cv_adult = 0.40,
        substrates = ["pravastatin", "rosuvastatin", "cimetidine", "estrone_sulfate", "benzylpenicillin"],
        clinical_impact = "Statin and antibiotic secretion; develops parallel to OAT1",
        references = ["Hunt 2024", "van Groen 2021"]
    ),

    :OCT2 => (
        name = "Organic Cation Transporter 2",
        gene = "SLC22A2",
        protein = PR_TERMS[:OCT2],
        location = :basolateral,
        cell_type = CL_TERMS[:proximal_tubule_cell],
        go_function = GO_TERMS[:organic_cation_transport],
        maturation = SigmoidalMaturation(
            152.0,    # TM50 = 5.1 months (faster than OATs)
            2.10,     # gamma (steeper)
            1.0,
            0.12,     # 12% at birth
            :days
        ),
        adult_expression = 6.0,
        cv_adult = 0.33,
        substrates = ["metformin", "cisplatin", "oxaliplatin", "amiloride", "cimetidine"],
        clinical_impact = "Cisplatin nephrotoxicity lower in infants; metformin clearance reduced",
        references = ["Hunt 2024", "Motohashi 2013"]
    ),

    # === APICAL EFFLUX ===
    :MATE1 => (
        name = "Multidrug and Toxin Extrusion 1",
        gene = "SLC47A1",
        protein = PR_TERMS[:MATE1],
        location = :apical,
        cell_type = CL_TERMS[:proximal_tubule_cell],
        go_function = GO_TERMS[:efflux_pump],
        maturation = SigmoidalMaturation(
            305.0,    # TM50 = 10.2 months (slower than OCT2)
            1.45,     # gamma
            1.0,
            0.15,     # 15% at birth
            :days
        ),
        adult_expression = 2.5,
        cv_adult = 0.35,
        substrates = ["metformin", "cimetidine", "oxaliplatin", "acyclovir"],
        clinical_impact = "Rate-limiting for OCT2 substrates; MATE1<OCT2 causes accumulation risk",
        references = ["Hunt 2024", "Motohashi 2013"]
    ),

    :MATE2K => (
        name = "Multidrug and Toxin Extrusion 2-K",
        gene = "SLC47A2",
        protein = PR_TERMS[:MATE2K],
        location = :apical,
        cell_type = CL_TERMS[:proximal_tubule_cell],
        go_function = GO_TERMS[:efflux_pump],
        maturation = SigmoidalMaturation(
            335.0,    # TM50 = 11.2 months
            1.38,     # gamma
            1.0,
            0.10,     # 10% at birth
            :days
        ),
        adult_expression = 1.5,
        cv_adult = 0.40,
        substrates = ["metformin", "oxaliplatin", "cimetidine"],
        clinical_impact = "Kidney-specific; complements MATE1",
        references = ["Hunt 2024"]
    ),

    # === APICAL EFFLUX (ABC) ===
    :P_gp_renal => (
        name = "P-glycoprotein (renal)",
        gene = "ABCB1",
        protein = PR_TERMS[:P_gp],
        location = :apical,
        cell_type = CL_TERMS[:proximal_tubule_cell],
        go_function = GO_TERMS[:ABC_transporter],
        maturation = SigmoidalMaturation(
            183.0,    # TM50 = 6.1 months
            1.95,     # gamma
            1.0,
            0.25,     # 25% at birth (higher baseline)
            :days
        ),
        adult_expression = 0.8,
        cv_adult = 0.45,
        substrates = ["digoxin", "tacrolimus", "cyclosporine", "loperamide", "fexofenadine"],
        clinical_impact = "Important for immunosuppressant dosing in transplant",
        references = ["Hunt 2024", "van Kalken 1992"]
    ),

    :BCRP_renal => (
        name = "Breast Cancer Resistance Protein (renal)",
        gene = "ABCG2",
        protein = PR_TERMS[:BCRP],
        location = :apical,
        cell_type = CL_TERMS[:proximal_tubule_cell],
        go_function = GO_TERMS[:ABC_transporter],
        maturation = SigmoidalMaturation(
            244.0,    # TM50 = 8.1 months
            1.62,     # gamma
            1.0,
            0.18,     # 18% at birth
            :days
        ),
        adult_expression = 1.2,
        cv_adult = 0.40,
        substrates = ["methotrexate", "rosuvastatin", "sulfasalazine", "topotecan"],
        clinical_impact = "MTX elimination; important for pediatric oncology",
        references = ["Hunt 2024", "Maliepaard 2001"]
    ),

    :MRP2_renal => (
        name = "Multidrug Resistance Protein 2 (renal)",
        gene = "ABCC2",
        protein = PR_TERMS[:MRP2],
        location = :apical,
        cell_type = CL_TERMS[:proximal_tubule_cell],
        go_function = GO_TERMS[:ABC_transporter],
        maturation = SigmoidalMaturation(
            274.0,    # TM50 = 9.1 months
            1.55,     # gamma
            1.0,
            0.12,     # 12% at birth
            :days
        ),
        adult_expression = 1.5,
        cv_adult = 0.42,
        substrates = ["methotrexate", "cisplatin_conjugates", "bilirubin_glucuronides"],
        clinical_impact = "Conjugate efflux; immature contributes to neonatal jaundice",
        references = ["Hunt 2024", "Nies 2004"]
    ),

    :MRP4_renal => (
        name = "Multidrug Resistance Protein 4 (renal)",
        gene = "ABCC4",
        protein = PR_TERMS[:MRP4],
        location = :apical,
        cell_type = CL_TERMS[:proximal_tubule_cell],
        go_function = GO_TERMS[:ABC_transporter],
        maturation = SigmoidalMaturation(
            213.0,    # TM50 = 7.1 months
            1.78,     # gamma
            1.0,
            0.20,     # 20% at birth
            :days
        ),
        adult_expression = 2.0,
        cv_adult = 0.38,
        substrates = ["adefovir", "tenofovir", "furosemide", "methotrexate", "cAMP", "cGMP"],
        clinical_impact = "Nucleotide analog efflux; antiviral dosing",
        references = ["Hunt 2024", "van Aubel 2002"]
    ),

    # === REABSORPTION ===
    :URAT1 => (
        name = "Urate Transporter 1",
        gene = "SLC22A12",
        protein = PR_TERMS[:URAT1],
        location = :apical,
        cell_type = CL_TERMS[:proximal_tubule_cell],
        go_function = GO_TERMS[:uptake_transporter],
        maturation = SigmoidalMaturation(
            122.0,    # TM50 = 4.1 months (early maturation)
            2.35,     # gamma (steep)
            1.0,
            0.35,     # 35% at birth (higher baseline)
            :days
        ),
        adult_expression = 2.2,
        cv_adult = 0.35,
        substrates = ["uric_acid", "lactate", "nicotinate"],
        clinical_impact = "Urate handling; neonates have low uric acid",
        references = ["Hunt 2024", "Enomoto 2002"]
    )
)

# =============================================================================
# HEPATIC TRANSPORTER ONTOGENY PROFILES
# =============================================================================
# Sources: Prasad et al. (2016), van Groen et al. (2021), Mooij et al. (2016)

const HEPATIC_TRANSPORTER_ONTOGENY = Dict{Symbol, NamedTuple}(
    # === SINUSOIDAL UPTAKE ===
    :OATP1B1 => (
        name = "Organic Anion Transporting Polypeptide 1B1",
        gene = "SLCO1B1",
        protein = PR_TERMS[:OATP1B1],
        location = :sinusoidal,
        cell_type = CL_TERMS[:hepatocyte],
        go_function = GO_TERMS[:organic_anion_transport],
        maturation = SigmoidalMaturation(
            548.0,    # TM50 = 1.5 years (slow maturation)
            1.42,     # gamma
            1.0,
            0.10,     # 10% at birth
            :days
        ),
        adult_expression = 3.8,
        cv_adult = 0.52,  # High variability (SLCO1B1*5, *15)
        zonal_distribution = (periportal = 1.0, centrilobular = 2.1),  # 2.1× higher centrilobular
        substrates = ["atorvastatin", "rosuvastatin", "pravastatin", "repaglinide", "rifampicin", "methotrexate"],
        clinical_impact = "Statin myopathy risk; bilirubin uptake → neonatal jaundice",
        references = ["Prasad 2016", "Mooij 2016", "CPT Pharmacometrics Sept 2024"]
    ),

    :OATP1B3 => (
        name = "Organic Anion Transporting Polypeptide 1B3",
        gene = "SLCO1B3",
        protein = PR_TERMS[:OATP1B3],
        location = :sinusoidal,
        cell_type = CL_TERMS[:hepatocyte],
        go_function = GO_TERMS[:organic_anion_transport],
        maturation = SigmoidalMaturation(
            487.0,    # TM50 = 1.33 years
            1.55,     # gamma
            1.0,
            0.08,     # 8% at birth
            :days
        ),
        adult_expression = 1.2,
        cv_adult = 0.48,
        zonal_distribution = (periportal = 1.0, centrilobular = 1.8),
        substrates = ["telmisartan", "docetaxel", "paclitaxel", "digoxin", "CCK-8"],
        clinical_impact = "Taxane hepatotoxicity; lower in children",
        references = ["Prasad 2016", "König 2006"]
    ),

    :OCT1 => (
        name = "Organic Cation Transporter 1",
        gene = "SLC22A1",
        protein = PR_TERMS[:OCT1],
        location = :sinusoidal,
        cell_type = CL_TERMS[:hepatocyte],
        go_function = GO_TERMS[:organic_cation_transport],
        maturation = SigmoidalMaturation(
            365.0,    # TM50 = 1 year
            1.68,     # gamma
            1.0,
            0.15,     # 15% at birth
            :days
        ),
        adult_expression = 2.8,
        cv_adult = 0.45,  # OCT1*2, *3, *4 polymorphisms
        zonal_distribution = (periportal = 1.2, centrilobular = 1.0),
        substrates = ["metformin", "morphine", "tramadol", "ondansetron", "tropisetron"],
        clinical_impact = "Metformin hepatic uptake; morphine glucuronidation",
        references = ["Prasad 2016", "Nies 2009"]
    ),

    :NTCP => (
        name = "Sodium Taurocholate Co-transporting Polypeptide",
        gene = "SLC10A1",
        protein = PR_TERMS[:NTCP],
        location = :sinusoidal,
        cell_type = CL_TERMS[:hepatocyte],
        go_function = GO_TERMS[:bile_acid_transport],
        maturation = SigmoidalMaturation(
            730.0,    # TM50 = 2 years (very slow)
            1.25,     # gamma (gradual)
            1.0,
            0.05,     # 5% at birth - critically low
            :days
        ),
        adult_expression = 1.5,
        cv_adult = 0.35,
        zonal_distribution = (periportal = 1.0, centrilobular = 0.6),
        substrates = ["taurocholate", "glycocholate", "rosuvastatin", "HBV_receptor"],
        clinical_impact = "Bile acid recycling; physiologic cholestasis in newborns",
        references = ["Mooij 2016", "Yan 2012"]
    ),

    # === CANALICULAR EFFLUX ===
    :BSEP => (
        name = "Bile Salt Export Pump",
        gene = "ABCB11",
        protein = PR_TERMS[:BSEP],
        location = :canalicular,
        cell_type = CL_TERMS[:hepatocyte],
        go_function = GO_TERMS[:bile_acid_transport],
        maturation = SigmoidalMaturation(
            913.0,    # TM50 = 2.5 years (very slow)
            1.18,     # gamma
            1.0,
            0.08,     # 8% at birth
            :days
        ),
        adult_expression = 2.2,
        cv_adult = 0.40,
        zonal_distribution = (periportal = 0.7, centrilobular = 1.0),
        substrates = ["taurocholate", "glycocholate", "pravastatin"],
        clinical_impact = "PFIC2 mutations; neonatal cholestasis; DILI risk",
        references = ["Mooij 2016", "Strautnieks 1998"]
    ),

    :MRP2_hepatic => (
        name = "Multidrug Resistance Protein 2 (hepatic)",
        gene = "ABCC2",
        protein = PR_TERMS[:MRP2],
        location = :canalicular,
        cell_type = CL_TERMS[:hepatocyte],
        go_function = GO_TERMS[:ABC_transporter],
        maturation = SigmoidalMaturation(
            639.0,    # TM50 = 1.75 years
            1.35,     # gamma
            1.0,
            0.12,     # 12% at birth
            :days
        ),
        adult_expression = 3.5,
        cv_adult = 0.42,
        zonal_distribution = (periportal = 0.8, centrilobular = 1.0),
        substrates = ["bilirubin_glucuronide", "leukotriene_C4", "methotrexate", "irinotecan_SN38G"],
        clinical_impact = "Dubin-Johnson syndrome; conjugate biliary excretion",
        references = ["Prasad 2016", "Paulusma 1997"]
    ),

    :P_gp_hepatic => (
        name = "P-glycoprotein (hepatic)",
        gene = "ABCB1",
        protein = PR_TERMS[:P_gp],
        location = :canalicular,
        cell_type = CL_TERMS[:hepatocyte],
        go_function = GO_TERMS[:ABC_transporter],
        maturation = SigmoidalMaturation(
            456.0,    # TM50 = 1.25 years
            1.72,     # gamma
            1.0,
            0.22,     # 22% at birth
            :days
        ),
        adult_expression = 1.8,
        cv_adult = 0.55,
        zonal_distribution = (periportal = 0.9, centrilobular = 1.0),
        substrates = ["digoxin", "loperamide", "tacrolimus", "cyclosporine", "verapamil"],
        clinical_impact = "Immunosuppressant biliary clearance",
        references = ["Prasad 2016", "van Groen 2021"]
    ),

    :BCRP_hepatic => (
        name = "Breast Cancer Resistance Protein (hepatic)",
        gene = "ABCG2",
        protein = PR_TERMS[:BCRP],
        location = :canalicular,
        cell_type = CL_TERMS[:hepatocyte],
        go_function = GO_TERMS[:ABC_transporter],
        maturation = SigmoidalMaturation(
            578.0,    # TM50 = 1.58 years
            1.48,     # gamma
            1.0,
            0.15,     # 15% at birth
            :days
        ),
        adult_expression = 1.0,
        cv_adult = 0.50,
        zonal_distribution = (periportal = 1.0, centrilobular = 0.85),
        substrates = ["rosuvastatin", "sulfasalazine", "methotrexate", "topotecan", "porphyrins"],
        clinical_impact = "Statin biliary efflux; photosensitivity risk",
        references = ["Prasad 2016", "Vlaming 2009"]
    ),

    # === BASOLATERAL EFFLUX ===
    :MRP3 => (
        name = "Multidrug Resistance Protein 3",
        gene = "ABCC3",
        protein = PR_TERMS[:MRP3],
        location = :sinusoidal,
        cell_type = CL_TERMS[:hepatocyte],
        go_function = GO_TERMS[:efflux_pump],
        maturation = SigmoidalMaturation(
            426.0,    # TM50 = 1.17 years
            1.62,     # gamma
            1.0,
            0.18,     # 18% at birth
            :days
        ),
        adult_expression = 0.9,
        cv_adult = 0.45,
        zonal_distribution = (periportal = 0.6, centrilobular = 1.0),
        substrates = ["morphine_3G", "morphine_6G", "bilirubin_glucuronides", "etoposide_glucuronide"],
        clinical_impact = "Alternative exit for glucuronides when MRP2 impaired",
        references = ["Prasad 2016", "Zelcer 2006"]
    ),

    :MRP4_hepatic => (
        name = "Multidrug Resistance Protein 4 (hepatic)",
        gene = "ABCC4",
        protein = PR_TERMS[:MRP4],
        location = :sinusoidal,
        cell_type = CL_TERMS[:hepatocyte],
        go_function = GO_TERMS[:efflux_pump],
        maturation = SigmoidalMaturation(
            365.0,    # TM50 = 1 year
            1.55,     # gamma
            1.0,
            0.22,     # 22% at birth
            :days
        ),
        adult_expression = 0.6,
        cv_adult = 0.48,
        zonal_distribution = (periportal = 0.8, centrilobular = 1.0),
        substrates = ["bile_acids", "cAMP", "cGMP", "methotrexate", "adefovir"],
        clinical_impact = "Upregulated in cholestasis; compensatory pathway",
        references = ["Rius 2006"]
    )
)

# =============================================================================
# INTESTINAL TRANSPORTER ONTOGENY PROFILES
# =============================================================================

const INTESTINAL_TRANSPORTER_ONTOGENY = Dict{Symbol, NamedTuple}(
    :P_gp_intestinal => (
        name = "P-glycoprotein (intestinal)",
        gene = "ABCB1",
        protein = PR_TERMS[:P_gp],
        location = :apical,
        cell_type = CL_TERMS[:enterocyte],
        go_function = GO_TERMS[:ABC_transporter],
        maturation = SigmoidalMaturation(
            274.0,    # TM50 = 9 months
            1.85,     # gamma
            1.0,
            0.30,     # 30% at birth - relatively high
            :days
        ),
        adult_expression = 2.5,  # Increases duodenum → ileum
        cv_adult = 0.55,
        regional_gradient = (duodenum = 0.6, jejunum = 0.8, ileum = 1.0, colon = 1.2),
        substrates = ["digoxin", "loperamide", "tacrolimus", "cyclosporine"],
        clinical_impact = "Oral bioavailability of P-gp substrates higher in neonates",
        references = ["van Groen 2021", "Fakhoury 2005"]
    ),

    :BCRP_intestinal => (
        name = "Breast Cancer Resistance Protein (intestinal)",
        gene = "ABCG2",
        protein = PR_TERMS[:BCRP],
        location = :apical,
        cell_type = CL_TERMS[:enterocyte],
        go_function = GO_TERMS[:ABC_transporter],
        maturation = SigmoidalMaturation(
            305.0,    # TM50 = 10 months
            1.68,     # gamma
            1.0,
            0.25,     # 25% at birth
            :days
        ),
        adult_expression = 1.8,
        cv_adult = 0.50,
        regional_gradient = (duodenum = 1.0, jejunum = 0.9, ileum = 0.7, colon = 0.5),
        substrates = ["sulfasalazine", "rosuvastatin", "methotrexate", "topotecan"],
        clinical_impact = "Affects oral bioavailability of BCRP substrates",
        references = ["van Groen 2021", "Maliepaard 2001"]
    ),

    :PEPT1 => (
        name = "Peptide Transporter 1",
        gene = "SLC15A1",
        protein = PR_TERMS[:PEPT1],
        location = :apical,
        cell_type = CL_TERMS[:enterocyte],
        go_function = GO_TERMS[:uptake_transporter],
        maturation = SigmoidalMaturation(
            91.0,     # TM50 = 3 months (early maturation)
            2.25,     # gamma (steep)
            1.0,
            0.45,     # 45% at birth - high for nutrition
            :days
        ),
        adult_expression = 3.2,
        cv_adult = 0.35,
        regional_gradient = (duodenum = 1.0, jejunum = 0.9, ileum = 0.6, colon = 0.1),
        substrates = ["cephalexin", "amoxicillin", "valacyclovir", "enalapril", "captopril"],
        clinical_impact = "β-lactam absorption; early maturation for amino acid nutrition",
        references = ["Mooij 2016", "Shen 1999"]
    ),

    :OATP2B1_intestinal => (
        name = "Organic Anion Transporting Polypeptide 2B1",
        gene = "SLCO2B1",
        protein = PR_TERMS[:OATP2B1],
        location = :apical,
        cell_type = CL_TERMS[:enterocyte],
        go_function = GO_TERMS[:organic_anion_transport],
        maturation = SigmoidalMaturation(
            183.0,    # TM50 = 6 months
            1.75,     # gamma
            1.0,
            0.20,     # 20% at birth
            :days
        ),
        adult_expression = 1.5,
        cv_adult = 0.42,
        regional_gradient = (duodenum = 1.0, jejunum = 0.85, ileum = 0.5, colon = 0.3),
        substrates = ["fexofenadine", "rosuvastatin", "atorvastatin", "glyburide"],
        clinical_impact = "Fruit juice interactions (naringin inhibition)",
        references = ["Mooij 2016", "Tamai 2000"]
    ),

    :MRP2_intestinal => (
        name = "Multidrug Resistance Protein 2 (intestinal)",
        gene = "ABCC2",
        protein = PR_TERMS[:MRP2],
        location = :apical,
        cell_type = CL_TERMS[:enterocyte],
        go_function = GO_TERMS[:ABC_transporter],
        maturation = SigmoidalMaturation(
            335.0,    # TM50 = 11 months
            1.52,     # gamma
            1.0,
            0.15,     # 15% at birth
            :days
        ),
        adult_expression = 1.2,
        cv_adult = 0.45,
        regional_gradient = (duodenum = 1.0, jejunum = 0.9, ileum = 0.8, colon = 0.4),
        substrates = ["MTX", "irinotecan_SN38G", "vinblastine"],
        clinical_impact = "Intestinal secretion of conjugates",
        references = ["van Groen 2021"]
    ),

    :MRP3_intestinal => (
        name = "Multidrug Resistance Protein 3 (intestinal)",
        gene = "ABCC3",
        protein = PR_TERMS[:MRP3],
        location = :basolateral,
        cell_type = CL_TERMS[:enterocyte],
        go_function = GO_TERMS[:efflux_pump],
        maturation = SigmoidalMaturation(
            244.0,    # TM50 = 8 months
            1.65,     # gamma
            1.0,
            0.22,     # 22% at birth
            :days
        ),
        adult_expression = 0.8,
        cv_adult = 0.40,
        regional_gradient = (duodenum = 0.7, jejunum = 0.8, ileum = 1.0, colon = 1.2),
        substrates = ["morphine_glucuronides", "etoposide_glucuronide", "fexofenadine"],
        clinical_impact = "Basolateral efflux → systemic circulation",
        references = ["van Groen 2021"]
    )
)

# =============================================================================
# BBB TRANSPORTER ONTOGENY PROFILES
# =============================================================================

const BBB_TRANSPORTER_ONTOGENY = Dict{Symbol, NamedTuple}(
    :P_gp_BBB => (
        name = "P-glycoprotein (BBB)",
        gene = "ABCB1",
        protein = PR_TERMS[:P_gp],
        location = :luminal,
        cell_type = CL_TERMS[:brain_endothelial],
        go_function = GO_TERMS[:ABC_transporter],
        maturation = BiphasicMaturation(
            61.0,     # Early TM50 = 2 months
            2.0,      # gamma early
            0.6,      # weight early
            730.0,    # Late TM50 = 2 years
            1.3,      # gamma late
            0.4,      # weight late
            :days
        ),
        adult_expression = 4.5,  # pmol/mg protein
        cv_adult = 0.45,
        substrates = ["loperamide", "cyclosporine", "quinidine", "verapamil", "dexamethasone"],
        clinical_impact = "CNS drug exposure higher in neonates; opioid sensitivity",
        references = ["Daood 2008", "Lam 2015"]
    ),

    :BCRP_BBB => (
        name = "Breast Cancer Resistance Protein (BBB)",
        gene = "ABCG2",
        protein = PR_TERMS[:BCRP],
        location = :luminal,
        cell_type = CL_TERMS[:brain_endothelial],
        go_function = GO_TERMS[:ABC_transporter],
        maturation = SigmoidalMaturation(
            365.0,    # TM50 = 1 year
            1.55,     # gamma
            1.0,
            0.20,     # 20% at birth
            :days
        ),
        adult_expression = 8.0,  # Higher than P-gp at BBB
        cv_adult = 0.50,
        substrates = ["sulfasalazine", "rosuvastatin", "topotecan", "mitoxantrone", "porphyrins"],
        clinical_impact = "Protoporphyrin IX accumulation in immature BBB",
        references = ["Ek 2012", "Strazielle 2015"]
    ),

    :LAT1 => (
        name = "L-type Amino Acid Transporter 1",
        gene = "SLC7A5",
        protein = PR_TERMS[:LAT1],
        location = :both_membranes,
        cell_type = CL_TERMS[:brain_endothelial],
        go_function = GO_TERMS[:uptake_transporter],
        maturation = SigmoidalMaturation(
            61.0,     # TM50 = 2 months (early - nutritional)
            2.50,     # gamma (steep)
            1.0,
            0.55,     # 55% at birth - high for brain development
            :days
        ),
        adult_expression = 3.5,
        cv_adult = 0.30,
        substrates = ["L-DOPA", "gabapentin", "melphalan", "phenylalanine", "tyrosine", "tryptophan"],
        clinical_impact = "Amino acid supply for brain growth; L-DOPA transport",
        references = ["Boado 1999", "Gynther 2008"]
    ),

    :GLUT1 => (
        name = "Glucose Transporter 1",
        gene = "SLC2A1",
        protein = PR_TERMS[:GLUT1],
        location = :both_membranes,
        cell_type = CL_TERMS[:brain_endothelial],
        go_function = GO_TERMS[:uptake_transporter],
        maturation = SigmoidalMaturation(
            45.0,     # TM50 = 1.5 months (very early)
            3.0,      # gamma (very steep)
            1.0,
            0.70,     # 70% at birth - critical for brain metabolism
            :days
        ),
        adult_expression = 45.0,  # Very high expression
        cv_adult = 0.25,
        substrates = ["glucose", "dehydroascorbic_acid", "mannose"],
        clinical_impact = "Essential for brain glucose supply; GLUT1 deficiency syndrome",
        references = ["Simpson 2007", "Pardridge 1990"]
    ),

    :MCT1 => (
        name = "Monocarboxylate Transporter 1",
        gene = "SLC16A1",
        protein = PR_TERMS[:MCT1],
        location = :both_membranes,
        cell_type = CL_TERMS[:brain_endothelial],
        go_function = GO_TERMS[:uptake_transporter],
        maturation = SigmoidalMaturation(
            30.0,     # TM50 = 1 month (extremely early)
            2.80,     # gamma (steep)
            1.2,      # Higher in neonates (ketone bodies!)
            0.80,     # 80% at birth
            :days
        ),
        adult_expression = 2.5,
        cv_adult = 0.35,
        substrates = ["lactate", "pyruvate", "ketone_bodies", "valproic_acid", "salicylate"],
        clinical_impact = "Ketone body transport critical for neonatal brain; valproate entry",
        references = ["Leino 1999", "Pierre 2007"]
    ),

    :OAT3_BBB => (
        name = "Organic Anion Transporter 3 (choroid plexus)",
        gene = "SLC22A8",
        protein = PR_TERMS[:OAT3],
        location = :abluminal,
        cell_type = CL_TERMS[:brain_endothelial],
        go_function = GO_TERMS[:organic_anion_transport],
        maturation = SigmoidalMaturation(
            548.0,    # TM50 = 1.5 years (slow)
            1.40,     # gamma
            1.0,
            0.10,     # 10% at birth
            :days
        ),
        adult_expression = 0.5,
        cv_adult = 0.45,
        substrates = ["penicillin", "cephalosporins", "methotrexate", "6-mercaptopurine"],
        clinical_impact = "CSF antibiotic clearance; reduced in neonates → higher CSF levels",
        references = ["Strazielle 2015", "Ek 2012"]
    )
)

# =============================================================================
# MAIN API FUNCTIONS
# =============================================================================

"""
    get_transporter_ontogeny(transporter::Symbol, age::PediatricAge; organ::Symbol=:auto) -> Float64

Get the ontogeny scaling factor for a transporter at a given age.

# Arguments
- `transporter`: Transporter symbol (e.g., :OAT1, :OATP1B1, :P_gp)
- `age`: PediatricAge struct
- `organ`: Organ context (:renal, :hepatic, :intestinal, :bbb, :auto)

# Returns
- Fraction of adult activity (0.0 - 1.0, or >1.0 for MCT1)

# Example
```julia
age = PediatricAge(months=6)
factor = get_transporter_ontogeny(:OAT1, age)  # ~0.5 at 6 months
```
"""
function get_transporter_ontogeny(transporter::Symbol, age::PediatricAge; organ::Symbol=:auto)::Float64
    # Find transporter in databases
    profile = nothing

    if organ == :auto || organ == :renal
        if haskey(RENAL_TRANSPORTER_ONTOGENY, transporter)
            profile = RENAL_TRANSPORTER_ONTOGENY[transporter]
        end
    end

    if profile === nothing && (organ == :auto || organ == :hepatic)
        if haskey(HEPATIC_TRANSPORTER_ONTOGENY, transporter)
            profile = HEPATIC_TRANSPORTER_ONTOGENY[transporter]
        end
    end

    if profile === nothing && (organ == :auto || organ == :intestinal)
        if haskey(INTESTINAL_TRANSPORTER_ONTOGENY, transporter)
            profile = INTESTINAL_TRANSPORTER_ONTOGENY[transporter]
        end
    end

    if profile === nothing && (organ == :auto || organ == :bbb)
        if haskey(BBB_TRANSPORTER_ONTOGENY, transporter)
            profile = BBB_TRANSPORTER_ONTOGENY[transporter]
        end
    end

    if profile === nothing
        @warn "Transporter $transporter not found in ontogeny database"
        return 1.0  # Default to adult
    end

    return calculate_maturation(profile.maturation, age)
end

"""
    get_renal_transporter_ontogeny(age::PediatricAge) -> Dict{Symbol, Float64}

Get ontogeny factors for all renal transporters at a given age.
"""
function get_renal_transporter_ontogeny(age::PediatricAge)::Dict{Symbol, Float64}
    result = Dict{Symbol, Float64}()
    for (name, profile) in RENAL_TRANSPORTER_ONTOGENY
        result[name] = calculate_maturation(profile.maturation, age)
    end
    return result
end

"""
    get_hepatic_transporter_ontogeny(age::PediatricAge) -> Dict{Symbol, Float64}

Get ontogeny factors for all hepatic transporters at a given age.
"""
function get_hepatic_transporter_ontogeny(age::PediatricAge)::Dict{Symbol, Float64}
    result = Dict{Symbol, Float64}()
    for (name, profile) in HEPATIC_TRANSPORTER_ONTOGENY
        result[name] = calculate_maturation(profile.maturation, age)
    end
    return result
end

"""
    get_intestinal_transporter_ontogeny(age::PediatricAge) -> Dict{Symbol, Float64}

Get ontogeny factors for all intestinal transporters at a given age.
"""
function get_intestinal_transporter_ontogeny(age::PediatricAge)::Dict{Symbol, Float64}
    result = Dict{Symbol, Float64}()
    for (name, profile) in INTESTINAL_TRANSPORTER_ONTOGENY
        result[name] = calculate_maturation(profile.maturation, age)
    end
    return result
end

"""
    get_bbb_transporter_ontogeny(age::PediatricAge) -> Dict{Symbol, Float64}

Get ontogeny factors for all BBB transporters at a given age.
"""
function get_bbb_transporter_ontogeny(age::PediatricAge)::Dict{Symbol, Float64}
    result = Dict{Symbol, Float64}()
    for (name, profile) in BBB_TRANSPORTER_ONTOGENY
        result[name] = calculate_maturation(profile.maturation, age)
    end
    return result
end

"""
    apply_ontogeny_scaling(adult_clearance::Float64, transporter::Symbol, age::PediatricAge;
                           organ::Symbol=:auto) -> Float64

Scale adult clearance value by ontogeny factor.

# Example
```julia
adult_CL = 50.0  # mL/min
age = PediatricAge(days=30)
pediatric_CL = apply_ontogeny_scaling(adult_CL, :OAT1, age)
```
"""
function apply_ontogeny_scaling(adult_clearance::Float64, transporter::Symbol, age::PediatricAge;
                                organ::Symbol=:auto)::Float64
    factor = get_transporter_ontogeny(transporter, age; organ=organ)
    return adult_clearance * factor
end

"""
    get_pediatric_disease_modifiers(doid::String) -> NamedTuple

Get transporter modification factors for a pediatric disease condition.
"""
function get_pediatric_disease_modifiers(doid::String)
    if haskey(DOID_PEDIATRIC_CONDITIONS, doid)
        return DOID_PEDIATRIC_CONDITIONS[doid]
    end
    return (name = "unknown", transporter_effects = Symbol[], severity = :unknown)
end

"""
    get_icd_codes_for_condition(condition::Symbol) -> NamedTuple

Get ICD-10 and ICD-11 codes for a pediatric condition.
"""
function get_icd_codes_for_condition(condition::Symbol)
    if haskey(ICD_CLASSIFICATIONS, condition)
        return ICD_CLASSIFICATIONS[condition]
    end
    return (icd10 = String[], icd11 = String[], doid = "", description = "Unknown condition")
end

"""
    get_doid_pediatric_conditions() -> Vector{String}

List all DOID terms in the pediatric conditions database.
"""
function get_doid_pediatric_conditions()::Vector{String}
    return collect(keys(DOID_PEDIATRIC_CONDITIONS))
end

# =============================================================================
# VALIDATION AND CLINICAL TOOLS
# =============================================================================

"""
    predict_pediatric_clearance(;
        drug::String,
        adult_clearance::Float64,
        age::PediatricAge,
        transporters::Vector{Symbol},
        fraction_each::Vector{Float64},
        disease::Union{Symbol, Nothing}=nothing
    ) -> NamedTuple

Predict pediatric clearance from adult value using transporter ontogeny.

# Arguments
- `drug`: Drug name for documentation
- `adult_clearance`: Adult renal/hepatic clearance (mL/min or L/h)
- `age`: Pediatric age
- `transporters`: Vector of transporters involved
- `fraction_each`: Fraction of clearance via each transporter (must sum to 1)
- `disease`: Optional disease state affecting transporters

# Returns
NamedTuple with:
- `pediatric_clearance`: Predicted clearance
- `ontogeny_factors`: Individual transporter factors
- `overall_factor`: Weighted average factor
- `confidence`: Qualitative confidence assessment
"""
function predict_pediatric_clearance(;
    drug::String,
    adult_clearance::Float64,
    age::PediatricAge,
    transporters::Vector{Symbol},
    fraction_each::Vector{Float64},
    disease::Union{Symbol, Nothing}=nothing
)
    @assert length(transporters) == length(fraction_each) "Transporters and fractions must match"
    @assert abs(sum(fraction_each) - 1.0) < 0.01 "Fractions must sum to 1"

    # Get ontogeny factors
    ontogeny_factors = Dict{Symbol, Float64}()
    for t in transporters
        ontogeny_factors[t] = get_transporter_ontogeny(t, age)
    end

    # Calculate weighted average
    overall_factor = sum(fraction_each[i] * ontogeny_factors[transporters[i]]
                        for i in 1:length(transporters))

    # Apply disease modifier if present
    disease_factor = 1.0
    if disease !== nothing && haskey(ICD_CLASSIFICATIONS, disease)
        doid = ICD_CLASSIFICATIONS[disease].doid
        if haskey(DOID_PEDIATRIC_CONDITIONS, doid)
            cond = DOID_PEDIATRIC_CONDITIONS[doid]
            # Apply severity-based reduction
            disease_factor = cond.severity == :major ? 0.6 :
                            cond.severity == :moderate ? 0.8 : 1.0
        end
    end

    pediatric_clearance = adult_clearance * overall_factor * disease_factor

    # Confidence assessment
    confidence = if age.category == PRETERM_NEONATE
        :low  # Limited data for premature
    elseif age.category == TERM_NEONATE
        :moderate
    elseif age.category in [INFANT, TODDLER]
        :good
    else
        :high
    end

    return (
        drug = drug,
        age_category = age.category,
        postnatal_days = age.postnatal_days,
        adult_clearance = adult_clearance,
        pediatric_clearance = pediatric_clearance,
        overall_factor = overall_factor,
        disease_factor = disease_factor,
        ontogeny_factors = ontogeny_factors,
        confidence = confidence
    )
end

"""
    generate_ontogeny_curve(transporter::Symbol;
                           age_range_days::Tuple{Float64,Float64}=(0.0, 6570.0),
                           n_points::Int=100) -> NamedTuple

Generate ontogeny curve data for plotting.
"""
function generate_ontogeny_curve(transporter::Symbol;
                                age_range_days::Tuple{Float64,Float64}=(0.0, 6570.0),
                                n_points::Int=100)
    ages_days = range(age_range_days[1], age_range_days[2], length=n_points)
    ages_years = ages_days ./ 365.25
    factors = [get_transporter_ontogeny(transporter, PediatricAge(d)) for d in ages_days]

    # Find TM50
    idx_50 = findfirst(f -> f >= 0.5, factors)
    TM50_days = idx_50 !== nothing ? ages_days[idx_50] : NaN

    return (
        transporter = transporter,
        ages_days = collect(ages_days),
        ages_years = collect(ages_years),
        factors = factors,
        TM50_days = TM50_days,
        TM50_months = TM50_days / 30.44
    )
end

"""
    validate_ontogeny_prediction(predicted::Float64, observed::Float64) -> NamedTuple

Calculate prediction accuracy metrics.
"""
function validate_ontogeny_prediction(predicted::Float64, observed::Float64)
    ratio = predicted / observed
    fold_error = ratio >= 1.0 ? ratio : 1.0 / ratio
    within_2fold = fold_error <= 2.0

    return (
        predicted = predicted,
        observed = observed,
        ratio = ratio,
        fold_error = fold_error,
        within_2fold = within_2fold,
        percent_error = (predicted - observed) / observed * 100
    )
end

# =============================================================================
# SEMANTIC WEB EXPORT
# =============================================================================

"""
    export_transporter_to_jsonld(transporter::Symbol; organ::Symbol=:auto) -> String

Export transporter ontogeny data as JSON-LD with OBO Foundry annotations.
"""
function export_transporter_to_jsonld(transporter::Symbol; organ::Symbol=:auto)::String
    profile = nothing
    organ_name = ""

    if haskey(RENAL_TRANSPORTER_ONTOGENY, transporter)
        profile = RENAL_TRANSPORTER_ONTOGENY[transporter]
        organ_name = "renal"
    elseif haskey(HEPATIC_TRANSPORTER_ONTOGENY, transporter)
        profile = HEPATIC_TRANSPORTER_ONTOGENY[transporter]
        organ_name = "hepatic"
    elseif haskey(INTESTINAL_TRANSPORTER_ONTOGENY, transporter)
        profile = INTESTINAL_TRANSPORTER_ONTOGENY[transporter]
        organ_name = "intestinal"
    elseif haskey(BBB_TRANSPORTER_ONTOGENY, transporter)
        profile = BBB_TRANSPORTER_ONTOGENY[transporter]
        organ_name = "bbb"
    end

    if profile === nothing
        return "{\"error\": \"Transporter not found\"}"
    end

    # Build JSON-LD
    jsonld = """
{
  "@context": {
    "obo": "http://purl.obolibrary.org/obo/",
    "PR": "http://purl.obolibrary.org/obo/PR_",
    "GO": "http://purl.obolibrary.org/obo/GO_",
    "UBERON": "http://purl.obolibrary.org/obo/UBERON_",
    "CL": "http://purl.obolibrary.org/obo/CL_",
    "HGNC": "http://identifiers.org/hgnc/",
    "UniProt": "http://identifiers.org/uniprot/",
    "darwin": "http://darwin-pbpk.org/ontogeny/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#"
  },
  "@type": "darwin:TransporterOntogenyProfile",
  "@id": "darwin:ontogeny/$(transporter)",
  "rdfs:label": "$(profile.name)",
  "darwin:gene": "$(profile.gene)",
  "darwin:organ": "$(organ_name)",
  "darwin:TM50_days": $(profile.maturation isa SigmoidalMaturation ? profile.maturation.TM50 : "null"),
  "darwin:adultExpression": $(profile.adult_expression),
  "darwin:coefficientOfVariation": $(profile.cv_adult),
  "darwin:proteinOntology": "$(profile.protein.id)",
  "darwin:uniprotId": "$(profile.protein.uniprot)",
  "darwin:hgncGene": "$(profile.protein.hgnc)",
  "darwin:cellType": "$(profile.cell_type.id)",
  "darwin:goFunction": "$(profile.go_function.id)",
  "darwin:substrates": $(profile.substrates),
  "darwin:clinicalImpact": "$(profile.clinical_impact)",
  "darwin:references": $(profile.references)
}
"""
    return jsonld
end

end # module TransporterOntogeny
