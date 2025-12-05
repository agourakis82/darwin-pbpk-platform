# =============================================================================
# Schema.org Full Medical Vocabulary - Darwin PBPK Semantic Layer
# =============================================================================
# Complete Schema.org integration for pharmacokinetics and drug discovery.
#
# Schema.org Types Covered:
# - Drug (with all PK-relevant properties)
# - MedicalStudy / MedicalTrial
# - DrugStrength / DoseSchedule
# - MedicalCondition / MedicalIndication
# - PropertyValue (for structured measurements)
# - MedicalCode (for ontology cross-references)
#
# Benefits of Full Schema.org Integration:
# 1. Web search engine visibility (Google, Bing knowledge panels)
# 2. Interoperability with health portals (WebMD, Drugs.com)
# 3. Compatibility with BioSchemas extensions
# 4. Standard vocabulary for LLM/RAG systems
#
# Reference: https://schema.org/Drug
#
# Author: Dr. Demetrios Agourakis
# Date: December 2025
# =============================================================================

module SchemaOrg

export SCHEMA_ORG_CONTEXT, SCHEMA_ORG_DRUG_CONTEXT, SCHEMA_ORG_STUDY_CONTEXT
export SchemaOrgDrug, SchemaOrgDrugStrength, SchemaOrgDoseSchedule
export SchemaOrgMedicalStudy, SchemaOrgPropertyValue, SchemaOrgMedicalCode
export to_schema_org_drug, to_schema_org_study, to_schema_org_property_value
export map_pk_parameter_to_schema, map_ddi_to_schema
export DrugPrescriptionStatus, DrugPregnancyCategory, MedicalStudyStatus

using JSON
using Dates

# =============================================================================
# Schema.org Enumerations
# =============================================================================

"""Schema.org DrugPrescriptionStatus enumeration"""
@enum DrugPrescriptionStatus begin
    OTC                    # Over the counter
    PrescriptionOnly       # Requires prescription
end

"""Schema.org DrugPregnancyCategory enumeration (FDA categories)"""
@enum DrugPregnancyCategory begin
    FDAcategoryA
    FDAcategoryB
    FDAcategoryC
    FDAcategoryD
    FDAcategoryX
    FDAnotEvaluated
end

"""Schema.org MedicalStudyStatus enumeration"""
@enum MedicalStudyStatus begin
    ActiveNotRecruiting
    Completed
    EnrollingByInvitation
    NotYetRecruiting
    Recruiting
    ResultsAvailable
    ResultsNotAvailable
    Suspended
    Terminated
    Withdrawn
end

# =============================================================================
# Schema.org Contexts
# =============================================================================

"""
Full Schema.org context for medical/pharmaceutical entities.
Uses @vocab for clean property names.
"""
const SCHEMA_ORG_CONTEXT = Dict{String, Any}(
    "@context" => Dict{String, Any}(
        "@version" => 1.1,
        "@vocab" => "https://schema.org/",

        # Darwin extensions under custom namespace
        "darwin" => "https://darwin-pbpk.org/schema/",

        # QUDT for precise units
        "qudt" => "http://qudt.org/schema/qudt/",
        "unit" => "http://qudt.org/vocab/unit/",

        # OBO for biomedical ontologies
        "obo" => "http://purl.obolibrary.org/obo/",

        # External identifiers
        "rxcui" => Dict("@id" => "https://schema.org/rxcui"),
        "drugBankId" => Dict("@id" => "darwin:drugBankId"),
        "chebiId" => Dict("@id" => "darwin:chebiId"),
        "pubchemCid" => Dict("@id" => "darwin:pubchemCid"),
        "uniprotId" => Dict("@id" => "darwin:uniprotId"),

        # PK-specific extensions
        "pharmacokinetics" => Dict("@id" => "darwin:pharmacokinetics", "@type" => "@id"),
        "clearance" => Dict("@id" => "darwin:clearance"),
        "volumeOfDistribution" => Dict("@id" => "darwin:volumeOfDistribution"),
        "halfLife" => Dict("@id" => "darwin:halfLife"),
        "bioavailability" => Dict("@id" => "darwin:bioavailability"),
        "proteinBinding" => Dict("@id" => "darwin:proteinBinding"),
        "metabolicPathway" => Dict("@id" => "darwin:metabolicPathway", "@type" => "@id"),

        # DDI extensions
        "interactionMechanism" => Dict("@id" => "darwin:interactionMechanism"),
        "aucFoldChange" => Dict("@id" => "darwin:aucFoldChange"),
        "cmaxFoldChange" => Dict("@id" => "darwin:cmaxFoldChange"),
        "clinicalSignificance" => Dict("@id" => "darwin:clinicalSignificance"),
    )
)

"""
Schema.org Drug-specific context with all drug properties.
"""
const SCHEMA_ORG_DRUG_CONTEXT = Dict{String, Any}(
    "@context" => merge(
        SCHEMA_ORG_CONTEXT["@context"],
        Dict{String, Any}(
            # Drug-specific Schema.org types
            "Drug" => "https://schema.org/Drug",
            "DrugClass" => "https://schema.org/DrugClass",
            "DrugStrength" => "https://schema.org/DrugStrength",
            "DoseSchedule" => "https://schema.org/DoseSchedule",
            "MaximumDoseSchedule" => "https://schema.org/MaximumDoseSchedule",
            "DrugCost" => "https://schema.org/DrugCost",
            "DrugLegalStatus" => "https://schema.org/DrugLegalStatus",

            # MedicalEntity types
            "MedicalCode" => "https://schema.org/MedicalCode",
            "MedicalCondition" => "https://schema.org/MedicalCondition",
            "MedicalIndication" => "https://schema.org/MedicalIndication",
            "MedicalContraindication" => "https://schema.org/MedicalContraindication",

            # Substance
            "Substance" => "https://schema.org/Substance",
        )
    )
)

"""
Schema.org MedicalStudy context for clinical trials.
"""
const SCHEMA_ORG_STUDY_CONTEXT = Dict{String, Any}(
    "@context" => merge(
        SCHEMA_ORG_CONTEXT["@context"],
        Dict{String, Any}(
            "MedicalStudy" => "https://schema.org/MedicalStudy",
            "MedicalTrial" => "https://schema.org/MedicalTrial",
            "MedicalObservationalStudy" => "https://schema.org/MedicalObservationalStudy",

            # Study properties
            "healthCondition" => Dict("@id" => "https://schema.org/healthCondition", "@type" => "@id"),
            "studySubject" => Dict("@id" => "https://schema.org/studySubject", "@type" => "@id"),
            "studyLocation" => Dict("@id" => "https://schema.org/studyLocation", "@type" => "@id"),
            "sponsor" => Dict("@id" => "https://schema.org/sponsor", "@type" => "@id"),
            "status" => Dict("@id" => "https://schema.org/status"),

            # Trial-specific
            "trialDesign" => Dict("@id" => "https://schema.org/trialDesign"),
            "phase" => Dict("@id" => "https://schema.org/phase"),
        )
    )
)

# =============================================================================
# Schema.org Data Structures
# =============================================================================

"""
    SchemaOrgMedicalCode

Represents a medical code from a controlled vocabulary.
Maps to schema:MedicalCode.
"""
struct SchemaOrgMedicalCode
    coding_system::String    # e.g., "RxNorm", "ICD-10", "SNOMED-CT", "ChEBI"
    code_value::String       # The actual code
    name::Union{String, Nothing}  # Human-readable name
    url::Union{String, Nothing}   # URL to the code definition
end

function SchemaOrgMedicalCode(coding_system::String, code_value::String)
    SchemaOrgMedicalCode(coding_system, code_value, nothing, nothing)
end

"""
    SchemaOrgPropertyValue

Represents a structured value with optional unit.
Maps to schema:PropertyValue with QUDT unit integration.
"""
struct SchemaOrgPropertyValue
    name::String
    value::Any              # Number, Text, Boolean, or StructuredValue
    unit_code::Union{String, Nothing}     # UN/CEFACT code or QUDT URI
    unit_text::Union{String, Nothing}     # Human-readable unit
    property_id::Union{String, Nothing}   # Standard property identifier
    min_value::Union{Float64, Nothing}
    max_value::Union{Float64, Nothing}
    measurement_method::Union{String, Nothing}
end

function SchemaOrgPropertyValue(name::String, value::Any;
    unit_code::Union{String,Nothing}=nothing,
    unit_text::Union{String,Nothing}=nothing,
    property_id::Union{String,Nothing}=nothing)
    SchemaOrgPropertyValue(name, value, unit_code, unit_text, property_id, nothing, nothing, nothing)
end

"""
    SchemaOrgDrugStrength

Represents available drug dosage strength.
Maps to schema:DrugStrength.
"""
struct SchemaOrgDrugStrength
    strength_value::Float64
    strength_unit::String
    active_ingredient::Union{String, Nothing}
    maximum_intake::Union{SchemaOrgPropertyValue, Nothing}
end

function SchemaOrgDrugStrength(value::Float64, unit::String)
    SchemaOrgDrugStrength(value, unit, nothing, nothing)
end

"""
    SchemaOrgDoseSchedule

Represents a dosing schedule.
Maps to schema:DoseSchedule.
"""
struct SchemaOrgDoseSchedule
    dose_value::Float64
    dose_unit::String
    frequency::String           # e.g., "QD", "BID", "TID"
    target_population::Union{String, Nothing}
end

"""
    SchemaOrgDrug

Complete Schema.org Drug representation with all PK-relevant properties.
"""
struct SchemaOrgDrug
    # Identifiers (Thing)
    name::String
    identifier::Union{String, Nothing}
    url::Union{String, Nothing}
    same_as::Vector{String}           # Links to other representations

    # Drug identifiers
    rxcui::Union{String, Nothing}     # RxNorm CUI
    non_proprietary_name::Union{String, Nothing}  # Generic name
    proprietary_name::Union{String, Nothing}      # Brand name

    # Cross-references (as MedicalCode)
    codes::Vector{SchemaOrgMedicalCode}

    # Drug properties
    active_ingredient::Union{String, Nothing}
    drug_class::Union{String, Nothing}
    dosage_form::Union{String, Nothing}
    administration_route::Union{String, Nothing}

    # Available strengths and dosing
    available_strength::Vector{SchemaOrgDrugStrength}
    dose_schedule::Vector{SchemaOrgDoseSchedule}
    maximum_intake::Union{SchemaOrgPropertyValue, Nothing}

    # Pharmacology (Schema.org clinicalPharmacology + Darwin extensions)
    clinical_pharmacology::Union{String, Nothing}
    mechanism_of_action::Union{String, Nothing}
    pharmacokinetics::Dict{String, SchemaOrgPropertyValue}  # Darwin extension

    # Interactions
    interacting_drug::Vector{String}  # Names or IRIs of interacting drugs

    # Warnings and status
    warning::Union{String, Nothing}
    alcohol_warning::Union{String, Nothing}
    food_warning::Union{String, Nothing}
    pregnancy_warning::Union{String, Nothing}
    pregnancy_category::Union{DrugPregnancyCategory, Nothing}
    breastfeeding_warning::Union{String, Nothing}
    overdosage::Union{String, Nothing}

    # Legal/regulatory
    prescription_status::Union{DrugPrescriptionStatus, Nothing}
    is_available_generically::Union{Bool, Nothing}
    legal_status::Union{String, Nothing}

    # References
    label_details::Union{String, Nothing}
    prescribing_info::Union{String, Nothing}
end

"""
    SchemaOrgMedicalStudy

Schema.org MedicalStudy representation for PK studies.
"""
struct SchemaOrgMedicalStudy
    name::String
    identifier::Union{String, Nothing}
    description::Union{String, Nothing}
    url::Union{String, Nothing}

    # Study specifics
    study_type::Symbol   # :trial, :observational
    status::Union{MedicalStudyStatus, Nothing}
    phase::Union{String, Nothing}  # For trials: "Phase 1", "Phase 2", etc.

    # Subjects
    health_condition::Vector{String}   # Conditions being studied
    study_subject::Vector{String}      # Drugs/therapies being studied

    # Location and sponsor
    study_location::Vector{String}
    sponsor::Union{String, Nothing}

    # Codes
    codes::Vector{SchemaOrgMedicalCode}  # ClinicalTrials.gov NCT, etc.
end

# =============================================================================
# JSON-LD Serialization Functions
# =============================================================================

"""
    to_schema_org_drug(drug::SchemaOrgDrug; include_context::Bool=true) -> Dict{String, Any}

Convert SchemaOrgDrug to full Schema.org JSON-LD.
"""
function to_schema_org_drug(drug::SchemaOrgDrug; include_context::Bool=true)::Dict{String, Any}
    result = Dict{String, Any}()

    if include_context
        result["@context"] = SCHEMA_ORG_DRUG_CONTEXT["@context"]
    end

    result["@type"] = "Drug"

    # Basic identity
    result["name"] = drug.name

    if drug.identifier !== nothing
        result["identifier"] = drug.identifier
        result["@id"] = drug.identifier
    end

    if drug.url !== nothing
        result["url"] = drug.url
    end

    if !isempty(drug.same_as)
        result["sameAs"] = drug.same_as
    end

    # RxNorm identifier (direct Schema.org property)
    if drug.rxcui !== nothing
        result["rxcui"] = drug.rxcui
    end

    # Names
    if drug.non_proprietary_name !== nothing
        result["nonProprietaryName"] = drug.non_proprietary_name
    end

    if drug.proprietary_name !== nothing
        result["proprietaryName"] = drug.proprietary_name
    end

    # Medical codes
    if !isempty(drug.codes)
        result["code"] = [to_jsonld(code) for code in drug.codes]
    end

    # Drug properties
    if drug.active_ingredient !== nothing
        result["activeIngredient"] = drug.active_ingredient
    end

    if drug.drug_class !== nothing
        result["drugClass"] = Dict{String, Any}(
            "@type" => "DrugClass",
            "name" => drug.drug_class
        )
    end

    if drug.dosage_form !== nothing
        result["dosageForm"] = drug.dosage_form
    end

    if drug.administration_route !== nothing
        result["administrationRoute"] = drug.administration_route
    end

    # Strengths
    if !isempty(drug.available_strength)
        result["availableStrength"] = [to_jsonld(s) for s in drug.available_strength]
    end

    # Dose schedules
    if !isempty(drug.dose_schedule)
        result["doseSchedule"] = [to_jsonld(ds) for ds in drug.dose_schedule]
    end

    # Maximum intake
    if drug.maximum_intake !== nothing
        result["maximumIntake"] = Dict{String, Any}(
            "@type" => "MaximumDoseSchedule",
            "doseValue" => to_jsonld(drug.maximum_intake)
        )
    end

    # Pharmacology - Schema.org text field
    if drug.clinical_pharmacology !== nothing
        result["clinicalPharmacology"] = drug.clinical_pharmacology
    end

    if drug.mechanism_of_action !== nothing
        result["mechanismOfAction"] = drug.mechanism_of_action
    end

    # PK Parameters - Darwin extension with QUDT
    if !isempty(drug.pharmacokinetics)
        pk_data = Dict{String, Any}(
            "@type" => "darwin:PharmacokineticProfile"
        )
        for (param_name, param_value) in drug.pharmacokinetics
            pk_data[param_name] = to_jsonld(param_value)
        end
        result["darwin:pharmacokinetics"] = pk_data
    end

    # Interactions
    if !isempty(drug.interacting_drug)
        result["interactingDrug"] = [
            Dict{String, Any}("@type" => "Drug", "name" => d)
            for d in drug.interacting_drug
        ]
    end

    # Warnings
    if drug.warning !== nothing
        result["warning"] = drug.warning
    end
    if drug.alcohol_warning !== nothing
        result["alcoholWarning"] = drug.alcohol_warning
    end
    if drug.food_warning !== nothing
        result["foodWarning"] = drug.food_warning
    end
    if drug.pregnancy_warning !== nothing
        result["pregnancyWarning"] = drug.pregnancy_warning
    end
    if drug.pregnancy_category !== nothing
        result["pregnancyCategory"] = string(drug.pregnancy_category)
    end
    if drug.breastfeeding_warning !== nothing
        result["breastfeedingWarning"] = drug.breastfeeding_warning
    end
    if drug.overdosage !== nothing
        result["overdosage"] = drug.overdosage
    end

    # Legal/regulatory
    if drug.prescription_status !== nothing
        result["prescriptionStatus"] = drug.prescription_status == OTC ?
            "https://schema.org/OTC" : "https://schema.org/PrescriptionOnly"
    end

    if drug.is_available_generically !== nothing
        result["isAvailableGenerically"] = drug.is_available_generically
    end

    if drug.legal_status !== nothing
        result["legalStatus"] = drug.legal_status
    end

    # References
    if drug.label_details !== nothing
        result["labelDetails"] = drug.label_details
    end
    if drug.prescribing_info !== nothing
        result["prescribingInfo"] = drug.prescribing_info
    end

    return result
end

"""
    to_jsonld(code::SchemaOrgMedicalCode) -> Dict{String, Any}

Convert MedicalCode to JSON-LD.
"""
function to_jsonld(code::SchemaOrgMedicalCode)::Dict{String, Any}
    result = Dict{String, Any}(
        "@type" => "MedicalCode",
        "codingSystem" => code.coding_system,
        "codeValue" => code.code_value
    )

    if code.name !== nothing
        result["name"] = code.name
    end
    if code.url !== nothing
        result["url"] = code.url
    end

    return result
end

"""
    to_jsonld(pv::SchemaOrgPropertyValue) -> Dict{String, Any}

Convert PropertyValue to JSON-LD with QUDT integration.
"""
function to_jsonld(pv::SchemaOrgPropertyValue)::Dict{String, Any}
    result = Dict{String, Any}(
        "@type" => "PropertyValue",
        "name" => pv.name,
        "value" => pv.value
    )

    if pv.unit_code !== nothing
        result["unitCode"] = pv.unit_code
        # Also add QUDT reference if it looks like a QUDT unit
        if startswith(pv.unit_code, "unit:")
            result["qudt:unit"] = Dict("@id" => pv.unit_code)
        end
    end

    if pv.unit_text !== nothing
        result["unitText"] = pv.unit_text
    end

    if pv.property_id !== nothing
        result["propertyID"] = pv.property_id
    end

    if pv.min_value !== nothing
        result["minValue"] = pv.min_value
    end

    if pv.max_value !== nothing
        result["maxValue"] = pv.max_value
    end

    if pv.measurement_method !== nothing
        result["measurementMethod"] = pv.measurement_method
    end

    return result
end

"""
    to_jsonld(strength::SchemaOrgDrugStrength) -> Dict{String, Any}

Convert DrugStrength to JSON-LD.
"""
function to_jsonld(strength::SchemaOrgDrugStrength)::Dict{String, Any}
    result = Dict{String, Any}(
        "@type" => "DrugStrength",
        "strengthValue" => Dict{String, Any}(
            "@type" => "PropertyValue",
            "value" => strength.strength_value,
            "unitText" => strength.strength_unit
        )
    )

    if strength.active_ingredient !== nothing
        result["activeIngredient"] = strength.active_ingredient
    end

    if strength.maximum_intake !== nothing
        result["maximumIntake"] = to_jsonld(strength.maximum_intake)
    end

    return result
end

"""
    to_jsonld(ds::SchemaOrgDoseSchedule) -> Dict{String, Any}

Convert DoseSchedule to JSON-LD.
"""
function to_jsonld(ds::SchemaOrgDoseSchedule)::Dict{String, Any}
    result = Dict{String, Any}(
        "@type" => "DoseSchedule",
        "doseValue" => Dict{String, Any}(
            "@type" => "PropertyValue",
            "value" => ds.dose_value,
            "unitText" => ds.dose_unit
        ),
        "frequency" => ds.frequency
    )

    if ds.target_population !== nothing
        result["targetPopulation"] = ds.target_population
    end

    return result
end

"""
    to_schema_org_study(study::SchemaOrgMedicalStudy; include_context::Bool=true) -> Dict{String, Any}

Convert MedicalStudy to JSON-LD.
"""
function to_schema_org_study(study::SchemaOrgMedicalStudy; include_context::Bool=true)::Dict{String, Any}
    result = Dict{String, Any}()

    if include_context
        result["@context"] = SCHEMA_ORG_STUDY_CONTEXT["@context"]
    end

    result["@type"] = study.study_type == :trial ? "MedicalTrial" : "MedicalObservationalStudy"
    result["name"] = study.name

    if study.identifier !== nothing
        result["identifier"] = study.identifier
        result["@id"] = study.identifier
    end

    if study.description !== nothing
        result["description"] = study.description
    end

    if study.url !== nothing
        result["url"] = study.url
    end

    if study.status !== nothing
        result["status"] = string(study.status)
    end

    if study.phase !== nothing
        result["phase"] = study.phase
    end

    if !isempty(study.health_condition)
        result["healthCondition"] = [
            Dict("@type" => "MedicalCondition", "name" => c)
            for c in study.health_condition
        ]
    end

    if !isempty(study.study_subject)
        result["studySubject"] = study.study_subject
    end

    if !isempty(study.study_location)
        result["studyLocation"] = study.study_location
    end

    if study.sponsor !== nothing
        result["sponsor"] = Dict("@type" => "Organization", "name" => study.sponsor)
    end

    if !isempty(study.codes)
        result["code"] = [to_jsonld(code) for code in study.codes]
    end

    return result
end

# =============================================================================
# Mapping Functions
# =============================================================================

"""
    to_schema_org_property_value(name::String, value::Float64, unit::String; kwargs...) -> Dict{String, Any}

Create a Schema.org PropertyValue with QUDT unit.
"""
function to_schema_org_property_value(
    name::String,
    value::Float64,
    unit::String;
    property_id::Union{String, Nothing}=nothing,
    min_value::Union{Float64, Nothing}=nothing,
    max_value::Union{Float64, Nothing}=nothing
)::Dict{String, Any}

    # Map common PK units to QUDT
    qudt_units = Dict(
        "L/h" => "unit:L-PER-HR",
        "L" => "unit:L",
        "h" => "unit:HR",
        "mg" => "unit:MilliGM",
        "mg/L" => "unit:MilliGM-PER-L",
        "%" => "unit:PERCENT",
        "L/h/kg" => "unit:L-PER-HR-KiloGM",
    )

    pv = SchemaOrgPropertyValue(
        name,
        value,
        get(qudt_units, unit, nothing),
        unit,
        property_id,
        min_value,
        max_value,
        nothing
    )

    return to_jsonld(pv)
end

"""
    map_pk_parameter_to_schema(param_name::String, value::Float64, unit::String; cv::Union{Float64,Nothing}=nothing) -> Dict{String, Any}

Map a PK parameter to Schema.org PropertyValue with optional variability.
"""
function map_pk_parameter_to_schema(
    param_name::String,
    value::Float64,
    unit::String;
    cv::Union{Float64, Nothing}=nothing,
    source::Union{String, Nothing}=nothing
)::Dict{String, Any}

    # Standard PK parameter property IDs (linking to PATO/QUDT)
    property_ids = Dict(
        "clearance" => "darwin:Clearance",
        "cl" => "darwin:Clearance",
        "volume" => "darwin:VolumeOfDistribution",
        "vd" => "darwin:VolumeOfDistribution",
        "vss" => "darwin:VolumeOfDistributionSteadyState",
        "half-life" => "darwin:HalfLife",
        "t1/2" => "darwin:HalfLife",
        "bioavailability" => "darwin:Bioavailability",
        "f" => "darwin:Bioavailability",
        "fu" => "darwin:FractionUnbound",
        "ka" => "darwin:AbsorptionRateConstant",
        "ke" => "darwin:EliminationRateConstant",
        "cmax" => "darwin:MaximumConcentration",
        "tmax" => "darwin:TimeToMaxConcentration",
        "auc" => "darwin:AreaUnderCurve",
    )

    lower_name = lowercase(param_name)
    prop_id = get(property_ids, lower_name, "darwin:PKParameter")

    result = to_schema_org_property_value(param_name, value, unit; property_id=prop_id)

    # Add variability if provided
    if cv !== nothing
        result["darwin:coefficientOfVariation"] = cv
        result["darwin:interIndividualVariability"] = Dict{String, Any}(
            "@type" => "PropertyValue",
            "name" => "CV%",
            "value" => cv * 100,
            "unitText" => "%"
        )
    end

    # Add source reference
    if source !== nothing
        result["darwin:source"] = source
    end

    return result
end

"""
    map_ddi_to_schema(perpetrator::String, victim::String, auc_ratio::Float64; kwargs...) -> Dict{String, Any}

Map a DDI to Schema.org Drug with interactingDrug property.
"""
function map_ddi_to_schema(
    perpetrator::String,
    victim::String,
    auc_ratio::Float64;
    mechanism::Union{String, Nothing}=nothing,
    clinical_significance::Union{String, Nothing}=nothing,
    warning::Union{String, Nothing}=nothing
)::Dict{String, Any}

    result = Dict{String, Any}(
        "@context" => SCHEMA_ORG_DRUG_CONTEXT["@context"],
        "@type" => "Drug",
        "name" => victim,
        "interactingDrug" => [
            Dict{String, Any}(
                "@type" => "Drug",
                "name" => perpetrator,
                "darwin:aucFoldChange" => auc_ratio
            )
        ]
    )

    # Classify severity for warning
    severity = if auc_ratio >= 5.0
        "strong"
    elseif auc_ratio >= 2.0
        "moderate"
    elseif auc_ratio >= 1.25
        "weak"
    else
        "none"
    end

    if mechanism !== nothing
        result["interactingDrug"][1]["darwin:interactionMechanism"] = mechanism
    end

    if clinical_significance !== nothing
        result["darwin:clinicalSignificance"] = clinical_significance
    elseif severity != "none"
        result["darwin:clinicalSignificance"] = severity
    end

    if warning !== nothing
        result["warning"] = warning
    elseif severity in ["strong", "moderate"]
        result["warning"] = "Concomitant use with $perpetrator may result in $(severity) increase in $victim exposure (AUC ratio: $(round(auc_ratio, digits=1)))"
    end

    return result
end

"""
    create_drug_from_darwin(darwin_drug::Dict) -> SchemaOrgDrug

Create a SchemaOrgDrug from Darwin PBPK drug data.
"""
function create_drug_from_darwin(darwin_drug::Dict)::SchemaOrgDrug
    # Extract identifiers
    codes = SchemaOrgMedicalCode[]

    if haskey(darwin_drug, :chebi_id) || haskey(darwin_drug, "chebi_id")
        chebi = get(darwin_drug, :chebi_id, get(darwin_drug, "chebi_id", ""))
        push!(codes, SchemaOrgMedicalCode(
            "ChEBI",
            chebi,
            nothing,
            "http://purl.obolibrary.org/obo/$(replace(chebi, ":" => "_"))"
        ))
    end

    if haskey(darwin_drug, :drugbank_id) || haskey(darwin_drug, "drugbank_id")
        db_id = get(darwin_drug, :drugbank_id, get(darwin_drug, "drugbank_id", ""))
        push!(codes, SchemaOrgMedicalCode(
            "DrugBank",
            db_id,
            nothing,
            "https://go.drugbank.com/drugs/$db_id"
        ))
    end

    # Build sameAs links
    same_as = String[]
    for code in codes
        if code.url !== nothing
            push!(same_as, code.url)
        end
    end

    # Extract PK parameters
    pk_params = Dict{String, SchemaOrgPropertyValue}()

    pk_data = get(darwin_drug, :pharmacokinetics, get(darwin_drug, "pharmacokinetics", Dict()))
    for (param_name, param_data) in pk_data
        if param_data isa Dict
            value = get(param_data, :value, get(param_data, "value", 0.0))
            unit = get(param_data, :unit, get(param_data, "unit", ""))
            pk_params[string(param_name)] = SchemaOrgPropertyValue(
                string(param_name),
                value;
                unit_text=unit
            )
        end
    end

    rxcui = get(darwin_drug, :rxnorm_cui, get(darwin_drug, "rxnorm_cui", nothing))
    name = get(darwin_drug, :name, get(darwin_drug, "name", "Unknown Drug"))

    return SchemaOrgDrug(
        name,                                    # name
        get(darwin_drug, :chebi_id, get(darwin_drug, "chebi_id", nothing)),  # identifier
        nothing,                                 # url
        same_as,                                # same_as
        rxcui,                                  # rxcui
        name,                                   # non_proprietary_name
        nothing,                                # proprietary_name
        codes,                                  # codes
        get(darwin_drug, :active_ingredient, get(darwin_drug, "active_ingredient", nothing)),
        get(darwin_drug, :drug_class, get(darwin_drug, "drug_class", nothing)),
        get(darwin_drug, :dosage_form, get(darwin_drug, "dosage_form", nothing)),
        get(darwin_drug, :route, get(darwin_drug, "route", nothing)),
        SchemaOrgDrugStrength[],               # available_strength
        SchemaOrgDoseSchedule[],               # dose_schedule
        nothing,                               # maximum_intake
        get(darwin_drug, :clinical_pharmacology, get(darwin_drug, "clinical_pharmacology", nothing)),
        get(darwin_drug, :mechanism_of_action, get(darwin_drug, "mechanism_of_action", nothing)),
        pk_params,                             # pharmacokinetics
        String[],                              # interacting_drug
        nothing, nothing, nothing, nothing,    # warnings
        nothing, nothing, nothing,
        nothing,                               # prescription_status
        nothing,                               # is_available_generically
        nothing,                               # legal_status
        nothing,                               # label_details
        nothing                                # prescribing_info
    )
end

end # module SchemaOrg
