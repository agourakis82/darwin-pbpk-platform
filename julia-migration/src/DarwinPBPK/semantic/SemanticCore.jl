# =============================================================================
# SemanticCore - Darwin PBPK Semantic Web Layer
# =============================================================================
# Main module for JSON-LD serialization and OBO Foundry ontology integration.
#
# Provides FAIR (Findable, Accessible, Interoperable, Reusable) data layer for:
# - Drug entities (ChEBI, DrOn, Schema.org)
# - DDI predictions (DINTO, DIDEO)
# - PK parameters (QUDT, PATO, IAO)
# - Provenance tracking (PROV-O)
#
# References:
# - JSON-LD 1.1: https://www.w3.org/TR/json-ld11/
# - OBO Foundry: http://obofoundry.org/
# - QUDT: http://qudt.org/
# - PROV-O: https://www.w3.org/TR/prov-o/
#
# Author: Dr. Demetrios Agourakis
# Date: December 2025
# Version: 1.0.0
# =============================================================================

module SemanticCore

# =============================================================================
# Submodule Includes
# =============================================================================

include("contexts.jl")
include("qudt_units.jl")
include("jsonld_serializer.jl")
include("provenance.jl")
include("schema_org.jl")
include("doid_ontology.jl")

# =============================================================================
# Re-exports from Contexts
# =============================================================================

using .Contexts
export DARWIN_CONTEXT, DRUG_CONTEXT, DDI_CONTEXT, PARAMETER_CONTEXT
export EVIDENCE_CONTEXT, PROVENANCE_CONTEXT
export get_context, merge_contexts, context_to_json

# =============================================================================
# Re-exports from QUDTUnits
# =============================================================================

using .QUDTUnits
export QUDTUnit, SemanticQuantity
export QUDT_UNITS, get_qudt_unit, validate_dimension, dimension_compatible
export convert_units, to_quantity_value

# =============================================================================
# Re-exports from JSONLDSerializer
# =============================================================================

using .JSONLDSerializer
export to_jsonld_drug, to_jsonld_ddi, to_jsonld_parameter
export to_jsonld_simulation, to_jsonld_evidence
export to_turtle, serialize_entity
export JSONLDDocument, add_entity, finalize_document

# =============================================================================
# Re-exports from Provenance
# =============================================================================

using .Provenance
export ProvenanceRecord, ProvenanceActivity, ProvenanceAgent
export create_prediction_provenance, create_simulation_provenance
export create_ddi_provenance
export add_derivation, add_attribution
export DARWIN_SOFTWARE_AGENT, DARWIN_PBPK_MODEL

# =============================================================================
# Re-exports from SchemaOrg
# =============================================================================

using .SchemaOrg
export SCHEMA_ORG_CONTEXT, SCHEMA_ORG_DRUG_CONTEXT, SCHEMA_ORG_STUDY_CONTEXT
export SchemaOrgDrug, SchemaOrgDrugStrength, SchemaOrgDoseSchedule
export SchemaOrgMedicalStudy, SchemaOrgPropertyValue, SchemaOrgMedicalCode
export to_schema_org_drug, to_schema_org_study, to_schema_org_property_value
export map_pk_parameter_to_schema, map_ddi_to_schema, create_drug_from_darwin
export DrugPrescriptionStatus, DrugPregnancyCategory, MedicalStudyStatus

# =============================================================================
# Re-exports from DOIDOntology
# =============================================================================

using .DOIDOntology
export DOIDTerm, DOIDIndex
export load_doid, search_doid, get_disease_by_id, get_disease_by_name
export get_disease_xrefs, get_disease_hierarchy, get_diseases_for_drug_class
export DOID_IRI_PREFIX

# =============================================================================
# Convenience Functions
# =============================================================================

"""
    semantic_version() -> String

Return the version of the SemanticCore module.
"""
function semantic_version()::String
    return "1.0.0"
end

"""
    list_supported_ontologies() -> Vector{String}

List all supported ontologies and vocabularies.
"""
function list_supported_ontologies()::Vector{String}
    return [
        # Schema.org Medical Types
        "Schema.org Drug - Web-standard drug representation",
        "Schema.org MedicalStudy - Clinical trial representation",
        "Schema.org DrugStrength - Dosage strength",
        "Schema.org DoseSchedule - Dosing regimen",
        "Schema.org MedicalCode - Ontology cross-references",
        "Schema.org PropertyValue - Structured measurements",

        # OBO Foundry Ontologies (fully integrated)
        "DOID - Human Disease Ontology [FULL - v2025-11-25]",
        "UO - Units Ontology [FULL]",
        "OBI - Ontology for Biomedical Investigations [FULL]",

        # OBO Foundry Ontologies (prefix support)
        "BFO - Basic Formal Ontology (upper-level)",
        "ChEBI - Chemical Entities of Biological Interest",
        "DrOn - Drug Ontology",
        "DINTO - Drug-Drug Interactions Ontology",
        "DIDEO - Drug-Drug Interaction Evidence Ontology",
        "OGMS - Ontology for General Medical Science",
        "PATO - Phenotype and Trait Ontology",
        "IAO - Information Artifact Ontology",
        "STATO - Statistics Ontology",
        "RO - Relations Ontology",
        "GO - Gene Ontology",
        "UBERON - Uber Anatomy Ontology",
        "HP - Human Phenotype Ontology",

        # Standards
        "QUDT - Quantities, Units, Dimensions, Types [FULL]",
        "PROV-O - W3C Provenance Ontology [FULL]",
    ]
end

"""
    list_supported_units() -> Vector{String}

List all supported PK units.
"""
function list_supported_units()::Vector{String}
    return collect(keys(QUDT_UNITS))
end

"""
    validate_jsonld(doc::Dict{String, Any}) -> NamedTuple

Validate a JSON-LD document structure.

# Returns
- NamedTuple with :valid, :errors, :warnings
"""
function validate_jsonld(doc::Dict{String, Any})::NamedTuple
    errors = String[]
    warnings = String[]

    # Check for @context
    if !haskey(doc, "@context")
        push!(warnings, "Missing @context - document may not be interoperable")
    end

    # Check for @type
    if !haskey(doc, "@type")
        push!(warnings, "Missing @type - entity type unknown")
    end

    # Check for @id
    if !haskey(doc, "@id")
        push!(warnings, "Missing @id - entity not uniquely identifiable")
    end

    # Validate @context structure if present
    if haskey(doc, "@context")
        ctx = doc["@context"]
        if !(ctx isa Dict)
            push!(errors, "@context must be a dictionary or array of contexts")
        end
    end

    return (
        valid = isempty(errors),
        errors = errors,
        warnings = warnings
    )
end

# =============================================================================
# High-Level API
# =============================================================================

"""
    create_semantic_drug(
        name::String;
        chebi_id::String = "",
        drugbank_id::String = "",
        rxnorm_cui::String = "",
        smiles::String = "",
        molecular_weight::Float64 = 0.0,
        atc_codes::Vector{String} = String[],
        mechanism_of_action::String = ""
    ) -> Dict{String, Any}

Create a semantic drug entity with JSON-LD representation.
"""
function create_semantic_drug(
    name::String;
    chebi_id::String = "",
    drugbank_id::String = "",
    rxnorm_cui::String = "",
    smiles::String = "",
    molecular_weight::Float64 = 0.0,
    atc_codes::Vector{String} = String[],
    mechanism_of_action::String = ""
)::Dict{String, Any}

    data = Dict{String, Any}("name" => name)

    if !isempty(chebi_id)
        data["chebi_id"] = chebi_id
    end
    if !isempty(drugbank_id)
        data["drugbank_id"] = drugbank_id
    end
    if !isempty(rxnorm_cui)
        data["rxnorm_cui"] = rxnorm_cui
    end
    if !isempty(smiles)
        data["smiles"] = smiles
    end
    if molecular_weight > 0
        data["molecular_weight"] = molecular_weight
    end
    if !isempty(atc_codes)
        data["atc_codes"] = atc_codes
    end
    if !isempty(mechanism_of_action)
        data["mechanism_of_action"] = mechanism_of_action
    end

    return to_jsonld_drug(data)
end

"""
    create_semantic_ddi(
        perpetrator::String,
        victim::String,
        mechanism::Symbol,
        auc_ratio::Float64;
        cmax_ratio::Union{Float64, Nothing} = nothing,
        evidence_level::Symbol = :unknown,
        pubmed_ids::Vector{Int} = Int[]
    ) -> Dict{String, Any}

Create a semantic DDI entity with JSON-LD representation.
"""
function create_semantic_ddi(
    perpetrator::String,
    victim::String,
    mechanism::Symbol,
    auc_ratio::Float64;
    cmax_ratio::Union{Float64, Nothing} = nothing,
    evidence_level::Symbol = :unknown,
    pubmed_ids::Vector{Int} = Int[]
)::Dict{String, Any}

    data = Dict{String, Any}(
        "perpetrator" => Dict("name" => perpetrator),
        "victim" => Dict("name" => victim),
        "mechanism" => mechanism,
        "auc_ratio" => auc_ratio
    )

    if cmax_ratio !== nothing
        data["cmax_ratio"] = cmax_ratio
    end

    if evidence_level != :unknown || !isempty(pubmed_ids)
        data["evidence"] = Dict{String, Any}(
            "level" => evidence_level,
            "pubmed_ids" => pubmed_ids
        )
    end

    return to_jsonld_ddi(data)
end

"""
    create_semantic_parameter(
        name::String,
        value::Float64,
        unit::String;
        symbol::String = "",
        omega::Union{Float64, Nothing} = nothing,
        source::Union{String, Nothing} = nothing
    ) -> Dict{String, Any}

Create a semantic PK parameter entity with JSON-LD representation.
"""
function create_semantic_parameter(
    name::String,
    value::Float64,
    unit::String;
    symbol::String = "",
    omega::Union{Float64, Nothing} = nothing,
    source::Union{String, Nothing} = nothing
)::Dict{String, Any}

    data = Dict{String, Any}(
        "name" => name,
        "value" => value,
        "unit" => unit
    )

    if !isempty(symbol)
        data["symbol"] = symbol
    end
    if omega !== nothing
        data["omega"] = omega
    end
    if source !== nothing
        data["source"] = source
    end

    return to_jsonld_parameter(data)
end

"""
    annotate_with_provenance(
        entity::Dict{String, Any},
        model_type::Symbol;
        derived_from::Vector{String} = String[]
    ) -> Dict{String, Any}

Add provenance annotations to an existing entity.
"""
function annotate_with_provenance(
    entity::Dict{String, Any},
    model_type::Symbol;
    derived_from::Vector{String} = String[]
)::Dict{String, Any}

    entity_iri = get(entity, "@id", "darwin:entity/unknown")

    prov = create_prediction_provenance(
        entity_iri,
        model_type,
        Dict{String, Any}();
        derived_from = derived_from
    )

    # Merge provenance into entity
    prov_jsonld = Provenance.to_jsonld(prov)

    entity["prov:wasGeneratedBy"] = prov_jsonld["prov:wasGeneratedBy"]

    if haskey(prov_jsonld, "prov:wasDerivedFrom")
        entity["prov:wasDerivedFrom"] = prov_jsonld["prov:wasDerivedFrom"]
    end

    if haskey(prov_jsonld, "prov:wasAttributedTo")
        entity["prov:wasAttributedTo"] = prov_jsonld["prov:wasAttributedTo"]
    end

    if haskey(prov_jsonld, "prov:generatedAtTime")
        entity["prov:generatedAtTime"] = prov_jsonld["prov:generatedAtTime"]
    end

    return entity
end

# =============================================================================
# Export Utilities
# =============================================================================

"""
    export_to_jsonld(entity::Dict{String, Any}; pretty::Bool=true) -> String

Export entity to JSON-LD string.
"""
function export_to_jsonld(entity::Dict{String, Any}; pretty::Bool = true)::String
    if pretty
        return JSON.json(entity, 2)
    else
        return JSON.json(entity)
    end
end

"""
    export_to_turtle(entity::Dict{String, Any}) -> String

Export entity to RDF/Turtle string.
"""
function export_to_turtle(entity::Dict{String, Any})::String
    return to_turtle(entity)
end

using JSON

end # module SemanticCore
