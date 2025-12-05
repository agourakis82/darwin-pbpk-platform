# =============================================================================
# JSON-LD Context Definitions - Darwin PBPK Semantic Layer
# =============================================================================
# Implements JSON-LD 1.1 contexts for semantic web integration.
#
# Integrated Standards:
# - Schema.org (web-facing vocabulary)
# - QUDT (units and quantities)
# - OBO Foundry (biomedical ontologies)
# - PROV-O (provenance)
#
# Author: Dr. Demetrios Agourakis
# Date: December 2025
# =============================================================================

module Contexts

export DARWIN_CONTEXT, DRUG_CONTEXT, DDI_CONTEXT, PARAMETER_CONTEXT
export EVIDENCE_CONTEXT, PROVENANCE_CONTEXT
export get_context, merge_contexts, context_to_json

using JSON

# =============================================================================
# OBO Foundry Prefix Definitions
# =============================================================================

"""
OBO Foundry ontology prefixes with proper JSON-LD 1.1 handling.
Uses @prefix: true for underscore-based CURIEs.
"""
const OBO_PREFIXES = Dict{String, Any}(
    # Upper-level
    "BFO" => Dict("@id" => "http://purl.obolibrary.org/obo/BFO_", "@prefix" => true),

    # Chemical entities
    "CHEBI" => Dict("@id" => "http://purl.obolibrary.org/obo/CHEBI_", "@prefix" => true),
    "DRON" => Dict("@id" => "http://purl.obolibrary.org/obo/DRON_", "@prefix" => true),

    # DDI ontologies
    "DINTO" => Dict("@id" => "http://purl.obolibrary.org/obo/DINTO_", "@prefix" => true),
    "DIDEO" => Dict("@id" => "http://purl.obolibrary.org/obo/DIDEO_", "@prefix" => true),

    # Investigations and studies
    "OBI" => Dict("@id" => "http://purl.obolibrary.org/obo/OBI_", "@prefix" => true),

    # Diseases
    "DOID" => Dict("@id" => "http://purl.obolibrary.org/obo/DOID_", "@prefix" => true),
    "OGMS" => Dict("@id" => "http://purl.obolibrary.org/obo/OGMS_", "@prefix" => true),

    # Qualities and units
    "PATO" => Dict("@id" => "http://purl.obolibrary.org/obo/PATO_", "@prefix" => true),
    "UO" => Dict("@id" => "http://purl.obolibrary.org/obo/UO_", "@prefix" => true),

    # Information artifacts
    "IAO" => Dict("@id" => "http://purl.obolibrary.org/obo/IAO_", "@prefix" => true),

    # Statistics
    "STATO" => Dict("@id" => "http://purl.obolibrary.org/obo/STATO_", "@prefix" => true),

    # Relations
    "RO" => Dict("@id" => "http://purl.obolibrary.org/obo/RO_", "@prefix" => true),

    # Gene Ontology
    "GO" => Dict("@id" => "http://purl.obolibrary.org/obo/GO_", "@prefix" => true),

    # Anatomy
    "UBERON" => Dict("@id" => "http://purl.obolibrary.org/obo/UBERON_", "@prefix" => true),

    # Human Phenotype
    "HP" => Dict("@id" => "http://purl.obolibrary.org/obo/HP_", "@prefix" => true),
)

# =============================================================================
# Core Darwin Context
# =============================================================================

"""
Core JSON-LD 1.1 context for Darwin PBPK Platform.
Provides namespace prefixes and property definitions.
"""
const DARWIN_CONTEXT = Dict{String, Any}(
    "@context" => Dict{String, Any}(
        # JSON-LD version
        "@version" => 1.1,

        # Base URIs
        "@vocab" => "https://schema.org/",
        "@base" => "https://darwin-pbpk.org/",

        # Darwin namespace
        "darwin" => "https://darwin-pbpk.org/schema/",

        # QUDT namespaces
        "qudt" => "http://qudt.org/schema/qudt/",
        "unit" => "http://qudt.org/vocab/unit/",
        "quantitykind" => "http://qudt.org/vocab/quantitykind/",

        # OBO generic prefix
        "obo" => "http://purl.obolibrary.org/obo/",

        # Standard vocabularies
        "rdfs" => "http://www.w3.org/2000/01/rdf-schema#",
        "xsd" => "http://www.w3.org/2001/XMLSchema#",
        "skos" => "http://www.w3.org/2004/02/skos/core#",
        "dcterms" => "http://purl.org/dc/terms/",
        "prov" => "http://www.w3.org/ns/prov#",

        # External drug databases
        "drugbank" => "http://wifo5-04.informatik.uni-mannheim.de/drugbank/resource/drugs/",
        "pubchem" => "https://pubchem.ncbi.nlm.nih.gov/compound/",
        "rxnorm" => "http://purl.bioontology.org/ontology/RXNORM/",
        "mesh" => "http://id.nlm.nih.gov/mesh/",
        "uniprot" => "http://purl.uniprot.org/uniprot/",

        # OBO Foundry prefixes
        OBO_PREFIXES...,

        # =================================================================
        # Standard OBO Relations (RO)
        # =================================================================
        "has_role" => Dict("@id" => "obo:RO_0000087", "@type" => "@id"),
        "has_quality" => Dict("@id" => "obo:RO_0000086", "@type" => "@id"),
        "has_part" => Dict("@id" => "obo:BFO_0000051", "@type" => "@id"),
        "part_of" => Dict("@id" => "obo:BFO_0000050", "@type" => "@id"),
        "realizes" => Dict("@id" => "obo:BFO_0000055", "@type" => "@id"),
        "participates_in" => Dict("@id" => "obo:RO_0000056", "@type" => "@id"),
        "has_participant" => Dict("@id" => "obo:RO_0000057", "@type" => "@id"),
        "is_about" => Dict("@id" => "obo:IAO_0000136", "@type" => "@id"),

        # OBI relations
        "has_specified_input" => Dict("@id" => "obo:OBI_0000293", "@type" => "@id"),
        "has_specified_output" => Dict("@id" => "obo:OBI_0000299", "@type" => "@id"),

        # =================================================================
        # QUDT Properties
        # =================================================================
        "numericValue" => "qudt:numericValue",
        "unitRef" => Dict("@id" => "qudt:unit", "@type" => "@id"),
        "quantityKind" => Dict("@id" => "qudt:hasQuantityKind", "@type" => "@id"),

        # UO crosslink
        "unitOBO" => Dict("@id" => "darwin:unitOBO", "@type" => "@id"),

        # =================================================================
        # Darwin-specific Properties
        # =================================================================

        # Drug properties
        "chebiId" => Dict("@id" => "darwin:chebiId"),
        "drugbankId" => Dict("@id" => "darwin:drugbankId"),
        "rxnormCui" => Dict("@id" => "darwin:rxnormCui"),
        "atcCode" => Dict("@id" => "darwin:atcCode"),
        "smiles" => Dict("@id" => "darwin:smiles"),
        "inchi" => Dict("@id" => "darwin:inchi"),
        "inchiKey" => Dict("@id" => "darwin:inchiKey"),
        "molecularFormula" => Dict("@id" => "darwin:molecularFormula"),
        "molecularWeight" => Dict("@id" => "darwin:molecularWeight"),

        # DDI properties
        "perpetrator" => Dict("@id" => "darwin:perpetrator", "@type" => "@id"),
        "victim" => Dict("@id" => "darwin:victim", "@type" => "@id"),
        "mechanism" => Dict("@id" => "darwin:mechanism", "@type" => "@id"),
        "aucRatio" => Dict("@id" => "darwin:aucRatio"),
        "cmaxRatio" => Dict("@id" => "darwin:cmaxRatio"),
        "fdaClassification" => Dict("@id" => "darwin:fdaClassification"),

        # PK parameter properties
        "clearance" => Dict("@id" => "darwin:clearance"),
        "volumeOfDistribution" => Dict("@id" => "darwin:volumeOfDistribution"),
        "halfLife" => Dict("@id" => "darwin:halfLife"),
        "bioavailability" => Dict("@id" => "darwin:bioavailability"),
        "fractionUnbound" => Dict("@id" => "darwin:fractionUnbound"),

        # Uncertainty properties
        "credibleInterval" => Dict("@id" => "darwin:credibleInterval"),
        "confidenceLevel" => Dict("@id" => "darwin:confidenceLevel"),
        "standardError" => Dict("@id" => "darwin:standardError"),
        "omega" => Dict("@id" => "darwin:omega"),  # IIV

        # Evidence properties
        "evidenceLevel" => Dict("@id" => "darwin:evidenceLevel"),
        "pubmedId" => Dict("@id" => "darwin:pubmedId"),
        "studyDesign" => Dict("@id" => "darwin:studyDesign"),
        "nSubjects" => Dict("@id" => "darwin:nSubjects"),
    )
)

# =============================================================================
# Specialized Contexts
# =============================================================================

"""
Drug-specific context with ChEBI, DrOn, and Schema.org Drug alignment.
"""
const DRUG_CONTEXT = Dict{String, Any}(
    "@context" => merge(
        DARWIN_CONTEXT["@context"],
        Dict{String, Any}(
            # Schema.org Drug type
            "Drug" => "https://schema.org/Drug",

            # ChEBI molecular entity
            "MolecularEntity" => "obo:CHEBI_24431",

            # DrOn drug product
            "DrugProduct" => "obo:DRON_00000001",

            # Active ingredient relation
            "hasActiveIngredient" => Dict("@id" => "darwin:hasActiveIngredient", "@type" => "@id"),

            # Chemical properties
            "logP" => Dict("@id" => "darwin:logP"),
            "pKa" => Dict("@id" => "darwin:pKa"),
            "polarSurfaceArea" => Dict("@id" => "darwin:polarSurfaceArea"),
            "hBondDonors" => Dict("@id" => "darwin:hBondDonors"),
            "hBondAcceptors" => Dict("@id" => "darwin:hBondAcceptors"),
            "rotatableBonds" => Dict("@id" => "darwin:rotatableBonds"),
        )
    )
)

"""
DDI-specific context with DINTO, DIDEO alignment.
"""
const DDI_CONTEXT = Dict{String, Any}(
    "@context" => merge(
        DARWIN_CONTEXT["@context"],
        Dict{String, Any}(
            # DINTO DDI type
            "DrugDrugInteraction" => "obo:DINTO_0000001",

            # Mechanism types
            "CYPInhibition" => "obo:DINTO_0000102",
            "CYPInduction" => "obo:DINTO_0000105",
            "TransporterInhibition" => "obo:DINTO_0000106",
            "ProteinBindingDisplacement" => "obo:DINTO_0000108",

            # DIDEO evidence types
            "InVitroStudy" => "obo:DIDEO_0000001",
            "ClinicalStudy" => "obo:DIDEO_0000015",
            "PopulationPKStudy" => "obo:DIDEO_0000020",

            # DDI-specific properties
            "clinicalSignificance" => Dict("@id" => "darwin:clinicalSignificance"),
            "managementRecommendation" => Dict("@id" => "darwin:managementRecommendation"),
            "interactionSeverity" => Dict("@id" => "darwin:interactionSeverity"),
        )
    )
)

"""
PK Parameter-specific context with QUDT, PATO, IAO alignment.
"""
const PARAMETER_CONTEXT = Dict{String, Any}(
    "@context" => merge(
        DARWIN_CONTEXT["@context"],
        Dict{String, Any}(
            # IAO data item
            "DataItem" => "obo:IAO_0000027",

            # QUDT quantity value
            "QuantityValue" => "qudt:QuantityValue",

            # PATO qualities
            "Concentration" => "obo:PATO_0000033",
            "Volume" => "obo:PATO_0000918",
            "Rate" => "obo:PATO_0001025",
            "Mass" => "obo:PATO_0000125",

            # Parameter metadata
            "parameterType" => Dict("@id" => "darwin:parameterType"),
            "populationMean" => Dict("@id" => "darwin:populationMean"),
            "populationVariance" => Dict("@id" => "darwin:populationVariance"),
            "covariateModel" => Dict("@id" => "darwin:covariateModel"),
        )
    )
)

"""
Evidence and study design context with OBI, STATO alignment.
"""
const EVIDENCE_CONTEXT = Dict{String, Any}(
    "@context" => merge(
        DARWIN_CONTEXT["@context"],
        Dict{String, Any}(
            # OBI investigation types
            "Investigation" => "obo:OBI_0000066",
            "StudyDesignExecution" => "obo:OBI_0000471",
            "Assay" => "obo:OBI_0000070",
            "Population" => "obo:OBI_0000181",

            # STATO statistical types
            "Variance" => "obo:STATO_0000039",
            "LogNormalDistribution" => "obo:STATO_0000227",
            "RandomEffect" => "obo:STATO_0000381",
            "MixedEffectsModel" => "obo:STATO_0000189",

            # Study metadata
            "studyPopulation" => Dict("@id" => "darwin:studyPopulation", "@type" => "@id"),
            "inclusionCriteria" => Dict("@id" => "darwin:inclusionCriteria"),
            "exclusionCriteria" => Dict("@id" => "darwin:exclusionCriteria"),
        )
    )
)

"""
Provenance context with PROV-O alignment.
"""
const PROVENANCE_CONTEXT = Dict{String, Any}(
    "@context" => merge(
        DARWIN_CONTEXT["@context"],
        Dict{String, Any}(
            # PROV-O types
            "Entity" => "prov:Entity",
            "Activity" => "prov:Activity",
            "Agent" => "prov:Agent",
            "SoftwareAgent" => "prov:SoftwareAgent",

            # PROV-O relations
            "wasGeneratedBy" => Dict("@id" => "prov:wasGeneratedBy", "@type" => "@id"),
            "wasDerivedFrom" => Dict("@id" => "prov:wasDerivedFrom", "@type" => "@id"),
            "wasAttributedTo" => Dict("@id" => "prov:wasAttributedTo", "@type" => "@id"),
            "wasAssociatedWith" => Dict("@id" => "prov:wasAssociatedWith", "@type" => "@id"),
            "used" => Dict("@id" => "prov:used", "@type" => "@id"),
            "generatedAtTime" => Dict("@id" => "prov:generatedAtTime", "@type" => "xsd:dateTime"),
            "startedAtTime" => Dict("@id" => "prov:startedAtTime", "@type" => "xsd:dateTime"),
            "endedAtTime" => Dict("@id" => "prov:endedAtTime", "@type" => "xsd:dateTime"),
        )
    )
)

# =============================================================================
# Context Utilities
# =============================================================================

"""
    get_context(context_type::Symbol) -> Dict{String, Any}

Get a specific JSON-LD context by type.

# Arguments
- `context_type`: One of :darwin, :drug, :ddi, :parameter, :evidence, :provenance

# Returns
- The corresponding JSON-LD context dictionary
"""
function get_context(context_type::Symbol)::Dict{String, Any}
    contexts = Dict{Symbol, Dict{String, Any}}(
        :darwin => DARWIN_CONTEXT,
        :drug => DRUG_CONTEXT,
        :ddi => DDI_CONTEXT,
        :parameter => PARAMETER_CONTEXT,
        :evidence => EVIDENCE_CONTEXT,
        :provenance => PROVENANCE_CONTEXT,
    )

    if !haskey(contexts, context_type)
        error("Unknown context type: $context_type. Valid types: $(keys(contexts))")
    end

    return contexts[context_type]
end

"""
    merge_contexts(contexts::Vector{Symbol}) -> Dict{String, Any}

Merge multiple contexts into a single context.

# Arguments
- `contexts`: Vector of context type symbols to merge

# Returns
- Merged JSON-LD context dictionary
"""
function merge_contexts(contexts::Vector{Symbol})::Dict{String, Any}
    if isempty(contexts)
        return DARWIN_CONTEXT
    end

    merged = Dict{String, Any}()

    for ctx_type in contexts
        ctx = get_context(ctx_type)
        for (k, v) in ctx["@context"]
            merged[k] = v
        end
    end

    return Dict{String, Any}("@context" => merged)
end

"""
    context_to_json(context::Dict{String, Any}; pretty::Bool=true) -> String

Serialize a context to JSON string.

# Arguments
- `context`: JSON-LD context dictionary
- `pretty`: Whether to pretty-print the JSON (default: true)

# Returns
- JSON string representation
"""
function context_to_json(context::Dict{String, Any}; pretty::Bool=true)::String
    if pretty
        return JSON.json(context, 2)
    else
        return JSON.json(context)
    end
end

end # module Contexts
