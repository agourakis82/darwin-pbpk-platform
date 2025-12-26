# =============================================================================
# JSON-LD Serializer - Darwin PBPK Semantic Layer
# =============================================================================
# Provides JSON-LD serialization for PBPK entities including drugs, DDIs,
# parameters, and simulation results.
#
# Implements:
# - Drug serialization with ChEBI/DrOn/Schema.org alignment
# - DDI serialization with DINTO/DIDEO alignment
# - PK parameter serialization with QUDT/PATO alignment
# - Simulation result serialization
#
# Author: Dr. Sounio Agourakis
# Date: December 2025
# =============================================================================

module JSONLDSerializer

using JSON
using Dates

# Import sibling modules
using ..Contexts: DARWIN_CONTEXT, DRUG_CONTEXT, DDI_CONTEXT, PARAMETER_CONTEXT
using ..Contexts: EVIDENCE_CONTEXT, PROVENANCE_CONTEXT, get_context, merge_contexts
using ..QUDTUnits: QUDTUnit, SemanticQuantity, QUDT_UNITS, get_qudt_unit, to_jsonld

export to_jsonld_drug, to_jsonld_ddi, to_jsonld_parameter, to_jsonld_simulation
export to_jsonld_evidence, to_turtle, serialize_entity
export JSONLDDocument, add_entity, finalize_document

# =============================================================================
# JSON-LD Document Builder
# =============================================================================

"""
    JSONLDDocument

Builder for constructing JSON-LD documents with multiple entities.
"""
mutable struct JSONLDDocument
    context::Dict{String, Any}
    graph::Vector{Dict{String, Any}}
    base_uri::String
end

function JSONLDDocument(; base_uri::String = "https://darwin-pbpk.org/")
    JSONLDDocument(
        DARWIN_CONTEXT["@context"],
        Vector{Dict{String, Any}}(),
        base_uri
    )
end

"""
    add_entity(doc::JSONLDDocument, entity::Dict{String, Any})

Add an entity to the document graph.
"""
function add_entity(doc::JSONLDDocument, entity::Dict{String, Any})
    push!(doc.graph, entity)
end

"""
    finalize_document(doc::JSONLDDocument) -> Dict{String, Any}

Finalize the document and return the complete JSON-LD structure.
"""
function finalize_document(doc::JSONLDDocument)::Dict{String, Any}
    if length(doc.graph) == 1
        # Single entity - inline
        result = Dict{String, Any}("@context" => doc.context)
        merge!(result, doc.graph[1])
        return result
    else
        # Multiple entities - use @graph
        return Dict{String, Any}(
            "@context" => doc.context,
            "@graph" => doc.graph
        )
    end
end

# =============================================================================
# Drug Serialization
# =============================================================================

"""
    to_jsonld_drug(drug_data::Dict; include_context::Bool=true) -> Dict{String, Any}

Convert drug data to JSON-LD with Schema.org/ChEBI/DrOn alignment.

# Arguments
- `drug_data`: Dictionary with drug properties:
  - `name`: Drug name (required)
  - `chebi_id`: ChEBI identifier (e.g., "CHEBI:6801")
  - `drugbank_id`: DrugBank identifier
  - `rxnorm_cui`: RxNorm CUI
  - `smiles`: SMILES string
  - `molecular_weight`: Molecular weight in g/mol
  - `atc_codes`: Vector of ATC codes
  - `mechanism_of_action`: GO term or description
- `include_context`: Whether to include @context

# Returns
- JSON-LD dictionary
"""
function to_jsonld_drug(drug_data::Dict; include_context::Bool = true)::Dict{String, Any}
    result = Dict{String, Any}()

    if include_context
        result["@context"] = DRUG_CONTEXT["@context"]
    end

    # Types: Schema.org Drug + ChEBI molecular entity
    result["@type"] = ["Drug", "obo:CHEBI_24431"]

    # ID from ChEBI if available
    if haskey(drug_data, :chebi_id) || haskey(drug_data, "chebi_id")
        chebi_id = get(drug_data, :chebi_id, get(drug_data, "chebi_id", ""))
        result["@id"] = chebi_id
    end

    # Name
    name = get(drug_data, :name, get(drug_data, "name", "Unknown Drug"))
    result["name"] = name
    result["rdfs:label"] = name

    # Description
    if haskey(drug_data, :description) || haskey(drug_data, "description")
        result["description"] = get(drug_data, :description, get(drug_data, "description", ""))
    end

    # Identifiers as PropertyValue array
    identifiers = Vector{Dict{String, Any}}()

    if haskey(drug_data, :chebi_id) || haskey(drug_data, "chebi_id")
        push!(identifiers, Dict{String, Any}(
            "@type" => "PropertyValue",
            "propertyID" => "ChEBI",
            "value" => get(drug_data, :chebi_id, get(drug_data, "chebi_id", ""))
        ))
    end

    if haskey(drug_data, :drugbank_id) || haskey(drug_data, "drugbank_id")
        push!(identifiers, Dict{String, Any}(
            "@type" => "PropertyValue",
            "propertyID" => "DrugBank",
            "value" => get(drug_data, :drugbank_id, get(drug_data, "drugbank_id", ""))
        ))
    end

    if haskey(drug_data, :rxnorm_cui) || haskey(drug_data, "rxnorm_cui")
        push!(identifiers, Dict{String, Any}(
            "@type" => "PropertyValue",
            "propertyID" => "RxNorm",
            "value" => get(drug_data, :rxnorm_cui, get(drug_data, "rxnorm_cui", ""))
        ))
    end

    if !isempty(identifiers)
        result["identifier"] = identifiers
    end

    # Chemical properties
    if haskey(drug_data, :smiles) || haskey(drug_data, "smiles")
        result["darwin:smiles"] = get(drug_data, :smiles, get(drug_data, "smiles", ""))
    end

    if haskey(drug_data, :inchi) || haskey(drug_data, "inchi")
        result["darwin:inchi"] = get(drug_data, :inchi, get(drug_data, "inchi", ""))
    end

    if haskey(drug_data, :inchi_key) || haskey(drug_data, "inchi_key")
        result["darwin:inchiKey"] = get(drug_data, :inchi_key, get(drug_data, "inchi_key", ""))
    end

    # Molecular weight with QUDT
    if haskey(drug_data, :molecular_weight) || haskey(drug_data, "molecular_weight")
        mw = get(drug_data, :molecular_weight, get(drug_data, "molecular_weight", 0.0))
        result["darwin:molecularWeight"] = Dict{String, Any}(
            "@type" => "qudt:QuantityValue",
            "numericValue" => mw,
            "unitRef" => "unit:GM-PER-MOL",
            "darwin:unitOBO" => Dict("@id" => "UO:0000088")
        )
    end

    # ATC codes
    if haskey(drug_data, :atc_codes) || haskey(drug_data, "atc_codes")
        result["darwin:atcCode"] = get(drug_data, :atc_codes, get(drug_data, "atc_codes", []))
    end

    # Mechanism of action
    if haskey(drug_data, :mechanism_of_action) || haskey(drug_data, "mechanism_of_action")
        moa = get(drug_data, :mechanism_of_action, get(drug_data, "mechanism_of_action", ""))
        result["mechanismOfAction"] = Dict{String, Any}(
            "@type" => "obo:GO_0008150",  # biological_process
            "name" => moa
        )
    end

    # ChEBI roles (e.g., enzyme inhibitor, transporter substrate)
    if haskey(drug_data, :roles) || haskey(drug_data, "roles")
        roles = get(drug_data, :roles, get(drug_data, "roles", []))
        result["has_role"] = [Dict("@id" => role) for role in roles]
    end

    return result
end

# =============================================================================
# DDI Serialization
# =============================================================================

"""
    to_jsonld_ddi(ddi_data::Dict; include_context::Bool=true) -> Dict{String, Any}

Convert DDI data to JSON-LD with DINTO/DIDEO alignment.

# Arguments
- `ddi_data`: Dictionary with DDI properties:
  - `perpetrator`: Perpetrator drug (symbol or dict)
  - `victim`: Victim drug (symbol or dict)
  - `mechanism`: DDI mechanism type
  - `auc_ratio`: AUC fold-change
  - `cmax_ratio`: Cmax fold-change (optional)
  - `evidence`: Evidence metadata (optional)
- `include_context`: Whether to include @context

# Returns
- JSON-LD dictionary
"""
function to_jsonld_ddi(ddi_data::Dict; include_context::Bool = true)::Dict{String, Any}
    result = Dict{String, Any}()

    if include_context
        result["@context"] = DDI_CONTEXT["@context"]
    end

    # Types
    result["@type"] = ["darwin:DrugDrugInteraction", "obo:DINTO_0000001"]

    # Generate ID
    perp_name = get_drug_name(get(ddi_data, :perpetrator, get(ddi_data, "perpetrator", "unknown")))
    vic_name = get_drug_name(get(ddi_data, :victim, get(ddi_data, "victim", "unknown")))
    result["@id"] = "darwin:ddi/$(perp_name)_$(vic_name)"

    # Perpetrator
    perpetrator = get(ddi_data, :perpetrator, get(ddi_data, "perpetrator", nothing))
    if perpetrator !== nothing
        if perpetrator isa Dict
            result["darwin:perpetrator"] = to_jsonld_drug(perpetrator; include_context=false)
        else
            result["darwin:perpetrator"] = Dict{String, Any}(
                "@type" => "Drug",
                "name" => string(perpetrator)
            )
        end
    end

    # Victim
    victim = get(ddi_data, :victim, get(ddi_data, "victim", nothing))
    if victim !== nothing
        if victim isa Dict
            result["darwin:victim"] = to_jsonld_drug(victim; include_context=false)
        else
            result["darwin:victim"] = Dict{String, Any}(
                "@type" => "Drug",
                "name" => string(victim)
            )
        end
    end

    # Mechanism
    mechanism = get(ddi_data, :mechanism, get(ddi_data, "mechanism", nothing))
    if mechanism !== nothing
        dinto_id = mechanism_to_dinto(mechanism)
        result["darwin:mechanism"] = Dict{String, Any}(
            "@type" => "obo:$(dinto_id)",
            "rdfs:label" => string(mechanism)
        )
    end

    # AUC ratio
    auc_ratio = get(ddi_data, :auc_ratio, get(ddi_data, "auc_ratio", nothing))
    if auc_ratio !== nothing
        result["darwin:aucRatio"] = auc_ratio
        result["darwin:fdaClassification"] = classify_ddi_severity(auc_ratio)
    end

    # Cmax ratio
    cmax_ratio = get(ddi_data, :cmax_ratio, get(ddi_data, "cmax_ratio", nothing))
    if cmax_ratio !== nothing
        result["darwin:cmaxRatio"] = cmax_ratio
    end

    # Evidence
    evidence = get(ddi_data, :evidence, get(ddi_data, "evidence", nothing))
    if evidence !== nothing
        result["darwin:evidence"] = to_jsonld_evidence(evidence; include_context=false)
    end

    return result
end

# =============================================================================
# Parameter Serialization
# =============================================================================

"""
    to_jsonld_parameter(param_data::Dict; include_context::Bool=true) -> Dict{String, Any}

Convert PK parameter to JSON-LD with QUDT/PATO/IAO alignment.

# Arguments
- `param_data`: Dictionary with parameter properties:
  - `name`: Parameter name (e.g., "Clearance")
  - `symbol`: Parameter symbol (e.g., "CL")
  - `value`: Numeric value
  - `unit`: Unit symbol (e.g., "L/h")
  - `omega`: Inter-individual variability (optional)
  - `source`: Literature source (optional)
- `include_context`: Whether to include @context

# Returns
- JSON-LD dictionary
"""
function to_jsonld_parameter(param_data::Dict; include_context::Bool = true)::Dict{String, Any}
    result = Dict{String, Any}()

    if include_context
        result["@context"] = PARAMETER_CONTEXT["@context"]
    end

    # Types
    result["@type"] = ["darwin:PKParameter", "obo:IAO_0000027"]

    # Name and symbol
    name = get(param_data, :name, get(param_data, "name", "Unknown"))
    symbol = get(param_data, :symbol, get(param_data, "symbol", ""))

    result["name"] = name
    result["rdfs:label"] = name
    if !isempty(symbol)
        result["darwin:symbol"] = symbol
        result["@id"] = "darwin:param/$(lowercase(symbol))"
    end

    # Value with QUDT unit
    value = get(param_data, :value, get(param_data, "value", nothing))
    unit_symbol = get(param_data, :unit, get(param_data, "unit", ""))

    if value !== nothing && !isempty(unit_symbol)
        unit = get_qudt_unit(unit_symbol)
        sq = SemanticQuantity(Float64(value), unit)
        result["darwin:value"] = to_jsonld(sq)
    elseif value !== nothing
        result["darwin:value"] = value
    end

    # PATO quality mapping
    pato = param_to_pato(name)
    if pato !== nothing
        result["is_about"] = Dict("@id" => pato)
    end

    # Inter-individual variability (omega)
    omega = get(param_data, :omega, get(param_data, "omega", nothing))
    if omega !== nothing
        result["darwin:interIndividualVariability"] = Dict{String, Any}(
            "@type" => "obo:STATO_0000039",  # variance
            "darwin:omega" => omega,
            "darwin:distribution" => "LogNormal"
        )
    end

    # Covariate model
    covariates = get(param_data, :covariates, get(param_data, "covariates", nothing))
    if covariates !== nothing
        result["darwin:covariateModel"] = covariates
    end

    # Source/provenance
    source = get(param_data, :source, get(param_data, "source", nothing))
    if source !== nothing
        result["prov:wasDerivedFrom"] = Dict("@id" => source)
    end

    return result
end

# =============================================================================
# Evidence Serialization
# =============================================================================

"""
    to_jsonld_evidence(evidence_data::Dict; include_context::Bool=true) -> Dict{String, Any}

Convert evidence/study data to JSON-LD with OBI/DIDEO alignment.
"""
function to_jsonld_evidence(evidence_data::Dict; include_context::Bool = true)::Dict{String, Any}
    result = Dict{String, Any}()

    if include_context
        result["@context"] = EVIDENCE_CONTEXT["@context"]
    end

    # Types
    evidence_type = get(evidence_data, :type, get(evidence_data, "type", :clinical))
    dideo_type = evidence_type_to_dideo(evidence_type)
    result["@type"] = ["darwin:Evidence", "obo:$(dideo_type)"]

    # Evidence level
    level = get(evidence_data, :level, get(evidence_data, "level", nothing))
    if level !== nothing
        result["darwin:evidenceLevel"] = string(level)
    end

    # Study design
    design = get(evidence_data, :study_design, get(evidence_data, "study_design", nothing))
    if design !== nothing
        result["darwin:studyDesign"] = string(design)
    end

    # Number of subjects
    n = get(evidence_data, :n_subjects, get(evidence_data, "n_subjects", nothing))
    if n !== nothing
        result["darwin:nSubjects"] = n
    end

    # PubMed IDs
    pubmed_ids = get(evidence_data, :pubmed_ids, get(evidence_data, "pubmed_ids", []))
    if !isempty(pubmed_ids)
        result["darwin:pubmedId"] = pubmed_ids
        result["dcterms:references"] = [
            Dict("@id" => "https://pubmed.ncbi.nlm.nih.gov/$id/") for id in pubmed_ids
        ]
    end

    # FDA label
    fda_label = get(evidence_data, :fda_label, get(evidence_data, "fda_label", false))
    if fda_label
        result["darwin:fdaLabelInfo"] = true
    end

    return result
end

# =============================================================================
# Simulation Result Serialization
# =============================================================================

"""
    to_jsonld_simulation(sim_result::Dict; include_context::Bool=true) -> Dict{String, Any}

Convert simulation result to JSON-LD.
"""
function to_jsonld_simulation(sim_result::Dict; include_context::Bool = true)::Dict{String, Any}
    result = Dict{String, Any}()

    if include_context
        result["@context"] = merge_contexts([:darwin, :parameter, :provenance])["@context"]
    end

    result["@type"] = ["darwin:SimulationResult", "obo:OBI_0000070"]  # assay
    result["@id"] = "darwin:simulation/$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

    # Timestamp
    result["prov:generatedAtTime"] = Dict(
        "@type" => "xsd:dateTime",
        "@value" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SSZ")
    )

    # Time points
    if haskey(sim_result, :time) || haskey(sim_result, "time")
        time = get(sim_result, :time, get(sim_result, "time", []))
        result["darwin:timePoints"] = Dict{String, Any}(
            "@type" => "qudt:QuantityValue",
            "darwin:values" => time,
            "unitRef" => "unit:HR"
        )
    end

    # Concentrations by compartment
    concentrations = Dict{String, Any}()
    for (key, value) in sim_result
        key_str = string(key)
        if key_str != "time" && value isa AbstractVector
            concentrations[key_str] = Dict{String, Any}(
                "@type" => "qudt:QuantityValue",
                "darwin:values" => collect(value),
                "unitRef" => "unit:MilliGM-PER-L"
            )
        end
    end

    if !isempty(concentrations)
        result["darwin:concentrations"] = concentrations
    end

    # PK metrics (if computed)
    if haskey(sim_result, :cmax) || haskey(sim_result, "cmax")
        result["darwin:cmax"] = Dict{String, Any}(
            "@type" => "qudt:QuantityValue",
            "numericValue" => get(sim_result, :cmax, get(sim_result, "cmax", 0.0)),
            "unitRef" => "unit:MilliGM-PER-L"
        )
    end

    if haskey(sim_result, :auc) || haskey(sim_result, "auc")
        result["darwin:auc"] = Dict{String, Any}(
            "@type" => "qudt:QuantityValue",
            "numericValue" => get(sim_result, :auc, get(sim_result, "auc", 0.0)),
            "unitRef" => "unit:MilliGM-HR-PER-L"
        )
    end

    if haskey(sim_result, :half_life) || haskey(sim_result, "half_life")
        result["darwin:halfLife"] = Dict{String, Any}(
            "@type" => "qudt:QuantityValue",
            "numericValue" => get(sim_result, :half_life, get(sim_result, "half_life", 0.0)),
            "unitRef" => "unit:HR"
        )
    end

    return result
end

# =============================================================================
# RDF/Turtle Export
# =============================================================================

"""
    to_turtle(entity::Dict{String, Any}) -> String

Convert JSON-LD entity to RDF/Turtle format.
"""
function to_turtle(entity::Dict{String, Any})::String
    buf = IOBuffer()

    # Prefixes
    println(buf, "@prefix darwin: <https://darwin-pbpk.org/schema/> .")
    println(buf, "@prefix qudt: <http://qudt.org/schema/qudt/> .")
    println(buf, "@prefix unit: <http://qudt.org/vocab/unit/> .")
    println(buf, "@prefix obo: <http://purl.obolibrary.org/obo/> .")
    println(buf, "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
    println(buf, "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .")
    println(buf, "@prefix prov: <http://www.w3.org/ns/prov#> .")
    println(buf, "@prefix schema: <https://schema.org/> .")
    println(buf)

    # Convert entity to triples
    entity_id = get(entity, "@id", "_:entity")
    entity_id = startswith(entity_id, "http") ? "<$entity_id>" : entity_id

    # Types
    types = get(entity, "@type", [])
    if types isa String
        types = [types]
    end

    for t in types
        println(buf, "$entity_id a $t .")
    end

    # Properties (simplified - full implementation would handle nested objects)
    for (key, value) in entity
        if startswith(key, "@")
            continue
        end

        if value isa String
            println(buf, "$entity_id $key \"$value\" .")
        elseif value isa Number
            println(buf, "$entity_id $key $value .")
        end
    end

    return String(take!(buf))
end

# =============================================================================
# Helper Functions
# =============================================================================

function get_drug_name(drug)::String
    if drug isa Symbol
        return string(drug)
    elseif drug isa String
        return drug
    elseif drug isa Dict
        return get(drug, :name, get(drug, "name", "unknown"))
    else
        return "unknown"
    end
end

function mechanism_to_dinto(mechanism)::String
    # Map mechanism types to DINTO IDs
    mapping = Dict(
        :CYP_COMPETITIVE_INHIBITION => "DINTO_0000102",
        :CYP_NONCOMPETITIVE_INHIBITION => "DINTO_0000103",
        :CYP_MECHANISM_BASED_INHIBITION => "DINTO_0000104",
        :CYP_INDUCTION => "DINTO_0000105",
        :TRANSPORTER_INHIBITION => "DINTO_0000106",
        :TRANSPORTER_INDUCTION => "DINTO_0000107",
        :PROTEIN_BINDING_DISPLACEMENT => "DINTO_0000108",
        :RENAL_COMPETITION => "DINTO_0000109",
    )

    mech_sym = mechanism isa Symbol ? mechanism : Symbol(mechanism)
    return get(mapping, mech_sym, "DINTO_0000001")
end

function classify_ddi_severity(auc_ratio::Float64)::String
    if auc_ratio < 1.25 && auc_ratio > 0.8
        return "no_effect"
    elseif auc_ratio < 2.0 && auc_ratio >= 1.25
        return "weak"
    elseif auc_ratio < 5.0 && auc_ratio >= 2.0
        return "moderate"
    elseif auc_ratio >= 5.0
        return "strong"
    elseif auc_ratio <= 0.5
        return "strong_induction"
    elseif auc_ratio <= 0.8
        return "moderate_induction"
    else
        return "unknown"
    end
end

function param_to_pato(param_name::String)::Union{String, Nothing}
    # Map PK parameters to PATO qualities
    mapping = Dict(
        "clearance" => "PATO:0001025",      # rate
        "cl" => "PATO:0001025",
        "volume" => "PATO:0000918",          # volume
        "vd" => "PATO:0000918",
        "v1" => "PATO:0000918",
        "concentration" => "PATO:0000033",   # concentration
        "bioavailability" => "PATO:0000033",
        "fraction" => "PATO:0000033",
        "fu" => "PATO:0000033",
        "half-life" => "PATO:0001309",       # duration
        "t1/2" => "PATO:0001309",
        "rate" => "PATO:0001025",
        "ka" => "PATO:0001025",
    )

    lower_name = lowercase(param_name)
    for (key, pato) in mapping
        if occursin(key, lower_name)
            return pato
        end
    end

    return nothing
end

function evidence_type_to_dideo(evidence_type)::String
    mapping = Dict(
        :in_vitro => "DIDEO_0000001",
        :in_vitro_enzyme => "DIDEO_0000001",
        :in_vitro_hepatocyte => "DIDEO_0000002",
        :clinical => "DIDEO_0000015",
        :clinical_pk_healthy => "DIDEO_0000015",
        :clinical_pk_patient => "DIDEO_0000016",
        :population_pk => "DIDEO_0000020",
        :case_report => "DIDEO_0000025",
        :pbpk_modeling => "DIDEO_0000030",
    )

    et_sym = evidence_type isa Symbol ? evidence_type : Symbol(evidence_type)
    return get(mapping, et_sym, "DIDEO_0000001")
end

"""
    serialize_entity(entity_type::Symbol, data::Dict; format::Symbol=:jsonld) -> String

Serialize an entity to the specified format.

# Arguments
- `entity_type`: One of :drug, :ddi, :parameter, :evidence, :simulation
- `data`: Entity data dictionary
- `format`: Output format (:jsonld or :turtle)

# Returns
- Serialized string
"""
function serialize_entity(entity_type::Symbol, data::Dict; format::Symbol = :jsonld)::String
    serializers = Dict(
        :drug => to_jsonld_drug,
        :ddi => to_jsonld_ddi,
        :parameter => to_jsonld_parameter,
        :evidence => to_jsonld_evidence,
        :simulation => to_jsonld_simulation,
    )

    if !haskey(serializers, entity_type)
        error("Unknown entity type: $entity_type")
    end

    jsonld = serializers[entity_type](data)

    if format == :jsonld
        return JSON.json(jsonld, 2)
    elseif format == :turtle
        return to_turtle(jsonld)
    else
        error("Unknown format: $format. Use :jsonld or :turtle")
    end
end

end # module JSONLDSerializer
