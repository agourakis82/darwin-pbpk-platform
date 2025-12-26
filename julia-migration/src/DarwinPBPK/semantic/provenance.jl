# =============================================================================
# PROV-O Provenance - Darwin PBPK Semantic Layer
# =============================================================================
# Implements W3C PROV-O provenance ontology for tracking prediction lineage.
#
# Provides:
# - Entity provenance (predictions, datasets, models)
# - Activity tracking (simulations, training runs)
# - Agent attribution (software, algorithms, authors)
# - Derivation chains (data → model → prediction)
#
# Reference: https://www.w3.org/TR/prov-o/
#
# Author: Dr. Sounio Agourakis
# Date: December 2025
# =============================================================================

module Provenance

using Dates
using JSON

export ProvenanceRecord, ProvenanceActivity, ProvenanceAgent
export create_prediction_provenance, create_simulation_provenance
export add_derivation, add_attribution, to_jsonld
export DARWIN_SOFTWARE_AGENT, DARWIN_PBPK_MODEL

# =============================================================================
# PROV-O Structures
# =============================================================================

"""
    ProvenanceAgent

Represents a PROV-O Agent (software, person, or organization).
"""
struct ProvenanceAgent
    iri::String
    type::Symbol           # :software, :person, :organization
    name::String
    version::Union{String, Nothing}
    description::Union{String, Nothing}
end

# Convenience constructor
function ProvenanceAgent(iri::String, type::Symbol, name::String)
    ProvenanceAgent(iri, type, name, nothing, nothing)
end

"""
    ProvenanceActivity

Represents a PROV-O Activity (simulation, training, prediction).
"""
struct ProvenanceActivity
    iri::String
    type::Symbol           # :simulation, :prediction, :training, :validation
    started_at::DateTime
    ended_at::Union{DateTime, Nothing}
    used_entities::Vector{String}       # IRIs of used entities
    associated_agents::Vector{String}   # IRIs of associated agents
    description::Union{String, Nothing}
    parameters::Dict{String, Any}       # Activity parameters
end

"""
    ProvenanceRecord

Complete provenance record for an entity.
"""
mutable struct ProvenanceRecord
    entity_iri::String
    entity_type::Symbol    # :prediction, :simulation, :dataset, :model
    generated_at::DateTime
    was_derived_from::Vector{String}      # Source entity IRIs
    was_attributed_to::Vector{ProvenanceAgent}
    was_generated_by::Union{ProvenanceActivity, Nothing}
    invalidated_at::Union{DateTime, Nothing}
    metadata::Dict{String, Any}
end

# Convenience constructor
function ProvenanceRecord(entity_iri::String, entity_type::Symbol)
    ProvenanceRecord(
        entity_iri,
        entity_type,
        now(),
        String[],
        ProvenanceAgent[],
        nothing,
        nothing,
        Dict{String, Any}()
    )
end

# =============================================================================
# Pre-defined Agents
# =============================================================================

"""
Darwin PBPK Platform software agent.
"""
const DARWIN_SOFTWARE_AGENT = ProvenanceAgent(
    "darwin:agent/darwin_pbpk",
    :software,
    "Darwin PBPK Platform",
    "2.5.0",
    "AI-powered PBPK prediction platform with Julia implementation"
)

"""
Darwin PBPK Model agent (for model-based predictions).
"""
const DARWIN_PBPK_MODEL = ProvenanceAgent(
    "darwin:agent/pbpk_model",
    :software,
    "Darwin PBPK ODE Model",
    "1.0.0",
    "14-compartment physiologically-based pharmacokinetic model"
)

"""
Darwin Dynamic GNN agent.
"""
const DARWIN_GNN_AGENT = ProvenanceAgent(
    "darwin:agent/dynamic_gnn",
    :software,
    "Darwin Dynamic GNN",
    "1.0.0",
    "Graph neural network for dynamic PBPK simulation"
)

"""
Darwin DDI Prediction agent.
"""
const DARWIN_DDI_AGENT = ProvenanceAgent(
    "darwin:agent/ddi_predictor",
    :software,
    "Darwin DDI Prediction Engine",
    "1.0.0",
    "Bayesian drug-drug interaction prediction model"
)

# =============================================================================
# Provenance Creation Functions
# =============================================================================

"""
    create_prediction_provenance(
        entity_iri::String,
        model_type::Symbol,
        input_data::Dict;
        kwargs...
    ) -> ProvenanceRecord

Create a provenance record for a PBPK prediction.

# Arguments
- `entity_iri`: IRI for the prediction result
- `model_type`: Type of model used (:pbpk, :gnn, :ddi)
- `input_data`: Dictionary of input parameters
- `start_time`: When prediction started (optional)
- `end_time`: When prediction ended (optional)
- `derived_from`: IRIs of source entities (optional)

# Returns
- ProvenanceRecord with complete attribution
"""
function create_prediction_provenance(
    entity_iri::String,
    model_type::Symbol,
    input_data::Dict;
    start_time::DateTime = now(),
    end_time::Union{DateTime, Nothing} = nothing,
    derived_from::Vector{String} = String[]
)::ProvenanceRecord

    # Select appropriate agent
    agent = if model_type == :pbpk
        DARWIN_PBPK_MODEL
    elseif model_type == :gnn
        DARWIN_GNN_AGENT
    elseif model_type == :ddi
        DARWIN_DDI_AGENT
    else
        DARWIN_SOFTWARE_AGENT
    end

    # Create activity
    activity = ProvenanceActivity(
        "darwin:activity/prediction_$(Dates.format(start_time, "yyyymmdd_HHMMSS"))",
        :prediction,
        start_time,
        end_time !== nothing ? end_time : now(),
        derived_from,
        [agent.iri, DARWIN_SOFTWARE_AGENT.iri],
        "PBPK prediction using $model_type model",
        input_data
    )

    # Create record
    record = ProvenanceRecord(entity_iri, :prediction)
    record.was_derived_from = derived_from
    record.was_attributed_to = [agent, DARWIN_SOFTWARE_AGENT]
    record.was_generated_by = activity
    record.metadata = input_data

    return record
end

"""
    create_simulation_provenance(
        entity_iri::String,
        simulation_params::Dict;
        kwargs...
    ) -> ProvenanceRecord

Create a provenance record for a PBPK simulation.
"""
function create_simulation_provenance(
    entity_iri::String,
    simulation_params::Dict;
    start_time::DateTime = now(),
    end_time::Union{DateTime, Nothing} = nothing,
    drug_iri::Union{String, Nothing} = nothing
)::ProvenanceRecord

    derived_from = String[]
    if drug_iri !== nothing
        push!(derived_from, drug_iri)
    end

    # Add model/database sources
    push!(derived_from, "darwin:model/pbpk_14compartment_v1.0")

    activity = ProvenanceActivity(
        "darwin:activity/simulation_$(Dates.format(start_time, "yyyymmdd_HHMMSS"))",
        :simulation,
        start_time,
        end_time !== nothing ? end_time : now(),
        derived_from,
        [DARWIN_PBPK_MODEL.iri, DARWIN_SOFTWARE_AGENT.iri],
        "PBPK ODE simulation",
        simulation_params
    )

    record = ProvenanceRecord(entity_iri, :simulation)
    record.was_derived_from = derived_from
    record.was_attributed_to = [DARWIN_PBPK_MODEL, DARWIN_SOFTWARE_AGENT]
    record.was_generated_by = activity
    record.metadata = simulation_params

    return record
end

"""
    create_ddi_provenance(
        entity_iri::String,
        perpetrator_iri::String,
        victim_iri::String,
        mechanism::Symbol;
        kwargs...
    ) -> ProvenanceRecord

Create a provenance record for a DDI prediction.
"""
function create_ddi_provenance(
    entity_iri::String,
    perpetrator_iri::String,
    victim_iri::String,
    mechanism::Symbol;
    start_time::DateTime = now(),
    evidence_sources::Vector{String} = String[]
)::ProvenanceRecord

    derived_from = [perpetrator_iri, victim_iri]
    append!(derived_from, evidence_sources)
    push!(derived_from, "darwin:database/ddi_knowledge_base_v1.0")

    activity = ProvenanceActivity(
        "darwin:activity/ddi_prediction_$(Dates.format(start_time, "yyyymmdd_HHMMSS"))",
        :prediction,
        start_time,
        now(),
        derived_from,
        [DARWIN_DDI_AGENT.iri, DARWIN_SOFTWARE_AGENT.iri],
        "Bayesian DDI prediction for $mechanism mechanism",
        Dict{String, Any}(
            "mechanism" => string(mechanism),
            "perpetrator" => perpetrator_iri,
            "victim" => victim_iri
        )
    )

    record = ProvenanceRecord(entity_iri, :prediction)
    record.was_derived_from = derived_from
    record.was_attributed_to = [DARWIN_DDI_AGENT, DARWIN_SOFTWARE_AGENT]
    record.was_generated_by = activity

    return record
end

# =============================================================================
# Provenance Modification
# =============================================================================

"""
    add_derivation(record::ProvenanceRecord, source_iri::String)

Add a derivation source to the provenance record.
"""
function add_derivation(record::ProvenanceRecord, source_iri::String)
    if !(source_iri in record.was_derived_from)
        push!(record.was_derived_from, source_iri)
    end
end

"""
    add_attribution(record::ProvenanceRecord, agent::ProvenanceAgent)

Add an agent attribution to the provenance record.
"""
function add_attribution(record::ProvenanceRecord, agent::ProvenanceAgent)
    if !any(a -> a.iri == agent.iri, record.was_attributed_to)
        push!(record.was_attributed_to, agent)
    end
end

# =============================================================================
# JSON-LD Serialization
# =============================================================================

"""
    to_jsonld(agent::ProvenanceAgent) -> Dict{String, Any}

Convert ProvenanceAgent to JSON-LD.
"""
function to_jsonld(agent::ProvenanceAgent)::Dict{String, Any}
    result = Dict{String, Any}(
        "@id" => agent.iri,
        "rdfs:label" => agent.name
    )

    # Type based on agent type
    if agent.type == :software
        result["@type"] = ["prov:Agent", "prov:SoftwareAgent"]
    elseif agent.type == :person
        result["@type"] = ["prov:Agent", "prov:Person"]
    elseif agent.type == :organization
        result["@type"] = ["prov:Agent", "prov:Organization"]
    else
        result["@type"] = "prov:Agent"
    end

    if agent.version !== nothing
        result["darwin:version"] = agent.version
    end

    if agent.description !== nothing
        result["rdfs:comment"] = agent.description
    end

    return result
end

"""
    to_jsonld(activity::ProvenanceActivity) -> Dict{String, Any}

Convert ProvenanceActivity to JSON-LD.
"""
function to_jsonld(activity::ProvenanceActivity)::Dict{String, Any}
    result = Dict{String, Any}(
        "@type" => "prov:Activity",
        "@id" => activity.iri,
        "prov:startedAtTime" => Dict(
            "@type" => "xsd:dateTime",
            "@value" => Dates.format(activity.started_at, "yyyy-mm-ddTHH:MM:SSZ")
        )
    )

    if activity.ended_at !== nothing
        result["prov:endedAtTime"] = Dict(
            "@type" => "xsd:dateTime",
            "@value" => Dates.format(activity.ended_at, "yyyy-mm-ddTHH:MM:SSZ")
        )
    end

    # Used entities
    if !isempty(activity.used_entities)
        result["prov:used"] = [Dict("@id" => iri) for iri in activity.used_entities]
    end

    # Associated agents
    if !isempty(activity.associated_agents)
        result["prov:wasAssociatedWith"] = [Dict("@id" => iri) for iri in activity.associated_agents]
    end

    if activity.description !== nothing
        result["rdfs:comment"] = activity.description
    end

    # Activity type mapping
    activity_types = Dict(
        :simulation => "darwin:Simulation",
        :prediction => "darwin:Prediction",
        :training => "darwin:ModelTraining",
        :validation => "darwin:Validation"
    )

    if haskey(activity_types, activity.type)
        current_type = result["@type"]
        result["@type"] = [current_type, activity_types[activity.type]]
    end

    # Parameters as qualified usage
    if !isempty(activity.parameters)
        result["darwin:parameters"] = activity.parameters
    end

    return result
end

"""
    to_jsonld(record::ProvenanceRecord) -> Dict{String, Any}

Convert ProvenanceRecord to JSON-LD.
"""
function to_jsonld(record::ProvenanceRecord)::Dict{String, Any}
    result = Dict{String, Any}(
        "@type" => "prov:Entity",
        "@id" => record.entity_iri,
        "prov:generatedAtTime" => Dict(
            "@type" => "xsd:dateTime",
            "@value" => Dates.format(record.generated_at, "yyyy-mm-ddTHH:MM:SSZ")
        )
    )

    # Entity type mapping
    entity_types = Dict(
        :prediction => "darwin:Prediction",
        :simulation => "darwin:SimulationResult",
        :dataset => "darwin:Dataset",
        :model => "darwin:Model"
    )

    if haskey(entity_types, record.entity_type)
        current_type = result["@type"]
        result["@type"] = [current_type, entity_types[record.entity_type]]
    end

    # Derivations
    if !isempty(record.was_derived_from)
        result["prov:wasDerivedFrom"] = [Dict("@id" => iri) for iri in record.was_derived_from]
    end

    # Attributions
    if !isempty(record.was_attributed_to)
        result["prov:wasAttributedTo"] = [to_jsonld(agent) for agent in record.was_attributed_to]
    end

    # Generation activity
    if record.was_generated_by !== nothing
        result["prov:wasGeneratedBy"] = to_jsonld(record.was_generated_by)
    end

    # Invalidation
    if record.invalidated_at !== nothing
        result["prov:invalidatedAtTime"] = Dict(
            "@type" => "xsd:dateTime",
            "@value" => Dates.format(record.invalidated_at, "yyyy-mm-ddTHH:MM:SSZ")
        )
    end

    # Additional metadata
    if !isempty(record.metadata)
        result["darwin:metadata"] = record.metadata
    end

    return result
end

"""
    to_jsonld_with_context(record::ProvenanceRecord) -> Dict{String, Any}

Convert ProvenanceRecord to JSON-LD with full context.
"""
function to_jsonld_with_context(record::ProvenanceRecord)::Dict{String, Any}
    result = to_jsonld(record)

    result["@context"] = Dict{String, Any}(
        "prov" => "http://www.w3.org/ns/prov#",
        "darwin" => "https://darwin-pbpk.org/schema/",
        "rdfs" => "http://www.w3.org/2000/01/rdf-schema#",
        "xsd" => "http://www.w3.org/2001/XMLSchema#",

        # PROV-O terms
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
        "invalidatedAtTime" => Dict("@id" => "prov:invalidatedAtTime", "@type" => "xsd:dateTime"),
    )

    return result
end

# =============================================================================
# Utility Functions
# =============================================================================

"""
    generate_entity_iri(entity_type::Symbol, identifier::String) -> String

Generate a Darwin IRI for an entity.
"""
function generate_entity_iri(entity_type::Symbol, identifier::String)::String
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    return "darwin:$(entity_type)/$(identifier)_$(timestamp)"
end

"""
    create_pubmed_reference(pubmed_id::Int) -> Dict{String, Any}

Create a reference to a PubMed article.
"""
function create_pubmed_reference(pubmed_id::Int)::Dict{String, Any}
    return Dict{String, Any}(
        "@type" => "darwin:LiteratureReference",
        "@id" => "https://pubmed.ncbi.nlm.nih.gov/$pubmed_id/",
        "darwin:pubmedId" => pubmed_id
    )
end

end # module Provenance
