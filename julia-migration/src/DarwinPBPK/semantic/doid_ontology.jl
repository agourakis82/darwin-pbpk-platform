"""
DOID Ontology Integration - Disease Ontology for Darwin PBPK

Provides access to the Disease Ontology (DOID) for:
- Disease identification and classification
- Disease-drug indication mappings
- ICD-10, MESH, OMIM cross-references
- Disease hierarchy navigation

DOID Version: 2025-11-25 (downloaded from OBO Foundry)
Reference: https://disease-ontology.org/
"""
module DOIDOntology

using JSON

export DOIDTerm, DOIDIndex
export load_doid, search_doid, get_disease_by_id, get_disease_by_name
export get_disease_xrefs, get_disease_hierarchy, get_diseases_for_drug_class
export DOID_IRI_PREFIX

# DOID IRI prefix
const DOID_IRI_PREFIX = "http://purl.obolibrary.org/obo/DOID_"

"""
    DOIDTerm

Represents a Disease Ontology term with full metadata.

Fields:
- id: DOID identifier (e.g., "DOID:9352" for diabetes mellitus)
- name: Disease name
- definition: Full definition text
- synonyms: Alternative names (exact, related, narrow, broad)
- xrefs: Cross-references to ICD-10, MESH, OMIM, etc.
- parents: Parent term IDs (is_a relationships)
- children: Child term IDs
- subset: Ontology subsets this term belongs to
"""
struct DOIDTerm
    id::String
    name::String
    definition::Union{String, Nothing}
    synonyms::Dict{String, Vector{String}}  # type => [synonyms]
    xrefs::Dict{String, Vector{String}}     # source => [ids]
    parents::Vector{String}
    children::Vector{String}
    subset::Vector{String}
    is_obsolete::Bool
end

"""
    DOIDIndex

In-memory index of the Disease Ontology for fast lookups.
"""
struct DOIDIndex
    terms::Dict{String, DOIDTerm}           # id => term
    name_index::Dict{String, String}        # lowercase name => id
    xref_index::Dict{String, Vector{String}} # xref => [ids]
    hierarchy::Dict{String, Vector{String}}  # id => child ids
    version::String
end

"""
    load_doid(json_path::String) -> DOIDIndex

Load DOID from JSON format (most efficient for Julia).
"""
function load_doid(json_path::String)::DOIDIndex
    @info "Loading DOID from $json_path..."

    data = JSON.parsefile(json_path)

    terms = Dict{String, DOIDTerm}()
    name_index = Dict{String, String}()
    xref_index = Dict{String, Vector{String}}()
    hierarchy = Dict{String, Vector{String}}()

    # Extract version from metadata
    version = get(get(data, "meta", Dict()), "version", "unknown")

    # Parse graph nodes
    graphs = get(data, "graphs", [])
    if isempty(graphs)
        @warn "No graphs found in DOID JSON"
        return DOIDIndex(terms, name_index, xref_index, hierarchy, version)
    end

    nodes = get(graphs[1], "nodes", [])
    edges = get(graphs[1], "edges", [])

    @info "Parsing $(length(nodes)) nodes..."

    for node in nodes
        id_raw = get(node, "id", "")

        # Only process DOID terms
        if !occursin("DOID_", id_raw)
            continue
        end

        # Convert to CURIE format
        id = replace(id_raw, "http://purl.obolibrary.org/obo/DOID_" => "DOID:")

        name = get(node, "lbl", "")
        if isempty(name)
            continue
        end

        # Parse metadata
        meta = get(node, "meta", Dict())

        definition = nothing
        def_obj = get(meta, "definition", nothing)
        if def_obj !== nothing
            definition = get(def_obj, "val", nothing)
        end

        # Parse synonyms
        synonyms = Dict{String, Vector{String}}()
        for syn in get(meta, "synonyms", [])
            syn_type = get(syn, "pred", "hasExactSynonym")
            syn_val = get(syn, "val", "")
            if !isempty(syn_val)
                key = replace(syn_type, "has" => "", "Synonym" => "")
                if !haskey(synonyms, key)
                    synonyms[key] = String[]
                end
                push!(synonyms[key], syn_val)
            end
        end

        # Parse xrefs
        xrefs = Dict{String, Vector{String}}()
        for xref in get(meta, "xrefs", [])
            xref_val = get(xref, "val", "")
            if occursin(":", xref_val)
                parts = split(xref_val, ":", limit=2)
                source = String(parts[1])
                ref_id = String(parts[2])
                if !haskey(xrefs, source)
                    xrefs[source] = String[]
                end
                push!(xrefs[source], ref_id)

                # Index xrefs
                if !haskey(xref_index, xref_val)
                    xref_index[xref_val] = String[]
                end
                push!(xref_index[xref_val], id)
            end
        end

        # Check obsolete
        is_obsolete = get(meta, "deprecated", false)

        # Parse subsets
        subset = String[]
        for s in get(meta, "subsets", [])
            push!(subset, String(s))
        end

        term = DOIDTerm(
            id,
            name,
            definition,
            synonyms,
            xrefs,
            String[],  # parents - filled from edges
            String[],  # children - filled from edges
            subset,
            is_obsolete
        )

        terms[id] = term
        name_index[lowercase(name)] = id

        # Index synonyms
        for (_, syns) in synonyms
            for syn in syns
                name_index[lowercase(syn)] = id
            end
        end
    end

    @info "Parsing $(length(edges)) edges..."

    # Build hierarchy from edges
    parents_map = Dict{String, Vector{String}}()
    children_map = Dict{String, Vector{String}}()

    for edge in edges
        sub = get(edge, "sub", "")
        obj = get(edge, "obj", "")
        pred = get(edge, "pred", "")

        if pred == "is_a" && occursin("DOID_", sub) && occursin("DOID_", obj)
            child_id = replace(sub, "http://purl.obolibrary.org/obo/DOID_" => "DOID:")
            parent_id = replace(obj, "http://purl.obolibrary.org/obo/DOID_" => "DOID:")

            if !haskey(parents_map, child_id)
                parents_map[child_id] = String[]
            end
            push!(parents_map[child_id], parent_id)

            if !haskey(children_map, parent_id)
                children_map[parent_id] = String[]
            end
            push!(children_map[parent_id], child_id)
        end
    end

    # Update terms with hierarchy
    for (id, term) in terms
        parents = get(parents_map, id, String[])
        children = get(children_map, id, String[])

        terms[id] = DOIDTerm(
            term.id,
            term.name,
            term.definition,
            term.synonyms,
            term.xrefs,
            parents,
            children,
            term.subset,
            term.is_obsolete
        )

        hierarchy[id] = children
    end

    @info "Loaded $(length(terms)) DOID terms"

    return DOIDIndex(terms, name_index, xref_index, hierarchy, version)
end

"""
    search_doid(index::DOIDIndex, query::String; limit=10) -> Vector{DOIDTerm}

Search DOID by name or synonym (case-insensitive).
"""
function search_doid(index::DOIDIndex, query::String; limit::Int=10)::Vector{DOIDTerm}
    query_lower = lowercase(query)
    results = DOIDTerm[]

    for (name, id) in index.name_index
        if occursin(query_lower, name)
            term = get(index.terms, id, nothing)
            if term !== nothing && !term.is_obsolete
                push!(results, term)
                if length(results) >= limit
                    break
                end
            end
        end
    end

    return results
end

"""
    get_disease_by_id(index::DOIDIndex, doid::String) -> Union{DOIDTerm, Nothing}

Get disease by DOID (e.g., "DOID:9352" for diabetes mellitus).
"""
function get_disease_by_id(index::DOIDIndex, doid::String)::Union{DOIDTerm, Nothing}
    return get(index.terms, doid, nothing)
end

"""
    get_disease_by_name(index::DOIDIndex, name::String) -> Union{DOIDTerm, Nothing}

Get disease by exact name match (case-insensitive).
"""
function get_disease_by_name(index::DOIDIndex, name::String)::Union{DOIDTerm, Nothing}
    id = get(index.name_index, lowercase(name), nothing)
    if id !== nothing
        return get(index.terms, id, nothing)
    end
    return nothing
end

"""
    get_disease_xrefs(term::DOIDTerm, source::String) -> Vector{String}

Get cross-references for a specific source (ICD10CM, MESH, OMIM, etc.).
"""
function get_disease_xrefs(term::DOIDTerm, source::String)::Vector{String}
    return get(term.xrefs, source, String[])
end

"""
    get_disease_hierarchy(index::DOIDIndex, doid::String; depth=2) -> Dict

Get disease hierarchy (ancestors and descendants).
"""
function get_disease_hierarchy(index::DOIDIndex, doid::String; depth::Int=2)::Dict{String, Any}
    term = get_disease_by_id(index, doid)
    if term === nothing
        return Dict{String, Any}()
    end

    # Get ancestors
    ancestors = String[]
    current_parents = term.parents
    for _ in 1:depth
        if isempty(current_parents)
            break
        end
        append!(ancestors, current_parents)
        next_parents = String[]
        for p in current_parents
            parent_term = get(index.terms, p, nothing)
            if parent_term !== nothing
                append!(next_parents, parent_term.parents)
            end
        end
        current_parents = next_parents
    end

    # Get descendants
    descendants = String[]
    current_children = term.children
    for _ in 1:depth
        if isempty(current_children)
            break
        end
        append!(descendants, current_children)
        next_children = String[]
        for c in current_children
            child_term = get(index.terms, c, nothing)
            if child_term !== nothing
                append!(next_children, child_term.children)
            end
        end
        current_children = next_children
    end

    return Dict{String, Any}(
        "term" => term,
        "ancestors" => unique(ancestors),
        "descendants" => unique(descendants)
    )
end

"""
    get_diseases_for_drug_class(index::DOIDIndex, drug_class::String) -> Vector{DOIDTerm}

Get diseases commonly treated by a drug class.
This is a placeholder - full implementation requires drug-disease mappings.
"""
function get_diseases_for_drug_class(index::DOIDIndex, drug_class::String)::Vector{DOIDTerm}
    # Common drug class to disease mappings (subset for PBPK relevance)
    drug_disease_map = Dict(
        "antidiabetic" => ["diabetes mellitus", "type 2 diabetes mellitus", "type 1 diabetes mellitus"],
        "antihypertensive" => ["hypertensive disease", "essential hypertension"],
        "statin" => ["hypercholesterolemia", "familial hypercholesterolemia"],
        "anticoagulant" => ["thrombosis", "venous thromboembolism"],
        "antibiotic" => ["bacterial infectious disease"],
        "antiviral" => ["viral infectious disease"],
        "antineoplastic" => ["cancer", "malignant neoplasm"],
        "immunosuppressant" => ["autoimmune disease"],
        "analgesic" => ["pain disorder"],
        "antidepressant" => ["depressive disorder", "major depressive disorder"],
    )

    diseases = String[]
    drug_class_lower = lowercase(drug_class)

    for (class, disease_names) in drug_disease_map
        if occursin(drug_class_lower, class) || occursin(class, drug_class_lower)
            append!(diseases, disease_names)
        end
    end

    results = DOIDTerm[]
    for disease_name in diseases
        term = get_disease_by_name(index, disease_name)
        if term !== nothing
            push!(results, term)
        end
    end

    return results
end

"""
    to_jsonld(term::DOIDTerm) -> Dict

Convert DOID term to JSON-LD format.
"""
function to_jsonld(term::DOIDTerm)::Dict{String, Any}
    jsonld = Dict{String, Any}(
        "@context" => Dict(
            "DOID" => "http://purl.obolibrary.org/obo/DOID_",
            "obo" => "http://purl.obolibrary.org/obo/",
            "rdfs" => "http://www.w3.org/2000/01/rdf-schema#",
            "oboInOwl" => "http://www.geneontology.org/formats/oboInOwl#"
        ),
        "@id" => "DOID:$(split(term.id, ':')[2])",
        "@type" => "obo:DOID_4",  # disease
        "rdfs:label" => term.name
    )

    if term.definition !== nothing
        jsonld["obo:IAO_0000115"] = term.definition  # definition
    end

    # Add xrefs
    if !isempty(term.xrefs)
        xrefs = String[]
        for (source, ids) in term.xrefs
            for id in ids
                push!(xrefs, "$source:$id")
            end
        end
        jsonld["oboInOwl:hasDbXref"] = xrefs
    end

    # Add synonyms
    if haskey(term.synonyms, "Exact")
        jsonld["oboInOwl:hasExactSynonym"] = term.synonyms["Exact"]
    end

    return jsonld
end

end # module
