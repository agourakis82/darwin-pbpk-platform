# ===========================================================================
# PUBCHEM API CLIENT
# ===========================================================================
# REST API client for fetching compound properties from PubChem
# Documentation: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
# ===========================================================================

module PubChemClient

using HTTP
using JSON

export fetch_compound_properties, fetch_compound_by_name, fetch_compound_by_cid
export PubChemCompound, search_compounds_by_name

const PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

"""
Compound data structure returned from PubChem
"""
struct PubChemCompound
    cid::Int
    name::String
    molecular_formula::String
    molecular_weight::Float64
    xlogp::Union{Float64, Nothing}      # Computed XLogP3
    hbd::Int                             # H-bond donors
    hba::Int                             # H-bond acceptors
    tpsa::Float64                        # Topological polar surface area
    rotatable_bonds::Int
    exact_mass::Float64
    canonical_smiles::String
    iupac_name::String
end

"""
Fetch compound properties by PubChem CID

# Arguments
- `cid::Int`: PubChem Compound ID

# Returns
- `PubChemCompound` with molecular properties
"""
function fetch_compound_by_cid(cid::Int)
    url = "$PUBCHEM_BASE_URL/compound/cid/$cid/property/MolecularFormula,MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,TPSA,RotatableBondCount,ExactMass,CanonicalSMILES,IUPACName/JSON"

    response = HTTP.get(url; status_exception=false)

    if response.status != 200
        error("PubChem API error for CID $cid: HTTP $(response.status)")
    end

    data = JSON.parse(String(response.body))
    props = data["PropertyTable"]["Properties"][1]

    # Get compound name separately
    name_url = "$PUBCHEM_BASE_URL/compound/cid/$cid/synonyms/JSON"
    name_response = HTTP.get(name_url; status_exception=false)
    name = "Unknown"
    if name_response.status == 200
        name_data = JSON.parse(String(name_response.body))
        synonyms = get(name_data["InformationList"]["Information"][1], "Synonym", String[])
        if !isempty(synonyms)
            name = synonyms[1]
        end
    end

    # Parse values, handling string returns from API
    parse_float(x) = x isa Number ? Float64(x) : parse(Float64, string(x))
    parse_int(x) = x isa Number ? Int(x) : parse(Int, string(x))

    mw = get(props, "MolecularWeight", nothing)
    xlogp = get(props, "XLogP", nothing)
    tpsa = get(props, "TPSA", nothing)
    exact_mass = get(props, "ExactMass", nothing)

    return PubChemCompound(
        cid,
        name,
        get(props, "MolecularFormula", ""),
        mw === nothing ? 0.0 : parse_float(mw),
        xlogp === nothing ? nothing : parse_float(xlogp),
        parse_int(get(props, "HBondDonorCount", 0)),
        parse_int(get(props, "HBondAcceptorCount", 0)),
        tpsa === nothing ? 0.0 : parse_float(tpsa),
        parse_int(get(props, "RotatableBondCount", 0)),
        exact_mass === nothing ? 0.0 : parse_float(exact_mass),
        get(props, "CanonicalSMILES", ""),
        get(props, "IUPACName", "")
    )
end

"""
Search for compound by name and fetch properties

# Arguments
- `name::String`: Drug/compound name

# Returns
- `PubChemCompound` or `nothing` if not found
"""
function fetch_compound_by_name(name::String)
    # First, search for the CID
    search_url = "$PUBCHEM_BASE_URL/compound/name/$(HTTP.escapeuri(name))/cids/JSON"

    response = HTTP.get(search_url; status_exception=false)

    if response.status != 200
        @warn "Compound not found in PubChem: $name"
        return nothing
    end

    data = JSON.parse(String(response.body))
    cids = get(data["IdentifierList"], "CID", Int[])

    if isempty(cids)
        return nothing
    end

    # Fetch properties for the first (most relevant) CID
    return fetch_compound_by_cid(cids[1])
end

"""
Batch fetch multiple compounds by name

# Arguments
- `names::Vector{String}`: List of drug names

# Returns
- `Dict{String, PubChemCompound}`: Map of name to compound data
"""
function fetch_compounds_batch(names::Vector{String}; delay_ms::Int=200)
    results = Dict{String, Union{PubChemCompound, Nothing}}()

    for (i, name) in enumerate(names)
        println("[$i/$(length(names))] Fetching: $name")
        try
            results[name] = fetch_compound_by_name(name)
            sleep(delay_ms / 1000)  # Rate limiting
        catch e
            @warn "Error fetching $name: $e"
            results[name] = nothing
        end
    end

    return results
end

"""
Fetch compound properties with specific fields

# Arguments
- `cid::Int`: PubChem CID
- `properties::Vector{String}`: List of property names

# Returns
- `Dict` with requested properties
"""
function fetch_compound_properties(cid::Int, properties::Vector{String})
    props_str = join(properties, ",")
    url = "$PUBCHEM_BASE_URL/compound/cid/$cid/property/$props_str/JSON"

    response = HTTP.get(url; status_exception=false)

    if response.status != 200
        error("PubChem API error: HTTP $(response.status)")
    end

    data = JSON.parse(String(response.body))
    return data["PropertyTable"]["Properties"][1]
end

"""
Search compounds by structural similarity (for finding analogs)

# Arguments
- `smiles::String`: Query SMILES
- `threshold::Float64`: Tanimoto similarity threshold (0-1)
- `max_results::Int`: Maximum number of results

# Returns
- `Vector{Int}`: List of similar compound CIDs
"""
function search_similar_compounds(smiles::String; threshold::Float64=0.9, max_results::Int=10)
    url = "$PUBCHEM_BASE_URL/compound/similarity/smiles/$(HTTP.escapeuri(smiles))/cids/JSON?Threshold=$threshold&MaxRecords=$max_results"

    response = HTTP.get(url; status_exception=false)

    if response.status != 200
        return Int[]
    end

    data = JSON.parse(String(response.body))
    return get(data["IdentifierList"], "CID", Int[])
end

end # module
