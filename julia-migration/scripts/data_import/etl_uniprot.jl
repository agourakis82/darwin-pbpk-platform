# =============================================================================
# ETL PIPELINE: UniProt -> DDI Ontology
# =============================================================================
# Transforms downloaded UniProt CYP450 enzyme data into Darwin PBPK ontology
# =============================================================================

using JSON3
using Dates

const RAW_DATA_DIR = joinpath(@__DIR__, "raw_data", "uniprot")
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "src", "DarwinPBPK", "medlang", "databases", "generated")

mkpath(OUTPUT_DIR)

# CYP enzyme UniProt IDs
const CYP_UNIPROT_IDS = Dict(
    "CYP3A4" => "P08684",
    "CYP2D6" => "P10635",
    "CYP2C9" => "P11712",
    "CYP2C19" => "P33261",
    "CYP1A2" => "P05177",
    "CYP2C8" => "P10632",
    "CYP2B6" => "P20813",
    "CYP2E1" => "P05181"
)

# =============================================================================
# EXTRACT: Load UniProt JSON files
# =============================================================================

"""
Load UniProt JSON file for a CYP enzyme.
"""
function load_uniprot_enzyme(enzyme::String, uniprot_id::String)
    json_path = joinpath(RAW_DATA_DIR, "$(enzyme)_$(uniprot_id).json")

    if !isfile(json_path)
        @warn "UniProt file not found: $json_path"
        return nothing
    end

    return JSON3.read(read(json_path, String))
end

"""
Load all CYP enzyme data.
"""
function load_all_cyp_enzymes()
    enzymes = Dict{String, Any}()

    for (enzyme, uniprot_id) in CYP_UNIPROT_IDS
        data = load_uniprot_enzyme(enzyme, uniprot_id)
        if !isnothing(data)
            enzymes[enzyme] = data
        end
    end

    return enzymes
end

# =============================================================================
# TRANSFORM: Extract enzyme properties
# =============================================================================

"""
Extract tissue expression data from UniProt.
"""
function extract_tissue_expression(data)
    tissues = String[]

    # Look in comments for tissue specificity
    comments = get(data, :comments, nothing)
    if !isnothing(comments)
        for comment in comments
            comment_type = get(comment, :commentType, "")
            if comment_type == "TISSUE SPECIFICITY"
                texts = get(comment, :texts, nothing)
                if !isnothing(texts)
                    for text in texts
                        value = get(text, :value, "")
                        push!(tissues, string(value))
                    end
                end
            end
        end
    end

    return tissues
end

"""
Extract subcellular location from UniProt.
"""
function extract_subcellular_location(data)
    locations = String[]

    comments = get(data, :comments, nothing)
    if !isnothing(comments)
        for comment in comments
            comment_type = get(comment, :commentType, "")
            if comment_type == "SUBCELLULAR LOCATION"
                sublocs = get(comment, :subcellularLocations, nothing)
                if !isnothing(sublocs)
                    for subloc in sublocs
                        loc = get(subloc, :location, nothing)
                        if !isnothing(loc)
                            value = get(loc, :value, "")
                            !isempty(string(value)) && push!(locations, string(value))
                        end
                    end
                end
            end
        end
    end

    return unique(locations)
end

"""
Extract catalytic activity data from UniProt.
"""
function extract_catalytic_activity(data)
    activities = String[]

    comments = get(data, :comments, nothing)
    if !isnothing(comments)
        for comment in comments
            comment_type = get(comment, :commentType, "")
            if comment_type == "CATALYTIC ACTIVITY"
                reaction = get(comment, :reaction, nothing)
                if !isnothing(reaction)
                    name = get(reaction, :name, "")
                    !isempty(string(name)) && push!(activities, string(name))
                end
            end
        end
    end

    return activities
end

"""
Extract function description from UniProt.
"""
function extract_function(data)
    func_texts = String[]

    comments = get(data, :comments, nothing)
    if !isnothing(comments)
        for comment in comments
            comment_type = get(comment, :commentType, "")
            if comment_type == "FUNCTION"
                texts = get(comment, :texts, nothing)
                if !isnothing(texts)
                    for text in texts
                        value = get(text, :value, "")
                        !isempty(string(value)) && push!(func_texts, string(value))
                    end
                end
            end
        end
    end

    return join(func_texts, " ")
end

"""
Extract protein sequence info from UniProt.
"""
function extract_sequence_info(data)
    seq = get(data, :sequence, nothing)
    if isnothing(seq)
        return (length=0, mass=0.0)
    end

    length_val = get(seq, :length, 0)
    mass_val = get(seq, :molWeight, 0)

    return (
        length = length_val,
        mass = mass_val / 1000.0  # Convert to kDa
    )
end

"""
Extract cross-references from UniProt.
"""
function extract_xrefs(data)
    xrefs = Dict{String, Vector{String}}()

    uniprotxrefs = get(data, :uniProtKBCrossReferences, nothing)
    if !isnothing(uniprotxrefs)
        for xref in uniprotxrefs
            db = string(get(xref, :database, ""))
            id = string(get(xref, :id, ""))

            if !isempty(db) && !isempty(id)
                if !haskey(xrefs, db)
                    xrefs[db] = String[]
                end
                push!(xrefs[db], id)
            end
        end
    end

    return xrefs
end

"""
Transform UniProt enzyme data to ontology format.
"""
function transform_cyp_enzymes(enzymes::Dict)
    cyp_data = Dict{Symbol, NamedTuple}()

    for (enzyme, data) in enzymes
        enzyme_sym = Symbol(enzyme)

        # Get basic info
        uniprot_id = string(get(data, :primaryAccession, ""))
        uniprot_kb_id = string(get(data, :uniProtkbId, ""))

        # Get protein description
        protein_desc = get(data, :proteinDescription, nothing)
        full_name = ""
        ec_numbers = String[]

        if !isnothing(protein_desc)
            rec_name = get(protein_desc, :recommendedName, nothing)
            if !isnothing(rec_name)
                full_name_obj = get(rec_name, :fullName, nothing)
                if !isnothing(full_name_obj)
                    full_name = string(get(full_name_obj, :value, ""))
                end

                ec_objs = get(rec_name, :ecNumbers, nothing)
                if !isnothing(ec_objs)
                    for ec in ec_objs
                        ec_val = get(ec, :value, "")
                        !isempty(string(ec_val)) && push!(ec_numbers, string(ec_val))
                    end
                end
            end
        end

        # Extract detailed info
        tissues = extract_tissue_expression(data)
        locations = extract_subcellular_location(data)
        activities = extract_catalytic_activity(data)
        func_desc = extract_function(data)
        seq_info = extract_sequence_info(data)
        xrefs = extract_xrefs(data)

        # Get key cross-references
        gene_ids = get(xrefs, "GeneID", String[])
        pdb_ids = get(xrefs, "PDB", String[])

        cyp_data[enzyme_sym] = (
            name = full_name,
            uniprot_id = uniprot_id,
            uniprot_kb_id = uniprot_kb_id,
            ec_numbers = join(ec_numbers, ","),
            gene_id = isempty(gene_ids) ? "" : first(gene_ids),
            pdb_ids = join(pdb_ids[1:min(5, length(pdb_ids))], ","),  # Limit to 5 PDB structures
            sequence_length = seq_info.length,
            molecular_weight_kda = round(seq_info.mass, digits=2),
            tissue_expression = join(tissues, " | "),
            subcellular_location = join(locations, ", "),
            function_description = func_desc,
            catalytic_activities = join(activities[1:min(10, length(activities))], " | "),  # Limit
            source = :uniprot
        )
    end

    return cyp_data
end

# =============================================================================
# LOAD: Generate Julia source files
# =============================================================================

"""
Generate Julia source code for CYP enzyme properties.
"""
function generate_cyp_enzymes_code(cyp_data::Dict, timestamp::DateTime)
    code = """
# =============================================================================
# CYP ENZYME PROPERTIES DATABASE - GENERATED FROM UNIPROT
# =============================================================================
# Auto-generated by etl_uniprot.jl
# Source: UniProt (https://www.uniprot.org)
# Generated: $timestamp
# DO NOT EDIT MANUALLY - regenerate from source data
# Total enzymes: $(length(cyp_data))
# =============================================================================

\"\"\"
CYP450 enzyme properties from UniProt.
Includes protein information, tissue expression, and cross-references.
\"\"\"
const UNIPROT_CYP_ENZYMES = Dict{Symbol, NamedTuple}(
"""

    for (enzyme, data) in sort(collect(cyp_data), by=x->string(x[1]))
        code *= "    :$enzyme => (\n"
        for (k, v) in pairs(data)
            code *= "        $k = $(repr(v)),\n"
        end
        code *= "    ),\n"
    end

    code *= """)

# Enzyme name lookup
const CYP_ENZYME_NAMES = Dict(
"""
    for (enzyme, data) in sort(collect(cyp_data), by=x->string(x[1]))
        code *= "    :$enzyme => $(repr(data.name)),\n"
    end
    code *= """)

# UniProt ID lookup
const CYP_UNIPROT_IDS = Dict(
"""
    for (enzyme, data) in sort(collect(cyp_data), by=x->string(x[1]))
        code *= "    :$enzyme => $(repr(data.uniprot_id)),\n"
    end
    code *= ")\n"

    return code
end

# =============================================================================
# MAIN ETL FUNCTION
# =============================================================================

"""
    run_uniprot_etl()

Run the complete ETL pipeline for UniProt CYP enzyme data.
"""
function run_uniprot_etl()
    timestamp = now()

    println("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║          UniProt CYP ETL PIPELINE                                 ║
    ║          Darwin PBPK Platform                                     ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Check if data exists
    if !isdir(RAW_DATA_DIR)
        @error "UniProt data not found. Run download_ddi_databases.jl first."
        return nothing
    end

    println("[1/3] Loading UniProt CYP enzyme data...")
    enzymes = load_all_cyp_enzymes()
    println("  Loaded $(length(enzymes)) CYP enzymes")

    println("\n[2/3] Transforming to enzyme properties...")
    cyp_data = transform_cyp_enzymes(enzymes)
    println("  Generated $(length(cyp_data)) enzyme entries")

    for (enzyme, data) in sort(collect(cyp_data), by=x->string(x[1]))
        println("    • $enzyme: $(data.name) ($(data.sequence_length) aa, $(data.molecular_weight_kda) kDa)")
    end

    println("\n[3/3] Generating Julia source files...")

    # Generate and save files
    enzymes_code = generate_cyp_enzymes_code(cyp_data, timestamp)
    enzymes_file = joinpath(OUTPUT_DIR, "uniprot_cyp_enzymes.jl")
    open(enzymes_file, "w") do f
        write(f, enzymes_code)
    end
    println("  ✓ Generated: uniprot_cyp_enzymes.jl")

    println("""

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  ETL COMPLETE                                                     ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Generated files in: databases/generated/                         ║
    ║    • uniprot_cyp_enzymes.jl   ($(length(cyp_data)) enzymes)
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    return cyp_data
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_uniprot_etl()
end
