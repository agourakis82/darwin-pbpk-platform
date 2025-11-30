# =============================================================================
# ETL PIPELINE: CPIC -> DDI Ontology
# =============================================================================
# Transforms downloaded CPIC data into Darwin PBPK ontology format
# Clinical Pharmacogenetics Implementation Consortium (CPIC) Guidelines
# =============================================================================

using JSON3
using Dates

const RAW_DATA_DIR = joinpath(@__DIR__, "raw_data", "cpic")
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "src", "DarwinPBPK", "medlang", "databases", "generated")

mkpath(OUTPUT_DIR)

# =============================================================================
# EXTRACT: Load CPIC JSON files
# =============================================================================

"""
Load JSON file from CPIC data directory.
"""
function load_cpic_json(filename::String)
    json_path = joinpath(RAW_DATA_DIR, filename)

    if !isfile(json_path)
        @warn "JSON file not found: $json_path"
        return nothing
    end

    return JSON3.read(read(json_path, String))
end

"""Load CPIC drugs."""
load_cpic_drugs() = load_cpic_json("drugs.json")

"""Load CPIC genes."""
load_cpic_genes() = load_cpic_json("genes.json")

"""Load CPIC alleles."""
load_cpic_alleles() = load_cpic_json("alleles.json")

"""Load CPIC recommendations."""
load_cpic_recommendations() = load_cpic_json("recommendations.json")

"""Load CPIC guidelines."""
load_cpic_guidelines() = load_cpic_json("guidelines.json")

# =============================================================================
# TRANSFORM: Convert to Ontology Format
# =============================================================================

# CYP genes of interest
const CYP_GENES = Set(["CYP2D6", "CYP2C19", "CYP2C9", "CYP3A4", "CYP3A5", "CYP1A2", "CYP2B6", "CYP2C8"])

"""
Transform CPIC alleles to genetic variants database.
"""
function transform_cpic_alleles(alleles, genes)
    variants = Dict{Symbol, Dict{Symbol, NamedTuple}}()

    isnothing(alleles) && return variants

    # Build gene symbol lookup
    gene_symbols = Dict{String, String}()
    if !isnothing(genes)
        for gene in genes
            gene_symbols[string(get(gene, :symbol, ""))] = string(get(gene, :symbol, ""))
        end
    end

    for allele in alleles
        gene_symbol = string(get(allele, :genesymbol, ""))

        # Only process CYP genes
        gene_symbol in CYP_GENES || continue

        gene_sym = Symbol(gene_symbol)
        if !haskey(variants, gene_sym)
            variants[gene_sym] = Dict{Symbol, NamedTuple}()
        end

        # Get allele name
        allele_name = string(get(allele, :name, ""))
        isempty(allele_name) && continue

        # Clean allele name for symbol - replace all special chars
        allele_clean = replace(allele_name,
            "*" => "star_",
            " " => "_",
            "/" => "_",
            ">" => "_gt_",
            "<" => "_lt_",
            "≥" => "_gte_",
            "≤" => "_lte_",
            "+" => "_plus_",
            "(" => "_",
            ")" => "_",
            "[" => "_",
            "]" => "_",
            "×" => "x",
            "-" => "_",
            ":" => "_"
        )
        # Remove consecutive underscores and trailing underscores
        allele_clean = replace(allele_clean, r"_+" => "_")
        allele_clean = strip(allele_clean, '_')
        # Ensure doesn't start with number or is just 'star' followed by number
        if !isempty(allele_clean) && (isdigit(allele_clean[1]) || startswith(allele_clean, "star"))
            allele_clean = "allele_" * allele_clean
        end
        allele_sym = Symbol(allele_clean)

        # Get function status
        func_str = string(get(allele, :clinicalfunctionalstatus, ""))
        func_status = if occursin("no function", lowercase(func_str))
            :no_function
        elseif occursin("decreased", lowercase(func_str))
            :decreased
        elseif occursin("increased", lowercase(func_str))
            :increased
        elseif occursin("normal", lowercase(func_str))
            :normal
        elseif occursin("uncertain", lowercase(func_str))
            :uncertain
        else
            :unknown
        end

        # Get activity value if available
        activity = get(allele, :activityvalue, nothing)
        activity_val = if isnothing(activity) || activity === ""
            1.0
        elseif activity isa Number
            Float64(activity)
        else
            tryparse(Float64, string(activity))
        end
        activity_val = isnothing(activity_val) ? 1.0 : activity_val

        variants[gene_sym][allele_sym] = (
            name = allele_name,
            function_status = func_status,
            activity_value = activity_val,
            clinical_functional_status = func_str,
            source = :cpic
        )
    end

    return variants
end

"""
Transform CPIC recommendations to dosing guidelines.
"""
function transform_cpic_recommendations(recommendations, drugs)
    guidelines = Dict{Symbol, Vector{NamedTuple}}()

    isnothing(recommendations) && return guidelines

    # Build drug name lookup
    drug_names = Dict{String, String}()
    if !isnothing(drugs)
        for drug in drugs
            drugid = string(get(drug, :drugid, ""))
            name = string(get(drug, :name, ""))
            drug_names[drugid] = name
        end
    end

    for rec in recommendations
        # Get drug info
        drugid = string(get(rec, :drugid, ""))
        drug_name = get(drug_names, drugid, "")

        # Skip if no drug name
        if isempty(drug_name)
            # Try to extract from drugid
            if startswith(drugid, "RxNorm:")
                drug_name = drugid
            else
                continue
            end
        end

        # Clean drug name for symbol
        drug_clean = lowercase(replace(drug_name, r"[^a-zA-Z0-9]" => "_"))
        # Ensure doesn't start with number
        if !isempty(drug_clean) && isdigit(drug_clean[1])
            drug_clean = "_" * drug_clean
        end
        drug_sym = Symbol(drug_clean)

        if !haskey(guidelines, drug_sym)
            guidelines[drug_sym] = Vector{NamedTuple}()
        end

        # Get recommendation details
        recommendation = string(get(rec, :drugrecommendation, ""))
        classification = string(get(rec, :classification, ""))
        implications = get(rec, :implications, Dict())
        phenotypes = get(rec, :phenotypes, Dict())
        lookupkey = get(rec, :lookupkey, Dict())

        # Convert implications dict to string
        impl_str = join(["$k: $v" for (k, v) in pairs(implications)], "; ")
        phenotype_str = join(["$k: $v" for (k, v) in pairs(phenotypes)], "; ")
        lookup_str = join(["$k: $v" for (k, v) in pairs(lookupkey)], "; ")

        push!(guidelines[drug_sym], (
            drug_name = drug_name,
            recommendation = recommendation,
            classification = classification,
            implications = impl_str,
            phenotypes = phenotype_str,
            lookup_key = lookup_str,
            source = :cpic
        ))
    end

    return guidelines
end

"""
Transform CPIC drugs to drug properties.
"""
function transform_cpic_drugs(drugs)
    drug_props = Dict{Symbol, NamedTuple}()

    isnothing(drugs) && return drug_props

    for drug in drugs
        name = string(get(drug, :name, ""))
        isempty(name) && continue

        drug_clean = lowercase(replace(name, r"[^a-zA-Z0-9]" => "_"))
        # Ensure doesn't start with number
        if !isempty(drug_clean) && isdigit(drug_clean[1])
            drug_clean = "_" * drug_clean
        end
        drug_sym = Symbol(drug_clean)

        # Extract IDs
        pharmgkb_id = get(drug, :pharmgkbid, nothing)
        rxnorm_id = get(drug, :rxnormid, nothing)
        drugbank_id = get(drug, :drugbankid, nothing)
        atc_ids = get(drug, :atcid, nothing)
        atc_str = if isnothing(atc_ids)
            ""
        else
            join([string(a) for a in atc_ids], ",")
        end

        drug_props[drug_sym] = (
            name = name,
            pharmgkb_id = isnothing(pharmgkb_id) ? "" : pharmgkb_id,
            rxnorm_id = isnothing(rxnorm_id) ? "" : rxnorm_id,
            drugbank_id = isnothing(drugbank_id) ? "" : drugbank_id,
            atc_ids = atc_str,
            has_cpic_guideline = !isnothing(get(drug, :guidelineid, nothing)),
            source = :cpic
        )
    end

    return drug_props
end

# =============================================================================
# LOAD: Generate Julia source files
# =============================================================================

"""
Generate Julia source code for CPIC genetic variants.
"""
function generate_cpic_variants_code(variants::Dict, timestamp::DateTime)
    total_count = sum(length(v) for v in values(variants); init=0)

    code = """
# =============================================================================
# GENETIC VARIANTS DATABASE - GENERATED FROM CPIC
# =============================================================================
# Auto-generated by etl_cpic.jl
# Source: CPIC (https://cpicpgx.org)
# Generated: $timestamp
# DO NOT EDIT MANUALLY - regenerate from source data
# Total alleles: $total_count across $(length(variants)) genes
# =============================================================================

"""

    for (gene, alleles) in sort(collect(variants), by=x->string(x[1]))
        code *= """
\"\"\"
$(gene) allele definitions from CPIC.
$(length(alleles)) alleles with clinical functional status.
\"\"\"
const CPIC_$(gene)_ALLELES = Dict{Symbol, NamedTuple}(
"""

        for (allele, data) in sort(collect(alleles), by=x->string(x[1]))
            fields = join(["$k = $(repr(v))" for (k, v) in pairs(data)], ", ")
            code *= "    :$allele => ($fields),\n"
        end

        code *= ")\n\n"
    end

    # Add summary
    code *= """
# Summary of all allele counts
const CPIC_ALLELE_COUNTS = Dict(
"""
    for (gene, alleles) in sort(collect(variants), by=x->string(x[1]))
        code *= "    :$gene => $(length(alleles)),\n"
    end
    code *= ")\n"

    return code
end

"""
Generate Julia source code for CPIC dosing guidelines.
"""
function generate_cpic_guidelines_code(guidelines::Dict, timestamp::DateTime)
    total_recs = sum(length(v) for v in values(guidelines); init=0)

    code = """
# =============================================================================
# CPIC DOSING GUIDELINES DATABASE
# =============================================================================
# Auto-generated by etl_cpic.jl
# Source: CPIC (https://cpicpgx.org)
# Generated: $timestamp
# DO NOT EDIT MANUALLY - regenerate from source data
# Total recommendations: $total_recs across $(length(guidelines)) drugs
# =============================================================================

\"\"\"
CPIC dosing recommendations by drug.
Each drug maps to a vector of recommendations based on genotype/phenotype.
\"\"\"
const CPIC_DOSING_GUIDELINES = Dict{Symbol, Vector{NamedTuple}}(
"""

    for (drug, recs) in sort(collect(guidelines), by=x->string(x[1]))
        code *= "    :$drug => [\n"
        for rec in recs
            fields = join(["$k = $(repr(v))" for (k, v) in pairs(rec)], ", ")
            code *= "        ($fields),\n"
        end
        code *= "    ],\n"
    end

    code *= """)

const CPIC_GUIDELINES_COUNT = $(length(guidelines))
const CPIC_RECOMMENDATIONS_COUNT = $total_recs
"""
    return code
end

"""
Generate Julia source code for CPIC drug properties.
"""
function generate_cpic_drugs_code(drug_props::Dict, timestamp::DateTime)
    code = """
# =============================================================================
# CPIC DRUG PROPERTIES DATABASE
# =============================================================================
# Auto-generated by etl_cpic.jl
# Source: CPIC (https://cpicpgx.org)
# Generated: $timestamp
# DO NOT EDIT MANUALLY - regenerate from source data
# Total drugs: $(length(drug_props))
# =============================================================================

\"\"\"
Drug properties from CPIC including cross-references to PharmGKB, RxNorm, DrugBank.
\"\"\"
const CPIC_DRUG_PROPERTIES = Dict{Symbol, NamedTuple}(
"""

    for (drug, data) in sort(collect(drug_props), by=x->string(x[1]))
        fields = join(["$k = $(repr(v))" for (k, v) in pairs(data)], ", ")
        code *= "    :$drug => ($fields),\n"
    end

    code *= """)

const CPIC_DRUG_COUNT = $(length(drug_props))

# Drugs with CPIC guidelines
const CPIC_GUIDELINE_DRUGS = [
"""
    guideline_drugs = [d for (d, props) in drug_props if props.has_cpic_guideline]
    for drug in sort(guideline_drugs)
        code *= "    :$drug,\n"
    end
    code *= "]\n"

    return code
end

# =============================================================================
# MAIN ETL FUNCTION
# =============================================================================

"""
    run_cpic_etl()

Run the complete ETL pipeline for CPIC data.
"""
function run_cpic_etl()
    timestamp = now()

    println("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║          CPIC ETL PIPELINE                                        ║
    ║          Darwin PBPK Platform                                     ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Check if data exists
    if !isdir(RAW_DATA_DIR)
        @error "CPIC data not found. Run download_ddi_databases.jl first."
        return nothing
    end

    println("[1/5] Loading CPIC data...")
    drugs = load_cpic_drugs()
    genes = load_cpic_genes()
    alleles = load_cpic_alleles()
    recommendations = load_cpic_recommendations()

    println("  Loaded $(isnothing(drugs) ? 0 : length(drugs)) drugs")
    println("  Loaded $(isnothing(genes) ? 0 : length(genes)) genes")
    println("  Loaded $(isnothing(alleles) ? 0 : length(alleles)) alleles")
    println("  Loaded $(isnothing(recommendations) ? 0 : length(recommendations)) recommendations")

    println("\n[2/5] Transforming to genetic variants (CYP alleles)...")
    variants = transform_cpic_alleles(alleles, genes)
    total_variants = sum(length(v) for v in values(variants); init=0)
    println("  Generated $total_variants allele entries across $(length(variants)) CYP genes")

    println("\n[3/5] Transforming to dosing guidelines...")
    guidelines = transform_cpic_recommendations(recommendations, drugs)
    total_recs = sum(length(v) for v in values(guidelines); init=0)
    println("  Generated $total_recs recommendations for $(length(guidelines)) drugs")

    println("\n[4/5] Transforming to drug properties...")
    drug_props = transform_cpic_drugs(drugs)
    println("  Generated $(length(drug_props)) drug entries")

    println("\n[5/5] Generating Julia source files...")

    # Generate and save files
    variants_code = generate_cpic_variants_code(variants, timestamp)
    variants_file = joinpath(OUTPUT_DIR, "cpic_genetic_variants.jl")
    open(variants_file, "w") do f
        write(f, variants_code)
    end
    println("  ✓ Generated: cpic_genetic_variants.jl")

    guidelines_code = generate_cpic_guidelines_code(guidelines, timestamp)
    guidelines_file = joinpath(OUTPUT_DIR, "cpic_dosing_guidelines.jl")
    open(guidelines_file, "w") do f
        write(f, guidelines_code)
    end
    println("  ✓ Generated: cpic_dosing_guidelines.jl")

    drugs_code = generate_cpic_drugs_code(drug_props, timestamp)
    drugs_file = joinpath(OUTPUT_DIR, "cpic_drug_properties.jl")
    open(drugs_file, "w") do f
        write(f, drugs_code)
    end
    println("  ✓ Generated: cpic_drug_properties.jl")

    println("""

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  ETL COMPLETE                                                     ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Generated files in: databases/generated/                         ║
    ║    • cpic_genetic_variants.jl   ($total_variants CYP alleles)
    ║    • cpic_dosing_guidelines.jl  ($total_recs recommendations)
    ║    • cpic_drug_properties.jl    ($(length(drug_props)) drugs)
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    return (variants=variants, guidelines=guidelines, drug_props=drug_props)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_cpic_etl()
end
