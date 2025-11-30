# =============================================================================
# ETL PIPELINE: PharmGKB -> DDI Ontology
# =============================================================================
# Transforms downloaded PharmGKB data into Darwin PBPK ontology format
# =============================================================================

using CSV
using DataFrames
using JSON3
using Dates

const RAW_DATA_DIR = joinpath(@__DIR__, "raw_data", "pharmgkb")
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "src", "DarwinPBPK", "medlang", "databases", "generated")

mkpath(OUTPUT_DIR)

# =============================================================================
# EXTRACT: Read PharmGKB TSV files (already extracted)
# =============================================================================

"""
Read TSV file from PharmGKB data directory.
"""
function load_pharmgkb_tsv(filename::String)
    tsv_path = joinpath(RAW_DATA_DIR, filename)

    if !isfile(tsv_path)
        @warn "TSV file not found: $tsv_path"
        return DataFrame()
    end

    return CSV.read(tsv_path, DataFrame; delim='\t', stringtype=String, missingstring=["", "NA"])
end

"""
Load PharmGKB clinical annotations.
"""
function load_clinical_annotations()
    return load_pharmgkb_tsv("clinical_annotations.tsv")
end

"""
Load PharmGKB drug-gene relationships.
"""
function load_relationships()
    return load_pharmgkb_tsv("relationships.tsv")
end

"""
Load PharmGKB drugs.
"""
function load_drugs()
    return load_pharmgkb_tsv("drugs.tsv")
end

"""
Load PharmGKB genes.
"""
function load_genes()
    return load_pharmgkb_tsv("genes.tsv")
end

# =============================================================================
# TRANSFORM: Convert to Ontology Format
# =============================================================================

# Column name mappings (PharmGKB uses spaces in column names)
const COL_ENTITY1_NAME = Symbol("Entity1_name")
const COL_ENTITY1_TYPE = Symbol("Entity1_type")
const COL_ENTITY1_ID = Symbol("Entity1_id")
const COL_ENTITY2_NAME = Symbol("Entity2_name")
const COL_ENTITY2_TYPE = Symbol("Entity2_type")
const COL_ENTITY2_ID = Symbol("Entity2_id")
const COL_EVIDENCE = Symbol("Evidence")
const COL_PK = Symbol("PK")

const COL_GENE = Symbol("Gene")
const COL_VARIANT = Symbol("Variant/Haplotypes")
const COL_LEVEL = Symbol("Level of Evidence")
const COL_PHENOTYPE_CAT = Symbol("Phenotype Category")
const COL_DRUGS = Symbol("Drug(s)")

"""
Transform PharmGKB relationships to CYP substrate data.
"""
function transform_to_cyp_substrates(relationships::DataFrame)
    cyp_genes = ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2", "CYP2C8", "CYP2B6", "CYP2E1"]

    # Dict to collect enzyme assignments per drug
    drug_enzymes = Dict{String, Dict{String, Float64}}()
    drug_ids = Dict{String, String}()

    if isempty(relationships)
        return Dict{Symbol, NamedTuple}()
    end

    for row in eachrow(relationships)
        # Get entity types
        e1_type = get(row, COL_ENTITY1_TYPE, missing)
        e2_type = get(row, COL_ENTITY2_TYPE, missing)

        ismissing(e1_type) && continue
        ismissing(e2_type) && continue

        # Find Chemical-Gene pairs
        drug_name = nothing
        drug_id = nothing
        gene_name = nothing

        if e1_type == "Chemical" && e2_type == "Gene"
            drug_name = get(row, COL_ENTITY1_NAME, missing)
            drug_id = get(row, COL_ENTITY1_ID, missing)
            gene_name = get(row, COL_ENTITY2_NAME, missing)
        elseif e1_type == "Gene" && e2_type == "Chemical"
            drug_name = get(row, COL_ENTITY2_NAME, missing)
            drug_id = get(row, COL_ENTITY2_ID, missing)
            gene_name = get(row, COL_ENTITY1_NAME, missing)
        else
            continue
        end

        ismissing(drug_name) && continue
        ismissing(gene_name) && continue

        # Check if it's a CYP gene
        gene_str = string(gene_name)
        cyp_match = nothing
        for cyp in cyp_genes
            if occursin(cyp, gene_str)
                cyp_match = cyp
                break
            end
        end

        cyp_match === nothing && continue

        # Check if it's a PK relationship (metabolism)
        pk_val = get(row, COL_PK, missing)
        is_pk = !ismissing(pk_val) && !isempty(string(pk_val))

        # Estimate fm based on evidence
        evidence = get(row, COL_EVIDENCE, "")
        evidence_str = ismissing(evidence) ? "" : lowercase(string(evidence))

        # Assign fm estimate based on relationship strength
        fm_estimate = if occursin("clinicalannotation", evidence_str)
            0.5  # Clinical evidence - moderate contribution
        elseif is_pk
            0.4  # PK relationship
        else
            0.3  # Other evidence
        end

        # Store in drug_enzymes dict
        drug_key = lowercase(string(drug_name))
        enzyme_key = lowercase(replace(cyp_match, "CYP" => ""))

        if !haskey(drug_enzymes, drug_key)
            drug_enzymes[drug_key] = Dict{String, Float64}()
        end

        # Keep highest fm value for each enzyme
        current_fm = get(drug_enzymes[drug_key], enzyme_key, 0.0)
        drug_enzymes[drug_key][enzyme_key] = max(current_fm, fm_estimate)

        if !ismissing(drug_id)
            drug_ids[drug_key] = string(drug_id)
        end
    end

    # Convert to ontology format
    substrates = Dict{Symbol, NamedTuple}()

    for (drug, enzymes) in drug_enzymes
        # Clean drug name for symbol (ensure doesn't start with number)
        drug_clean = replace(drug, r"[^a-z0-9_]" => "_")
        # Prepend underscore if starts with number
        if !isempty(drug_clean) && isdigit(drug_clean[1])
            drug_clean = "_" * drug_clean
        end
        drug_sym = Symbol(drug_clean)

        # Build named tuple with fm values
        fields = Dict{Symbol, Any}(:source => :pharmgkb)

        if haskey(drug_ids, drug)
            fields[:pharmgkb_id] = drug_ids[drug]
        end

        for (enzyme, fm) in enzymes
            fields[Symbol("fm_$enzyme")] = round(fm, digits=2)
        end

        substrates[drug_sym] = (; (k => v for (k, v) in fields)...)
    end

    return substrates
end

"""
Transform PharmGKB clinical annotations to genetic variants database.
"""
function transform_to_genetic_variants(annotations::DataFrame)
    variants = Dict{Symbol, Dict{Symbol, NamedTuple}}()

    # CYP gene mapping
    cyp_genes = Dict(
        "CYP2D6" => :CYP2D6,
        "CYP2C19" => :CYP2C19,
        "CYP2C9" => :CYP2C9,
        "CYP3A4" => :CYP3A4,
        "CYP3A5" => :CYP3A5,
        "CYP1A2" => :CYP1A2,
        "CYP2B6" => :CYP2B6,
        "CYP2C8" => :CYP2C8
    )

    if isempty(annotations)
        return variants
    end

    for row in eachrow(annotations)
        gene = get(row, COL_GENE, missing)
        ismissing(gene) && continue
        gene_str = string(gene)

        # Check if it's a CYP gene we care about
        !haskey(cyp_genes, gene_str) && continue
        gene_sym = cyp_genes[gene_str]

        if !haskey(variants, gene_sym)
            variants[gene_sym] = Dict{Symbol, NamedTuple}()
        end

        # Get variant/haplotype info
        variant_str = get(row, COL_VARIANT, missing)
        ismissing(variant_str) && continue

        # Parse multiple variants (comma-separated)
        for variant in split(string(variant_str), ",")
            variant = strip(variant)
            isempty(variant) && continue

            # Clean variant name for symbol
            variant_clean = replace(variant, "*" => "star", " " => "_", "/" => "_", "x" => "x")
            # Prepend underscore if starts with number
            if !isempty(variant_clean) && isdigit(variant_clean[1])
                variant_clean = "_" * variant_clean
            end
            variant_sym = Symbol(variant_clean)

            # Get phenotype and evidence info
            phenotype = get(row, COL_PHENOTYPE_CAT, "")
            level = get(row, COL_LEVEL, "")
            drugs = get(row, COL_DRUGS, "")

            phenotype_str = ismissing(phenotype) ? "" : string(phenotype)
            level_str = ismissing(level) ? "" : string(level)
            drugs_str = ismissing(drugs) ? "" : string(drugs)

            # Determine functional status from phenotype
            func_status = if occursin("Metabolism", phenotype_str)
                if occursin("Poor", phenotype_str) || occursin("Reduced", phenotype_str)
                    :reduced
                elseif occursin("Ultra", phenotype_str) || occursin("Increased", phenotype_str)
                    :increased
                else
                    :normal
                end
            else
                :unknown
            end

            variants[gene_sym][variant_sym] = (
                phenotype = phenotype_str,
                evidence_level = level_str,
                func_status = func_status,
                drugs = drugs_str,
                source = :pharmgkb
            )
        end
    end

    return variants
end

"""
Transform PharmGKB clinical annotations to clinical DDI data.
"""
function transform_to_clinical_ddis(annotations::DataFrame, relationships::DataFrame)
    ddis = Dict{Symbol, NamedTuple}()

    # Look for drug-drug pairs in relationships
    if !isempty(relationships)
        for row in eachrow(relationships)
            e1_type = get(row, COL_ENTITY1_TYPE, missing)
            e2_type = get(row, COL_ENTITY2_TYPE, missing)

            # Find Chemical-Chemical pairs
            if !ismissing(e1_type) && !ismissing(e2_type) &&
               e1_type == "Chemical" && e2_type == "Chemical"

                drug1 = get(row, COL_ENTITY1_NAME, missing)
                drug2 = get(row, COL_ENTITY2_NAME, missing)

                ismissing(drug1) && continue
                ismissing(drug2) && continue

                # Create DDI key
                d1 = lowercase(string(drug1))
                d2 = lowercase(string(drug2))
                d1_clean = replace(d1, r"[^a-z0-9]" => "_")
                d2_clean = replace(d2, r"[^a-z0-9]" => "_")
                # Prepend underscore if starts with number
                if !isempty(d1_clean) && isdigit(d1_clean[1])
                    d1_clean = "_" * d1_clean
                end
                ddi_key = Symbol("$(d1_clean)_$(d2_clean)")

                evidence = get(row, COL_EVIDENCE, "")
                evidence_str = ismissing(evidence) ? "" : string(evidence)

                ddis[ddi_key] = (
                    perpetrator = d1,
                    victim = d2,
                    evidence = evidence_str,
                    source = :pharmgkb
                )
            end
        end
    end

    return ddis
end

# =============================================================================
# LOAD: Generate Julia source files
# =============================================================================

"""
Generate Julia source code for CYP substrates database.
"""
function generate_cyp_substrates_code(substrates::Dict, timestamp::DateTime)
    code = """
# =============================================================================
# CYP SUBSTRATES DATABASE - GENERATED FROM PHARMGKB
# =============================================================================
# Auto-generated by etl_pharmgkb.jl
# Source: PharmGKB (https://www.pharmgkb.org)
# Generated: $timestamp
# DO NOT EDIT MANUALLY - regenerate from source data
# =============================================================================

\"\"\"
CYP substrate data extracted from PharmGKB relationships.
Keys are drug names (symbols), values are NamedTuples with:
- fm_XXXX: fraction metabolized by CYP enzyme (estimated)
- source: data source (:pharmgkb)
- pharmgkb_id: PharmGKB accession ID
\"\"\"
const PHARMGKB_CYP_SUBSTRATES = Dict{Symbol, NamedTuple}(
"""

    for (drug, data) in sort(collect(substrates), by=x->string(x[1]))
        fields = join(["$k = $(repr(v))" for (k, v) in pairs(data)], ", ")
        code *= "    :$drug => ($fields),\n"
    end

    code *= """)

# Export count for verification
const PHARMGKB_SUBSTRATES_COUNT = $(length(substrates))
"""
    return code
end

"""
Generate Julia source code for genetic variants database.
"""
function generate_genetic_variants_code(variants::Dict, timestamp::DateTime)
    total_count = sum(length(v) for v in values(variants); init=0)

    code = """
# =============================================================================
# GENETIC VARIANTS DATABASE - GENERATED FROM PHARMGKB
# =============================================================================
# Auto-generated by etl_pharmgkb.jl
# Source: PharmGKB (https://www.pharmgkb.org)
# Generated: $timestamp
# DO NOT EDIT MANUALLY - regenerate from source data
# Total variants: $total_count across $(length(variants)) genes
# =============================================================================

"""

    for (gene, alleles) in sort(collect(variants), by=x->string(x[1]))
        code *= """
\"\"\"
$(gene) genetic variants from PharmGKB clinical annotations.
$(length(alleles)) variants.
\"\"\"
const PHARMGKB_$(gene)_VARIANTS = Dict{Symbol, NamedTuple}(
"""

        for (allele, data) in sort(collect(alleles), by=x->string(x[1]))
            fields = join(["$k = $(repr(v))" for (k, v) in pairs(data)], ", ")
            code *= "    :$allele => ($fields),\n"
        end

        code *= ")\n\n"
    end

    # Add summary
    code *= """
# Summary of all variant counts
const PHARMGKB_VARIANT_COUNTS = Dict(
"""
    for (gene, alleles) in sort(collect(variants), by=x->string(x[1]))
        code *= "    :$gene => $(length(alleles)),\n"
    end
    code *= ")\n"

    return code
end

"""
Generate Julia source code for clinical DDIs database.
"""
function generate_clinical_ddis_code(ddis::Dict, timestamp::DateTime)
    code = """
# =============================================================================
# CLINICAL DDI DATABASE - GENERATED FROM PHARMGKB
# =============================================================================
# Auto-generated by etl_pharmgkb.jl
# Source: PharmGKB (https://www.pharmgkb.org)
# Generated: $timestamp
# DO NOT EDIT MANUALLY - regenerate from source data
# =============================================================================

\"\"\"
Drug-drug interaction pairs from PharmGKB.
$(length(ddis)) DDI pairs.
\"\"\"
const PHARMGKB_CLINICAL_DDIS = Dict{Symbol, NamedTuple}(
"""

    for (ddi_key, data) in sort(collect(ddis), by=x->string(x[1]))
        fields = join(["$k = $(repr(v))" for (k, v) in pairs(data)], ", ")
        code *= "    :$ddi_key => ($fields),\n"
    end

    code *= """)

const PHARMGKB_DDI_COUNT = $(length(ddis))
"""
    return code
end

# =============================================================================
# MAIN ETL FUNCTION
# =============================================================================

"""
    run_pharmgkb_etl()

Run the complete ETL pipeline for PharmGKB data.
"""
function run_pharmgkb_etl()
    timestamp = now()

    println("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║          PharmGKB ETL PIPELINE                                    ║
    ║          Darwin PBPK Platform                                     ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Check if data exists
    if !isdir(RAW_DATA_DIR)
        @error "PharmGKB data not found. Run download_ddi_databases.jl first."
        return nothing
    end

    println("[1/5] Loading PharmGKB data...")
    annotations = load_clinical_annotations()
    relationships = load_relationships()
    drugs = load_drugs()

    println("  Loaded $(nrow(annotations)) clinical annotations")
    println("  Loaded $(nrow(relationships)) relationships")
    println("  Loaded $(nrow(drugs)) drugs")

    println("\n[2/5] Transforming to CYP substrates...")
    substrates = transform_to_cyp_substrates(relationships)
    println("  Generated $(length(substrates)) substrate entries")

    println("\n[3/5] Transforming to genetic variants...")
    variants = transform_to_genetic_variants(annotations)
    total_variants = sum(length(v) for v in values(variants); init=0)
    println("  Generated $total_variants variant entries across $(length(variants)) genes")

    println("\n[4/5] Transforming to clinical DDIs...")
    ddis = transform_to_clinical_ddis(annotations, relationships)
    println("  Generated $(length(ddis)) DDI entries")

    println("\n[5/5] Generating Julia source files...")

    # Generate and save files
    substrates_code = generate_cyp_substrates_code(substrates, timestamp)
    substrates_file = joinpath(OUTPUT_DIR, "pharmgkb_cyp_substrates.jl")
    open(substrates_file, "w") do f
        write(f, substrates_code)
    end
    println("  ✓ Generated: pharmgkb_cyp_substrates.jl")

    variants_code = generate_genetic_variants_code(variants, timestamp)
    variants_file = joinpath(OUTPUT_DIR, "pharmgkb_genetic_variants.jl")
    open(variants_file, "w") do f
        write(f, variants_code)
    end
    println("  ✓ Generated: pharmgkb_genetic_variants.jl")

    ddis_code = generate_clinical_ddis_code(ddis, timestamp)
    ddis_file = joinpath(OUTPUT_DIR, "pharmgkb_clinical_ddis.jl")
    open(ddis_file, "w") do f
        write(f, ddis_code)
    end
    println("  ✓ Generated: pharmgkb_clinical_ddis.jl")

    println("""

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  ETL COMPLETE                                                     ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Generated files in: databases/generated/                         ║
    ║    • pharmgkb_cyp_substrates.jl   ($(length(substrates)) substrates)
    ║    • pharmgkb_genetic_variants.jl ($total_variants variants)
    ║    • pharmgkb_clinical_ddis.jl    ($(length(ddis)) DDIs)
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    return (substrates=substrates, variants=variants, ddis=ddis)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_pharmgkb_etl()
end
