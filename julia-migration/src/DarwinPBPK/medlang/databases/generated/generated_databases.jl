# =============================================================================
# GENERATED DATABASES INTEGRATION MODULE
# =============================================================================
# Darwin PBPK Platform v2.10.0
#
# This module loads all auto-generated databases from public sources:
# - PharmGKB: CYP substrates, genetic variants, clinical DDIs
# - CPIC: Genetic variants, dosing guidelines, drug properties
# - UniProt: CYP enzyme properties
#
# To regenerate these files, run the ETL scripts in scripts/data_import/
# =============================================================================

module GeneratedDatabases

# =============================================================================
# INCLUDE ALL GENERATED DATABASE FILES
# =============================================================================

# PharmGKB databases
include("pharmgkb_cyp_substrates.jl")
include("pharmgkb_genetic_variants.jl")
include("pharmgkb_clinical_ddis.jl")

# CPIC databases
include("cpic_genetic_variants.jl")
include("cpic_dosing_guidelines.jl")
include("cpic_drug_properties.jl")

# UniProt databases
include("uniprot_cyp_enzymes.jl")

# =============================================================================
# EXPORTS
# =============================================================================

# PharmGKB exports
export PHARMGKB_CYP_SUBSTRATES, PHARMGKB_SUBSTRATES_COUNT
export PHARMGKB_CYP2D6_VARIANTS, PHARMGKB_CYP2C19_VARIANTS, PHARMGKB_CYP2C9_VARIANTS
export PHARMGKB_CYP3A4_VARIANTS, PHARMGKB_CYP1A2_VARIANTS, PHARMGKB_CYP2B6_VARIANTS
export PHARMGKB_VARIANT_COUNTS
export PHARMGKB_CLINICAL_DDIS, PHARMGKB_DDI_COUNT

# CPIC exports
export CPIC_CYP2D6_ALLELES, CPIC_CYP2C19_ALLELES, CPIC_CYP2C9_ALLELES
export CPIC_CYP3A5_ALLELES, CPIC_CYP2B6_ALLELES
export CPIC_ALLELE_COUNTS
export CPIC_DOSING_GUIDELINES, CPIC_GUIDELINES_COUNT, CPIC_RECOMMENDATIONS_COUNT
export CPIC_DRUG_PROPERTIES, CPIC_DRUG_COUNT, CPIC_GUIDELINE_DRUGS

# UniProt exports
export UNIPROT_CYP_ENZYMES, CYP_ENZYME_NAMES, CYP_UNIPROT_IDS

# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

"""
    generated_databases_summary()

Print a summary of all generated databases.
"""
function generated_databases_summary()
    println("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║          GENERATED DDI DATABASES SUMMARY                          ║
    ║          Darwin PBPK Platform                                     ║
    ╚═══════════════════════════════════════════════════════════════════╝

    📊 PharmGKB Data:
       • CYP Substrates: $(PHARMGKB_SUBSTRATES_COUNT) drugs
       • Genetic Variants: $(sum(values(PHARMGKB_VARIANT_COUNTS))) across $(length(PHARMGKB_VARIANT_COUNTS)) genes
       • Clinical DDIs: $(PHARMGKB_DDI_COUNT) drug pairs

    📊 CPIC Data:
       • CYP Alleles: $(sum(values(CPIC_ALLELE_COUNTS))) across $(length(CPIC_ALLELE_COUNTS)) genes
       • Dosing Guidelines: $(CPIC_RECOMMENDATIONS_COUNT) recommendations for $(CPIC_GUIDELINES_COUNT) drugs
       • Drug Properties: $(CPIC_DRUG_COUNT) drugs
       • Drugs with Guidelines: $(length(CPIC_GUIDELINE_DRUGS))

    📊 UniProt Data:
       • CYP Enzymes: $(length(UNIPROT_CYP_ENZYMES)) enzymes

    ═══════════════════════════════════════════════════════════════════
    """)
end

export generated_databases_summary

# =============================================================================
# UNIFIED QUERY FUNCTIONS
# =============================================================================

"""
    get_cyp_substrate_info(drug::Symbol)

Get CYP substrate information for a drug from PharmGKB.
Returns the drug's CYP metabolism data or `nothing` if not found.
"""
function get_cyp_substrate_info(drug::Symbol)
    return get(PHARMGKB_CYP_SUBSTRATES, drug, nothing)
end

"""
    get_cpic_dosing(drug::Symbol)

Get CPIC dosing recommendations for a drug.
Returns a vector of recommendations or empty vector if not found.
"""
function get_cpic_dosing(drug::Symbol)
    return get(CPIC_DOSING_GUIDELINES, drug, NamedTuple[])
end

"""
    get_enzyme_info(enzyme::Symbol)

Get UniProt enzyme information for a CYP enzyme.
"""
function get_enzyme_info(enzyme::Symbol)
    return get(UNIPROT_CYP_ENZYMES, enzyme, nothing)
end

"""
    get_allele_info(gene::Symbol, allele::Symbol)

Get allele information from CPIC or PharmGKB.
Prefers CPIC data if available.
"""
function get_allele_info(gene::Symbol, allele::Symbol)
    # Try CPIC first
    cpic_db = if gene == :CYP2D6
        CPIC_CYP2D6_ALLELES
    elseif gene == :CYP2C19
        CPIC_CYP2C19_ALLELES
    elseif gene == :CYP2C9
        CPIC_CYP2C9_ALLELES
    elseif gene == :CYP3A5
        CPIC_CYP3A5_ALLELES
    elseif gene == :CYP2B6
        CPIC_CYP2B6_ALLELES
    else
        nothing
    end

    if !isnothing(cpic_db) && haskey(cpic_db, allele)
        return cpic_db[allele]
    end

    # Try PharmGKB
    pharmgkb_db = if gene == :CYP2D6 && @isdefined(PHARMGKB_CYP2D6_VARIANTS)
        PHARMGKB_CYP2D6_VARIANTS
    elseif gene == :CYP2C19 && @isdefined(PHARMGKB_CYP2C19_VARIANTS)
        PHARMGKB_CYP2C19_VARIANTS
    elseif gene == :CYP2C9 && @isdefined(PHARMGKB_CYP2C9_VARIANTS)
        PHARMGKB_CYP2C9_VARIANTS
    else
        nothing
    end

    if !isnothing(pharmgkb_db) && haskey(pharmgkb_db, allele)
        return pharmgkb_db[allele]
    end

    return nothing
end

export get_cyp_substrate_info, get_cpic_dosing, get_enzyme_info, get_allele_info

end # module GeneratedDatabases
