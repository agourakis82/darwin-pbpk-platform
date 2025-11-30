# =============================================================================
# DDI DATABASE DOWNLOAD SCRIPTS
# =============================================================================
# Darwin PBPK Platform
#
# Downloads public DDI data from authoritative sources for ontology population
# =============================================================================

using HTTP
using JSON3
using CSV
using DataFrames
using Downloads
using DotEnv

const DATA_DIR = joinpath(@__DIR__, "raw_data")
mkpath(DATA_DIR)

# Load environment variables from .env file
const ENV_FILE = joinpath(@__DIR__, ".env")
if isfile(ENV_FILE)
    env_dict = DotEnv.parse(read(ENV_FILE, String))
    for (k, v) in env_dict
        ENV[k] = v
    end
    println("Loaded API keys from .env")
end

# API Keys from environment
const S2_API_KEY = get(ENV, "S2_API_KEY", "")
const DRUGBANK_API_KEY = get(ENV, "DRUGBANK_API_KEY", "")

# =============================================================================
# 1. DRUGBANK - Requires academic license (free for academic use)
# =============================================================================
# Register at: https://go.drugbank.com/releases/latest
# Download: drugbank_all_full_database.xml.zip (requires login)

"""
DrugBank provides:
- Drug-drug interactions
- Drug-enzyme relationships (CYP substrates/inhibitors/inducers)
- Drug-transporter relationships
- Pharmacokinetic parameters
"""
function download_drugbank_info()
    println("""
    ═══════════════════════════════════════════════════════════════════
    DRUGBANK DATABASE
    ═══════════════════════════════════════════════════════════════════

    DrugBank requires free academic registration:

    1. Go to: https://go.drugbank.com/releases/latest
    2. Create account (free for academic/non-commercial)
    3. Download: drugbank_all_full_database.xml.zip
    4. Place in: $(DATA_DIR)/drugbank/

    Contains:
    - 15,000+ drug entries
    - Drug-drug interactions with mechanisms
    - CYP450 substrate/inhibitor/inducer data
    - Transporter interactions
    - PK parameters (Vd, CL, t1/2, bioavailability)
    ═══════════════════════════════════════════════════════════════════
    """)
end

# =============================================================================
# 2. PHARMGKB - Public download available
# =============================================================================
# https://www.pharmgkb.org/downloads

const PHARMGKB_URLS = Dict(
    "clinical_annotations" => "https://api.pharmgkb.org/v1/download/file/data/clinicalAnnotations.zip",
    "drug_labels" => "https://api.pharmgkb.org/v1/download/file/data/drugLabels.zip",
    "automated_annotations" => "https://api.pharmgkb.org/v1/download/file/data/automatedAnnotations.zip",
    "clinical_variants" => "https://api.pharmgkb.org/v1/download/file/data/clinicalVariants.zip",
    "relationships" => "https://api.pharmgkb.org/v1/download/file/data/relationships.zip",
    "drugs" => "https://api.pharmgkb.org/v1/download/file/data/drugs.zip",
    "genes" => "https://api.pharmgkb.org/v1/download/file/data/genes.zip",
    "phenotypes" => "https://api.pharmgkb.org/v1/download/file/data/phenotypes.zip",
    "variants" => "https://api.pharmgkb.org/v1/download/file/data/variants.zip"
)

"""
Download PharmGKB public datasets.
"""
function download_pharmgkb()
    println("Downloading PharmGKB datasets...")
    pharmgkb_dir = joinpath(DATA_DIR, "pharmgkb")
    mkpath(pharmgkb_dir)

    for (name, url) in PHARMGKB_URLS
        outfile = joinpath(pharmgkb_dir, "$(name).zip")
        if !isfile(outfile)
            println("  Downloading $name...")
            try
                Downloads.download(url, outfile)
                println("    ✓ Downloaded: $outfile")
            catch e
                println("    ✗ Failed: $e")
            end
        else
            println("  ✓ Already exists: $name")
        end
    end

    println("\nPharmGKB data downloaded to: $pharmgkb_dir")
    return pharmgkb_dir
end

# =============================================================================
# 3. FDA DRUG INTERACTIONS - Public
# =============================================================================
# https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers

"""
FDA provides clinical DDI tables (manually curated from labels).
Available as web tables - need to scrape or use cached versions.
"""
function download_fda_ddi_tables()
    println("""
    ═══════════════════════════════════════════════════════════════════
    FDA DRUG INTERACTION TABLES
    ═══════════════════════════════════════════════════════════════════

    FDA Clinical DDI tables available at:
    https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers

    Contains:
    - CYP substrates (sensitive, moderate sensitivity)
    - CYP inhibitors (strong, moderate, weak)
    - CYP inducers (strong, moderate, weak)
    - Transporter substrates and inhibitors

    These are HTML tables - need manual download or scraping.
    Recommend: Copy tables to CSV files manually.
    ═══════════════════════════════════════════════════════════════════
    """)

    fda_dir = joinpath(DATA_DIR, "fda")
    mkpath(fda_dir)
    return fda_dir
end

# =============================================================================
# 4. UNIPROT/SWISSPROT - Enzyme data
# =============================================================================

const UNIPROT_CYP_IDS = Dict(
    "CYP3A4" => "P08684",
    "CYP2D6" => "P10635",
    "CYP2C9" => "P11712",
    "CYP2C19" => "P33261",
    "CYP1A2" => "P05177",
    "CYP2C8" => "P10632",
    "CYP2B6" => "P20813",
    "CYP2E1" => "P05181"
)

"""
Download UniProt enzyme data for CYP450 isoforms.
"""
function download_uniprot_cyp_data()
    println("Downloading UniProt CYP450 data...")
    uniprot_dir = joinpath(DATA_DIR, "uniprot")
    mkpath(uniprot_dir)

    for (enzyme, uniprot_id) in UNIPROT_CYP_IDS
        url = "https://rest.uniprot.org/uniprotkb/$(uniprot_id).json"
        outfile = joinpath(uniprot_dir, "$(enzyme)_$(uniprot_id).json")

        if !isfile(outfile)
            println("  Downloading $enzyme ($uniprot_id)...")
            try
                Downloads.download(url, outfile)
                println("    ✓ Downloaded")
            catch e
                println("    ✗ Failed: $e")
            end
        else
            println("  ✓ Already exists: $enzyme")
        end
    end

    return uniprot_dir
end

# =============================================================================
# 5. PUBCHEM - Chemical properties
# =============================================================================

"""
Download PubChem compound data for physicochemical properties.
"""
function download_pubchem_compound(cid::Int)
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/$(cid)/JSON"
    response = HTTP.get(url)
    return JSON3.read(String(response.body))
end

function download_pubchem_properties(cid::Int)
    props = ["MolecularWeight", "XLogP", "TPSA", "HBondDonorCount", "HBondAcceptorCount"]
    props_str = join(props, ",")
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/$(cid)/property/$(props_str)/JSON"
    response = HTTP.get(url)
    return JSON3.read(String(response.body))
end

# =============================================================================
# 6. CHEMBL - Bioactivity data
# =============================================================================

const CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

"""
Download ChEMBL drug metabolism data.
"""
function download_chembl_metabolism(chembl_id::String)
    url = "$(CHEMBL_API)/metabolism.json?drug_chembl_id=$(chembl_id)"
    response = HTTP.get(url)
    return JSON3.read(String(response.body))
end

function download_chembl_drug_indication(chembl_id::String)
    url = "$(CHEMBL_API)/drug_indication.json?molecule_chembl_id=$(chembl_id)"
    response = HTTP.get(url)
    return JSON3.read(String(response.body))
end

# =============================================================================
# 7. OATP/TRANSPORTER DATA - IUPHAR/BPS Guide to Pharmacology
# =============================================================================

const IUPHAR_API = "https://www.guidetopharmacology.org/services"

"""
Download IUPHAR transporter data.
"""
function download_iuphar_transporters()
    println("Downloading IUPHAR transporter data...")
    iuphar_dir = joinpath(DATA_DIR, "iuphar")
    mkpath(iuphar_dir)

    # Get transporter families
    transporters_url = "$(IUPHAR_API)/targets?type=transporter"

    try
        response = HTTP.get(transporters_url, ["Accept" => "application/json"])
        data = JSON3.read(String(response.body))

        outfile = joinpath(iuphar_dir, "transporters.json")
        open(outfile, "w") do f
            JSON3.write(f, data)
        end
        println("  ✓ Downloaded transporter list")
        return data
    catch e
        println("  ✗ Failed: $e")
        return nothing
    end
end

# =============================================================================
# 8. CPIC GUIDELINES - Pharmacogenomics
# =============================================================================

const CPIC_API = "https://api.cpicpgx.org/v1"

"""
Download CPIC pharmacogenomics guidelines.
"""
function download_cpic_guidelines()
    println("Downloading CPIC guidelines...")
    cpic_dir = joinpath(DATA_DIR, "cpic")
    mkpath(cpic_dir)

    endpoints = [
        "guideline" => "guidelines.json",
        "drug" => "drugs.json",
        "gene" => "genes.json",
        "allele" => "alleles.json",
        "diplotype" => "diplotypes.json",
        "recommendation" => "recommendations.json"
    ]

    for (endpoint, filename) in endpoints
        url = "$(CPIC_API)/$(endpoint)"
        outfile = joinpath(cpic_dir, filename)

        if !isfile(outfile)
            println("  Downloading $endpoint...")
            try
                response = HTTP.get(url)
                open(outfile, "w") do f
                    write(f, response.body)
                end
                println("    ✓ Downloaded")
            catch e
                println("    ✗ Failed: $e")
            end
        else
            println("  ✓ Already exists: $endpoint")
        end
    end

    return cpic_dir
end

# =============================================================================
# 9. FLOCKHART TABLE - University of Indiana
# =============================================================================

"""
Flockhart P450 Drug Interaction Table.
Manual download required from: https://drug-interactions.medicine.iu.edu/
"""
function download_flockhart_info()
    println("""
    ═══════════════════════════════════════════════════════════════════
    FLOCKHART P450 DRUG INTERACTION TABLE
    ═══════════════════════════════════════════════════════════════════

    The Flockhart Table is available at:
    https://drug-interactions.medicine.iu.edu/MainTable.aspx

    Manual steps:
    1. Visit the website
    2. Export/copy each enzyme table (1A2, 2B6, 2C8, 2C9, 2C19, 2D6, 2E1, 3A4)
    3. Save as CSV files in: $(DATA_DIR)/flockhart/

    Format: CYP[enzyme]_substrates.csv, CYP[enzyme]_inhibitors.csv, etc.
    ═══════════════════════════════════════════════════════════════════
    """)

    flockhart_dir = joinpath(DATA_DIR, "flockhart")
    mkpath(flockhart_dir)
    return flockhart_dir
end

# =============================================================================
# 10. SEMANTIC SCHOLAR - DDI Literature Search
# =============================================================================

const S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

"""
Search Semantic Scholar for DDI-related papers.
"""
function search_s2_ddi_papers(query::String; limit::Int=100)
    if isempty(S2_API_KEY)
        @warn "S2_API_KEY not set. Using rate-limited public API."
    end

    headers = ["Content-Type" => "application/json"]
    if !isempty(S2_API_KEY)
        push!(headers, "x-api-key" => S2_API_KEY)
    end

    # URL encode the query
    encoded_query = HTTP.escapeuri(query)
    url = "$(S2_API_BASE)/paper/search?query=$(encoded_query)&limit=$(limit)&fields=paperId,title,abstract,year,authors,citationCount,journal"

    try
        response = HTTP.get(url, headers)
        return JSON3.read(String(response.body))
    catch e
        @error "S2 API error: $e"
        return nothing
    end
end

"""
Get paper details from Semantic Scholar.
"""
function get_s2_paper(paper_id::String)
    headers = ["Content-Type" => "application/json"]
    if !isempty(S2_API_KEY)
        push!(headers, "x-api-key" => S2_API_KEY)
    end

    url = "$(S2_API_BASE)/paper/$(paper_id)?fields=paperId,title,abstract,year,authors,citationCount,references,citations"

    try
        response = HTTP.get(url, headers)
        return JSON3.read(String(response.body))
    catch e
        @error "S2 API error: $e"
        return nothing
    end
end

"""
Download DDI literature from Semantic Scholar.
"""
function download_s2_ddi_literature()
    println("Downloading DDI literature from Semantic Scholar...")
    s2_dir = joinpath(DATA_DIR, "semantic_scholar")
    mkpath(s2_dir)

    # DDI-related search queries
    queries = [
        "drug-drug interaction CYP450",
        "cytochrome P450 inhibition clinical",
        "CYP3A4 substrate inhibitor",
        "CYP2D6 polymorphism drug interaction",
        "transporter mediated drug interaction OATP",
        "P-glycoprotein drug interaction",
        "pharmacokinetic drug interaction AUC",
        "DDI prediction PBPK model"
    ]

    all_papers = Dict{String, Any}()

    for query in queries
        println("  Searching: $query")
        result = search_s2_ddi_papers(query; limit=50)

        if result !== nothing && haskey(result, :data)
            for paper in result.data
                paper_id = paper.paperId
                if !haskey(all_papers, paper_id)
                    all_papers[paper_id] = paper
                end
            end
            println("    Found $(length(result.data)) papers")
        end

        sleep(1)  # Rate limiting
    end

    # Save results
    outfile = joinpath(s2_dir, "ddi_papers.json")
    open(outfile, "w") do f
        JSON3.write(f, collect(values(all_papers)))
    end

    println("  Total unique papers: $(length(all_papers))")
    println("  Saved to: $outfile")

    return s2_dir
end

# =============================================================================
# MAIN DOWNLOAD FUNCTION
# =============================================================================

"""
    download_all_databases()

Download all available public DDI databases.
"""
function download_all_databases()
    println("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║          DDI DATABASE DOWNLOAD UTILITY                            ║
    ║          Darwin PBPK Platform v2.10.0                             ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    println("\n[1/9] DrugBank (requires registration)")
    download_drugbank_info()

    println("\n[2/9] PharmGKB (public download)")
    download_pharmgkb()

    println("\n[3/9] FDA DDI Tables (manual)")
    download_fda_ddi_tables()

    println("\n[4/9] UniProt CYP450 data")
    download_uniprot_cyp_data()

    println("\n[5/9] IUPHAR Transporter data")
    download_iuphar_transporters()

    println("\n[6/9] CPIC Pharmacogenomics")
    download_cpic_guidelines()

    println("\n[7/9] Flockhart Table (manual)")
    download_flockhart_info()

    println("\n[8/9] Semantic Scholar DDI Literature")
    if !isempty(S2_API_KEY)
        download_s2_ddi_literature()
    else
        println("  ⚠ S2_API_KEY not set - skipping Semantic Scholar download")
        println("  Add your key to .env file to enable")
    end

    println("""

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  DOWNLOAD SUMMARY                                                 ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Automatic downloads:                                             ║
    ║    ✓ PharmGKB clinical annotations                                ║
    ║    ✓ UniProt CYP450 enzyme data                                   ║
    ║    ✓ IUPHAR transporter data                                      ║
    ║    ✓ CPIC pharmacogenomics guidelines                             ║
    ║    ✓ Semantic Scholar DDI literature (if API key set)             ║
    ║                                                                   ║
    ║  Manual downloads required:                                       ║
    ║    → DrugBank (free academic license)                             ║
    ║    → FDA DDI Tables (web scrape or manual)                        ║
    ║    → Flockhart Table (manual copy)                                ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Data directory: $(DATA_DIR)

    Next step: Run ETL scripts to transform data into ontology format.
    """)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    download_all_databases()
end
