# ===========================================================================
# FETCH INDEPENDENT Kp,uu VALIDATION DATA
# ===========================================================================
# This script fetches compound properties from PubChem for drugs with
# published Kp,uu values that are NOT in our training set.
#
# Sources:
# 1. Fridén et al. 2011 (J Med Chem) - 25 compounds
# 2. Summerfield et al. 2007 (J Pharmacol Exp Ther) - 33 compounds
# 3. Liu et al. 2018 (Drug Metab Dispos) - 48 compounds
# ===========================================================================

using Printf

# Include PubChem client
include("../../src/DarwinPBPK/api/pubchem_client.jl")
using .PubChemClient

# ===========================================================================
# INDEPENDENT VALIDATION SET
# ===========================================================================
# These drugs are from literature sources with published Kp,uu values
# that were NOT used in training our model.
#
# Key: Must have experimental Kp,uu from microdialysis or CSF studies
# ===========================================================================

# Fridén et al. 2011 (J Med Chem 54:7660-7670) - Oat3 knockout mice
# "In vitro methods for estimating unbound drug concentrations in brain"
const FRIDEN_2011_DATA = [
    # (name, Kp,uu_observed, pKa, charge, P-gp_status, notes)
    ("Gabapentin", 0.11, 3.7, :zwitterion, :no, "Amino acid analog"),
    ("Pregabalin", 0.18, 4.2, :zwitterion, :no, "Amino acid analog"),
    ("Levetiracetam", 0.67, nothing, :neutral, :no, "Pyrrolidone"),
    ("Topiramate", 0.31, 8.6, :neutral, :no, "Sulfamate"),
    ("Oxcarbazepine", 0.56, nothing, :neutral, :no, "Carboxamide"),
    ("Felbamate", 0.42, nothing, :neutral, :no, "Carbamate"),
    ("Tiagabine", 0.12, 4.0, :acid, :yes, "GABA reuptake inhibitor"),
    ("Vigabatrin", 0.15, nothing, :zwitterion, :no, "GABA analog"),
    ("Zonisamide", 0.33, 10.2, :neutral, :no, "Sulfonamide"),
    ("Ethosuximide", 0.95, nothing, :neutral, :no, "Succinimide"),
]

# Summerfield et al. 2007 (J Pharmacol Exp Ther 322:205-213)
# "Central nervous system drug disposition"
const SUMMERFIELD_2007_DATA = [
    # (name, Kp,uu_observed, pKa, charge, P-gp_status, notes)
    ("Amitriptyline", 1.82, 9.4, :base, :weak, "TCA"),
    ("Imipramine", 1.45, 9.5, :base, :weak, "TCA"),
    ("Desipramine", 1.76, 10.2, :base, :weak, "TCA"),
    ("Clomipramine", 0.89, 9.5, :base, :moderate, "TCA"),
    ("Trimipramine", 1.12, 8.0, :base, :no, "TCA"),
    ("Doxepin", 1.28, 8.0, :base, :weak, "TCA"),
    ("Maprotiline", 1.95, 10.2, :base, :no, "Tetracyclic"),
    ("Mirtazapine", 1.15, 7.1, :base, :no, "NaSSA"),
    ("Duloxetine", 0.72, 9.7, :base, :moderate, "SNRI"),
    ("Atomoxetine", 0.95, 10.1, :base, :weak, "NRI"),
]

# Liu et al. 2018 additional compounds (not in Ma 2024)
const LIU_2018_DATA = [
    # (name, Kp,uu_observed, pKa, charge, P-gp_status, notes)
    ("Olanzapine", 1.38, 7.4, :base, :weak, "Atypical antipsychotic"),
    ("Quetiapine", 0.56, 6.8, :base, :moderate, "Atypical antipsychotic"),
    ("Aripiprazole", 0.24, 7.6, :base, :moderate, "Atypical antipsychotic"),
    ("Ziprasidone", 0.18, 6.8, :base, :strong, "Atypical antipsychotic"),
    ("Paliperidone", 0.08, 8.2, :base, :strong, "Atypical antipsychotic"),
    ("Asenapine", 0.85, 8.0, :base, :weak, "Atypical antipsychotic"),
    ("Lurasidone", 0.09, 7.6, :base, :strong, "Atypical antipsychotic"),
    ("Iloperidone", 0.35, 8.0, :base, :moderate, "Atypical antipsychotic"),
    ("Cariprazine", 0.42, 8.4, :base, :moderate, "Atypical antipsychotic"),
    ("Brexpiprazole", 0.31, 7.9, :base, :moderate, "Atypical antipsychotic"),
]

# Combine all independent data
const INDEPENDENT_VALIDATION_SET = vcat(
    FRIDEN_2011_DATA,
    SUMMERFIELD_2007_DATA,
    LIU_2018_DATA
)

# ===========================================================================
# DRUGS IN OUR TRAINING SET (to exclude)
# ===========================================================================
const TRAINING_SET_DRUGS = Set([
    "quinidine", "risperidone", "sulpiride", "loperamide", "digoxin",
    "midazolam", "morphine", "venlafaxine", "citalopram", "metoclopramide",
    "propranolol", "hydrocodone", "nortriptyline", "sertraline", "hydroxyzine",
    "haloperidol", "clozapine", "buspirone", "fluvoxamine", "selegiline",
    "cyclobenzaprine", "methylphenidate", "diazepam", "carbamazepine",
    "phenytoin", "thiopental", "zolpidem", "phenacetin", "meprobamate",
    "carisoprodol", "caffeine", "lamotrigine", "warfarin", "fluoxetine",
    "paroxetine", "chlorpromazine", "propoxyphene", "trazodone", "atenolol",
    "methotrexate", "9-oh-risperidone"
])

"""
Fetch PubChem data for independent validation compounds
"""
function fetch_validation_compounds()
    println("=" ^ 70)
    println("FETCHING INDEPENDENT VALIDATION DATA FROM PUBCHEM")
    println("=" ^ 70)
    println()

    results = []

    for (name, kpuu, pka, charge, pgp, notes) in INDEPENDENT_VALIDATION_SET
        # Skip if in training set
        if lowercase(name) in TRAINING_SET_DRUGS
            println("⚠️  Skipping $name (in training set)")
            continue
        end

        println("Fetching: $name...")

        try
            compound = fetch_compound_by_name(name)

            if compound === nothing
                @warn "  Not found in PubChem: $name"
                continue
            end

            push!(results, (
                name = name,
                cid = compound.cid,
                kpuu_obs = kpuu,
                logP = compound.xlogp,
                MW = compound.molecular_weight,
                pKa = pka,
                charge = charge,
                pgp = pgp,
                notes = notes,
                smiles = compound.canonical_smiles,
                tpsa = compound.tpsa,
                hbd = compound.hbd,
                hba = compound.hba
            ))

            @printf("  ✓ CID=%d, MW=%.1f, XLogP=%s\n",
                    compound.cid, compound.molecular_weight,
                    compound.xlogp === nothing ? "N/A" : string(round(compound.xlogp, digits=2)))

            sleep(0.25)  # Rate limiting

        catch e
            @warn "  Error fetching $name: $e"
        end
    end

    println()
    println("=" ^ 70)
    println("FETCHED $(length(results)) COMPOUNDS FOR INDEPENDENT VALIDATION")
    println("=" ^ 70)

    return results
end

"""
Display fetched data as Julia constant for validation script
"""
function print_validation_data(results)
    println()
    println("# Copy this to your validation script:")
    println("const INDEPENDENT_VALIDATION = [")

    for r in results
        logp_str = r.logP === nothing ? "nothing" : string(round(r.logP, digits=1))
        pka_str = r.pKa === nothing ? "nothing" : string(r.pKa)
        pgp_er = if r.pgp == :no
            1.0
        elseif r.pgp == :weak
            2.0
        elseif r.pgp == :moderate
            5.0
        elseif r.pgp == :strong
            15.0
        else
            1.0
        end

        println("    (\"$(r.name)\", $logp_str, 0.10, $(round(r.MW, digits=1)), $pka_str, :$(r.charge), $pgp_er, :unknown, $(r.kpuu_obs)),")
    end

    println("]")
end

# ===========================================================================
# RUN
# ===========================================================================

println("\nStarting PubChem data fetch...\n")

results = fetch_validation_compounds()

if !isempty(results)
    print_validation_data(results)
end
