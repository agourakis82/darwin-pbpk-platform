#!/usr/bin/env julia
# Test External PK Datasets

using HTTP, JSON3, CSV, DataFrames, Statistics

DATA_DIR = joinpath(@__DIR__, "..", "data", "external_pk_datasets")

println("=" ^ 60)
println("EXTERNAL PK DATASETS - VALIDATION TEST")
println("=" ^ 60)

# Load datasets
println("\n[1] Loading OSP DDI Database...")
ddi = CSV.read(joinpath(DATA_DIR, "OSP_DDI.csv"), DataFrame)
println("    Records: $(nrow(ddi))")
println("    Unique victims: $(length(unique(ddi.Victim)))")
println("    Unique perpetrators: $(length(unique(ddi.Perpetrator)))")

println("\n[2] Loading OSP Pediatrics Database...")
ped = CSV.read(joinpath(DATA_DIR, "OSP_Pediatrics.csv"), DataFrame)
println("    Records: $(nrow(ped))")
println("    Unique analytes: $(unique(ped.Analyte))")

println("\n[3] Loading Zenodo Beta-Lactam ICU Dataset...")
cov = CSV.read(joinpath(DATA_DIR, "Zenodo_BetaLactam_CriticallyIll_covariates.csv"), DataFrame)
out = CSV.read(joinpath(DATA_DIR, "Zenodo_BetaLactam_CriticallyIll_outcomes.csv"), DataFrame)
println("    Studies: $(nrow(cov))")
println("    Outcome records: $(nrow(out))")
println("    Beta-lactams: $(unique(cov.betalactam_studied))")

# DDI Analysis
println("\n" * "=" ^ 60)
println("DDI AUC RATIO ANALYSIS")
println("=" ^ 60)

auc_col = Symbol("AUCR Avg")

# Midazolam + Itraconazole (CYP3A4 inhibition)
itra = filter(row ->
    lowercase(string(row.Victim)) == "midazolam" &&
    lowercase(string(row.Perpetrator)) == "itraconazole",
    ddi
)
auc_vals = [row[auc_col] for row in eachrow(itra) if !ismissing(row[auc_col])]
println("\nMidazolam + Itraconazole (CYP3A4 inhibition):")
println("  N studies: $(length(auc_vals))")
println("  Mean AUC ratio: $(round(mean(auc_vals), digits=2))")
println("  Range: $(round(minimum(auc_vals), digits=2)) - $(round(maximum(auc_vals), digits=2))")
println("  Expected: >3 (strong inhibition)")
println("  PASS: $(mean(auc_vals) > 3 ? "YES" : "NO")")

# Midazolam + Rifampicin (CYP3A4 induction)
rifamp = filter(row ->
    lowercase(string(row.Victim)) == "midazolam" &&
    lowercase(string(row.Perpetrator)) == "rifampicin",
    ddi
)
auc_vals2 = [row[auc_col] for row in eachrow(rifamp) if !ismissing(row[auc_col])]
println("\nMidazolam + Rifampicin (CYP3A4 induction):")
println("  N studies: $(length(auc_vals2))")
println("  Mean AUC ratio: $(round(mean(auc_vals2), digits=3))")
println("  Range: $(round(minimum(auc_vals2), digits=3)) - $(round(maximum(auc_vals2), digits=3))")
println("  Expected: <0.2 (strong induction)")
println("  PASS: $(mean(auc_vals2) < 0.2 ? "YES" : "NO")")

# PK-DB API
println("\n" * "=" ^ 60)
println("PK-DB REST API TEST")
println("=" ^ 60)

try
    response = HTTP.get(
        "https://pk-db.com/api/v1/studies/?format=json&page=1",
        headers=["Accept" => "application/json"],
        readtimeout=15
    )
    data = JSON3.read(String(response.body))

    println("\nAPI Status: CONNECTED")
    println("Total studies in database: $(data.data.count)")
    println("\nSample studies:")
    for (i, study) in enumerate(data.data.data[1:5])
        substances = haskey(study, :substances) ?
            join([s.name for s in study.substances], ", ") : "N/A"
        println("  $i. $(study.name)")
        println("     Individuals: $(study.individual_count), Outputs: $(study.output_count)")
        println("     Substances: $substances")
    end
catch e
    println("\nAPI Status: UNAVAILABLE")
    println("Error: $e")
end

# Summary
println("\n" * "=" ^ 60)
println("VALIDATION DATA SUMMARY")
println("=" ^ 60)
total = nrow(ddi) + nrow(ped) + nrow(cov) + nrow(out)
println("\nDataset                    Records")
println("-" ^ 40)
println("OSP DDI interactions       $(nrow(ddi))")
println("OSP Pediatric PK           $(nrow(ped))")
println("Zenodo ICU studies         $(nrow(cov))")
println("Zenodo outcome records     $(nrow(out))")
println("-" ^ 40)
println("TOTAL                      $total")
println("\n✓ External validation datasets ready!")
println("=" ^ 60)
