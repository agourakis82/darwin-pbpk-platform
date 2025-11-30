# Test phenotype-aware DDI predictions

include("../src/DarwinPBPK/medlang/ddi_prediction.jl")
using .DDIPrediction

println("=" ^ 70)
println("PHENOTYPE-AWARE DDI PREDICTION TEST")
println("=" ^ 70)

println("\n1. Paroxetine + Desipramine (CYP2D6 substrate)")
println("-" ^ 50)

for phenotype in [:PM, :IM, :NM, :UM]
    result = predict_ddi_by_phenotype(:paroxetine, :desipramine, phenotype)
    println("\n  $phenotype ($(result.phenotype_name)):")
    println("    Baseline AUC multiplier: $(result.baseline_auc_multiplier)x")
    println("    DDI AUC ratio (NM reference): $(round(result.ddi_auc_ratio_nm, digits=1))x")
    println("    DDI AUC ratio in $phenotype: $(result.ddi_auc_ratio_in_phenotype)x")
    println("    Note: $(result.clinical_note)")
    println("    Population frequency: $(round(result.population_frequency*100, digits=1))%")
end

println("\n" * "=" ^ 70)
println("2. Quinidine + Codeine (CYP2D6 prodrug activation)")
println("-" ^ 50)

for phenotype in [:PM, :IM, :NM, :UM]
    result = predict_ddi_by_phenotype(:quinidine, :codeine, phenotype)
    if hasproperty(result, :clinical_note)
        println("\n  $phenotype: $(result.clinical_note)")
    else
        println("\n  $phenotype: Not a CYP2D6 substrate in our database")
    end
end

println("\n" * "=" ^ 70)
println("3. Clinical Implications Summary")
println("-" ^ 50)

println("\nCYP2D6 Phenotype Distribution (Caucasian):")
for (pheno, data) in CYP2D6_PHENOTYPES
    freq = round(data.frequency_caucasian * 100, digits=1)
    println("  $pheno: $(freq)% - $(data.description)")
end

println("\n" * "=" ^ 70)
