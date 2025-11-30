# Test multi-perpetrator DDI prediction

include("../src/DarwinPBPK/medlang/ddi_prediction.jl")
using .DDIPrediction

println("=" ^ 60)
println("MULTI-PERPETRATOR DDI TEST")
println("=" ^ 60)

# Test 1: Two CYP3A4 inhibitors
println("\n1. Clarithromycin + Diltiazem + Midazolam")
result = predict_multi_ddi([:clarithromycin, :diltiazem], :midazolam)
println("   Individual effects:")
for r in result.individual_results
    ratio = round(r.auc_ratio, digits=1)
    println("     $(r.perpetrator): $(ratio)x")
end
combined = round(result.combined_auc_ratio, digits=1)
println("   Combined AUC ratio: $(combined)x")
println("   Net effect: $(result.net_effect)")
println("   Significance: $(result.clinical_significance)")

# Test 2: Inhibitor + Inducer
println("\n2. Ritonavir + Rifampin + Midazolam (HIV/TB co-treatment)")
result = predict_multi_ddi([:ritonavir, :rifampin], :midazolam)
println("   Individual effects:")
for r in result.individual_results
    ratio = round(r.auc_ratio, digits=2)
    println("     $(r.perpetrator): $(ratio)x ($(r.mechanism))")
end
combined = round(result.combined_auc_ratio, digits=2)
println("   Combined AUC ratio: $(combined)x")
println("   Net effect: $(result.net_effect)")
println("   Warnings: $(result.warnings)")

# Test 3: Multi-pathway inhibition
println("\n3. Fluconazole + Paroxetine + Codeine (CYP3A4 + CYP2D6)")
result = predict_multi_ddi([:fluconazole, :paroxetine], :codeine)
println("   Individual effects:")
for r in result.individual_results
    ratio = round(r.auc_ratio, digits=1)
    println("     $(r.perpetrator) via $(r.enzyme): $(ratio)x")
end
combined = round(result.combined_auc_ratio, digits=1)
println("   Combined AUC ratio: $(combined)x")
println("   Significance: $(result.clinical_significance)")

println("\n" * "=" ^ 60)
