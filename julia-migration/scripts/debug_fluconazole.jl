# Debug fluconazole predictions

include("../src/DarwinPBPK/medlang/ddi_prediction.jl")
using .DDIPrediction

println("=== FLUCONAZOLE DEBUG ===")

# Check inhibitor params
inhib = get_inhibitor_params(:fluconazole)
println("\nFluconazole inhibits:")
for (enz, params) in inhib
    ki = params.ki_um
    auc = params.auc_ratio
    println("  $enz: Ki=$(ki)uM, auc_ratio=$(auc)")
end

# Check warfarin substrate
sub = get_substrate_params(:warfarin_s)
println("\nWarfarin_s substrate params: $sub")

# Predict
result = predict_ddi(:fluconazole, :warfarin_s)
auc = round(result.auc_ratio, digits=2)
println("\nPrediction for fluconazole + warfarin_s:")
println("  AUC ratio: $auc")
println("  Enzyme: $(result.enzyme)")
println("  Mechanism: $(result.mechanism)")
println("  Params: $(result.parameters_used)")
println("  Clinical observed: 2.3x")

# The issue: fluconazole has entries for CYP2C9, CYP2C19, and CYP3A4
# Warfarin_s is primarily CYP2C9 (should use that Ki/auc_ratio)
# But Dict iteration order may be picking CYP3A4 first

println("\n=== CHECKING ENZYME MATCHING ===")

# Check what FDA substrates say about warfarin
for (enzyme, substrates) in FDA_CYP_SUBSTRATES
    if haskey(substrates, :warfarin_s)
        println("Warfarin_s is in FDA_CYP_SUBSTRATES[$enzyme]: $(substrates[:warfarin_s])")
    end
end

# Check midazolam too
println("\n=== FLUCONAZOLE + MIDAZOLAM ===")
result = predict_ddi(:fluconazole, :midazolam)
auc = round(result.auc_ratio, digits=2)
println("  AUC ratio: $auc (observed: 3.5x)")
println("  Enzyme: $(result.enzyme)")
