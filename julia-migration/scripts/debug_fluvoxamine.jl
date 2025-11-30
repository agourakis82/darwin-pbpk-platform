# Debug fluvoxamine + caffeine prediction

include("../src/DarwinPBPK/medlang/ddi_prediction.jl")
using .DDIPrediction

println("=== FLUVOXAMINE + CAFFEINE DEBUG ===\n")

# Check parameters
println("1. Inhibitor params (fluvoxamine):")
inhib = get_inhibitor_params(:fluvoxamine)
for (enzyme, params) in inhib
    println("   $enzyme: $params")
end

println("\n2. Substrate params (caffeine):")
sub = get_substrate_params(:caffeine)
println("   $sub")

println("\n3. Prediction:")
result = predict_ddi(:fluvoxamine, :caffeine)
pred = round(result.auc_ratio, digits=1)
println("   Predicted: $(pred)x")
println("   Observed: 5x")
println("   Mechanism: $(result.mechanism)")
println("   Enzyme: $(result.enzyme)")
println("   Params used: $(result.parameters_used)")
println("   Warnings: $(result.warnings)")

# The issue: fluvoxamine has auc_ratio=33.0 (for tizanidine)
# But clinical caffeine AUC ratio is only 5x
# This is because caffeine metabolism is NOT as sensitive to CYP1A2 inhibition
# Need substrate-specific calibration

println("\n=== ANALYSIS ===")
println("Fluvoxamine's auc_ratio=33.0 is calibrated for TIZANIDINE")
println("Caffeine clinical data shows only 5x increase")
println("This is NOT a modeling error - it's a data calibration issue")
println("Solution: Use substrate-specific sensitivity from FDA classification")
