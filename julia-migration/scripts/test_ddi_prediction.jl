# Test DDI prediction module
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src", "DarwinPBPK", "medlang"))
include(joinpath(@__DIR__, "..", "src", "DarwinPBPK", "medlang", "ddi_prediction.jl"))
using .DDIPrediction

println("=" ^ 60)
println("DDI PREDICTION MODULE TEST")
println("=" ^ 60)

# Test 1: Strong CYP3A4 inhibition
println("\n1. Itraconazole + Midazolam (strong CYP3A4 inhibition)")
result = predict_ddi(:itraconazole, :midazolam)
println("   Predicted AUC ratio: ", round(result.auc_ratio, digits=1))
println("   Clinical observed: ~10x")
println("   Mechanism: ", result.mechanism)

# Test 2: MBI inhibition
println("\n2. Clarithromycin + Midazolam (MBI)")
result = predict_ddi(:clarithromycin, :midazolam)
println("   Predicted AUC ratio: ", round(result.auc_ratio, digits=1))
println("   Clinical observed: ~8x")
println("   Mechanism: ", result.mechanism)

# Test 3: Ritonavir (potent MBI)
println("\n3. Ritonavir + Midazolam (potent MBI)")
result = predict_ddi(:ritonavir, :midazolam)
println("   Predicted AUC ratio: ", round(result.auc_ratio, digits=1))
println("   Clinical observed: ~28x")
println("   Mechanism: ", result.mechanism)

# Test 4: Strong induction
println("\n4. Rifampin + Midazolam (strong induction)")
result = predict_ddi(:rifampin, :midazolam)
println("   Predicted AUC ratio: ", round(result.auc_ratio, digits=3))
println("   Clinical observed: ~0.04")
println("   Mechanism: ", result.mechanism)

# Test 5: CYP2D6 inhibition
println("\n5. Quinidine + Desipramine (CYP2D6)")
result = predict_ddi(:quinidine, :desipramine)
println("   Predicted AUC ratio: ", round(result.auc_ratio, digits=1))
println("   Clinical observed: ~7.5x")

# Test 6: Moderate inhibition
println("\n6. Erythromycin + Midazolam (moderate MBI)")
result = predict_ddi(:erythromycin, :midazolam)
println("   Predicted AUC ratio: ", round(result.auc_ratio, digits=1))
println("   Clinical observed: ~4.4x")

# Test 7: CYP2C8 MBI (gemfibrozil)
println("\n7. Gemfibrozil + Repaglinide (CYP2C8 MBI)")
result = predict_ddi(:gemfibrozil_glucuronide, :repaglinide)
println("   Predicted AUC ratio: ", round(result.auc_ratio, digits=1))
println("   Clinical observed: ~8x")

# Validate all predictions
println("\n" * "=" ^ 60)
println("VALIDATION AGAINST CLINICAL DATA")
println("=" ^ 60)
metrics = validate_predictions()
println("Number of CYP-mediated DDI studies: ", metrics.n)
println("Average Fold Error (bias): ", metrics.AFE)
println("Absolute Average Fold Error (precision): ", metrics.AAFE)
println("Within 2-fold accuracy: ", metrics.within_2fold, "%")
println("Within 3-fold accuracy: ", metrics.within_3fold, "%")

# Get detailed validation
println("\n" * "=" ^ 60)
println("DETAILED PREDICTIONS VS OBSERVED")
println("=" ^ 60)
validation = validate_predictions(verbose=true)
println("\nTop overpredictions:")
sorted_by_error = sort(validation.details, by=x->-x.fold_error)
for (i, d) in enumerate(sorted_by_error[1:min(5, length(sorted_by_error))])
    println("  $(d.perpetrator) + $(d.victim): predicted=$(round(d.predicted, digits=1)), observed=$(round(d.observed, digits=1)), error=$(round(d.fold_error, digits=1))x")
end
