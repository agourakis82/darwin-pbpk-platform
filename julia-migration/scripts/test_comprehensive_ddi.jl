# Test comprehensive DDI predictions

include("../src/DarwinPBPK/medlang/ddi_prediction.jl")
using .DDIPrediction

println("=" ^ 70)
println("COMPREHENSIVE DDI PREDICTION TEST")
println("=" ^ 70)

# Test gemfibrozil + repaglinide (CYP2C8 MBI + OATP1B1)
println("\n1. Gemfibrozil + Repaglinide (CYP2C8 + OATP1B1)")
comp = predict_ddi_comprehensive(:gemfibrozil, :repaglinide)
println("   CYP2C8 contribution: $(round(comp.cyp_result.auc_ratio, digits=2))x")
println("   OATP1B1 contribution: $(round(comp.transporter_result.auc_ratio, digits=2))x")
println("   Combined prediction: $(round(comp.result.auc_ratio, digits=2))x")
println("   Clinical observed: 8.1x")
println("   Fold error: $(round(max(comp.result.auc_ratio/8.1, 8.1/comp.result.auc_ratio), digits=2))x")

# Test cyclosporine + repaglinide
println("\n2. Cyclosporine + Repaglinide (CYP3A4 + OATP1B1)")
comp = predict_ddi_comprehensive(:cyclosporine, :repaglinide)
println("   CYP contribution: $(round(comp.cyp_result.auc_ratio, digits=2))x")
println("   OATP1B1 contribution: $(round(comp.transporter_result.auc_ratio, digits=2))x")
println("   Combined prediction: $(round(comp.result.auc_ratio, digits=2))x")
println("   Clinical observed: ~2.4x")

# Test rifampin single dose (OATP1B1 inhibitor before induction)
println("\n3. Rifampin (single dose) + Atorvastatin (OATP1B1)")
# Single dose rifampin is an OATP1B1 inhibitor
trans_result = predict_transporter_ddi(:rifampin, :atorvastatin)
println("   OATP1B1 inhibition: $(round(trans_result.auc_ratio, digits=2))x")
println("   Clinical observed (single dose): ~7x")

# Test ritonavir + digoxin (P-gp)
println("\n4. Ritonavir + Digoxin (P-gp)")
comp = predict_ddi_comprehensive(:ritonavir, :digoxin)
println("   CYP contribution: $(round(comp.cyp_result.auc_ratio, digits=2))x")
println("   P-gp contribution: $(round(comp.transporter_result.auc_ratio, digits=2))x")
println("   Combined: $(round(comp.result.auc_ratio, digits=2))x")
println("   Clinical observed: ~2x")

println("\n" * "=" ^ 70)
println("VALIDATION: Predictions for dual-mechanism DDIs")
println("=" ^ 70)

# List of known dual-mechanism DDIs
dual_ddis = [
    (:gemfibrozil, :repaglinide, 8.1, "CYP2C8 + OATP1B1"),
    (:cyclosporine, :rosuvastatin, 7.1, "OATP1B1 + BCRP"),
    (:ritonavir, :digoxin, 2.0, "CYP3A4 + P-gp"),
]

within_2fold = 0
for (perp, vic, observed, mechanism) in dual_ddis
    comp = predict_ddi_comprehensive(perp, vic)
    predicted = comp.result.auc_ratio
    fold_error = max(predicted/observed, observed/predicted)
    status = fold_error <= 2.0 ? "OK" : "MISS"
    within_2fold += fold_error <= 2.0 ? 1 : 0

    println("\n$perp + $vic ($mechanism)")
    println("  Predicted: $(round(predicted, digits=2))x | Observed: $(observed)x | Error: $(round(fold_error, digits=2))x [$status]")
end

println("\n" * "=" ^ 70)
println("Within 2-fold: $(within_2fold)/$(length(dual_ddis)) ($(round(within_2fold/length(dual_ddis)*100, digits=1))%)")
println("=" ^ 70)
