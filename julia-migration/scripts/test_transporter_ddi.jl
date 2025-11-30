# Test transporter DDI predictions

include("../src/DarwinPBPK/medlang/ddi_prediction.jl")
using .DDIPrediction

println("=" ^ 60)
println("TRANSPORTER DDI PREDICTION TEST")
println("=" ^ 60)

# Test P-gp interactions
println("\n1. Cyclosporine + Digoxin (P-gp inhibition)")
result = predict_transporter_ddi(:cyclosporine, :digoxin)
println("   Predicted AUC ratio: $(round(result.auc_ratio, digits=2))x")
println("   Transporter: $(result.enzyme)")
println("   Clinical observed: ~2.5x")

println("\n2. Ritonavir + Digoxin (P-gp)")
result = predict_transporter_ddi(:ritonavir, :digoxin)
println("   Predicted AUC ratio: $(round(result.auc_ratio, digits=2))x")
println("   Transporter: $(result.enzyme)")
println("   Clinical observed: ~2.0x")

# Test OATP1B1 interactions
println("\n3. Cyclosporine + Rosuvastatin (OATP1B1/BCRP)")
result = predict_transporter_ddi(:cyclosporine, :rosuvastatin)
println("   Predicted AUC ratio: $(round(result.auc_ratio, digits=2))x")
println("   Transporter: $(result.enzyme)")
println("   Clinical observed: ~7x")

println("\n4. Rifampin (single dose) + Repaglinide (OATP1B1)")
result = predict_transporter_ddi(:rifampin, :repaglinide)
println("   Predicted AUC ratio: $(round(result.auc_ratio, digits=2))x")
println("   Note: Single dose rifampin inhibits OATP1B1 before induction kicks in")

# Combined CYP + transporter
println("\n" * "=" ^ 60)
println("COMBINED CYP + TRANSPORTER DDI")
println("=" ^ 60)

println("\n5. Gemfibrozil + Repaglinide (CYP2C8 + OATP1B1)")
cyp_result = predict_ddi(:gemfibrozil, :repaglinide)
transporter_result = predict_transporter_ddi(:gemfibrozil, :repaglinide)
combined = cyp_result.auc_ratio * transporter_result.auc_ratio
println("   CYP2C8 contribution: $(round(cyp_result.auc_ratio, digits=2))x")
println("   OATP1B1 contribution: $(round(transporter_result.auc_ratio, digits=2))x")
println("   Combined prediction: $(round(combined, digits=2))x")
println("   Clinical observed: 8.1x")

println("\n" * "=" ^ 60)
