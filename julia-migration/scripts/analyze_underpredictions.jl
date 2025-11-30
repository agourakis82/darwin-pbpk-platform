# Analyze underpredictions in DDI system

include("../src/DarwinPBPK/medlang/ddi_prediction.jl")
using .DDIPrediction

println("=== ANALYZING UNDERPREDICTIONS ===\n")

# Check ciprofloxacin + tizanidine
println("1. Ciprofloxacin + Tizanidine (CYP1A2)")
result = predict_ddi(:ciprofloxacin, :tizanidine)
pred = round(result.auc_ratio, digits=2)
println("   Predicted: $(pred)x, Observed: 10x")
println("   Params: $(result.parameters_used)")
println("   Warnings: $(result.warnings)")

# Ciprofloxacin concentration check
if haskey(TYPICAL_CLINICAL_CMAX, :ciprofloxacin)
    pk = TYPICAL_CLINICAL_CMAX[:ciprofloxacin]
    unbound = pk.fu_p * pk.cmax_um
    println("   Cipro Cmax: $(pk.cmax_um) uM, fu: $(pk.fu_p), unbound: $(unbound) uM")
end

println("\n2. Gemfibrozil + Repaglinide (CYP2C8 + OATP1B1)")
result = predict_ddi(:gemfibrozil, :repaglinide)
pred = round(result.auc_ratio, digits=2)
println("   Predicted: $(pred)x, Observed: 8.1x")
println("   Mechanism: $(result.mechanism)")
println("   Params: $(result.parameters_used)")

# Check gemfibrozil - it's MBI on CYP2C8
inhib = get_inhibitor_params(:gemfibrozil)
println("   Gemfibrozil inhibitor params: ", inhib)

println("\n3. Quinidine + Dextromethorphan (CYP2D6)")
result = predict_ddi(:quinidine, :dextromethorphan)
pred = round(result.auc_ratio, digits=2)
println("   Predicted: $(pred)x, Observed: 30x")
println("   Params: $(result.parameters_used)")

# Check dextromethorphan fm
sub = get_substrate_params(:dextromethorphan)
println("   DXM substrate params: ", sub)

println("\n4. Fluvoxamine + Tizanidine (CYP1A2)")
result = predict_ddi(:fluvoxamine, :tizanidine)
pred = round(result.auc_ratio, digits=2)
println("   Predicted: $(pred)x, Observed: 33x")
println("   Params: $(result.parameters_used)")

# Check what fm values we have
println("\n=== FM VALUES CHECK ===")
println("Tizanidine: ", get_substrate_params(:tizanidine))
println("Dextromethorphan: ", get_substrate_params(:dextromethorphan))
println("Repaglinide: ", get_substrate_params(:repaglinide))
