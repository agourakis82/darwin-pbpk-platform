using DarwinPBPK

# Test validation metrics
pred = [1.0, 2.0, 3.0, 4.0, 5.0]
obs = [1.1, 2.1, 2.9, 4.2, 4.8]

println("📊 REGULATORY METRICS VALIDATION:")

try
    fe = DarwinPBPK.Validation.fold_error(pred, obs)
    println("  Fold Error (FE): ", fe)
catch e
    println("  Fold Error: Error - ", e)
end

try
    gmfe = DarwinPBPK.Validation.geometric_mean_fold_error(pred, obs)
    println("  Geometric Mean FE: ", gmfe)
catch e
    println("  Geometric Mean FE: Error - ", e)
end

try
    r2 = DarwinPBPK.Validation.r_squared(pred, obs)
    println("  R²: ", r2)
catch e
    println("  R²: Error - ", e)
end

try
    within_2fold = DarwinPBPK.Validation.percent_within_fold(pred, obs, 2.0)
    println("  % within 2-fold: ", within_2fold, "%")
catch e
    println("  % within 2-fold: Error - ", e)
end

println("✅ Validation metrics test completed!")
