# Analyze remaining prediction errors to identify improvements

include("../src/DarwinPBPK/medlang/ddi_prediction.jl")
using .DDIPrediction

println("=" ^ 70)
println("DETAILED ERROR ANALYSIS")
println("=" ^ 70)

# Get detailed validation
results = validate_predictions(verbose=true)

println("\nOverall Metrics:")
println("  AFE (bias): $(results.metrics.AFE)")
println("  AAFE (precision): $(results.metrics.AAFE)")
println("  Within 2-fold: $(results.metrics.within_2fold)%")
println("  Within 3-fold: $(results.metrics.within_3fold)%")

# Sort by fold error
sorted = sort(results.details, by=x->x.fold_error, rev=true)

println("\n" * "=" ^ 70)
println("TOP 10 PREDICTION ERRORS (sorted by fold error)")
println("=" ^ 70)

for (i, d) in enumerate(sorted[1:min(10, length(sorted))])
    direction = d.predicted > d.observed ? "OVER" : "UNDER"
    println("\n$i. $(d.perpetrator) + $(d.victim)")
    println("   Predicted: $(round(d.predicted, digits=2))x")
    println("   Observed:  $(round(d.observed, digits=2))x")
    println("   Fold error: $(round(d.fold_error, digits=2))x ($direction)")

    # Debug the prediction
    result = predict_ddi(d.perpetrator, d.victim)
    println("   Mechanism: $(result.mechanism)")
    println("   Enzyme: $(result.enzyme)")
    if !isempty(result.warnings)
        println("   Warnings: $(result.warnings)")
    end
end

println("\n" * "=" ^ 70)
println("PREDICTIONS WITHIN 2-FOLD (GOOD)")
println("=" ^ 70)

good = filter(x -> x.fold_error <= 2.0, sorted)
println("\n$(length(good)) predictions within 2-fold:")
for d in good
    println("  $(d.perpetrator) + $(d.victim): pred=$(round(d.predicted, digits=1))x, obs=$(round(d.observed, digits=1))x")
end

println("\n" * "=" ^ 70)
println("ANALYSIS BY MECHANISM")
println("=" ^ 70)

# Group by mechanism
mechanisms = Dict{Symbol, Vector{NamedTuple}}()
for d in results.details
    result = predict_ddi(d.perpetrator, d.victim)
    mech = result.mechanism
    if !haskey(mechanisms, mech)
        mechanisms[mech] = []
    end
    push!(mechanisms[mech], d)
end

for (mech, details) in mechanisms
    errors = [d.fold_error for d in details]
    within2 = count(e -> e <= 2.0, errors) / length(errors) * 100
    println("\n$mech (n=$(length(details))):")
    println("  Mean fold error: $(round(mean(errors), digits=2))")
    println("  Within 2-fold: $(round(within2, digits=1))%")
end

using Statistics
