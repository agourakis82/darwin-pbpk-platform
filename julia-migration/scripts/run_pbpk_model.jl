#!/usr/bin/env julia
#=
Run MedLang PBPK Model
=#

using Pkg
Pkg.activate("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration")

# Include modules
include("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration/src/DarwinPBPK/ode_solver.jl")
using .ODEPBPKSolver

include("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration/src/DarwinPBPK/medlang/parser.jl")
include("/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration/src/DarwinPBPK/medlang/transpiler.jl")

using .MedLangParser
using .MedLangTranspiler

println("=" ^ 60)
println("MedLang PBPK Model Execution")
println("=" ^ 60)

# Load and parse the MedLang file
medlang_file = "/home/agourakis82/workspace/darwin-pbpk-platform/julia-migration/src/DarwinPBPK/medlang/examples/standard_pbpk.medlang"
println("\nLoading: ", medlang_file)

source = read(medlang_file, String)
ast = parse_medlang(source)

println("\nParsed models: ", length(ast.models))
for name in keys(ast.models)
    println("  - ", name)
end

# Transpile to PBPKParams
println("\nTranspiling StandardPBPK_14Compartment...")
params = transpile_to_pbpk_params(source; model_name="StandardPBPK_14Compartment")

println("\nPBPK Parameters:")
println("  Hepatic CL: ", params.clearance_hepatic, " L/h")
println("  Renal CL: ", params.clearance_renal, " L/h")
println("  Blood volume: ", params.volumes[1], " L")
println("  Liver volume: ", params.volumes[2], " L")

# Run simulation
println("\n" * "=" ^ 60)
println("Running PBPK Simulation")
println("=" ^ 60)

dose = 100.0  # mg IV bolus

println("\nDose: ", dose, " mg IV")
println("Time points: 0 to 24 h (100 points)")

result = simulate(params, dose; t_max=24.0, num_points=100)

time_points = result["time"]
plasma = result["blood"]

# Calculate PK metrics
cmax = maximum(plasma)
cmax_idx = argmax(plasma)
tmax = time_points[cmax_idx]

# AUC by trapezoidal rule
auc = sum(0.5 * (plasma[i] + plasma[i-1]) * (time_points[i] - time_points[i-1]) for i in 2:length(time_points))

println("\nPK Results:")
println("  Cmax: ", round(cmax, digits=3), " mg/L")
println("  Tmax: ", round(tmax, digits=2), " h")
println("  AUC(0-24h): ", round(auc, digits=2), " mg*h/L")

# Calculate half-life
terminal_idx = findfirst(c -> c < plasma[cmax_idx] / 2, plasma[cmax_idx:end])
if terminal_idx !== nothing
    t_half = time_points[cmax_idx + terminal_idx - 1] - time_points[cmax_idx]
    println("  t1/2 (approx): ", round(t_half, digits=2), " h")
end

# Print concentration profile
println("\nConcentration Profile (Plasma):")
println("  Time (h)    Conc (mg/L)")
println("  " * "-" ^ 25)
for i in 1:10:length(time_points)
    t = time_points[i]
    c = plasma[i]
    println("  ", lpad(round(t, digits=1), 6), "    ", round(c, digits=4))
end

# Organ concentrations at Tmax
println("\nOrgan Concentrations at Tmax:")
for organ in ["liver", "kidney", "brain", "muscle", "adipose"]
    c = result[organ][cmax_idx]
    println("  ", rpad(organ, 10), ": ", round(c, digits=4), " mg/L")
end

println("\n" * "=" ^ 60)
println("Simulation Complete!")
println("=" ^ 60)
