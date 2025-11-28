"""Debug MedLang validation"""

include(joinpath(@__DIR__, "..", "..", "src", "DarwinPBPK.jl"))
using .DarwinPBPK
using .DarwinPBPK.MedLang
using .DarwinPBPK.ODEPBPKSolver

# Try JSON
try
    @eval using JSON
catch
    println("JSON not available, using simple approach")
end

# Load one drug
data = JSON.parsefile("/mnt/f/DARWIN_VALIDATION/datasets/ULTIMATE_DATASET_v1_normalized_with_smiles.json")
drug = data[1]

println("Drug: ", drug["drug_name"])

name = get(drug, "drug_name", "Unknown")
safe_name = replace(name, r"[^a-zA-Z0-9_]" => "_")

cl = get(drug, "CL_lit", 10.0)
vd = get(drug, "Vd_lit", 50.0)
f = get(drug, "bioavailability", 0.8)
fu = get(drug, "fu", 0.5)

cl = cl === nothing ? 10.0 : Float64(cl)
vd = vd === nothing ? 50.0 : Float64(vd)
f = f === nothing ? 0.8 : Float64(f)
fu = fu === nothing ? 0.5 : Float64(fu)

println("Raw params: cl=$cl, vd=$vd, f=$f, fu=$fu")

cl = max(cl, 0.1)
vd = max(vd, 1.0)
f = clamp(f, 0.01, 1.0)
fu = clamp(fu, 0.001, 1.0)

cl_hepatic = cl * (1.0 - fu * 0.3)
cl_renal = cl * fu * 0.3

base_kp = max(0.1, vd / 50.0)
safe_kp(x) = max(0.01, round(x, digits=3))

cl_hepatic = max(0.01, cl_hepatic)
cl_renal = max(0.0, cl_renal)

println("Calculated: cl_hepatic=$cl_hepatic, cl_renal=$cl_renal, base_kp=$base_kp")

# Build MedLang source
liver_kp = safe_kp(base_kp * 1.2)
kidney_kp = safe_kp(base_kp * 0.8)
brain_kp = safe_kp(base_kp * 0.3 * max(fu, 0.01))

println("Kp values: liver=$liver_kp, kidney=$kidney_kp, brain=$brain_kp")

medlang_source = """
model $(safe_name)_PBPK {
    clearance hepatic: $(cl_hepatic)_L/h
    clearance renal: $(cl_renal)_L/h

    organ blood { V: 5.0_L, Q: 0.0_L/h, Kp: 1.0 }
    organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: $liver_kp }
    organ kidney { V: 0.31_L, Q: 60.0_L/h, Kp: $kidney_kp }
    organ brain { V: 1.4_L, Q: 50.0_L/h, Kp: $brain_kp }
    organ heart { V: 0.33_L, Q: 20.0_L/h, Kp: $(safe_kp(base_kp * 0.9)) }
    organ lung { V: 0.5_L, Q: 300.0_L/h, Kp: $(safe_kp(base_kp * 0.7)) }
    organ muscle { V: 30.0_L, Q: 75.0_L/h, Kp: $(safe_kp(base_kp * 0.6)) }
    organ adipose { V: 15.0_L, Q: 12.0_L/h, Kp: $(safe_kp(base_kp * 1.5)) }
    organ gut { V: 1.1_L, Q: 45.0_L/h, Kp: $(safe_kp(base_kp * 0.8)) }
    organ skin { V: 3.3_L, Q: 10.0_L/h, Kp: $(safe_kp(base_kp * 0.5)) }
    organ bone { V: 10.0_L, Q: 5.0_L/h, Kp: $(safe_kp(base_kp * 0.2)) }
    organ spleen { V: 0.18_L, Q: 15.0_L/h, Kp: $(safe_kp(base_kp * 0.9)) }
    organ pancreas { V: 0.1_L, Q: 5.0_L/h, Kp: $(safe_kp(base_kp * 0.7)) }
    organ other { V: 5.0_L, Q: 20.0_L/h, Kp: $(safe_kp(base_kp * 0.5)) }
}
"""

println("\n=== Generated MedLang ===")
println(medlang_source)
println("=========================\n")

println("Compiling MedLang model...")
try
    params = compile_model(medlang_source)
    println("SUCCESS! Compiled to PBPKParams")
    println("  clearance_hepatic: ", params.clearance_hepatic)
    println("  clearance_renal: ", params.clearance_renal)
    println("  volumes: ", params.volumes)
    println("  partition_coeffs: ", params.partition_coeffs)

    dose = Float64(get(drug, "dose", 100.0))
    println("\nSimulating with dose=$dose mg...")
    results = ODEPBPKSolver.simulate(params, dose; t_max=24.0, num_points=100)

    println("Blood concentration range: ", minimum(results["blood"]), " - ", maximum(results["blood"]))
    println("Cmax: ", maximum(results["blood"]))

catch e
    println("ERROR: ", e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
