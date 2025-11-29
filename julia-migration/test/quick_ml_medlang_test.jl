using DarwinPBPK
using DarwinPBPK.MedLang

println("="^60)
println("ML → MEDLANG INTEGRATION VERIFICATION")
println("="^60)

# Test parameter estimation functions
println("\n1. Testing Ka estimation...")
transporters = [:PEPT1, :OCT1]
km_values = Dict(:PEPT1 => 50.0, :OCT1 => 200.0)
ka = estimate_ka_from_transporters(transporters, km_values; baseline_ka=1.0)
println("   Ka from PEPT1+OCT1: $(round(ka, digits=2)) /h")

println("\n2. Testing Fg estimation...")
fg = estimate_fg_from_transporters([:PEPT1, :PGP]; cyp_substrate=:CYP3A4)
println("   Fg with P-gp + CYP3A4: $(round(fg, digits=2))")

println("\n3. Testing Fh estimation...")
fh = estimate_fh_from_clearance(27.0)  # Midazolam-like
println("   Fh (CLh=27 L/h): $(round(fh, digits=2))")

# Test MedLang code generation
println("\n4. Generating MedLang model from ML predictions...")
ml_result = (
    uptake_transporters = [:PEPT1, :OATP2B1],
    is_pgp_substrate = true,
    pgp_efflux_ratio = 8.5,
    carrier_km_values = Dict(:PEPT1 => 80.0, :OATP2B1 => 150.0, :PGP => 50.0),
    predictions = []
)

medlang_code = generate_full_medlang_model(
    ml_result;
    drug_name = "TestDrug",
    mw = 400.0,
    cl_hepatic = 15.0,
    cyp_substrate = :CYP3A4
)

println("\n--- Generated MedLang Model ---")
println(medlang_code)

# Compile and simulate
println("\n5. Compiling MedLang to PBPKParams...")
params = compile_model(medlang_code)
println("   Hepatic CL: $(params.clearance_hepatic) L/h")
println("   Liver Kp: $(params.partition_coeffs[2])")  # liver is index 2

println("\n6. Simulating oral dose with MedLang DSL...")
results = simulate_oral(medlang_code, 100.0; t_max=24.0, num_points=50)
println("   Cmax: $(round(results["cmax"], digits=3)) mg/L")
println("   Tmax: $(round(results["tmax"], digits=2)) h")
println("   AUC: $(round(results["auc"], digits=2)) mg*h/L")

println("\n" * "="^60)
println("ML -> MEDLANG INTEGRATION WORKING!")
println("="^60)
println("""
Architecture verified:
  TransporterPredictor (ML)
         |
  MLMedLangIntegration.generate_full_medlang_model()
         |
  MedLang DSL Code (route, absorption, firstpass blocks)
         |
  MedLang.compile_model() -> PBPKParams
         |
  MedLang.simulate_oral() -> C(t) profiles
""")
