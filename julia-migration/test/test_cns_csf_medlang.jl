using DarwinPBPK
using DarwinPBPK.MedLang

println("="^70)
println("CNS/CSF COMPARTMENT MODEL IN MEDLANG")
println("="^70)

# ===========================================================================
# Test 1: Risperidone (P-gp substrate - antipsychotic)
# ===========================================================================
println("\n1. Testing Risperidone (P-gp substrate, antipsychotic)...")

risperidone = drug_preset(:risperidone)

println("   Drug: $(risperidone.drug_name)")
println("   MW: $(risperidone.MW) Da")
println("   logP: $(risperidone.logP)")
println("   P-gp substrate: $(risperidone.is_pgp_substrate)")
println("   fu,plasma: $(risperidone.fu_plasma)")
println("   fu,brain: $(risperidone.fu_brain)")
println("   Kp,brain: $(risperidone.Kp_brain)")

# Calculate Kp,uu values
bbb = default_bbb_transporters()
bcsfb = default_bcsfb_transporters()

kpuu_bbb = calculate_kpuu_bbb(risperidone, bbb)
kpuu_bcsfb = calculate_kpuu_bcsfb(risperidone, bcsfb)

println("\n   Calculated Kp,uu values:")
println("   - Kp,uu,BBB: $(round(kpuu_bbb, digits=3)) (P-gp restricts brain entry)")
println("   - Kp,uu,BCSFB: $(round(kpuu_bcsfb, digits=3)) (P-gp pumps INTO CSF!)")

# Simulate
println("\n   Simulating 4 mg IV dose...")
results = simulate_cns_distribution(risperidone, 4.0; t_max_h=24.0)

println("   ECF/CSF ratio: $(round(results["ECF_to_CSF_ratio"], digits=2))")
println("   Peak Cu,brain_ECF: $(round(maximum(results["Cu_brain_ECF"]), digits=4)) uM")
println("   Peak C,CSF_CM (cisternal): $(round(maximum(results["C_CSF_CM"]), digits=4)) uM")
println("   Peak C,CSF_SAS (lumbar): $(round(maximum(results["C_CSF_SAS"]), digits=4)) uM")

# ===========================================================================
# Test 2: Generate MedLang DSL code
# ===========================================================================
println("\n2. Generating MedLang DSL code for Risperidone...")

medlang_code = generate_cns_medlang(risperidone)

println("\n--- Generated MedLang (first 80 lines) ---")
lines = split(medlang_code, "\n")
for line in lines[1:min(80, length(lines))]
    println(line)
end
println("... ($(length(lines)) total lines)")

# ===========================================================================
# Test 3: Morphine (P-gp substrate, opioid - cisternal relevant)
# ===========================================================================
println("\n3. Testing Morphine (hydrophilic, P-gp substrate)...")

morphine = drug_preset(:morphine)

println("   Drug: $(morphine.drug_name)")
println("   logP: $(morphine.logP) (hydrophilic)")
println("   fu,plasma: $(morphine.fu_plasma) (low binding)")

kpuu_bbb_morph = calculate_kpuu_bbb(morphine, bbb)
kpuu_bcsfb_morph = calculate_kpuu_bcsfb(morphine, bcsfb)

println("   Kp,uu,BBB: $(round(kpuu_bbb_morph, digits=3))")
println("   Kp,uu,BCSFB: $(round(kpuu_bcsfb_morph, digits=3))")

results_morph = simulate_cns_distribution(morphine, 10.0; t_max_h=12.0)
println("   ECF/CSF ratio: $(round(results_morph["ECF_to_CSF_ratio"], digits=2))")

# ===========================================================================
# Test 4: Gabapentin (LAT1 substrate, not P-gp)
# ===========================================================================
println("\n4. Testing Gabapentin (LAT1 substrate, zwitterion)...")

gabapentin = drug_preset(:gabapentin)

println("   Drug: $(gabapentin.drug_name)")
println("   logP: $(gabapentin.logP) (very hydrophilic)")
println("   P-gp substrate: $(gabapentin.is_pgp_substrate)")
println("   LAT1 substrate: $(gabapentin.is_lat1_substrate)")

kpuu_bbb_gaba = calculate_kpuu_bbb(gabapentin, bbb)
kpuu_bcsfb_gaba = calculate_kpuu_bcsfb(gabapentin, bcsfb)

println("   Kp,uu,BBB: $(round(kpuu_bbb_gaba, digits=3)) (no P-gp efflux)")
println("   Kp,uu,BCSFB: $(round(kpuu_bcsfb_gaba, digits=3))")

# Note: Gabapentin relies on LAT1 for brain entry, but our model
# doesn't fully capture carrier-mediated influx yet

# ===========================================================================
# Test 5: Custom CNS drug parameters
# ===========================================================================
println("\n5. Testing custom CNS drug (high P-gp affinity)...")

custom_drug = CNSParams(
    "HighPgpDrug",
    450.0,          # MW
    3.5,            # logP
    8.5,            # pKa
    :base,
    0.05,           # fu_plasma (highly bound)
    0.02,           # fu_brain
    25.0,           # Kp_brain (high tissue binding)
    1.0e-4,         # Papp_BBB (good permeability)
    5.0e-5,         # Papp_BCSFB
    true,           # P-gp substrate
    5.0,            # Low Km (high affinity)
    true,           # BCRP substrate
    10.0,
    false, 0.0,     # MRP
    false,          # OATP
    false, false,   # LAT1, GLUT
    :brain_ecf
)

kpuu_custom = calculate_kpuu_bbb(custom_drug, bbb)
kpuu_csf_custom = calculate_kpuu_bcsfb(custom_drug, bcsfb)

println("   High P-gp affinity drug:")
println("   - Kp,uu,BBB: $(round(kpuu_custom, digits=3)) (heavily restricted)")
println("   - Kp,uu,BCSFB: $(round(kpuu_csf_custom, digits=3)) (P-gp pumps to CSF)")
println("   - Ratio CSF/ECF: $(round(kpuu_csf_custom/kpuu_custom, digits=1))x higher in CSF!")

# ===========================================================================
# Summary
# ===========================================================================
println("\n" * "="^70)
println("CNS/CSF MODEL SUMMARY")
println("="^70)
println("""
Key insights captured in the model:

1. BBB vs BCSFB transporter orientation:
   - BBB: P-gp on luminal side -> efflux TO BLOOD (restricts brain entry)
   - BCSFB: P-gp on apical side -> efflux INTO CSF (increases CSF!)

2. CSF compartments in series:
   LV (ventricle) -> TFV -> CM (cisterna) -> SAS (lumbar)

3. For P-gp substrates:
   - Brain ECF may be LOW (P-gp restricts BBB entry)
   - CSF may be HIGHER than expected (P-gp pumps into CSF)
   - Lumbar CSF does NOT reflect brain exposure!

4. Cisternal CSF (CSF_CM) relevant for:
   - Antipsychotics (D2 in brainstem)
   - Opioids (brainstem receptors)
   - Anatomical proximity to targets

5. Bulk flow (glymphatic):
   - ECF drains to CSF (ISF -> ventricular system)
   - Creates "washing" effect
   - Tissue binding maintains ECF > CSF gradient

MedLang DSL captures all these mechanisms!
""")
