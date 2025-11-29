# TEST QUANTUM DESCRIPTORS AND BLOOD WORK INTEGRATION
# =====================================================
#
# Tests the new modules:
# 1. QuantumDescriptors - Electronic structure descriptors
# 2. BloodWorkIntegration - Personalized PBPK parameters

using Printf

println("=" ^ 70)
println("TESTING QUANTUM DESCRIPTORS MODULE")
println("=" ^ 70)

# Include the quantum descriptors module
include("../../src/DarwinPBPK/ml/quantum_descriptors.jl")
using .QuantumDescriptors

# Test drugs with known properties
TEST_DRUGS = [
    ("Propranolol", "CC(C)NCC(O)COc1cccc2ccccc12", :base),
    ("Atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(c3ccc(F)cc3)n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c1c4ccc(F)cc4", :acid),
    ("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13", :neutral),
    ("Metformin", "CN(C)C(=N)NC(=N)N", :base),
    ("Imipramine", "CN(C)CCCN1c2ccccc2CCc3ccccc13", :base),
    ("Warfarin", "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O", :acid),
]

println("\n--- Quantum Descriptors for Test Drugs ---\n")

for (name, smiles, drug_type) in TEST_DRUGS
    qd = calculate_quantum_descriptors(smiles)

    if qd === nothing
        println("$name: FAILED to parse SMILES")
        continue
    end

    println("$name ($drug_type):")
    println("  Electronic Structure:")
    @printf("    HOMO: %.2f eV, LUMO: %.2f eV, Gap: %.2f eV\n",
            qd.homo, qd.lumo, qd.homo_lumo_gap)
    @printf("    Electronegativity: %.2f, Hardness: %.2f, Electrophilicity: %.2f\n",
            qd.electronegativity, qd.chemical_hardness, qd.electrophilicity)

    println("  Polarizability:")
    @printf("    α: %.1f Å³, PSA: %.1f Å², MR: %.1f\n",
            qd.polarizability, qd.polar_surface_area, qd.molar_refractivity)

    println("  Abraham Descriptors:")
    @printf("    E=%.2f, S=%.2f, A=%.2f, B=%.2f, V=%.2f\n",
            qd.E, qd.S, qd.A, qd.B, qd.V)

    println("  Shape:")
    @printf("    MW: %.1f, Rotatable: %.0f, Rings: %.0f, Aromatic: %.0f, fsp3: %.2f\n",
            qd.molecular_weight, qd.n_rotatable_bonds, qd.n_rings,
            qd.n_aromatic_rings, qd.fraction_sp3)

    println()
end

# Test descriptor vector
println("--- Descriptor Vector Test ---")
qd = calculate_quantum_descriptors("CC(C)NCC(O)COc1cccc2ccccc12")  # Propranolol
v = vec(qd)
println("Vector length: $(length(v)) (expected: $QUANTUM_DESCRIPTOR_DIM)")
println("Descriptor names: $(descriptor_names()[1:6])...")

# Test normalization
println("\n--- Normalization Test ---")
v_norm = normalize_descriptors(qd)
println("Normalized range: [$(minimum(v_norm)), $(maximum(v_norm))]")
println("All in [0,1]: $(all(0 .<= v_norm .<= 1))")

println("\n" * "=" ^ 70)
println("TESTING BLOOD WORK INTEGRATION MODULE")
println("=" ^ 70)

# Include blood work module
include("../../src/DarwinPBPK/clinical/blood_work_integration.jl")
using .BloodWorkIntegration

# Test Case 1: Normal patient
println("\n--- Test Case 1: Normal Patient ---")
normal_patient = PatientBloodWork(
    age_years = 45.0,
    weight_kg = 70.0,
    sex = :male,
    albumin_gdL = 4.2,
    AGP_mgdL = 75.0,
    ALT_UL = 25.0,
    AST_UL = 22.0,
    bilirubin_total_mgdL = 0.8,
    INR = 1.0,
    creatinine_mgdL = 1.0,
    platelets_K = 250.0,
    CRP_mgL = 0.5
)

params_normal = calculate_personalized_parameters(normal_patient)
println(clinical_summary(params_normal))

# Test Case 2: Cirrhotic patient (Child-Pugh B)
println("\n--- Test Case 2: Cirrhotic Patient (Child-Pugh B) ---")
cirrhotic_patient = PatientBloodWork(
    age_years = 58.0,
    weight_kg = 75.0,
    sex = :male,
    albumin_gdL = 2.8,  # Low!
    AGP_mgdL = 90.0,
    ALT_UL = 85.0,      # Elevated
    AST_UL = 95.0,      # Elevated
    bilirubin_total_mgdL = 2.5,  # Elevated
    INR = 1.8,          # Prolonged
    creatinine_mgdL = 1.1,
    platelets_K = 95.0,  # Low (hypersplenism)
    CRP_mgL = 8.0        # Inflammation
)

params_cirrhotic = calculate_personalized_parameters(cirrhotic_patient)
println(clinical_summary(params_cirrhotic))

# Test Case 3: CKD patient
println("\n--- Test Case 3: CKD Stage 4 Patient ---")
ckd_patient = PatientBloodWork(
    age_years = 72.0,
    weight_kg = 65.0,
    sex = :female,
    albumin_gdL = 3.5,
    AGP_mgdL = 110.0,   # Elevated (inflammation)
    ALT_UL = 18.0,
    bilirubin_total_mgdL = 0.6,
    INR = 1.0,
    creatinine_mgdL = 3.5,  # Elevated!
    eGFR = 22.0,            # Stage 4 CKD
    platelets_K = 180.0,
    CRP_mgL = 12.0          # Chronic inflammation
)

params_ckd = calculate_personalized_parameters(ckd_patient)
println(clinical_summary(params_ckd))

# Test Case 4: Patient with pharmacogenomics
println("\n--- Test Case 4: Patient with SLCO1B1*5 (Statin Risk) ---")
pgx_patient = PatientBloodWork(
    age_years = 55.0,
    weight_kg = 80.0,
    sex = :male,
    albumin_gdL = 4.0,
    ALT_UL = 30.0,
    creatinine_mgdL = 0.9,
    SLCO1B1_genotype = Symbol("*5"),  # Reduced OATP function
    CYP2D6_phenotype = :IM            # Intermediate metabolizer
)

params_pgx = calculate_personalized_parameters(pgx_patient)
println(clinical_summary(params_pgx))

# Demonstrate fup adjustment
println("\n--- fup Adjustment Examples ---")

# Warfarin (acidic, albumin-bound)
fup_warfarin_ref = 0.01
fup_warfarin_cirrhotic = personalize_fup(
    fup_reference = fup_warfarin_ref,
    drug_type = :acid,
    params = params_cirrhotic
)
@printf("Warfarin fup: Reference %.3f → Cirrhotic %.3f (%.1fx increase)\n",
        fup_warfarin_ref, fup_warfarin_cirrhotic,
        fup_warfarin_cirrhotic / fup_warfarin_ref)

# Propranolol (basic, AGP-bound)
fup_propranolol_ref = 0.10
fup_propranolol_inflamed = personalize_fup(
    fup_reference = fup_propranolol_ref,
    drug_type = :base,
    params = params_ckd  # Has elevated AGP due to inflammation
)
@printf("Propranolol fup: Reference %.3f → Inflamed %.3f (%.1fx change)\n",
        fup_propranolol_ref, fup_propranolol_inflamed,
        fup_propranolol_inflamed / fup_propranolol_ref)

# Clinical scores
println("\n--- Clinical Scoring ---")
@printf("Cirrhotic patient: Child-Pugh %d (Class %s), MELD %.1f\n",
        params_cirrhotic.child_pugh_score,
        params_cirrhotic.child_pugh_score <= 6 ? "A" :
        params_cirrhotic.child_pugh_score <= 9 ? "B" : "C",
        params_cirrhotic.meld_score)

println("\n" * "=" ^ 70)
println("ALL TESTS COMPLETED SUCCESSFULLY!")
println("=" ^ 70)
