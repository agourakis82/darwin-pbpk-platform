#!/usr/bin/env julia
# ===========================================================================
# TEST: Renal Elimination MedLang Model
# ===========================================================================
# Tests:
# 1. Basic renal clearance calculations (GFR, secretion, reabsorption)
# 2. pH-dependent ionization (Henderson-Hasselbalch)
# 3. CKD adaptation (tubular flow rate increase)
# 4. Fanconi syndrome (mTORC1-mediated transporter dysfunction)
# 5. Cystinosis modeling
# 6. Drug-specific examples (metformin, tenofovir, amphetamine)
# ===========================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DarwinPBPK
using DarwinPBPK.MedLang

println("="^70)
println("RENAL ELIMINATION MODEL IN MEDLANG")
println("="^70)

# ===========================================================================
# 1. Test Metformin (OCT2/MATE substrate, cation)
# ===========================================================================
println("\n1. Testing Metformin (OCT2/MATE substrate, organic cation)...")

metformin = drug_renal_preset(:metformin)
println("   Drug: $(metformin.drug_name)")
println("   MW: $(metformin.MW) Da")
println("   pKa: $(metformin.pKa) ($(metformin.charge_type))")
println("   fu,plasma: $(metformin.fu_plasma)")
println("   OCT2 substrate: $(metformin.is_oct2_substrate)")
println("   MATE substrate: $(metformin.is_mate_substrate)")

# Calculate renal clearance - healthy
transporters = default_renal_transporters()
cl_metformin = calculate_clr(metformin, transporters)

println("\n   Renal clearance (healthy):")
println("   - CL_filtration: $(round(cl_metformin["CL_filtration"], digits=1)) mL/min")
println("   - CL_secretion: $(round(cl_metformin["CL_secretion"], digits=1)) mL/min")
println("   - F_reabsorbed: $(round(cl_metformin["F_reabsorbed"] * 100, digits=1))%")
println("   - CL_renal: $(round(cl_metformin["CL_renal"], digits=1)) mL/min")
println("   Note: Metformin CLr ~450-600 mL/min (exceeds GFR = active secretion)")

# ===========================================================================
# 2. Test pH-dependent reabsorption (Amphetamine)
# ===========================================================================
println("\n2. Testing Amphetamine (weak base, pH-dependent reabsorption)...")

amphetamine = drug_renal_preset(:amphetamine)
println("   Drug: $(amphetamine.drug_name)")
println("   pKa: $(amphetamine.pKa) ($(amphetamine.charge_type))")

# Test at different urine pH
for urine_ph in [5.0, 6.0, 7.0, 8.0]
    f_ionized = henderson_hasselbalch_ionized_fraction(
        amphetamine.pKa, urine_ph, amphetamine.charge_type
    )
    cl_amph = calculate_clr(amphetamine, transporters; urine_ph=urine_ph)

    println("   pH $(urine_ph): $(round(f_ionized * 100, digits=1))% ionized, " *
            "F_reab=$(round(cl_amph["F_reabsorbed"] * 100, digits=1))%, " *
            "CLr=$(round(cl_amph["CL_renal"], digits=1)) mL/min")
end
println("   Clinical: Acidify urine in overdose -> more ionized -> trapped -> enhanced excretion")

# ===========================================================================
# 3. Test Aspirin (weak acid, opposite pH effect)
# ===========================================================================
println("\n3. Testing Aspirin (weak acid, pH-dependent reabsorption)...")

aspirin = drug_renal_preset(:aspirin)
println("   Drug: $(aspirin.drug_name)")
println("   pKa: $(aspirin.pKa) ($(aspirin.charge_type))")

for urine_ph in [5.0, 6.0, 7.0, 8.0]
    f_ionized = henderson_hasselbalch_ionized_fraction(
        aspirin.pKa, urine_ph, aspirin.charge_type
    )
    cl_asp = calculate_clr(aspirin, transporters; urine_ph=urine_ph)

    println("   pH $(urine_ph): $(round(f_ionized * 100, digits=1))% ionized, " *
            "F_reab=$(round(cl_asp["F_reabsorbed"] * 100, digits=1))%, " *
            "CLr=$(round(cl_asp["CL_renal"], digits=1)) mL/min")
end
println("   Clinical: Alkalinize urine in overdose -> more ionized -> trapped -> enhanced excretion")

# ===========================================================================
# 4. Test CKD stages
# ===========================================================================
println("\n4. Testing CKD stages (Tenofovir - OAT1/OAT3 substrate)...")

tenofovir = drug_renal_preset(:tenofovir)
println("   Drug: $(tenofovir.drug_name)")
println("   OAT1 substrate: $(tenofovir.is_oat1_substrate)")
println("   OAT3 substrate: $(tenofovir.is_oat3_substrate)")

println("\n   CKD Stage | GFR | OAT expr | CLr | Dose adjustment")
println("   " * "-"^55)

for stage in 1:5
    ckd = ckd_stage(stage)
    cl_ckd = calculate_clr(tenofovir, transporters; ckd=ckd)
    cl_healthy = calculate_clr(tenofovir, transporters)
    dose_adj = cl_ckd["CL_renal"] / cl_healthy["CL_renal"]

    println("   Stage $(stage)   | $(Int(ckd.gfr)) | $(round(ckd.oat_expression, digits=2))    | " *
            "$(round(cl_ckd["CL_renal"], digits=1)) | $(round(dose_adj * 100, digits=0))%")
end

# ===========================================================================
# 5. Test Fanconi Syndrome (mTORC1 mechanism)
# ===========================================================================
println("\n5. Testing Fanconi Syndrome (mTORC1-mediated dysfunction)...")

println("\n   mTORC1 activity | Transporter expr | ATP avail | OAT function | Clinical features")
println("   " * "-"^75)

for mtorc1 in [0.0, 0.3, 0.5, 0.7, 1.0]
    fanconi = fanconi_syndrome(mtorc1_activity=mtorc1)

    features = String[]
    fanconi.phosphaturia && push!(features, "PO4")
    fanconi.glucosuria && push!(features, "Glc")
    fanconi.aminoaciduria && push!(features, "AA")
    fanconi.metabolic_acidosis && push!(features, "RTA")

    println("   $(round(mtorc1, digits=1))            | $(round(fanconi.transporter_expression, digits=2))              | " *
            "$(round(fanconi.atp_availability, digits=2))       | $(round(fanconi.oat_function, digits=2))          | " *
            (isempty(features) ? "none" : join(features, ", ")))
end

# ===========================================================================
# 6. Test Cystinosis
# ===========================================================================
println("\n6. Testing Cystinosis (CTNS mutation -> mTORC1 hyperactivation)...")

# Untreated cystinosis
cyst_untreated = cystinosis(
    wbc_cystine=3.5,
    on_cysteamine=false
)
println("\n   Untreated cystinosis (WBC cystine: 3.5 nmol/mg):")
println("   - mTORC1 activity: $(round(cyst_untreated.fanconi.mtorc1_activity, digits=2))")
println("   - Transporter expression: $(round(cyst_untreated.fanconi.transporter_expression, digits=2))")
println("   - ATP availability: $(round(cyst_untreated.fanconi.atp_availability, digits=2))")
println("   - Clinical: phosphaturia=$(cyst_untreated.fanconi.phosphaturia), " *
        "glucosuria=$(cyst_untreated.fanconi.glucosuria)")

# Treated cystinosis (good compliance)
cyst_treated = cystinosis(
    wbc_cystine=3.5,
    on_cysteamine=true,
    cysteamine_compliance=0.9
)
println("\n   Treated cystinosis (cysteamine, 90% compliance):")
println("   - mTORC1 activity: $(round(cyst_treated.fanconi.mtorc1_activity, digits=2))")
println("   - Transporter expression: $(round(cyst_treated.fanconi.transporter_expression, digits=2))")
println("   - ATP availability: $(round(cyst_treated.fanconi.atp_availability, digits=2))")
println("   - Clinical: phosphaturia=$(cyst_treated.fanconi.phosphaturia), " *
        "glucosuria=$(cyst_treated.fanconi.glucosuria)")

# ===========================================================================
# 7. Drug clearance in Fanconi syndrome
# ===========================================================================
println("\n7. Drug clearance comparison: Healthy vs Fanconi syndrome...")

fanconi_moderate = fanconi_syndrome(mtorc1_activity=0.6)

println("\n   Drug         | Healthy CLr | Fanconi CLr | Ratio")
println("   " * "-"^50)

for drug_sym in [:metformin, :tenofovir, :penicillin_g, :gabapentin]
    drug = drug_renal_preset(drug_sym)
    cl_healthy = calculate_clr(drug, transporters)
    cl_fanconi = calculate_clr(drug, transporters; fanconi=fanconi_moderate)
    ratio = cl_fanconi["CL_renal"] / cl_healthy["CL_renal"]

    name_padded = rpad(drug.drug_name, 12)
    println("   $(name_padded) | $(lpad(string(round(cl_healthy["CL_renal"], digits=1)), 7)) | " *
            "$(lpad(string(round(cl_fanconi["CL_renal"], digits=1)), 7))     | $(round(ratio, digits=2))")
end

println("\n   Note: Secreted drugs (OAT/OCT substrates) most affected by transporter loss")

# ===========================================================================
# 8. Generate MedLang code
# ===========================================================================
println("\n8. Generating MedLang DSL code for Metformin...")

medlang_code = generate_renal_medlang(metformin)
lines = split(medlang_code, '\n')
println("\n--- Generated MedLang (first 80 lines) ---")
for (i, line) in enumerate(lines[1:min(80, length(lines))])
    println(line)
end
println("... ($(length(lines)) total lines)")

# ===========================================================================
# 9. Generate MedLang with disease states
# ===========================================================================
println("\n9. Generating MedLang for Tenofovir in CKD Stage 3 + Fanconi...")

ckd3 = ckd_stage(3)
fanconi_mild = fanconi_syndrome(etiology=:drug_induced, mtorc1_activity=0.4)

medlang_disease = generate_renal_medlang(tenofovir; ckd=ckd3, fanconi=fanconi_mild)
lines_disease = split(medlang_disease, '\n')
println("\n--- Generated MedLang with disease states (first 100 lines) ---")
for (i, line) in enumerate(lines_disease[1:min(100, length(lines_disease))])
    println(line)
end
println("... ($(length(lines_disease)) total lines)")

# ===========================================================================
# 10. Simulation
# ===========================================================================
println("\n10. Simulating renal elimination (Metformin 500mg)...")

sim_result = simulate_renal_elimination(
    metformin, 500.0;
    t_max_h=24.0,
    urine_ph=6.0
)

println("   Half-life: $(round(sim_result["half_life_h"], digits=2)) h")
println("   CLr: $(round(sim_result["CL_renal_mL_min"], digits=1)) mL/min")
println("   fe: $(round(sim_result["fe"], digits=2))")
println("   Total urinary excretion (24h): $(round(sim_result["A_urine_mg"][end], digits=1)) mg")

# ===========================================================================
# Summary
# ===========================================================================
println("\n" * "="^70)
println("RENAL ELIMINATION MODEL SUMMARY")
println("="^70)

println("""
Key mechanisms captured in the model:

1. RENAL CLEARANCE EQUATION:
   CLr = (CL_filtration + CL_secretion) x (1 - F_reabsorbed)
   - Filtration: GFR x fu,plasma
   - Secretion: OAT1/OAT3 (anions), OCT2/MATE (cations)
   - Reabsorption: pH-dependent passive diffusion

2. TRANSPORTER LOCALIZATION (vectorial transport):
   Basolateral (blood -> cell): OAT1, OAT3, OCT2
   Apical (cell -> urine): MATE1, MATE2-K, MRP2, MRP4

3. pH-DEPENDENT IONIZATION (Henderson-Hasselbalch):
   Weak acids: alkaline urine -> ionized -> trapped -> excretion
   Weak bases: acidic urine -> ionized -> trapped -> excretion

4. CKD ADAPTATION (the "deep detail"):
   - Remaining nephrons increase tubular flow rate per nephron
   - Nonlinear reduction in reabsorption
   - Permeable drugs: CLr reduction < GFR reduction

5. FANCONI SYNDROME / CYSTINOSIS (mTORC1 mechanism):
   CTNS mutation -> cystine accumulation -> mTORC1 hyperactivation
   -> transporter EXPRESSION reduced (not just inhibited!)
   -> ATP depletion -> "sick nephrons" vs "fewer nephrons"

6. CLINICAL APPLICATIONS:
   - Dose adjustment in CKD and Fanconi
   - DDI prediction (transporter inhibition)
   - Urine pH manipulation in overdose

MedLang DSL captures all these mechanisms!
""")
