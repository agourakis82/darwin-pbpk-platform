#!/usr/bin/env julia
# ===========================================================================
# TEST: Hepatic Metabolism MedLang Model
# ===========================================================================
# Tests:
# 1. Fractal sinusoidal architecture
# 2. Classical vs Fractal Michaelis-Menten kinetics
# 3. Hepatic zonation
# 4. CYP enzyme metabolism
# 5. Transporter-enzyme interplay
# 6. DDI mechanisms (inhibition, induction, MBI)
# 7. Cirrhosis (Child-Pugh with fractal dimension collapse)
# 8. Drug-specific examples
# ===========================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DarwinPBPK
using DarwinPBPK.MedLang

println("="^70)
println("HEPATIC METABOLISM MODEL IN MEDLANG")
println("With Fractal Sinusoidal Architecture")
println("="^70)

# ===========================================================================
# 1. Fractal Sinusoidal Architecture
# ===========================================================================
println("\n1. Fractal Sinusoidal Architecture...")

fractal_healthy = fractal_sinusoid()
println("   Healthy liver:")
println("   - Fractal dimension (Df): $(fractal_healthy.Df)")
println("   - Spectral dimension (ds): $(fractal_healthy.ds)")
println("   - Fractal exponent (h): $(round(fractal_healthy.h, digits=3))")
println("   - Walk dimension (dw): $(round(fractal_healthy.walk_dimension, digits=2))")

# In cirrhosis, Df decreases
println("\n   Cirrhosis causes fractal dimension COLLAPSE:")
println("   - Healthy: Df ≈ 1.70 (complex, branching sinusoids)")
println("   - Child-Pugh A: Df ≈ 1.55")
println("   - Child-Pugh B: Df ≈ 1.45")
println("   - Child-Pugh C: Df ≈ 1.35 (simplified, linear sinusoids)")
println("   → Less branching = less hepatocyte contact = reduced clearance")

# ===========================================================================
# 2. Classical vs Fractal Michaelis-Menten
# ===========================================================================
println("\n2. Classical vs Fractal Michaelis-Menten Kinetics...")

Vmax = 100.0
Km = 10.0
h = 0.17  # DLA fractal exponent

println("\n   Substrate [S] | Classical MM | Fractal MM | Ratio")
println("   " * "-"^55)

for S in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
    v_classical = classical_michaelis_menten(S, Vmax, Km)
    v_fractal = fractal_michaelis_menten(S, Vmax, Km, h)
    ratio = v_fractal / v_classical

    println("   $(lpad(string(S), 6)) µM    | $(lpad(string(round(v_classical, digits=2)), 8)) | " *
            "$(lpad(string(round(v_fractal, digits=2)), 8))   | $(round(ratio, digits=3))")
end

println("\n   Note: Fractal kinetics gives ~5-10% lower rates due to")
println("   anomalous diffusion in spatially constrained environments")

# ===========================================================================
# 3. Hepatic Zonation
# ===========================================================================
println("\n3. Hepatic Zonation...")

zonation = default_zonation()
println("\n   CYP Distribution (Zone 1 = Periportal, Zone 3 = Pericentral):")
println("   Enzyme    | Zone 1 | Zone 3 | Predominant Zone")
println("   " * "-"^55)
println("   CYP3A4    | $(Int(zonation.CYP3A4_zone1*100))%    | $(Int(zonation.CYP3A4_zone3*100))%    | Pericentral")
println("   CYP2E1    | $(Int(zonation.CYP2E1_zone1*100))%    | $(Int(zonation.CYP2E1_zone3*100))%    | Pericentral (↑in NASH)")
println("   CYP2D6    | $(Int(round(zonation.CYP2D6_zone1*100)))%    | $(Int(round(zonation.CYP2D6_zone3*100)))%    | ~Uniform")
println("   CYP1A2    | $(Int(zonation.CYP1A2_zone1*100))%    | $(Int(zonation.CYP1A2_zone3*100))%    | Pericentral")

println("\n   OATP1B1   | $(Int(zonation.OATP1B1_zone1*100))%    | $(Int(zonation.OATP1B1_zone3*100))%    | Periportal")
println("\n   SPATIAL MISMATCH for OATP1B1-CYP3A4 substrates (statins):")
println("   Drug enters periportally (OATP) but is metabolized pericentrally (CYP)")
println("   → Transit time through sinusoid matters!")

# ===========================================================================
# 4. DDI Mechanisms
# ===========================================================================
println("\n4. DDI Mechanisms...")

# Competitive inhibition
S, I, Ki = 10.0, 5.0, 2.0
v_uninhibited = classical_michaelis_menten(S, Vmax, Km)
v_competitive = ddi_competitive(S, I, Vmax, Km, Ki)
println("\n   Competitive inhibition (ketoconazole-like):")
println("   - Uninhibited rate: $(round(v_uninhibited, digits=2))")
println("   - With inhibitor: $(round(v_competitive, digits=2))")
println("   - Ratio: $(round(v_competitive/v_uninhibited, digits=2)) (↑apparent Km)")

# MBI
println("\n   Mechanism-based inactivation (clarithromycin-like):")
kinact = 0.05  # min⁻¹
KI = 5.0       # µM
kdeg = 0.00032 # min⁻¹ (CYP3A4)
fm = 0.9

for I_mbi in [0.5, 1.0, 5.0, 10.0]
    auc_ratio = ddi_mbi(I_mbi, kinact, KI, kdeg, fm)
    println("   - [I] = $(I_mbi) µM → AUC ratio: $(round(auc_ratio, digits=1))x")
end
println("   Note: MBI is TIME-DEPENDENT and IRREVERSIBLE")

# Induction
println("\n   Enzyme induction (rifampicin):")
rif_params = inducer_preset(:rifampicin)
kdeg_cyp3a4 = 0.019  # h⁻¹

for t_days in [1, 3, 7, 14, 21]
    fold = enzyme_induction_dynamics(Float64(t_days), 10.0, rif_params, kdeg_cyp3a4)
    println("   - Day $(lpad(t_days, 2)): $(round(fold, digits=2))-fold CYP3A4")
end
println("   Note: Full induction takes 2 weeks (protein turnover)")

# ===========================================================================
# 5. Cirrhosis with Fractal Dimension Collapse
# ===========================================================================
println("\n5. Cirrhosis - Child-Pugh Classification with Fractal Changes...")

for cp_class in [:A, :B, :C]
    cirr = cirrhosis_state(cp_class)
    println("\n   Child-Pugh $(cp_class) (Score $(cirr.child_pugh.total_score), MELD $(Int(round(cirr.meld_score)))):")
    println("   - Sinusoid Df: $(cirr.sinusoid_Df) (healthy=1.70)")
    println("   - Blood flow: $(Int(round(cirr.hepatic_blood_flow_fraction*100)))%")
    println("   - Portal shunt: $(Int(round(cirr.portal_shunt_fraction*100)))%")
    println("   - CYP3A4: $(Int(round(cirr.cyp3a4_expression*100)))%")
    println("   - OATP1B1: $(Int(round(cirr.oatp1b1_expression*100)))%")
    println("   - Albumin: $(Int(round(cirr.albumin_fraction*100)))% → fu ↑")
end

# ===========================================================================
# 6. Drug Examples
# ===========================================================================
println("\n6. Drug-Specific Hepatic Clearance...")

cyp_enzymes = default_cyp_enzymes()
transporters = default_hepatic_transporters()

println("\n   Drug          | Primary CYP | Eh     | CLh (mL/min) | Class")
println("   " * "-"^65)

for drug_sym in [:midazolam, :atorvastatin, :caffeine, :warfarin, :ketoconazole]
    drug = drug_hepatic_preset(drug_sym)
    cl = calculate_clh(drug, cyp_enzymes, transporters)
    eh_class = calculate_extraction_ratio(cl["Eh"])

    name = rpad(drug.drug_name, 12)
    cyp = rpad(string(drug.primary_cyp), 7)
    println("   $(name) | $(cyp)     | $(lpad(string(round(cl["Eh"], digits=2)), 5)) | " *
            "$(lpad(string(round(cl["CLh_mL_min"], digits=1)), 8))     | $(eh_class)")
end

# ===========================================================================
# 7. Cirrhosis Effect on Drug Clearance
# ===========================================================================
println("\n7. Effect of Cirrhosis on Atorvastatin (OATP1B1-CYP3A4 substrate)...")

atorvastatin = drug_hepatic_preset(:atorvastatin)

println("\n   Condition      | Df   | CLh (mL/min) | Eh    | AUC ratio")
println("   " * "-"^60)

# Healthy
cl_healthy = calculate_clh(atorvastatin, cyp_enzymes, transporters)
println("   Healthy        | 1.70 | $(lpad(string(round(cl_healthy["CLh_mL_min"], digits=1)), 8))     | " *
        "$(round(cl_healthy["Eh"], digits=2))  | 1.0x")

# Cirrhosis stages
for cp_class in [:A, :B, :C]
    cirr = cirrhosis_state(cp_class)
    cl_cirr = calculate_clh(atorvastatin, cyp_enzymes, transporters; cirrhosis=cirr)
    auc_ratio = cl_healthy["CLh_mL_min"] / max(cl_cirr["CLh_mL_min"], 0.1)

    println("   Child-Pugh $(cp_class)  | $(cirr.sinusoid_Df) | $(lpad(string(round(cl_cirr["CLh_mL_min"], digits=1)), 8))     | " *
            "$(round(cl_cirr["Eh"], digits=2))  | $(round(auc_ratio, digits=1))x")
end

println("\n   Why the large increase?")
println("   1. Sinusoid Df ↓ → less hepatocyte contact")
println("   2. OATP1B1 ↓ → less uptake (rate-limiting for statins)")
println("   3. CYP3A4 ↓ → less metabolism")
println("   4. Portal shunting → drug bypasses liver")
println("   5. Albumin ↓ → fu ↑ → partially compensates")

# ===========================================================================
# 8. Generate MedLang Code
# ===========================================================================
println("\n8. Generating MedLang DSL code for Atorvastatin...")

medlang_code = generate_hepatic_medlang(atorvastatin)
lines = split(medlang_code, '\n')
println("\n--- Generated MedLang (first 100 lines) ---")
for (i, line) in enumerate(lines[1:min(100, length(lines))])
    println(line)
end
println("... ($(length(lines)) total lines)")

# ===========================================================================
# 9. Generate MedLang with Cirrhosis
# ===========================================================================
println("\n9. Generating MedLang for Midazolam in Child-Pugh C...")

midazolam = drug_hepatic_preset(:midazolam)
cirr_c = cirrhosis_state(:C)

medlang_cirr = generate_hepatic_medlang(midazolam; cirrhosis=cirr_c)
lines_cirr = split(medlang_cirr, '\n')
println("\n--- Generated MedLang with Cirrhosis (first 80 lines) ---")
for (i, line) in enumerate(lines_cirr[1:min(80, length(lines_cirr))])
    println(line)
end
println("... ($(length(lines_cirr)) total lines)")

# ===========================================================================
# 10. Simulation
# ===========================================================================
println("\n10. Simulating hepatic clearance (Midazolam 5mg IV)...")

sim_healthy = simulate_hepatic_clearance(midazolam, 5.0; t_max_h=12.0)
sim_cirrhosis = simulate_hepatic_clearance(midazolam, 5.0; t_max_h=12.0, cirrhosis=cirr_c)

println("\n   Parameter      | Healthy | Child-Pugh C")
println("   " * "-"^45)
println("   CLh (mL/min)   | $(lpad(string(round(sim_healthy["CLh_mL_min"], digits=1)), 7)) | $(round(sim_cirrhosis["CLh_mL_min"], digits=1))")
println("   Eh             | $(lpad(string(round(sim_healthy["Eh"], digits=2)), 7)) | $(round(sim_cirrhosis["Eh"], digits=2))")
println("   Half-life (h)  | $(lpad(string(round(sim_healthy["half_life_h"], digits=2)), 7)) | $(round(sim_cirrhosis["half_life_h"], digits=2))")
println("   Fractal Df     | $(lpad(string(round(sim_healthy["fractal_Df"], digits=2)), 7)) | $(round(sim_cirrhosis["fractal_Df"], digits=2))")

# ===========================================================================
# Summary
# ===========================================================================
println("\n" * "="^70)
println("HEPATIC METABOLISM MODEL SUMMARY")
println("="^70)

println("""
Key mechanisms captured in the model:

1. FRACTAL SINUSOIDAL ARCHITECTURE (DLA-based):
   - Healthy liver: Df ≈ 1.70 (complex branching)
   - Cirrhosis: Df ↓ (simplified, linear sinusoids)
   - Less branching = less hepatocyte contact = reduced clearance
   - This is WHY cirrhotic patients have unpredictable PK!

2. FRACTAL MICHAELIS-MENTEN KINETICS:
   - Classical MM assumes homogeneous, well-stirred
   - Reality: CYP on ER membrane (2D), anomalous diffusion
   - Fractal MM: v = Vmax × [S]^(1-h) / (Km' + [S]^(1-h))
   - Captures membrane-bound enzyme behavior

3. HEPATIC ZONATION:
   - Zone 1 (periportal): High O2, OATP1B1
   - Zone 3 (pericentral): Low O2, CYP3A4
   - SPATIAL MISMATCH for OATP1B1-CYP3A4 substrates!
   - Drug enters periportally, metabolized pericentrally

4. TRANSPORTER-ENZYME INTERPLAY:
   - Rate-limiting step depends on Df (sinusoid access)
   - High Df: metabolism-limited
   - Low Df: uptake-limited

5. DDI MECHANISMS:
   - Competitive inhibition: ↑Km (reversible)
   - Non-competitive: ↓Vmax (reversible)
   - MBI: Time-dependent, irreversible
   - Induction: Delayed onset (1-2 weeks), PXR/CAR/AhR

6. CIRRHOSIS:
   - Fractal dimension collapse (architecture)
   - Portal shunting (bypass liver)
   - Enzyme expression ↓
   - Transporter expression ↓
   - Albumin ↓ → fu ↑
   - Child-Pugh alone is INSUFFICIENT - need to capture architecture!

MedLang DSL captures geometry-function relationships in hepatic clearance!
""")
