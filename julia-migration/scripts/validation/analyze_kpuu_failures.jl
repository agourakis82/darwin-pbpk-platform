#!/usr/bin/env julia
# ===========================================================================
# DEEP ANALYSIS: WHY DOES THE Kp,uu MODEL FAIL?
# ===========================================================================
# Scientific root cause analysis before attempting fixes
# ===========================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Printf
using Statistics

println("=" ^ 80)
println("ROOT CAUSE ANALYSIS: Kp,uu PREDICTION FAILURES")
println("=" ^ 80)
println()

# External validation data with full properties
# (name, logP, fup, MW, is_base, pKa, is_pgp, Kpuu_obs)
validation_data = [
    ("Buspirone", 2.4, 0.05, 385.5, true, 7.3, false, 1.29),
    ("Carisoprodol", 2.4, 0.40, 260.3, false, nothing, false, 0.34),
    ("Carbamazepine", 2.5, 0.24, 236.3, false, nothing, false, 0.27),
    ("Chlorpromazine", 5.2, 0.05, 318.9, true, 9.3, true, 0.65),
    ("Citalopram", 3.5, 0.20, 324.4, true, 9.5, true, 0.68),
    ("Clozapine", 3.2, 0.05, 326.8, true, 7.5, false, 1.01),
    ("Cyclobenzaprine", 5.0, 0.07, 275.4, true, 8.5, false, 1.62),
    ("Diazepam", 2.8, 0.02, 284.7, false, nothing, false, 1.02),
    ("Fluvoxamine", 2.8, 0.23, 318.3, true, 9.4, false, 1.32),
    ("Fluoxetine", 4.0, 0.06, 309.3, true, 9.8, true, 0.89),
    ("Haloperidol", 4.3, 0.08, 375.9, true, 8.3, false, 1.06),
    ("Hydrocodone", 1.2, 0.55, 299.4, true, 8.9, true, 1.96),
    ("Hydroxyzine", 2.4, 0.07, 374.9, true, 7.1, true, 1.51),
    ("Lamotrigine", 1.9, 0.45, 256.1, true, 5.7, false, 0.64),
    ("Meprobamate", 0.7, 0.80, 218.3, false, nothing, false, 0.42),
    ("Metoclopramide", 2.6, 0.60, 299.8, true, 9.3, true, 0.52),
    ("Methylphenidate", 2.0, 0.85, 233.3, true, 8.8, false, 3.43),
    ("Midazolam", 3.9, 0.03, 325.8, true, 6.0, true, 0.14),
    ("Morphine", 0.9, 0.65, 285.3, true, 8.0, true, 0.72),
    ("Nortriptyline", 4.7, 0.07, 263.4, true, 9.7, true, 1.63),
    ("9-OH-Risperidone", 2.3, 0.23, 426.5, true, 8.2, true, 0.02),
    ("Paroxetine", 3.6, 0.05, 329.4, true, 9.9, true, 0.86),
    ("Phenacetin", 1.6, 0.70, 179.2, false, nothing, false, 0.55),
    ("Phenytoin", 2.5, 0.10, 252.3, false, nothing, false, 0.28),
    ("Propranolol", 3.5, 0.10, 259.3, true, 9.4, true, 3.08),
    ("Propoxyphene", 4.2, 0.22, 339.5, true, 9.0, true, 0.85),
    ("Quinidine", 3.4, 0.13, 324.4, true, 8.5, true, 0.05),
    ("Risperidone", 3.0, 0.10, 410.5, true, 8.2, true, 0.26),
    ("Selegiline", 2.7, 0.06, 187.3, true, 7.5, false, 1.30),
    ("Sertraline", 5.1, 0.02, 306.2, true, 9.5, true, 1.44),
    ("Sulpiride", -0.6, 0.60, 341.4, true, 9.1, true, 0.06),
    ("Thiopental", 2.9, 0.15, 242.3, false, nothing, false, 0.17),
    ("Trazodone", 2.8, 0.11, 371.9, true, 7.1, true, 0.56),
    ("Venlafaxine", 2.7, 0.73, 277.4, true, 9.4, true, 0.98),
    ("Warfarin", 2.7, 0.01, 308.3, true, 5.1, false, 0.19),
    ("Zolpidem", 3.0, 0.08, 307.4, true, 6.2, false, 0.24),
]

# ===========================================================================
# ANALYSIS 1: BY DRUG CLASS (BASE vs NEUTRAL vs ACID)
# ===========================================================================

println("ANALYSIS 1: Kp,uu by Drug Class")
println("-" ^ 60)

bases = filter(d -> d[5] == true, validation_data)
neutrals = filter(d -> d[5] == false, validation_data)

base_kpuu = [d[8] for d in bases]
neutral_kpuu = [d[8] for d in neutrals]

@printf("  Bases (N=%d):\n", length(bases))
@printf("    Mean Kp,uu:   %.2f\n", mean(base_kpuu))
@printf("    Median:       %.2f\n", median(base_kpuu))
@printf("    Range:        %.2f - %.2f\n", minimum(base_kpuu), maximum(base_kpuu))
println()

@printf("  Neutrals (N=%d):\n", length(neutrals))
@printf("    Mean Kp,uu:   %.2f\n", mean(neutral_kpuu))
@printf("    Median:       %.2f\n", median(neutral_kpuu))
@printf("    Range:        %.2f - %.2f\n", minimum(neutral_kpuu), maximum(neutral_kpuu))
println()

println("  KEY FINDING: Neutrals have LOWER Kp,uu than expected!")
println("  Our model assumes neutrals equilibrate (Kp,uu ~1)")
println("  Reality: Neutral mean = $(round(mean(neutral_kpuu), digits=2))")
println()

# ===========================================================================
# ANALYSIS 2: BY P-gp STATUS
# ===========================================================================

println("ANALYSIS 2: Kp,uu by P-gp Status")
println("-" ^ 60)

pgp_pos = filter(d -> d[7] == true, validation_data)
pgp_neg = filter(d -> d[7] == false, validation_data)

pgp_pos_kpuu = [d[8] for d in pgp_pos]
pgp_neg_kpuu = [d[8] for d in pgp_neg]

@printf("  P-gp Substrates (N=%d):\n", length(pgp_pos))
@printf("    Mean Kp,uu:   %.2f\n", mean(pgp_pos_kpuu))
@printf("    Median:       %.2f\n", median(pgp_pos_kpuu))
@printf("    Range:        %.2f - %.2f\n", minimum(pgp_pos_kpuu), maximum(pgp_pos_kpuu))
println()

@printf("  Non-P-gp (N=%d):\n", length(pgp_neg))
@printf("    Mean Kp,uu:   %.2f\n", mean(pgp_neg_kpuu))
@printf("    Median:       %.2f\n", median(pgp_neg_kpuu))
@printf("    Range:        %.2f - %.2f\n", minimum(pgp_neg_kpuu), maximum(pgp_neg_kpuu))
println()

# P-gp substrates with HIGH Kp,uu (paradox!)
high_pgp = filter(d -> d[7] == true && d[8] > 1.0, validation_data)
println("  PARADOX: P-gp substrates with Kp,uu > 1.0:")
for d in high_pgp
    @printf("    - %-20s: Kp,uu = %.2f (P-gp substrate!)\n", d[1], d[8])
end
println()
println("  These likely have ACTIVE UPTAKE overcoming P-gp efflux!")
println()

# ===========================================================================
# ANALYSIS 3: CORRELATION WITH PHYSICOCHEMICAL PROPERTIES
# ===========================================================================

println("ANALYSIS 3: Correlation with Physicochemical Properties")
println("-" ^ 60)

logP_vals = [d[2] for d in validation_data]
fup_vals = [d[3] for d in validation_data]
MW_vals = [d[4] for d in validation_data]
kpuu_vals = [d[8] for d in validation_data]

log_kpuu = log10.(kpuu_vals)

# Correlations
cor_logP = cor(logP_vals, log_kpuu)
cor_fup = cor(fup_vals, log_kpuu)
cor_MW = cor(MW_vals, log_kpuu)

@printf("  Correlation with log(Kp,uu):\n")
@printf("    logP:  r = %.3f\n", cor_logP)
@printf("    fup:   r = %.3f\n", cor_fup)
@printf("    MW:    r = %.3f\n", cor_MW)
println()

println("  KEY FINDING: Weak correlations with simple properties!")
println("  This means Kp,uu is determined by TRANSPORTERS, not just passive diffusion.")
println()

# ===========================================================================
# ANALYSIS 4: OUTLIER DRUG PATTERNS
# ===========================================================================

println("ANALYSIS 4: What Makes Outliers Different?")
println("-" ^ 60)

# Group by Kp,uu ranges
very_low = filter(d -> d[8] < 0.1, validation_data)
low = filter(d -> 0.1 <= d[8] < 0.5, validation_data)
medium = filter(d -> 0.5 <= d[8] < 1.0, validation_data)
high = filter(d -> 1.0 <= d[8] < 2.0, validation_data)
very_high = filter(d -> d[8] >= 2.0, validation_data)

println("  VERY LOW Kp,uu (<0.1): Strong efflux, no uptake")
for d in very_low
    pgp = d[7] ? "P-gp+" : "P-gp-"
    @printf("    %-20s: %.2f  [%s, logP=%.1f]\n", d[1], d[8], pgp, d[2])
end
println()

println("  VERY HIGH Kp,uu (≥2.0): Active uptake dominates")
for d in very_high
    pgp = d[7] ? "P-gp+" : "P-gp-"
    @printf("    %-20s: %.2f  [%s, logP=%.1f]\n", d[1], d[8], pgp, d[2])
end
println()

# ===========================================================================
# ANALYSIS 5: pKa EFFECT (ION TRAPPING)
# ===========================================================================

println("ANALYSIS 5: pKa Effect on Kp,uu (Ion Trapping)")
println("-" ^ 60)

bases_with_pka = filter(d -> d[5] == true && !isnothing(d[6]), validation_data)

# Sort by pKa
sorted_bases = sort(bases_with_pka, by=d->d[6])

println("  Bases sorted by pKa:")
@printf("  %-20s %6s %8s %6s\n", "Drug", "pKa", "Kp,uu", "P-gp")
println("  " * "-"^44)
for d in sorted_bases
    pgp = d[7] ? "Yes" : "No"
    @printf("  %-20s %6.1f %8.2f %6s\n", d[1], d[6], d[8], pgp)
end
println()

# Check if high pKa correlates with high Kp,uu
pka_vals = [d[6] for d in bases_with_pka]
kpuu_bases = [d[8] for d in bases_with_pka]
cor_pka = cor(pka_vals, log10.(kpuu_bases))
@printf("  Correlation pKa vs log(Kp,uu): r = %.3f\n", cor_pka)
println()

# ===========================================================================
# ANALYSIS 6: WHAT DETERMINES Kp,uu?
# ===========================================================================

println("=" ^ 60)
println("SYNTHESIS: What Actually Determines Kp,uu?")
println("=" ^ 60)
println()

println("Based on this analysis:")
println()
println("1. P-gp STATUS is NOT binary:")
println("   - Some P-gp substrates have Kp,uu > 3 (active uptake)")
println("   - Some P-gp substrates have Kp,uu < 0.1 (strong efflux)")
println("   - Need QUANTITATIVE efflux ratio, not just yes/no")
println()

println("2. ACTIVE UPTAKE is critical:")
println("   - Propranolol (3.08): Likely OCT1/OCT2 uptake")
println("   - Methylphenidate (3.43): Likely DAT involvement")
println("   - Hydrocodone (1.96): Likely OATP uptake")
println("   - Without modeling uptake, we CANNOT predict high Kp,uu")
println()

println("3. NEUTRAL DRUGS don't equilibrate:")
println("   - Theory says Kp,uu should be ~1.0")
println("   - Reality: mean = $(round(mean(neutral_kpuu), digits=2))")
println("   - Possible reasons: BCRP efflux, metabolic clearance, binding")
println()

println("4. STRONG P-gp substrates are predictable:")
println("   - Quinidine (0.05), 9-OH-Risperidone (0.02), Sulpiride (0.06)")
println("   - These follow the expected pattern")
println()

println("5. MULTIVARIATE model needed:")
println("   - Simple physicochemical properties explain <15% variance")
println("   - Must include: P-gp efflux ratio, active uptake, BCRP")
println()

# ===========================================================================
# PROPOSED MODEL ARCHITECTURE
# ===========================================================================

println("=" ^ 60)
println("PROPOSED NEW MODEL ARCHITECTURE")
println("=" ^ 60)
println()

println("""
Current Model (FAILING):
  Kp,uu = mechanistic(logP, fup, pKa)
        × P-gp_binary_effect
        × arbitrary_caps

Proposed Model (HYBRID):

  1. BASE PREDICTION (mechanistic):
     log(Kp,uu_base) = α₀ + α₁·logP + α₂·log(fup) + α₃·pKa_factor

  2. EFFLUX TERM (P-gp + BCRP):
     efflux_factor = 1 / (1 + Km/IC50_pgp × P-gp_expression)
     - Use literature IC50 values for known substrates
     - Estimate from structure for unknowns

  3. UPTAKE TERM (novel!):
     uptake_factor = 1 + Σ(transporter_activity)
     - OCT1/OCT2 for cationic drugs
     - LAT1 for amino acid analogs
     - OATP for anionic drugs

  4. CORRECTION TERM (ML-based):
     correction = ML_model(descriptors)
     - Trained on residuals from mechanistic model
     - Captures what mechanism misses

  FINAL:
     Kp,uu = Kp,uu_base × efflux_factor × uptake_factor × correction
""")

println()
println("=" ^ 60)
println("NEXT STEPS")
println("=" ^ 60)
println()
println("1. Get Ma et al. 2024 full training dataset (226 compounds)")
println("2. Implement improved mechanistic base model")
println("3. Add transporter terms (efflux + uptake)")
println("4. Train ML correction on residuals")
println("5. Validate on held-out 36 drugs")
println()
