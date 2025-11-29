#!/usr/bin/env julia
# ===========================================================================
# OUTLIER ANALYSIS: What's left to fix?
# ===========================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

println("=" ^ 80)
println("OUTLIER ANALYSIS - What's Different About These Drugs?")
println("=" ^ 80)
println()

# Outliers from validation
outliers = [
    # (name, logP, fup, MW, pKa, charge, pgp_er, obs, pred, issue)
    ("Hydroxyzine", 2.4, 0.07, 374.9, 7.1, :base, 3.0, 1.51, 0.34, "Active uptake? Antihistamine"),
    ("Zolpidem", 3.0, 0.08, 307.4, 6.2, :base, 1.0, 0.24, 1.05, "Possible BCRP efflux?"),
    ("Sertraline", 5.1, 0.02, 306.2, 9.5, :base, 2.5, 1.44, 0.37, "High lipophilicity base - OCT uptake?"),
    ("Thiopental", 2.9, 0.15, 242.3, nothing, :neutral, 1.0, 0.17, 0.62, "Neutral - why so low?"),
    ("Quinidine", 0.05, 0.15, 324.4, 8.5, :base, 15.0, 0.05, 0.15, "Strong P-gp but still overpredicted"),
    ("Nortriptyline", 4.7, 0.07, 263.4, 9.7, :base, 2.0, 1.63, 0.56, "TCA - OCT/NET uptake?"),
    ("Propranolol", 3.5, 0.10, 259.3, 9.4, :base, 2.0, 3.08, 1.24, "Known OCT substrate!"),
    ("9-OH-Risperidone", 2.3, 0.23, 426.5, 8.2, :base, 25.0, 0.02, 0.04, "Very strong P-gp"),
    ("Hydrocodone", 1.2, 0.55, 299.4, 8.9, :base, 2.5, 1.96, 0.91, "Opioid - OATP uptake?"),
    ("Sulpiride", -0.6, 0.60, 341.4, 9.1, :base, 20.0, 0.06, 0.03, "Very polar base + P-gp"),
]

println("UNDERPREDICTED (Active Uptake Candidates):")
println("-" ^ 60)

underpredicted = filter(o -> o[9] > o[8] * 2, outliers)
for o in underpredicted
    name, logP, fup, MW, pKa, charge, pgp, obs, pred, note = o
    ratio = pred / obs
    println()
    println("$name:")
    println("  Observed: $obs, Predicted: $pred ($(round(ratio, digits=2))x)")
    println("  Properties: logP=$logP, pKa=$pKa, MW=$MW, fup=$fup")
    println("  P-gp ER: $pgp")
    println("  Note: $note")

    # Hypothesis
    if obs > 1.0 && pgp > 1.5
        println("  HYPOTHESIS: Active uptake overcoming P-gp efflux")
        println("    - Consider adding OCT1/OCT2 term for cationic drugs")
        println("    - Or LAT1 term for amino acid-like structures")
    end
end

println()
println("=" ^ 60)
println("OVERPREDICTED (Unmodeled Efflux?):")
println("-" ^ 60)

overpredicted = filter(o -> o[9] < o[8] * 0.5, outliers)
for o in overpredicted
    name, logP, fup, MW, pKa, charge, pgp, obs, pred, note = o
    ratio = pred / obs
    println()
    println("$name:")
    println("  Observed: $obs, Predicted: $pred ($(round(ratio, digits=2))x)")
    println("  Properties: logP=$logP, pKa=$pKa, MW=$MW")
    println("  Note: $note")

    if charge == :neutral && obs < 0.3
        println("  HYPOTHESIS: BCRP efflux for neutral drugs")
    end
end

println()
println("=" ^ 80)
println("SPECIFIC RECOMMENDATIONS")
println("=" ^ 80)
println()

println("""
1. PROPRANOLOL (Obs=3.08, Pred=1.24):
   - Well-documented OCT1/OCT2 substrate
   - Need to add explicit OCT transporter term
   - OCT uptake factor ~2.5x for cationic beta-blockers

2. HYDROXYZINE (Obs=1.51, Pred=0.34):
   - Antihistamine with Kp,uu > 1 despite P-gp
   - Possible H1 receptor-mediated uptake?
   - Or OCT-mediated uptake (cationic at physiological pH)

3. SERTRALINE (Obs=1.44, Pred=0.37):
   - Very lipophilic (logP 5.1) but high Kp,uu
   - Possible OCT or NET transporter involvement
   - Or P-gp efflux ratio overestimated

4. ZOLPIDEM (Obs=0.24, Pred=1.05):
   - Neutral/weak base - behaves like neutral
   - Likely BCRP substrate (imidazopyridine)
   - Need to add BCRP efflux term for this class

5. THIOPENTAL (Obs=0.17, Pred=0.62):
   - Barbiturate - neutral drug
   - Very low Kp,uu despite good logP
   - Possible MRP efflux? Or metabolism in brain?

6. HYDROCODONE, NORTRIPTYLINE:
   - Both show active uptake pattern
   - Hydrocodone: possible OATP involvement
   - Nortriptyline: possible NET involvement (TCA)
""")

println()
println("=" ^ 80)
println("PROPOSED MODEL IMPROVEMENTS")
println("=" ^ 80)
println()

println("""
To reach >80% within 2-fold:

1. ADD OCT1/OCT2 UPTAKE TERM:
   - For cationic drugs (pKa > 8.0)
   - With logP 1.5-4.0 and MW < 400
   - Estimated uptake factor: 1.5-3.0x
   - Key drugs: propranolol, hydroxyzine, sertraline

2. ADD BCRP EFFLUX TERM:
   - For neutral drugs with imidazole/pyridine rings
   - Estimated efflux factor: 2-4x reduction
   - Key drugs: zolpidem, possibly others

3. REFINE NEUTRAL DRUG MODEL:
   - Current correction may be too uniform
   - Some neutrals equilibrate (diazepam: 1.02)
   - Some have strong efflux (thiopental: 0.17)
   - Need structural features to differentiate

4. TRANSPORTER SUBSTRATE PREDICTION:
   - Use structural alerts for P-gp, BCRP, OCT
   - Or use ML model trained on transporter data
   - Would improve predictions without explicit substrate info
""")
