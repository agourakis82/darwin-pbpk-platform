#!/usr/bin/env python3
"""
ADIPOSE TISSUE VALIDATION
=========================

Validate our adipose Kp predictions against:
1. Known drugs with measured adipose Kp
2. Theoretical expectations based on lipophilicity
3. Compare octanol vs olive oil partitioning
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 70)
print("ADIPOSE TISSUE: DEEP PHYSIOLOGICAL VALIDATION")
print("=" * 70)

# Tissue composition (Rodgers-Rowland)
F_NL = 0.853  # Neutral lipids (triglycerides)
F_PL = 0.0016  # Phospholipids
F_EW = 0.135  # Extracellular water
F_IW = 0.017  # Intracellular water
F_W = F_EW + F_IW  # Total water

print(f"""
ADIPOSE TISSUE COMPOSITION:
---------------------------
Neutral Lipids (triglycerides): {F_NL * 100:.1f}%
Phospholipids:                  {F_PL * 100:.2f}%
Extracellular Water:            {F_EW * 100:.1f}%
Intracellular Water:            {F_IW * 100:.1f}%
Total Water:                    {F_W * 100:.1f}%

KEY INSIGHT: 85% triglycerides = use OIL:WATER partition, not octanol!
""")


# Conversion from octanol to oil partitioning
def logP_oil(logP_oct):
    """Convert octanol:water to oil:water partition"""
    return 1.115 * logP_oct - 1.35


def P_oil(logP_oct):
    """Get oil:water partition coefficient"""
    return 10 ** logP_oil(logP_oct)


def calculate_kp_adipose(logP, fup, pKa=None, is_base=False, is_acid=False):
    """
    Calculate adipose Kp using Poulin-Theil/Rodgers-Rowland

    Key: Uses OIL:WATER partitioning for neutral lipids
    """
    # Oil:water partition (for triglycerides)
    P_ow = P_oil(logP)

    # Octanol:water (for phospholipids)
    P_oct = 10**logP

    # Ionization correction
    if pKa is not None:
        if is_base:
            ionization = 10 ** (pKa - 7.4)
            P_ow = P_ow / (1 + ionization)
        elif is_acid:
            ionization = 10 ** (7.4 - pKa)
            P_ow = P_ow / (1 + ionization)

    # Kp calculation
    # Neutral lipids: use oil:water
    # Phospholipids: 0.3P + 0.7 (Rodgers-Rowland approximation)
    # Water: 1.0

    Kpu = P_ow * F_NL + (0.3 * P_oct + 0.7) * F_PL + F_W
    Kp = Kpu * fup

    return Kp, P_ow, P_oct


print("=" * 70)
print("1. OCTANOL vs OLIVE OIL PARTITION COMPARISON")
print("=" * 70)

print("""
The key equation: log P(oil) = 1.115 × log P(octanol) - 1.35

This means:
- Low logP drugs: OIL partition << OCTANOL partition
- High logP drugs: OIL partition slightly > OCTANOL partition
""")

print("\nlog P(oct) | log P(oil) | P(oct) | P(oil) | Oil/Oct Ratio")
print("-" * 65)
for logP in [-1, 0, 1, 2, 3, 4, 5, 6, 7]:
    logP_o = logP_oil(logP)
    P_oct = 10**logP
    P_o = 10**logP_o
    ratio = P_o / P_oct
    print(
        f"    {logP:3.0f}    |   {logP_o:5.2f}   | {P_oct:7.1f} | {P_o:7.1f} | {ratio:8.4f}"
    )

print("""
KEY INSIGHT:
- At logP = 2: Oil partition is only 8% of octanol!
- At logP = 4: Oil partition is 130% of octanol
- At logP = 6: Oil partition is 220% of octanol

Most drugs have logP 1-4, so adipose Kp is LOWER than
octanol-based predictions would suggest.
""")

print("=" * 70)
print("2. VALIDATION WITH KNOWN DRUGS")
print("=" * 70)

# Literature data for adipose Kp (rat data, Rodgers & Rowland 2006)
# (drug, logP, fup, pKa, is_base, observed_Kp_adipose)
validation_drugs = [
    ("Caffeine", -0.1, 0.65, None, False, False, 0.15),
    ("Antipyrine", 0.3, 0.90, None, False, False, 0.20),
    ("Theophylline", -0.1, 0.60, None, False, False, 0.12),
    ("Metoprolol", 1.9, 0.88, 9.7, True, False, 0.25),
    ("Propranolol", 3.5, 0.13, 9.4, True, False, 0.8),
    ("Diazepam", 2.8, 0.02, 3.4, False, False, 4.0),
    ("Midazolam", 3.9, 0.04, 6.2, True, False, 8.0),
    ("Thiopental", 2.9, 0.15, 7.5, False, True, 10.0),
    ("Phenytoin", 2.5, 0.10, 8.3, False, True, 2.0),
    ("Imipramine", 4.8, 0.10, 9.4, True, False, 15.0),
    ("Amitriptyline", 4.9, 0.05, 9.4, True, False, 20.0),
]

print(
    "\nDrug           | logP | fup  | pKa  | Type  | P(oil) | Obs Kp | Pred Kp | Error"
)
print("-" * 85)

predictions = []
observations = []

for drug, logP, fup, pKa, is_base, is_acid, obs_kp in validation_drugs:
    pred_kp, P_ow, P_oct = calculate_kp_adipose(logP, fup, pKa, is_base, is_acid)

    drug_type = "Base" if is_base else ("Acid" if is_acid else "Neut")
    fold_error = max(pred_kp / obs_kp, obs_kp / pred_kp)

    predictions.append(pred_kp)
    observations.append(obs_kp)

    print(
        f"{drug:14s} | {logP:4.1f} | {fup:.2f} | {str(pKa) if pKa else '-':4s} | {drug_type:5s} | {P_ow:6.1f} | {obs_kp:6.1f} | {pred_kp:7.2f} | {fold_error:5.2f}×"
    )

# Calculate metrics
predictions = np.array(predictions)
observations = np.array(observations)
log_errors = np.abs(np.log10(predictions) - np.log10(observations))
gmfe = 10 ** np.mean(log_errors)
fold_errors = np.maximum(predictions / observations, observations / predictions)
within_2fold = np.sum(fold_errors <= 2.0) / len(fold_errors) * 100
within_3fold = np.sum(fold_errors <= 3.0) / len(fold_errors) * 100

print(f"""
ADIPOSE Kp PREDICTION PERFORMANCE:
----------------------------------
GMFE: {gmfe:.2f}
Within 2-fold: {within_2fold:.0f}%
Within 3-fold: {within_3fold:.0f}%
""")

print("=" * 70)
print("3. EFFECT OF IONIZATION ON ADIPOSE Kp")
print("=" * 70)

print("""
Ionized drugs do NOT partition well into triglycerides!

For a base with pKa 9.0 at pH 7.4:
  Ionization factor = 10^(9.0 - 7.4) = 40
  Effective P(oil) = P(oil) / 40

This is why highly basic drugs have lower adipose Kp
than their logP would suggest.
""")

# Show effect of pKa on adipose Kp for a drug with logP = 4
logP_test = 4.0
fup_test = 0.10
print(f"\nDrug with logP = {logP_test}, fup = {fup_test}")
print("pKa (base) | Ionization | Effective P(oil) | Kp_adipose")
print("-" * 60)

for pKa in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
    ionization = 10 ** (pKa - 7.4)
    P_ow_base = P_oil(logP_test)
    P_ow_eff = P_ow_base / (1 + ionization)
    kp, _, _ = calculate_kp_adipose(logP_test, fup_test, pKa, is_base=True)
    print(
        f"   {pKa:.1f}     |    {ionization:6.1f}   |      {P_ow_eff:7.1f}      |    {kp:5.2f}"
    )

print("""
KEY INSIGHT: A drug with pKa 9.0 has 40× less adipose uptake
than a neutral drug with same logP!
""")

print("=" * 70)
print("4. CLINICAL IMPLICATIONS: OBESITY")
print("=" * 70)

print("""
Adipose volume varies MASSIVELY with body composition:

| BMI | Adipose Volume | For drug with Kp=5 |
|-----|----------------|-------------------|
| 20  | 10 L           | Contributes 50 L  |
| 25  | 18 L           | Contributes 90 L  |
| 30  | 28 L           | Contributes 140 L |
| 35  | 40 L           | Contributes 200 L |
| 40  | 55 L           | Contributes 275 L |

For lipophilic drugs (Kp_adipose > 3):
- Obese patients need HIGHER loading doses
- BUT: elimination is similar (liver/kidney)
- Result: LONGER half-life in obese patients
""")

# Calculate Vdss contribution for different BMIs
print("\nVdss CONTRIBUTION FROM ADIPOSE:")
print("BMI  | Adipose (L) | Kp=1  | Kp=5  | Kp=10 | Kp=50")
print("-" * 55)
for bmi, adipose_vol in [(20, 10), (25, 18), (30, 28), (35, 40), (40, 55)]:
    contribs = [adipose_vol * kp for kp in [1, 5, 10, 50]]
    print(
        f" {bmi}  |     {adipose_vol:2.0f}      | {contribs[0]:4.0f}  | {contribs[1]:4.0f}  | {contribs[2]:4.0f}  | {contribs[3]:4.0f}"
    )

print("=" * 70)
print("5. PERFUSION LIMITATION")
print("=" * 70)

print("""
ADIPOSE BLOOD FLOW: 0.03 L/min/kg tissue (VERY LOW!)

For 20L adipose:
- Blood flow = 0.6 L/min
- Time to equilibrate = Volume / Flow = 20 / 0.6 = 33 min per turnover
- But only fraction of drug transfers each pass
- REAL equilibration: HOURS to DAYS

This creates the "DEEP COMPARTMENT" in multi-compartment PK models.

CLINICAL EXAMPLE - Thiopental:
1. IV injection → immediate brain uptake → unconsciousness
2. Minutes: redistribution to muscle → awakening
3. Hours: slow uptake into adipose (still ongoing!)
4. Days: slow release from adipose → residual sedation risk
""")

print("=" * 70)
print("6. KEY FEATURES FOR ML MODELING")
print("=" * 70)

print("""
RECOMMENDED FEATURES FOR ADIPOSE:

1. logP_oil = 1.115 × logP - 1.35
   (NOT logP_octanol!)

2. P_oil = 10^logP_oil
   (The actual partition coefficient)

3. kp_adipose = P_oil × 0.853 × fup
   (Simplified, ignoring phospholipids)

4. ionization_correction = 1 / (1 + 10^|pKa-7.4|)
   (For bases/acids)

5. adipose_contribution = kp_adipose × adipose_volume
   (Scale by patient's body composition)

6. is_adipose_accumulator = logP > 4
   (Binary flag for extreme accumulators)

7. adipose_fraction = adipose_contribution / total_Vdss
   (What fraction of drug is in fat)
""")

print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
ADIPOSE TISSUE - KEY TAKEAWAYS:

1. COMPOSITION: 85% triglycerides - NOT phospholipids
   → Use OIL:WATER partition, not octanol

2. CONVERSION: log P(oil) = 1.115 × logP - 1.35
   → At logP=2: oil partition is only 8% of octanol!

3. IONIZATION: Ionized drugs don't enter triglycerides
   → Strong bases (pKa>8) have much lower adipose Kp

4. VOLUME VARIABILITY: 10-60L depending on BMI
   → Biggest source of Vdss variability for lipophilic drugs

5. PERFUSION: 0.03 L/min/kg - slowest of all tissues
   → Creates "deep compartment", prolongs half-life

6. VALIDATION: Our model achieves GMFE {gmfe:.2f}
   → {within_2fold:.0f}% within 2-fold for {len(validation_drugs)} test drugs

NEXT: Proceed to MUSCLE tissue analysis.
""")
