#!/usr/bin/env python3
"""
MUSCLE TISSUE VALIDATION
========================

Validate muscle Kp predictions with focus on:
1. Ion trapping (pH 7.0 vs 7.4)
2. Acidic phospholipid binding for bases
3. High water content for hydrophilic drugs
"""

import numpy as np

print("=" * 70)
print("MUSCLE TISSUE: DEEP PHYSIOLOGICAL VALIDATION")
print("=" * 70)

# Tissue composition (Rodgers-Rowland)
F_IW = 0.630  # Intracellular water (HIGH!)
F_EW = 0.118  # Extracellular water
F_NL = 0.010  # Neutral lipids (low)
F_NPL = 0.0072  # Neutral phospholipids
F_APL = 0.00153  # Acidic phospholipids
F_WATER = F_IW + F_EW

print(f"""
MUSCLE TISSUE COMPOSITION:
--------------------------
Intracellular Water: {F_IW * 100:.1f}% ◄── HIGH (largest aqueous reservoir)
Extracellular Water: {F_EW * 100:.1f}%
Total Water:         {F_WATER * 100:.1f}%
Neutral Lipids:      {F_NL * 100:.2f}% (low - unlike adipose)
Neutral Phospholipids: {F_NPL * 100:.3f}%
Acidic Phospholipids:  {F_APL * 100:.4f}% ◄── Important for bases!

VOLUME: ~30 L (43% body weight) - LARGEST COMPARTMENT
""")

print("=" * 70)
print("1. ION TRAPPING: THE pH GRADIENT EFFECT")
print("=" * 70)

print("""
Muscle intracellular pH = 7.0
Plasma pH = 7.4
ΔpH = 0.4

For BASIC drugs:
- More ionized in muscle (lower pH)
- Ionized form can't easily leave
- Results in ACCUMULATION
""")

print("\nION TRAPPING QUANTIFICATION:")
print("pKa  | % Ion (plasma) | % Ion (muscle) | Trapping Ratio")
print("-" * 60)

for pKa in [5.0, 6.0, 7.0, 7.4, 8.0, 9.0, 10.0]:
    # For bases: ionization = 10^(pKa - pH) / (1 + 10^(pKa - pH))
    ion_plasma = 10 ** (pKa - 7.4) / (1 + 10 ** (pKa - 7.4)) * 100
    ion_muscle = 10 ** (pKa - 7.0) / (1 + 10 ** (pKa - 7.0)) * 100

    # X and Y factors
    X = 10 ** (pKa - 7.0)
    Y = 10 ** (pKa - 7.4)
    trapping_ratio = (1 + X) / (1 + Y)

    print(
        f"{pKa:4.1f} |     {ion_plasma:5.1f}%     |     {ion_muscle:5.1f}%     |     {trapping_ratio:.2f}×"
    )

print("""
KEY INSIGHT:
- Weak bases (pKa 5-6): Minimal ion trapping (~1.0×)
- Moderate bases (pKa 7-8): Moderate trapping (1.2-1.6×)
- Strong bases (pKa 9-10): Significant trapping (2.0-2.5×)

Maximum trapping effect is ~2.5× (when drug is fully ionized at both pH)
""")

print("=" * 70)
print("2. ACIDIC PHOSPHOLIPID BINDING")
print("=" * 70)

print("""
Muscle contains 0.153% acidic phospholipids (PS, PI).
These have NEGATIVE charges that bind POSITIVE bases.

Binding is described by Ka_AP (association constant).
Ka_AP is typically 10-500 depending on drug structure.
""")

# Calculate effect of Ka_AP on Kp
print("\nEFFECT OF Ka_AP ON MUSCLE Kp (for pKa 9 base, logP 3, fup 0.1):")
print("Ka_AP | APL Binding Term | Water Term | Lipid Term | Kp_muscle")
print("-" * 70)

logP = 3.0
pKa = 9.0
fup = 0.1
P = 10**logP
X = 10 ** (pKa - 7.0)  # 100
Y = 10 ** (pKa - 7.4)  # 40
denom = 1 + Y

for Ka_AP in [10, 50, 100, 200, 500]:
    water_term = F_EW + ((1 + X) / denom) * F_IW
    lipid_term = (P * F_NL + (0.3 * P + 0.7) * F_NPL) / denom
    apl_term = (Ka_AP * F_APL * X) / denom

    Kp = (water_term + lipid_term + apl_term) * fup

    print(
        f" {Ka_AP:3.0f}  |      {apl_term:6.3f}       |   {water_term:5.2f}    |   {lipid_term:6.4f}   |   {Kp:.2f}"
    )

print("""
KEY INSIGHT:
- Water term dominates for hydrophilic drugs
- Acidic PL binding becomes significant at Ka_AP > 100
- Strong bases (high Ka_AP) can have Kp_muscle >> 1
""")

print("=" * 70)
print("3. VALIDATION WITH KNOWN DRUGS")
print("=" * 70)


def calculate_kp_muscle(logP, fup, pKa=None, is_base=False, is_acid=False):
    """Calculate muscle Kp using Rodgers-Rowland approach"""
    P = 10**logP

    # Ionization factors
    X = 0
    Y = 0
    if pKa and is_base:
        X = 10 ** (pKa - 7.0)
        Y = 10 ** (pKa - 7.4)
    elif pKa and is_acid:
        X = 10 ** (7.0 - pKa)
        Y = 10 ** (7.4 - pKa)

    denom = max(1 + Y, 1e-10)

    # Ka_AP estimation
    if is_base and pKa and pKa > 7:
        Ka_AP = 40 * (1 + 0.25 * (pKa - 7))
        Ka_AP = min(Ka_AP, 200)
    else:
        Ka_AP = 10

    # Kp components
    water_term = F_EW + ((1 + X) / denom) * F_IW
    lipid_term = (P * F_NL + (0.3 * P + 0.7) * F_NPL) / denom

    if is_base:
        apl_term = (Ka_AP * F_APL * X) / denom
    else:
        apl_term = 0

    Kp = (water_term + lipid_term + apl_term) * fup

    return Kp, water_term, lipid_term, apl_term


# Literature data (rat muscle Kp, Rodgers & Rowland 2006)
validation_drugs = [
    ("Caffeine", -0.1, 0.65, None, False, False, 0.52),
    ("Antipyrine", 0.3, 0.90, None, False, False, 0.72),
    ("Theophylline", -0.1, 0.60, None, False, False, 0.48),
    ("Metoprolol", 1.9, 0.88, 9.7, True, False, 1.8),
    ("Propranolol", 3.5, 0.13, 9.4, True, False, 2.8),
    ("Diazepam", 2.8, 0.02, 3.4, False, False, 0.45),
    ("Phenytoin", 2.5, 0.10, 8.3, False, True, 0.35),
    ("Warfarin", 2.6, 0.01, 5.0, False, True, 0.08),
    ("Imipramine", 4.8, 0.10, 9.4, True, False, 5.2),
    ("Quinidine", 3.4, 0.15, 8.5, True, False, 3.5),
]

print(
    "\nDrug         | logP | fup  | pKa | Type | Water | Lipid | APL  | Pred | Obs  | Error"
)
print("-" * 95)

predictions = []
observations = []

for drug, logP, fup, pKa, is_base, is_acid, obs_kp in validation_drugs:
    pred_kp, water, lipid, apl = calculate_kp_muscle(logP, fup, pKa, is_base, is_acid)

    drug_type = "Base" if is_base else ("Acid" if is_acid else "Neut")
    fold_error = max(pred_kp / obs_kp, obs_kp / pred_kp)

    predictions.append(pred_kp)
    observations.append(obs_kp)

    print(
        f"{drug:12s} | {logP:4.1f} | {fup:.2f} | {str(pKa) if pKa else '-':3s} | {drug_type:4s} | {water:.3f} | {lipid:.4f} | {apl:.3f} | {pred_kp:4.2f} | {obs_kp:4.2f} | {fold_error:.2f}×"
    )

# Calculate metrics
predictions = np.array(predictions)
observations = np.array(observations)
log_errors = np.abs(
    np.log10(np.clip(predictions, 0.001, 100))
    - np.log10(np.clip(observations, 0.001, 100))
)
gmfe = 10 ** np.mean(log_errors)
fold_errors = np.maximum(predictions / observations, observations / predictions)
within_2fold = np.sum(fold_errors <= 2.0) / len(fold_errors) * 100
within_3fold = np.sum(fold_errors <= 3.0) / len(fold_errors) * 100

print(f"""
MUSCLE Kp PREDICTION PERFORMANCE:
---------------------------------
GMFE: {gmfe:.2f}
Within 2-fold: {within_2fold:.0f}%
Within 3-fold: {within_3fold:.0f}%
""")

print("=" * 70)
print("4. COMPONENT ANALYSIS")
print("=" * 70)

print("""
Breaking down what drives Kp for different drug types:

HYDROPHILIC NEUTRALS (Caffeine, Theophylline):
  - Water term dominates (~0.75 × fup)
  - Lipid term negligible
  - APL term = 0
  - Kp ≈ 0.5-0.7

LIPOPHILIC NEUTRALS (Diazepam):
  - Water term: ~0.75 × fup
  - Lipid term: adds 0.01-0.05
  - APL term = 0 (not a base)
  - Kp depends mainly on fup

STRONG BASES (Propranolol, Imipramine):
  - Water term: enhanced by ion trapping (2×)
  - Lipid term: moderate
  - APL term: SIGNIFICANT (Ka_AP × 0.00153 × X)
  - Kp can be >> 1 even with low fup!

ACIDS (Warfarin, Phenytoin):
  - Water term: reduced by ion trapping in plasma
  - APL term = 0 (electrostatic repulsion)
  - Kp tends to be LOW
""")

print("=" * 70)
print("5. CLINICAL IMPLICATIONS: PATIENT VARIABILITY")
print("=" * 70)

print("""
MUSCLE VOLUME VARIES SIGNIFICANTLY:

Population              | Muscle (L) | % of Reference
------------------------|------------|---------------
Young male athlete      |    40      |     133%
Reference (young male)  |    30      |     100%
Young female            |    25      |      83%
Elderly male (70+)      |    22      |      73%
Elderly female (70+)    |    18      |      60%
Cancer cachexia         |    15      |      50%

IMPACT ON Vdss:
For a drug with Kp_muscle = 2:
  - Athlete: contributes 80 L
  - Elderly female: contributes 36 L
  - DIFFERENCE: 44 L!

This is why Vdss varies between patients even for same fup!
""")

print("=" * 70)
print("6. EXERCISE EFFECTS")
print("=" * 70)

print("""
MUSCLE BLOOD FLOW CHANGES DRAMATICALLY WITH EXERCISE:

Condition        | Flow (L/min/kg) | For 30L muscle | Equilibration
-----------------|-----------------|----------------|---------------
Rest             |     0.025       |    0.75 L/min  |    ~40 min
Light exercise   |     0.10        |    3.0 L/min   |    ~10 min
Moderate exercise|     0.25        |    7.5 L/min   |     ~4 min
Heavy exercise   |     0.50        |   15.0 L/min   |     ~2 min
Maximum          |     1.0+        |   30+ L/min    |     <1 min

CLINICAL RELEVANCE:
- Exercise during therapy = faster muscle uptake
- May affect drug response timing
- Important for performance drugs, insulin, etc.
""")

print("=" * 70)
print("7. KEY FEATURES FOR ML MODELING")
print("=" * 70)

print("""
RECOMMENDED MUSCLE-SPECIFIC FEATURES:

1. Ion Trapping Ratio
   ion_trap = (1 + 10^(pKa - 7.0)) / (1 + 10^(pKa - 7.4))
   Range: 1.0 (neutral) to 2.5 (strong base)

2. APL Binding Potential
   apl_binding = Ka_AP × 0.00153 × X / (1 + Y)
   Significant for strong bases only

3. Muscle Kp
   kp_muscle = (water_term + lipid_term + apl_term) × fup

4. Muscle Contribution
   contrib_muscle = kp_muscle × muscle_volume

5. Muscle Fraction
   frac_muscle = contrib_muscle / total_Vdss

6. Strong Base Flag
   is_strong_base = pKa > 8

7. Patient-Specific Volume
   muscle_volume = f(weight, age, sex, condition)
""")

print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
MUSCLE TISSUE - KEY TAKEAWAYS:

1. LARGEST COMPARTMENT (30L, 43% body weight)
   → Dominates Vdss for most drugs
   → Even Kp = 0.5 contributes 15L

2. HIGH WATER CONTENT (75%)
   → Hydrophilic drugs: Kp ≈ 0.5-0.7
   → Main reservoir for water-soluble drugs

3. pH GRADIENT (7.0 vs 7.4)
   → Bases are 2.5× more ionized in muscle
   → Ion trapping drives accumulation

4. ACIDIC PHOSPHOLIPIDS (0.153%)
   → Bind positively charged bases
   → Ka_AP is key unknown parameter
   → Strong bases: Kp can be >> 1

5. VALIDATION: GMFE = {gmfe:.2f}
   → {within_2fold:.0f}% within 2-fold
   → Reasonable but needs better Ka_AP estimation

6. PATIENT VARIABILITY
   → Muscle volume varies 2× (age, sex, cachexia)
   → Major source of Vdss inter-individual variability

NEXT: Proceed to LIVER tissue analysis
""")
