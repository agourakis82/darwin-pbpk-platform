#!/usr/bin/env python3
"""
Test Rodgers-Rowland predictions with ionization classification
Using RDKit to classify drugs by ionizable groups and assign typical pKa values
"""

import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

warnings.filterwarnings("ignore")

print("=" * 60)
print("RODGERS-ROWLAND WITH IONIZATION CLASSIFICATION")
print("=" * 60)

# Tissue composition (Rodgers-Rowland 2005, 2006)
TISSUE_COMPOSITION = {
    # tissue: (f_n_l, f_n_pl, f_a_pl, f_ew, f_iw, AR, LR, volume_L)
    "adipose": (0.853, 0.0016, 0.0004, 0.135, 0.017, 0.049, 0.068, 12.0),
    "bone": (0.017, 0.0017, 0.00067, 0.100, 0.346, 0.100, 0.050, 4.0),
    "brain": (0.039, 0.0015, 0.0004, 0.162, 0.620, 0.048, 0.041, 1.4),
    "gut": (0.038, 0.0125, 0.00241, 0.282, 0.475, 0.158, 0.141, 1.2),
    "heart": (0.014, 0.0111, 0.00225, 0.320, 0.456, 0.157, 0.160, 0.33),
    "kidney": (0.012, 0.0242, 0.00503, 0.273, 0.483, 0.130, 0.137, 0.31),
    "liver": (0.014, 0.0240, 0.00456, 0.161, 0.573, 0.086, 0.161, 1.8),
    "lung": (0.022, 0.0128, 0.00391, 0.336, 0.446, 0.212, 0.168, 1.0),
    "muscle": (0.010, 0.0072, 0.00153, 0.118, 0.630, 0.064, 0.059, 30.0),
    "pancreas": (0.041, 0.0093, 0.00167, 0.120, 0.664, 0.060, 0.060, 0.1),
    "skin": (0.060, 0.0044, 0.00132, 0.382, 0.291, 0.277, 0.096, 3.0),
    "spleen": (0.0077, 0.0113, 0.00318, 0.207, 0.579, 0.097, 0.207, 0.18),
    "rbc": (0.0017, 0.0029, 0.0005, 0.0, 0.603, 0.0, 0.0, 2.5),
    "plasma": (0.0023, 0.0013, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0),
}

# pH constants
pH_IW = 7.0
pH_P = 7.4
pH_RBC = 7.22
HCT = 0.45


def classify_ionization(smiles):
    """
    Classify drug ionization type from SMILES using RDKit SMARTS patterns
    Returns: (ion_type, estimated_pKa_list)

    Ion types:
    1 = neutral
    2 = monoprotic acid
    3 = monoprotic base
    4 = diprotic acid
    5 = diprotic base
    6 = zwitterion
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 1, []  # Default to neutral

    # Acidic groups with typical pKa
    acidic_patterns = [
        ("C(=O)[OH]", 4.0),  # Carboxylic acid
        ("S(=O)(=O)[OH]", 1.0),  # Sulfonic acid
        ("P(=O)([OH])", 2.0),  # Phosphoric acid
        ("[OH]c1ccccc1", 10.0),  # Phenol
        ("S[H]", 8.0),  # Thiol
    ]

    # Basic groups with typical pKa
    basic_patterns = [
        ("[NH2]c1ccccc1", 4.5),  # Aniline
        ("[NH2][CH2]", 10.5),  # Primary aliphatic amine
        ("[NH]([CH2,CH3])([CH2,CH3])", 10.5),  # Secondary aliphatic amine
        ("[N]([CH2,CH3])([CH2,CH3])([CH2,CH3])", 9.5),  # Tertiary aliphatic amine
        ("c1nccnc1", 5.5),  # Pyrimidine
        ("c1ccncc1", 5.2),  # Pyridine
        ("C1=NCCC1", 7.0),  # Imidazoline
        ("c1c[nH]cn1", 7.0),  # Imidazole
        ("[NH]=C(N)", 12.5),  # Guanidine
        ("c1ccc2[nH]ccc2c1", 3.5),  # Indole (weak)
    ]

    acidic_pkas = []
    basic_pkas = []

    for pattern, pka in acidic_patterns:
        try:
            pat = Chem.MolFromSmarts(pattern)
            if pat and mol.HasSubstructMatch(pat):
                matches = mol.GetSubstructMatches(pat)
                acidic_pkas.extend([pka] * len(matches))
        except:
            pass

    for pattern, pka in basic_patterns:
        try:
            pat = Chem.MolFromSmarts(pattern)
            if pat and mol.HasSubstructMatch(pat):
                matches = mol.GetSubstructMatches(pat)
                basic_pkas.extend([pka] * len(matches))
        except:
            pass

    n_acid = len(acidic_pkas)
    n_base = len(basic_pkas)

    # Classify
    if n_acid == 0 and n_base == 0:
        return 1, []  # Neutral
    elif n_acid >= 1 and n_base == 0:
        return 2, [min(acidic_pkas)]  # Acid (use strongest)
    elif n_acid == 0 and n_base >= 1:
        return 3, [max(basic_pkas)]  # Base (use strongest)
    else:
        # Zwitterion - return both
        return 6, [min(acidic_pkas), max(basic_pkas)]


def calculate_ionization_factors(pka_list, ion_type, pH):
    """Calculate ionization factor at given pH"""
    if ion_type == 1 or len(pka_list) == 0:  # Neutral
        return 0.0
    elif ion_type == 2:  # Acid
        return 10 ** (pH - pka_list[0])
    elif ion_type == 3:  # Base
        return 10 ** (pka_list[0] - pH)
    elif ion_type == 6 and len(pka_list) >= 2:  # Zwitterion
        return 10 ** (pka_list[1] - pH) + 10 ** (pH - pka_list[0])
    return 0.0


def calculate_kp_rodgers_rowland(logP, fup, pka_list, ion_type, tissue_name, BP=1.0):
    """
    Calculate tissue:plasma partition coefficient using Rodgers-Rowland equations
    """
    tissue = TISSUE_COMPOSITION[tissue_name]
    f_n_l, f_n_pl, f_a_pl, f_ew, f_iw, AR, LR, vol = tissue

    plasma = TISSUE_COMPOSITION["plasma"]
    rbc = TISSUE_COMPOSITION["rbc"]

    P = 10**logP
    # Oil:water from octanol:water for adipose
    P_OW = 10 ** (1.115 * logP - 1.35) if tissue_name == "adipose" else P
    P_tissue = P_OW if tissue_name == "adipose" else P

    # Ionization factors
    X = calculate_ionization_factors(pka_list, ion_type, pH_IW)
    Y = calculate_ionization_factors(pka_list, ion_type, pH_P)
    Z = (
        calculate_ionization_factors(pka_list, ion_type, pH_RBC)
        if ion_type == 3
        else 1.0
    )

    # Blood cell partition
    Kpu_bc = (HCT - 1 + BP) / (HCT * fup) if fup > 0 else 1.0

    # Affinity constants
    denom_Y = max(1 + Y, 1e-10)
    Ka_PR = max(
        0, (1 / fup - 1 - (P * plasma[0] + (0.3 * P + 0.7) * plasma[1]) / denom_Y)
    )

    # For bases: acidic phospholipid binding
    Z_safe = max(Z, 1e-10)
    f_a_pl_rbc = rbc[2]
    Ka_AP = max(
        0,
        (
            Kpu_bc
            - (1 + Z) / denom_Y * rbc[4]
            - (P * rbc[0] + (0.3 * P + 0.7) * rbc[1]) / denom_Y
        )
        * denom_Y
        / (f_a_pl_rbc * Z_safe),
    )

    # Determine calculation type
    is_strong_base = ion_type == 3 and len(pka_list) > 0 and pka_list[0] > 7

    # Calculate Kp
    if is_strong_base:
        # Use acidic phospholipid binding for strong bases
        Kp = (
            f_ew
            + ((1 + X) / denom_Y) * f_iw
            + (P_tissue * f_n_l + (0.3 * P_tissue + 0.7) * f_n_pl) / denom_Y
            + (Ka_AP * f_a_pl * X) / denom_Y
        ) * fup
    elif ion_type == 2 or ion_type == 6:
        # Acids and zwitterions: albumin binding
        Kp = (
            f_ew
            + ((1 + X) / denom_Y) * f_iw
            + (P_tissue * f_n_l + (0.3 * P_tissue + 0.7) * f_n_pl) / denom_Y
            + (Ka_PR * AR * X) / denom_Y
        ) * fup
    else:
        # Neutrals and weak bases: lipoprotein binding
        Kp = (
            f_ew
            + ((1 + X) / denom_Y) * f_iw
            + (P_tissue * f_n_l + (0.3 * P_tissue + 0.7) * f_n_pl) / denom_Y
            + (Ka_PR * LR * X) / denom_Y
        ) * fup

    return max(Kp, 0.01)


def calculate_vdss_rr(logP, fup, pka_list, ion_type, BP=1.0):
    """Calculate Vdss using Rodgers-Rowland Kp values"""
    Vp = TISSUE_COMPOSITION["plasma"][7]  # Plasma volume

    vdss = Vp
    for tissue_name, tissue_data in TISSUE_COMPOSITION.items():
        if tissue_name not in ["rbc", "plasma"]:
            kp = calculate_kp_rodgers_rowland(
                logP, fup, pka_list, ion_type, tissue_name, BP
            )
            vdss += kp * tissue_data[7]  # Kp × volume

    # RBC contribution
    Kp_rbc = BP * (1 - HCT) / HCT + 1
    vdss += Kp_rbc * TISSUE_COMPOSITION["rbc"][7]

    return vdss / 70.0  # Convert to L/kg


def calculate_vdss_simple(logP, fup):
    """Simple Øie-Tozer with empirical fut (our current best)"""
    P = 10**logP
    fut = 1 / (1 + 0.05 * np.clip(P, 0.001, 1e6))
    fut = np.clip(fut, 0.01, 0.99)

    Vp = 0.043  # L/kg
    Ve = 0.15  # L/kg
    Vr = 0.45  # L/kg

    return Vp + (Ve + Vr) * (fup / fut)


# Load dataset
print("\nLoading Lombardo dataset...")
df = pd.read_csv(
    "/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/obach_lombardo_1352_drugs.csv"
)
print(f"Total compounds: {len(df)}")

# Filter valid data
df = df.dropna(subset=["human_VDss_L_kg", "human_fup", "MoKa.LogP"])
df = df[(df["human_fup"] > 0) & (df["human_fup"] < 1)]
df = df[(df["human_VDss_L_kg"] > 0)]
df = df[(df["MoKa.LogP"] > -5) & (df["MoKa.LogP"] < 8)]
print(f"Valid compounds: {len(df)}")

# Classify ionization for all compounds
print("\nClassifying ionization types...")
ion_types = []
pka_lists = []
for smiles in df["smiles_r"]:
    ion_type, pka_list = classify_ionization(smiles)
    ion_types.append(ion_type)
    pka_lists.append(pka_list)

df["ion_type"] = ion_types
df["pka_list"] = pka_lists

# Count by type
type_names = {1: "Neutral", 2: "Acid", 3: "Base", 6: "Zwitterion"}
print("\nIonization distribution:")
for t, name in type_names.items():
    count = sum(1 for x in ion_types if x == t)
    print(f"  {name}: {count} ({100 * count / len(ion_types):.1f}%)")

# Calculate predictions
print("\n" + "=" * 60)
print("COMPARING PREDICTION METHODS")
print("=" * 60)

vdss_obs = df["human_VDss_L_kg"].values
vdss_rr = []
vdss_simple = []

for idx, row in df.iterrows():
    logP = row["MoKa.LogP"]
    fup = row["human_fup"]
    ion_type = row["ion_type"]
    pka_list = row["pka_list"]

    # Rodgers-Rowland prediction
    vdss_rr.append(calculate_vdss_rr(logP, fup, pka_list, ion_type))

    # Simple prediction
    vdss_simple.append(calculate_vdss_simple(logP, fup))

vdss_rr = np.array(vdss_rr)
vdss_simple = np.array(vdss_simple)


# Calculate GMFE
def gmfe(pred, obs):
    log_err = np.abs(np.log10(pred) - np.log10(obs))
    return 10 ** np.mean(log_err)


def within_fold(pred, obs, fold):
    fe = np.maximum(pred / obs, obs / pred)
    return 100 * np.sum(fe <= fold) / len(fe)


print("\n--- All Compounds ---")
print(f"Rodgers-Rowland GMFE: {gmfe(vdss_rr, vdss_obs):.3f}")
print(f"  Within 2-fold: {within_fold(vdss_rr, vdss_obs, 2):.1f}%")
print(f"  Within 3-fold: {within_fold(vdss_rr, vdss_obs, 3):.1f}%")
print(f"\nSimple Øie-Tozer GMFE: {gmfe(vdss_simple, vdss_obs):.3f}")
print(f"  Within 2-fold: {within_fold(vdss_simple, vdss_obs, 2):.1f}%")
print(f"  Within 3-fold: {within_fold(vdss_simple, vdss_obs, 3):.1f}%")

# By ionization type
print("\n--- By Ionization Type ---")
for t, name in type_names.items():
    mask = df["ion_type"] == t
    if mask.sum() > 10:
        obs_t = vdss_obs[mask]
        rr_t = vdss_rr[mask]
        simple_t = vdss_simple[mask]
        print(f"\n{name} (n={mask.sum()}):")
        print(
            f"  R-R GMFE: {gmfe(rr_t, obs_t):.3f}, Simple GMFE: {gmfe(simple_t, obs_t):.3f}"
        )

# Hybrid approach: use best method per drug type
print("\n" + "=" * 60)
print("HYBRID APPROACH")
print("=" * 60)

# For each compound, use the method that works better for its type
# Based on analysis, we can choose:
# - Strong bases (pKa > 7): R-R may be better
# - Acids: Simple may be better
# - Neutrals: Test both

vdss_hybrid = []
for idx, row in df.iterrows():
    i = df.index.get_loc(idx)
    ion_type = row["ion_type"]
    pka_list = row["pka_list"]

    # Use ionization-aware weighting
    if ion_type == 3 and len(pka_list) > 0 and pka_list[0] > 8:
        # Strong base - Rodgers-Rowland may capture phospholipid binding
        vdss_hybrid.append(vdss_rr[i])
    else:
        # Default to simple for now (we know it works)
        vdss_hybrid.append(vdss_simple[i])

vdss_hybrid = np.array(vdss_hybrid)
print(f"\nHybrid GMFE: {gmfe(vdss_hybrid, vdss_obs):.3f}")
print(f"  Within 2-fold: {within_fold(vdss_hybrid, vdss_obs, 2):.1f}%")

# Feature engineering: use R-R Kp ratios as features
print("\n" + "=" * 60)
print("USING R-R INSIGHTS AS FEATURES FOR ML")
print("=" * 60)

# Calculate tissue Kp ratios as features
print("\nCalculating mechanistic features...")
features = []
for idx, row in df.iterrows():
    logP = row["MoKa.LogP"]
    fup = row["human_fup"]
    ion_type = row["ion_type"]
    pka_list = row["pka_list"]

    # Calculate key tissue Kps
    kp_muscle = calculate_kp_rodgers_rowland(logP, fup, pka_list, ion_type, "muscle")
    kp_adipose = calculate_kp_rodgers_rowland(logP, fup, pka_list, ion_type, "adipose")
    kp_liver = calculate_kp_rodgers_rowland(logP, fup, pka_list, ion_type, "liver")
    kp_kidney = calculate_kp_rodgers_rowland(logP, fup, pka_list, ion_type, "kidney")
    kp_brain = calculate_kp_rodgers_rowland(logP, fup, pka_list, ion_type, "brain")

    # Mechanistic Vdss
    vdss_mech = calculate_vdss_rr(logP, fup, pka_list, ion_type)

    # Effective fut (volume-weighted)
    total_vol = sum(
        t[7] for name, t in TISSUE_COMPOSITION.items() if name not in ["rbc", "plasma"]
    )
    kpu_sum = 0
    for tissue_name, tissue_data in TISSUE_COMPOSITION.items():
        if tissue_name not in ["rbc", "plasma"]:
            kp = calculate_kp_rodgers_rowland(
                logP, fup, pka_list, ion_type, tissue_name
            )
            kpu_sum += (kp / fup) * tissue_data[7]
    fut_eff = total_vol / kpu_sum if kpu_sum > 0 else 0.5

    # Simple fut
    P = 10**logP
    fut_simple = 1 / (1 + 0.05 * np.clip(P, 0.001, 1e6))

    features.append(
        {
            "fup": fup,
            "log_fup": np.log(fup),
            "logP": logP,
            "fut_rr": fut_eff,
            "log_fut_rr": np.log(max(fut_eff, 0.001)),
            "fut_simple": fut_simple,
            "fup_fut_rr": fup / max(fut_eff, 0.001),
            "fup_fut_simple": fup / fut_simple,
            "log_vdss_rr": np.log(max(vdss_mech, 0.001)),
            "log_vdss_simple": np.log(calculate_vdss_simple(logP, fup)),
            "kp_muscle": kp_muscle,
            "kp_adipose": kp_adipose,
            "kp_liver": kp_liver,
            "ion_type": ion_type,
            "is_base": 1 if ion_type == 3 else 0,
            "is_acid": 1 if ion_type == 2 else 0,
            "is_zwitterion": 1 if ion_type == 6 else 0,
        }
    )

feature_df = pd.DataFrame(features)

# Random Forest with mechanistic features
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

X = feature_df[
    [
        "fup",
        "log_fup",
        "logP",
        "fut_rr",
        "log_fut_rr",
        "fut_simple",
        "fup_fut_rr",
        "fup_fut_simple",
        "log_vdss_rr",
        "log_vdss_simple",
        "kp_muscle",
        "kp_adipose",
        "kp_liver",
        "is_base",
        "is_acid",
        "is_zwitterion",
    ]
].values
y = np.log(vdss_obs)

# Remove any inf/nan
valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[valid_mask]
y = y[valid_mask]
vdss_obs_clean = vdss_obs[valid_mask]

print(f"\nValid samples for ML: {len(y)}")

# Cross-validation
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
y_pred_cv = cross_val_predict(rf, X, y, cv=5)
vdss_pred = np.exp(y_pred_cv)

print(f"\nRandom Forest + R-R Features GMFE: {gmfe(vdss_pred, vdss_obs_clean):.3f}")
print(f"  Within 2-fold: {within_fold(vdss_pred, vdss_obs_clean, 2):.1f}%")
print(f"  Within 3-fold: {within_fold(vdss_pred, vdss_obs_clean, 3):.1f}%")

# Feature importance
rf.fit(X, y)
feature_names = [
    "fup",
    "log_fup",
    "logP",
    "fut_rr",
    "log_fut_rr",
    "fut_simple",
    "fup_fut_rr",
    "fup_fut_simple",
    "log_vdss_rr",
    "log_vdss_simple",
    "kp_muscle",
    "kp_adipose",
    "kp_liver",
    "is_base",
    "is_acid",
    "is_zwitterion",
]
importance = rf.feature_importances_
print("\nFeature Importance (Top 10):")
for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1])[:10]:
    print(f"  {name}: {imp:.3f}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("CONCLUSION")
print("="*60)
print("""
The Rodgers-Rowland mechanistic approach provides:
1. Ionization-aware tissue partition prediction
2. Physiologically meaningful features (Kp by tissue)
3. Better understanding of WHY drugs distribute

Combined with ML, the mechanistic features may improve predictions
by incorporating tissue-specific binding insights.
""")
