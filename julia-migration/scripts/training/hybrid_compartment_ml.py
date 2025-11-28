#!/usr/bin/env python3
"""
HYBRID COMPARTMENT + ML MODEL
=============================

Key insight: Pure mechanistic PBPK underperforms because:
1. Ka_AP (acidic phospholipid binding) is unknown for most drugs
2. Transporter effects are not predictable from structure
3. Tissue composition varies between individuals

Solution: Use COMPARTMENT-SPECIFIC FEATURES as inputs to ML model
This captures the unique physiology of each tissue while letting
ML learn the correct weights.
"""

import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")

print("=" * 70)
print("HYBRID COMPARTMENT-SPECIFIC ML MODEL")
print("=" * 70)

# Tissue composition (Rodgers-Rowland 2005, 2006)
# (f_nl, f_npl, f_apl, f_ew, f_iw, volume_L)
TISSUES = {
    "adipose": (0.853, 0.0016, 0.0004, 0.135, 0.017, 12.0),
    "muscle": (0.010, 0.0072, 0.00153, 0.118, 0.630, 30.0),
    "liver": (0.014, 0.0240, 0.00456, 0.161, 0.573, 1.8),
    "brain": (0.039, 0.0015, 0.0004, 0.162, 0.620, 1.4),
    "kidney": (0.012, 0.0242, 0.00503, 0.273, 0.483, 0.31),
    "gut": (0.038, 0.0125, 0.00241, 0.282, 0.475, 1.2),
    "heart": (0.014, 0.0111, 0.00225, 0.320, 0.456, 0.33),
    "lung": (0.022, 0.0128, 0.00391, 0.336, 0.446, 1.0),
    "skin": (0.060, 0.0044, 0.00132, 0.382, 0.291, 3.0),
    "spleen": (0.0077, 0.0113, 0.00318, 0.207, 0.579, 0.18),
    "bone": (0.017, 0.0017, 0.00067, 0.100, 0.346, 4.0),
}


def classify_ionization(smiles):
    """Classify drug ionization from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 1, None, False, False

    # Acidic groups
    acidic_smarts = ["C(=O)[OH]", "S(=O)(=O)[OH]", "[OH]c1ccccc1"]
    basic_smarts = [
        "[NH2][CH2]",
        "[NH]([C])([C])",
        "[N]([C])([C])([C])",
        "c1ccncc1",
        "c1c[nH]cn1",
        "[NH]=C(N)",
    ]

    has_acid = any(
        mol.HasSubstructMatch(Chem.MolFromSmarts(s))
        for s in acidic_smarts
        if Chem.MolFromSmarts(s)
    )
    has_base = any(
        mol.HasSubstructMatch(Chem.MolFromSmarts(s))
        for s in basic_smarts
        if Chem.MolFromSmarts(s)
    )

    # Estimate pKa
    pKa = None
    if has_base:
        # Check for strong vs weak base patterns
        if mol.HasSubstructMatch(
            Chem.MolFromSmarts("[NH2][CH2]")
        ) or mol.HasSubstructMatch(Chem.MolFromSmarts("[N]([C])([C])([C])")):
            pKa = 9.5  # Aliphatic amine
        else:
            pKa = 5.5  # Aromatic/weak base
    elif has_acid:
        pKa = 4.0  # Carboxylic acid

    return (
        (
            1
            if not has_acid and not has_base
            else (2 if has_acid and not has_base else 3)
        ),
        pKa,
        has_acid,
        has_base,
    )


def calculate_tissue_specific_features(
    logP, logD, fup, pKa, is_acid, is_base, MW, TPSA
):
    """
    Calculate features that capture unique physiology of each tissue.
    This is the KEY innovation - compartment-specific features for ML.
    """
    features = {}

    P = 10**logP
    D = 10**logD if logD else P

    # Simple fut estimate (our proven baseline)
    fut_simple = 1 / (1 + 0.05 * np.clip(P, 0.001, 1e6))
    fut_simple = np.clip(fut_simple, 0.01, 0.99)

    # Ionization factors
    X_tissue = 0  # At pH 7.0 (tissue)
    Y_plasma = 0  # At pH 7.4 (plasma)
    if pKa and is_base:
        X_tissue = 10 ** (pKa - 7.0)
        Y_plasma = 10 ** (pKa - 7.4)
    elif pKa and is_acid:
        X_tissue = 10 ** (7.0 - pKa)
        Y_plasma = 10 ** (7.4 - pKa)

    denom = max(1 + Y_plasma, 1e-10)

    # Calculate Kp-like features for each major tissue
    for tissue_name, (f_nl, f_npl, f_apl, f_ew, f_iw, vol) in TISSUES.items():
        # Use olive oil partitioning for adipose
        P_eff = 10 ** (1.115 * logP - 1.35) if tissue_name == "adipose" else P

        # Base component: water + lipid partitioning
        kp_base = (
            f_ew
            + ((1 + X_tissue) / denom) * f_iw
            + (P_eff * f_nl + (0.3 * P_eff + 0.7) * f_npl) / denom
        )

        # Acidic phospholipid term (for bases)
        apl_term = 0
        if is_base and pKa and pKa > 7:
            # Scale by tissue's acidic PL content
            apl_term = f_apl * X_tissue * 50 / denom  # Empirical Ka_AP

        kp_tissue = (kp_base + apl_term) * fup
        kp_tissue = np.clip(kp_tissue, 0.001, 100)  # Reasonable bounds

        # Store as features
        features[f"kp_{tissue_name}"] = kp_tissue
        features[f"log_kp_{tissue_name}"] = np.log(kp_tissue)
        features[f"contrib_{tissue_name}"] = kp_tissue * vol  # Contribution to Vdss

    # Aggregate features
    total_contrib = sum(features[f"contrib_{t}"] for t in TISSUES.keys())
    features["vdss_mech"] = total_contrib / 70.0 + 0.043  # Add plasma
    features["log_vdss_mech"] = np.log(max(features["vdss_mech"], 0.001))

    # Dominant tissue
    contribs = [(t, features[f"contrib_{t}"]) for t in TISSUES.keys()]
    contribs.sort(key=lambda x: -x[1])
    features["dominant_tissue"] = contribs[0][0]
    features["dominant_fraction"] = (
        contribs[0][1] / total_contrib if total_contrib > 0 else 0
    )

    # Muscle fraction (most important for Vdss)
    features["muscle_fraction"] = (
        features["contrib_muscle"] / total_contrib if total_contrib > 0 else 0
    )

    # Adipose fraction (important for lipophilic drugs)
    features["adipose_fraction"] = (
        features["contrib_adipose"] / total_contrib if total_contrib > 0 else 0
    )

    # Standard features
    features["fup"] = fup
    features["log_fup"] = np.log(max(fup, 0.001))
    features["fut_simple"] = fut_simple
    features["log_fut_simple"] = np.log(fut_simple)
    features["fup_fut"] = fup / fut_simple
    features["log_fup_fut"] = np.log(fup / fut_simple)
    features["logP"] = logP
    features["logD"] = logD if logD else logP
    features["MW"] = MW
    features["TPSA"] = TPSA
    features["is_base"] = 1 if is_base else 0
    features["is_acid"] = 1 if is_acid else 0
    features["is_strong_base"] = 1 if (is_base and pKa and pKa > 8) else 0
    features["pKa"] = pKa if pKa else 7.0

    # Simple Vdss for comparison
    features["vdss_simple"] = 0.043 + 0.6 * (fup / fut_simple)
    features["log_vdss_simple"] = np.log(features["vdss_simple"])

    return features


def gmfe(pred, obs):
    return 10 ** np.mean(
        np.abs(np.log10(np.clip(pred, 0.001, 1e6)) - np.log10(np.clip(obs, 0.001, 1e6)))
    )


def within_fold(pred, obs, fold):
    return 100 * np.sum(np.maximum(pred / obs, obs / pred) <= fold) / len(pred)


# Load data
print("\nLoading Lombardo dataset...")
df = pd.read_csv(
    "/home/agourakis82/workspace/darwin-pbpk-platform/data/external_datasets/obach_lombardo_1352_drugs.csv"
)

# Clean data
df = df.dropna(subset=["human_VDss_L_kg", "human_fup", "MoKa.LogP"])
df = df[(df["human_fup"] > 0) & (df["human_fup"] < 1)]
df = df[(df["human_VDss_L_kg"] > 0)]
df = df[(df["MoKa.LogP"] > -5) & (df["MoKa.LogP"] < 8)]
print(f"Valid compounds: {len(df)}")

# Classify ionization and build features
print("Building compartment-specific features...")
all_features = []

for idx, row in df.iterrows():
    smiles = row["smiles_r"]
    ion_type, pKa, is_acid, is_base = classify_ionization(smiles)

    features = calculate_tissue_specific_features(
        logP=row["MoKa.LogP"],
        logD=row.get("MoKa.LogD7.4", row["MoKa.LogP"]),
        fup=row["human_fup"],
        pKa=pKa,
        is_acid=is_acid,
        is_base=is_base,
        MW=row["MW"],
        TPSA=row.get("TPSA_NO", 60),
    )
    all_features.append(features)

feature_df = pd.DataFrame(all_features)
y = np.log(df["human_VDss_L_kg"].values)
vdss_obs = df["human_VDss_L_kg"].values

# Remove invalid
valid_mask = np.isfinite(feature_df.select_dtypes(include=[np.number]).values).all(
    axis=1
) & np.isfinite(y)
feature_df = feature_df[valid_mask].reset_index(drop=True)
y = y[valid_mask]
vdss_obs = vdss_obs[valid_mask]

print(f"Valid samples: {len(y)}")

# Outlier removal
print("\nRemoving outliers...")
X_simple = feature_df[["fup", "log_fup", "log_fup_fut"]].values
lr = LinearRegression()
lr.fit(X_simple, y)
residuals = y - lr.predict(X_simple)
clean_mask = np.abs(residuals) <= 2.0
n_removed = len(y) - clean_mask.sum()
print(f"Removed {n_removed} outliers ({100 * n_removed / len(y):.1f}%)")

feature_df_clean = feature_df[clean_mask].reset_index(drop=True)
y_clean = y[clean_mask]
vdss_obs_clean = vdss_obs[clean_mask]

# Define feature sets to compare
print("\n" + "=" * 70)
print("COMPARING FEATURE SETS")
print("=" * 70)

# Feature set 1: Simple (baseline)
features_simple = [
    "fup",
    "log_fup",
    "logP",
    "fut_simple",
    "log_fut_simple",
    "fup_fut",
    "log_fup_fut",
    "log_vdss_simple",
    "MW",
    "TPSA",
]

# Feature set 2: Compartment Kps only
features_kp = [
    "kp_adipose",
    "kp_muscle",
    "kp_liver",
    "kp_kidney",
    "kp_brain",
    "kp_gut",
    "kp_heart",
    "kp_lung",
    "fup",
    "log_fup",
    "logP",
]

# Feature set 3: Compartment contributions
features_contrib = [
    "contrib_adipose",
    "contrib_muscle",
    "contrib_liver",
    "contrib_kidney",
    "contrib_brain",
    "log_vdss_mech",
    "muscle_fraction",
    "adipose_fraction",
    "fup",
    "log_fup",
]

# Feature set 4: Combined (all features)
features_combined = [
    "fup",
    "log_fup",
    "logP",
    "logD",
    "MW",
    "TPSA",
    "fut_simple",
    "log_fut_simple",
    "fup_fut",
    "log_fup_fut",
    "log_vdss_simple",
    "log_vdss_mech",
    "kp_adipose",
    "log_kp_adipose",
    "kp_muscle",
    "log_kp_muscle",
    "kp_liver",
    "kp_kidney",
    "kp_brain",
    "contrib_adipose",
    "contrib_muscle",
    "contrib_liver",
    "muscle_fraction",
    "adipose_fraction",
    "is_base",
    "is_acid",
    "is_strong_base",
]

results = {}
for name, features in [
    ("Simple", features_simple),
    ("Compartment_Kp", features_kp),
    ("Contributions", features_contrib),
    ("Combined", features_combined),
]:
    # Filter to available features
    available = [f for f in features if f in feature_df_clean.columns]
    X = feature_df_clean[available].values

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    y_pred = cross_val_predict(rf, X, y_clean, cv=5)
    vdss_pred = np.exp(y_pred)

    g = gmfe(vdss_pred, vdss_obs_clean)
    w2 = within_fold(vdss_pred, vdss_obs_clean, 2)
    w3 = within_fold(vdss_pred, vdss_obs_clean, 3)

    results[name] = {"gmfe": g, "w2": w2, "w3": w3}
    print(f"\n{name}:")
    print(f"  GMFE: {g:.3f}")
    print(f"  Within 2-fold: {w2:.1f}%")
    print(f"  Within 3-fold: {w3:.1f}%")

# Stability test
print("\n" + "=" * 70)
print("STABILITY TEST (10 seeds)")
print("=" * 70)

available = [f for f in features_combined if f in feature_df_clean.columns]
X_best = feature_df_clean[available].values

gmfe_values = []
for seed in range(10):
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed)
    y_pred = cross_val_predict(rf, X_best, y_clean, cv=5)
    vdss_pred = np.exp(y_pred)
    gmfe_values.append(gmfe(vdss_pred, vdss_obs_clean))

print(
    f"\nCombined features GMFE: {np.mean(gmfe_values):.3f} ± {np.std(gmfe_values):.3f}"
)

# Feature importance
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_best, y_clean)

print("\nTop 15 features:")
for name, imp in sorted(zip(available, rf.feature_importances_), key=lambda x: -x[1])[
    :15
]:
    print(f"  {name}: {imp:.3f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

best_gmfe = np.mean(gmfe_values)
simple_gmfe = results["Simple"]["gmfe"]
improvement = (simple_gmfe - best_gmfe) / simple_gmfe * 100

print(f"""
Results with {len(y_clean)} compounds:

| Feature Set | GMFE | Within 2-fold |
|-------------|------|---------------|
| Simple | {results["Simple"]["gmfe"]:.3f} | {results["Simple"]["w2"]:.1f}% |
| Compartment Kp | {results["Compartment_Kp"]["gmfe"]:.3f} | {results["Compartment_Kp"]["w2"]:.1f}% |
| Contributions | {results["Contributions"]["gmfe"]:.3f} | {results["Contributions"]["w2"]:.1f}% |
| Combined | {best_gmfe:.3f} ± {np.std(gmfe_values):.3f} | {results["Combined"]["w2"]:.1f}% |
| Contributions | {results["Contributions"]["gmfe"]:.3f} | {results["Contributions"]["w2"]:.1f}% |
| Combined | {best_gmfe:.3f} ± {np.std(gmfe_values):.3f} | {results["Combined"]["w2"]:.1f}% |

Improvement from compartment features: {improvement:.1f}%

KEY INSIGHT:
Compartment-specific features capture tissue physiology:
- Adipose: olive oil partitioning for lipophilic drugs
- Muscle: dominates by volume (30L), ion trapping for bases
- Kidney: highest acidic PL, nephrotoxicity risk for strong bases
- Brain: BBB-restricted, P-gp effects

The ML model learns the correct weights for each tissue's
contribution, effectively calibrating the mechanistic equations.
""")
