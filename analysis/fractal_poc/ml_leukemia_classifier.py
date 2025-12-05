#!/usr/bin/env python3
"""
Machine Learning Classifier for Leukemia Detection
===================================================

Combines fractal dimension features with morphological metrics
to classify normal WBC vs leukemia cells.

Features:
- Fractal dimension (combined mask)
- Fractal dimension (edges)
- Fractal dimension (cell distribution)
- Mean cell circularity
- Mean cell area
- Cell count per image
- Confidence scores

Models:
- Random Forest
- Gradient Boosting
- SVM
- Logistic Regression

Created: 2025-12-05
"""

import json
import logging
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent


@dataclass
class ImageFeatures:
    """Features extracted from a single image."""

    source: str
    cell_type: str
    label: int  # 0 = normal, 1 = leukemia
    n_cells: int
    df_combined: float
    df_edges: float
    df_distribution: float
    mean_circularity: float
    mean_df_edge: float
    std_df_edge: float
    mean_area: float = 0.0
    mean_score: float = 0.0


def run_sam3_segmentation(
    image_dir: Path, output_dir: Path, cell_type: str, max_images: int = 50
) -> int:
    """Run SAM-3 segmentation on images."""
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    existing_npz = list(output_dir.glob("*.npz"))
    if len(existing_npz) >= max_images:
        logger.info(f"Skipping {cell_type}: {len(existing_npz)} already processed")
        return len(existing_npz)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "export_sam3_masks.py"),
        "--image-dir",
        str(image_dir),
        "--output-dir",
        str(output_dir),
        "--cell-type",
        cell_type,
        "--max-images",
        str(max_images),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"SAM-3 failed for {cell_type}: {result.stderr}")
        return 0

    summary_path = output_dir / "batch_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        return summary.get("total_cells", 0)

    return 0


def run_julia_fractal_analysis(masks_dir: Path) -> List[Dict]:
    """Run Julia fractal analysis and return per-image results."""
    import subprocess

    julia_script = """
    push!(LOAD_PATH, joinpath(@__DIR__, "src", "DarwinPBPK", "image_analysis"))
    include(joinpath(@__DIR__, "src", "DarwinPBPK", "image_analysis", "sam3_integration.jl"))
    using .SAM3Integration
    using JSON3

    masks_dir = ARGS[1]
    results = analyze_batch_sam3(masks_dir)

    # Return per-image details
    image_results = []
    for r in results
        # Calculate mean area from cell metrics
        areas = [m.area for m in r.cell_metrics if m.area > 0]
        mean_area = isempty(areas) ? 0.0 : sum(areas) / length(areas)

        push!(image_results, Dict(
            "source" => r.source_image,
            "n_cells" => r.n_cells,
            "df_combined" => r.df_combined,
            "df_edges" => r.df_edges,
            "df_distribution" => r.df_distribution,
            "mean_df_edge" => r.mean_df_edge,
            "std_df_edge" => r.std_df_edge,
            "mean_circularity" => r.mean_circularity,
            "mean_area" => mean_area
        ))
    end

    println(JSON3.write(image_results))
    """

    julia_dir = SCRIPT_DIR.parent.parent / "julia-migration"

    result = subprocess.run(
        ["julia", "--project=.", "-e", julia_script, str(masks_dir)],
        cwd=str(julia_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Julia analysis failed: {result.stderr}")
        return []

    output_lines = result.stdout.strip().split("\n")
    json_line = output_lines[-1]

    try:
        return json.loads(json_line)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse Julia output")
        return []


def extract_features(
    results: List[Dict], cell_type: str, is_leukemia: bool
) -> List[ImageFeatures]:
    """Extract features from Julia analysis results."""
    features = []

    for r in results:
        # Skip invalid results
        if r.get("n_cells", 0) == 0:
            continue

        df_combined = r.get("df_combined", 0)
        if df_combined is None or np.isnan(df_combined):
            continue

        feat = ImageFeatures(
            source=r.get("source", ""),
            cell_type=cell_type,
            label=1 if is_leukemia else 0,
            n_cells=r.get("n_cells", 0),
            df_combined=df_combined,
            df_edges=r.get("df_edges", 0) or 0,
            df_distribution=r.get("df_distribution", 0) or 0,
            mean_circularity=r.get("mean_circularity", 0) or 0,
            mean_df_edge=r.get("mean_df_edge", 0) or 0,
            std_df_edge=r.get("std_df_edge", 0) or 0,
            mean_area=r.get("mean_area", 0) or 0,
        )
        features.append(feat)

    return features


def features_to_array(features: List[ImageFeatures]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert features to numpy arrays for ML."""
    X = []
    y = []

    for f in features:
        X.append(
            [
                f.df_combined,
                f.df_edges,
                f.df_distribution,
                f.mean_circularity,
                f.mean_df_edge,
                f.std_df_edge,
                f.n_cells,
                f.mean_area,
            ]
        )
        y.append(f.label)

    return np.array(X), np.array(y)


def train_and_evaluate(X: np.ndarray, y: np.ndarray, output_dir: Path) -> Dict:
    """Train multiple models and evaluate performance."""

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=42, class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            random_state=42, class_weight="balanced", max_iter=1000
        ),
    }

    results = {}
    best_model = None
    best_auc = 0

    feature_names = [
        "df_combined",
        "df_edges",
        "df_distribution",
        "mean_circularity",
        "mean_df_edge",
        "std_df_edge",
        "n_cells",
        "mean_area",
    ]

    for name, model in models.items():
        logger.info(f"Training {name}...")

        # Use scaled data for SVM and LR, unscaled for tree-based
        if name in ["SVM", "Logistic Regression"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        # Cross-validation
        if name in ["SVM", "Logistic Regression"]:
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring="roc_auc"
            )
        else:
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="roc_auc"
            )

        results[name] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_roc": float(auc),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        # Feature importance for tree-based models
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            results[name]["feature_importance"] = {
                feature_names[i]: float(importances[i])
                for i in range(len(feature_names))
            }

        if auc > best_auc:
            best_auc = auc
            best_model = (
                name,
                model,
                scaler if name in ["SVM", "Logistic Regression"] else None,
            )

    # Save best model
    if best_model:
        model_path = output_dir / "best_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "name": best_model[0],
                    "model": best_model[1],
                    "scaler": best_model[2],
                    "feature_names": feature_names,
                },
                f,
            )
        logger.info(f"Best model saved: {best_model[0]} (AUC={best_auc:.3f})")

    return results


def print_results(results: Dict, n_normal: int, n_leukemia: int):
    """Print formatted results."""

    print("\n" + "=" * 70)
    print("MACHINE LEARNING LEUKEMIA CLASSIFIER RESULTS")
    print("=" * 70)

    print(f"\nDataset: {n_normal} normal images, {n_leukemia} leukemia images")
    print(f"Total samples: {n_normal + n_leukemia}")

    print("\n" + "-" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("-" * 70)
    print(
        f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'AUC':>8} {'CV AUC':>12}"
    )
    print("-" * 70)

    for name, metrics in sorted(results.items(), key=lambda x: -x[1]["auc_roc"]):
        print(
            f"{name:<22} {metrics['accuracy']:>10.3f} {metrics['precision']:>10.3f} "
            f"{metrics['recall']:>10.3f} {metrics['f1_score']:>8.3f} {metrics['auc_roc']:>8.3f} "
            f"{metrics['cv_auc_mean']:>6.3f}±{metrics['cv_auc_std']:.3f}"
        )

    # Best model details
    best_name = max(results.keys(), key=lambda x: results[x]["auc_roc"])
    best = results[best_name]

    print("\n" + "-" * 70)
    print(f"BEST MODEL: {best_name}")
    print("-" * 70)

    cm = np.array(best["confusion_matrix"])
    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Normal   Leukemia")
    print(f"Actual Normal     {tn:>6}   {fp:>6}")
    print(f"       Leukemia   {fn:>6}   {tp:>6}")

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"\nClinical Metrics:")
    print(f"  Sensitivity (Recall): {sensitivity * 100:.1f}%")
    print(f"  Specificity:          {specificity * 100:.1f}%")
    print(f"  PPV (Precision):      {ppv * 100:.1f}%")
    print(f"  NPV:                  {npv * 100:.1f}%")
    print(f"  AUC-ROC:              {best['auc_roc']:.3f}")

    # Feature importance
    if "feature_importance" in best:
        print("\nFeature Importance:")
        sorted_features = sorted(
            best["feature_importance"].items(), key=lambda x: -x[1]
        )
        for feat, imp in sorted_features:
            bar = "█" * int(imp * 50)
            print(f"  {feat:<20} {imp:.3f} {bar}")

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ML Leukemia Classifier")
    parser.add_argument(
        "--max-images", type=int, default=50, help="Max images per cell type"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ml_classifier",
        help="Output directory",
    )
    parser.add_argument(
        "--skip-segmentation", action="store_true", help="Skip SAM-3 segmentation"
    )

    args = parser.parse_args()

    output_dir = SCRIPT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_data = SCRIPT_DIR / "data" / "leukocytes"

    # Datasets - use 'normal' directory which has organized data
    datasets = {
        # Normal WBC types
        "neutrophils": (base_data / "normal/neutrophils", False),
        "lymphocytes": (base_data / "normal/lymphocytes", False),
        "monocytes": (base_data / "normal/monocytes", False),
        "eosinophils": (base_data / "normal/eosinophils", False),
        # Leukemia subtypes
        "leukemia_pre": (base_data / "leukemia_ALL_raw/Original/Pre", True),
        "leukemia_early": (base_data / "leukemia_ALL_raw/Original/Early", True),
        "leukemia_pro": (base_data / "leukemia_ALL_raw/Original/Pro", True),
    }

    all_features = []

    for cell_type, (image_dir, is_leukemia) in datasets.items():
        if not image_dir.exists():
            logger.warning(f"Dataset not found: {image_dir}")
            continue

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing: {cell_type.upper()}")
        logger.info(f"{'=' * 50}")

        masks_dir = output_dir / "masks" / cell_type

        # SAM-3 segmentation
        if not args.skip_segmentation:
            logger.info("Running SAM-3 segmentation...")
            # Map to prompt type
            prompt_type = "leukemia" if is_leukemia else cell_type.split("_")[0]
            n_cells = run_sam3_segmentation(
                image_dir, masks_dir, prompt_type, args.max_images
            )
            logger.info(f"Segmented {n_cells} cells")

        # Julia fractal analysis
        if masks_dir.exists() and any(masks_dir.glob("*.npz")):
            logger.info("Running Julia fractal analysis...")
            results = run_julia_fractal_analysis(masks_dir)

            if results:
                features = extract_features(results, cell_type, is_leukemia)
                all_features.extend(features)
                logger.info(f"Extracted features from {len(features)} images")
        else:
            logger.warning(f"No masks found for {cell_type}")

    if not all_features:
        logger.error("No features extracted!")
        return

    # Convert to arrays
    X, y = features_to_array(all_features)

    n_normal = sum(1 for f in all_features if f.label == 0)
    n_leukemia = sum(1 for f in all_features if f.label == 1)

    logger.info(f"\nTotal features: {len(all_features)}")
    logger.info(f"Normal: {n_normal}, Leukemia: {n_leukemia}")

    # Train and evaluate
    logger.info("\nTraining ML models...")
    results = train_and_evaluate(X, y, output_dir)

    # Save results
    results_path = output_dir / "ml_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "n_samples": len(all_features),
                "n_normal": n_normal,
                "n_leukemia": n_leukemia,
                "models": results,
            },
            f,
            indent=2,
        )

    # Print results
    print_results(results, n_normal, n_leukemia)

    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
