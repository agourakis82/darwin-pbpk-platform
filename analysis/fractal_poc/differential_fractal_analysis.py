#!/usr/bin/env python3
"""
Differential Fractal Analysis of Leukocyte Subpopulations
==========================================================

Compares fractal dimensions across:
1. Normal WBC types: Neutrophils, Lymphocytes, Monocytes, Eosinophils
2. Pathological: Leukemia (ALL - Acute Lymphoblastic Leukemia)

Output: Statistical comparison and clinical validation metrics

Created: 2025-12-04
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent


@dataclass
class CellTypeAnalysis:
    """Fractal analysis results for a cell type."""

    cell_type: str
    n_images: int = 0
    n_cells: int = 0
    df_combined: List[float] = field(default_factory=list)
    df_edges: List[float] = field(default_factory=list)
    df_distribution: List[float] = field(default_factory=list)
    mean_circularity: List[float] = field(default_factory=list)
    cell_areas: List[int] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)


def load_analysis_results(results_dir: Path) -> CellTypeAnalysis:
    """Load fractal analysis results from Julia output."""
    analysis = CellTypeAnalysis(cell_type=results_dir.name)

    json_files = list(results_dir.glob("*_masks.json"))

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        analysis.n_images += 1
        analysis.n_cells += data.get("n_cells", 0)

        # Extract cell properties
        for cell in data.get("cell_properties", []):
            analysis.cell_areas.append(cell.get("area", 0))
            analysis.confidence_scores.append(cell.get("score", 0))

    return analysis


def run_sam3_segmentation(
    image_dir: Path, output_dir: Path, cell_type: str, max_images: int = 10
) -> int:
    """Run SAM-3 segmentation on images."""
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Count cells from batch summary
    summary_path = output_dir / "batch_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        return summary.get("total_cells", 0)

    return 0


def run_julia_fractal_analysis(masks_dir: Path) -> Dict:
    """Run Julia fractal analysis on exported masks."""
    import subprocess

    julia_script = """
    push!(LOAD_PATH, joinpath(@__DIR__, "src", "DarwinPBPK", "image_analysis"))
    include(joinpath(@__DIR__, "src", "DarwinPBPK", "image_analysis", "sam3_integration.jl"))
    using .SAM3Integration
    using JSON3

    masks_dir = ARGS[1]
    results = analyze_batch_sam3(masks_dir)
    report = generate_fractal_report(results)

    # Add per-image details
    image_results = []
    for r in results
        push!(image_results, Dict(
            "source" => r.source_image,
            "n_cells" => r.n_cells,
            "df_combined" => r.df_combined,
            "df_edges" => r.df_edges,
            "df_distribution" => r.df_distribution,
            "mean_df_edge" => r.mean_df_edge,
            "std_df_edge" => r.std_df_edge,
            "mean_circularity" => r.mean_circularity
        ))
    end
    report["image_results"] = image_results

    println(JSON3.write(report))
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
        return {}

    # Parse JSON output (last line)
    output_lines = result.stdout.strip().split("\n")
    json_line = output_lines[-1]

    try:
        return json.loads(json_line)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse Julia output: {json_line[:200]}")
        return {}


def statistical_comparison(results: Dict[str, Dict]) -> Dict:
    """Perform statistical comparison between cell types."""
    comparisons = {}

    cell_types = list(results.keys())

    # Extract Df values for each type
    df_values = {}
    for ct, data in results.items():
        if "image_results" in data:
            df_values[ct] = [
                r["df_combined"]
                for r in data["image_results"]
                if r["df_combined"] and not np.isnan(r["df_combined"])
            ]

    # Pairwise comparisons
    for i, ct1 in enumerate(cell_types):
        for ct2 in cell_types[i + 1 :]:
            if ct1 in df_values and ct2 in df_values:
                vals1 = df_values[ct1]
                vals2 = df_values[ct2]

                if len(vals1) >= 2 and len(vals2) >= 2:
                    # Mann-Whitney U test (non-parametric)
                    stat, p_value = stats.mannwhitneyu(
                        vals1, vals2, alternative="two-sided"
                    )

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((np.std(vals1) ** 2 + np.std(vals2) ** 2) / 2)
                    cohens_d = (
                        (np.mean(vals1) - np.mean(vals2)) / pooled_std
                        if pooled_std > 0
                        else 0
                    )

                    comparisons[f"{ct1}_vs_{ct2}"] = {
                        "mann_whitney_U": float(stat),
                        "p_value": float(p_value),
                        "cohens_d": float(cohens_d),
                        "significant": p_value < 0.05,
                        "mean_diff": float(np.mean(vals1) - np.mean(vals2)),
                        "n1": len(vals1),
                        "n2": len(vals2),
                    }

    # ANOVA if we have 3+ groups
    if len(df_values) >= 3:
        groups = [v for v in df_values.values() if len(v) >= 2]
        if len(groups) >= 3:
            stat, p_value = stats.kruskal(*groups)  # Non-parametric ANOVA
            comparisons["kruskal_wallis"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            }

    return comparisons


def clinical_validation(results: Dict[str, Dict]) -> Dict:
    """
    Clinical validation metrics.

    Compares:
    - Normal WBC (pooled) vs Leukemia
    - Sensitivity/Specificity for leukemia detection based on Df threshold
    """
    validation = {}

    # Get leukemia data
    leukemia_df = []
    if "leukemia" in results and "image_results" in results["leukemia"]:
        leukemia_df = [
            r["df_combined"]
            for r in results["leukemia"]["image_results"]
            if r["df_combined"] and not np.isnan(r["df_combined"])
        ]

    # Pool normal WBC types
    normal_types = ["neutrophils", "lymphocytes", "monocytes", "eosinophils"]
    normal_df = []
    for ct in normal_types:
        if ct in results and "image_results" in results[ct]:
            normal_df.extend(
                [
                    r["df_combined"]
                    for r in results[ct]["image_results"]
                    if r["df_combined"] and not np.isnan(r["df_combined"])
                ]
            )

    if leukemia_df and normal_df:
        # Statistical comparison
        stat, p_value = stats.mannwhitneyu(
            normal_df, leukemia_df, alternative="two-sided"
        )

        validation["normal_vs_leukemia"] = {
            "normal_mean_df": float(np.mean(normal_df)),
            "normal_std_df": float(np.std(normal_df)),
            "leukemia_mean_df": float(np.mean(leukemia_df)),
            "leukemia_std_df": float(np.std(leukemia_df)),
            "mann_whitney_p": float(p_value),
            "significant": p_value < 0.05,
            "n_normal": len(normal_df),
            "n_leukemia": len(leukemia_df),
        }

        # ROC-like analysis: find optimal threshold
        all_df = normal_df + leukemia_df
        labels = [0] * len(normal_df) + [1] * len(leukemia_df)

        best_threshold = None
        best_youden = -1
        best_sens = 0
        best_spec = 0

        thresholds = np.linspace(min(all_df), max(all_df), 50)

        for thresh in thresholds:
            # Predict leukemia if Df > threshold (hypothesis: leukemia has higher Df)
            predictions = [1 if df > thresh else 0 for df in all_df]

            tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
            tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
            fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            youden = sensitivity + specificity - 1

            if youden > best_youden:
                best_youden = youden
                best_threshold = thresh
                best_sens = sensitivity
                best_spec = specificity

        validation["diagnostic_performance"] = {
            "optimal_threshold_df": float(best_threshold) if best_threshold else None,
            "sensitivity": float(best_sens),
            "specificity": float(best_spec),
            "youden_index": float(best_youden),
            "interpretation": interpret_diagnostic(best_sens, best_spec),
        }

    return validation


def interpret_diagnostic(sensitivity: float, specificity: float) -> str:
    """Interpret diagnostic performance."""
    if sensitivity >= 0.9 and specificity >= 0.9:
        return "Excellent diagnostic performance"
    elif sensitivity >= 0.8 and specificity >= 0.8:
        return "Good diagnostic performance"
    elif sensitivity >= 0.7 or specificity >= 0.7:
        return "Moderate diagnostic performance - further validation needed"
    else:
        return "Poor diagnostic performance - fractal dimension alone insufficient"


def generate_report(
    results: Dict[str, Dict], comparisons: Dict, validation: Dict, output_path: Path
) -> None:
    """Generate comprehensive analysis report."""

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {},
        "per_cell_type": {},
        "statistical_comparisons": comparisons,
        "clinical_validation": validation,
    }

    # Summary per cell type
    total_cells = 0
    for ct, data in results.items():
        if "summary" in data:
            n_cells = data["summary"].get("total_cells", 0)
            total_cells += n_cells

            fd = data.get("fractal_dimensions", {})
            report["per_cell_type"][ct] = {
                "n_cells": n_cells,
                "n_images": data["summary"].get("n_images", 0),
                "df_combined_mean": fd.get("combined_mask", {}).get("mean"),
                "df_combined_std": fd.get("combined_mask", {}).get("std"),
                "df_edges_mean": fd.get("edge_mask", {}).get("mean"),
                "mean_circularity": data.get("morphology", {}).get("mean_circularity"),
            }

    report["summary"]["total_cells_analyzed"] = total_cells
    report["summary"]["cell_types_analyzed"] = list(results.keys())

    # Save report (convert numpy types to Python native)
    def convert_numpy(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    report = convert_numpy(report)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {output_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("DIFFERENTIAL FRACTAL ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nTotal cells analyzed: {total_cells}")
    print(f"Cell types: {', '.join(results.keys())}")

    print("\n" + "-" * 70)
    print("FRACTAL DIMENSIONS BY CELL TYPE")
    print("-" * 70)
    print(
        f"{'Cell Type':<15} {'N cells':>8} {'Df (mean)':>12} {'Df (std)':>10} {'Circ':>8}"
    )
    print("-" * 70)

    for ct, data in report["per_cell_type"].items():
        df_mean = data.get("df_combined_mean", 0) or 0
        df_std = data.get("df_combined_std", 0) or 0
        circ = data.get("mean_circularity", 0) or 0
        print(
            f"{ct:<15} {data['n_cells']:>8} {df_mean:>12.3f} {df_std:>10.3f} {circ:>8.3f}"
        )

    # Statistical comparisons
    if comparisons:
        print("\n" + "-" * 70)
        print("STATISTICAL COMPARISONS")
        print("-" * 70)

        for comp_name, comp_data in comparisons.items():
            if comp_name == "kruskal_wallis":
                sig = (
                    "***"
                    if comp_data["p_value"] < 0.001
                    else (
                        "**"
                        if comp_data["p_value"] < 0.01
                        else ("*" if comp_data["significant"] else "")
                    )
                )
                print(
                    f"Kruskal-Wallis: H={comp_data['statistic']:.2f}, p={comp_data['p_value']:.4f} {sig}"
                )
            else:
                sig = (
                    "***"
                    if comp_data["p_value"] < 0.001
                    else (
                        "**"
                        if comp_data["p_value"] < 0.01
                        else ("*" if comp_data["significant"] else "")
                    )
                )
                print(
                    f"{comp_name}: U={comp_data['mann_whitney_U']:.1f}, p={comp_data['p_value']:.4f}, d={comp_data['cohens_d']:.2f} {sig}"
                )

    # Clinical validation
    if validation:
        print("\n" + "-" * 70)
        print("CLINICAL VALIDATION (Normal vs Leukemia)")
        print("-" * 70)

        if "normal_vs_leukemia" in validation:
            v = validation["normal_vs_leukemia"]
            print(
                f"Normal WBC:  Df = {v['normal_mean_df']:.3f} +/- {v['normal_std_df']:.3f} (n={v['n_normal']})"
            )
            print(
                f"Leukemia:    Df = {v['leukemia_mean_df']:.3f} +/- {v['leukemia_std_df']:.3f} (n={v['n_leukemia']})"
            )
            sig = "SIGNIFICANT" if v["significant"] else "not significant"
            print(f"Mann-Whitney p = {v['mann_whitney_p']:.4f} ({sig})")

        if "diagnostic_performance" in validation:
            d = validation["diagnostic_performance"]
            print(f"\nDiagnostic Performance:")
            print(f"  Optimal Df threshold: {d['optimal_threshold_df']:.3f}")
            print(f"  Sensitivity: {d['sensitivity'] * 100:.1f}%")
            print(f"  Specificity: {d['specificity'] * 100:.1f}%")
            print(f"  Youden's J: {d['youden_index']:.3f}")
            print(f"  Interpretation: {d['interpretation']}")

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Differential Fractal Analysis")
    parser.add_argument(
        "--max-images", type=int, default=10, help="Max images per cell type"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/differential_analysis",
        help="Output directory",
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip SAM-3 segmentation (use existing)",
    )

    args = parser.parse_args()

    output_dir = SCRIPT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define datasets
    base_data = SCRIPT_DIR / "data" / "leukocytes"

    datasets = {
        "neutrophils": base_data
        / "wbc_classification_raw/dataset2-master/dataset2-master/images/TRAIN/NEUTROPHIL",
        "lymphocytes": base_data
        / "wbc_classification_raw/dataset2-master/dataset2-master/images/TRAIN/LYMPHOCYTE",
        "monocytes": base_data
        / "wbc_classification_raw/dataset2-master/dataset2-master/images/TRAIN/MONOCYTE",
        "eosinophils": base_data
        / "wbc_classification_raw/dataset2-master/dataset2-master/images/TRAIN/EOSINOPHIL",
        "leukemia": base_data / "leukemia_ALL_raw/Original/Pre",  # Pre-B ALL
    }

    results = {}

    for cell_type, image_dir in datasets.items():
        if not image_dir.exists():
            logger.warning(f"Dataset not found: {image_dir}")
            continue

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing: {cell_type.upper()}")
        logger.info(f"{'=' * 50}")

        masks_dir = output_dir / cell_type

        # Step 1: SAM-3 segmentation
        if not args.skip_segmentation:
            logger.info("Running SAM-3 segmentation...")
            n_cells = run_sam3_segmentation(
                image_dir, masks_dir, cell_type, args.max_images
            )
            logger.info(f"Segmented {n_cells} cells")

        # Step 2: Julia fractal analysis
        if masks_dir.exists() and any(masks_dir.glob("*.npz")):
            logger.info("Running Julia fractal analysis...")
            analysis = run_julia_fractal_analysis(masks_dir)
            if analysis:
                results[cell_type] = analysis
                logger.info(
                    f"Analysis complete: {analysis.get('summary', {}).get('total_cells', 0)} cells"
                )
        else:
            logger.warning(f"No masks found for {cell_type}")

    if not results:
        logger.error("No results to analyze!")
        return

    # Step 3: Statistical comparisons
    logger.info("\nPerforming statistical comparisons...")
    comparisons = statistical_comparison(results)

    # Step 4: Clinical validation
    logger.info("Running clinical validation...")
    validation = clinical_validation(results)

    # Step 5: Generate report
    report_path = output_dir / "differential_analysis_report.json"
    generate_report(results, comparisons, validation, report_path)


if __name__ == "__main__":
    main()
