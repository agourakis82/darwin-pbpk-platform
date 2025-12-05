#!/usr/bin/env python3
"""
Validate SAM-3 Test Results
============================

Analyzes and validates comprehensive test results for SAM-3 leukocyte segmentation.

Created: 2025-12-01
Author: Darwin PBPK Platform
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_test_results(results_file: Path) -> Dict:
    """Load test results JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def validate_subpopulation_results(results: List[Dict], subpop_name: str) -> Dict:
    """
    Validate results for a subpopulation.

    Returns validation metrics.
    """
    if not results:
        return {"valid": False, "reason": "No test results", "n_tests": 0}

    validation = {
        "valid": True,
        "n_tests": len(results),
        "total_cells": 0,
        "avg_score": 0.0,
        "issues": [],
    }

    for result in results:
        stats = result.get("stats", {})
        n_cells = stats.get("n_cells", 0)
        score_mean = stats.get("score_mean", 0.0)

        validation["total_cells"] += n_cells

        # Validation checks
        if n_cells == 0:
            validation["issues"].append(
                f"No cells detected in {Path(result['image_path']).name}"
            )
            validation["valid"] = False

        if score_mean < 0.3:
            validation["issues"].append(f"Low confidence (score {score_mean:.3f})")

    validation["avg_score"] = np.mean(
        [r.get("stats", {}).get("score_mean", 0.0) for r in results if r.get("stats")]
    )

    validation["avg_cells_per_image"] = (
        validation["total_cells"] / len(results) if results else 0
    )

    return validation


def validate_comprehensive_results(all_results: Dict) -> Dict:
    """
    Validate comprehensive test results.

    Returns validation report.
    """
    validation_report = {
        "timestamp": datetime.now().isoformat(),
        "overall_valid": True,
        "subpopulations": {},
        "pathological": {},
        "summary": {"total_valid": 0, "total_tests": 0, "coverage": {}},
    }

    # Validate subpopulations
    for subpop_name, results in all_results.get("subpopulations", {}).items():
        validation = validate_subpopulation_results(results, subpop_name)
        validation_report["subpopulations"][subpop_name] = validation

        if validation["valid"]:
            validation_report["summary"]["total_valid"] += 1
        validation_report["summary"]["total_tests"] += 1

    # Validate pathological conditions
    for condition_name, results in all_results.get("pathological", {}).items():
        validation = validate_subpopulation_results(results, condition_name)
        validation_report["pathological"][condition_name] = validation

        if validation["valid"]:
            validation_report["summary"]["total_valid"] += 1
        validation_report["summary"]["total_tests"] += 1

    # Calculate coverage
    summary_stats = all_results.get("summary", {})
    validation_report["summary"]["coverage"] = {
        "subpopulations_tested": len(all_results.get("subpopulations", {})),
        "pathological_tested": len(all_results.get("pathological", {})),
        "total_cells_detected": summary_stats.get("total_cells_detected", 0),
        "validation_rate": validation_report["summary"]["total_valid"]
        / validation_report["summary"]["total_tests"]
        if validation_report["summary"]["total_tests"] > 0
        else 0,
    }

    # Overall validity
    validation_report["overall_valid"] = (
        validation_report["summary"]["total_valid"]
        == validation_report["summary"]["total_tests"]
    )

    return validation_report


def print_validation_report(report: Dict):
    """Print validation report."""
    print("=" * 80)
    print("✅ VALIDAÇÃO COMPLETA - SAM-3 LEUCOCITOS")
    print("=" * 80)
    print()

    print("SUBPOPULAÇÕES:")
    print("-" * 80)
    for subpop, validation in report["subpopulations"].items():
        status = "✅" if validation["valid"] else "❌"
        print(f"{status} {subpop.upper()}:")
        print(f"   Testes: {validation['n_tests']}")
        if validation["n_tests"] > 0 and "total_cells" in validation:
            print(f"   Células detectadas: {validation['total_cells']}")
            print(
                f"   Média por imagem: {validation.get('avg_cells_per_image', 0):.1f}"
            )
            print(f"   Score médio: {validation.get('avg_score', 0):.3f}")
        else:
            print(f"   Razão: {validation.get('reason', 'Sem dados')}")
        if validation.get("issues"):
            for issue in validation["issues"]:
                print(f"   ⚠️  {issue}")
        print()

    print("CONDIÇÕES PATOLÓGICAS:")
    print("-" * 80)
    for condition, validation in report["pathological"].items():
        status = "✅" if validation["valid"] else "❌"
        print(f"{status} {condition.upper()}:")
        print(f"   Testes: {validation['n_tests']}")
        if validation["n_tests"] > 0 and "total_cells" in validation:
            print(f"   Células detectadas: {validation['total_cells']}")
        else:
            print(f"   Razão: {validation.get('reason', 'Sem dados')}")
        if validation.get("issues"):
            for issue in validation["issues"]:
                print(f"   ⚠️  {issue}")
        print()

    print("=" * 80)
    print("RESUMO FINAL")
    print("=" * 80)
    print(
        f"Status geral: {'✅ VALIDADO' if report['overall_valid'] else '❌ FALHAS ENCONTRADAS'}"
    )
    print(f"Taxa de validação: {report['summary']['coverage']['validation_rate']:.1%}")
    print(
        f"Total de células detectadas: {report['summary']['coverage']['total_cells_detected']}"
    )
    print("=" * 80)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate SAM-3 test results")
    parser.add_argument("results_file", type=str, help="Path to test results JSON file")

    args = parser.parse_args()

    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"❌ Arquivo não encontrado: {results_file}")
        sys.exit(1)

    # Load and validate
    print(f"📂 Carregando resultados: {results_file.name}")
    all_results = load_test_results(results_file)

    print("🔍 Validando resultados...")
    validation_report = validate_comprehensive_results(all_results)

    # Print report
    print_validation_report(validation_report)

    # Save validation report
    validation_file = results_file.parent / f"validation_{results_file.stem}.json"
    with open(validation_file, "w") as f:
        json.dump(validation_report, f, indent=2, default=str)

    print(f"\n💾 Relatório de validação salvo: {validation_file.name}")


if __name__ == "__main__":
    main()
