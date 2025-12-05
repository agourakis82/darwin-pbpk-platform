#!/usr/bin/env python3
"""
Comprehensive Test Suite for SAM-3 Leukocyte Segmentation
==========================================================

Tests SAM-3 segmentation for all WBC types (neutrophils, lymphocytes,
monocytes, eosinophils, basophils) and pathological conditions.

Created: 2025-12-01
Author: Darwin PBPK Platform
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add sam3 to path
SCRIPT_DIR = Path(__file__).parent
SAM3_DIR = SCRIPT_DIR / "sam3"
sys.path.insert(0, str(SAM3_DIR))

DATA_DIR = SCRIPT_DIR / "data" / "leukocytes"
OUTPUT_DIR = SCRIPT_DIR / "results" / "sam3_comprehensive_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

WBC_SUBPOPULATIONS = {
    "neutrophils": {
        "prompts": [
            "neutrophils",
            "neutrophil white blood cells",
            "neutrophils with segmented nuclei",
            "polymorphonuclear neutrophils",
        ],
        "normal_dir": DATA_DIR / "normal" / "neutrophils",
        "pathology": {
            "sepsis": "abnormal neutrophils in sepsis",
            "neutrophilia": "increased neutrophils",
        },
    },
    "lymphocytes": {
        "prompts": [
            "lymphocytes",
            "lymphocyte white blood cells",
            "lymphocytes with round nuclei",
            "small lymphocytes",
        ],
        "normal_dir": DATA_DIR / "normal" / "lymphocytes",
        "pathology": {
            "leukemia": "leukemia lymphocytes",
            "leukemia_all": "ALL acute lymphoblastic leukemia cells",
            "atypical": "atypical lymphocytes",
        },
    },
    "monocytes": {
        "prompts": [
            "monocytes",
            "monocyte white blood cells",
            "monocytes with kidney-shaped nuclei",
            "large monocytes",
        ],
        "normal_dir": DATA_DIR / "normal" / "monocytes",
        "pathology": {},
    },
    "eosinophils": {
        "prompts": [
            # Optimized prompts - specific terms don't work, descriptive ones do
            "granular white blood cells",  # Best: 96 cells, score 0.771
            "cells with orange granules",  # 98 cells, score 0.733
            "bilobed nucleus cells",  # 56 cells, score 0.696
            "orange stained cells",  # 73 cells, score 0.728
        ],
        "normal_dir": DATA_DIR / "normal" / "eosinophils",
        "pathology": {},
    },
    "basophils": {
        "prompts": [
            "basophils",
            "basophil white blood cells",
            "basophils with S-shaped nuclei",
        ],
        "normal_dir": DATA_DIR / "normal" / "basophils",
        "pathology": {},
    },
    "all_wbc": {
        "prompts": [
            "white blood cells",
            "leukocytes",
            "all white blood cells",
        ],
        "normal_dir": DATA_DIR / "normal" / "all",
        "pathology": {},
    },
}

PATHOLOGICAL_CONDITIONS = {
    "leukemia": {
        "dir": DATA_DIR / "leukemia" / "lymphocytes",
        "prompts": [
            # Optimized prompts - descriptive terms work much better than medical terms
            "round cells",  # Best: 792 cells, score 0.694, 100% success
            "cells with large nuclei",  # 273 cells, score 0.617, 100% success
            "large abnormal cells",  # 172 cells, score 0.617, 100% success
            "abnormal lymphocytes",  # 48 cells, score 0.688, 80% success
        ],
    },
    "sepsis": {
        "dir": None,  # May need to create or use existing
        "prompts": [
            "abnormal neutrophils in sepsis",
            "toxic neutrophils",
            "sepsis neutrophils",
        ],
    },
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================


def load_sam3_model(device: str = "cuda"):
    """Load SAM-3 model."""
    logger.info("🔄 Carregando modelo SAM-3...")

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    # Enable TF32
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info("   Construindo modelo...")
    model = build_sam3_image_model()

    logger.info(f"   Movendo para {device}...")
    model = model.to(device)
    model.eval()

    logger.info("   Criando processor...")
    processor = Sam3Processor(model)

    logger.info("✅ Modelo carregado!")
    return model, processor


def segment_with_prompts(
    image_path: Path, processor, prompts: List[str], device: str = "cuda"
) -> Dict:
    """
    Segment image with multiple prompts, return best result.

    Returns:
        dict with masks, scores, best_prompt, metadata
    """
    logger.info(f"📸 Processando: {image_path.name}")

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Set image
    inference_state = processor.set_image(image)

    best_result = None
    best_n_masks = 0

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        for prompt in prompts:
            try:
                processor.reset_all_prompts(inference_state)
                inference_state = processor.set_text_prompt(
                    state=inference_state, prompt=prompt
                )

                masks = inference_state.get("masks", [])
                boxes = inference_state.get("boxes", [])
                scores = inference_state.get("scores", [])

                if len(masks) > best_n_masks:
                    best_n_masks = len(masks)
                    best_result = {
                        "masks": masks,
                        "boxes": boxes,
                        "scores": scores,
                        "prompt": prompt,
                        "image_path": str(image_path),
                        "image_size": image.size,
                    }

                    logger.info(f"   ✅ '{prompt}': {len(masks)} células")

            except Exception as e:
                logger.warning(f"   ⚠️  Erro com '{prompt}': {e}")
                continue

    return best_result


def convert_tensors_to_numpy(data):
    """Convert PyTorch tensors to numpy arrays."""
    if isinstance(data, list):
        return [convert_tensors_to_numpy(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_tensors_to_numpy(v) for k, v in data.items()}
    elif torch.is_tensor(data):
        return data.cpu().float().numpy()
    else:
        return data


# ============================================================================
# TEST SUITES
# ============================================================================


def test_subpopulation(
    subpop_name: str, config: Dict, processor, device: str = "cuda", n_images: int = 5
) -> List[Dict]:
    """
    Test segmentation for a specific WBC subpopulation.

    Args:
        subpop_name: Name of subpopulation (e.g., "neutrophils")
        config: Configuration dict with prompts and directories
        processor: SAM-3 processor
        device: Device to run on
        n_images: Number of images to test

    Returns:
        List of test results
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"🧪 TESTE: {subpop_name.upper()}")
    logger.info("=" * 80)

    results = []

    # Find test images
    test_dir = config.get("normal_dir")
    if test_dir and test_dir.exists():
        images = (
            list(test_dir.glob("*.jpg"))
            + list(test_dir.glob("*.png"))
            + list(test_dir.glob("*.jpeg"))
        )
        images = images[:n_images]

        logger.info(f"📁 Encontradas {len(images)} imagens para teste")
        logger.info(f"📝 Prompts: {config['prompts']}")
        logger.info("")

        for i, img_path in enumerate(images, 1):
            logger.info(f"[{i}/{len(images)}] {img_path.name}")

            result = segment_with_prompts(
                image_path=img_path,
                processor=processor,
                prompts=config["prompts"],
                device=device,
            )

            if result:
                # Convert tensors to numpy for JSON serialization
                result_clean = convert_tensors_to_numpy(result)

                # Calculate statistics
                scores = result.get("scores", [])
                if len(scores) > 0:
                    scores_array = [
                        float(s.cpu()) if torch.is_tensor(s) else float(s)
                        for s in scores
                    ]
                    result_clean["stats"] = {
                        "n_cells": len(result["masks"]),
                        "score_mean": float(np.mean(scores_array)),
                        "score_std": float(np.std(scores_array)),
                        "score_min": float(np.min(scores_array)),
                        "score_max": float(np.max(scores_array)),
                    }
                else:
                    result_clean["stats"] = {
                        "n_cells": len(result["masks"]),
                        "score_mean": 0.0,
                        "score_std": 0.0,
                        "score_min": 0.0,
                        "score_max": 0.0,
                    }

                results.append(result_clean)
                logger.info(
                    f"   ✅ {result_clean['stats']['n_cells']} células, "
                    f"score médio: {result_clean['stats']['score_mean']:.3f}"
                )
            else:
                logger.warning(f"   ❌ Nenhuma célula detectada")

    else:
        logger.warning(f"⚠️  Diretório não encontrado: {test_dir}")

    return results


def test_pathological_condition(
    condition_name: str,
    config: Dict,
    processor,
    device: str = "cuda",
    n_images: int = 5,
) -> List[Dict]:
    """Test segmentation for pathological condition."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"🔬 TESTE PATOLÓGICO: {condition_name.upper()}")
    logger.info("=" * 80)

    results = []

    test_dir = config.get("dir")
    if test_dir and test_dir.exists():
        images = (
            list(test_dir.glob("*.jpg"))
            + list(test_dir.glob("*.png"))
            + list(test_dir.glob("*.jpeg"))
        )
        images = images[:n_images]

        logger.info(f"📁 Encontradas {len(images)} imagens")
        logger.info(f"📝 Prompts: {config['prompts']}")
        logger.info("")

        for i, img_path in enumerate(images, 1):
            logger.info(f"[{i}/{len(images)}] {img_path.name}")

            result = segment_with_prompts(
                image_path=img_path,
                processor=processor,
                prompts=config["prompts"],
                device=device,
            )

            if result:
                result_clean = convert_tensors_to_numpy(result)

                scores = result.get("scores", [])
                if len(scores) > 0:
                    scores_array = [
                        float(s.cpu()) if torch.is_tensor(s) else float(s)
                        for s in scores
                    ]
                    result_clean["stats"] = {
                        "n_cells": len(result["masks"]),
                        "score_mean": float(np.mean(scores_array)),
                        "score_std": float(np.std(scores_array)),
                        "score_min": float(np.min(scores_array)),
                        "score_max": float(np.max(scores_array)),
                    }
                else:
                    result_clean["stats"] = {
                        "n_cells": len(result["masks"]),
                        "score_mean": 0.0,
                        "score_std": 0.0,
                        "score_min": 0.0,
                        "score_max": 0.0,
                    }

                results.append(result_clean)
                logger.info(f"   ✅ {result_clean['stats']['n_cells']} células")
    else:
        logger.warning(f"⚠️  Diretório não encontrado: {test_dir}")

    return results


def run_comprehensive_test_suite(
    device: str = "cuda", images_per_test: int = 5
) -> Dict:
    """
    Run comprehensive test suite for all WBC types.

    Returns:
        Dict with all test results
    """
    logger.info("=" * 80)
    logger.info("🧪 SUÍTE COMPLETA DE TESTES - SAM-3 LEUCOCITOS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Imagens por teste: {images_per_test}")
    logger.info(f"Dispositivo: {device}")
    logger.info("")

    # Load model once
    model, processor = load_sam3_model(device=device)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "subpopulations": {},
        "pathological": {},
        "summary": {},
    }

    # Test each subpopulation
    for subpop_name, config in WBC_SUBPOPULATIONS.items():
        results = test_subpopulation(
            subpop_name=subpop_name,
            config=config,
            processor=processor,
            device=device,
            n_images=images_per_test,
        )
        all_results["subpopulations"][subpop_name] = results

    # Test pathological conditions
    for condition_name, config in PATHOLOGICAL_CONDITIONS.items():
        results = test_pathological_condition(
            condition_name=condition_name,
            config=config,
            processor=processor,
            device=device,
            n_images=images_per_test,
        )
        all_results["pathological"][condition_name] = results

    # Calculate summary statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 CALCULANDO ESTATÍSTICAS")
    logger.info("=" * 80)

    summary = calculate_summary_statistics(all_results)
    all_results["summary"] = summary

    # Print summary
    print_summary(summary)

    # Save results
    results_file = (
        OUTPUT_DIR / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("")
    logger.info(f"💾 Resultados salvos: {results_file.name}")

    return all_results


def calculate_summary_statistics(all_results: Dict) -> Dict:
    """Calculate summary statistics from all test results."""
    summary = {
        "total_tests": 0,
        "total_cells_detected": 0,
        "subpopulations": {},
        "pathological": {},
    }

    # Subpopulations
    for subpop_name, results in all_results["subpopulations"].items():
        if results:
            n_tests = len(results)
            total_cells = sum(r.get("stats", {}).get("n_cells", 0) for r in results)
            avg_score = np.mean(
                [
                    r.get("stats", {}).get("score_mean", 0)
                    for r in results
                    if r.get("stats")
                ]
            )

            summary["subpopulations"][subpop_name] = {
                "n_tests": n_tests,
                "total_cells": int(total_cells),
                "avg_cells_per_image": float(total_cells / n_tests)
                if n_tests > 0
                else 0,
                "avg_score": float(avg_score) if not np.isnan(avg_score) else 0,
            }
            summary["total_tests"] += n_tests
            summary["total_cells_detected"] += total_cells

    # Pathological
    for condition_name, results in all_results["pathological"].items():
        if results:
            n_tests = len(results)
            total_cells = sum(r.get("stats", {}).get("n_cells", 0) for r in results)
            avg_score = np.mean(
                [
                    r.get("stats", {}).get("score_mean", 0)
                    for r in results
                    if r.get("stats")
                ]
            )

            summary["pathological"][condition_name] = {
                "n_tests": n_tests,
                "total_cells": int(total_cells),
                "avg_cells_per_image": float(total_cells / n_tests)
                if n_tests > 0
                else 0,
                "avg_score": float(avg_score) if not np.isnan(avg_score) else 0,
            }
            summary["total_tests"] += n_tests
            summary["total_cells_detected"] += total_cells

    return summary


def print_summary(summary: Dict):
    """Print summary statistics."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 RESUMO GERAL")
    logger.info("=" * 80)
    logger.info(f"Total de testes: {summary['total_tests']}")
    logger.info(f"Total de células detectadas: {summary['total_cells_detected']}")
    logger.info("")

    logger.info("SUBPOPULAÇÕES NORMAIS:")
    logger.info("-" * 80)
    for subpop, stats in summary["subpopulations"].items():
        logger.info(f"  {subpop.upper()}:")
        logger.info(f"    Testes: {stats['n_tests']}")
        logger.info(f"    Total células: {stats['total_cells']}")
        logger.info(f"    Média por imagem: {stats['avg_cells_per_image']:.1f}")
        logger.info(f"    Score médio: {stats['avg_score']:.3f}")

    logger.info("")
    logger.info("CONDIÇÕES PATOLÓGICAS:")
    logger.info("-" * 80)
    for condition, stats in summary["pathological"].items():
        logger.info(f"  {condition.upper()}:")
        logger.info(f"    Testes: {stats['n_tests']}")
        logger.info(f"    Total células: {stats['total_cells']}")
        logger.info(f"    Média por imagem: {stats['avg_cells_per_image']:.1f}")
        logger.info(f"    Score médio: {stats['avg_score']:.3f}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive SAM-3 test suite for all WBC types"
    )
    parser.add_argument(
        "--n-images", type=int, default=5, help="Number of images per test (default: 5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )

    args = parser.parse_args()

    # Run comprehensive tests
    results = run_comprehensive_test_suite(
        device=args.device, images_per_test=args.n_images
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ TESTES COMPLETOS FINALIZADOS!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
