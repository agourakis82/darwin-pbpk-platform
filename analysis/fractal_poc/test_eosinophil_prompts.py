#!/usr/bin/env python3
"""
Test Alternative Prompts for Eosinophil Detection with SAM-3
============================================================

Tests various text prompts to find the best approach for detecting eosinophils.

Created: 2025-12-04
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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

DATA_DIR = SCRIPT_DIR / "data" / "leukocytes" / "normal" / "eosinophils"
OUTPUT_DIR = SCRIPT_DIR / "results" / "eosinophil_prompt_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Alternative prompts to test for eosinophils
EOSINOPHIL_PROMPTS = [
    # Original prompts (failed)
    "eosinophils",
    "eosinophil white blood cells",
    "eosinophils with bilobed nuclei",
    "eosinophils with orange granules",
    # Color-based prompts
    "cells with orange granules",
    "cells with red granules",
    "cells with pink granules",
    "orange stained cells",
    "red-orange granular cells",
    # Morphology-based prompts
    "bilobed nucleus cells",
    "cells with two-lobed nucleus",
    "granulocytes",
    "granular white blood cells",
    # Generic cell prompts
    "blood cells",
    "white blood cells",
    "leukocytes",
    "round cells",
    "stained cells",
    # Descriptive prompts
    "cells with granules",
    "large granular cells",
    "cells with cytoplasmic granules",
    # Medical terminology
    "acidophilic granulocytes",
    "polymorphonuclear cells",
]


def load_sam3_model(device: str = "cuda"):
    """Load SAM-3 model."""
    logger.info("Loading SAM-3 model...")

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_sam3_image_model()
    model = model.to(device)
    model.eval()

    processor = Sam3Processor(model)
    logger.info("Model loaded!")
    return model, processor


def test_prompt(image_path: Path, processor, prompt: str, device: str = "cuda") -> Dict:
    """Test a single prompt on an image."""
    image = Image.open(image_path).convert("RGB")
    inference_state = processor.set_image(image)

    result = {
        "prompt": prompt,
        "n_cells": 0,
        "scores": [],
        "score_mean": 0.0,
        "success": False,
    }

    try:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state, prompt=prompt
            )

            masks = inference_state.get("masks", [])
            scores = inference_state.get("scores", [])

            if len(masks) > 0:
                scores_list = [
                    float(s.cpu()) if torch.is_tensor(s) else float(s) for s in scores
                ]
                result["n_cells"] = len(masks)
                result["scores"] = scores_list
                result["score_mean"] = float(np.mean(scores_list))
                result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def run_prompt_tests(n_images: int = 5, device: str = "cuda"):
    """Run tests for all prompts on multiple images."""
    logger.info("=" * 80)
    logger.info("EOSINOPHIL PROMPT TESTING")
    logger.info("=" * 80)

    # Get test images
    images = (
        list(DATA_DIR.glob("*.jpeg"))
        + list(DATA_DIR.glob("*.jpg"))
        + list(DATA_DIR.glob("*.png"))
    )
    images = images[:n_images]

    logger.info(f"Testing {len(EOSINOPHIL_PROMPTS)} prompts on {len(images)} images")
    logger.info("")

    # Load model
    model, processor = load_sam3_model(device)

    # Results structure
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "n_images": len(images),
        "n_prompts": len(EOSINOPHIL_PROMPTS),
        "prompts": {},
        "summary": {},
    }

    # Test each prompt
    for prompt in EOSINOPHIL_PROMPTS:
        logger.info(f"Testing prompt: '{prompt}'")

        prompt_results = {
            "total_cells": 0,
            "successful_images": 0,
            "scores": [],
            "per_image": [],
        }

        for img_path in images:
            result = test_prompt(img_path, processor, prompt, device)
            prompt_results["per_image"].append({"image": img_path.name, **result})

            if result["success"]:
                prompt_results["total_cells"] += result["n_cells"]
                prompt_results["successful_images"] += 1
                prompt_results["scores"].extend(result["scores"])

        # Calculate averages
        if prompt_results["scores"]:
            prompt_results["avg_score"] = float(np.mean(prompt_results["scores"]))
            prompt_results["avg_cells_per_image"] = prompt_results["total_cells"] / len(
                images
            )
        else:
            prompt_results["avg_score"] = 0.0
            prompt_results["avg_cells_per_image"] = 0.0

        all_results["prompts"][prompt] = prompt_results

        # Log summary for this prompt
        if prompt_results["total_cells"] > 0:
            logger.info(
                f"  ✅ {prompt_results['total_cells']} cells, avg score: {prompt_results['avg_score']:.3f}"
            )
        else:
            logger.info(f"  ❌ No cells detected")

    # Generate summary - rank prompts by effectiveness
    logger.info("")
    logger.info("=" * 80)
    logger.info("PROMPT RANKING (by total cells detected)")
    logger.info("=" * 80)

    ranked_prompts = sorted(
        all_results["prompts"].items(),
        key=lambda x: (x[1]["total_cells"], x[1]["avg_score"]),
        reverse=True,
    )

    all_results["summary"]["ranked_prompts"] = []

    for i, (prompt, results) in enumerate(ranked_prompts[:10], 1):
        logger.info(f"{i}. '{prompt}'")
        logger.info(
            f"   Cells: {results['total_cells']}, Avg Score: {results['avg_score']:.3f}, Success Rate: {results['successful_images']}/{len(images)}"
        )

        all_results["summary"]["ranked_prompts"].append(
            {
                "rank": i,
                "prompt": prompt,
                "total_cells": results["total_cells"],
                "avg_score": results["avg_score"],
                "success_rate": results["successful_images"] / len(images),
            }
        )

    # Best prompt
    if ranked_prompts and ranked_prompts[0][1]["total_cells"] > 0:
        best_prompt = ranked_prompts[0][0]
        best_results = ranked_prompts[0][1]
        all_results["summary"]["best_prompt"] = {
            "prompt": best_prompt,
            "total_cells": best_results["total_cells"],
            "avg_score": best_results["avg_score"],
        }
        logger.info("")
        logger.info(f"🏆 BEST PROMPT: '{best_prompt}'")
        logger.info(f"   Total cells: {best_results['total_cells']}")
        logger.info(f"   Average score: {best_results['avg_score']:.3f}")
    else:
        all_results["summary"]["best_prompt"] = None
        logger.info("")
        logger.info("❌ No effective prompt found for eosinophils")

    # Save results
    results_file = (
        OUTPUT_DIR
        / f"eosinophil_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("")
    logger.info(f"Results saved: {results_file.name}")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test eosinophil detection prompts")
    parser.add_argument(
        "--n-images", type=int, default=5, help="Number of images to test"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    run_prompt_tests(n_images=args.n_images, device=args.device)


if __name__ == "__main__":
    main()
