#!/usr/bin/env python3
"""
Export SAM-3 Masks for Julia Fractal Analysis
==============================================

Segments leukocytes with SAM-3 and exports masks in formats
compatible with Julia's LeukocyteFractalAnalysis module.

Output formats:
- NPZ: NumPy compressed arrays (masks, scores, metadata)
- PNG: Individual mask images
- JSON: Metadata and cell properties

Created: 2025-12-04
"""

import json
import logging
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

SCRIPT_DIR = Path(__file__).parent
SAM3_DIR = SCRIPT_DIR / "sam3"
sys.path.insert(0, str(SAM3_DIR))

# Optimized prompts from testing
OPTIMIZED_PROMPTS = {
    "neutrophils": ["neutrophils", "neutrophil white blood cells"],
    "lymphocytes": ["lymphocytes", "small lymphocytes"],
    "monocytes": ["monocytes", "monocyte white blood cells"],
    "eosinophils": ["granular white blood cells", "cells with orange granules"],
    "basophils": ["cells with dark granules", "granular cells"],
    "all_wbc": ["white blood cells", "round cells"],
    "leukemia": ["round cells", "cells with large nuclei", "abnormal lymphocytes"],
}


class SAM3MaskExporter:
    """Export SAM-3 segmentation masks for Julia analysis."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        """Load SAM-3 model."""
        if self.model is not None:
            return

        logger.info("Loading SAM-3 model...")
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model = build_sam3_image_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = Sam3Processor(self.model)
        logger.info("Model loaded!")

    def segment_image(self, image_path: Path, prompts: List[str]) -> Dict:
        """
        Segment image with multiple prompts, return best result.

        Returns dict with masks, scores, boxes, and metadata.
        """
        self.load_model()

        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        inference_state = self.processor.set_image(image)

        best_result = None
        best_n_masks = 0

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            for prompt in prompts:
                try:
                    self.processor.reset_all_prompts(inference_state)
                    inference_state = self.processor.set_text_prompt(
                        state=inference_state, prompt=prompt
                    )

                    masks = inference_state.get("masks", [])
                    boxes = inference_state.get("boxes", [])
                    scores = inference_state.get("scores", [])

                    if len(masks) > best_n_masks:
                        best_n_masks = len(masks)

                        # Convert tensors to numpy
                        masks_np = []
                        for m in masks:
                            if torch.is_tensor(m):
                                mask_np = m.cpu().numpy().squeeze()
                                # Ensure binary
                                mask_np = (mask_np > 0.5).astype(np.uint8)
                            else:
                                mask_np = np.array(m).squeeze().astype(np.uint8)
                            masks_np.append(mask_np)

                        boxes_np = (
                            boxes.cpu().numpy()
                            if torch.is_tensor(boxes)
                            else np.array(boxes)
                        )
                        scores_np = np.array(
                            [
                                float(s.cpu()) if torch.is_tensor(s) else float(s)
                                for s in scores
                            ]
                        )

                        best_result = {
                            "masks": masks_np,
                            "boxes": boxes_np,
                            "scores": scores_np,
                            "prompt": prompt,
                            "n_cells": len(masks_np),
                            "image_shape": image_array.shape[:2],
                        }

                except Exception as e:
                    logger.warning(f"Error with prompt '{prompt}': {e}")
                    continue

        if best_result is None:
            best_result = {
                "masks": [],
                "boxes": np.array([]),
                "scores": np.array([]),
                "prompt": None,
                "n_cells": 0,
                "image_shape": image_array.shape[:2],
            }

        return best_result

    def export_masks_npz(
        self, image_path: Path, output_path: Path, cell_type: str = "all_wbc"
    ) -> Dict:
        """
        Export masks to NPZ format for Julia.

        NPZ contains:
        - masks: (N, H, W) binary mask array
        - scores: (N,) confidence scores
        - boxes: (N, 4) bounding boxes [x1, y1, x2, y2]
        - combined_mask: (H, W) all cells combined
        - metadata: JSON string with additional info
        """
        prompts = OPTIMIZED_PROMPTS.get(cell_type, OPTIMIZED_PROMPTS["all_wbc"])

        logger.info(f"Segmenting: {image_path.name}")
        result = self.segment_image(image_path, prompts)

        if result["n_cells"] == 0:
            logger.warning(f"No cells detected in {image_path.name}")
            return result

        # Stack masks into single array
        masks_array = np.stack(result["masks"], axis=0)

        # Create combined mask (all cells)
        combined_mask = np.any(masks_array, axis=0).astype(np.uint8)

        # Create edge mask for fractal analysis
        from scipy import ndimage

        edge_mask = np.zeros_like(combined_mask)
        for mask in result["masks"]:
            # Sobel edge detection on each cell
            edges = ndimage.sobel(mask.astype(float))
            edge_mask = np.logical_or(edge_mask, np.abs(edges) > 0.1)
        edge_mask = edge_mask.astype(np.uint8)

        # Calculate cell properties
        cell_properties = []
        for i, mask in enumerate(result["masks"]):
            coords = np.where(mask > 0)
            if len(coords[0]) > 0:
                centroid_y = float(np.mean(coords[0]))
                centroid_x = float(np.mean(coords[1]))
                area = int(np.sum(mask))

                cell_properties.append(
                    {
                        "id": i,
                        "area": area,
                        "centroid_x": centroid_x,
                        "centroid_y": centroid_y,
                        "score": float(result["scores"][i])
                        if i < len(result["scores"])
                        else 0.0,
                        "bbox": result["boxes"][i].tolist()
                        if i < len(result["boxes"])
                        else [],
                    }
                )

        # Metadata
        metadata = {
            "source_image": str(image_path),
            "cell_type": cell_type,
            "prompt_used": result["prompt"],
            "n_cells": result["n_cells"],
            "image_shape": list(result["image_shape"]),
            "timestamp": datetime.now().isoformat(),
            "cell_properties": cell_properties,
        }

        # Save NPZ (without metadata - Julia NPZ can't read Unicode strings)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            masks=masks_array,
            scores=result["scores"],
            boxes=result["boxes"],
            combined_mask=combined_mask,
            edge_mask=edge_mask,
            # Note: metadata saved separately as JSON for Julia compatibility
        )

        # Save metadata as separate JSON file (Julia-compatible)
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Exported {result['n_cells']} cells to {output_path.name}")

        return {
            **result,
            "output_path": str(output_path),
            "cell_properties": cell_properties,
        }

    def export_batch(
        self,
        image_dir: Path,
        output_dir: Path,
        cell_type: str = "all_wbc",
        max_images: int = None,
    ) -> List[Dict]:
        """Export masks for multiple images."""

        image_extensions = ["*.jpg", "*.jpeg", "*.png"]
        images = []
        for ext in image_extensions:
            images.extend(image_dir.glob(ext))

        if max_images:
            images = images[:max_images]

        logger.info(f"Processing {len(images)} images from {image_dir}")

        results = []
        for i, img_path in enumerate(images, 1):
            logger.info(f"[{i}/{len(images)}] {img_path.name}")

            output_path = output_dir / f"{img_path.stem}_masks.npz"
            result = self.export_masks_npz(img_path, output_path, cell_type)
            results.append(result)

        # Save batch summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "source_dir": str(image_dir),
            "cell_type": cell_type,
            "n_images": len(images),
            "total_cells": sum(r.get("n_cells", 0) for r in results),
            "files": [r.get("output_path") for r in results if r.get("output_path")],
        }

        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Batch complete: {summary['total_cells']} cells from {len(images)} images"
        )

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export SAM-3 masks for Julia")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image-dir", type=str, help="Directory of images")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--cell-type",
        type=str,
        default="all_wbc",
        choices=list(OPTIMIZED_PROMPTS.keys()),
        help="Cell type to segment",
    )
    parser.add_argument(
        "--max-images", type=int, default=None, help="Max images to process"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    exporter = SAM3MaskExporter(device=args.device)
    output_dir = Path(args.output_dir)

    if args.image:
        # Single image
        image_path = Path(args.image)
        output_path = output_dir / f"{image_path.stem}_masks.npz"
        exporter.export_masks_npz(image_path, output_path, args.cell_type)

    elif args.image_dir:
        # Batch processing
        image_dir = Path(args.image_dir)
        exporter.export_batch(image_dir, output_dir, args.cell_type, args.max_images)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
