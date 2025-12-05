#!/usr/bin/env python3
"""
Segment Leukocytes with SAM-3
==============================

Segment white blood cells from blood smear images using SAM-3 text prompts.

Created: 2025-12-01
Author: Darwin PBPK Platform
"""

import os
import sys
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add sam3 to path
SCRIPT_DIR = Path(__file__).parent
SAM3_DIR = SCRIPT_DIR / "sam3"
sys.path.insert(0, str(SAM3_DIR))

DATA_DIR = SCRIPT_DIR / "data" / "leukocytes"
OUTPUT_DIR = SCRIPT_DIR / "results" / "sam3_segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sam3_model(device: str = "cuda"):
    """Load SAM-3 model from HuggingFace or local checkpoint."""
    logger.info("🔄 Carregando modelo SAM-3...")
    
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        # Enable TF32 for Ampere GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        logger.info("   Construindo modelo (baixará do HuggingFace se necessário)...")
        # build_sam3_image_model() will handle downloading from HuggingFace
        model = build_sam3_image_model()
        
        logger.info(f"   Movendo modelo para {device}...")
        model = model.to(device)
        model.eval()
        
        logger.info("   Criando processor...")
        processor = Sam3Processor(model)
        
        logger.info("✅ Modelo SAM-3 carregado com sucesso!")
        return model, processor
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        logger.error("   Verifique se você tem acesso aos checkpoints no HuggingFace")
        logger.error("   Acesse: https://huggingface.co/facebook/sam3")
        logger.error("   Execute: hf auth login (se necessário)")
        raise


def segment_leukocytes(
    image_path: Path,
    processor,
    prompts: List[str] = ["white blood cells"],
    device: str = "cuda"
) -> Tuple[List[np.ndarray], List[float], dict]:
    """
    Segment leukocytes from blood smear image using SAM-3.
    
    Args:
        image_path: Path to blood smear image
        processor: SAM-3 processor instance
        prompts: List of text prompts to try
        device: Device to run inference on
    
    Returns:
        masks: List of binary masks (one per detected cell)
        scores: Confidence scores for each mask
        metadata: Additional information
    """
    logger.info(f"📸 Processando imagem: {image_path.name}")
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
        logger.info(f"   Dimensões: {image.size}")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar imagem: {e}")
        return [], [], {}
    
    # Try each prompt until we get results
    all_masks = []
    all_scores = []
    successful_prompt = None
    
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # Set image in processor
        inference_state = processor.set_image(image)
        
        for prompt in prompts:
            logger.info(f"   Tentando prompt: '{prompt}'...")
            
            try:
                # Reset prompts
                processor.reset_all_prompts(inference_state)
                
                # Set text prompt (returns updated inference_state)
                inference_state = processor.set_text_prompt(
                    state=inference_state,
                    prompt=prompt
                )
                
                # Get masks, boxes, and scores from inference_state
                masks = inference_state.get("masks", [])
                boxes = inference_state.get("boxes", [])
                scores = inference_state.get("scores", [])
                
                if len(masks) > 0:
                    logger.info(f"   ✅ Encontradas {len(masks)} células com prompt '{prompt}'")
                    all_masks.extend(masks)
                    all_scores.extend(scores)
                    if successful_prompt is None:
                        successful_prompt = prompt
                else:
                    logger.info(f"   ⚠️  Nenhuma célula encontrada com prompt '{prompt}'")
                    
            except Exception as e:
                logger.warning(f"   ⚠️  Erro com prompt '{prompt}': {e}")
                continue
    
    metadata = {
        "image_path": str(image_path),
        "image_size": image.size,
        "successful_prompt": successful_prompt,
        "n_masks": len(all_masks),
        "prompts_tried": prompts
    }
    
    return all_masks, all_scores, metadata


def visualize_segmentation(
    image_path: Path,
    masks: List[np.ndarray],
    scores: List[float],
    output_path: Path,
    max_masks: int = 50
):
    """Visualize segmentation results."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        image = Image.open(image_path).convert('RGB')
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f'SAM-3 Segmentation: {image_path.name}\n{len(masks)} cells detected')
        
        # Overlay masks (limited to max_masks for performance)
        for i, (mask, score) in enumerate(zip(masks[:max_masks], scores[:max_masks])):
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Handle different mask shapes
            if len(mask.shape) == 3:
                mask = mask.squeeze()
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            # Create overlay
            mask_resized = np.array(Image.fromarray(mask).resize(image.size))
            
            # Apply colored overlay
            color = np.random.rand(3)
            overlay = np.zeros_like(np.array(image), dtype=float)
            mask_bool = mask_resized > 128
            overlay[mask_bool] = color
            
            ax.imshow(overlay, alpha=0.3)
            
            # Add bounding box
            if mask_bool.any():
                y_coords, x_coords = np.where(mask_bool)
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=1, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   💾 Visualização salva: {output_path.name}")
        
    except Exception as e:
        logger.warning(f"   ⚠️  Erro ao criar visualização: {e}")


def test_leukocyte_segmentation(
    image_path: Optional[Path] = None,
    prompts: Optional[List[str]] = None,
    visualize: bool = True
):
    """Test leukocyte segmentation on a single image."""
    
    if prompts is None:
        prompts = [
            "white blood cells",
            "leukocytes",
            "white blood cell",
            "neutrophils",
            "lymphocytes"
        ]
    
    # Find test image if not provided
    if image_path is None:
        logger.info("🔍 Procurando imagem de teste...")
        
        # Try to find an image from our organized datasets
        possible_dirs = [
            DATA_DIR / "normal" / "all",
            DATA_DIR / "normal" / "lymphocytes",
            DATA_DIR / "normal" / "neutrophils",
        ]
        
        for test_dir in possible_dirs:
            if test_dir.exists():
                images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
                if images:
                    image_path = images[0]
                    logger.info(f"   ✅ Usando imagem: {image_path.name}")
                    break
        
        if image_path is None:
            logger.error("❌ Nenhuma imagem de teste encontrada!")
            return
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️  Usando dispositivo: {device}")
    
    try:
        model, processor = load_sam3_model(device=device)
    except Exception as e:
        logger.error(f"❌ Falha ao carregar modelo: {e}")
        logger.error("   Verifique se os checkpoints estão disponíveis")
        return
    
    # Segment
    masks, scores, metadata = segment_leukocytes(
        image_path=image_path,
        processor=processor,
        prompts=prompts,
        device=device
    )
    
    # Results
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 RESULTADOS")
    logger.info("=" * 80)
    logger.info(f"Imagem: {image_path.name}")
    logger.info(f"Células detectadas: {len(masks)}")
    logger.info(f"Prompt bem-sucedido: {metadata.get('successful_prompt', 'N/A')}")
    if scores:
        # Convert tensors to numpy if needed
        if isinstance(scores, list):
            scores_array = [float(s.cpu()) if torch.is_tensor(s) else float(s) for s in scores]
            scores_array = np.array(scores_array)
        else:
            if torch.is_tensor(scores):
                scores_array = scores.cpu().float().numpy()
            else:
                scores_array = np.array(scores, dtype=float)
        
        logger.info(f"Score médio: {np.mean(scores_array):.3f}")
        logger.info(f"Score mínimo: {np.min(scores_array):.3f}")
        logger.info(f"Score máximo: {np.max(scores_array):.3f}")
    
    # Visualize if requested
    if visualize and len(masks) > 0:
        output_path = OUTPUT_DIR / f"sam3_{image_path.stem}.png"
        visualize_segmentation(image_path, masks, scores, output_path)
    
    return masks, scores, metadata


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Segment leukocytes with SAM-3")
    parser.add_argument("--image", type=str, help="Path to blood smear image")
    parser.add_argument("--prompt", type=str, default="white blood cells",
                       help="Text prompt for segmentation")
    parser.add_argument("--no-visualize", action="store_true",
                       help="Don't create visualization")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🩸 SEGMENTAÇÃO DE LEUCÓCITOS COM SAM-3")
    logger.info("=" * 80)
    logger.info("")
    
    image_path = Path(args.image) if args.image else None
    prompts = [args.prompt] if args.prompt else None
    
    test_leukocyte_segmentation(
        image_path=image_path,
        prompts=prompts,
        visualize=not args.no_visualize
    )


if __name__ == "__main__":
    main()

