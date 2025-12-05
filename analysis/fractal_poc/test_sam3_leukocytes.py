#!/usr/bin/env python3
"""
Test SAM-3 for Leukocyte Segmentation
=====================================

Test script to evaluate SAM-3 capabilities for segmenting white blood cells
using text prompts.

Created: 2025-12-01
Author: Darwin PBPK Platform
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add sam3 to path
SCRIPT_DIR = Path(__file__).parent
SAM3_DIR = SCRIPT_DIR / "sam3"
sys.path.insert(0, str(SAM3_DIR))

DATA_DIR = SCRIPT_DIR / "data" / "leukocytes"

def check_sam3_installation():
    """Check if SAM-3 is properly installed."""
    logger.info("🔍 Verificando instalação do SAM-3...")
    
    try:
        import sam3
        logger.info("✅ SAM-3 importado com sucesso")
        logger.info(f"   Localização: {sam3.__file__}")
        return True
    except ImportError as e:
        logger.error(f"❌ Erro ao importar SAM-3: {e}")
        return False


def check_model_files():
    """Check if model files/checkpoints are available."""
    logger.info("🔍 Verificando arquivos do modelo...")
    
    sam3_root = Path(SAM3_DIR)
    
    # Check for BPE vocabulary
    bpe_path = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if bpe_path.exists():
        logger.info(f"✅ BPE vocabulary encontrado: {bpe_path}")
    else:
        logger.warning(f"⚠️  BPE vocabulary não encontrado: {bpe_path}")
    
    # Check for model checkpoints directory
    checkpoints_dirs = [
        sam3_root / "checkpoints",
        sam3_root / "models",
        Path.home() / ".cache" / "sam3",
    ]
    
    for ckpt_dir in checkpoints_dirs:
        if ckpt_dir.exists():
            logger.info(f"✅ Diretório de checkpoints encontrado: {ckpt_dir}")
            files = list(ckpt_dir.rglob("*.pt")) + list(ckpt_dir.rglob("*.pth"))
            if files:
                logger.info(f"   Encontrados {len(files)} arquivos de modelo")
    
    return True


def check_test_images():
    """Check if we have test leukocyte images available."""
    logger.info("🔍 Verificando imagens de teste...")
    
    if not DATA_DIR.exists():
        logger.warning(f"⚠️  Diretório de dados não encontrado: {DATA_DIR}")
        return False
    
    # Look for images
    image_files = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png"))
    
    if image_files:
        logger.info(f"✅ Encontradas {len(image_files)} imagens de leucócitos")
        logger.info(f"   Exemplos: {[f.name for f in image_files[:3]]}")
        return True, image_files[:5]  # Return first 5 for testing
    else:
        logger.warning("⚠️  Nenhuma imagem encontrada")
        return False, []


def test_sam3_imports():
    """Test importing SAM-3 modules."""
    logger.info("🔍 Testando imports do SAM-3...")
    
    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.visualization_utils import plot_results
        logger.info("✅ Módulos principais importados com sucesso")
        return True
    except ImportError as e:
        logger.error(f"❌ Erro ao importar módulos: {e}")
        return False


def test_model_loading():
    """Test loading SAM-3 model (if checkpoints available)."""
    logger.info("🔍 Testando carregamento do modelo...")
    
    try:
        import torch
        from sam3 import build_sam3_image_model
        
        sam3_root = Path(SAM3_DIR)
        bpe_path = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        
        if not bpe_path.exists():
            logger.warning("⚠️  BPE vocabulary não encontrado. Pulando teste de carregamento.")
            return False
        
        logger.info("   Tentando construir modelo...")
        # This might fail if checkpoints are not available
        # model = build_sam3_image_model(bpe_path=str(bpe_path))
        logger.info("   (Modelo requer checkpoints - verificar acesso)")
        return False  # Return False as checkpoints need access
    except Exception as e:
        logger.warning(f"⚠️  Não foi possível carregar modelo (checkpoints podem não estar disponíveis): {e}")
        return False


def check_huggingface_access():
    """Check if we can access HuggingFace (where checkpoints might be)."""
    logger.info("🔍 Verificando acesso ao HuggingFace...")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        logger.info("✅ HuggingFace Hub disponível")
        logger.info("   Nota: Checkpoints podem estar no HuggingFace")
        logger.info("   Verificar: https://huggingface.co/facebook/sam3-*")
        return True
    except ImportError:
        logger.warning("⚠️  huggingface_hub não instalado")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Erro ao acessar HuggingFace: {e}")
        return False


def create_segmentation_function_template():
    """Create a template function for leukocyte segmentation."""
    template = '''
def segment_leukocytes_with_sam3(image_path: str, prompt: str = "white blood cells"):
    """
    Segment leukocytes from blood smear image using SAM-3.
    
    Args:
        image_path: Path to blood smear image
        prompt: Text prompt describing cells to segment
    
    Returns:
        masks: List of binary masks (one per cell)
        scores: Confidence scores for each mask
    """
    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Build model (requires checkpoints)
    sam3_root = Path(__file__).parent / "sam3"
    bpe_path = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=str(bpe_path))
    
    # Create processor
    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)
    
    # Set text prompt
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
    
    # Run inference
    masks, scores = processor.get_masks(inference_state)
    
    return masks, scores
'''
    
    logger.info("📝 Template de função criado (ver código acima)")
    return template


def main():
    """Main test function."""
    logger.info("=" * 80)
    logger.info("🧪 TESTE SAM-3 PARA SEGMENTAÇÃO DE LEUCÓCITOS")
    logger.info("=" * 80)
    logger.info("")
    
    results = {}
    
    # 1. Check installation
    logger.info("1️⃣  VERIFICAÇÃO DE INSTALAÇÃO")
    logger.info("-" * 80)
    results['installation'] = check_sam3_installation()
    logger.info("")
    
    # 2. Check model files
    logger.info("2️⃣  VERIFICAÇÃO DE ARQUIVOS DO MODELO")
    logger.info("-" * 80)
    results['model_files'] = check_model_files()
    logger.info("")
    
    # 3. Check imports
    logger.info("3️⃣  VERIFICAÇÃO DE IMPORTS")
    logger.info("-" * 80)
    results['imports'] = test_sam3_imports()
    logger.info("")
    
    # 4. Check test images
    logger.info("4️⃣  VERIFICAÇÃO DE IMAGENS DE TESTE")
    logger.info("-" * 80)
    has_images, test_images = check_test_images()
    results['test_images'] = has_images
    logger.info("")
    
    # 5. Check HuggingFace access
    logger.info("5️⃣  VERIFICAÇÃO DE ACESSO HUGGINGFACE")
    logger.info("-" * 80)
    results['huggingface'] = check_huggingface_access()
    logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("📊 RESUMO")
    logger.info("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        logger.info(f"{status} {test_name}: {'PASSOU' if passed else 'FALHOU'}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("📝 PRÓXIMOS PASSOS")
    logger.info("=" * 80)
    logger.info("")
    
    if not results.get('installation'):
        logger.info("❌ SAM-3 não está instalado corretamente")
        logger.info("   Execute: cd analysis/fractal_poc/sam3 && pip install -e .")
    else:
        logger.info("✅ SAM-3 está instalado")
        
        if not results.get('model_files'):
            logger.info("⚠️  Checkpoints do modelo não encontrados")
            logger.info("   Acesso necessário: https://huggingface.co/facebook/sam3-*")
            logger.info("   Ou verificar: https://ai.meta.com/sam3")
        
        logger.info("")
        logger.info("📚 Para usar SAM-3:")
        logger.info("   1. Obter acesso aos checkpoints")
        logger.info("   2. Baixar pesos do modelo")
        logger.info("   3. Testar segmentação com prompts textuais")
        logger.info("")
        logger.info("💡 Exemplo de prompts para leucócitos:")
        logger.info("   - 'white blood cells'")
        logger.info("   - 'neutrophils'")
        logger.info("   - 'lymphocytes'")
        logger.info("   - 'leukemia cells'")


if __name__ == "__main__":
    main()

