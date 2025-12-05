#!/usr/bin/env python3
"""
Test SAM-3 Integration for Leukocyte Segmentation
==================================================

Script de teste para verificar disponibilidade e viabilidade do SAM-3
para segmentação de leucócitos em análises fractais.

Created: 2025-12-01
Author: Darwin PBPK Platform
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data" / "leukocytes"


def check_sam3_availability() -> Tuple[bool, Optional[str]]:
    """
    Verifica se SAM-3 está disponível e qual método pode ser usado.
    
    Returns:
        (is_available, method) - method pode ser "python", "api", "web"
    """
    logger.info("🔍 Verificando disponibilidade do SAM-3...")
    
    # Método 1: Tentar importar biblioteca Python
    try:
        import sam3
        logger.info("✅ SAM-3 library encontrada via 'import sam3'")
        return True, "python"
    except ImportError:
        logger.info("   ❌ 'sam3' não encontrado via import direto")
    
    # Método 2: Tentar segment-anything-3 ou similar
    try:
        import segment_anything_3 as sam3
        logger.info("✅ SAM-3 encontrada via 'segment_anything_3'")
        return True, "python"
    except ImportError:
        logger.info("   ❌ 'segment_anything_3' não encontrado")
    
    # Método 3: Verificar se há API web disponível
    try:
        import requests
        # Tentar acessar Segment Anything Playground
        response = requests.get("https://segment-anything-playground.com", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Segment Anything Playground disponível via web")
            return True, "web"
    except Exception as e:
        logger.info(f"   ❌ API web não acessível: {e}")
    
    # Método 4: Verificar Hugging Face
    try:
        from transformers import AutoModel
        # Tentar verificar se modelo existe
        logger.info("   Verificando Hugging Face...")
        # Note: Não vamos baixar agora, só verificar disponibilidade
        return False, None  # Aguardar release oficial
    except ImportError:
        logger.info("   ❌ transformers não instalado")
    
    logger.warning("⚠️  SAM-3 não encontrado em nenhum método testado")
    return False, None


def check_sam2_availability() -> Tuple[bool, Optional[str]]:
    """
    Verifica disponibilidade do SAM-2 (versão anterior) como fallback.
    
    SAM-2 não tem prompts textuais, mas pode ser útil para comparação.
    """
    logger.info("🔍 Verificando SAM-2 como alternativa...")
    
    try:
        from segment_anything import sam_model_registry, SamPredictor
        logger.info("✅ SAM-2 encontrado (versão anterior)")
        logger.info("   Nota: SAM-2 não suporta prompts textuais, apenas pontos/boxes")
        return True, "sam2"
    except ImportError:
        logger.info("   ❌ SAM-2 não encontrado")
        return False, None


def test_segmentation_with_current_method(image_path: Path) -> dict:
    """Testa segmentação com método atual (threshold) para baseline."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    
    logger.info(f"📸 Testando método atual em {image_path.name}...")
    
    # Load image
    img = Image.open(image_path).convert('L')
    arr = np.array(img) / 255.0
    
    # Current method: threshold
    threshold = np.mean(arr) - 0.5 * np.std(arr)
    binary = arr < threshold
    
    # Clean up
    binary = ndimage.binary_opening(binary, iterations=2)
    binary = ndimage.binary_closing(binary, iterations=2)
    
    # Count cells
    labeled, n_features = ndimage.label(binary)
    
    # Filter by size
    valid_cells = 0
    for i in range(1, n_features + 1):
        cell_mask = labeled == i
        area = np.sum(cell_mask)
        if 100 < area < 10000:
            valid_cells += 1
    
    return {
        "method": "threshold_current",
        "n_cells_detected": valid_cells,
        "n_components": n_features,
        "image_shape": arr.shape
    }


def test_segmentation_with_sam3(image_path: Path, prompt: str = "white blood cells") -> Optional[dict]:
    """
    Testa segmentação com SAM-3 (se disponível).
    
    Args:
        image_path: Caminho para imagem
        prompt: Prompt textual para segmentação
    
    Returns:
        Dict com resultados ou None se não disponível
    """
    logger.info(f"🤖 Tentando segmentação SAM-3 com prompt: '{prompt}'...")
    
    available, method = check_sam3_availability()
    
    if not available:
        logger.warning("SAM-3 não disponível. Pulando teste.")
        return None
    
    # Aqui implementaríamos o teste real quando SAM-3 estiver disponível
    logger.info("   ⚠️  Implementação completa aguarda release oficial do SAM-3")
    
    return {
        "method": "sam3",
        "available": True,
        "method_type": method,
        "status": "pending_implementation"
    }


def compare_methods(image_path: Path):
    """Compara métodos de segmentação."""
    logger.info("=" * 80)
    logger.info("📊 COMPARAÇÃO DE MÉTODOS DE SEGMENTAÇÃO")
    logger.info("=" * 80)
    
    # Teste método atual
    current_results = test_segmentation_with_current_method(image_path)
    logger.info(f"\n✅ Método Atual (Threshold):")
    logger.info(f"   Células detectadas: {current_results['n_cells_detected']}")
    logger.info(f"   Componentes totais: {current_results['n_components']}")
    
    # Teste SAM-3 (se disponível)
    sam3_results = test_segmentation_with_sam3(image_path, "white blood cells")
    if sam3_results:
        logger.info(f"\n🤖 SAM-3:")
        logger.info(f"   Status: {sam3_results['status']}")
        logger.info(f"   Método: {sam3_results['method_type']}")
    
    # Teste SAM-2 (fallback)
    sam2_available, _ = check_sam2_availability()
    if sam2_available:
        logger.info(f"\n📦 SAM-2 (Fallback):")
        logger.info(f"   Disponível: Sim")
        logger.info(f"   Limitação: Não suporta prompts textuais")
    
    logger.info("\n" + "=" * 80)


def check_segment_anything_playground():
    """Verifica se Segment Anything Playground está acessível."""
    logger.info("🌐 Verificando Segment Anything Playground...")
    
    try:
        import requests
        from urllib.parse import urljoin
        
        base_urls = [
            "https://segment-anything-playground.com",
            "https://segment-anything.com",
            "https://sam3.metademolab.com",  # Possível URL alternativa
        ]
        
        for url in base_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Playground acessível: {url}")
                    return True, url
            except Exception as e:
                logger.debug(f"   {url}: {e}")
        
        logger.info("   ❌ Playground não acessível")
        return False, None
        
    except ImportError:
        logger.warning("   ⚠️  'requests' não instalado. Instale com: pip install requests")
        return False, None


def main():
    """Função principal."""
    logger.info("🧪 SAM-3 Integration Test")
    logger.info("=" * 80)
    logger.info("")
    
    # 1. Verificar disponibilidade SAM-3
    logger.info("1️⃣  VERIFICANDO DISPONIBILIDADE")
    logger.info("-" * 80)
    available, method = check_sam3_availability()
    
    if available:
        logger.info(f"\n✅ SAM-3 DISPONÍVEL via: {method}")
    else:
        logger.info("\n❌ SAM-3 NÃO DISPONÍVEL")
        logger.info("   Verificando alternativas...")
        check_sam2_availability()
    
    logger.info("")
    
    # 2. Verificar Playground web
    logger.info("2️⃣  VERIFICANDO WEB PLAYGROUND")
    logger.info("-" * 80)
    playground_available, playground_url = check_segment_anything_playground()
    if playground_available:
        logger.info(f"   Acesse em: {playground_url}")
    
    logger.info("")
    
    # 3. Testar com imagem real (se disponível)
    logger.info("3️⃣  TESTANDO COM IMAGEM REAL")
    logger.info("-" * 80)
    
    # Procurar imagem de teste
    test_images = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png"))
    if test_images:
        test_image = test_images[0]
        logger.info(f"   Imagem de teste: {test_image.name}")
        compare_methods(test_image)
    else:
        logger.warning("   ⚠️  Nenhuma imagem encontrada para teste")
        logger.info(f"   Procurando em: {DATA_DIR}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("📝 PRÓXIMOS PASSOS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("1. Aguardar release oficial do SAM-3 (Meta)")
    logger.info("2. Verificar documentação oficial quando disponível")
    logger.info("3. Testar Segment Anything Playground manualmente")
    logger.info("4. Avaliar precisão comparada a métodos atuais")
    logger.info("")
    logger.info("📚 Documentação completa: docs/SAM3_LEUKOCYTE_SEGMENTATION_ANALYSIS.md")


if __name__ == "__main__":
    main()

