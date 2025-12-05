#!/usr/bin/env python3
"""
Check SAM-3 Access Status
=========================

Check if we have access to SAM-3 model on HuggingFace.

Created: 2025-12-01
Author: Darwin PBPK Platform
"""

import sys
from pathlib import Path

# Add sam3 to path
SCRIPT_DIR = Path(__file__).parent
SAM3_DIR = SCRIPT_DIR / "sam3"
sys.path.insert(0, str(SAM3_DIR))

def check_access():
    """Check SAM-3 model access."""
    print("=" * 80)
    print("🔍 VERIFICANDO ACESSO AO SAM-3")
    print("=" * 80)
    print()
    
    # Check HuggingFace authentication
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Autenticado no HuggingFace como: {user_info.get('name', 'N/A')}")
        print()
    except Exception as e:
        print(f"❌ Erro de autenticação: {e}")
        print("   Execute: hf auth login")
        return False
    
    # Check model info
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        model_info = api.model_info('facebook/sam3')
        
        print("📋 INFORMAÇÕES DO MODELO")
        print("-" * 80)
        print(f"Modelo: {model_info.id}")
        print(f"Gated: {model_info.gated}")
        print(f"Private: {model_info.private}")
        print(f"Downloads: {model_info.downloads:,}")
        print()
        
        if model_info.gated:
            print("⚠️  ATENÇÃO: Modelo requer aprovação manual (gated)")
            print()
            print("📝 PARA OBTER ACESSO:")
            print("   1. Acesse: https://huggingface.co/facebook/sam3")
            print("   2. Clique em 'Request access' (solicitar acesso)")
            print("   3. Aguarde aprovação (geralmente rápida)")
            print("   4. Depois, execute este script novamente")
            print()
    except Exception as e:
        print(f"❌ Erro ao obter informações do modelo: {e}")
        return False
    
    # Try to download a small file to test access
    print("🧪 TESTANDO ACESSO...")
    print("-" * 80)
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Try to download config file (small)
        print("   Tentando baixar config.json...")
        config_path = hf_hub_download(
            repo_id='facebook/sam3',
            filename='config.json',
            local_files_only=False
        )
        print(f"✅ Acesso confirmado! Arquivo baixado: {config_path}")
        print()
        return True
        
    except Exception as e:
        error_msg = str(e)
        if '403' in error_msg or 'Forbidden' in error_msg:
            print("❌ Acesso negado (403 Forbidden)")
            print()
            print("📝 AÇÃO NECESSÁRIA:")
            print("   1. Acesse: https://huggingface.co/facebook/sam3")
            print("   2. Clique em 'Request access' na página do modelo")
            print("   3. Aguarde aprovação")
            print("   4. Execute este script novamente após aprovação")
            print()
            return False
        else:
            print(f"⚠️  Erro ao testar acesso: {e}")
            print()
            return False


def main():
    """Main function."""
    has_access = check_access()
    
    print("=" * 80)
    if has_access:
        print("✅ STATUS: Acesso confirmado ao SAM-3!")
        print()
        print("🚀 Próximo passo:")
        print("   python segment_leukocytes_sam3.py")
    else:
        print("⚠️  STATUS: Acesso pendente")
        print()
        print("📝 Próximo passo:")
        print("   1. Solicitar acesso em: https://huggingface.co/facebook/sam3")
        print("   2. Aguardar aprovação")
        print("   3. Executar este script novamente")
    print("=" * 80)


if __name__ == "__main__":
    main()

