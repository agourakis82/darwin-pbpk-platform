#!/usr/bin/env python3
"""
📤 Upload Datasets para Zenodo via API

Automatiza o upload dos datasets do Darwin PBPK Platform para o Zenodo.

Uso:
    python scripts/upload_to_zenodo.py [--sandbox] [--token TOKEN]

Author: Dr. Demetrios Chiuratto Agourakis
Date: November 6, 2025
"""

import os
import sys
import json
import requests
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import time

# Base URLs
ZENODO_API_URL = "https://zenodo.org/api"
ZENODO_SANDBOX_URL = "https://sandbox.zenodo.org/api"

# Metadados do dataset
METADATA = {
    "metadata": {
        "title": "Darwin PBPK Platform - Training Datasets v1.0.0",
        "upload_type": "dataset",
        "description": """Training datasets for Darwin PBPK Platform v1.0.0, including:

- **consolidated_pbpk_v1.parquet**: Processed PBPK data for 44,779 compounds (ChEMBL + TDC + KEC)
- **chemberta_embeddings_consolidated.npz**: ChemBERTa embeddings (768d, 44,779 molecules)
- **molecular_graphs.pkl**: Molecular graphs in PyTorch Geometric format

**Dataset Details:**
- Total compounds: 44,779
- Train/Val/Test split: 80/10/10 (scaffold-based, zero leakage)
- PBPK parameters: Fu, Vd, CL
- Sources: ChEMBL, TDC (Therapeutics Data Commons), KEC

**Related Software:**
- Repository: https://github.com/agourakis82/darwin-pbpk-platform
- Software DOI: https://doi.org/10.5281/zenodo.17536674""",
        "creators": [
            {
                "name": "Agourakis, Demetrios Chiuratto",
                "affiliation": "PUCRS - Pontifícia Universidade Católica do Rio Grande do Sul"
            }
        ],
        "keywords": [
            "pharmacokinetics",
            "PBPK",
            "machine learning",
            "drug discovery",
            "ADMET",
            "ChEMBL",
            "molecular graphs",
            "ChemBERTa",
            "deep learning",
            "GNN"
        ],
        "related_identifiers": [
            {
                "identifier": "10.5281/zenodo.17536674",
                "relation": "isSupplementTo",
                "resource_type": "software"
            }
        ],
        "license": "cc-by-4.0",
        "publication_date": "2025-11-06",
        "version": "1.0.0"
    }
}


def get_zenodo_token(sandbox: bool = False) -> Optional[str]:
    """Obtém o token do Zenodo das variáveis de ambiente ou arquivo de configuração"""
    
    # Tentar variável de ambiente
    token = os.getenv("ZENODO_TOKEN")
    if token:
        return token
    
    # Tentar arquivo de configuração
    config_file = Path.home() / ".zenodo_token"
    if config_file.exists():
        with open(config_file, 'r') as f:
            token = f.read().strip()
            if token:
                return token
    
    # Tentar arquivo no projeto
    project_config = Path(__file__).parent.parent / ".zenodo_token"
    if project_config.exists():
        with open(project_config, 'r') as f:
            token = f.read().strip()
            if token:
                return token
    
    return None


def create_deposition(api_url: str, token: str) -> Dict:
    """Cria um novo depósito no Zenodo"""
    print("📦 Criando novo depósito no Zenodo...")
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{api_url}/deposit/depositions",
        json={},
        headers=headers,
        params={"access_token": token}
    )
    
    if response.status_code not in [200, 201]:
        raise Exception(f"Erro ao criar depósito: {response.status_code} - {response.text}")
    
    deposition = response.json()
    print(f"✅ Depósito criado: ID {deposition['id']}")
    return deposition


def upload_file(api_url: str, token: str, deposition_id: int, file_path: Path) -> Dict:
    """Faz upload de um arquivo para o depósito"""
    print(f"📤 Fazendo upload: {file_path.name} ({file_path.stat().st_size / (1024**2):.1f} MB)...")
    
    # Criar bucket de upload
    bucket_url = f"{api_url}/deposit/depositions/{deposition_id}/files"
    headers = {"Authorization": f"Bearer {token}"}
    
    with open(file_path, 'rb') as f:
        files = {'file': (file_path.name, f)}
        data = {'name': file_path.name}
        
        response = requests.post(
            bucket_url,
            files=files,
            data=data,
            headers=headers,
            params={"access_token": token}
        )
    
    if response.status_code not in [200, 201]:
        raise Exception(f"Erro ao fazer upload: {response.status_code} - {response.text}")
    
    file_info = response.json()
    print(f"✅ Upload concluído: {file_path.name}")
    return file_info


def update_metadata(api_url: str, token: str, deposition_id: int, metadata: Dict) -> Dict:
    """Atualiza os metadados do depósito"""
    print("📝 Atualizando metadados...")
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.put(
        f"{api_url}/deposit/depositions/{deposition_id}",
        json=metadata,
        headers=headers,
        params={"access_token": token}
    )
    
    if response.status_code not in [200, 201]:
        raise Exception(f"Erro ao atualizar metadados: {response.status_code} - {response.text}")
    
    deposition = response.json()
    print("✅ Metadados atualizados")
    return deposition


def publish_deposition(api_url: str, token: str, deposition_id: int) -> Dict:
    """Publica o depósito"""
    print("🚀 Publicando depósito...")
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{api_url}/deposit/depositions/{deposition_id}/actions/publish",
        headers=headers,
        params={"access_token": token}
    )
    
    if response.status_code not in [200, 201]:
        raise Exception(f"Erro ao publicar: {response.status_code} - {response.text}")
    
    deposition = response.json()
    print("✅ Depósito publicado!")
    return deposition


def get_doi(deposition: Dict) -> str:
    """Extrai o DOI do depósito"""
    doi = deposition.get("doi", "")
    if not doi and "metadata" in deposition:
        doi = deposition["metadata"].get("doi", "")
    return doi


def main():
    parser = argparse.ArgumentParser(description="Upload datasets para Zenodo via API")
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Usar Zenodo Sandbox (para testes)"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Token de acesso do Zenodo (ou use ZENODO_TOKEN env var)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas simular, não fazer upload real"
    )
    parser.add_argument(
        "--files-dir",
        type=str,
        default="/tmp/darwin-pbpk-datasets-v1.0.0",
        help="Diretório com os arquivos para upload"
    )
    
    args = parser.parse_args()
    
    # Determinar URL da API
    api_url = ZENODO_SANDBOX_URL if args.sandbox else ZENODO_API_URL
    env_name = "SANDBOX" if args.sandbox else "PRODUCTION"
    
    print("=" * 80)
    print(f"📤 UPLOAD PARA ZENODO {env_name}")
    print("=" * 80)
    print()
    
    # Obter token
    token = args.token or get_zenodo_token(args.sandbox)
    if not token:
        print("❌ Token do Zenodo não encontrado!")
        print()
        print("Para obter um token:")
        print("1. Acesse: https://zenodo.org/account/settings/applications/tokens/new/")
        if args.sandbox:
            print("   (Sandbox: https://sandbox.zenodo.org/account/settings/applications/tokens/new/)")
        print("2. Crie um token com permissões de 'deposit:write' e 'deposit:actions'")
        print("3. Configure uma das opções:")
        print("   - Variável de ambiente: export ZENODO_TOKEN='seu_token'")
        print("   - Arquivo: echo 'seu_token' > ~/.zenodo_token")
        print("   - Ou passe via --token")
        print()
        sys.exit(1)
    
    # Verificar diretório de arquivos
    files_dir = Path(args.files_dir)
    if not files_dir.exists():
        print(f"❌ Diretório não encontrado: {files_dir}")
        print("Execute primeiro: bash scripts/prepare_zenodo_upload.sh")
        sys.exit(1)
    
    # Listar arquivos para upload
    files_to_upload = [
        files_dir / "consolidated_pbpk_v1.parquet",
        files_dir / "chemberta_embeddings_consolidated.npz",
        files_dir / "molecular_graphs.pkl",
        files_dir / "README.md"
    ]
    
    # Verificar se todos os arquivos existem
    missing_files = [f for f in files_to_upload if not f.exists()]
    if missing_files:
        print("❌ Arquivos não encontrados:")
        for f in missing_files:
            print(f"   - {f}")
        print()
        print("Execute primeiro: bash scripts/prepare_zenodo_upload.sh")
        sys.exit(1)
    
    print(f"📋 Arquivos para upload ({len(files_to_upload)}):")
    total_size = 0
    for f in files_to_upload:
        size = f.stat().st_size
        total_size += size
        print(f"   - {f.name} ({size / (1024**2):.1f} MB)")
    print(f"   Total: {total_size / (1024**2):.1f} MB")
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN - Nenhum upload será feito")
        print()
        print("Metadados que seriam enviados:")
        print(json.dumps(METADATA, indent=2, ensure_ascii=False))
        return
    
    # Confirmar
    print(f"⚠️  Você está prestes a fazer upload para Zenodo {env_name}")
    if not args.sandbox:
        print("⚠️  ATENÇÃO: Isso publicará os dados em PRODUÇÃO!")
    response = input("Continuar? (yes/no): ")
    if response.lower() not in ['yes', 'y', 'sim', 's']:
        print("❌ Cancelado pelo usuário")
        return
    
    try:
        # 1. Criar depósito
        deposition = create_deposition(api_url, token)
        deposition_id = deposition['id']
        print()
        
        # 2. Fazer upload dos arquivos
        for file_path in files_to_upload:
            upload_file(api_url, token, deposition_id, file_path)
            time.sleep(1)  # Pequena pausa entre uploads
        print()
        
        # 3. Atualizar metadados
        update_metadata(api_url, token, deposition_id, METADATA)
        print()
        
        # 4. Publicar
        if not args.sandbox:
            publish_response = input("Publicar o depósito agora? (yes/no): ")
            if publish_response.lower() not in ['yes', 'y', 'sim', 's']:
                print("⚠️  Depósito criado mas NÃO publicado")
                print(f"   URL: {api_url.replace('/api', '')}/deposit/{deposition_id}")
                print("   Você pode publicar manualmente depois")
                return
        
        deposition = publish_deposition(api_url, token, deposition_id)
        print()
        
        # 5. Obter DOI
        doi = get_doi(deposition)
        if doi:
            print("=" * 80)
            print("🎉 UPLOAD CONCLUÍDO COM SUCESSO!")
            print("=" * 80)
            print()
            print(f"📄 DOI: {doi}")
            print(f"🔗 URL: https://doi.org/{doi}")
            print()
            print("📝 Próximos passos:")
            print("1. Copie o DOI acima")
            print("2. Execute: python scripts/update_readme_with_doi.py --doi", doi)
            print("   OU atualize manualmente README.md e RELEASE_DESCRIPTION.md")
            print()
        else:
            print("⚠️  DOI ainda não disponível (pode levar alguns minutos)")
            print(f"   Verifique em: {api_url.replace('/api', '')}/deposit/{deposition_id}")
        
    except Exception as e:
        print()
        print("❌ Erro durante o upload:")
        print(f"   {str(e)}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()

