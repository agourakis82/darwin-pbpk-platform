#!/usr/bin/env python3
"""
Atualiza README.md e RELEASE_DESCRIPTION.md com o DOI dos datasets
"""
import argparse
import re
from pathlib import Path


def update_file(file_path: Path, old_doi: str, new_doi: str) -> bool:
    """Atualiza o DOI em um arquivo"""
    if not file_path.exists():
        print(f"⚠️  Arquivo não encontrado: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Substituir DOI
    updated = content.replace(old_doi, new_doi)
    
    if updated == content:
        print(f"⚠️  Nenhuma substituição feita em {file_path.name}")
        return False
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated)
    
    print(f"✅ {file_path.name} atualizado")
    return True


def main():
    parser = argparse.ArgumentParser(description="Atualiza READMEs com DOI dos datasets")
    parser.add_argument(
        "--doi",
        type=str,
        required=True,
        help="DOI dos datasets (ex: 10.5281/zenodo.123456)"
    )
    parser.add_argument(
        "--old-doi",
        type=str,
        default="zenodo.YYYYYY",
        help="DOI antigo a substituir (padrão: zenodo.YYYYYY)"
    )
    
    args = parser.parse_args()
    
    # Normalizar DOI
    doi = args.doi
    if not doi.startswith("10.5281/"):
        if doi.startswith("zenodo."):
            doi = f"10.5281/{doi}"
        else:
            print(f"❌ DOI inválido: {doi}")
            print("   Formato esperado: 10.5281/zenodo.XXXXXX ou zenodo.XXXXXX")
            return
    
    # Formato para substituição (sem https://doi.org/)
    doi_short = doi.replace("10.5281/", "zenodo.")
    
    print("=" * 80)
    print("📝 ATUALIZANDO READMEs COM DOI DOS DATASETS")
    print("=" * 80)
    print()
    print(f"DOI: {doi}")
    print(f"Substituindo: {args.old_doi}")
    print()
    
    base_dir = Path(__file__).parent.parent
    
    # Arquivos para atualizar
    files_to_update = [
        base_dir / "README.md",
        base_dir / "RELEASE_DESCRIPTION.md"
    ]
    
    updated_count = 0
    for file_path in files_to_update:
        if update_file(file_path, args.old_doi, doi_short):
            updated_count += 1
    
    print()
    if updated_count > 0:
        print(f"✅ {updated_count} arquivo(s) atualizado(s)")
        print()
        print("📝 Próximos passos:")
        print("1. Revise as mudanças: git diff")
        print("2. Commit: git add README.md RELEASE_DESCRIPTION.md")
        print("3. Commit: git commit -m 'docs: Add Zenodo dataset DOI'")
        print("4. Push: git push origin main")
    else:
        print("⚠️  Nenhum arquivo foi atualizado")
        print("   Verifique se o DOI antigo está correto")


if __name__ == "__main__":
    main()

