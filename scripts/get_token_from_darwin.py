#!/usr/bin/env python3
"""
🔑 Darwin Agent - Obter Token Zenodo do Darwin

Usa o sistema Darwin para solicitar o token do Zenodo de forma interativa.
Verifica múltiplos repositórios Darwin para encontrar o token.

Author: Dr. Demetrios Chiuratto Agourakis
Date: November 6, 2025
"""
import os
import sys
import getpass
from pathlib import Path
import subprocess


def check_darwin_repos():
    """Verifica repositórios Darwin para encontrar token"""
    workspace = Path.home() / "workspace"
    repos_to_check = [
        "kec-biomaterials-scaffolds",
        "pcs-meta-repo",
        "darwin-pbpk-platform"
    ]
    
    print("🔍 Procurando token em repositórios Darwin...")
    print()
    
    for repo_name in repos_to_check:
        repo_path = workspace / repo_name
        if not repo_path.exists():
            continue
        
        print(f"📁 Verificando: {repo_name}")
        
        # Verificar variável de ambiente no contexto do repo
        try:
            result = subprocess.run(
                ["bash", "-c", f"cd {repo_path} && echo $ZENODO_TOKEN"],
                capture_output=True,
                text=True,
                timeout=5
            )
            token = result.stdout.strip()
            if token and len(token) > 10:
                print(f"   ✅ Token encontrado em variável de ambiente!")
                return token
        except:
            pass
        
        # Verificar arquivo .zenodo_token
        token_file = repo_path / ".zenodo_token"
        if token_file.exists():
            try:
                token = token_file.read_text().strip()
                if token and len(token) > 10:
                    print(f"   ✅ Token encontrado em: {token_file}")
                    return token
            except:
                pass
        
        # Verificar ~/.zenodo_token
        home_token = Path.home() / ".zenodo_token"
        if home_token.exists():
            try:
                token = home_token.read_text().strip()
                if token and len(token) > 10:
                    print(f"   ✅ Token encontrado em: ~/.zenodo_token")
                    return token
            except:
                pass
        
        print(f"   ❌ Não encontrado")
    
    return None


def request_token_interactive():
    """Solicita token do usuário de forma interativa"""
    print()
    print("=" * 80)
    print("🔑 SOLICITAR TOKEN ZENODO")
    print("=" * 80)
    print()
    print("📋 Instruções:")
    print()
    print("1. Acesse: https://zenodo.org/account/settings/applications/tokens/new/")
    print("   (Sandbox: https://sandbox.zenodo.org/account/settings/applications/tokens/new/)")
    print()
    print("2. Preencha:")
    print("   - Name: Darwin PBPK Platform Upload")
    print("   - Scopes:")
    print("     ✅ deposit:write")
    print("     ✅ deposit:actions")
    print()
    print("3. Clique em 'Create token'")
    print()
    print("4. COPIE o token gerado (você só verá uma vez!)")
    print()
    
    input("Pressione ENTER quando tiver o token pronto... ")
    print()
    
    # Solicitar token (não será exibido)
    token1 = getpass.getpass("🔐 Cole o token (não será exibido): ")
    
    if not token1:
        print("❌ Token vazio. Cancelado.")
        return None
    
    # Confirmar
    token2 = getpass.getpass("🔐 Cole novamente para confirmar: ")
    
    if token1 != token2:
        print("❌ Tokens não coincidem. Cancelado.")
        return None
    
    return token1


def save_token(token: str, output_path: Path):
    """Salva token em arquivo seguro"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(token)
        output_path.chmod(0o600)  # rw-------
        return True
    except Exception as e:
        print(f"❌ Erro ao salvar token: {e}")
        return False


def main():
    print("=" * 80)
    print("🔑 DARWIN AGENT - Obter Token Zenodo")
    print("=" * 80)
    print()
    
    # 1. Procurar token existente
    token = check_darwin_repos()
    
    if token:
        print()
        print("✅ Token encontrado!")
        print()
        
        # Perguntar se quer usar ou sobrescrever
        response = input("Deseja usar este token? (yes/no) [yes]: ").strip().lower()
        if response in ['', 'yes', 'y', 'sim', 's']:
            # Salvar no local padrão
            output_file = Path.home() / ".zenodo_token"
            if save_token(token, output_file):
                print(f"✅ Token salvo em: {output_file}")
                print()
                print("📝 Próximos passos:")
                print("   python scripts/upload_to_zenodo.py")
                return
            else:
                print("⚠️  Erro ao salvar, mas token está disponível")
                return
    
    # 2. Solicitar token interativamente
    print()
    token = request_token_interactive()
    
    if not token:
        print("❌ Nenhum token fornecido")
        sys.exit(1)
    
    # 3. Salvar token
    output_file = Path.home() / ".zenodo_token"
    if save_token(token, output_file):
        print()
        print(f"✅ Token salvo em: {output_file}")
        print()
        print("📝 Próximos passos:")
        print("   python scripts/upload_to_zenodo.py")
    else:
        print("❌ Falha ao salvar token")
        sys.exit(1)


if __name__ == "__main__":
    main()

