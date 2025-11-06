#!/usr/bin/env python3
"""
🔑 Darwin Agent - Obter Token Zenodo

Solicita token do Zenodo de forma interativa e segura.
Integra com o sistema Darwin para gerenciamento de credenciais.

Uso:
    python scripts/get_zenodo_token.py [--sandbox]
"""
import os
import sys
import getpass
import argparse
import requests
from pathlib import Path


def get_token_interactive(sandbox: bool = False) -> str:
    """Solicita token do usuário de forma interativa"""
    
    env_name = "SANDBOX" if sandbox else "PRODUCTION"
    base_url = "sandbox.zenodo.org" if sandbox else "zenodo.org"
    
    print("=" * 80)
    print(f"🔑 OBTER TOKEN ZENODO {env_name}")
    print("=" * 80)
    print()
    print("📋 Instruções:")
    print()
    print(f"1. Acesse: https://{base_url}/account/settings/applications/tokens/new/")
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
        sys.exit(1)
    
    # Confirmar
    token2 = getpass.getpass("🔐 Cole novamente para confirmar: ")
    
    if token1 != token2:
        print("❌ Tokens não coincidem. Cancelado.")
        sys.exit(1)
    
    return token1


def save_token(token: str, file_path: Path) -> bool:
    """Salva token em arquivo seguro"""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(token)
        file_path.chmod(0o600)  # rw-------
        return True
    except Exception as e:
        print(f"❌ Erro ao salvar token: {e}")
        return False


def test_token(token: str, sandbox: bool = False) -> bool:
    """Testa se o token é válido"""
    base_url = "sandbox.zenodo.org" if sandbox else "zenodo.org"
    
    print("🧪 Testando token...")
    
    try:
        response = requests.get(
            f"https://{base_url}/api/deposit/depositions",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        
        if response.status_code in [200, 201]:
            print("✅ Token válido! Conexão com Zenodo OK.")
            return True
        else:
            print(f"⚠️  Teste retornou código {response.status_code}")
            print("   Isso pode ser normal. O token foi salvo mesmo assim.")
            return False
    except Exception as e:
        print(f"⚠️  Erro ao testar token: {e}")
        print("   O token foi salvo mesmo assim.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Obter token do Zenodo")
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Usar Zenodo Sandbox"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Arquivo para salvar token (padrão: ~/.zenodo_token)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Testar token após salvar"
    )
    
    args = parser.parse_args()
    
    # Verificar se já existe
    output_file = Path(args.output) if args.output else Path.home() / ".zenodo_token"
    
    if output_file.exists():
        print(f"⚠️  Token já existe em: {output_file}")
        response = input("Deseja sobrescrever? (yes/no): ")
        if response.lower() not in ['yes', 'y', 'sim', 's']:
            print("✅ Mantendo token existente")
            return
    
    # Verificar variável de ambiente
    if os.getenv("ZENODO_TOKEN"):
        print("⚠️  Variável ZENODO_TOKEN já está configurada")
        response = input("Deseja configurar um novo token? (yes/no): ")
        if response.lower() not in ['yes', 'y', 'sim', 's']:
            print("✅ Usando token da variável de ambiente")
            return
    
    # Obter token
    token = get_token_interactive(args.sandbox)
    
    # Salvar
    if save_token(token, output_file):
        print()
        print(f"✅ Token salvo em: {output_file}")
    else:
        print("❌ Falha ao salvar token")
        sys.exit(1)
    
    # Testar se solicitado
    if args.test:
        print()
        test_token(token, args.sandbox)
    
    # Próximos passos
    print()
    print("📝 Próximos passos:")
    print()
    print("1. O token está salvo em:", output_file)
    print("2. Execute o upload:")
    print("   python scripts/upload_to_zenodo.py")
    if args.sandbox:
        print("   python scripts/upload_to_zenodo.py --sandbox")
    print()
    print("   OU configure variável de ambiente:")
    print(f"   export ZENODO_TOKEN=$(cat {output_file})")
    print()


if __name__ == "__main__":
    main()

