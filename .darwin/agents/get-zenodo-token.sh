#!/bin/bash
# Darwin Agent - Get Zenodo Token
# Version: 1.0.0
# Description: Solicita token do Zenodo de forma interativa e segura

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║  🔑 DARWIN AGENT - Obter Token Zenodo                                ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Verificar se já existe token
if [ -f "$HOME/.zenodo_token" ]; then
    echo "⚠️  Token já existe em: ~/.zenodo_token"
    read -p "Deseja sobrescrever? (yes/no): " overwrite
    if [[ ! "$overwrite" =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "✅ Mantendo token existente"
        exit 0
    fi
fi

# Verificar variável de ambiente
if [ -n "${ZENODO_TOKEN:-}" ]; then
    echo "⚠️  Variável ZENODO_TOKEN já está configurada"
    read -p "Deseja configurar um novo token? (yes/no): " new_token
    if [[ ! "$new_token" =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "✅ Usando token da variável de ambiente"
        exit 0
    fi
fi

echo "📋 Instruções para obter token do Zenodo:"
echo ""
echo "1. Acesse: https://zenodo.org/account/settings/applications/tokens/new/"
echo "   (Sandbox: https://sandbox.zenodo.org/account/settings/applications/tokens/new/)"
echo ""
echo "2. Preencha:"
echo "   - Name: Darwin PBPK Platform Upload"
echo "   - Scopes:"
echo "     ✅ deposit:write"
echo "     ✅ deposit:actions"
echo ""
echo "3. Clique em 'Create token'"
echo ""
echo "4. COPIE o token gerado (você só verá uma vez!)"
echo ""

read -p "Pressione ENTER quando tiver o token pronto... "

echo ""
echo "🔐 Cole o token abaixo (não será exibido na tela):"
read -s ZENODO_TOKEN_INPUT

if [ -z "$ZENODO_TOKEN_INPUT" ]; then
    echo "❌ Token vazio. Cancelado."
    exit 1
fi

# Confirmar token
echo ""
echo "🔐 Cole o token novamente para confirmar:"
read -s ZENODO_TOKEN_CONFIRM

if [ "$ZENODO_TOKEN_INPUT" != "$ZENODO_TOKEN_CONFIRM" ]; then
    echo "❌ Tokens não coincidem. Cancelado."
    exit 1
fi

# Salvar token
echo "$ZENODO_TOKEN_INPUT" > "$HOME/.zenodo_token"
chmod 600 "$HOME/.zenodo_token"

echo ""
echo "✅ Token salvo em: ~/.zenodo_token"
echo ""

# Perguntar se quer testar
read -p "Deseja testar o token agora? (yes/no): " test_token
if [[ "$test_token" =~ ^[Yy][Ee][Ss]$ ]]; then
    echo ""
    echo "🧪 Testando token..."
    
    # Testar com API do Zenodo (sandbox)
    response=$(curl -s -H "Authorization: Bearer $ZENODO_TOKEN_INPUT" \
        "https://sandbox.zenodo.org/api/deposit/depositions" \
        -w "\n%{http_code}" || echo "000")
    
    http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
        echo "✅ Token válido! Conexão com Zenodo Sandbox OK."
    else
        echo "⚠️  Teste falhou (código: $http_code)"
        echo "   Isso pode ser normal se o token for de produção"
        echo "   O token foi salvo mesmo assim"
    fi
fi

echo ""
echo "📝 Próximos passos:"
echo ""
echo "1. O token está salvo em: ~/.zenodo_token"
echo "2. Execute o upload:"
echo "   python scripts/upload_to_zenodo.py"
echo ""
echo "   OU configure variável de ambiente:"
echo "   export ZENODO_TOKEN=\$(cat ~/.zenodo_token)"
echo ""

