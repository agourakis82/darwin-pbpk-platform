#!/bin/bash
# Script para remover TODOS os arquivos Python do repositório
# ⚠️ ATENÇÃO: Execução permanente!

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================================================="
echo "REMOÇÃO COMPLETA DE ARQUIVOS PYTHON"
echo "=================================================================================="
echo ""
echo "⚠️  ATENÇÃO: Este script remove PERMANENTEMENTE todos os arquivos Python!"
echo ""
read -p "Tem certeza que deseja continuar? (digite 'SIM' para confirmar): " confirm

if [ "$confirm" != "SIM" ]; then
    echo "❌ Operação cancelada."
    exit 1
fi

echo ""
echo "🗑️  Removendo arquivos Python..."

# Remover arquivos .py (exceto julia-migration)
find "$ROOT" -name "*.py" -type f ! -path "*/julia-migration/*" -delete

# Remover __pycache__
find "$ROOT" -type d -name "__pycache__" ! -path "*/julia-migration/*" -exec rm -rf {} + 2>/dev/null || true

# Remover .pyc
find "$ROOT" -name "*.pyc" -type f ! -path "*/julia-migration/*" -delete

# Remover requirements.txt
if [ -f "$ROOT/requirements.txt" ]; then
    rm "$ROOT/requirements.txt"
    echo "  ✅ Removido: requirements.txt"
fi

# Remover setup.py (se existir e for Python-only)
if [ -f "$ROOT/setup.py" ]; then
    # Verificar se é Python-only (não tem Julia)
    if ! grep -q "Julia" "$ROOT/setup.py" 2>/dev/null; then
        rm "$ROOT/setup.py"
        echo "  ✅ Removido: setup.py"
    fi
fi

# Remover diretórios Python vazios
find "$ROOT/apps" -type d -empty -delete 2>/dev/null || true
find "$ROOT/scripts" -type d -empty -delete 2>/dev/null || true

echo ""
echo "✅ Remoção completa!"
echo ""
echo "Verificando arquivos Python restantes..."
remaining=$(find "$ROOT" -name "*.py" -type f ! -path "*/julia-migration/*" | wc -l)
if [ "$remaining" -eq 0 ]; then
    echo "  ✅ Nenhum arquivo Python encontrado!"
else
    echo "  ⚠️  Ainda existem $remaining arquivos Python:"
    find "$ROOT" -name "*.py" -type f ! -path "*/julia-migration/*" | head -10
fi

echo ""
echo "=================================================================================="
echo "✅ PROCESSO COMPLETO!"
echo "=================================================================================="
