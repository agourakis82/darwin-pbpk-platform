#!/bin/bash
# Script de teste simplificado para debug do job Kubernetes
# Este script testa o ambiente antes de executar o treinamento completo

set -e
set -o pipefail

LOG_DIR="/workspace/darwin-pbpk-platform/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/clearance_sota_debug_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "🔍 DEBUG: Teste de Ambiente - Treinamento Clearance SOTA"
echo "================================================================================"
echo "📍 Node: $(hostname)"
echo "⏰ Início: $(date)"
echo ""

# Teste 1: Comandos básicos
echo "✅ Teste 1: Comandos básicos"
python3 --version || echo "❌ python3 não encontrado"
pip --version || echo "❌ pip não encontrado"
echo ""

# Teste 2: PyTorch e CUDA
echo "✅ Teste 2: PyTorch e CUDA"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponível: {torch.cuda.is_available()}')" || echo "❌ Erro ao importar PyTorch"
echo ""

# Teste 3: Workspace
echo "✅ Teste 3: Workspace"
if [ -d "/workspace/darwin-pbpk-platform" ]; then
    echo "✅ Workspace encontrado: /workspace/darwin-pbpk-platform"
    cd /workspace/darwin-pbpk-platform
    echo "   Diretório atual: $(pwd)"
    echo "   Arquivos em apps/training/:"
    ls -la apps/training/ | head -5 || echo "   ❌ Diretório não encontrado"
else
    echo "❌ Workspace não encontrado"
fi
echo ""

# Teste 4: Imports Python
echo "✅ Teste 4: Imports Python"
cd /workspace/darwin-pbpk-platform || exit 1
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
print('✅ Path configurado')
" || echo "❌ Erro ao configurar path"
echo ""

# Teste 5: Import MultimodalEncoder
echo "✅ Teste 5: Import MultimodalMolecularEncoder"
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
try:
    from apps.pbpk_core.ml.multimodal import MultimodalMolecularEncoder
    print('✅ MultimodalMolecularEncoder importado com sucesso')
except Exception as e:
    print(f'❌ Erro ao importar: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" || echo "❌ Falha no teste de import"
echo ""

# Teste 6: Script de treinamento
echo "✅ Teste 6: Script de treinamento"
if [ -f "apps/training/03_single_task_clearance_multimodal.py" ]; then
    echo "✅ Script encontrado"
    python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
code = open('apps/training/03_single_task_clearance_multimodal.py').read()
compile(code, 'apps/training/03_single_task_clearance_multimodal.py', 'exec')
print('✅ Script Python válido')
" || echo "❌ Script tem erros de sintaxe"
else
    echo "❌ Script não encontrado"
fi
echo ""

echo "================================================================================"
echo "✅ Testes de ambiente concluídos"
echo "📝 Log salvo em: $LOG_FILE"
echo "⏰ Fim: $(date)"
echo "================================================================================"

