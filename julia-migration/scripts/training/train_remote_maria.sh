#!/bin/bash
# Script para executar treinamento remoto no servidor Maria
# Servidor: 10.100.0.2
# Usuário: maria
# GPU: L4 24GB
# RAM: 256GB
# Rede: 100Gbps

set -e

REMOTE_HOST="10.100.0.2"
REMOTE_USER="maria"
REMOTE_DIR="~/darwin-pbpk-platform"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo "================================================"
echo "TREINAMENTO REMOTO - SERVIDOR MARIA"
echo "================================================"
echo ""
echo "Servidor: $REMOTE_USER@$REMOTE_HOST"
echo "GPU: L4 24GB"
echo "RAM: 256GB"
echo ""

# Configurar SSH sem senha (usando expect se necessário)
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE_USER@$REMOTE_HOST" exit 2>/dev/null; then
    echo "⚠️  SSH sem senha não configurado. Usando sshpass..."

    # Tentar com sshpass (se disponível)
    if command -v sshpass &> /dev/null; then
        SSH_CMD="sshpass -p '123456' ssh -o StrictHostKeyChecking=no"
        SCP_CMD="sshpass -p '123456' scp -o StrictHostKeyChecking=no"
    else
        echo "💡 Instalando sshpass..."
        sudo apt-get update && sudo apt-get install -y sshpass 2>/dev/null || {
            echo "⚠️  sshpass não disponível. Configure SSH key:"
            echo "   ssh-copy-id $REMOTE_USER@$REMOTE_HOST"
            exit 1
        }
        SSH_CMD="sshpass -p '123456' ssh -o StrictHostKeyChecking=no"
        SCP_CMD="sshpass -p '123456' scp -o StrictHostKeyChecking=no"
    fi
else
    SSH_CMD="ssh"
    SCP_CMD="scp"
fi

echo "📂 Sincronizando arquivos..."
echo ""

# Criar diretório remoto
$SSH_CMD "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_DIR/julia-migration/{src,scripts,data}"

# Sincronizar código fonte (apenas julia-migration necessário)
echo "   Sincronizando código Julia..."
rsync -avz --progress \
    "$LOCAL_DIR/julia-migration/src/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/julia-migration/src/" \
    2>/dev/null || $SCP_CMD -r "$LOCAL_DIR/julia-migration/src/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/julia-migration/"

echo "   Sincronizando scripts..."
rsync -avz --progress \
    "$LOCAL_DIR/julia-migration/scripts/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/julia-migration/scripts/" \
    2>/dev/null || $SCP_CMD -r "$LOCAL_DIR/julia-migration/scripts/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/julia-migration/"

echo "   Sincronizando dataset..."
# Criar diretório primeiro
$SSH_CMD "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_DIR/data/processed/pbpk_enriched"
if [ -f "$LOCAL_DIR/data/processed/pbpk_enriched/dynamic_gnn_dataset_enriched_v4.npz" ]; then
    echo "   Copiando dataset (14.7 MB)..."
    $SCP_CMD "$LOCAL_DIR/data/processed/pbpk_enriched/dynamic_gnn_dataset_enriched_v4.npz" \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/data/processed/pbpk_enriched/"
else
    echo "   ⚠️  Dataset local não encontrado. Usando dados sintéticos no remoto."
fi

echo "   Sincronizando Project.toml..."
$SCP_CMD "$LOCAL_DIR/julia-migration/Project.toml" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/julia-migration/"

echo ""
echo "✅ Sincronização completa!"
echo ""

# Executar treinamento remoto
echo "🚀 Iniciando treinamento remoto..."
echo ""

$SSH_CMD "$REMOTE_USER@$REMOTE_HOST" bash << 'REMOTE_SCRIPT'
cd ~/darwin-pbpk-platform/julia-migration

# Verificar Julia
if ! command -v julia &> /dev/null; then
    echo "⚠️  Julia não encontrado. Instalando..."
    # Instalar Julia (ajustar versão se necessário)
    wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz
    tar -xzf julia-1.10.0-linux-x86_64.tar.gz
    export PATH="$PWD/julia-1.10.0/bin:$PATH"
fi

# Verificar GPU
echo "📊 Verificando GPU..."
nvidia-smi || echo "⚠️  GPU não detectada (pode ser problema de driver)"

# Instalar dependências
echo "📦 Instalando dependências Julia..."
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Executar treinamento
echo ""
echo "🎯 Executando treinamento..."
julia --project=. -e 'using CUDA; println("CUDA disponível: ", CUDA.functional())'
julia --project=. julia-migration/scripts/training/train_with_regularization_gpu.jl
REMOTE_SCRIPT

echo ""
echo "================================================"
echo "✅ TREINAMENTO REMOTO CONCLUÍDO"
echo "================================================"
echo ""
echo "📁 Resultados salvos em: $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/models/dynamic_gnn_regularized_gpu/"
echo ""
