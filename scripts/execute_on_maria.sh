#!/bin/bash
# Script para executar no node maria via RDMA/filesystem compartilhado
# Este script pode ser executado diretamente no node maria

echo "================================================================================"
echo "TREINAMENTO DYNAMIC GNN - Node Maria (L4 24GB)"
echo "================================================================================"
echo ""

# Verificar se estamos no node maria
HOSTNAME=$(hostname)
echo "📍 Hostname: $HOSTNAME"

# Verificar GPU
echo ""
echo "🔍 Verificando GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader)
    echo "$GPU_INFO"
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "   GPUs detectadas: $GPU_COUNT"
    
    # Verificar se é L4
    if echo "$GPU_INFO" | grep -qi "L4"; then
        echo "✅ L4 24GB detectada!"
        BATCH_SIZE=32  # Otimizado para L4
    else
        echo "⚠️  GPU diferente detectada, usando batch_size padrão"
        BATCH_SIZE=16
    fi
else
    echo "❌ nvidia-smi não encontrado!"
    exit 1
fi

echo ""

# Verificar dataset
echo "📦 Verificando dataset..."
DATA_PATH="data/dynamic_gnn_training_full/training_data.npz"

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Dataset não encontrado: $DATA_PATH"
    echo ""
    echo "Opções:"
    echo "1. Copiar do node atual (se workspace não for compartilhado):"
    echo "   scp SounioPCS:~/workspace/darwin-pbpk-platform/$DATA_PATH $DATA_PATH"
    echo ""
    echo "2. Gerar novo dataset:"
    echo "   python3 scripts/generate_dynamic_gnn_training_data.py \\"
    echo "       --num-samples 1000 \\"
    echo "       --output-dir data/dynamic_gnn_training_full"
    exit 1
fi

DATA_SIZE=$(du -h "$DATA_PATH" | cut -f1)
echo "✅ Dataset encontrado: $DATA_PATH ($DATA_SIZE)"
echo ""

# Verificar se treinamento já está rodando
if pgrep -f "train_dynamic_gnn_pbpk.py" > /dev/null; then
    echo "⚠️  Treinamento já está rodando!"
    PID=$(pgrep -f "train_dynamic_gnn_pbpk.py" | head -1)
    echo "   PID: $PID"
    echo ""
    echo "📊 Para monitorar:"
    echo "   tail -f training_maria.log"
    echo ""
    echo "🛑 Para parar:"
    echo "   kill $PID"
    exit 0
fi

# Configurações
OUTPUT_DIR="models/dynamic_gnn_maria"
EPOCHS=50
LR=1e-3

echo "⚙️  Configuração:"
echo "   Dataset: $DATA_PATH"
echo "   Output: $OUTPUT_DIR"
echo "   Épocas: $EPOCHS"
echo "   Batch size: $BATCH_SIZE (otimizado para GPU disponível)"
echo "   Learning rate: $LR"
echo ""

# Criar diretório de output
mkdir -p "$OUTPUT_DIR"

# Iniciar treinamento
echo "🚀 Iniciando treinamento..."
echo "   Log: training_maria.log"
echo "   Modelos: $OUTPUT_DIR"
echo ""

nohup python3 scripts/train_dynamic_gnn_pbpk.py \
    --data "$DATA_PATH" \
    --output "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --device cuda \
    > training_maria.log 2>&1 &

TRAIN_PID=$!
echo "✅ Treinamento iniciado! PID: $TRAIN_PID"
echo ""
echo "📊 Monitorar progresso:"
echo "   tail -f training_maria.log"
echo ""
echo "🛑 Parar treinamento:"
echo "   kill $TRAIN_PID"
echo ""

# Aguardar e mostrar início
sleep 5
echo "📋 Primeiras linhas do log:"
echo "--------------------------------------------------------------------------------"
tail -20 training_maria.log 2>/dev/null || echo "Log ainda não disponível, aguarde alguns segundos..."

echo ""
echo "================================================================================"
echo "✅ TREINAMENTO INICIADO!"
echo "================================================================================"

