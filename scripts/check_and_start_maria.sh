#!/bin/bash
# Script para verificar e iniciar treinamento no node maria

echo "================================================================================"
echo "VERIFICAÇÃO E INÍCIO - Treinamento Dynamic GNN (Node Maria)"
echo "================================================================================"
echo ""

# Verificar se estamos no node maria
HOSTNAME=$(hostname)
echo "📍 Hostname: $HOSTNAME"

if [[ "$HOSTNAME" != *"maria"* ]]; then
    echo "⚠️  ATENÇÃO: Este script deve ser executado no node maria!"
    echo "   Execute: ssh maria"
    echo "   Depois: cd ~/workspace/darwin-pbpk-platform && ./scripts/check_and_start_maria.sh"
    exit 1
fi

echo "✅ Node correto detectado!"
echo ""

# Verificar GPU
echo "🔍 Verificando GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "   GPUs detectadas: $GPU_COUNT"
    echo ""
else
    echo "❌ nvidia-smi não encontrado! GPU pode não estar disponível."
    exit 1
fi

# Verificar dataset
echo "📦 Verificando dataset..."
DATA_PATH="data/dynamic_gnn_training_full/training_data.npz"

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Dataset não encontrado: $DATA_PATH"
    echo ""
    echo "Opções:"
    echo "1. Copiar do node atual:"
    echo "   scp DemetriosPCS:~/workspace/darwin-pbpk-platform/$DATA_PATH $DATA_PATH"
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
    read -p "Deseja parar e reiniciar? (s/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[SsYy]$ ]]; then
        echo "🛑 Parando treinamento atual..."
        pkill -f train_dynamic_gnn_pbpk.py
        sleep 2
    else
        echo "✅ Mantendo treinamento atual rodando."
        exit 0
    fi
fi

# Configurações
OUTPUT_DIR="models/dynamic_gnn_maria"
EPOCHS=50
BATCH_SIZE=32  # L4 tem 24GB, pode usar batch maior!
LR=1e-3

echo "⚙️  Configuração:"
echo "   Dataset: $DATA_PATH"
echo "   Output: $OUTPUT_DIR"
echo "   Épocas: $EPOCHS"
echo "   Batch size: $BATCH_SIZE (otimizado para L4 24GB)"
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
echo "   ./scripts/monitor_training.sh"
echo ""
echo "🛑 Parar treinamento:"
echo "   kill $TRAIN_PID"
echo ""

# Aguardar alguns segundos e mostrar início
sleep 5
echo "📋 Primeiras linhas do log:"
echo "--------------------------------------------------------------------------------"
tail -20 training_maria.log 2>/dev/null || echo "Log ainda não disponível, aguarde alguns segundos..."

echo ""
echo "================================================================================"
echo "✅ TUDO PRONTO! Treinamento rodando em background."
echo "================================================================================"

