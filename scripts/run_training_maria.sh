#!/bin/bash
# Script para rodar treinamento no node maria (L4 24GB)

# Parar treinamento atual se estiver rodando
pkill -f train_dynamic_gnn_pbpk.py 2>/dev/null

# Configurações
DATA_PATH="data/dynamic_gnn_training_full/training_data.npz"
OUTPUT_DIR="models/dynamic_gnn_maria"
EPOCHS=50
BATCH_SIZE=16
LR=1e-3

echo "================================================================================"
echo "TREINAMENTO DYNAMIC GNN - Node Maria (L4 24GB)"
echo "================================================================================"
echo ""
echo "Configuração:"
echo "   Dataset: $DATA_PATH"
echo "   Output: $OUTPUT_DIR"
echo "   Épocas: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
echo "   Learning rate: $LR"
echo "   GPU: L4 24GB"
echo ""

# Verificar GPU
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Rodar treinamento
echo "🚀 Iniciando treinamento..."
python3 scripts/train_dynamic_gnn_pbpk.py \
    --data "$DATA_PATH" \
    --output "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --device cuda \
    2>&1 | tee training_maria.log

echo ""
echo "✅ Treinamento concluído!"
echo "   Log: training_maria.log"
echo "   Modelos: $OUTPUT_DIR"

