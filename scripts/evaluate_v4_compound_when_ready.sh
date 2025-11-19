#!/bin/bash
# Script para avaliar o modelo v4_compound quando o treino terminar
# Criado: 2025-11-16 22:10 -03

set -e

CHECKPOINT="models/dynamic_gnn_v4_compound/best_model.pt"
DATA="data/processed/pbpk_enriched/dynamic_gnn_dataset_enriched_v4.npz"
OUTPUT_DIR="models/dynamic_gnn_v4_compound/evaluation_robust"

echo "🔍 Verificando se o treino v4_compound terminou..."
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint não encontrado. O treino ainda está em andamento."
    exit 1
fi

echo "✅ Checkpoint encontrado. Iniciando avaliação robusta..."
mkdir -p "$OUTPUT_DIR"

python scripts/evaluate_dynamic_gnn_robust.py \
    --data "$DATA" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --hidden-dim 128 \
    --num-gnn-layers 4 \
    --num-temporal-steps 120 \
    --dt 0.1 \
    --device cuda \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

echo ""
echo "✅ Avaliação robusta concluída!"
echo "📊 Resultados em: $OUTPUT_DIR"
echo ""
echo "🔄 Atualizando comparação com v4_compound..."

# Atualizar comparação incluindo v4_compound
python scripts/compare_robust_evaluations.py \
    --evals \
        "sweep_b:models/dynamic_gnn_sweep_b/evaluation_robust/robust_eval.json" \
        "sweep_c:models/dynamic_gnn_sweep_c/evaluation_robust/robust_eval.json" \
        "v4_compound:$OUTPUT_DIR/robust_eval.json" \
    --output models/comparison_robust_all

echo ""
echo "✅ Comparação atualizada com v4_compound!"
echo "📊 Resultados em: models/comparison_robust_all/"


