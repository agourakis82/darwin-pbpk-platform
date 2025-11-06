#!/bin/bash
# Script para monitorar treinamento do Dynamic GNN

LOG_FILE="training.log"
OUTPUT_DIR="models/dynamic_gnn_full"

echo "================================================================================"
echo "MONITORAMENTO: Treinamento Dynamic GNN PBPK"
echo "================================================================================"
echo ""

# Verificar se processo está rodando
if pgrep -f "train_dynamic_gnn_pbpk.py" > /dev/null; then
    echo "✅ Processo de treinamento está rodando"
    PID=$(pgrep -f "train_dynamic_gnn_pbpk.py" | head -1)
    echo "   PID: $PID"
else
    echo "⚠️  Processo de treinamento não está rodando"
fi

echo ""
echo "📊 Últimas linhas do log:"
echo "--------------------------------------------------------------------------------"
tail -20 "$LOG_FILE" 2>/dev/null || echo "Log file não encontrado ainda"

echo ""
echo "📁 Arquivos gerados:"
if [ -d "$OUTPUT_DIR" ]; then
    ls -lh "$OUTPUT_DIR" 2>/dev/null | tail -5
else
    echo "   Diretório ainda não criado"
fi

echo ""
echo "💡 Comandos úteis:"
echo "   tail -f $LOG_FILE          # Seguir log em tempo real"
echo "   ps aux | grep train_dynamic # Ver processo"
echo "   kill <PID>                  # Parar treinamento"

