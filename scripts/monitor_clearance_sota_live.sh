#!/bin/bash
# Monitoramento em tempo real do treinamento científico SOTA
# Autor: Dr. Sounio Agourakis
# Data: 2025-11-08

set -e

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml 2>/dev/null || true

JOB_NAME="clearance-sota-training"
NAMESPACE="darwin-pbpk-platform"
PROJECT_ROOT="/home/agourakis82/workspace/darwin-pbpk-platform"

echo "=================================================================================="
echo "🔬 MONITORAMENTO: Treinamento Científico SOTA - Single-Task Clearance"
echo "=================================================================================="
echo ""

# Função para obter status do job
get_job_status() {
    kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.conditions[0].type}' 2>/dev/null || echo "Unknown"
}

# Função para obter pod atual
get_current_pod() {
    kubectl get pods -l component=training,version=sota-clearance -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo ""
}

# Função para obter fase do pod
get_pod_phase() {
    local pod=$1
    kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown"
}

# Função para verificar logs no filesystem
check_filesystem_logs() {
    find "$PROJECT_ROOT/logs" -name "clearance_sota_training_*.log" -type f -mmin -60 2>/dev/null | sort -r | head -1
}

# Loop de monitoramento
POD=$(get_current_pod)
if [ -z "$POD" ]; then
    echo "⚠️  Pod não encontrado. Aguardando criação..."
    sleep 5
    POD=$(get_current_pod)
fi

if [ -z "$POD" ]; then
    echo "❌ Não foi possível encontrar o pod. Verifique o job manualmente."
    exit 1
fi

echo "📦 Pod: $POD"
echo ""

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))

    # Status do job
    JOB_STATUS=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status}' 2>/dev/null || echo "{}")
    COMPLETIONS=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "0")

    # Status do pod
    PHASE=$(get_pod_phase "$POD")

    # Verificar logs no filesystem
    LOG_FILE=$(check_filesystem_logs)

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⏰ Iteração $ITERATION - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Status do Job:"
    echo "   - Completions: $COMPLETIONS/1"
    echo "   - Pod Phase: $PHASE"
    echo ""

    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        echo "📝 Log encontrado: $(basename "$LOG_FILE")"
        echo "   Tamanho: $(du -h "$LOG_FILE" | cut -f1)"
        echo ""
        echo "📄 Últimas 15 linhas do log:"
        echo "────────────────────────────────────────────────────────────────────────────"
        tail -15 "$LOG_FILE" 2>/dev/null | sed 's/^/   /'
        echo "────────────────────────────────────────────────────────────────────────────"
    else
        echo "⏳ Log ainda não disponível no filesystem (pod pode estar em fase inicial)"
    fi

    echo ""

    # Verificar se job completou
    if [ "$COMPLETIONS" = "1" ]; then
        echo "✅ Job concluído com sucesso!"
        break
    fi

    # Verificar se pod falhou
    if [ "$PHASE" = "Failed" ] || [ "$PHASE" = "Error" ]; then
        echo "❌ Pod falhou! Verifique os logs para mais detalhes."
        break
    fi

    # Verificar se pod está completado
    if [ "$PHASE" = "Succeeded" ]; then
        echo "✅ Pod concluído!"
        break
    fi

    echo "🔄 Aguardando 30 segundos antes da próxima verificação..."
    echo ""
    sleep 30
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Status Final:"
kubectl get job "$JOB_NAME" -n "$NAMESPACE" 2>/dev/null || true
echo ""
echo "📁 Verificar resultados em:"
find "$PROJECT_ROOT/models" -name "clearance_sota_*" -type d -mmin -120 2>/dev/null | head -3 || echo "   (diretórios ainda não criados)"
echo ""





