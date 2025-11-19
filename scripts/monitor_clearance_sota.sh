#!/bin/bash
# Script para monitorar treinamento científico SOTA (Single-Task Clearance)
# no cluster K8s

echo "================================================================================"
echo "📊 MONITORAMENTO: Treinamento Científico SOTA - Single-Task Clearance"
echo "================================================================================"
echo ""

JOB_NAME="clearance-sota-training"
NAMESPACE="darwin-pbpk-platform"

# Verificar se kubectl está disponível
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl não encontrado!"
    exit 1
fi

# Status do Job
echo "📋 Status do Job:"
echo "────────────────────────────────────────────────────────────────────────────────"
kubectl get jobs "$JOB_NAME" -n "$NAMESPACE" -o wide 2>&1 || echo "⚠️  Job não encontrado"
echo ""

# Status dos Pods
echo "📦 Status dos Pods:"
echo "────────────────────────────────────────────────────────────────────────────────"
kubectl get pods -l component=training,version=sota-clearance -n "$NAMESPACE" -o wide 2>&1
echo ""

# Informações do Pod
POD_NAME=$(kubectl get pods -l component=training,version=sota-clearance -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
NODE=$(kubectl get pods -l component=training,version=sota-clearance -n "$NAMESPACE" -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null)
POD_STATUS=$(kubectl get pods -l component=training,version=sota-clearance -n "$NAMESPACE" -o jsonpath='{.items[0].status.phase}' 2>/dev/null)

if [ -n "$POD_NAME" ]; then
    echo "🔍 Detalhes do Pod:"
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "   Nome: $POD_NAME"
    echo "   Node: ${NODE:-N/A}"
    echo "   Status: ${POD_STATUS:-N/A}"
    echo ""

    # Uso de recursos
    echo "💻 Uso de Recursos:"
    echo "────────────────────────────────────────────────────────────────────────────────"
    kubectl top pod "$POD_NAME" -n "$NAMESPACE" 2>/dev/null || echo "   ⚠️  Métricas não disponíveis (métricas-server pode não estar instalado)"
    echo ""

    # Logs recentes
    echo "📝 Últimas 30 linhas dos logs:"
    echo "────────────────────────────────────────────────────────────────────────────────"
    kubectl logs "$POD_NAME" -n "$NAMESPACE" --tail=30 2>&1 | tail -30 || echo "⚠️  Não foi possível obter logs"
    echo ""

    # Verificar se está rodando
    if [ "$POD_STATUS" = "Running" ]; then
        echo "✅ Pod está rodando!"
        echo ""
        echo "📊 Para acompanhar logs em tempo real:"
        echo "   kubectl logs -f $POD_NAME -n $NAMESPACE"
    elif [ "$POD_STATUS" = "Succeeded" ]; then
        echo "✅ Treinamento concluído com sucesso!"
        echo ""
        echo "📁 Verificar resultados em:"
        echo "   /workspace/darwin-pbpk-platform/models/clearance_sota_*/"
        echo "   /workspace/darwin-pbpk-platform/logs/clearance_sota_training_*.log"
    elif [ "$POD_STATUS" = "Failed" ]; then
        echo "❌ Pod falhou!"
        echo ""
        echo "🔍 Ver logs completos:"
        echo "   kubectl logs $POD_NAME -n $NAMESPACE"
        echo ""
        echo "🔍 Ver eventos:"
        echo "   kubectl describe pod $POD_NAME -n $NAMESPACE"
    else
        echo "⏳ Status: $POD_STATUS"
    fi
else
    echo "⚠️  Nenhum pod encontrado para este job"
    echo ""
    echo "🔍 Verificar jobs:"
    echo "   kubectl get jobs -n $NAMESPACE"
    echo ""
    echo "🔍 Ver todos os pods:"
    echo "   kubectl get pods -n $NAMESPACE"
fi

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "📚 Comandos úteis:"
echo ""
echo "   # Ver logs completos:"
echo "   kubectl logs $POD_NAME -n $NAMESPACE"
echo ""
echo "   # Acompanhar logs em tempo real:"
echo "   kubectl logs -f $POD_NAME -n $NAMESPACE"
echo ""
echo "   # Ver detalhes do pod:"
echo "   kubectl describe pod $POD_NAME -n $NAMESPACE"
echo ""
echo "   # Ver eventos do namespace:"
echo "   kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'"
echo ""
echo "   # Deletar job (parar treinamento):"
echo "   kubectl delete job $JOB_NAME -n $NAMESPACE"
echo ""

