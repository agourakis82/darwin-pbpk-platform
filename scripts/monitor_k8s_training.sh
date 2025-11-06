#!/bin/bash
# Script para monitorar treinamento no cluster K8s

echo "================================================================================"
echo "MONITORAMENTO: Treinamento Dynamic GNN (K8s Job)"
echo "================================================================================"
echo ""

# Verificar job
echo "📊 Status do Job:"
kubectl get jobs dynamic-gnn-training-maria -o wide 2>&1
echo ""

# Verificar pods
echo "📦 Pods:"
kubectl get pods -l app=darwin-pbpk-platform,component=training -o wide 2>&1
echo ""

# Verificar em qual node está rodando
NODE=$(kubectl get pods -l app=darwin-pbpk-platform,component=training -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null)
POD_NAME=$(kubectl get pods -l app=darwin-pbpk-platform,component=training -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -n "$NODE" ]; then
    echo "📍 Node: $NODE"
    echo "📍 Pod: $POD_NAME"
    echo ""
    
    # Tentar logs
    echo "📋 Últimas linhas do log (tentando...):"
    echo "--------------------------------------------------------------------------------"
    kubectl logs "$POD_NAME" --tail=30 2>&1 | tail -20 || echo "⚠️  Não foi possível obter logs via kubectl"
    echo ""
    
    # Se está no node maria, podemos acessar diretamente
    if [ "$NODE" == "maria" ]; then
        echo "💡 Dica: Pod está no node maria (10.100.0.2)"
        echo "   Para ver logs completos, acesse o node maria:"
        echo "   ssh agourakis82@10.100.0.2"
        echo "   kubectl logs $POD_NAME"
    fi
else
    echo "⚠️  Pod não encontrado ou ainda não iniciado"
fi

echo ""
echo "🔧 Comandos úteis:"
echo "   kubectl get jobs dynamic-gnn-training-maria"
echo "   kubectl get pods -l component=training"
echo "   kubectl logs <pod-name>"
echo "   kubectl delete job dynamic-gnn-training-maria  # Para parar"

