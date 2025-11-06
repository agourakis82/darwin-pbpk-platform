#!/bin/bash
# Script para submeter job de treinamento no cluster K8s (node maria)

echo "================================================================================"
echo "SUBMISSÃO: Treinamento Dynamic GNN no Cluster K8s (Node Maria)"
echo "================================================================================"
echo ""

# Verificar se kubectl está disponível
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl não encontrado!"
    exit 1
fi

# Verificar node maria
echo "🔍 Verificando node maria..."
if kubectl get nodes maria &> /dev/null; then
    echo "✅ Node maria encontrado no cluster"
    kubectl get nodes maria -o wide | grep maria
else
    echo "❌ Node maria não encontrado no cluster!"
    exit 1
fi

echo ""

# Verificar se job já existe
if kubectl get jobs dynamic-gnn-training-maria &> /dev/null; then
    echo "⚠️  Job já existe!"
    read -p "Deseja deletar e recriar? (s/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[SsYy]$ ]]; then
        echo "🗑️  Deletando job existente..."
        kubectl delete job dynamic-gnn-training-maria
        sleep 2
    else
        echo "✅ Mantendo job existente"
        exit 0
    fi
fi

# Aplicar job
echo "🚀 Submetendo job..."
kubectl apply -f .darwin/cluster/k8s/training-job-maria.yaml

if [ $? -eq 0 ]; then
    echo "✅ Job submetido com sucesso!"
    echo ""
    echo "📊 Monitorar:"
    echo "   ./scripts/monitor_k8s_training.sh"
    echo "   kubectl get pods -l component=training"
    echo "   kubectl logs <pod-name>"
    echo ""
    echo "🛑 Parar job:"
    echo "   kubectl delete job dynamic-gnn-training-maria"
else
    echo "❌ Erro ao submeter job"
    exit 1
fi

