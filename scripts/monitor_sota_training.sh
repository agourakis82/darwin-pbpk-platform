#!/bin/bash
# Monitoramento dos treinamentos SOTA

echo "================================================================================"
echo "MONITORAMENTO: Treinamento SOTA Dynamic GNN"
echo "================================================================================"
echo ""

# RTX 4000 (Node Atual)
echo "📊 Node DemetriosPCS (RTX 4000 Ada):"
if [ -f "training_sota.log" ]; then
    echo "   ✅ Log encontrado"
    echo "   📝 Últimas linhas:"
    tail -10 training_sota.log | sed 's/^/      /'
    echo ""
    if ps aux | grep -q "[p]ython.*train_dynamic_gnn_sota"; then
        echo "   ✅ Processo rodando"
    else
        echo "   ⚠️  Processo não encontrado"
    fi
else
    echo "   ⚠️  Log não encontrado"
fi
echo ""

# L4 24GB (Node Maria - K8s)
echo "📊 Node Maria (L4 24GB) - Kubernetes:"
JOB_STATUS=$(kubectl get jobs dynamic-gnn-training-sota -o jsonpath='{.status.conditions[0].type}' 2>/dev/null || echo "Unknown")
POD_NAME=$(kubectl get pods -l component=training,version=sota -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "None")
POD_STATUS=$(kubectl get pods -l component=training,version=sota -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Unknown")
NODE=$(kubectl get pods -l component=training,version=sota -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || echo "Unknown")

echo "   Job Status: $JOB_STATUS"
echo "   Pod: $POD_NAME"
echo "   Pod Status: $POD_STATUS"
echo "   Node: $NODE"
echo ""

if [ "$POD_STATUS" != "Unknown" ] && [ "$POD_STATUS" != "None" ]; then
    echo "   📋 Tentando obter logs..."
    kubectl logs "$POD_NAME" --tail=20 2>&1 | tail -15 | sed 's/^/      /' || echo "      ⚠️  Logs inacessíveis (proxy error comum em k3s)"
fi

echo ""
echo "🔧 Comandos úteis:"
echo "   # Ver logs RTX 4000:"
echo "   tail -f training_sota.log"
echo ""
echo "   # Ver status K8s:"
echo "   kubectl get jobs dynamic-gnn-training-sota"
echo "   kubectl get pods -l component=training,version=sota"
echo "   kubectl logs <pod-name>"
echo ""
echo "   # Parar treinamentos:"
echo "   pkill -f train_dynamic_gnn_sota"
echo "   kubectl delete job dynamic-gnn-training-sota"

