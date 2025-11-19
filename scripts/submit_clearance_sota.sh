#!/bin/bash
# Script para submeter job de treinamento científico SOTA (Single-Task Clearance)
# no cluster K8s (node maria)

echo "================================================================================"
echo "🔬 SUBMISSÃO: Treinamento Científico SOTA - Single-Task Clearance"
echo "Cluster K8s (Node Maria - L4 24GB)"
echo "================================================================================"
echo ""

# Verificar se kubectl está disponível
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl não encontrado!"
    echo "   Instale kubectl ou configure o contexto do cluster"
    exit 1
fi

# Verificar contexto do cluster
echo "🔍 Verificando contexto do cluster..."
CURRENT_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "none")
echo "   Contexto atual: $CURRENT_CONTEXT"

# Verificar node maria
echo ""
echo "🔍 Verificando node maria..."
if kubectl get nodes maria &> /dev/null; then
    echo "✅ Node maria encontrado no cluster"
    kubectl get nodes maria -o wide | grep maria
    NODE_GPU=$(kubectl describe node maria | grep -i "nvidia.com/gpu" | head -1 || echo "GPU não detectada")
    echo "   $NODE_GPU"
else
    echo "⚠️  Node maria não encontrado no cluster!"
    echo "   Nodes disponíveis:"
    kubectl get nodes 2>/dev/null || echo "   Não foi possível listar nodes"
    read -p "Deseja continuar mesmo assim? (s/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[SsYy]$ ]]; then
        exit 1
    fi
fi

echo ""

# Verificar se job já existe
JOB_NAME="clearance-sota-training"
if kubectl get jobs "$JOB_NAME" -n darwin-pbpk-platform &> /dev/null; then
    echo "⚠️  Job '$JOB_NAME' já existe!"
    echo ""
    echo "Status atual:"
    kubectl get jobs "$JOB_NAME" -n darwin-pbpk-platform
    echo ""
    read -p "Deseja deletar e recriar? (s/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[SsYy]$ ]]; then
        echo "🗑️  Deletando job existente..."
        kubectl delete job "$JOB_NAME" -n darwin-pbpk-platform
        sleep 3
        echo "✅ Job deletado"
    else
        echo "✅ Mantendo job existente"
        echo ""
        echo "📊 Para monitorar:"
        echo "   ./scripts/monitor_clearance_sota.sh"
        echo "   kubectl get pods -l component=training,version=sota-clearance -n darwin-pbpk-platform"
        exit 0
    fi
fi

# Verificar namespace
echo "🔍 Verificando namespace..."
if ! kubectl get namespace darwin-pbpk-platform &> /dev/null; then
    echo "⚠️  Namespace 'darwin-pbpk-platform' não existe!"
    read -p "Deseja criar? (s/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[SsYy]$ ]]; then
        kubectl create namespace darwin-pbpk-platform
        echo "✅ Namespace criado"
    else
        echo "❌ Abortando (namespace necessário)"
        exit 1
    fi
else
    echo "✅ Namespace 'darwin-pbpk-platform' existe"
fi

# Verificar arquivo YAML
YAML_FILE=".darwin/cluster/k8s/training-job-clearance-sota.yaml"
if [ ! -f "$YAML_FILE" ]; then
    echo "❌ Arquivo YAML não encontrado: $YAML_FILE"
    exit 1
fi
echo "✅ Arquivo YAML encontrado: $YAML_FILE"

# Aplicar job
echo ""
echo "🚀 Submetendo job científico SOTA..."
kubectl apply -f "$YAML_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Job submetido com sucesso!"
    echo ""
    echo "📊 Monitorar progresso:"
    echo "   ./scripts/monitor_clearance_sota.sh"
    echo "   kubectl get jobs $JOB_NAME -n darwin-pbpk-platform"
    echo "   kubectl get pods -l component=training,version=sota-clearance -n darwin-pbpk-platform"
    echo ""
    echo "📝 Ver logs em tempo real:"
    POD_NAME=$(kubectl get pods -l component=training,version=sota-clearance -n darwin-pbpk-platform -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$POD_NAME" ]; then
        echo "   kubectl logs -f $POD_NAME -n darwin-pbpk-platform"
    else
        echo "   kubectl logs -l component=training,version=sota-clearance -n darwin-pbpk-platform --tail=50"
    fi
    echo ""
    echo "🛑 Parar job:"
    echo "   kubectl delete job $JOB_NAME -n darwin-pbpk-platform"
    echo ""
    echo "📁 Resultados serão salvos em:"
    echo "   /workspace/darwin-pbpk-platform/models/clearance_sota_*/"
    echo "   /workspace/darwin-pbpk-platform/logs/clearance_sota_training_*.log"
else
    echo "❌ Erro ao submeter job"
    exit 1
fi

