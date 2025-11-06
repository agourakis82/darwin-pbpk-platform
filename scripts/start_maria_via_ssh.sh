#!/bin/bash
# Script para iniciar treinamento no maria via SSH com autenticação

MARIA_HOST="10.100.0.2"
MARIA_USER="agourakis82"
PROJECT_DIR="~/workspace/darwin-pbpk-platform"

echo "================================================================================"
echo "INICIANDO TREINAMENTO NO NODE MARIA (10.100.0.2)"
echo "================================================================================"
echo ""

# Verificar conectividade
echo "🔍 Verificando conectividade..."
if ping -c 1 -W 2 $MARIA_HOST > /dev/null 2>&1; then
    echo "✅ Node maria acessível (10.100.0.2)"
else
    echo "❌ Node maria não acessível"
    exit 1
fi

echo ""
echo "📋 INSTRUÇÕES PARA EXECUTAR NO NODE MARIA:"
echo "================================================================================"
echo ""
echo "Opção 1: SSH Manual (Recomendado)"
echo "------------------------------------"
echo "ssh $MARIA_USER@$MARIA_HOST"
echo "cd $PROJECT_DIR"
echo "./scripts/check_and_start_maria.sh"
echo ""
echo "Opção 2: SSH com Comando Direto"
echo "--------------------------------"
echo "ssh $MARIA_USER@$MARIA_HOST 'cd $PROJECT_DIR && ./scripts/check_and_start_maria.sh'"
echo ""
echo "Opção 3: Se workspace é compartilhado (NFS/Lustre)"
echo "--------------------------------------------------"
echo "# Se /home/agourakis82/workspace é montado via NFS,"
echo "# você pode executar diretamente quando estiver no node maria:"
echo "cd $PROJECT_DIR"
echo "./scripts/check_and_start_maria.sh"
echo ""
echo "================================================================================"
echo ""
echo "🚀 Tentando executar via SSH (pode pedir senha)..."
echo ""

# Tentar executar (pode pedir senha)
ssh -t $MARIA_USER@$MARIA_HOST << 'ENDSSH'
cd ~/workspace/darwin-pbpk-platform
git pull origin main 2>/dev/null || true
chmod +x scripts/check_and_start_maria.sh 2>/dev/null || true
./scripts/check_and_start_maria.sh
ENDSSH

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Treinamento iniciado no node maria!"
    echo ""
    echo "📊 Monitorar:"
    echo "   ssh $MARIA_USER@$MARIA_HOST 'tail -f ~/workspace/darwin-pbpk-platform/training_maria.log'"
else
    echo ""
    echo "⚠️  Execute manualmente no node maria:"
    echo "   ssh $MARIA_USER@$MARIA_HOST"
    echo "   cd ~/workspace/darwin-pbpk-platform"
    echo "   ./scripts/check_and_start_maria.sh"
fi

