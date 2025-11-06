#!/bin/bash
# Script para executar treinamento remotamente no node maria
# Usa SSH com diferentes métodos de autenticação

MARIA_HOST="10.100.0.2"
MARIA_USER="${USER:-agourakis82}"
PROJECT_DIR="~/workspace/darwin-pbpk-platform"

echo "================================================================================"
echo "EXECUÇÃO REMOTA - Node Maria (L4 24GB)"
echo "================================================================================"
echo ""

# Tentar diferentes métodos de conexão
echo "🔍 Tentando conectar ao node maria ($MARIA_HOST)..."

# Método 1: SSH direto
SSH_CMD="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o LogLevel=ERROR"

# Verificar se workspace é compartilhado (NFS, etc)
if [ -d "/shared/workspace" ] || [ -d "/mnt/shared" ]; then
    echo "✅ Filesystem compartilhado detectado!"
    echo "   Executando localmente (workspace compartilhado)"
    
    # Se workspace é compartilhado, podemos executar diretamente
    # Mas precisamos verificar se estamos no node correto
    HOSTNAME=$(hostname)
    if [[ "$HOSTNAME" == *"maria"* ]]; then
        echo "✅ Já estamos no node maria!"
        cd ~/workspace/darwin-pbpk-platform
        ./scripts/check_and_start_maria.sh
        exit 0
    fi
fi

# Método 2: SSH remoto
echo "📡 Conectando via SSH..."

# Comando completo para executar no maria
REMOTE_CMD="cd $PROJECT_DIR && \
    git pull origin main 2>/dev/null || true && \
    chmod +x scripts/check_and_start_maria.sh 2>/dev/null || true && \
    ./scripts/check_and_start_maria.sh"

# Tentar executar
$SSH_CMD $MARIA_USER@$MARIA_HOST "$REMOTE_CMD" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Comando executado com sucesso no node maria!"
    echo ""
    echo "📊 Para monitorar:"
    echo "   ssh $MARIA_USER@$MARIA_HOST 'tail -f ~/workspace/darwin-pbpk-platform/training_maria.log'"
else
    echo ""
    echo "⚠️  Falha na conexão SSH. Tentando métodos alternativos..."
    echo ""
    echo "Opções:"
    echo "1. Executar manualmente:"
    echo "   ssh $MARIA_USER@$MARIA_HOST"
    echo "   cd ~/workspace/darwin-pbpk-platform"
    echo "   ./scripts/check_and_start_maria.sh"
    echo ""
    echo "2. Se workspace é compartilhado, executar localmente no node maria"
    echo ""
    echo "3. Verificar autenticação SSH:"
    echo "   ssh-copy-id $MARIA_USER@$MARIA_HOST"
fi

