# 🚀 Execução no Node Maria via RDMA

**Conexão:** RDMA 100Gbps  
**Node Maria:** 10.100.0.2  
**GPU:** L4 24GB

---

## 📋 MÉTODO 1: Se Workspace é Compartilhado (NFS/RDMA)

Se o workspace está montado via NFS/Lustre e acessível de ambos os nodes:

```bash
# No node maria (10.100.0.2)
cd ~/workspace/darwin-pbpk-platform
./scripts/execute_on_maria.sh
```

**Vantagem:** Execução direta, sem cópia de arquivos!

---

## 📋 MÉTODO 2: SSH Manual (Recomendado)

```bash
# 1. Conectar ao node maria
ssh agourakis82@10.100.0.2

# 2. Ir para workspace
cd ~/workspace/darwin-pbpk-platform

# 3. Atualizar código (se necessário)
git pull origin main

# 4. Executar script
./scripts/execute_on_maria.sh
```

---

## 📋 MÉTODO 3: SSH com Comando Único

```bash
ssh agourakis82@10.100.0.2 "cd ~/workspace/darwin-pbpk-platform && ./scripts/execute_on_maria.sh"
```

---

## 📋 MÉTODO 4: Via RDMA Direct (Se Configurado)

Se há acesso RDMA direto configurado:

```bash
# Verificar se há ferramentas RDMA
which ibstat ibdev2netdev 2>/dev/null

# Se disponível, pode executar diretamente via RDMA
# (depende da configuração específica do cluster)
```

---

## 🔍 VERIFICAÇÕES

### 1. Verificar Conectividade RDMA:
```bash
ping -c 2 10.100.0.2
```

### 2. Verificar GPU no Maria:
```bash
ssh agourakis82@10.100.0.2 "nvidia-smi"
```

### 3. Verificar Workspace Compartilhado:
```bash
# No node atual
df -h ~/workspace

# No node maria
ssh agourakis82@10.100.0.2 "df -h ~/workspace"
```

Se ambos apontam para o mesmo filesystem (NFS/Lustre), o workspace é compartilhado!

---

## ⚡ CONFIGURAÇÃO OTIMIZADA

O script `execute_on_maria.sh` detecta automaticamente:
- **L4 24GB:** batch_size = 32
- **Outras GPUs:** batch_size = 16

---

## 📊 MONITORAMENTO

Após iniciar:

```bash
# No node maria
tail -f ~/workspace/darwin-pbpk-platform/training_maria.log

# Ou remotamente
ssh agourakis82@10.100.0.2 "tail -f ~/workspace/darwin-pbpk-platform/training_maria.log"
```

---

## 🎯 RESULTADO ESPERADO

- **Tempo estimado:** ~6-7 horas (L4 com batch_size 32)
- **Modelo salvo:** `models/dynamic_gnn_maria/best_model.pt`
- **Log:** `training_maria.log`

---

**Status:** ✅ Scripts prontos para execução no node maria!

