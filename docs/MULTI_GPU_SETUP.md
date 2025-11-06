# 🚀 Setup Multi-GPU para Treinamento Dynamic GNN

**Data:** 06 de Novembro de 2025  
**GPUs Disponíveis:**
- Node atual: NVIDIA RTX 4000 Ada Generation (19.2 GB)
- Node maria: NVIDIA L4 24GB

---

## 📊 OPÇÕES DE CONFIGURAÇÃO

### Opção 1: Treinamento Separado (Recomendado)

Rodar treinamento em cada node separadamente:

#### Node Atual (RTX 4000):
```bash
# Já está rodando
python3 scripts/train_dynamic_gnn_pbpk.py \
    --data data/dynamic_gnn_training_full/training_data.npz \
    --output models/dynamic_gnn_full \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3 \
    --device cuda
```

#### Node Maria (L4 24GB):
```bash
# SSH para node maria
ssh maria

# Rodar treinamento
cd ~/workspace/darwin-pbpk-platform
./scripts/run_training_maria.sh

# Ou manualmente:
python3 scripts/train_dynamic_gnn_pbpk.py \
    --data data/dynamic_gnn_training_full/training_data.npz \
    --output models/dynamic_gnn_maria \
    --epochs 50 \
    --batch-size 32 \  # L4 tem mais memória, pode usar batch maior
    --lr 1e-3 \
    --device cuda
```

**Vantagens:**
- ✅ Simples de configurar
- ✅ Cada GPU trabalha independentemente
- ✅ Pode usar batch sizes diferentes (L4 tem mais memória)
- ✅ Fácil de monitorar

**Desvantagens:**
- ⚠️ Não sincroniza gradientes (mas OK para este caso)

---

### Opção 2: DistributedDataParallel (DDP) Multi-Node

Usar ambas as GPUs com sincronização de gradientes:

#### Setup:

1. **No node master (atual):**
```bash
# Iniciar processo master
python3 scripts/train_dynamic_gnn_ddp.py \
    --data data/dynamic_gnn_training_full/training_data.npz \
    --output models/dynamic_gnn_ddp \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3 \
    --world-size 2 \
    --master-addr $(hostname -I | awk '{print $1}') \
    --master-port 12355
```

2. **No node maria:**
```bash
# SSH e rodar worker
ssh maria
cd ~/workspace/darwin-pbpk-platform

# Rodar worker (mesmo comando, mas precisa do master_addr)
python3 scripts/train_dynamic_gnn_ddp.py \
    --data data/dynamic_gnn_training_full/training_data.npz \
    --output models/dynamic_gnn_ddp \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3 \
    --world-size 2 \
    --master-addr <IP_DO_NODE_MASTER> \
    --master-port 12355
```

**Vantagens:**
- ✅ Sincroniza gradientes entre nodes
- ✅ Treinamento mais consistente
- ✅ Melhor para datasets muito grandes

**Desvantagens:**
- ⚠️ Mais complexo de configurar
- ⚠️ Requer comunicação de rede entre nodes
- ⚠️ Pode ser mais lento devido a overhead de comunicação

---

## 🎯 RECOMENDAÇÃO

**Para este caso, recomendo Opção 1 (Treinamento Separado):**

1. **Node atual (RTX 4000):** Já está rodando, deixar continuar
2. **Node maria (L4 24GB):** Rodar treinamento separado com batch_size maior

**Razões:**
- Dataset não é tão grande (1000 amostras)
- Cada GPU pode completar treinamento independentemente
- L4 tem mais memória (24GB vs 19.2GB), pode usar batch_size 32
- Mais simples e menos propenso a erros

---

## 📊 COMPARAÇÃO DE PERFORMANCE

### RTX 4000 Ada (Node Atual):
- **Memória:** 19.2 GB
- **Batch size:** 16
- **Tempo/época:** ~14-15 min
- **Tempo total:** ~12-13 horas

### L4 24GB (Node Maria):
- **Memória:** 24 GB
- **Batch size:** 32 (recomendado, 2x mais rápido)
- **Tempo/época estimado:** ~7-8 min
- **Tempo total estimado:** ~6-7 horas

**Com ambas rodando:**
- Você terá 2 modelos treinados
- Pode comparar resultados
- Ou usar ensemble dos 2 modelos

---

## 🚀 QUICK START - Node Maria

```bash
# 1. SSH para node maria
ssh maria

# 2. Ir para diretório do projeto
cd ~/workspace/darwin-pbpk-platform

# 3. Rodar script
./scripts/run_training_maria.sh

# 4. Monitorar
tail -f training_maria.log
```

---

## 📁 ARQUIVOS

- `scripts/train_dynamic_gnn_ddp.py` - Script DDP (multi-node)
- `scripts/run_training_maria.sh` - Script para node maria
- `models/dynamic_gnn_full/` - Modelo do node atual
- `models/dynamic_gnn_maria/` - Modelo do node maria (será criado)

---

**Status:** ✅ Scripts prontos para usar ambas as GPUs

