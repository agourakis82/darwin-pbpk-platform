# 🔬 Treinamento Científico SOTA - Cluster Distribuído

**Criado:** 2025-11-08
**Autor:** Dr. Demetrios Chiuratto Agourakis
**Status:** ✅ Pronto para execução no cluster

---

## 📋 Resumo

Configuração completa para execução do treinamento científico **Single-Task Clearance** no cluster Kubernetes distribuído (node Maria - L4 24GB).

---

## 🎯 Objetivo Científico

- **Target:** R² > 0.50 para Clearance
- **Metodologia:** Single-task model com encoder multimodal completo (976d)
- **Rigor:** Validação 5-fold cross-validation
- **Comparação:** Benchmarks da literatura (TDC, ChEMBL)

---

## 📁 Arquivos Criados

### 1. Job Kubernetes
**Arquivo:** `.darwin/cluster/k8s/training-job-clearance-sota.yaml`

- **Imagem:** `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`
- **Node:** Maria (L4 24GB)
- **Recursos:**
  - GPU: 1x NVIDIA L4
  - CPU: 6-8 cores
  - Memória: 16-24 GiB
- **Volume:** Workspace compartilhado (`/workspace`)
- **TTL:** 48 horas após conclusão

### 2. Script de Submissão
**Arquivo:** `scripts/submit_clearance_sota.sh`

- Verifica kubectl e contexto do cluster
- Verifica node Maria
- Verifica/cria namespace `darwin-pbpk-platform`
- Submete job com validações

### 3. Script de Monitoramento
**Arquivo:** `scripts/monitor_clearance_sota.sh`

- Status do job e pods
- Uso de recursos (CPU, memória)
- Logs recentes (últimas 30 linhas)
- Comandos úteis para debugging

### 4. Script de Treinamento Atualizado
**Arquivo:** `apps/training/03_single_task_clearance_multimodal.py`

- ✅ Suporte a argumentos de linha de comando (`argparse`)
- ✅ Configurável via parâmetros do job K8s
- ✅ Fallback automático para TDC se dataset consolidado não existir

---

## 🚀 Como Usar

### 1. Submeter Job

```bash
cd /home/agourakis82/workspace/darwin-pbpk-platform
./scripts/submit_clearance_sota.sh
```

### 2. Monitorar Progresso

```bash
# Monitoramento rápido
./scripts/monitor_clearance_sota.sh

# Logs em tempo real
kubectl logs -f -l component=training,version=sota-clearance -n darwin-pbpk-platform

# Status do job
kubectl get jobs clearance-sota-training -n darwin-pbpk-platform

# Status dos pods
kubectl get pods -l component=training,version=sota-clearance -n darwin-pbpk-platform
```

### 3. Verificar Resultados

Após conclusão, os resultados estarão em:

```
/workspace/darwin-pbpk-platform/models/clearance_sota_YYYYMMDD_HHMMSS/
├── best_model_fold_1.pt
├── best_model_fold_2.pt
├── best_model_fold_3.pt
├── best_model_fold_4.pt
├── best_model_fold_5.pt
├── training.log
└── results.json

/workspace/darwin-pbpk-platform/logs/
└── clearance_sota_training_YYYYMMDD_HHMMSS.log
```

### 4. Parar Job (se necessário)

```bash
kubectl delete job clearance-sota-training -n darwin-pbpk-platform
```

---

## ⚙️ Configuração do Job

O job executa o seguinte comando:

```bash
python3 apps/training/03_single_task_clearance_multimodal.py \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --batch-size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 15 \
    --num-folds 5
```

**Parâmetros configuráveis:**
- `--output-dir`: Diretório de saída (com timestamp automático)
- `--device`: cuda/cpu
- `--batch-size`: Tamanho do batch (padrão: 32)
- `--epochs`: Número de épocas (padrão: 100)
- `--lr`: Learning rate (padrão: 1e-4)
- `--patience`: Early stopping patience (padrão: 15)
- `--num-folds`: Número de folds para CV (padrão: 5)

---

## 📊 Dependências Instaladas no Job

O job instala automaticamente:

- **PyTorch:** Pré-instalado na imagem
- **PyTorch Geometric:** `torch-geometric` + extensions CUDA
- **Transformers:** Para ChemBERTa
- **RDKit:** Para descritores moleculares
- **PyTDC:** Para carregar datasets TDC diretamente
- **Scikit-learn:** Para métricas e cross-validation
- **Outras:** numpy, scipy, pandas, matplotlib, tqdm, pydantic

---

## 🔍 Troubleshooting

### Job não inicia
```bash
# Verificar eventos do namespace
kubectl get events -n darwin-pbpk-platform --sort-by='.lastTimestamp'

# Ver detalhes do pod
kubectl describe pod <pod-name> -n darwin-pbpk-platform
```

### GPU não disponível
```bash
# Verificar se node Maria tem GPU
kubectl describe node maria | grep -i gpu

# Verificar se device plugin está configurado
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, gpu: .status.capacity."nvidia.com/gpu"}'
```

### Erro de memória
- Reduzir `--batch-size` no job YAML
- Aumentar `memory.limits` no job YAML

### Dataset não encontrado
- O script tem fallback automático para TDC
- Verificar logs para mensagens de fallback

---

## 📝 Próximos Passos

1. ✅ Job Kubernetes criado
2. ✅ Scripts de submissão e monitoramento criados
3. ✅ Script de treinamento atualizado com argparse
4. ⏳ **Submeter job e monitorar execução**
5. ⏳ Validar resultados (R² > 0.50)
6. ⏳ Comparar com benchmarks da literatura
7. ⏳ Publicar resultados científicos

---

## 🎓 Metodologia Científica

- **Single-task learning:** Foco em Clearance (evita missing data de multi-task)
- **Multimodal encoder:** 976d (ChemBERTa 768d + GNN 128d + KEC 32d + 3D 16d + QM 32d)
- **Cross-validation:** 5-fold independente (rigor científico)
- **Early stopping:** Patience 15 épocas (evita overfitting)
- **Métricas:** R², RMSE, MAE (padrão da literatura)

---

**Última atualização:** 2025-11-08

