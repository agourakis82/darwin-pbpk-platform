# 🚀 Quick Start - Treinamento no Node Maria

## Passo a Passo

### 1. Conectar ao Node Maria
```bash
ssh maria
```

### 2. Ir para o Diretório do Projeto
```bash
cd ~/workspace/darwin-pbpk-platform
```

### 3. Verificar e Iniciar Treinamento
```bash
./scripts/check_and_start_maria.sh
```

**OU manualmente:**

```bash
# Verificar GPU
nvidia-smi

# Verificar dataset
ls -lh data/dynamic_gnn_training_full/training_data.npz

# Iniciar treinamento
python3 scripts/train_dynamic_gnn_pbpk.py \
    --data data/dynamic_gnn_training_full/training_data.npz \
    --output models/dynamic_gnn_maria \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3 \
    --device cuda \
    > training_maria.log 2>&1 &
```

### 4. Monitorar Progresso
```bash
# Seguir log em tempo real
tail -f training_maria.log

# Ver apenas épocas e losses
tail -f training_maria.log | grep -E "(Epoch|Loss|✅)"

# Verificar processo
ps aux | grep train_dynamic_gnn_pbpk
```

### 5. Verificar Modelos
```bash
# Ver modelos salvos
ls -lh models/dynamic_gnn_maria/*.pt

# Ver curva de treinamento
ls -lh models/dynamic_gnn_maria/training_curve.png
```

---

## ⚡ Configuração Otimizada para L4 24GB

- **Batch size:** 32 (vs 16 no RTX 4000)
- **Tempo estimado:** ~6-7 horas (vs 12-13h)
- **Memória:** 24GB disponível (pode aumentar batch se necessário)

---

## 🔧 Troubleshooting

### Dataset não encontrado:
```bash
# Copiar do node atual
scp SounioPCS:~/workspace/darwin-pbpk-platform/data/dynamic_gnn_training_full/training_data.npz \
    data/dynamic_gnn_training_full/
```

### GPU não detectada:
```bash
# Verificar drivers
nvidia-smi

# Verificar PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Treinamento muito lento:
- Verificar se está usando GPU: `nvidia-smi` deve mostrar processo
- Verificar batch_size (32 é ideal para L4)
- Verificar se há outros processos usando GPU

---

## 📊 Comparação de Performance

| GPU | Memória | Batch Size | Tempo/Época | Tempo Total |
|-----|---------|------------|-------------|-------------|
| RTX 4000 Ada | 19.2 GB | 16 | ~14-15 min | ~12-13h |
| **L4 24GB** | **24 GB** | **32** | **~7-8 min** | **~6-7h** |

**L4 é ~2x mais rápido!** ⚡

---

## ✅ Checklist

- [ ] SSH para node maria
- [ ] Dataset disponível
- [ ] GPU detectada (nvidia-smi)
- [ ] Treinamento iniciado
- [ ] Log sendo gerado
- [ ] Modelos sendo salvos

---

**Boa sorte com o treinamento!** 🚀

