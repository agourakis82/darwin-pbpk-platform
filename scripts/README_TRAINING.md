# 🚀 Treinamento Dynamic GNN PBPK

## Quick Start

### 1. Gerar Dataset

```bash
# Dataset pequeno (teste rápido)
python3 scripts/generate_dynamic_gnn_training_data.py \
    --num-samples 100 \
    --output-dir data/dynamic_gnn_training \
    --seed 42

# Dataset completo (treinamento real)
python3 scripts/generate_dynamic_gnn_training_data.py \
    --num-samples 5000 \
    --output-dir data/dynamic_gnn_training_full \
    --seed 42
```

### 2. Treinar Modelo

```bash
# Treinamento rápido (teste)
python3 scripts/train_dynamic_gnn_pbpk.py \
    --data data/dynamic_gnn_training/training_data.npz \
    --output models/dynamic_gnn \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-3 \
    --device cpu

# Treinamento completo (GPU recomendado)
python3 scripts/train_dynamic_gnn_pbpk.py \
    --data data/dynamic_gnn_training_full/training_data.npz \
    --output models/dynamic_gnn_full \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-3 \
    --device cuda
```

## Arquivos Gerados

- `training_data.npz`: Dataset de treinamento
- `metadata.json`: Metadados do dataset
- `best_model.pt`: Melhor modelo (menor val loss)
- `final_model.pt`: Modelo final (última época)
- `training_curve.png`: Curva de treinamento

## Status

⚠️ **Nota:** Há um bug conhecido com shapes que precisa ser corrigido antes do treinamento completo. O código está funcional mas precisa ajustes finos.

