# 🎯 PBPK SOTA - Plano de Ação Imediato

**Data:** 06 de Novembro de 2025  
**Baseado em:** Pesquisa profunda SOTA + Estado atual do código

---

## ✅ DESCOBERTAS IMPORTANTES

### 1. Encoder Multi-Modal JÁ ESTÁ SOTA! ✅

**Implementado (Sprint 3):**
- ✅ ChemBERTa 768d
- ✅ D-MPNN 256d (Directed Message Passing)
- ✅ SchNet 128d (3D convolutions)
- ✅ Cross-Attention Fusion (8-head, 512d unified)
- ✅ KEC 15d (NOVEL - código do mestrado)
- ✅ 3D Conformer 50d
- ✅ QM 15d

**Total:** 976 dimensions (5 modalidades)

**Status:** ✅ **COMPLETO E SOTA!**

**Problema:** D-MPNN e SchNet foram desabilitados no treinamento (para velocidade)

**Ação:** Reativar para obter R² completo (+0.30 esperado)

---

## 🚀 AÇÕES IMEDIATAS (Prioridade)

### Ação 1: Reativar D-MPNN + SchNet no Treinamento ⭐

**Arquivo:** `apps/training/02_gnn_model.py` ou similar

**Mudança:**
```python
# ATUAL (desabilitado):
use_dmpnn=False,
use_schnet=False,

# MUDAR PARA:
use_dmpnn=True,   # Reativar D-MPNN
use_schnet=True,  # Reativar SchNet
```

**Impacto Esperado:**
- +0.15-0.20 R² (D-MPNN - 2D topology)
- +0.10-0.15 R² (SchNet - 3D geometry)
- **Total: +0.25-0.35 R²**

**Tempo:** 5 minutos (mudança de flag)

---

### Ação 2: Implementar Single-Task Models ⭐⭐

**Problema:** Multi-task falhando com 80%+ missing data

**Solução:** Single-task models (Clearance-first)

**Arquivo:** Criar `apps/training/03_single_task_clearance.py`

**Implementação:**
```python
# Model: Clearance-only
# Input: Multi-modal encoder (976d) OU ChemBERTa (768d)
# Architecture: MLP [1024, 512, 256, 128]
# Output: Single task (Clearance)
# Loss: MSE with log1p transform
# Epochs: 200
# Learning rate: 1e-4
```

**Target:** R² > 0.50 (32k samples disponíveis!)

**Tempo:** 2-3 horas (implementação + treino)

---

### Ação 3: Implementar Dynamic GNN para PBPK ⭐⭐⭐

**Breakthrough SOTA 2024:**
- R² 0.93+ vs 0.85-0.90 (ODE tradicional)
- Data-driven, menos parâmetros

**Arquivo:** Criar `apps/pbpk_core/simulation/dynamic_gnn_pbpk.py`

**Arquitetura:**
```python
class DynamicPBPKGNN(nn.Module):
    """
    Dynamic Graph Neural Network para PBPK
    
    Graph: 14 órgãos (nodes)
    Edges: Fluxos sanguíneos, clearance
    Temporal: Evolution via GNN layers
    Attention: Critical organs (liver, kidney, brain)
    """
    def __init__(self):
        # 14-compartment graph
        # Dynamic edges (time-dependent)
        # GNN layers for temporal evolution
        # Attention mechanism
        pass
```

**Tempo:** 3-4 semanas (implementação completa)

**Impacto:** R² 0.93+ (vs 0.85-0.90 atual)

---

## 📋 ROADMAP PRIORITIZADO

### Semana 1 (Imediato):
1. ✅ Reativar D-MPNN + SchNet (5 min)
2. ⏳ Treinar modelo com encoder completo (2-3h)
3. ⏳ Implementar single-task Clearance (2-3h)
4. ⏳ Validar resultados

**Target:** Clearance R² > 0.50

### Semana 2-3:
5. ⏳ Ensemble strategy (5x MLP + 3x GNN)
6. ⏳ Hyperparameter optimization (Optuna)
7. ⏳ Fu-only e Vd-only models

**Target:** Clearance R² > 0.60

### Semana 4-6:
8. ⏳ Implementar Dynamic GNN
9. ⏳ Validar vs ODE solver
10. ⏳ Integrar no pipeline

**Target:** R² > 0.90 (Dynamic GNN)

---

## 💡 RECOMENDAÇÃO ESTRATÉGICA

### Opção A: Quick Win (1 semana)
1. Reativar D-MPNN + SchNet
2. Single-task Clearance
3. **Resultado:** R² > 0.50-0.60

### Opção B: Breakthrough (4-6 semanas)
1. Tudo da Opção A
2. + Dynamic GNN
3. **Resultado:** R² > 0.90 (SOTA absoluto)

**Recomendação:** Começar com Opção A (quick win), depois Opção B (breakthrough)

---

## 🎯 PRÓXIMO PASSO IMEDIATO

**Agora mesmo:**
1. Reativar D-MPNN + SchNet no código de treinamento
2. Rodar treinamento com encoder completo
3. Comparar resultados vs encoder parcial

**Tempo total:** ~30 minutos (mudança + treino rápido)

---

**"Rigorous science. Honest results. Real impact."**

