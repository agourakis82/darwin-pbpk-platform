# 🎯 PLANO DE MELHORIAS PARA PBPK

**Data:** 28 de outubro de 2025  
**Objetivo:** Aumentar R² de 0.054 para >0.30 e validar clinicamente

---

## 📊 SITUAÇÃO ATUAL

### Trial 84 (Melhor modelo atual)
```
✅ Val R²:  0.2333
❌ Test R²: 0.0540
   - Fu:        0.126
   - Vd:        0.098  
   - Clearance: -0.063 (NEGATIVO!)
```

### Comparação com Benchmark Externo
```
✅ Ensemble (XGB+RF+NN): R² = 0.438
   - 2-fold accuracy: ~70%
   - Clinicamente aceitável
```

---

## 🔍 PROBLEMAS IDENTIFICADOS

1. **❌ Clearance com R² negativo**
   - Modelo não aprende nada para clearance
   - Predições piores que baseline (mean)

2. **❌ Overfitting severo**
   - Val R² = 0.233 vs Test R² = 0.054
   - Gap de -76%!

3. **❌ Dataset pequeno**
   - 478 moléculas (99.4% missing data)
   - Não generalize para test

4. **❌ Transforms complexos**
   - Logit/log1p funcionam, mas não resolvem tudo

---

## 💡 3 ESTRATÉGIAS PARA MELHORAR

### Estratégia 1: DATA AUGMENTATION 📊
**Problema:** 478 moléculas é muito pouco  
**Solução:** Usar dados externos para pré-treino

**Implementação:**
1. **ChEMBL ADME (29k moléculas)**
   - Pre-train multi-task em ChEMBL
   - Fine-tune em KEC
   - ✅ JÁ TENTADO - Falhou (-28%)

2. **PubChem (~10k moléculas biomateriais)**
   - Unsupervised pre-training
   - Denoising autoencoder
   - ✅ JÁ TENTADO - Falhou (-200%)

3. **💡 NOVA ABORDAGEM: Semi-Supervised Learning**
   - Usar ~100k moléculas PubChem
   - Pseudo-labeling com modelo atual
   - Treinar em dados reais + pseudo-labels
   - Confidence-weighted loss

**Status:** ⚠️  Tentativas falharam, precisa abordagem diferente

---

### Estratégia 2: PHYSICS-INFORMED FINE-TUNING 🧪
**Problema:** Modelo ignora física PBPK  
**Solução:** Reativar physics loss com peso ajustado

**Implementação:**
1. **Physics Loss Components:**
   ```python
   L_total = L_data + λ * L_physics
   
   L_physics = w1*L_mass_balance  # Conservação de massa
             + w2*L_hepatic_flow   # Limite de clearance hepático
             + w3*L_clvd_ratio     # CL/Vd ratio constraints
   ```

2. **Adaptive Physics Weight:**
   - λ = 0 no início
   - Aumentar gradualmente durante treino
   - Evita dominação de physics loss

3. **Target-specific physics:**
   - Fu: Bound check (0 < fu < 1)
   - Vd: Volume plausível (0.1 < Vd < 10 L/kg)
   - CL: Hepatic flow limit (< 1.5 L/min)

**Status:** ⏳ NÃO IMPLEMENTADO AINDA

---

### Estratégia 3: ENSEMBLE COM DIVERSIDADE 🎲
**Problema:** Single model não capta toda variabilidade  
**Solução:** Ensemble com DIFERENTES arquiteturas

**Implementação:**
1. **Diverse Ensemble Members:**
   - Model 1: GNN (graph structure)
   - Model 2: Transformer (sequence)
   - Model 3: Residual MLP (features)
   - Model 4: XGBoost (baseline)

2. **Weighted Average:**
   ```python
   y_pred = Σ wi * yi
   onde wi baseado em validation performance
   ```

3. **Stacking (Nível 2):**
   - Meta-learner aprende a combinar
   - Usa outputs do Level 1 como features

**Status:** ✅ JÁ TENTADO - 10x Trial 84 com seeds diferentes
   - Resultado: Pior que single model (-36%)
   - **Precisa DIVERSIDADE, não apenas seeds!**

---

## 🎯 ESTRATÉGIA RECOMENDADA (3 PASSOS)

### Passo 1: PHYSICS-INFORMED FINE-TUNING (1-2 dias)
```bash
# Script a criar
python scripts/finetune_physics_informed.py \
  --model results/trial84_evaluation/trial84_best.pt \
  --physics-weight 0.01 \
  --adaptive-lambda \
  --epochs 50
```

**Expectativa:** R² 0.05 → 0.15 (+200%)

---

### Passo 2: HETEROGENEOUS ENSEMBLE (2-3 dias)
```bash
# Treinar 4 modelos DIFERENTES
python scripts/train_gnn_pbpk.py      # GNN
python scripts/train_transformer_pbpk.py  # Transformer
python scripts/train_residual_pbpk.py     # ResNet-like
python scripts/train_xgboost_pbpk.py      # XGBoost

# Combinar
python scripts/ensemble_heterogeneous.py \
  --models gnn,transformer,residual,xgboost \
  --weighting validation
```

**Expectativa:** R² 0.15 → 0.25 (+67%)

---

### Passo 3: SEMI-SUPERVISED COM PUBCHEM (3-5 dias)
```bash
# Gerar pseudo-labels para 100k PubChem
python scripts/generate_pseudo_labels.py \
  --model results/ensemble_best.pt \
  --pubchem-smiles data/pubchem_100k.txt \
  --confidence-threshold 0.7

# Treinar com dados reais + pseudo-labels
python scripts/train_semisupervised.py \
  --real-data data/processed/kec_dataset_split.pkl \
  --pseudo-data data/pubchem_pseudo_labels.pkl \
  --pseudo-weight 0.3
```

**Expectativa:** R² 0.25 → 0.35 (+40%)

---

## 📈 ROADMAP COMPLETO

| Fase | Ação | Tempo | R² Esperado |
|------|------|-------|-------------|
| ✅ Atual | Trial 84 | - | 0.054 |
| 🟡 Fase 1 | Physics-informed fine-tuning | 2 dias | 0.15 |
| 🟡 Fase 2 | Heterogeneous ensemble | 3 dias | 0.25 |
| 🟡 Fase 3 | Semi-supervised learning | 5 dias | 0.35 |
| 🟢 **Meta** | **Sistema completo** | **10 dias** | **>0.30** |

---

## ✅ VALIDAÇÃO CLÍNICA

### Métricas de Sucesso
1. **R² > 0.30** (estatístico)
2. **2-fold accuracy > 50%** (clínico)
3. **3-fold accuracy > 80%** (excelente)

### Datasets de Validação
- ✅ KEC Test (242 drugs)
- ⏳ DrugBank (100 drugs clínicos)
- ⏳ PK-DB (50 concentration-time curves)
- ⏳ FDA Real Data (30 drugs aprovados)

### Benchmark Comparação
- Literature PBPK: 2-fold ~ 70-80%
- ML ensembles: R² ~ 0.40-0.50
- **Target:** R² > 0.30 + 2-fold > 60%

---

## 🚀 PRÓXIMOS PASSOS IMEDIATOS

1. **Implementar physics-informed fine-tuning**
   ```bash
   cd scripts
   nano finetune_physics_informed.py
   ```

2. **Treinar GNN para ensemble**
   ```bash
   nano train_gnn_pbpk.py
   ```

3. **Setup validação externa**
   ```bash
   nano validate_external_datasets.py
   ```

---

## 📊 MÉTRICAS DE PROGRESSO

**Acompanhar:**
- R² test (alvo: >0.30)
- 2-fold accuracy (alvo: >60%)
- Bias per parameter
- Calibration (ECE)
- Inference time (<100ms)

**Gráficos:**
- Predicted vs True
- Bland-Altman plots
- Error distribution
- Per-drug-class performance

---

## 💡 CONCLUSÃO

**PROBLEMA REAL:** Overfitting + dataset pequeno + clearance ruim

**SOLUÇÃO:** 
1. Physics constraints (regulação)
2. Ensemble heterogêneo (redução de variância)
3. Semi-supervised (mais dados)

**TIMELINE:** 10 dias para R² > 0.30

**PRIORIDADE:** Começar por Physics-informed (quick win!)

---

**Última atualização:** 28/10/2025 08:30 UTC

