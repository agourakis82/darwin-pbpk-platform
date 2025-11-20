# Release v0.2.0 - Validação Rigorosa Completa

**Data:** 2025-11-18
**Versão:** 0.2.0
**Tag:** `v0.2.0-rigorous-validation`

---

## 🎯 Resumo da Release

Esta release inclui a implementação completa de validação científica rigorosa do modelo Dynamic GNN PBPK, com validação sistemática de parâmetros estimados contra literatura científica, investigação detalhada do problema de escala do Cmax, e criação de dataset expandido.

---

## ✨ Principais Funcionalidades

### 1. Validação Rigorosa de Parâmetros
- Validação sistemática de parâmetros estimados vs literatura científica
- Base de dados de literatura com 9 fármacos comuns
- Análise de Fold Error (FE) para CL, Kp, Vd
- Identificação de discrepâncias críticas

### 2. Investigação Detalhada
- Investigação do problema de escala do Cmax
- Análise de resíduos detalhada (Shapiro-Wilk, t-test)
- Análise por composto específico
- Verificação de normalização e unidades

### 3. Dataset Expandido
- Criação de dataset expandido (6,951 amostras)
- 200 exemplos com doses baixas (< 10 mg)
- 307 exemplos com Kp extremos (< 0.5 ou > 5.0)
- Balanceamento por dose e Kp

### 4. Scripts de Validação
- `validate_parameters_with_literature.py` - Validação vs literatura
- `investigate_cmax_scale_issue.py` - Investigação de escala
- `analyze_residuals_detailed.py` - Análise de resíduos
- `analyze_specific_compounds.py` - Análise por composto
- `verify_normalization_units.py` - Verificação de normalização
- `create_expanded_dataset.py` - Criação de dataset expandido

---

## 🔍 Descobertas Críticas

### 1. Parâmetros Estimados Estão Incorretos
- **CL Total:** FE médio = 228.68× (deveria ser < 2.0×)
- **CL Hepático:** FE médio = 224.19×
- **CL Renal:** FE médio = 243.69×
- **0% dos parâmetros atendem critério** (≥67% com FE ≤ 2.0×)

### 2. Problema de Escala do Cmax Explicado
- **Causa:** CL estimado 100-1000× menor que correto
- **Consequência:** Cmax previsto ~290× maior que observado
- **Equação:** Cmax ≈ Dose / (CL × Vd)

### 3. Discrepâncias Críticas Identificadas
- Ibuprofen: CL est 0.004 vs lit 5.0 L/h (FE = 1,125×)
- Rivaroxaban: CL est 0.010 vs lit 10.0 L/h (FE = 1,040×)
- Caffeine: CL est 0.007 vs lit 2.0 L/h (FE = 300×)

---

## 📊 Estatísticas

### Validação de Parâmetros:
- Compostos validados: **17**
- Parâmetros validados: **6** (CL hepático, CL renal, CL total, Kp liver, Kp kidney, Kp brain)
- **0% dos parâmetros atendem critério** (≥67% com FE ≤ 2.0×)

### Dataset Expandido:
- Tamanho original: **6,551 amostras**
- Tamanho expandido: **6,951 amostras**
- Doses baixas adicionadas: **200 (2.9%)**
- Kp extremos adicionados: **307 (4.4%)**

---

## 📁 Arquivos Adicionados

### Scripts:
- `scripts/validate_parameters_with_literature.py`
- `scripts/investigate_cmax_scale_issue.py`
- `scripts/analyze_residuals_detailed.py`
- `scripts/analyze_specific_compounds.py`
- `scripts/verify_normalization_units.py`
- `scripts/create_expanded_dataset.py`
- `scripts/revalidate_after_finetuning.py`
- `scripts/finetune_on_experimental.py`
- `scripts/calibrate_model_scale.py`

### Documentação:
- `docs/RIGOROUS_VALIDATION_FINAL_REPORT.md`
- `docs/COMPLETE_INVESTIGATION_SUMMARY.md`
- `docs/INVESTIGATION_FINDINGS.md`
- `docs/FINAL_VALIDATION_REPORT.md`
- `docs/SOTA_IMPROVEMENTS_SUMMARY.md`
- `docs/FINETUNING_STATUS.md`

### Dados:
- `data/processed/pbpk_enriched/dynamic_gnn_dataset_enriched_v4_expanded.npz`

---

## 🚀 Próximos Passos

### Crítico:
1. Corrigir algoritmo de estimativa de CL
2. Re-estimar todos os parâmetros com valores corrigidos
3. Re-gerar dataset de treino com parâmetros corrigidos

### Alta Prioridade:
4. Treinar modelo com dataset expandido e parâmetros corrigidos
5. Revalidar em dados experimentais

---

## 📝 Notas de Release

Esta release representa um marco importante na validação científica do modelo, revelando problemas críticos nos parâmetros estimados que explicam o problema de escala do Cmax. As descobertas fornecem direção clara para correções futuras.

---

## 🔗 Links

- **GitHub:** https://github.com/darwin-biomaterials/darwin-pbpk-platform
- **Tag:** `v0.2.0-rigorous-validation`
- **Zenodo:** (a ser criado)

---

**Autor:** Dr. Demetrios Agourakis + AI Assistant
**Data:** 2025-11-18

