# 🔬 PLANO CIENTÍFICO SOTA - Darwin PBPK Platform

**Data:** 2025-11-08
**Objetivo:** Alcançar performance SOTA (R² > 0.70) para publicação Q1
**Foco:** CIÊNCIA RIGOROSA, não funcionalidades

---

## 📊 SITUAÇÃO ATUAL (HONESTA)

### Performance Atual vs Target SOTA

| Parâmetro | R² Atual | Target SOTA | Gap | % Target |
|-----------|----------|-------------|-----|----------|
| **Clearance** | 0.18 | 0.70 | -0.52 | **26%** ❌ |
| **Vd** | 0.24 | 0.60 | -0.36 | **40%** ❌ |
| **Fu** | 0.19 | 0.50 | -0.31 | **38%** ❌ |

### Problemas Identificados

1. **Encoder Multimodal DESABILITADO**
   - D-MPNN (256d): ❌ Desabilitado → Perdendo +0.15-0.20 R²
   - SchNet (128d): ❌ Desabilitado → Perdendo +0.10-0.15 R²
   - **Total perdido: +0.25-0.35 R²**

2. **Multi-task falhando**
   - 80%+ missing data impede multi-task eficaz
   - **Solução SOTA:** Single-task models primeiro

3. **Dynamic GNN não validado**
   - Implementado mas não comparado rigorosamente com ODE
   - Precisa validação científica adequada

---

## 🎯 OBJETIVOS CIENTÍFICOS

### Objetivo 1: Reativar Encoder Multimodal Completo
**Target:** R² 0.48-0.59 (70-98% do target SOTA)

**Ações:**
1. Criar script de treinamento usando `MultimodalMolecularEncoder` completo
2. Habilitar D-MPNN + SchNet
3. Treinar modelos single-task (Clearance, Vd, Fu separadamente)
4. Validar com métricas rigorosas

**Métricas de Sucesso:**
- Clearance: R² > 0.48 (vs 0.18 atual)
- Vd: R² > 0.54 (vs 0.24 atual)
- Fu: R² > 0.49 (vs 0.19 atual)

### Objetivo 2: Implementar Single-Task Models
**Target:** R² > 0.50 para Clearance (32k samples disponíveis)

**Justificativa Científica:**
- Multi-task não funciona com missing data extensivo
- Literatura mostra que single-task supera multi-task neste cenário
- Clearance tem dataset maior (32k vs 6-7k para Fu/Vd)

**Ações:**
1. Criar `apps/training/03_single_task_clearance.py`
2. Usar encoder multimodal completo (976d)
3. Arquitetura: MLP [1024, 512, 256, 128]
4. Loss: MSE com log1p transform
5. Validação: 5-fold cross-validation

### Objetivo 3: Validar Dynamic GNN Cientificamente
**Target:** R² > 0.90 (vs 0.85-0.90 ODE tradicional)

**Ações:**
1. Gerar dataset de validação (1000+ simulações)
2. Comparar Dynamic GNN vs ODE solver
3. Métricas: R², RMSE, MAE, 2-fold accuracy
4. Análise estatística rigorosa (t-test, effect size)

### Objetivo 4: Comparação com Benchmarks da Literatura
**Target:** Superar ou igualar benchmarks publicados

**Benchmarks a Comparar:**
- TDC ADME Benchmark (R² 0.44 ensemble)
- ChEMBL PK predictions (R² 0.35-0.55)
- Yang et al. 2019 (R² 0.70-0.75)

**Métricas:**
- R² por parâmetro
- 2-fold accuracy
- RMSE normalizado
- Teste estatístico vs baseline

---

## 📋 PLANO DE EXECUÇÃO (CIENTÍFICO)

### Fase 1: Quick Win Científico (1 semana)

**Semana 1, Dia 1-2: Reativar Encoder Multimodal**
- [ ] Criar script `apps/training/04_multimodal_full.py`
- [ ] Usar `MultimodalMolecularEncoder` com D-MPNN + SchNet habilitados
- [ ] Treinar modelos single-task (Clearance-first)
- [ ] Documentar resultados vs baseline

**Semana 1, Dia 3-4: Single-Task Clearance**
- [ ] Implementar modelo Clearance-only
- [ ] Treinar em dataset completo (32k samples)
- [ ] Validação 5-fold cross-validation
- [ ] Comparar com literatura

**Semana 1, Dia 5: Análise e Documentação**
- [ ] Análise estatística completa
- [ ] Comparação com benchmarks
- [ ] Documentar metodologia rigorosamente

**Target Fase 1:** Clearance R² > 0.50

### Fase 2: Validação Rigorosa (2 semanas)

**Semana 2: Validação Dynamic GNN**
- [ ] Gerar dataset de validação (1000+ simulações)
- [ ] Comparar Dynamic GNN vs ODE solver
- [ ] Análise estatística (t-test, effect size)
- [ ] Documentar vantagens/limitações

**Semana 3: Validação Externa**
- [ ] Testar em datasets externos (DrugBank, PK-DB)
- [ ] Comparar com Simcyp/GastroPlus (se possível)
- [ ] Análise de erro por classe de droga
- [ ] Identificar casos de falha

**Target Fase 2:** Dynamic GNN R² > 0.90 validado

### Fase 3: Refinamento e Publicação (3-4 semanas)

**Semana 4-5: Ensemble e Otimização**
- [ ] Implementar ensemble strategy (5x MLP + 3x GNN)
- [ ] Hyperparameter optimization (Optuna)
- [ ] Data augmentation (SMILES enumeration)
- [ ] Validação final

**Semana 6-7: Preparação para Publicação**
- [ ] Redação de metodologia rigorosa
- [ ] Tabelas de resultados completas
- [ ] Figuras de qualidade publicação
- [ ] Comparação detalhada com literatura

**Target Fase 3:** R² > 0.70 em todos os parâmetros

---

## 🔬 METODOLOGIA CIENTÍFICA

### 1. Validação Rigorosa

**Train/Val/Test Split:**
- Train: 80% (scaffold-based split)
- Val: 10% (early stopping)
- Test: 10% (avaliação final)

**Métricas:**
- R² (coeficiente de determinação)
- RMSE (root mean square error)
- MAE (mean absolute error)
- 2-fold accuracy (% dentro de 2x do valor real)
- Pearson correlation

**Análise Estatística:**
- Teste t para comparar modelos
- Effect size (Cohen's d)
- Intervalos de confiança (95%)
- Análise de resíduos

### 2. Comparação com Literatura

**Benchmarks:**
- TDC ADME Benchmark
- ChEMBL PK predictions
- Yang et al. 2019
- Outros trabalhos relevantes

**Métricas de Comparação:**
- R² por parâmetro
- Dataset size (ajustar por tamanho)
- Método utilizado
- Limitações identificadas

### 3. Reproduzibilidade

**Requisitos:**
- Seeds fixos (42)
- Versões de pacotes documentadas
- Scripts completos e comentados
- Datasets com DOI

---

## 📊 RESULTADOS ESPERADOS

### Performance Projetada

| Fase | Clearance R² | Vd R² | Fu R² | Status |
|------|--------------|-------|-------|--------|
| **Atual** | 0.18 | 0.24 | 0.19 | ❌ |
| **Fase 1** | 0.48-0.53 | 0.54-0.59 | 0.49-0.54 | ⏳ |
| **Fase 2** | 0.55-0.65 | 0.60-0.70 | 0.55-0.65 | ⏳ |
| **Fase 3** | **0.70+** | **0.65+** | **0.60+** | ⏳ |

### Publicação Q1

**Target Journals:**
- Nature Machine Intelligence
- Journal of Chemical Information and Modeling (JCIM)
- Bioinformatics

**Requisitos:**
- R² > 0.70 para pelo menos 2 parâmetros
- Validação externa robusta
- Comparação com comerciais (se possível)
- Código open-source com DOI

---

## 🚀 PRÓXIMOS PASSOS IMEDIATOS

### Agora (Hoje):

1. **Criar script de treinamento multimodal completo**
   - Arquivo: `apps/training/04_multimodal_full.py`
   - Usar `MultimodalMolecularEncoder` com D-MPNN + SchNet
   - Single-task Clearance primeiro

2. **Treinar modelo Clearance-only**
   - Dataset: 32k samples
   - Encoder: Multimodal completo (976d)
   - Target: R² > 0.50

3. **Documentar resultados**
   - Métricas completas
   - Comparação com baseline
   - Análise de erros

---

## 📝 NOTAS IMPORTANTES

### Rigor Científico

- **NÃO** aceitar resultados sem validação adequada
- **NÃO** comparar com benchmarks sem ajustar por tamanho de dataset
- **SEMPRE** documentar limitações e falhas
- **SEMPRE** usar métricas padrão da literatura

### Transparência

- Documentar TODAS as decisões metodológicas
- Reportar TODOS os resultados (não apenas os melhores)
- Identificar casos de falha
- Comparar honestamente com literatura

---

**"Rigorous science. Honest results. Real impact."**

**Próximo passo:** Criar script de treinamento multimodal completo AGORA.

