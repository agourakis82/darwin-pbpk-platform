# Recomendações Científicas para Validação do DynamicPBPKGNN

**Data:** 2025-11-17
**Autor:** AI Assistant + Dr. Sounio Agourakis
**Contexto:** R² muito alto (~0.99999) em modelos PBPK com dados simulados

---

## 🎯 Problema Identificado

O modelo DynamicPBPKGNN alcança R² ~0.99999 mesmo após:
- Split estrito por composto (sem data leakage)
- Dose variável (50-200 mg)
- Ruído fisiológico adicionado
- Avaliação por janelas temporais
- Transformação log1p
- Comparação com baselines

**Hipótese:** O problema é inerentemente simples ou o dataset (gerado por simulação determinística) é muito regular.

---

## 📊 Métricas Científicas Adequadas para PBPK

### 1. **Fold Error (FE) - Padrão Ouro em PBPK**

**Definição:**
```
FE = max(predicted/observed, observed/predicted)
```

**Critérios de Aceitação (FDA/EMA):**
- **Aceitável:** FE ≤ 2.0 para ≥67% das previsões
- **Excelente:** FE ≤ 1.5 para ≥67% das previsões
- **Ideal:** FE ≤ 1.25 para ≥67% das previsões

**Vantagens sobre R²:**
- Não é inflado por valores extremos
- Interpretação clínica direta (erro de 2x é clinicamente relevante)
- Padrão regulatório aceito

### 2. **Geometric Mean Fold Error (GMFE)**

**Definição:**
```
GMFE = 10^(mean(|log10(predicted/observed)|))
```

**Critérios:**
- GMFE < 1.5: Excelente
- GMFE < 2.0: Aceitável
- GMFE > 2.0: Inaceitável

### 3. **Mean Absolute Error (MAE) e Root Mean Squared Error (RMSE)**

**Por órgão e janela temporal:**
- MAE em escala log10 para concentrações
- RMSE normalizado pela média observada (CV%)

### 4. **Percentual de Previsões Dentro de Faixas**

- % dentro de 1.25x, 1.5x, 2.0x do observado
- Visualização: scatter plots com linhas de 2x

---

## 🔬 Validação Externa: O Padrão Científico

### 1. **Split por Composto (Já Implementado)**
✅ **Status:** Implementado no v4_compound

### 2. **Validação Externa com Dados Experimentais**

**Recomendação Crítica:**
- Avaliar o modelo em dados experimentais reais (não simulados)
- Fontes potenciais:
  - ChEMBL (dados experimentais de PK)
  - PubChem BioAssay
  - Literatura científica (extração manual)
  - Dados internos/proprietários

**Métricas em Dados Experimentais:**
- FE, GMFE, MAE, RMSE
- Comparação com modelos ODE tradicionais
- Análise de resíduos por órgão

### 3. **Validação Temporal (Time-based Split)**

- Treinar em compostos com dados até 2020
- Validar em compostos com dados 2021+
- Simula evolução temporal do conhecimento

### 4. **Validação por Scaffold Molecular**

- Split por scaffolds químicos (estruturas base)
- Garante generalização para novas classes químicas
- Implementação: usar fingerprints moleculares (ECFP, MACCS)

---

## 🧪 Análise de Robustez Adicional

### 1. **Perturbação de Parâmetros**

**Teste de Sensibilidade:**
- Adicionar ruído gaussiano aos parâmetros de entrada (clearances, Kp)
- Verificar degradação do R²
- Se R² permanece >0.99 com ruído significativo → modelo muito simples

**Implementação:**
```python
# Adicionar ruído aos parâmetros
noise_level = 0.1  # 10% de ruído
perturbed_params = original_params * (1 + np.random.normal(0, noise_level))
```

### 2. **Teste de Generalização a Novos Compostos**

**Leave-One-Compound-Out (LOCO) Cross-Validation:**
- Treinar em N-1 compostos
- Validar no composto restante
- Repetir para todos os compostos
- Calcular FE médio e desvio padrão

### 3. **Comparação com Modelos Simples**

**Baselines Adicionais:**
- **Regressão Linear:** Clearance → Concentração (por órgão)
- **kNN:** k=5, features = clearances + Kp médio
- **Random Forest:** Features = clearances + Kp + dose
- **ODE Solver:** Simulação PBPK tradicional (sem ML)

**Critério de Sucesso:**
- GNN deve superar significativamente (p<0.05) os baselines
- Se GNN ≈ kNN → modelo não está aprendendo padrões complexos

### 4. **Análise de Resíduos**

**Padrões a Verificar:**
- Resíduos devem ser aleatórios (sem padrão)
- Sem heterocedasticidade (variância constante)
- Sem viés sistemático por órgão ou tempo

**Visualizações:**
- Resíduos vs. predito
- Resíduos vs. observado
- Resíduos por órgão (boxplots)
- Resíduos por janela temporal

---

## 📈 Métricas por Contexto Clínico

### 1. **Concentrações Terapêuticas**

- Focar em faixas clinicamente relevantes
- Exemplo: concentrações > IC50 ou > Cmin terapêutico
- Calcular FE apenas nesses pontos

### 2. **Fase de Eliminação**

- Analisar separadamente fase de distribuição (0-12h) vs. eliminação (12h+)
- Fase de eliminação é mais crítica para doseamento
- FE na fase de eliminação deve ser <1.5

### 3. **Órgãos Críticos**

- Liver, kidney, brain (BBB) são mais críticos
- Ponderar métricas por importância clínica
- Exemplo: peso 2x para liver/kidney, 1.5x para brain

---

## 🎓 Padrões de Publicação Científica

### 1. **Transparência Metodológica**

**Obrigatório Reportar:**
- Número de compostos únicos no treino/validação
- Critérios de split (por composto, por scaffold, temporal)
- Distribuição de doses, clearances, Kp
- Número de parâmetros do modelo
- Tempo de treinamento e recursos computacionais

### 2. **Comparação com Estado da Arte**

**Baselines Obrigatórios:**
- ODE solver tradicional
- Modelos ML simples (linear, RF)
- Modelos GNN estáticos (sem evolução temporal)
- Modelos da literatura (se disponíveis)

### 3. **Análise de Limitações**

**Reconhecer:**
- Dataset sintético (não experimental)
- Limitações da simulação determinística
- Possível overfitting a padrões simples
- Necessidade de validação em dados experimentais

### 4. **Visualizações Científicas**

**Obrigatórias:**
- Scatter plots: predito vs. observado (com linhas 2x)
- Resíduos vs. predito (por órgão)
- Curvas de concentração: observado vs. predito (exemplos representativos)
- Distribuição de FE (histograma)
- Heatmap de FE por órgão × janela temporal

---

## 🔧 Implementação Recomendada

### Script: `evaluate_dynamic_gnn_scientific.py`

**Métricas a Implementar:**
1. Fold Error (FE) e % dentro de 1.25x, 1.5x, 2.0x
2. Geometric Mean Fold Error (GMFE)
3. MAE e RMSE (escala log10)
4. Comparação com baselines (linear, kNN, RF, ODE)
5. Análise de resíduos
6. LOCO cross-validation (opcional)

**Saídas:**
- JSON com todas as métricas
- Gráficos científicos (scatter, resíduos, FE distribution)
- Tabela comparativa (modelo vs. baselines)
- Relatório Markdown formatado para publicação

### Script: `validate_on_experimental_data.py`

**Quando dados experimentais estiverem disponíveis:**
- Carregar dados experimentais (ChEMBL, PubChem, literatura)
- Prever com modelo treinado
- Calcular FE, GMFE, MAE, RMSE
- Comparar com ODE solver
- Gerar relatório de validação externa

---

## 🎯 Critérios de Sucesso Científico

### Mínimo Aceitável (para Publicação):
- FE ≤ 2.0 para ≥67% das previsões (validação externa)
- GMFE < 2.0
- Supera significativamente (p<0.05) modelos simples (linear, kNN)
- Análise de resíduos sem padrões sistemáticos

### Excelente (para Publicação Q1):
- FE ≤ 1.5 para ≥67% das previsões
- GMFE < 1.5
- Supera ODE solver em ≥50% dos casos
- Validação em dados experimentais independentes
- Análise de resíduos robusta

### Ideal (SOTA):
- FE ≤ 1.25 para ≥67% das previsões
- GMFE < 1.25
- Supera ODE solver consistentemente
- Validação em múltiplos datasets experimentais
- Generalização a novas classes químicas

---

## 📚 Referências Científicas

1. **FDA Guidance for Industry:** "Physiologically Based Pharmacokinetic Analyses — Format and Content" (2018)
2. **EMA Guideline:** "Guideline on the reporting of physiologically based pharmacokinetic (PBPK) modelling and simulation" (2018)
3. **Rowland & Tozer:** "Clinical Pharmacokinetics and Pharmacodynamics" (5th ed.) - Padrão ouro em PK
4. **Sheiner & Beal:** "Evaluation of methods for estimating population pharmacokinetic parameters" (1980) - Métricas de validação
5. **Bergstrand et al.:** "Prediction-Corrected Visual Predictive Checks for Diagnosing Nonlinear Mixed-Effects Models" (2011)

---

## ✅ Checklist Pré-Publicação

- [ ] FE calculado e reportado (não apenas R²)
- [ ] GMFE < 2.0 (idealmente < 1.5)
- [ ] % dentro de 2x reportado
- [ ] Comparação com baselines (linear, kNN, RF, ODE)
- [ ] Análise de resíduos realizada
- [ ] Validação externa (dados experimentais ou LOCO)
- [ ] Limitações do dataset sintético reconhecidas
- [ ] Visualizações científicas adequadas
- [ ] Métodos descritos com transparência total
- [ ] Código e dados disponibilizados (se possível)

---

**"Rigorous science. Honest results. Real impact."**

**Última atualização:** 2025-11-17


