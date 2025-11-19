# Validação Externa Final - DynamicPBPKGNN v4_compound

**Data:** 2025-11-17
**Modelo:** DynamicPBPKGNN v4_compound
**Dataset Experimental:** 150 compostos de dados clínicos reais
**Unidades:** Corrigidas (ng/mL → mg/L, ng·h/mL → mg·h/L)

---

## 📊 Resultados Finais (Unidades Corrigidas)

### Cmax (Concentração Máxima)

| Métrica | Valor | Critério | Status |
|---------|-------|----------|--------|
| **Fold Error (FE) médio** | 296.22 | ≤ 2.0 | ❌ FALHOU |
| **Fold Error (FE) mediano** | 67.50 | ≤ 2.0 | ❌ FALHOU |
| **Fold Error (FE) p67** | 162.52 | ≤ 2.0 | ❌ FALHOU |
| **Geometric Mean Fold Error (GMFE)** | 61.07 | < 2.0 | ❌ FALHOU |
| **% dentro de 2.0×** | 0.0% | ≥ 67% | ❌ FALHOU |

### AUC (Área Sob a Curva)

| Métrica | Valor | Critério | Status |
|---------|-------|----------|--------|
| **Fold Error (FE) médio** | 48.78 | ≤ 2.0 | ❌ FALHOU |
| **Fold Error (FE) mediano** | 16.67 | ≤ 2.0 | ❌ FALHOU |
| **Fold Error (FE) p67** | 48.01 | ≤ 2.0 | ❌ FALHOU |
| **Geometric Mean Fold Error (GMFE)** | 16.61 | < 2.0 | ❌ FALHOU |
| **% dentro de 2.0×** | 10.0% | ≥ 67% | ❌ FALHOU |

---

## 🔍 Análise Detalhada

### Comparação de Escalas

**Cmax:**
- Previsto: min=1.09, max=4000.0, **média=63.24 mg/L**
- Observado: min=0.001, max=25.0, **média=4.77 mg/L**
- **Razão média:** 13.3× (modelo superestima)

**AUC:**
- Previsto: min=4.11, max=213.59, **média=7.55 mg·h/L**
- Observado: min=0.02, max=90.0, **média=16.19 mg·h/L**
- **Razão média:** 0.47× (modelo subestima)

### Problemas Identificados

1. **Discrepância de Escala:**
   - Modelo prevê Cmax ~13× maior que observado
   - Modelo prevê AUC ~0.5× menor que observado
   - Sugere problema fundamental na escala das previsões

2. **Estimativas de Parâmetros:**
   - Clearance hepático/renal: estimado (70%/30% do total)
   - Partition coefficients: estimados a partir de Vd
   - **Impacto:** Incerteza propagada para previsões

3. **Dataset de Treino vs. Experimental:**
   - Treino: dados sintéticos (simulação determinística)
   - Validação: dados experimentais (variabilidade real)
   - **Impacto:** Modelo pode não generalizar bem

4. **Conversão de Unidades:**
   - ✅ Corrigida: ng/mL → mg/L (divisão por 1000)
   - ⚠️ Mas ainda há discrepância de escala

---

## 🎯 Conclusões

### ✅ Pontos Positivos

1. **Validação Externa Implementada:**
   - ✅ Script funcional para validação em dados experimentais
   - ✅ 150 compostos validados
   - ✅ Métricas científicas calculadas (FE, GMFE)
   - ✅ Conversão de unidades corrigida

2. **Infraestrutura Completa:**
   - ✅ Carregamento de múltiplas fontes de dados
   - ✅ Conversão para formato PBPK
   - ✅ Visualizações geradas

### ❌ Problemas Críticos

1. **Modelo não generaliza para dados experimentais:**
   - FE médio > 50 (muito acima do aceitável)
   - Apenas 0-10% das previsões dentro de 2.0×

2. **Discrepância de escala:**
   - Cmax previsto ~13× maior que observado
   - Sugere problema na normalização/escala do modelo

3. **Estimativas de parâmetros imprecisas:**
   - CL hepático/renal estimados (não medidos)
   - Kp estimados (não medidos)

---

## 🔧 Próximos Passos Críticos

### 1. Investigar Discrepância de Escala (Prioritário)

**Hipóteses:**
- Modelo foi treinado com normalização diferente?
- Doses estão sendo interpretadas incorretamente?
- Concentrações do dataset de treino estão em escala diferente?

**Ações:**
- Verificar normalização no dataset de treino
- Comparar escala de concentrações (treino vs. experimental)
- Verificar se há fator de conversão faltando

### 2. Refinar Estimativas de Parâmetros

**Melhorar:**
- Usar dados experimentais de clearance hepático/renal quando disponíveis
- Usar dados experimentais de Kp quando disponíveis
- Implementar estimativas mais sofisticadas (QSAR, ML)

### 3. Fine-tuning em Dados Experimentais

**Estratégias:**
- Transfer learning: ajustar modelo treinado em sintéticos
- Re-treino parcial: treinar apenas camadas finais em experimentais
- Ensemble: combinar modelo sintético com modelo experimental

### 4. Análise de Casos Específicos

**Identificar:**
- Compostos com melhor/pior desempenho
- Padrões de erro (subestimação/sobrestimacao)
- Dependência de propriedades moleculares

---

## 📈 Comparação: Antes vs. Depois da Correção de Unidades

| Métrica | Antes (unidades erradas) | Depois (unidades corretas) | Melhoria |
|---------|-------------------------|----------------------------|----------|
| **Cmax FE médio** | 67.18 | 296.22 | ❌ Piorou |
| **Cmax GMFE** | 17.56 | 61.07 | ❌ Piorou |
| **AUC FE médio** | 1838.49 | 48.78 | ✅ Melhorou 37.7× |
| **AUC GMFE** | 155.42 | 16.61 | ✅ Melhorou 9.4× |
| **AUC % dentro de 2.0×** | 0.0% | 10.0% | ✅ Melhorou |

**Observação:** A correção de unidades melhorou AUC significativamente, mas Cmax piorou. Isso sugere que há problemas adicionais além da conversão de unidades.

---

## ✅ Status Final

**Validação Externa:** ✅ Implementada e funcional
**Conversão de Unidades:** ✅ Corrigida
**Resultados:** ❌ Modelo não atende critérios científicos (FE > 2.0)
**Recomendação:** Necessário investigar discrepância de escala e refinar parâmetros antes de publicação

---

**"Rigorous science. Honest results. Real impact."**

**Última atualização:** 2025-11-17


