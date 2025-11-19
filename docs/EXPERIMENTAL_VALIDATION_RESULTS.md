# Resultados da Validação Externa - DynamicPBPKGNN v4_compound

**Data:** 2025-11-17
**Modelo:** DynamicPBPKGNN v4_compound
**Dataset Experimental:** 150 compostos de dados clínicos reais
**Fontes:** real_clinical_pk_data.json, ULTIMATE_DATASET_v1, PKDB

---

## 📊 Métricas de Validação Externa

### Cmax (Concentração Máxima)

| Métrica | Valor | Critério de Aceitação | Status |
|---------|-------|----------------------|--------|
| **Fold Error (FE) médio** | 67.181 | ≤ 2.0 | ❌ FALHOU |
| **Fold Error (FE) mediano** | 19.286 | ≤ 2.0 | ❌ FALHOU |
| **Fold Error (FE) p67** | 70.635 | ≤ 2.0 | ❌ FALHOU |
| **Geometric Mean Fold Error (GMFE)** | 17.560 | < 2.0 | ❌ FALHOU |
| **% dentro de 2.0×** | 20.0% | ≥ 67% | ❌ FALHOU |

### AUC (Área Sob a Curva)

| Métrica | Valor | Critério de Aceitação | Status |
|---------|-------|----------------------|--------|
| **Fold Error (FE) médio** | 1838.489 | ≤ 2.0 | ❌ FALHOU |
| **Fold Error (FE) mediano** | 141.459 | ≤ 2.0 | ❌ FALHOU |
| **Fold Error (FE) p67** | 628.404 | ≤ 2.0 | ❌ FALHOU |
| **Geometric Mean Fold Error (GMFE)** | 155.421 | < 2.0 | ❌ FALHOU |
| **% dentro de 2.0×** | 0.0% | ≥ 67% | ❌ FALHOU |

---

## 🔍 Análise dos Resultados

### Problemas Identificados

1. **Conversão de Unidades:**
   - Dados experimentais: ng/mL (Cmax), ng·h/mL (AUC)
   - Modelo prevê: mg/L (assumido)
   - **Problema:** Fator de conversão depende da massa molar de cada composto
   - **Impacto:** Erros sistemáticos grandes (FE médio > 60)

2. **Generalização do Modelo:**
   - Modelo treinado em dados sintéticos (simulação determinística)
   - Dados experimentais têm variabilidade real (ruído, variabilidade inter-individual)
   - **Impacto:** Modelo pode não generalizar bem para dados experimentais

3. **Estimativas de Parâmetros:**
   - Clearance hepático/renal estimados (70%/30% do total)
   - Partition coefficients estimados a partir de Vd
   - **Impacto:** Incerteza propagada para previsões

### Pontos Positivos

1. **Validação Externa Implementada:**
   - ✅ Script funcional para validação em dados experimentais
   - ✅ 150 compostos validados
   - ✅ Métricas científicas calculadas (FE, GMFE)

2. **Infraestrutura Completa:**
   - ✅ Carregamento de múltiplas fontes de dados
   - ✅ Conversão para formato PBPK
   - ✅ Visualizações geradas

---

## 🔧 Próximos Passos para Melhorar Validação

### 1. Correção de Unidades (Prioritário)

**Implementar conversão adequada:**
- Obter massa molar de cada composto (SMILES → MW)
- Converter ng/mL → mg/L: `mg/L = (ng/mL) / (MW * 1000)`
- Converter ng·h/mL → mg·h/L: `mg·h/L = (ng·h/mL) / (MW * 1000)`

**Script:** `scripts/convert_experimental_units.py`

### 2. Refinamento de Parâmetros

**Melhorar estimativas:**
- Usar dados experimentais de clearance hepático/renal quando disponíveis
- Usar dados experimentais de Kp quando disponíveis
- Implementar estimativas mais sofisticadas (ex: QSAR para Kp)

### 3. Re-treino com Dados Experimentais

**Estratégias:**
- Fine-tuning do modelo em dados experimentais
- Transfer learning: treinar em sintéticos, ajustar em experimentais
- Ensemble: combinar modelo sintético com modelo experimental

### 4. Análise de Casos Específicos

**Identificar:**
- Compostos com melhor/pior desempenho
- Padrões de erro (subestimação/sobrestimacao)
- Dependência de propriedades moleculares

---

## 📈 Comparação: Sintético vs. Experimental

| Métrica | Dataset Sintético (v4) | Dataset Experimental |
|---------|------------------------|---------------------|
| **FE médio (Cmax)** | 1.000 | 67.181 |
| **GMFE (Cmax)** | 1.000 | 17.560 |
| **% dentro de 2.0× (Cmax)** | 99.999% | 20.0% |
| **FE médio (AUC)** | 1.000 | 1838.489 |
| **GMFE (AUC)** | 1.000 | 155.421 |
| **% dentro de 2.0× (AUC)** | 99.999% | 0.0% |

**Conclusão:** O modelo tem excelente desempenho em dados sintéticos, mas falha em dados experimentais. Isso indica:
1. Problema de unidades (principal)
2. Necessidade de ajuste/fine-tuning em dados experimentais
3. Limitações do dataset sintético para generalização

---

## ✅ Conclusão

A validação externa foi **implementada com sucesso**, mas os resultados mostram que o modelo precisa de **ajustes significativos** para funcionar em dados experimentais reais:

1. **Correção de unidades** é crítica (FE médio reduziria drasticamente)
2. **Fine-tuning** em dados experimentais pode melhorar generalização
3. **Estimativas de parâmetros** precisam ser refinadas

**Status:** Validação externa funcional, mas resultados indicam necessidade de melhorias antes de publicação científica.

---

**"Rigorous science. Honest results. Real impact."**

**Última atualização:** 2025-11-17


