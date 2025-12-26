# Análise de Discrepância de Escala - Dynamic GNN PBPK

**Data:** 2025-11-17
**Autor:** AI Assistant + Dr. Sounio Agourakis
**Status:** Em Investigação

---

## 📊 Resumo Executivo

A validação externa do modelo Dynamic GNN em dados experimentais revelou uma **discrepância crítica de escala**:

- **Cmax previsto vs observado:** Razão média de **9.13×** (previsto muito maior)
- **AUC previsto vs observado:** Razão média de **0.47×** (previsto menor)
- **% dentro de 2.0×:** Apenas **0-10%** (muito abaixo do critério de 67%)

---

## 🔍 Análise Detalhada

### 1. Comparação de Doses

| Métrica | Dataset de Treino | Dados Experimentais |
|---------|-------------------|---------------------|
| Min | 50.00 mg | 0.10 mg |
| Max | 199.97 mg | **20,000.00 mg** |
| Média | 126.03 mg | 315.80 mg |

**Observação:** Doses experimentais têm faixa muito ampla, incluindo valores extremos (até 20,000 mg).

### 2. Comparação de Clearances

| Métrica | Dataset de Treino | Dados Experimentais |
|---------|-------------------|---------------------|
| Min | 0.07 L/h | 0.00 L/h |
| Max | 134.62 L/h | 441.00 L/h |
| Média | 23.14 L/h | 19.97 L/h |

**Observação:** Clearances são similares, mas dados experimentais têm valores extremos.

### 3. Concentrações do Dataset de Treino

- **Cmax (blood):** 10.00 - 39.99 mg/L (média: 25.21 mg/L)
- **AUC (blood):** 23.39 - 41.94 mg·h/L (média: 32.74 mg·h/L)

**Observação:** Concentrações do treino parecem razoáveis para doses de 50-200 mg.

### 4. Previsões vs Observados

| Métrica | Previsto | Observado | Razão |
|---------|----------|-----------|-------|
| Cmax (média) | 43.59 mg/L | 4.77 mg/L | **9.13×** |
| AUC (média) | 7.55 mg·h/L | 16.19 mg·h/L | **0.47×** |

**Observação:**
- Cmax previsto é ~9× maior que observado
- AUC previsto é ~2× menor que observado
- Sugere que o modelo prevê picos mais altos mas eliminação mais rápida

### 5. Análise de Escala Esperada

Para doses experimentais (média: 315.80 mg) e volume de sangue (5 L):
- **Concentração inicial esperada:** 63.16 mg/L
- **Cmax observado (média):** 4.77 mg/L
- **Razão (Cmax_obs / Conc_inicial):** 0.0756

**Interpretação:** Cmax observado é apenas 7.6% da concentração inicial esperada, o que é razoável devido a:
- Distribuição para tecidos (Kp > 1 em alguns órgãos)
- Clearance rápido
- Tempo de pico (Tmax) > 0

### 6. Correlações

- **Correlação dose vs razão Cmax:** -0.39 (fraca)
- **Correlação clearance vs razão Cmax:** -0.34 (fraca)

**Interpretação:** Correlações fracas sugerem que o problema é **sistêmico**, não dependente de dose ou clearance específicos.

---

## 🎯 Hipóteses sobre a Discrepância

### Hipótese 1: Doses Experimentais Incorretas
- **Evidência:** Faixa muito ampla (0.10 - 20,000 mg)
- **Ação:** Verificar unidades e conversões

### Hipótese 2: Parâmetros Estimados Incorretos
- **Evidência:** CL e Kp são estimados, não medidos
- **Ação:** Refinar estimativas usando dados experimentais quando disponíveis

### Hipótese 3: Problema de Normalização no Modelo
- **Evidência:** Modelo prevê concentrações consistentemente maiores
- **Ação:** Verificar normalização no forward pass

### Hipótese 4: Modelo Não Generaliza para Dados Experimentais
- **Evidência:** Alto R² em dados sintéticos, mas baixa performance em dados reais
- **Ação:** Fine-tuning em dados experimentais

### Hipótese 5: Conversão de Unidades Incorreta
- **Evidência:** Cmax observado pode estar em unidades diferentes
- **Ação:** Verificar conversão ng/mL → mg/L

---

## 💡 Recomendações Prioritárias

### 1. Verificar Doses Experimentais (Prioritário)
- [ ] Auditar conversão de unidades (mg, g, µg)
- [ ] Verificar se doses extremas (20,000 mg) são reais ou erros
- [ ] Filtrar outliers antes da validação

### 2. Refinar Estimativas de Parâmetros
- [ ] Usar dados experimentais de clearance quando disponíveis
- [ ] Melhorar estimativa de Kp usando Vd experimental
- [ ] Implementar separação CL hepático/renal mais precisa

### 3. Verificar Normalização no Modelo
- [ ] Auditar cálculo de concentração inicial (dose/volume)
- [ ] Verificar se há normalização implícita no forward pass
- [ ] Comparar com ODE solver tradicional

### 4. Fine-tuning em Dados Experimentais
- [ ] Criar dataset de fine-tuning com dados experimentais
- [ ] Treinar modelo com loss ponderada (mais peso para dados experimentais)
- [ ] Validar em conjunto de teste separado

### 5. Implementar Calibração de Escala
- [ ] Calibrar modelo usando fator de escala baseado em dados experimentais
- [ ] Implementar correção pós-processamento
- [ ] Validar calibração em conjunto independente

---

## 📈 Próximos Passos

1. **Auditar dados experimentais** (doses, unidades, outliers)
2. **Refinar estimativas de parâmetros** (CL, Kp)
3. **Verificar normalização** no modelo
4. **Implementar fine-tuning** em dados experimentais
5. **Calibrar escala** do modelo

---

## 📁 Arquivos Relacionados

- `scripts/investigate_scale_issue.py` - Script de investigação
- `scripts/analyze_scale_discrepancy.py` - Análise de discrepância
- `models/dynamic_gnn_v4_compound/scale_analysis/` - Resultados da análise
- `docs/EXPERIMENTAL_VALIDATION_RESULTS.md` - Resultados de validação

---

**Última atualização:** 2025-11-17


