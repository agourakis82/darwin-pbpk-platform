# Descobertas da Investigação - Problema de Escala Cmax

**Data:** 2025-11-18
**Autor:** AI Assistant + Dr. Sounio Agourakis
**Status:** ✅ Investigação Completa

---

## 🎯 Resumo Executivo

A investigação detalhada do problema de escala do Cmax revelou um **problema crítico e sistemático**: o modelo está prevendo Cmax **~100-300× maior** que os valores observados experimentalmente.

### Descobertas Principais:

1. **Razão média (pred/obs): 289.88×** (deveria ser ~1.0×)
2. **Razão mediana: 100.00×**
3. **Cmax previsto médio: 26.17 mg/L**
4. **Cmax observado médio: 3.64 mg/L**
5. **Resíduo % médio: 28,887%** (extremamente alto!)

---

## 📊 Análise Detalhada

### 1. Problema de Escala por Faixa de Cmax Observado

| Faixa Cmax Obs (mg/L) | Razão Média | Razão Mediana | N Compostos | Interpretação |
|----------------------|-------------|---------------|-------------|---------------|
| **< 0.01** | **762.97×** | 964.30× | 3 | ⚠️ **CRÍTICO** - Modelo prevê ~760× maior |
| 0.01 - 0.1 | 98.21× | 98.21× | 2 | ⚠️ Muito alto |
| 0.1 - 1.0 | 57.09× | 57.09× | 2 | ⚠️ Alto |
| 1.0 - 10 | 6.15× | 6.15× | 1 | ⚠️ Moderado |
| **> 10** | **3.20×** | 3.20× | 1 | ✅ Melhor (mas ainda alto) |

**Conclusão:** O problema é **muito mais grave para Cmax baixos** (< 0.01 mg/L), onde a razão chega a **762×**!

### 2. Problema por Faixa de Dose

| Faixa Dose (mg) | Razão Média | Razão Mediana | N Compostos |
|----------------|-------------|---------------|-------------|
| **< 50** | **416.59×** | 201.95× | 6 | ⚠️ **CRÍTICO** |
| 50 - 100 | - | - | 0 | - |
| 100 - 200 | 6.15× | 6.15× | 1 | ⚠️ Moderado |
| 200 - 500 | 51.60× | 51.60× | 2 | ⚠️ Alto |
| > 500 | - | - | 0 | - |

**Conclusão:** Doses baixas (< 50 mg) têm razão média de **416×**, indicando que o problema é mais grave para doses pequenas.

### 3. Compostos Mais Problemáticos

| Composto | Dose (mg) | Cmax Pred (mg/L) | Cmax Obs (mg/L) | Razão |
|----------|-----------|------------------|-----------------|-------|
| **Warfarin** | 5.0 | 1.19 | 0.0011 | **1,080×** |
| **Digoxin** | 0.25 | 1.16 | 0.0012 | **964×** |
| **Atorvastatin** | 10.0 | 2.00 | 0.0082 | **244×** |
| **Propranolol** | 40.0 | 8.00 | 0.0500 | **160×** |
| **Metformin** | 500.0 | 100.00 | 1.0000 | **100×** |

**Padrão identificado:** Compostos com **doses muito baixas** e **Cmax observado muito baixo** (< 0.01 mg/L) são os mais problemáticos.

### 4. Compostos com Melhor Ajuste

| Composto | Dose (mg) | Cmax Pred (mg/L) | Cmax Obs (mg/L) | Razão |
|----------|-----------|------------------|-----------------|-------|
| **Ibuprofen** | 400.0 | 80.00 | 25.00 | **3.2×** |
| **Caffeine** | 200.0 | 40.00 | 6.50 | **6.2×** |
| Rivaroxaban | 10.0 | 2.00 | 0.141 | 14.2× |

**Conclusão:** Compostos com **doses altas** e **Cmax observado alto** (> 1.0 mg/L) têm melhor ajuste, mas ainda estão longe do ideal (razão deveria ser ~1.0×).

---

## 📈 Análise de Resíduos

### Estatísticas de Resíduos (Cmax)

- **Resíduo médio:** 22.53 mg/L (muito alto!)
- **Resíduo mediano:** 1.99 mg/L
- **Desvio padrão:** 34.38 mg/L (alta variabilidade)
- **Resíduo % médio:** 28,887% (extremamente alto!)
- **Resíduo % mediano:** 9,900%

### Estatísticas de Resíduos (AUC)

- **Resíduo médio:** -10.55 mg·h/L (negativo = subestimação)
- **Resíduo mediano:** 2.05 mg·h/L
- **Desvio padrão:** 28.78 mg·h/L
- **Resíduo % médio:** 2,546%
- **Resíduo % mediano:** 1,181%

### Testes Estatísticos

1. **Teste de Normalidade (Shapiro-Wilk):**
   - Cmax: W=0.712, p=0.002 ⚠️ **Resíduos NÃO são normais**
   - AUC: W=0.536, p<0.001 ⚠️ **Resíduos NÃO são normais**

   **Interpretação:** Distribuição não-normal indica problemas estruturais no modelo.

2. **Teste de Viés (t-test):**
   - Cmax: t=1.97, p=0.085 ✅ **Sem viés significativo** (mas resíduo médio é enorme!)
   - AUC: t=-1.10, p=0.304 ✅ **Sem viés significativo**

   **Interpretação:** Embora não haja viés estatisticamente significativo, os resíduos são enormes em magnitude absoluta.

---

## 🔍 Interpretação Científica

### Por que o problema é tão grave?

1. **Problema de escala não-linear:**
   - O problema é **muito mais grave para Cmax baixos** (< 0.01 mg/L)
   - Razão média de 762× para Cmax < 0.01 mg/L
   - Razão média de apenas 3.2× para Cmax > 10 mg/L
   - Indica que o problema **não é apenas de escala linear** (um fator constante)

2. **Doses baixas são mais problemáticas:**
   - Doses < 50 mg: razão média de 416×
   - Doses 200-500 mg: razão média de 51.6×
   - Sugere que o modelo pode ter problemas com **normalização de dose**

3. **Resíduos não-normais:**
   - Distribuição não-normal indica **heterocedasticidade** ou **padrões sistemáticos**
   - Pode indicar que o modelo não está capturando corretamente a variabilidade

### Possíveis Causas

1. **Problema de normalização:**
   - O modelo pode estar usando normalização incorreta
   - Volume de distribuição pode estar errado
   - Unidades podem estar inconsistentes

2. **Problema de parâmetros estimados:**
   - CL e Kp estimados podem estar incorretos
   - Especialmente para compostos com doses baixas e Cmax baixos

3. **Problema estrutural do modelo:**
   - O modelo pode não estar aprendendo corretamente a relação dose-Cmax
   - Pode haver vazamento de dados ou overfitting no treino sintético

4. **Problema de dados experimentais:**
   - Alguns valores observados podem estar incorretos
   - Unidades podem estar inconsistentes
   - Parâmetros estimados (CL, Kp) podem estar errados

---

## 🚀 Recomendações Imediatas

### 1. Verificar Normalização e Unidades (ALTA PRIORIDADE)

- [ ] Verificar se o modelo está usando volume de distribuição correto
- [ ] Verificar se as unidades estão consistentes (mg/L vs ng/mL)
- [ ] Comparar normalização do modelo com ODE solver
- [ ] Verificar se há problema de escala na entrada (dose)

### 2. Refinar Parâmetros Experimentais (ALTA PRIORIDADE)

- [ ] Validar CL e Kp estimados com literatura
- [ ] Re-estimar parâmetros para compostos problemáticos (Warfarin, Digoxin, Atorvastatin)
- [ ] Usar múltiplas fontes de dados para validar parâmetros

### 3. Análise por Composto (MÉDIA PRIORIDADE)

- [ ] Investigar por que Warfarin e Digoxin têm razão > 1000×
- [ ] Verificar se há problema específico com compostos de baixa dose
- [ ] Analisar se há padrão molecular (estrutura química)

### 4. Ajustar Modelo (LONGO PRAZO)

- [ ] Considerar normalização adaptativa por faixa de Cmax
- [ ] Treinar modelo separado para Cmax baixos vs altos
- [ ] Implementar correção pós-processamento baseada em dose/Cmax observado

---

## 📁 Arquivos Gerados

- `models/dynamic_gnn_v4_compound/investigation/cmax_scale_investigation.png` - Visualizações
- `models/dynamic_gnn_v4_compound/investigation/cmax_scale_investigation.json` - Estatísticas
- `models/dynamic_gnn_v4_compound/investigation/cmax_scale_investigation.csv` - Dados brutos
- `models/dynamic_gnn_v4_compound/investigation/residuals_analysis.png` - Análise de resíduos
- `models/dynamic_gnn_v4_compound/investigation/residuals_analysis.json` - Estatísticas de resíduos
- `models/dynamic_gnn_v4_compound/investigation/residuals_analysis.csv` - Dados brutos

---

## ✅ Conclusão

A investigação revelou que o problema de escala do Cmax é **sistemático e não-linear**:

- **Muito mais grave para Cmax baixos** (< 0.01 mg/L): razão ~760×
- **Melhor para Cmax altos** (> 10 mg/L): razão ~3×
- **Resíduos não-normais** indicam problemas estruturais
- **Doses baixas são mais problemáticas** (razão ~416×)

**Próximos passos críticos:**
1. Verificar normalização e unidades
2. Refinar parâmetros experimentais
3. Investigar compostos específicos (Warfarin, Digoxin)

---

**Última atualização:** 2025-11-18

