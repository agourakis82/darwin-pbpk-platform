# Resumo Executivo - Análise de Discrepância de Escala

**Data:** 2025-11-17  
**Status:** Investigação Concluída

## 🎯 Problema Identificado

O modelo Dynamic GNN prevê concentrações **~9× maiores** que observadas em dados experimentais reais.

## 📊 Achados Principais

1. **Doses experimentais:** Maioria na faixa razoável (mediana: 100 mg), mas há outliers (1 dose de 20,000 mg)
2. **Clearances:** Similares entre treino e experimental (média ~20 L/h)
3. **Concentrações do treino:** Razoáveis (Cmax média: 25.21 mg/L para doses 50-200 mg)
4. **Discrepância:** Sistêmica (correlação fraca com dose/clearance)

## 💡 Causas Prováveis

1. **Parâmetros estimados incorretos** (CL, Kp não medidos, apenas estimados)
2. **Problema de normalização** no modelo
3. **Modelo não generaliza** para dados experimentais (treinado apenas em sintéticos)

## 🔧 Ações Recomendadas

1. ✅ **Auditar dados experimentais** (doses, unidades, outliers)
2. ⏳ **Refinar estimativas de parâmetros** (prioritário)
3. ⏳ **Verificar normalização** no modelo
4. ⏳ **Fine-tuning** em dados experimentais
5. ⏳ **Calibrar escala** do modelo

## 📈 Próximo Passo Imediato

Refinar estimativas de parâmetros (CL hepático/renal, Kp) usando dados experimentais quando disponíveis.
