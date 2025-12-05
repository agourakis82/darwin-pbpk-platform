# SAM-3 Comprehensive Test Plan - Leucócitos

**Data**: 2025-12-01  
**Status**: ✅ Script Criado - Executando Testes

---

## 🎯 OBJETIVO

Validar SAM-3 para segmentação de **TODOS os tipos de leucócitos**:
- Neutrófilos
- Linfócitos (T, B, NK)
- Monócitos
- Eosinófilos
- Basófilos

E condições patológicas:
- Leucemia
- Sepse

---

## 📋 SUÍTE DE TESTES

### 1. Subpopulações Normais

#### Neutrófilos
- Prompts: "neutrophils", "neutrophil white blood cells", "neutrophils with segmented nuclei"
- Diretório: `data/leukocytes/normal/neutrophils/`
- Validação: Detecta células com núcleos segmentados

#### Linfócitos
- Prompts: "lymphocytes", "lymphocyte white blood cells", "lymphocytes with round nuclei"
- Diretório: `data/leukocytes/normal/lymphocytes/`
- Validação: Detecta células com núcleos redondos

#### Monócitos
- Prompts: "monocytes", "monocyte white blood cells", "monocytes with kidney-shaped nuclei"
- Diretório: `data/leukocytes/normal/monocytes/`
- Validação: Detecta células grandes com núcleos em formato de rim

#### Eosinófilos
- Prompts: "eosinophils", "eosinophil white blood cells", "eosinophils with bilobed nuclei"
- Diretório: `data/leukocytes/normal/eosinophils/`
- Validação: Detecta células com grânulos alaranjados

#### Basófilos
- Prompts: "basophils", "basophil white blood cells"
- Diretório: `data/leukocytes/normal/basophils/`
- Validação: Detecta células raras com núcleos em S

### 2. Condições Patológicas

#### Leucemia (ALL)
- Prompts: "leukemia cells", "leukemia lymphocytes", "ALL acute lymphoblastic leukemia"
- Diretório: `data/leukocytes/leukemia/lymphocytes/`
- Validação: Detecta células anormais/blastos

#### Sepse
- Prompts: "abnormal neutrophils in sepsis", "toxic neutrophils"
- Validação: Detecta neutrófilos com morfologia alterada

---

## 📊 MÉTRICAS DE VALIDAÇÃO

Para cada tipo, medir:
1. **Número de células detectadas**
2. **Score médio de confiança**
3. **Taxa de sucesso** (células detectadas / células esperadas)
4. **Precisão relativa** (comparar com método atual)

---

## 🔄 FLUXO DE VALIDAÇÃO

```
1. Carregar Modelo SAM-3
   ↓
2. Para cada subpopulação:
   - Encontrar imagens de teste
   - Testar múltiplos prompts
   - Coletar resultados
   ↓
3. Para cada condição patológica:
   - Encontrar imagens de teste
   - Testar prompts específicos
   - Coletar resultados
   ↓
4. Calcular estatísticas agregadas
   ↓
5. Gerar relatório completo
   ↓
6. Validar precisão e cobertura
```

---

## ✅ CRITÉRIOS DE SUCESSO

### Mínimo Aceitável
- ✅ Detecta células em todas as subpopulações
- ✅ Score médio > 0.5
- ✅ Pelo menos 50% das células detectadas (vs. esperado)

### Ideal
- ✅ Score médio > 0.7
- ✅ Detecta 80%+ das células
- ✅ Distingue corretamente entre subpopulações
- ✅ Detecta células patológicas

---

## 📁 OUTPUT ESPERADO

1. **JSON com resultados detalhados**
   - Por subpopulação
   - Por imagem
   - Estatísticas agregadas

2. **Log completo**
   - Todos os testes
   - Prompts testados
   - Resultados

3. **Relatório de validação**
   - Resumo executivo
   - Comparação com método atual
   - Recomendações

---

**Script**: `analysis/fractal_poc/test_sam3_all_wbc_types.py`  
**Status**: ✅ Criado - Executando...

