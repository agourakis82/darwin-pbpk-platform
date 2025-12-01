# Módulo: Fetal-Placental Pharmacokinetics

## Status: 🔴 A Implementar

## Problema que Resolve
Modelos tratam feto como "mais um compartimento".
Realidade: Circulação fetal é RADICALMENTE diferente do adulto.

## Componentes

### 1. Shunts Fetais Únicos

**Ductus Venosus:**
- Conecta veia umbilical → veia cava inferior
- Desvia ~50% do sangue oxigenado do fígado
- Afeta primeira passagem hepática fetal

**Foramen Ovale:**
- Shunt atrial direito → esquerdo
- ~33% do retorno venoso
- Direciona sangue oxigenado ao cérebro/coração

**Ductus Arteriosus:**
- Conecta artéria pulmonar → aorta
- ~90% do output ventricular direito
- Desvia sangue dos pulmões (não funcionais)

### 2. Hemodinâmica Fetal
| Parâmetro | Feto | Adulto |
|-----------|------|--------|
| PO₂ arterial | ~30 mmHg | ~100 mmHg |
| Hematócrito | 45-65% | 38-50% |
| Resistência pulmonar | ALTA | BAIXA |
| Resistência sistêmica | BAIXA | ALTA |
| Hemoglobina | HbF | HbA |

### 3. Placenta como Órgão Ativo

**Transportadores de INFLUXO (mãe→feto):**
- OATPs (organic anion)
- OCTs (organic cation)
- OATs
- LAT1, LAT2

**Transportadores de EFLUXO (feto→mãe, proteção):**
- P-glycoprotein (P-gp/ABCB1)
- BCRP (ABCG2)
- MRPs (ABCC family)

**Expressão varia com idade gestacional:**
| Transportador | 1º Tri | 2º Tri | 3º Tri |
|---------------|--------|--------|--------|
| P-gp | ++++ | +++ | ++ |
| BCRP | + | ++ | +++ |
| OCT3 | ++ | ++ | ++ |

### 4. Metabolismo Placentário
- CYP19 (aromatase) - alta expressão
- CYP1A1 - induzível
- UGTs - presentes
- SULTs - presentes
- Pode metabolizar drogas ANTES de chegarem ao feto

### 5. Ligação Proteica Diferencial
| Proteína | Feto/Mãe Ratio |
|----------|----------------|
| Albumina | 0.8 |
| α1-AGP | 0.3 |
| → Fração livre MAIOR no feto para drogas básicas

### 6. Hemoglobina Fetal (HbF)
- Maior afinidade por O₂ que HbA
- Curva de dissociação deslocada à esquerda
- Facilita transferência de O₂ da mãe
- Afeta entrega de O₂ aos tecidos fetais

## Fluxo Sanguíneo Placentário
```
Fluxo = f(idade_gestacional)
  12 semanas: ~50 mL/min
  20 semanas: ~200 mL/min
  Termo: ~500-700 mL/min

Afetado por:
  - Posição materna
  - Exercício
  - Estresse
  - Patologias (pré-eclâmpsia)
```

## Modelo Proposto
```
dC_feto/dt = (Q_placenta × fu_mãe × C_mãe / Kp_placenta) 
           - (Q_placenta × fu_feto × C_feto)
           - (CL_fetal × C_feto)
           + Influx_transporters
           - Efflux_transporters

Com ajustes para:
  - Shunts (ductus venosus reduz primeira passagem)
  - Idade gestacional (todos os parâmetros mudam)
  - HbF vs HbA (afeta ligação de algumas drogas)
```

## Dados Necessários
- [ ] Expressão de transportadores por idade gestacional
- [ ] Fluxo placentário por idade gestacional
- [ ] Razões de ligação proteica feto/mãe
- [ ] Maturação de CYPs fetais

## Prioridade: ALTA
**Razão:** Área de maior risco (teratogenicidade) e menor conhecimento

## Referências Chave
1. Staud et al. (2021) PMC - Placental drug transport update
2. Myllynen & Vähäkangas (2013) - Placental transfer mechanisms
3. Abduljalil et al. - PBPK in pregnancy (vários papers)

