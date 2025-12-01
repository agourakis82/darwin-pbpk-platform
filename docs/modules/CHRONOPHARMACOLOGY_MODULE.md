# Módulo: Chronopharmacology (Cronofarmacocinética)

## Status: 🔴 A Implementar

## Problema que Resolve
Modelos tratam TODOS os parâmetros como constantes.
Realidade: TUDO varia ao longo das 24 horas.

## Componentes

### 1. Variação Circadiana de Parâmetros ADME

**Absorção:**
- Esvaziamento gástrico: mais rápido de manhã
- Motilidade intestinal: variação circadiana
- Fluxo sanguíneo GI: pico pós-prandial
- pH gástrico: variação diurna

**Distribuição:**
- Fluxo sanguíneo tecidual: variação circadiana
- Hematócrito: pico matinal
- Proteínas plasmáticas: variação diurna
- Permeabilidade vascular: influenciada por cortisol

**Metabolismo:**
- CYP3A4: expressão máxima à noite (em roedores)
- CYP2D6: menos estudado
- Atividade de UGTs: ritmo circadiano
- Fluxo hepático: variação com viscosidade

**Excreção:**
- GFR: pico durante o dia (posição ereta)
- Fluxo sanguíneo renal: variação circadiana
- pH urinário: variação diurna
- Transportadores renais: expressão rítmica

### 2. Ritmos Biológicos Relevantes

**Hormônios:**
| Hormônio | Pico | Impacto PK |
|----------|------|------------|
| Cortisol | 6-8am | Indução CYP, permeabilidade |
| Melatonina | 2-4am | Sono, fluxo glinfático |
| GH | Sono profundo | Metabolismo |
| Insulina | Pós-refeições | Transporte, metabolismo |

**Sinais Moleculares:**
- CLOCK/BMAL1 - master regulators
- PER1/2, CRY1/2 - feedback negativo
- REV-ERBα - ligação com metabolismo

### 3. Cronotoxicologia

**Exemplos documentados:**
| Droga | Melhor horário | Razão |
|-------|----------------|-------|
| Estatinas (curta ação) | Noite | Síntese colesterol noturna |
| Corticoides | Manhã | Mimetizar ritmo natural |
| Quimioterapia (alguns) | Tarde | Menor toxicidade medular |
| Anti-hipertensivos | Noite (alguns) | Controlar dipping noturno |

### 4. Modelo de Oscilação

**Função genérica de ritmo circadiano:**
```
P(t) = P_mesor + P_amplitude × cos(2π(t - φ)/24)

Onde:
- P_mesor = valor médio do parâmetro
- P_amplitude = amplitude de oscilação
- φ = acrofase (hora do pico)
- t = hora do dia (0-24)
```

**Aplicação a parâmetros PK:**
```
CL(t) = CL_mesor × [1 + A_CL × cos(2π(t - φ_CL)/24)]
Vd(t) = Vd_mesor × [1 + A_Vd × cos(2π(t - φ_Vd)/24)]
ka(t) = ka_mesor × [1 + A_ka × cos(2π(t - φ_ka)/24)]
```

### 5. Interação com Sono/Vigília

**Sistema Glinfático:**
- Clearance cerebral 60% maior durante sono
- Espaço extracelular aumenta 60%
- Implicações para drogas CNS

**Variabilidade de Resposta:**
- PA mais baixa à noite → risco de hipotensão
- Coagulabilidade maior de manhã → risco trombótico
- Broncoconstrição pior à noite → asma noturna

## Dados Necessários
- [ ] Ritmos circadianos de CYPs em humanos
- [ ] Variação de fluxo hepático ao longo do dia
- [ ] Ritmo de GFR em humanos
- [ ] Acrofases de proteínas plasmáticas

## Prioridade: MÉDIA-ALTA
**Razão:** Fácil de implementar, alto impacto clínico

## Referências Chave
1. Ruben et al. (2019) Science - Circadian medicine review
2. Dallmann et al. (2014) - Chronopharmacokinetics
3. Levi & Schibler (2007) - Circadian timing in cancer

