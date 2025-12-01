# Módulo: Blood-Brain Barrier & CNS

## Status: 🔴 A Implementar

## Problema que Resolve
Modelos tratam BBB como barreira uniforme com permeabilidade fixa.
Realidade: BBB é heterogênea, dinâmica, e varia por região cerebral.

## Componentes

### 1. Heterogeneidade Regional da BBB
**Descoberta chave (Nature Neuroscience 2024):**
- "Hundreds of molecular differences" entre regiões
- Células endoteliais variam por região
- Pericitos têm cobertura diferente
- Astrócitos têm fenótipos regionais

**Regiões com BBB "vazante" (Órgãos Circunventriculares):**
- Área postrema
- Eminência mediana
- Órgão subfornical
- Neurohipófise

### 2. Transportadores por Região
| Transportador | Córtex | Hipocampo | Cerebelo | CVOs |
|---------------|--------|-----------|----------|------|
| P-gp (ABCB1) | ++++ | +++ | +++ | + |
| BCRP (ABCG2) | ++++ | +++ | ++++ | + |
| MRP1 | ++ | ++ | ++ | + |
| OATP1A2 | ++ | ++ | + | +++ |
| LAT1 | +++ | +++ | +++ | +++ |

### 3. Sistema Glinfático (Descoberto 2012!)
**Função:** Clearance de resíduos cerebrais via CSF

**Fluxo:**
1. CSF entra pelo espaço periarterial (AQP4)
2. Atravessa parênquima
3. Coleta resíduos (β-amiloide, tau, etc.)
4. Drena pelo espaço perivenoso
5. Conecta com linfáticos meníngeos

**Controle Circadiano:**
- Influxo MÁXIMO durante SONO
- Espaço extracelular aumenta 60% no sono
- Clearance de β-amiloide 60% maior à noite
- Norepinefrina (acordado) REDUZ fluxo

### 4. Acoplamento Neurovascular
- Fluxo cerebral regional ajusta-se à atividade neural
- Heterogêneo por região
- Afetado por idade, doença, medicamentos

### 5. Permeabilidade Dinâmica
**Fatores que alteram BBB:**
- Inflamação → ↑permeabilidade
- Hipóxia → ↑permeabilidade
- Estresse oxidativo → ↑permeabilidade
- Idade → ↑permeabilidade basal
- Doenças (Alzheimer, MS, etc.)

## Impacto em Farmacocinética Cerebral

1. **Kp_brain** deveria ser região-específico
2. **Clearance cerebral** varia com sono/vigília
3. **Vd_brain** muda ao longo do dia
4. **Eficácia** de drogas CNS depende de timing

## Modelo Proposto

```
Para cada região cerebral r:
  - PS_r(t) = permeabilidade superfície (tempo-dependente)
  - fu_brain_r = fração livre no cérebro (região-específica)
  - CLint_brain_r = clearance intrínseco cerebral
  - Q_brain_r(t) = fluxo sanguíneo cerebral (circadiano)

Transferência materno-fetal considera:
  Kp_brain_r = (fu_plasma / fu_brain_r) × 
               (1 + efflux_ratio_r) × 
               f_glymphatic(t_sleep)
```

## Dados Necessários
- [ ] Expressão de transportadores por região (scRNA-seq)
- [ ] Permeabilidade por região (PET data)
- [ ] Ritmo circadiano do sistema glinfático
- [ ] Parâmetros de acoplamento neurovascular

## Prioridade: ALTA
**Razão:** Drogas CNS são área de maior necessidade não atendida

## Referências Chave
1. Pfau et al. (2024) Nature Neuroscience - BBB heterogeneity
2. Iliff et al. (2012) Science Translational Medicine - Glymphatic
3. Xie et al. (2013) Science - Sleep and glymphatic clearance

