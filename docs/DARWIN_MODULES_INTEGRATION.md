# Darwin: Integração de Módulos e Roadmap

## Arquitetura de Módulos

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DARWIN CORE ENGINE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   BLOOD     │  │   BBB/CNS   │  │   FETAL/    │  │   CHRONO    │            │
│  │  RHEOLOGY   │  │   MODULE    │  │  PLACENTAL  │  │   PHARM     │            │
│  │             │  │             │  │             │  │             │            │
│  │ • Viscosity │  │ • Regional  │  │ • Shunts    │  │ • Circadian │            │
│  │ • Rouleaux  │  │   BBB       │  │ • Transport │  │   rhythms   │            │
│  │ • RBC deform│  │ • Glymph.   │  │ • HbF/HbA   │  │ • Sleep     │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         │                │                │                │                   │
│         └────────────────┼────────────────┼────────────────┘                   │
│                          │                │                                     │
│                    ┌─────┴─────┐    ┌─────┴─────┐                               │
│                    │  FRACTAL  │    │  HELIO    │                               │
│                    │    PK     │    │  BIOLOGY  │                               │
│                    │           │    │(experim.) │                               │
│                    │ • k(t)    │    │ • Solar   │                               │
│                    │ • Anomal. │    │ • Geomagn │                               │
│                    │   diffus. │    │ • Schumann│                               │
│                    └───────────┘    └───────────┘                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Matriz de Dependências

| Módulo | Depende de | Fornece para |
|--------|-----------|--------------|
| Blood Rheology | - | Todos (fluxo é base) |
| BBB/CNS | Blood Rheology, Chrono | Fetal (imaturidade BBB) |
| Fetal/Placental | Blood, Chrono | - |
| Chronopharmacology | Blood Rheology | Todos |
| Fractal PK | Blood Rheology | Todos (modificador) |
| Heliobiology | Chrono | Todos (modificador) |

## Priorização (MoSCoW)

### MUST HAVE (MVP)
1. **Blood Rheology Base** - Viscosidade não-Newtoniana
2. **Chronopharmacology Base** - Ritmo circadiano de CL, Vd
3. **PBPK Core** - Integração com engine existente

### SHOULD HAVE (v1.0)
4. **BBB Heterogeneity** - Regionalização cerebral
5. **Fetal/Placental Base** - Shunts e transportadores
6. **Fractal PK** - k(t) para distribuição

### COULD HAVE (v1.x)
7. **Glymphatic System** - Clearance cerebral sono-dependente
8. **Advanced Fetal** - Maturação por idade gestacional
9. **Heliobiology** - Modo experimental

### WON'T HAVE (now)
- Integração com wearables (HRV em tempo real)
- Predição de eventos adversos por tempestade solar
- Personalização por cronótipo individual

## Roadmap de Implementação

### FASE 1: Fundação (Semanas 1-4)
```
□ Blood Rheology
  □ Implementar modelo Carreau-Yasuda
  □ Adicionar dependência de hematócrito
  □ Integrar variação circadiana básica
  □ Testes unitários

□ Chronopharmacology
  □ Função de oscilação circadiana genérica
  □ Aplicar a CL, Vd, ka
  □ Dados de acrofase para drogas comuns
  □ Testes de integração
```

### FASE 2: Barreiras (Semanas 5-8)
```
□ BBB/CNS
  □ Segmentação regional do cérebro
  □ Permeabilidades por região
  □ Transportadores por região
  □ Integração com PBPK core

□ Fetal/Placental
  □ Modelo de shunts fetais
  □ Transportadores placentários
  □ Ligação proteica diferencial
  □ Validação com dados clínicos
```

### FASE 3: Avançado (Semanas 9-12)
```
□ Fractal PK
  □ Implementar k(t) = k₀ × t^(-h)
  □ Difusão anômala para tecidos
  □ Comparar com modelo clássico

□ Glymphatic
  □ Clearance cerebral sono-dependente
  □ Integração com cronofarmacologia

□ Heliobiology (experimental)
  □ API para dados solares/geomagnéticos
  □ Modificador experimental de parâmetros
  □ Flag de "modo experimental"
```

## Métricas de Sucesso

| Módulo | Métrica | Target |
|--------|---------|--------|
| Blood Rheology | Predição de viscosidade | R² > 0.9 vs literatura |
| Chronopharm | Variação de AUC por horário | ±20% validado |
| BBB/CNS | Kp_brain regional | Validado vs PET data |
| Fetal | Razão feto/mãe | Validado vs cordocentese |
| Fractal | Fitting vs compartmental | AIC melhor em >50% casos |

## Documento Gerado: 2024-12-01

