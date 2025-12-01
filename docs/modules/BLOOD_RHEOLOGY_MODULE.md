# Módulo: Blood Rheology (Reologia Sanguínea)

## Status: 🔴 A Implementar

## Problema que Resolve
Modelos PBPK tradicionais tratam sangue como fluido Newtoniano homogêneo.
Realidade: Sangue é não-Newtoniano, heterogêneo, e varia temporalmente.

## Componentes

### 1. Viscosidade Dependente de Shear Rate
**Equação proposta (Carreau-Yasuda):**
```
η(γ̇) = η_∞ + (η_0 - η_∞) × [1 + (λγ̇)^a]^((n-1)/a)

Onde:
- η_∞ = viscosidade em shear infinito (~3.5 cP)
- η_0 = viscosidade em shear zero (~50-60 cP)
- λ = tempo de relaxação
- a = parâmetro de transição
- n = índice de shear-thinning
- γ̇ = shear rate (varia por vaso)
```

**Referência:** Nader et al., Frontiers in Physiology, 2019

### 2. Efeito do Hematócrito
```
η = η_plasma × (1 + 2.5×Hct + 7.35×Hct²)

Variação diurna do Hct: ±5-10%
Pico: manhã (6-8am)
```

### 3. Formação de Rouleaux (Agregação Eritrocitária)
- Ocorre em baixo shear (<1 s⁻¹)
- Depende de fibrinogênio e globulinas
- Aumenta viscosidade dramaticamente
- Reversível em alto shear

### 4. Deformabilidade do Eritrócito
**Fatores que afetam:**
- pH (↓pH → ↓deformabilidade)
- ATP intracelular
- Ca²⁺ intracelular
- NO (aumenta deformabilidade via S-nitrosilação)
- Idade do RBC

**Impacto:** 15% ↓deformabilidade → 75% ↑resistência ao fluxo

### 5. Variação Circadiana
| Parâmetro | Pico | Variação |
|-----------|------|----------|
| Viscosidade | 6-8am | 10-20% |
| Hematócrito | manhã | 5-10% |
| PAI-1 | 6-8am | 50%+ |
| Agregação plaquetária | manhã | significativa |

### 6. Efeitos de Campo Magnético (Experimental)
- Hemoglobina: paramagnética (desoxi) vs diamagnética (oxi)
- Eritrócitos se orientam em campos magnéticos
- Força de Lorentz em íons sanguíneos
- Base do fMRI (BOLD signal)

## Impacto em Farmacocinética

1. **Fluxo hepático** varia com viscosidade → Clearance varia
2. **Distribuição tecidual** depende de fluxo microvascular
3. **Ligação eritrocitária** afeta concentração plasmática
4. **Hora do dia** afeta todos os parâmetros

## Dados Necessários
- [ ] Shear rates por tipo de vaso (literatura)
- [ ] Parâmetros Carreau-Yasuda para sangue humano
- [ ] Variação circadiana de proteínas plasmáticas
- [ ] Efeito de temperatura em viscosidade

## Prioridade: ALTA
**Razão:** Afeta TODOS os outros módulos (fluxo = base de tudo)

## Referências Chave
1. Nader et al. (2019) Front Physiol - Blood Rheology Review
2. Lanotte et al. (2016) PNAS - RBC shape transitions
3. Baskurt & Meiselman (2003) - Hemorheology fundamentals

