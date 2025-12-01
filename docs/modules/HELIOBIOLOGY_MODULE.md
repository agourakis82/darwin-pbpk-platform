# Módulo: Heliobiology (Experimental)

## Status: 🟡 Pesquisa/Experimental

## ⚠️ NOTA IMPORTANTE
Este módulo é EXPERIMENTAL e baseado em literatura emergente.
Não é mainstream, mas há evidências crescentes.
Implementar como feature opcional/"advanced mode".

## Problema que Resolve
Modelos ignoram completamente influências geofísicas/heliofísicas.
Evidências sugerem correlações com eventos cardiovasculares e função autonômica.

## Base Histórica

### Alexander Chizhevsky (1897-1964)
- Pioneiro russo da heliobiologia
- Observou correlações entre ciclo solar e eventos históricos/biológicos
- Trabalho suprimido durante era soviética
- Reavaliado com dados modernos

## Componentes

### 1. Efeitos Geomagnéticos Documentados

**Paper HeartMath/NASA (2017):**
- 31 dias de monitoramento de HRV
- Correlações significativas encontradas:

| Fator | Correlação com HRV | p-valor |
|-------|-------------------|---------|
| Velocidade vento solar | Negativa | <0.001 |
| Índice Kp (perturbação) | Negativa | <0.001 |
| Raios cósmicos | Positiva | <0.001 |
| Ressonância de Schumann | Positiva | <0.01 |

**Achado surpreendente:**
- Participantes em locais DIFERENTES
- HRV sincronizada com período de ~2.5 dias
- Sincronização com campo geomagnético terrestre

### 2. Tempestades Geomagnéticas e Saúde

**Meta-análises mostram aumento durante tempestades de:**
- Infarto do miocárdio (+10-15%)
- AVC (+5-10%)
- Arritmias cardíacas
- Internações psiquiátricas
- Suicídios

**Mecanismos propostos:**
- Alteração da melatonina (pineal sensível a campos magnéticos)
- Efeito em sistema nervoso autônomo
- Alteração da coagulabilidade sanguínea
- Estresse oxidativo

### 3. Hemoglobina e Magnetismo

**Fatos físicos estabelecidos:**
- Deoxihemoglobina é PARAMAGNÉTICA
- Oxihemoglobina é DIAMAGNÉTICA
- Base do fMRI (BOLD signal)
- ~10²² átomos de Fe por litro de sangue

**Implicação teórica:**
- Campos magnéticos variáveis deveriam afetar sangue
- Força de Lorentz em íons sanguíneos
- Orientação de eritrócitos em campos magnéticos (demonstrado)

### 4. Ressonância de Schumann

**Frequência fundamental: 7.83 Hz**
- Ressonância eletromagnética Terra-ionosfera
- Frequência próxima a ondas α cerebrais (8-12 Hz)
- Correlações com estados mentais/meditativos
- Possível sincronização biológica

### 5. Ciclo Solar (~11 anos)

**Observações históricas:**
- Correlações com epidemias (Chizhevsky)
- Eventos cardiovasculares
- Atividade solar → partículas energéticas → efeitos biológicos

## Modelo Proposto (Experimental)

### Índice de Perturbação Heliobiológica
```
HBI(t) = w₁ × Kp(t) + w₂ × Dst(t) + w₃ × F10.7(t) + w₄ × CR(t)

Onde:
- Kp = índice de perturbação geomagnética
- Dst = disturbance storm time
- F10.7 = fluxo de rádio solar
- CR = raios cósmicos
- w_i = pesos a calibrar
```

### Modificador de Parâmetros PK
```
CL_adj = CL_base × (1 + α_HBI × (HBI - HBI_mean))
HRV_adj = HRV_base × (1 + β_HBI × (HBI - HBI_mean))
```

## Dados Necessários
- [ ] Séries temporais de Kp, Dst, F10.7 (disponíveis publicamente)
- [ ] Dados clínicos com timestamp para correlacionar
- [ ] Parâmetros PK medidos durante eventos geomagnéticos
- [ ] HRV durante tempestades solares

## Prioridade: BAIXA (mas fascinante)
**Razão:** Especulativo, mas pode explicar variabilidade "inexplicável"

## Como Implementar
1. Feature OPCIONAL - desativada por padrão
2. Modo "researcher" ou "experimental"
3. Clara indicação de status não-validado
4. Permitir usuários contribuírem dados para validação

## Referências Chave
1. McCraty et al. (2017) Int J Environ Res Public Health - HeartMath
2. Alabdulgader et al. (2018) - HRV and geomagnetic activity
3. Stoupel (vários) - Heliobiology clinical correlations
4. Palmer et al. (2006) - Geomagnetic storms and health

