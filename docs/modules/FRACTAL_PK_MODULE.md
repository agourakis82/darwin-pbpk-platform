# Módulo: Fractal Pharmacokinetics

## Status: 🔴 A Implementar

## Problema que Resolve
Modelos usam rate constants FIXAS (k = constante).
Realidade: Constantes cinéticas variam com o tempo em meios heterogêneos.

## Base Teórica

### Cinética Fractal (Kopelman, 1988)
Em meios heterogêneos/fractais, constantes de taxa NÃO são constantes:
```
k(t) = k₀ × t^(-h)

Onde:
- k₀ = rate constant inicial
- h = expoente fractal (0 ≤ h < 1)
- h = 0 → cinética clássica
- h > 0 → cinética fractal (desaceleração com tempo)
```

### Por que isso importa?
Tecidos biológicos NÃO são:
- Homogêneos
- Bem misturados ("well-stirred")
- Uniformemente vascularizados

Tecidos biológicos SÃO:
- Heterogêneos em escala micro
- Fractais em estrutura (vasculatura, parênquima)
- Sujeitos a difusão anômala

## Componentes

### 1. Difusão Anômala
**Difusão Normal (Browniana):**
```
⟨r²⟩ = 6Dt  (MSD proporcional a t)
```

**Difusão Anômala:**
```
⟨r²⟩ = 6Dₐt^α

Onde:
- α < 1 → Subdifusão (movimento restrito)
- α > 1 → Superdifusão (movimento facilitado)
- α = 1 → Difusão normal
```

Subdifusão ocorre em:
- Citoplasma celular (crowding molecular)
- Matriz extracelular
- Tecido tumoral
- Barreiras biológicas

### 2. Modelo de Clearance Fractal
**Modelo Clássico:**
```
dC/dt = -k × C
Solução: C(t) = C₀ × e^(-kt)
```

**Modelo Fractal:**
```
dC/dt = -k₀ × t^(-h) × C
Solução: C(t) = C₀ × exp(-k₀ × t^(1-h) / (1-h))
```

### 3. Distribuição Fractal de Drogas
**Modelo Macheras (1996):**
Aplicou cinética fractal à distribuição de cálcio.
Mostrou que modelo fractal descreve melhor dados experimentais.

**Implicações:**
- "Constantes" de distribuição mudam com tempo
- Fases iniciais são mais rápidas
- Equilíbrio é mais lento que previsto por modelos clássicos

### 4. Heterogeneidade Vascular
A vasculatura é FRACTAL:
```
Dimensão fractal da vasculatura: ~2.7 (3D)
Distribuição de fluxo: Lei de potência
Tempo de trânsito: Distribuição heterogênea
```

### 5. Cálculo Fracional
**Derivada de Ordem Fracional:**
```
D^α f(t)  onde 0 < α < 1

Permite modelar:
- Memória do sistema
- Processos não-Markovianos
- Relaxação anômala
```

**Equação de Difusão Fracional:**
```
∂C/∂t = D × ∂^α C/∂x^α  (derivada espacial fracional)
ou
∂^β C/∂t^β = D × ∂²C/∂x²  (derivada temporal fracional)
```

## Aplicações em PBPK

### 1. Distribuição Tecidual
```
Kp(t) = Kp_∞ × (1 - e^(-k_eq × t^(1-h)))

Em vez de Kp constante
```

### 2. Liberação de Depot
```
Liberação de formulações de depósito segue:
M(t)/M_∞ = k × t^n  (Korsmeyer-Peppas)

Onde n indica mecanismo:
- n = 0.5 → Difusão Fickiana
- 0.5 < n < 1 → Transporte anômalo
- n = 1 → Relaxação/erosão
```

### 3. Clearance Hepático
```
CL_H(t) = CL_H,0 × f(fluxo(t), viscosidade(t), heterogeneidade)
```

## Dados Necessários
- [ ] Expoentes fractais para diferentes tecidos
- [ ] Parâmetros de difusão anômala em tecidos humanos
- [ ] Dimensão fractal da vasculatura por órgão
- [ ] Validação com dados PK de drogas reais

## Prioridade: MÉDIA
**Razão:** Conceito avançado, mas pode explicar "anomalias" em PK

## Referências Chave
1. Kopelman (1988) Science - Fractal reaction kinetics
2. Macheras (1996) Pharm Res - Fractal drug distribution
3. Dokoumetzidis & Macheras (2009) - Fractional kinetics in PK
4. Weiss (2014) - Anomalous diffusion in biology

