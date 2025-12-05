# 🩸 BLOOD COMPARTMENT - SÉRIE BRANCA (Leucócitos/WBCs)

**Status:** TÓPICO CRÍTICO NÃO DISCUTIDO  
**Relevância Clínica:** ALTA para fármacos imunomoduladores, anti-infecciosos, quimioterápicos  
**Timestamp:** 2025-12-01T12:10:00-03:00

---

## 🎯 POR QUE A SÉRIE BRANCA É RELEVANTE PARA PBPK?

### **Cenários Clínicos Onde WBCs Afetam PK:**

1. **Fármacos com Binding a Leucócitos**
   - Azitromicina: Acúmulo intracelular em neutrófilos (Vd ↑ 30×)
   - Clindamicina: Binding a leucócitos → t½ prolongado
   - Antimaláricos: Internalização em monócitos/macrófagos
   - Imunossupressores: Targeting de linfócitos T/B

2. **Leucocitose/Leucopenia**
   - Infecção aguda: WBC ↑ 5-20× → ↑ binding pool → Vd ↑
   - Leucopenia (quimio): WBC ↓ 90% → ↓ binding → Vd ↓
   - Leucemia: Massa leucocitária ↑ 100× → efeito dramático

3. **Transporte Ativo em WBCs**
   - Anticorpos monoclonais: Internalização via FcγR
   - Nanopartículas: Fagocitose por macrófagos
   - Lipossomas: Uptake por células dendríticas

4. **Metabolismo em WBCs**
   - Conversão de pró-fármacos (ex: clopidogrel → metabólito ativo)
   - Desativação de fármacos (ex: GSH conjugation)

---

## 📊 COMPOSIÇÃO FISIOLÓGICA DA SÉRIE BRANCA

### **Populações Normais (70kg adulto, sangue total 5L):**

```
Tipo de Leucócito    | Contagem (células/μL) | % Total | Volume Total | Massa
---------------------|----------------------|---------|--------------|------
Neutrófilos          | 3,000-7,000          | 50-70%  | ~0.04 L      | ~2.5g
Linfócitos           | 1,500-4,000          | 20-40%  | ~0.02 L      | ~1.2g
Monócitos            | 200-800              | 2-8%    | ~0.003 L     | ~0.2g
Eosinófilos          | 50-500               | 1-4%    | ~0.001 L     | ~0.06g
Basófilos            | 20-50                | <1%     | ~0.0003 L    | ~0.02g
---------------------|----------------------|---------|--------------|------
TOTAL WBCs           | 5,000-11,000         | 100%    | ~0.065 L     | ~4.0g

Volume fracional: ~1.3% do sangue total (vs. ~45% RBCs, ~54% plasma)
```

### **Características Relevantes para PK:**

#### **1. NEUTRÓFILOS (Polymorphonuclear leukocytes)**

**Fisiologia:**
- Volume celular: ~330 fL (vs. 90 fL para RBC)
- Superfície: ~170 μm²
- Vida média: 6-8 horas no sangue, 1-2 dias tecidual
- Fagocitose: Alta capacidade (20-30 partículas/célula/hora)

**Relevância PK:**
- **Binding a fármacos:** Membrana rica em fosfolipídios → lipofílicos
- **Internalização:** Via endocitose (caveolae, clathrin)
- **Azitromicina:** Kd ≈ 10 nM, Bmax ≈ 10⁶ sites/célula
- **Fármacos beta-lactâmicos:** Binding moderado (fu_cell ≈ 0.3-0.7)

**Equação de Binding:**
```
C_WBC = C_plasma × (1 + (N_neutrophils × Bmax × Kd) / (Kd + C_plasma))
```
Onde:
- N_neutrophils = contagem de neutrófilos (células/L sangue)
- Bmax = capacidade máxima de binding (mol/célula)
- Kd = constante de dissociação (mol/L)

#### **2. LINFÓCITOS (T, B, NK)**

**Fisiologia:**
- Volume celular: ~200 fL (pequenos), ~500 fL (grandes)
- Vida média: Dias a anos (memória imunológica)
- Recirculação: 48 horas entre sangue ↔ tecidos linfoides

**Relevância PK:**
- **Targeting específico:**
  - **Ciclosporina/Tacrolimus:** Binding a ciclofilinas em linfócitos T
  - **Rituximab (anti-CD20):** Binding seletivo a linfócitos B
  - **Alemtuzumab (anti-CD52):** Depleção de linfócitos

**Modelo de Internalização:**
```
dC_lymphocyte/dt = k_uptake × C_plasma × N_CD20+ - k_efflux × C_lymphocyte
```

**Parâmetros Típicos:**
- Rituximab: k_uptake ≈ 0.5 L/(mol·h), t½ internalização ≈ 2h
- Ciclosporina: Binding a ciclofilina-A (Kd ≈ 10 nM)

#### **3. MONÓCITOS/MACRÓFAGOS**

**Fisiologia:**
- Volume celular: ~400 fL (monócitos), maior quando diferenciados
- Vida média: 1-3 dias no sangue, semanas/meses como macrófagos
- Fagocitose: Muito alta (especializada em partículas grandes)

**Relevância PK:**
- **Fármacos antimaláricos:**
  - Cloroquina: Acúmulo em macrófagos do baço/fígado (C_tissue/C_plasma ≈ 1000×)
  - Mefloquina: Similar, mas menor acúmulo
- **Nanopartículas:** Uptake preferencial (vírus, lipossomas, PLGA)
- **Produtos biológicos:** Fagocitose de complexos imune

**Equação de Acúmulo:**
```
C_monocyte = C_plasma × (1 + Kp_monocyte × (1 + V_lysosome/V_cytosol))
```
Onde Kp_monocyte ≈ 10-1000 (depende do fármaco)

#### **4. EOSINÓFILOS**

**Fisiologia:**
- Volume celular: ~400 fL
- Vida média: 8-12 horas no sangue
- Relevância: Fármacos anti-helmínticos, alergias

**Relevância PK:**
- Binding limitado para maioria dos fármacos
- Importante para: Ivermectina, albendazol (binding moderado)

#### **5. BASÓFILOS**

**Fisiologia:**
- Volume celular: ~300 fL
- Vida média: horas no sangue
- Relevância: Mínima para PK (população muito pequena)

---

## 🔬 MECANISMOS DE INTERAÇÃO FÁRMACO-LEUCÓCITO

### **1. BINDING À MEMBRANA (Passivo)**

**Mecanismo:**
- Interações hidrofóbicas com fosfolipídios
- Binding a receptores de superfície
- Adsorção não-específica

**Modelo Matemático:**
```
fu_WBC = 1 / (1 + K_binding × [WBC] × f_lipid_WBC)
```

Onde:
- K_binding = constante de binding (L/mol)
- [WBC] = concentração de leucócitos (células/L)
- f_lipid_WBC = fração lipídica da membrana (~0.5)

**Exemplos:**
- Azitromicina: fu_WBC ≈ 0.01-0.05 (altamente bound)
- Amoxicilina: fu_WBC ≈ 0.7-0.9 (fracamente bound)

### **2. INTERNALIZAÇÃO ATIVA (Endocitose/Fagocitose)**

**Mecanismos:**
- **Receptores Fc (FcγR, FcεR):** Anticorpos, imunocomplexos
- **Receptores de complemento:** C3b, C5a
- **Pattern recognition receptors (TLR, NLR):** PAMPs, DAMPs
- **Scavenger receptors:** Lipoproteínas, partículas

**Cinética de Internalização:**
```
dC_intracellular/dt = V_max × C_extracellular / (K_m + C_extracellular) - k_efflux × C_intracellular
```

**Parâmetros Típicos:**
- Rituximab: V_max ≈ 1.5×10⁻⁹ mol/(célula·h), K_m ≈ 10⁻⁹ mol/L
- Azitromicina: V_max ≈ 3×10⁻¹² mol/(célula·h), K_m ≈ 10⁻⁹ mol/L

### **3. TRAPING INTRACELULAR (Ion Trapping, Lysosomotropia)**

**Mecanismo:**
- Fármacos básicos (pKa > 7) acumulam em compartimentos ácidos
- Lisossomos: pH ≈ 4.5-5.5
- Azurofílicos em neutrófilos: pH ≈ 5.5

**Equação (Henderson-Hasselbalch modificada):**
```
C_lysosome / C_cytosol = (1 + 10^(pKa - pH_lysosome)) / (1 + 10^(pKa - pH_cytosol))
```

**Exemplos:**
- Cloroquina (pKa = 10.2): C_lysosome/C_plasma ≈ 1000×
- Azitromicina (pKa = 8.7): C_lysosome/C_plasma ≈ 300×
- Amoxicilina (pKa = 2.7): Sem trapping (ácido fraco)

### **4. METABOLISMO INTRACELULAR**

**Enzimas Relevantes:**
- **Esterases:** Conversão de pró-fármacos
- **Citocromo P450:** CYP2D6, CYP3A4 (expressão baixa)
- **Glutationa-S-transferase:** Conjugação
- **Peroxidases:** Oxidação (mieloperoxidase em neutrófilos)

**Taxa de Metabolismo:**
```
CL_intracellular = V_max_metabolism / (K_m_metabolism + C_intracellular)
```

---

## 📐 MODELO PBPK PARA SÉRIE BRANCA

### **Estrutura do Modelo:**

```
┌─────────────────────────────────────────────────────────┐
│                    BLOOD COMPARTMENT                     │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   PLASMA     │  │     RBCs     │  │    WBCs      │  │
│  │              │  │              │  │              │  │
│  │  C_plasma    │  │  C_RBC       │  │  C_WBC       │  │
│  │              │  │              │  │              │  │
│  │  Volume:     │  │  Volume:     │  │  Volume:     │  │
│  │  54% (2.7L)  │  │  45% (2.25L) │  │  1.3% (65mL) │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
│         └─────────────────┴─────────────────┘           │
│                  Exchange kinetics                      │
└─────────────────────────────────────────────────────────┘
```

### **Equações Diferenciais:**

#### **Equação para Plasma:**
```
dC_plasma/dt = (Q_in × C_in - Q_out × C_plasma) / V_plasma
              - k_RBC→plasma × C_RBC
              - k_WBC→plasma × C_WBC
              + k_plasma→RBC × C_plasma
              + k_plasma→WBC × C_plasma
              - CL_plasma × C_plasma
```

#### **Equação para WBCs:**
```
dC_WBC/dt = k_plasma→WBC × C_plasma 
            - k_WBC→plasma × C_WBC
            - k_internalization × C_WBC
            - CL_WBC × C_WBC
```

#### **Equação para Compartimento Intracelular:**
```
dC_WBC_intracellular/dt = k_internalization × C_WBC
                          - k_efflux × C_WBC_intracellular
                          - CL_intracellular × C_WBC_intracellular
```

### **Parâmetros do Modelo:**

```julia
struct WhiteBloodCellCompartment
    # Contagens celulares (células/L sangue)
    neutrophils::Float64      # 3,000-7,000 /μL → 3-7×10⁹ /L
    lymphocytes::Float64      # 1,500-4,000 /μL → 1.5-4×10⁹ /L
    monocytes::Float64        # 200-800 /μL → 0.2-0.8×10⁹ /L
    eosinophils::Float64      # 50-500 /μL → 0.05-0.5×10⁹ /L
    basophils::Float64        # 20-50 /μL → 0.02-0.05×10⁹ /L
    
    # Volumes celulares (fL por célula)
    V_neutrophil::Float64     # ~330 fL
    V_lymphocyte::Float64     # ~200 fL (pequeno), ~500 fL (grande)
    V_monocyte::Float64       # ~400 fL
    
    # Binding parameters
    K_binding_neutrophil::Dict{String, Float64}  # Kd (mol/L) por fármaco
    Bmax_neutrophil::Dict{String, Float64}       # Bmax (mol/célula) por fármaco
    
    # Internalization parameters
    k_internalization::Dict{String, Float64}     # 1/h por fármaco
    k_efflux::Dict{String, Float64}              # 1/h por fármaco
    
    # Compartimento intracelular
    V_lysosome_fraction::Float64  # Fração de volume lisossômico (~0.1)
    pH_lysosome::Float64          # pH lisossomal (~5.0)
    
    # Clearance intracelular
    CL_intracellular::Dict{String, Float64}  # L/(h·célula) por fármaco
end
```

---

## 🏥 ESTADOS PATOLÓGICOS E LEUCOCITOSE/LEUCOPENIA

### **1. LEUCOCITOSE (WBC ↑)**

**Causas:**
- Infecção bacteriana aguda: Neutrófilos ↑ 5-20× (3,000 → 15,000-60,000 /μL)
- Leucemia: WBC ↑ 100× (5,000 → 500,000 /μL)
- Inflamação crônica: WBC ↑ 2-5×

**Efeito no PK:**
```
Vd_WBC = N_WBC_normal × V_cell × Kp_WBC
Vd_WBC_pathological = N_WBC_pathological × V_cell × Kp_WBC

ΔVd = Vd_WBC_pathological / Vd_WBC_normal = N_WBC_pathological / N_WBC_normal
```

**Exemplo:**
- Azitromicina (normal): Vd_WBC ≈ 0.5 L
- Azitromicina (leucocitose 10×): Vd_WBC ≈ 5.0 L
- **Efeito total:** Vd_total ↑ 10-20%

### **2. LEUCOPENIA (WBC ↓)**

**Causas:**
- Quimioterapia: WBC ↓ 90% (5,000 → 500 /μL)
- Agranulocitose: Neutrófilos ↓ 99%
- Imunossupressão: Linfócitos ↓ 80-90%

**Efeito no PK:**
- Vd_WBC ↓ proporcionalmente
- Clearance pode ↑ (menos binding pool)
- Exposição livre ↑ (mais fármaco disponível)

**Exemplo:**
- Rituximab (normal): Binding a 1.5×10⁹ linfócitos B/L
- Rituximab (leucopenia): Binding a 0.15×10⁹ linfócitos B/L
- **Efeito:** Clearance ↑ 50-100% (menos alvos)

### **3. LEUCEMIA**

**Características:**
- Massa celular ↑ 100-1000×
- Volume de WBC pode alcançar 10-20% do sangue total
- Efeito dramático em fármacos com alto binding

**Modelo Especial:**
```
V_WBC_leukemia = N_blasts × V_blast × (1 + f_immature)
```
Onde:
- N_blasts = contagem de blastos (células/L)
- V_blast = volume médio do blasto (~500 fL)
- f_immature = fator de correção para células imaturas (↑ binding)

---

## 💊 FÁRMACOS COM INTERAÇÃO CRÍTICA COM WBCs

### **1. AZITROMICINA**

**Mecanismo:**
- Binding extensivo a neutrófilos
- Internalização ativa
- Acúmulo lisossomal (pKa = 8.7)

**Parâmetros PK:**
```
K_binding_neutrophil = 10⁻⁹ mol/L
Bmax = 10⁶ sites/célula
Kp_lysosome = 300 (trapping ácido)
t½_WBC = 50-70 horas (vs. t½_plasma = 11-14 horas)
```

**Modelo:**
```
fu_WBC_azithromycin = 0.01-0.05  (95-99% bound)
Vd_WBC = N_neutrophils × V_neutrophil × Kp_lysosome × (1 - fu_WBC)
```

### **2. CLOROQUINA/HIDROXICLOROQUINA**

**Mecanismo:**
- Binding a monócitos/macrófagos
- Acúmulo lisossomal extremo (pKa = 10.2)

**Parâmetros PK:**
```
Kp_monocyte = 1000-5000
C_lysosome/C_plasma = 10³-10⁴
t½_tissue = semanas a meses
```

**Relevância Clínica:**
- Toxicidade ocular (acúmulo em retina)
- Cardiotoxicidade (acúmulo em coração)

### **3. RITUXIMAB (Anti-CD20)**

**Mecanismo:**
- Binding específico a linfócitos B (CD20+)
- Internalização via endocitose
- Depleção de células B

**Parâmetros PK:**
```
Target density: ~10⁵ CD20 receptors/célula
Kd = 8 nM
k_internalization = 0.5 L/(mol·h)
Clearance dependente de alvo: CL = k_depletion × N_CD20+
```

**Farmacocinética não-linear:**
- Dose baixa: Clearance alto (muitos alvos)
- Dose alta: Clearance baixo (depleção de alvos)

### **4. CICLOSPORINA/TACROLIMUS**

**Mecanismo:**
- Binding a ciclofilinas em linfócitos T
- Inibição de calcineurina

**Parâmetros PK:**
```
Binding a linfócitos T: 80-90% bound
Kd_ciclofilina = 10 nM
Distribuição preferencial: Tecidos linfoides
```

---

## 🎯 IMPLEMENTAÇÃO NO DARWIN PBPK

### **Estrutura Proposta:**

```julia
# julia-migration/src/DarwinPBPK/compartment_models.jl

"""
WhiteBloodCellPhase - Fase de leucócitos no compartimento sanguíneo
"""
struct WhiteBloodCellPhase <: BloodPhase
    name::String
    volume_fraction::Float64  # ~0.013 (1.3%)
    
    # Contagens celulares
    neutrophils::Float64      # células/L
    lymphocytes::Float64
    monocytes::Float64
    eosinophils::Float64
    basophils::Float64
    
    # Volumes celulares (L/célula)
    V_neutrophil::Float64
    V_lymphocyte::Float64
    V_monocyte::Float64
    
    # Binding parameters (por fármaco)
    binding_constants::Dict{String, Dict{String, Float64}}  # Kd, Bmax
    
    # Internalization parameters
    internalization_rates::Dict{String, Float64}  # k_internalization
    efflux_rates::Dict{String, Float64}           # k_efflux
    
    # Intracellular compartment
    V_lysosome_fraction::Float64
    pH_lysosome::Float64
end

function calculate_WBC_volume_fraction(wbc_phase::WhiteBloodCellPhase)
    V_total = (wbc_phase.neutrophils * wbc_phase.V_neutrophil +
               wbc_phase.lymphocytes * wbc_phase.V_lymphocyte +
               wbc_phase.monocytes * wbc_phase.V_monocyte +
               wbc_phase.eosinophils * wbc_phase.V_neutrophil +  # Similar size
               wbc_phase.basophils * wbc_phase.V_neutrophil)     # Similar size
    return V_total / 1e15  # Convert fL to L
end

function calculate_WBC_partition_coefficient(drug::String, wbc_phase::WhiteBloodCellPhase, drug_pKa::Float64)
    """
    Calcula coeficiente de partição WBC/plasma considerando:
    1. Binding à membrana
    2. Internalização
    3. Ion trapping (se aplicável)
    """
    if !haskey(wbc_phase.binding_constants, drug)
        return 1.0  # Sem binding conhecido
    end
    
    params = wbc_phase.binding_constants[drug]
    Kd = params["Kd"]
    Bmax = params["Bmax"]
    
    # Binding à membrana
    Kp_membrane = 1.0 + (Bmax / Kd)
    
    # Ion trapping (para bases)
    if drug_pKa > 7.0
        Kp_lysosome = (1 + 10^(drug_pKa - wbc_phase.pH_lysosome)) / 
                      (1 + 10^(drug_pKa - 7.4))
        Kp_total = Kp_membrane * (1 + wbc_phase.V_lysosome_fraction * (Kp_lysosome - 1))
    else
        Kp_total = Kp_membrane
    end
    
    return Kp_total
end
```

### **Integração com FractalBlood:**

```julia
# Adicionar fase WBC ao FractalBloodModel

function create_blood_phases(patient::PatientProfile.PatientData, disease_state::String)
    phases = BloodPhase[]
    
    # Plasma phase
    push!(phases, BloodPhase("plasma", 0.54, 1.0, 1.0, 0.0))
    
    # RBC phase
    push!(phases, BloodPhase("RBC", patient.hematocrit, 0.8, 1.2, 10.0))
    
    # WBC phase
    wbc_phase = create_WBC_phase(patient, disease_state)
    push!(phases, BloodPhase("WBC", 
                             calculate_WBC_volume_fraction(wbc_phase),
                             0.7,  # Velocidade relativa menor
                             calculate_WBC_partition_coefficient("drug", wbc_phase, 8.7),
                             5.0))  # Taxa de troca
    
    return phases
end
```

---

## 🧪 VALIDAÇÃO E CASOS DE TESTE

### **Casos Clínicos para Validar:**

1. **Azitromicina em Infecção Bacteriana**
   - Baseline: WBC = 5,000 /μL, Vd = 31 L
   - Leucocitose: WBC = 25,000 /μL, Vd esperado = 35-40 L
   - Validar: Predição de Vd e clearance

2. **Rituximab em Linfoma**
   - Baseline: Linfócitos B = 500 /μL, CL = 0.5 L/h
   - Após depleção: Linfócitos B = 50 /μL, CL esperado = 0.1-0.2 L/h
   - Validar: Clearance não-linear dependente de alvo

3. **Cloroquina em Malária**
   - Acúmulo em macrófagos do baço: C_tissue/C_plasma = 1000-5000×
   - Validar: Predição de acúmulo tecidual

---

## ❓ QUESTÕES PARA DISCUSSÃO

1. **Quais fármacos são prioritários?**
   - Azitromicina? Rituximab? Antimaláricos? Outros?

2. **Nível de detalhe:**
   - Modelo agregado (WBC total) vs. subpopulações (neutrófilos, linfócitos, etc.)?

3. **Estados patológicos:**
   - Leucocitose/leucopenia? Leucemia? Infecção?

4. **Implementação:**
   - Integrar no FractalBlood existente ou módulo separado?

---

**Próximos Passos:** Aguardando sua direção sobre prioridades e nível de implementação! 🎯

