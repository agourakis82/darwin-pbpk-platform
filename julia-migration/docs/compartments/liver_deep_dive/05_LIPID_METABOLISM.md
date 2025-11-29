# HEPATIC LIPID METABOLISM: Fat Processing Central

## 1. Overview: The Liver as Lipid Traffic Controller

The liver is the **central hub** for lipid metabolism:
- Synthesizes fatty acids, triglycerides, cholesterol
- Packages lipids into lipoproteins for export
- Oxidizes fatty acids for energy
- Produces ketone bodies during fasting
- Converts excess carbohydrates to fat

```
                     DIETARY FAT                    ADIPOSE TISSUE
                         │                               │
                    (chylomicron                    (free fatty
                     remnants)                       acids)
                         │                               │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
    ┌────────────────────────────────────────────────────────────────┐
    │                          HEPATOCYTE                            │
    │                                                                │
    │   INPUTS:                                                      │
    │   • Dietary fat (chylomicron remnants)                        │
    │   • Adipose FFA (during fasting/lipolysis)                    │
    │   • De novo lipogenesis (from glucose)                         │
    │                                                                │
    │                    ┌─────────────┐                             │
    │                    │ FATTY ACID  │                             │
    │                    │    POOL     │                             │
    │                    └──────┬──────┘                             │
    │                           │                                    │
    │         ┌─────────────────┼─────────────────┐                 │
    │         │                 │                 │                  │
    │         ▼                 ▼                 ▼                  │
    │   ┌──────────┐     ┌──────────┐     ┌──────────┐             │
    │   │   TG     │     │  β-OX    │     │ KETONE   │             │
    │   │SYNTHESIS │     │(energy)  │     │  BODIES  │             │
    │   └────┬─────┘     └──────────┘     └────┬─────┘             │
    │        │                                  │                   │
    │        ▼                                  ▼                   │
    │   ┌──────────┐                     ┌──────────┐              │
    │   │  VLDL    │                     │Acetoacetate│             │
    │   │ASSEMBLY  │                     │β-hydroxy- │             │
    │   └────┬─────┘                     │butyrate   │             │
    │        │                            └────┬─────┘             │
    └────────┼─────────────────────────────────┼───────────────────┘
             │                                  │
             ▼                                  ▼
        PERIPHERAL                          BRAIN, MUSCLE
         TISSUES                           (during fasting)
    (Muscle, Adipose)
```

---

## 2. Fatty Acid Uptake and Activation

### 2.1 Fatty Acid Transport into Hepatocyte

```
FATTY ACID UPTAKE MECHANISMS:

1. FREE DIFFUSION (minor at low concentrations)
   FFA ─────────► Hepatocyte
   (flip-flop across membrane)

2. PROTEIN-MEDIATED TRANSPORT (major):

   ┌─────────────────────────────────────────────────────────────┐
   │                                                             │
   │  BLOOD                        HEPATOCYTE                   │
   │    │                               │                        │
   │    │   ┌─────┐   ┌─────┐          │                        │
   │  FFA──►│FAT  │──►│FATP │──►────────┼──► FFA                │
   │    │   │/CD36│   │2/5  │          │                        │
   │    │   └─────┘   └─────┘          │                        │
   │                                    │                        │
   │  FAT/CD36: Scavenger receptor     │                        │
   │  FATP: Fatty Acid Transport Proteins                       │
   │                                                             │
   └─────────────────────────────────────────────────────────────┘

3. LIPOPROTEIN RECEPTOR-MEDIATED:
   • LDL receptor (LDLR)
   • LRP (LDL receptor-related protein)
   • Scavenger receptors
```

### 2.2 Fatty Acid Activation

```
FATTY ACID ACTIVATION (Required for all subsequent metabolism):

Fatty acid + CoA + ATP ──ACSL──► Fatty acyl-CoA + AMP + PPi

ACSL = Acyl-CoA Synthetase Long-chain

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Location: Outer mitochondrial membrane, ER, peroxisomes   │
│                                                             │
│  Isoforms:                                                  │
│  • ACSL1: Major hepatic isoform, mitochondrial            │
│  • ACSL3: ER, lipid droplets                               │
│  • ACSL4: Prefers polyunsaturated FA                       │
│  • ACSL5: Mitochondrial                                    │
│                                                             │
│  This is a COMMITTED STEP:                                  │
│  Once activated, FA must be:                                │
│  • Oxidized (β-oxidation)                                   │
│  • Incorporated into complex lipids (TG, PL)               │
│  • Cannot simply diffuse back out                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. De Novo Lipogenesis (DNL)

### 3.1 Overview

```
DE NOVO LIPOGENESIS: Making Fat from Carbohydrates

FED STATE (High insulin, high glucose):

Glucose → Pyruvate → Acetyl-CoA → Malonyl-CoA → Palmitate → TG → VLDL

Location: CYTOSOL (requires export of acetyl-CoA from mitochondria)
```

### 3.2 The Pathway

```
DETAILED DNL PATHWAY:

MITOCHONDRIA                                  CYTOSOL
┌───────────────────┐                    ┌───────────────────────────┐
│                   │                    │                           │
│   Pyruvate        │                    │                           │
│      │            │                    │                           │
│      ▼            │                    │                           │
│   Acetyl-CoA      │                    │                           │
│      │            │                    │                           │
│      ▼            │                    │                           │
│   CITRATE ────────┼───────────────────►│   CITRATE                │
│                   │  (citrate shuttle) │      │                    │
│                   │                    │      ▼                    │
│                   │                    │   ATP-CITRATE LYASE       │
│                   │                    │      │                    │
│                   │                    │      ▼                    │
│                   │                    │   Acetyl-CoA              │
│                   │                    │      │                    │
│                   │                    │      ▼                    │
│                   │                    │   ACETYL-CoA              │
│                   │                    │   CARBOXYLASE (ACC)       │
│                   │                    │      │ (RATE-LIMITING!)   │
│                   │                    │      ▼                    │
│                   │                    │   Malonyl-CoA             │
│                   │                    │      │                    │
│                   │                    │      ▼                    │
│                   │                    │   FATTY ACID              │
│                   │                    │   SYNTHASE (FAS)          │
│                   │                    │      │                    │
│                   │                    │      ▼                    │
│                   │                    │   PALMITATE (C16:0)       │
│                   │                    │                           │
└───────────────────┘                    └───────────────────────────┘
```

### 3.3 Regulation of DNL

```
REGULATION OF ACC (Acetyl-CoA Carboxylase) - THE KEY ENZYME:

ALLOSTERIC:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ACTIVATORS:            INHIBITORS:                         │
│  • Citrate              • Long-chain acyl-CoA              │
│                         • Malonyl-CoA (product)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

COVALENT (Phosphorylation):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  AMPK phosphorylates ACC → INACTIVATION                    │
│                                                             │
│  This occurs during:                                        │
│  • Fasting (low ATP/AMP ratio)                             │
│  • Exercise                                                 │
│  • Metformin treatment (activates AMPK)                    │
│                                                             │
│  AMPK = "Fuel gauge" - when energy low, stops fat synthesis│
│                                                             │
└─────────────────────────────────────────────────────────────┘

TRANSCRIPTIONAL:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  SREBP-1c (Sterol Regulatory Element Binding Protein):     │
│  • Activated by insulin                                     │
│  • Increases transcription of ACC, FAS, SCD1               │
│  • Master regulator of lipogenesis                          │
│                                                             │
│  ChREBP (Carbohydrate Response Element Binding Protein):   │
│  • Activated by glucose metabolites                         │
│  • Also increases lipogenic gene expression                │
│                                                             │
│  LXR (Liver X Receptor):                                   │
│  • Activated by oxysterols                                  │
│  • Promotes SREBP-1c expression                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘

DRUG IMPLICATIONS:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  METFORMIN:                                                 │
│  • Activates AMPK → Inhibits ACC → ↓ Lipogenesis          │
│  • Part of mechanism for ↓ hepatic glucose output          │
│                                                             │
│  STATINS:                                                   │
│  • Inhibit HMG-CoA reductase (cholesterol pathway)         │
│  • Can paradoxically ↑ SREBP → ↑ Lipogenesis              │
│                                                             │
│  ALCOHOL:                                                   │
│  • ↑ NADH/NAD ratio → ↑ Lipogenesis                       │
│  • ↑ SREBP-1c → Fatty liver                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. β-Oxidation: Burning Fat for Energy

### 4.1 Overview

```
β-OXIDATION: Breaking Down Fatty Acids for ATP

FASTING STATE (Low insulin, high glucagon):

Fatty acyl-CoA → Acetyl-CoA → TCA cycle → ATP

Location: MITOCHONDRIAL MATRIX (requires carnitine shuttle)
          Also: Peroxisomes (for very long-chain FA)
```

### 4.2 The Carnitine Shuttle

```
CARNITINE SHUTTLE (Entry into mitochondria):

CYTOSOL                    OUTER MEMBRANE       INNER MEMBRANE        MATRIX
   │                            │                    │                  │
   │                            │                    │                  │
Acyl-CoA ──────────────────────►│                    │                  │
   │       CPT I                │                    │                  │
   │    (inhibited by           │                    │                  │
   │     malonyl-CoA!)          │                    │                  │
   │                            │                    │                  │
   ▼                            │                    │                  │
Acyl-Carnitine ─────────────────┼────────────────────┼──────────────────►│
   │                            │      CACT          │                  │
   │                            │   (translocase)    │                  │
   │                            │                    │                  │
                                │                    │            Acyl-Carnitine
                                │                    │                  │
                                │                    │               CPT II
                                │                    │                  │
                                │                    │                  ▼
                                │                    │            Acyl-CoA
                                │                    │            (in matrix)
                                │                    │                  │
                                │                    │                  ▼
                                │                    │           β-OXIDATION

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  CPT I = Carnitine Palmitoyltransferase I                  │
│        = RATE-LIMITING STEP of β-oxidation                 │
│                                                             │
│  MALONYL-CoA INHIBITS CPT I:                               │
│  • Fed state: DNL produces malonyl-CoA → blocks β-ox      │
│  • Fasted state: No malonyl-CoA → β-ox proceeds           │
│                                                             │
│  This is the "METABOLIC SWITCH":                           │
│  You can't synthesize AND burn fat at the same time!       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 The β-Oxidation Spiral

```
β-OXIDATION SPIRAL (Each cycle removes 2 carbons):

            Palmitoyl-CoA (C16)
                   │
    ┌──────────────┴──────────────┐
    │                             │
    │  1. ACYL-CoA DEHYDROGENASE │    FAD → FADH₂ (→ 1.5 ATP)
    │     (VLCAD, LCAD, MCAD,    │
    │      SCAD for different    │
    │      chain lengths)        │
    │                             │
    │  2. ENOYL-CoA HYDRATASE    │    + H₂O
    │                             │
    │  3. 3-HYDROXYACYL-CoA      │    NAD⁺ → NADH (→ 2.5 ATP)
    │     DEHYDROGENASE          │
    │                             │
    │  4. THIOLASE               │    + CoA → Acetyl-CoA
    │                             │
    └──────────────┬──────────────┘
                   │
                   ▼
            Myristoyl-CoA (C14) + Acetyl-CoA
                   │
                   ▼
            (Repeat 6 more times)
                   │
                   ▼
            8 Acetyl-CoA + 7 FADH₂ + 7 NADH

ATP YIELD FROM PALMITATE:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  7 FADH₂ × 1.5 ATP = 10.5 ATP                              │
│  7 NADH × 2.5 ATP  = 17.5 ATP                              │
│  8 Acetyl-CoA × 10 ATP (via TCA) = 80 ATP                  │
│  Minus 2 ATP for activation                                 │
│  ────────────────────────────────                           │
│  TOTAL: ~106 ATP per palmitate!                            │
│                                                             │
│  (Compare: Glucose = 30-32 ATP)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Disorders of β-Oxidation

```
FATTY ACID OXIDATION DISORDERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MCAD DEFICIENCY (Medium-Chain Acyl-CoA Dehydrogenase):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Frequency: 1:10,000-15,000 (most common FAO disorder)     │
│                                                             │
│  Presentation:                                              │
│  • Hypoketotic hypoglycemia during fasting                 │
│  • Lethargy, vomiting, seizures                            │
│  • Can cause sudden infant death                           │
│                                                             │
│  Diagnosis:                                                 │
│  • Elevated C8 (octanoylcarnitine) on newborn screen       │
│  • Urine organic acids: Hexanoylglycine, suberylglycine   │
│                                                             │
│  Treatment:                                                 │
│  • AVOID FASTING                                            │
│  • Frequent meals, cornstarch at bedtime                   │
│  • IV glucose during illness                                │
│                                                             │
│  DRUG IMPLICATIONS:                                         │
│  • Valproic acid contraindicated (inhibits β-ox)          │
│  • Aspirin caution (Reye's-like syndrome)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

CPT II DEFICIENCY:
  • Myopathic form: Exercise-induced rhabdomyolysis
  • Hepatocardiomyopathic form: Severe, early-onset
```

---

## 5. Ketogenesis: Fuel for the Brain During Fasting

### 5.1 Overview

```
KETOGENESIS: Making Ketone Bodies from Acetyl-CoA

WHEN: Prolonged fasting, starvation, diabetic ketoacidosis
WHERE: LIVER MITOCHONDRIA (only liver makes ketones!)
WHY: Brain can't use fatty acids (don't cross BBB)
     But brain CAN use ketones (cross BBB)
```

### 5.2 The Pathway

```
KETOGENESIS PATHWAY:

                    2 Acetyl-CoA
                         │
                         ▼ THIOLASE
                    Acetoacetyl-CoA
                         │
                         │ + Acetyl-CoA
                         ▼ HMG-CoA SYNTHASE (rate-limiting)
                    HMG-CoA
                         │
                         ▼ HMG-CoA LYASE
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
    ACETOACETATE               (Acetyl-CoA released)
          │
    ┌─────┴─────┐
    │           │
    │           ▼ β-HYDROXYBUTYRATE
    │           │ DEHYDROGENASE
    │           │ (NADH → NAD⁺)
    │           │
    │           ▼
    │    β-HYDROXYBUTYRATE
    │    (main circulating
    │     ketone body)
    │
    ▼
  SPONTANEOUS
  DECARBOXYLATION
    │
    ▼
  ACETONE
  (exhaled - "fruity breath"
   in DKA)


NORMAL FASTING:     Ketones ~0.3-0.5 mM
PROLONGED FASTING:  Ketones ~3-5 mM
DKA:                Ketones >10 mM (can be >25 mM)
```

### 5.3 Regulation

```
REGULATION OF KETOGENESIS:

1. SUBSTRATE SUPPLY:
   • ↑ FFA from adipose → ↑ β-oxidation → ↑ Acetyl-CoA → ↑ Ketones
   • Driven by: Low insulin, High glucagon, High catecholamines

2. MALONYL-CoA (the metabolic switch):
   • Fed: ↑ Malonyl-CoA → Inhibits CPT I → No β-ox → No ketones
   • Fasted: ↓ Malonyl-CoA → β-ox proceeds → Ketones made

3. OXALOACETATE AVAILABILITY:
   • OAA needed to burn Acetyl-CoA in TCA cycle
   • Fasting: Gluconeogenesis consumes OAA
   • Acetyl-CoA "backs up" → Diverted to ketogenesis

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  WHY DOES DKA OCCUR IN DIABETES?                           │
│                                                             │
│  Type 1 DM: Absolute insulin deficiency                    │
│                                                             │
│  • No insulin → Unrestrained lipolysis → Massive FFA      │
│  • No insulin → No ACC activation → No malonyl-CoA        │
│  • No insulin → Gluconeogenesis runs → OAA depleted       │
│                                                             │
│  Result: Massive ketone production → Metabolic acidosis    │
│                                                             │
│  Type 2 DM: Usually enough insulin to prevent DKA          │
│  (but can occur in severe stress, infection, HHS)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. VLDL Assembly and Secretion

### 6.1 Overview

```
VLDL: THE LIVER'S LIPID EXPORT PACKAGE

Purpose: Export triglycerides and cholesterol to peripheral tissues

Composition:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│          VLDL PARTICLE (~30-80 nm diameter)                │
│                                                             │
│    ┌───────────────────────────────────────────┐           │
│    │  ╭──────────────────────────────────╮    │           │
│    │  │      CORE (hydrophobic)          │    │           │
│    │  │                                  │    │           │
│    │  │    TRIGLYCERIDES (~55%)         │    │           │
│    │  │    Cholesterol esters (~15%)    │    │           │
│    │  │                                  │    │           │
│    │  ╰──────────────────────────────────╯    │           │
│    │                                          │           │
│    │  SURFACE (amphipathic):                  │           │
│    │  • Phospholipids                         │           │
│    │  • Free cholesterol                      │           │
│    │  • ApoB-100 (1 per particle - required!) │           │
│    │  • ApoE, ApoC-II, ApoC-III              │           │
│    │                                          │           │
│    └───────────────────────────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 VLDL Assembly

```
VLDL ASSEMBLY IN THE ER:

┌─────────────────────────────────────────────────────────────┐
│                 ENDOPLASMIC RETICULUM                       │
│                                                             │
│   1. ApoB-100 translation begins on ribosome               │
│      │                                                      │
│      ▼                                                      │
│   2. As ApoB enters ER lumen, MTP adds lipids              │
│      │                                                      │
│      │   MTP = Microsomal Triglyceride Transfer Protein    │
│      │   (CRITICAL - mutations cause abetalipoproteinemia) │
│      │                                                      │
│      ▼                                                      │
│   3. Pre-VLDL particle formed (partially lipidated ApoB)   │
│      │                                                      │
│      ▼                                                      │
│   4. Transport to Golgi                                     │
│      │                                                      │
│      ▼                                                      │
│   5. Further TG addition, maturation                        │
│      │                                                      │
│      ▼                                                      │
│   6. Secretion via secretory vesicles                       │
│      │                                                      │
│      ▼                                                      │
│   SPACE OF DISSE → SINUSOID → CIRCULATION                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

IF MTP IS DEFICIENT OR ApoB-100 DEFICIENT:
  • TG cannot be exported → Accumulates in liver → FATTY LIVER
  • This is why MTP inhibitors (lomitapide) cause steatosis
```

### 6.3 VLDL Metabolism in Circulation

```
VLDL → IDL → LDL CASCADE:

           VLDL
            │
            │ Lipoprotein Lipase (LPL)
            │ in capillaries (muscle, adipose)
            │ releases fatty acids
            ▼
           IDL (Intermediate-Density)
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼
  Hepatic        Further
  uptake         processing
  (LDL-R,        by Hepatic
   LRP)          Lipase
                  │
                  ▼
                 LDL
                  │
            To peripheral
            tissues for
            cholesterol
            delivery
```

---

## 7. Cholesterol Metabolism

### 7.1 Synthesis

```
CHOLESTEROL SYNTHESIS (Mevalonate Pathway):

Acetyl-CoA
    │
    ▼
Acetoacetyl-CoA
    │
    ▼
HMG-CoA
    │
    ▼ HMG-CoA REDUCTASE ◄──── STATINS INHIBIT HERE!
    │ (RATE-LIMITING)
    ▼
Mevalonate
    │
    ▼ (Multiple steps)
    │
Squalene
    │
    ▼ (Multiple steps)
    │
CHOLESTEROL


REGULATION:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  SREBP-2 (Sterol Regulatory Element Binding Protein 2):    │
│                                                             │
│  • Low cholesterol: SREBP-2 released from ER → nucleus     │
│    → ↑ transcription of HMG-CoA reductase, LDLR           │
│                                                             │
│  • High cholesterol: SREBP-2 retained in ER                │
│    → ↓ cholesterol synthesis and uptake                    │
│                                                             │
│  STATINS:                                                   │
│  • Inhibit HMG-CoA reductase → ↓ cholesterol synthesis     │
│  • ↓ Cholesterol → ↑ SREBP-2 → ↑ LDLR                     │
│  • Result: ↑ LDL clearance (main mechanism of LDL ↓)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Bile Acid Synthesis

```
BILE ACID SYNTHESIS (Major route of cholesterol elimination):

Cholesterol
    │
    ▼ CYP7A1 (rate-limiting)
    │ (7α-hydroxylase)
    │
    ▼
7α-Hydroxycholesterol
    │
    ▼ (Multiple steps)
    │
    ├──────────────────────────────────┐
    │                                  │
    ▼                                  ▼
CHOLIC ACID                    CHENODEOXYCHOLIC ACID
(primary bile acid)            (primary bile acid)
    │                                  │
    │ Conjugation with                │
    │ glycine or taurine              │
    ▼                                  ▼
GLYCOCHOLATE                   TAUROCHENODEOXYCHOLATE
TAUROCHOLATE                   GLYCOCHENODEOXYCHOLATE
    │                                  │
    └──────────────┬───────────────────┘
                   │
                   ▼
              BILE → INTESTINE
                   │
                   ▼
         SECONDARY BILE ACIDS
         (bacterial modification)
         • Deoxycholic acid
         • Lithocholic acid
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
    95% REABSORBED        5% EXCRETED
    (enterohepatic        (only route of
     circulation)          cholesterol loss!)
```

---

## 8. Drug Implications of Hepatic Lipid Metabolism

### 8.1 Drugs Affecting Lipid Metabolism

```
DRUGS TARGETING HEPATIC LIPID PATHWAYS:

┌───────────────┬─────────────────────────────────────────────┐
│ DRUG CLASS    │ MECHANISM & HEPATIC EFFECT                  │
├───────────────┼─────────────────────────────────────────────┤
│ STATINS       │ • Inhibit HMG-CoA reductase                │
│               │ • ↑ LDLR expression                         │
│               │ • Mild ↑ DNL (SREBP activation)            │
├───────────────┼─────────────────────────────────────────────┤
│ FIBRATES      │ • PPARα agonists                           │
│               │ • ↑ β-oxidation (↑ CPT I, LCAD)           │
│               │ • ↑ LPL → ↓ TG                             │
│               │ • ↓ VLDL secretion                         │
├───────────────┼─────────────────────────────────────────────┤
│ EZETIMIBE     │ • Inhibits NPC1L1 (intestinal)             │
│               │ • ↓ Cholesterol absorption                  │
│               │ • Compensatory ↑ hepatic synthesis         │
├───────────────┼─────────────────────────────────────────────┤
│ PCSK9         │ • ↑ LDLR on hepatocyte surface            │
│ INHIBITORS    │ • Dramatic LDL lowering                    │
├───────────────┼─────────────────────────────────────────────┤
│ MTP           │ • Block VLDL assembly                      │
│ INHIBITORS    │ • ↓ LDL (and TG)                          │
│ (Lomitapide)  │ • CAUSE FATTY LIVER (TG can't exit!)      │
├───────────────┼─────────────────────────────────────────────┤
│ BILE ACID     │ • Bind bile acids in gut                   │
│ SEQUESTRANTS  │ • Interrupt enterohepatic circulation      │
│               │ • ↑ Hepatic bile acid synthesis            │
│               │ • ↑ LDLR                                   │
├───────────────┼─────────────────────────────────────────────┤
│ OMEGA-3 FA    │ • ↓ SREBP-1c → ↓ DNL                      │
│               │ • ↑ β-oxidation                            │
│               │ • ↓ VLDL secretion                         │
└───────────────┴─────────────────────────────────────────────┘
```

### 8.2 Drug-Induced Fatty Liver (Steatosis)

```
DRUGS CAUSING HEPATIC STEATOSIS:

┌───────────────────────────────────────────────────────────────────┐
│ MECHANISM              │ DRUGS                                    │
├────────────────────────┼──────────────────────────────────────────┤
│ ↑ DNL                  │ Valproic acid, Corticosteroids,         │
│                        │ Tamoxifen, Alcohol                       │
├────────────────────────┼──────────────────────────────────────────┤
│ ↓ β-oxidation          │ Valproic acid, Aspirin (children),      │
│                        │ Tetracycline, Amiodarone                │
├────────────────────────┼──────────────────────────────────────────┤
│ ↓ VLDL secretion       │ MTP inhibitors (lomitapide),            │
│                        │ Tetracycline                            │
├────────────────────────┼──────────────────────────────────────────┤
│ ↑ FFA delivery         │ Corticosteroids (lipolysis),            │
│                        │ Alcohol                                  │
├────────────────────────┼──────────────────────────────────────────┤
│ Mitochondrial toxicity │ NRTIs (stavudine, didanosine),          │
│                        │ Amiodarone, Valproic acid               │
└────────────────────────┴──────────────────────────────────────────┘
```

---

## 9. Implications for Liver Kp

```
LIPID METABOLISM AND LIVER Kp:

1. LIPOPHILIC DRUG PARTITIONING:
   • Liver has significant lipid content
   • Steatotic liver: ↑ lipid → ↑ Kp for lipophilic drugs
   • NAFLD/NASH patients may have altered hepatic drug distribution

2. LIPOPROTEIN BINDING:
   • Drugs can bind to lipoproteins (VLDL, LDL, HDL)
   • Lipoprotein-bound drug delivered TO liver
   • May affect hepatic uptake and Kp

3. FATTY ACID COMPETITION:
   • Free fatty acids compete for albumin binding
   • High FFA (fasting, diabetes): ↑ free fraction of acidic drugs
   • May affect distribution

4. FEATURES FOR ML MODELS:
   • logP (lipophilicity)
   • is_steatosis_substrate (partitions into fat)
   • protein_binding_affected_by_FFA
   • lipoprotein_binding_fraction

5. DISEASE STATES:
   • NAFLD: ↑ hepatic TG → altered Kp for lipophilic drugs
   • Cirrhosis: ↓ lipoprotein synthesis → altered distribution
   • Diabetes: ↑ FFA → altered binding and distribution
```

---

**NEXT**: Part 6 - Hepatic Hormone Production and First-Pass Effect
