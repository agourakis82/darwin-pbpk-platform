# PHASE II METABOLISM: Conjugation Reactions

## 1. Overview: The Purpose of Conjugation

Phase II metabolism **attaches polar groups** to drugs (or Phase I metabolites), making them:
- More water-soluble
- Easier to excrete (bile or urine)
- Usually pharmacologically INACTIVE (with exceptions)

```
PHASE I vs PHASE II:

PHASE I (Functionalization):
  Drug ──CYP450──► Drug-OH (add functional group)
                      │
                      │ Still somewhat lipophilic
                      │ May still be active
                      ▼

PHASE II (Conjugation):
  Drug-OH ──UGT──► Drug-O-Glucuronide
                      │
                      │ Very hydrophilic (MW +176)
                      │ Usually inactive
                      │ Ready for excretion
                      ▼
                  BILE or URINE
```

---

## 2. The Major Phase II Enzymes

### 2.1 Overview of Conjugation Reactions

```
┌─────────────────────────────────────────────────────────────────────┐
│ ENZYME FAMILY    │ COFACTOR           │ GROUP ADDED    │ MW ADDED  │
├──────────────────┼────────────────────┼────────────────┼───────────┤
│ UGT              │ UDP-glucuronic acid│ Glucuronide    │ +176      │
│ (Glucuronidation)│                    │                │           │
├──────────────────┼────────────────────┼────────────────┼───────────┤
│ SULT             │ PAPS               │ Sulfate        │ +80       │
│ (Sulfation)      │                    │                │           │
├──────────────────┼────────────────────┼────────────────┼───────────┤
│ GST              │ Glutathione        │ GSH            │ +307      │
│ (Glutathione)    │                    │                │           │
├──────────────────┼────────────────────┼────────────────┼───────────┤
│ NAT              │ Acetyl-CoA         │ Acetyl         │ +42       │
│ (Acetylation)    │                    │                │           │
├──────────────────┼────────────────────┼────────────────┼───────────┤
│ MT               │ SAM                │ Methyl         │ +14       │
│ (Methylation)    │                    │                │           │
├──────────────────┼────────────────────┼────────────────┼───────────┤
│ Amino acid       │ Glycine, Glutamine │ Amino acid     │ +75/+146  │
│ conjugation      │ Taurine            │                │           │
└──────────────────┴────────────────────┴────────────────┴───────────┘
```

---

## 3. UGT (UDP-Glucuronosyltransferases) - The Major Pathway

### 3.1 Overview

```
GLUCURONIDATION - The Most Important Phase II Reaction

Reaction:
  Drug-OH + UDP-Glucuronic Acid ──UGT──► Drug-O-Glucuronide + UDP
  
                O
                ‖
  Drug─OH  +   COOH        ──►    Drug─O─Glucuronide + UDP
               │                        │
              HO─┤                      HO─┤
                 │                         │
              HO─┤                      HO─┤
                 │                         │
              ─O─UDP                     COOH
              
  (UDP-Glucuronic Acid)            (β-D-Glucuronide)

Location: Endoplasmic Reticulum (ER) membrane
          Active site faces ER lumen (unique topology!)
```

### 3.2 UGT Families

```
UGT NOMENCLATURE AND SUBSTRATES:

UGT1A FAMILY (all from one gene with alternative first exons):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────┬────────────────────────────────────────────────────────┐
│ ENZYME  │ KEY SUBSTRATES                                         │
├─────────┼────────────────────────────────────────────────────────┤
│ UGT1A1  │ BILIRUBIN (critical!), Estradiol, SN-38 (irinotecan)  │
│         │ Gilbert's syndrome = UGT1A1*28 (↓ activity)           │
├─────────┼────────────────────────────────────────────────────────┤
│ UGT1A3  │ NSAIDs, Bile acids, Statins                           │
├─────────┼────────────────────────────────────────────────────────┤
│ UGT1A4  │ Tertiary amines: Lamotrigine, Olanzapine, Trifluopera-│
│         │ zine (N-glucuronidation)                               │
├─────────┼────────────────────────────────────────────────────────┤
│ UGT1A6  │ Small phenols: Acetaminophen, Serotonin               │
├─────────┼────────────────────────────────────────────────────────┤
│ UGT1A9  │ Propofol, Mycophenolic acid, NSAIDs                   │
└─────────┴────────────────────────────────────────────────────────┘

UGT2B FAMILY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────┬────────────────────────────────────────────────────────┐
│ ENZYME  │ KEY SUBSTRATES                                         │
├─────────┼────────────────────────────────────────────────────────┤
│ UGT2B7  │ Morphine (→ M3G, M6G), Zidovudine, NSAIDs, Steroids   │
│         │ MOST IMPORTANT for opioid metabolism                   │
├─────────┼────────────────────────────────────────────────────────┤
│ UGT2B15 │ Oxazepam, Lorazepam, S-Oxazepam                       │
├─────────┼────────────────────────────────────────────────────────┤
│ UGT2B17 │ Testosterone, Dihydrotestosterone (androgens)         │
└─────────┴────────────────────────────────────────────────────────┘
```

### 3.3 Clinical Implications

```
UGT1A1 AND BILIRUBIN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bilirubin ──UGT1A1──► Bilirubin Glucuronide ──MRP2──► BILE
(unconjugated)        (conjugated)

GILBERT'S SYNDROME (UGT1A1*28):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Frequency: 5-10% of population (very common!)                 │
│                                                                 │
│  Cause: (TA)₇ instead of (TA)₆ in promoter                     │
│         → 30-50% reduction in UGT1A1 expression                │
│                                                                 │
│  Effect: Mild unconjugated hyperbilirubinemia                  │
│          (typically 1-3 mg/dL, increases with fasting/stress)  │
│                                                                 │
│  DRUG IMPLICATIONS:                                             │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ IRINOTECAN:                                                │ │
│  │ • Irinotecan → SN-38 (active) → SN-38G (inactive)         │ │
│  │                           ↑                                │ │
│  │                        UGT1A1                              │ │
│  │                                                            │ │
│  │ UGT1A1*28/*28: ↓ SN-38 glucuronidation                   │ │
│  │              → ↑ SN-38 levels                              │ │
│  │              → SEVERE NEUTROPENIA & DIARRHEA              │ │
│  │                                                            │ │
│  │ FDA: Reduce irinotecan dose in UGT1A1*28 homozygotes      │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

CRIGLER-NAJJAR SYNDROME:
  Type I: Complete absence of UGT1A1 → Fatal without treatment
  Type II: Severely reduced UGT1A1 → Responds to phenobarbital
```

```
MORPHINE GLUCURONIDATION (UGT2B7):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                        UGT2B7
Morphine ──────────────────────────────────► Morphine-3-Glucuronide (M3G)
    │                                         (INACTIVE, may be neuroexcitatory)
    │
    │                   UGT2B7
    └──────────────────────────────────────► Morphine-6-Glucuronide (M6G)
                                              (ACTIVE! More potent than morphine)

Clinical Implications:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  M6G is 50-100× more potent analgesic than morphine            │
│  M6G accumulates in RENAL FAILURE (excreted by kidney)         │
│                                                                 │
│  Renal failure patients:                                        │
│  • M6G accumulates → Prolonged, enhanced opioid effect         │
│  • Risk of respiratory depression                               │
│  • Reduce morphine dose or choose alternative (fentanyl)       │
│                                                                 │
│  M3G may contribute to:                                         │
│  • Opioid tolerance                                             │
│  • Neuroexcitation (myoclonus, allodynia)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Glucuronidation and Drug Interactions

```
UGT INDUCTION:
  • Rifampin (via PXR) - induces multiple UGTs
  • Phenobarbital (via CAR)
  • Ritonavir (some UGTs)

UGT INHIBITION:
  • Probenecid (competes for glucuronidation)
  • Valproic acid (inhibits UGT2B7)
  • Atazanavir (inhibits UGT1A1 → hyperbilirubinemia!)

ENTEROHEPATIC CIRCULATION:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Drug-Glucuronide ──BILE──► INTESTINE                          │
│                                   │                             │
│                                   │ β-glucuronidase             │
│                                   │ (bacterial)                 │
│                                   ▼                             │
│                              Drug (free)                        │
│                                   │                             │
│                                   │ Reabsorbed                  │
│                                   ▼                             │
│                           PORTAL CIRCULATION                    │
│                                   │                             │
│                                   ▼                             │
│                                LIVER                            │
│                                   │                             │
│                           Re-glucuronidation                    │
│                                   │                             │
│                                   ▼                             │
│                         CYCLE CONTINUES...                      │
│                                                                 │
│  This prolongs drug half-life and creates multiple peaks       │
│  Examples: Mycophenolate, Ethinyl estradiol, some NSAIDs      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. SULT (Sulfotransferases) - The High-Affinity Pathway

### 4.1 Overview

```
SULFATION REACTION:

  Drug-OH + PAPS ──SULT──► Drug-O-SO₃⁻ + PAP
  
  (PAPS = 3'-Phosphoadenosine-5'-phosphosulfate)
  
Location: CYTOSOL (unlike UGTs which are in ER)

Characteristics:
  • HIGH AFFINITY (low Km) - works at low drug concentrations
  • LOW CAPACITY (limited PAPS supply)
  • Complements glucuronidation
```

### 4.2 SULT Families

```
MAJOR SULT ENZYMES:

┌──────────┬─────────────────────────────────────────────────────┐
│ ENZYME   │ SUBSTRATES                                          │
├──────────┼─────────────────────────────────────────────────────┤
│ SULT1A1  │ Small phenols: Acetaminophen, Minoxidil            │
│          │ Catecholamines, Estrogens                           │
│          │ MOST IMPORTANT for drug metabolism                  │
├──────────┼─────────────────────────────────────────────────────┤
│ SULT1A3  │ Catecholamines (dopamine, norepinephrine)          │
│          │ Phenylephrine                                       │
├──────────┼─────────────────────────────────────────────────────┤
│ SULT1E1  │ Estrogens (estrone, estradiol)                     │
│          │ Important in breast cancer                          │
├──────────┼─────────────────────────────────────────────────────┤
│ SULT2A1  │ DHEA, Androgens, Bile acids                        │
│          │ (Hydroxysteroid sulfotransferase)                   │
└──────────┴─────────────────────────────────────────────────────┘
```

### 4.3 Sulfation vs Glucuronidation

```
ACETAMINOPHEN METABOLISM - Classic Example:

                    ACETAMINOPHEN
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
       SULFATION    GLUCURONIDATION   CYP2E1
       (SULT1A1)    (UGT1A6, 1A9)       │
          │              │              │
          │              │              ▼
          │              │           NAPQI
          │              │         (toxic!)
          │              │              │
          ▼              ▼              ▼
    APAP-Sulfate   APAP-Glucuronide   GSH conjugate
    (~30% at low   (~55% at normal    (~5% normally)
     doses)         doses)

AT LOW DOSES:
  Sulfation dominates (high affinity)
  
AT HIGH DOSES:
  Sulfation saturates (limited PAPS)
  Glucuronidation takes over
  
AT TOXIC DOSES:
  Both pathways saturate
  More goes through CYP2E1 → NAPQI → TOXICITY
```

---

## 5. GST (Glutathione S-Transferases) - The Detoxifier

### 5.1 Overview

```
GLUTATHIONE CONJUGATION:

  Electrophile + GSH ──GST──► GS-conjugate
  
  GSH = γ-Glutamyl-Cysteinyl-Glycine (tripeptide)
  
             O    H O       H O
             ‖    │ ‖       │ ‖
  HOOC─CH─(CH₂)₂─C─N─CH─C─N─CH₂─COOH
       │             │
      NH₂           SH  ◄── Reactive thiol
       
      Glu          Cys      Gly

Function: DETOXIFICATION of reactive metabolites
Location: Cytosol (major), ER, mitochondria, nucleus
```

### 5.2 GST Classes

```
HUMAN GST CLASSES:

┌─────────┬─────────────────────────────────────────────────────┐
│ CLASS   │ FUNCTION                                            │
├─────────┼─────────────────────────────────────────────────────┤
│ GSTA    │ Alpha - Major hepatic class                         │
│         │ Detoxifies lipid peroxidation products              │
├─────────┼─────────────────────────────────────────────────────┤
│ GSTM    │ Mu - Detoxifies carcinogens                        │
│         │ GSTM1 null: 50% of population!                      │
│         │ ↑ cancer risk from polycyclic aromatic hydrocarbons│
├─────────┼─────────────────────────────────────────────────────┤
│ GSTP    │ Pi - Major extrahepatic class                      │
│         │ Overexpressed in tumors (drug resistance)          │
├─────────┼─────────────────────────────────────────────────────┤
│ GSTT    │ Theta - Halogenated compounds                      │
│         │ GSTT1 null: 20-60% depending on ethnicity          │
└─────────┴─────────────────────────────────────────────────────┘
```

### 5.3 Clinical Importance

```
NAPQI DETOXIFICATION (Acetaminophen Toxicity):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NAPQI + GSH ──GST──► NAPQI-GSH conjugate ──► Mercapturic acid ──► Urine
(toxic)                (non-toxic)

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Normal: Sufficient GSH to detoxify NAPQI                      │
│                                                                 │
│  Overdose: GSH depleted → NAPQI binds to hepatocyte proteins  │
│           → HEPATOCYTE NECROSIS                                │
│                                                                 │
│  Treatment: N-ACETYLCYSTEINE (NAC)                             │
│            • NAC → Cysteine → GSH synthesis                    │
│            • Repletes GSH stores                                │
│            • Most effective within 8 hours of overdose         │
│                                                                 │
│  GSH Depletion Risk Factors:                                   │
│  • Chronic alcohol use (↓ GSH synthesis)                       │
│  • Fasting/malnutrition (↓ cysteine availability)             │
│  • HIV (chronic oxidative stress)                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

CANCER DRUG RESISTANCE:
  • GSTP1 overexpression in tumors
  • Conjugates and inactivates:
    - Cisplatin
    - Doxorubicin
    - Chlorambucil
  • Target for cancer therapy enhancement
```

---

## 6. NAT (N-Acetyltransferases) - The Acetylator Pathway

### 6.1 Overview

```
ACETYLATION REACTION:

  Drug-NH₂ + Acetyl-CoA ──NAT──► Drug-NH-COCH₃ + CoA
  
Location: Cytosol
Tissues: Liver, intestine
```

### 6.2 NAT Polymorphisms - The "Fast" and "Slow" Acetylators

```
NAT2 ACETYLATOR STATUS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BIMODAL DISTRIBUTION (one of the first pharmacogenetic discoveries!)

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Population Distribution:                                       │
│                                                                 │
│  ▲                                                              │
│  │     ┌───┐                    ┌───┐                          │
│  │     │   │                    │   │                          │
│  │     │   │                    │   │                          │
│  │   ┌─┤   ├─┐              ┌───┤   ├───┐                      │
│  │   │ │   │ │              │   │   │   │                      │
│  └───┴─┴───┴─┴──────────────┴───┴───┴───┴───► Acetylation Rate │
│         SLOW                     FAST                          │
│       (50% White)             (50% White)                      │
│       (10% Asian)             (90% Asian)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

CLINICAL IMPLICATIONS:

ISONIAZID (Classic Example):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Isoniazid ──NAT2──► Acetyl-isoniazid ──► Hydrolysis          │
│                                               │                 │
│                                               ▼                 │
│                                         Acetylhydrazine        │
│                                               │                 │
│                                          CYP2E1                │
│                                               │                 │
│                                               ▼                 │
│                                         Toxic metabolites      │
│                                               │                 │
│                                               ▼                 │
│                                         HEPATOTOXICITY         │
│                                                                 │
│  SLOW ACETYLATORS:                                             │
│  • ↑ Isoniazid levels → ↑ Efficacy but ↑ Peripheral neuropathy│
│  • Paradoxically, may have ↓ hepatotoxicity (less hydrazine)  │
│                                                                 │
│  FAST ACETYLATORS:                                             │
│  • ↓ Isoniazid levels → May need higher doses                 │
│  • ↑ Acetylhydrazine → ↑ Hepatotoxicity risk                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Other NAT2 Substrates:
  • Hydralazine (SLE-like syndrome in slow acetylators)
  • Procainamide (drug-induced lupus)
  • Sulfonamides
  • Dapsone
  • Caffeine (minor pathway)
```

---

## 7. Methylation

### 7.1 Overview

```
METHYLATION REACTIONS:

  Drug-OH/NH₂/SH + SAM ──MT──► Drug-OCH₃/NHCH₃/SCH₃ + SAH
  
  SAM = S-Adenosylmethionine (universal methyl donor)
  SAH = S-Adenosylhomocysteine
  
Location: Cytosol
```

### 7.2 Key Methyltransferases

```
IMPORTANT METHYLTRANSFERASES:

┌──────────┬─────────────────────────────────────────────────────┐
│ ENZYME   │ SUBSTRATES & NOTES                                  │
├──────────┼─────────────────────────────────────────────────────┤
│ COMT     │ Catecholamines: L-DOPA, Dopamine, Norepinephrine   │
│          │ Target for Parkinson's therapy (entacapone)         │
│          │ COMT polymorphisms affect pain sensitivity         │
├──────────┼─────────────────────────────────────────────────────┤
│ TPMT     │ Thiopurines: 6-Mercaptopurine, Azathioprine        │
│          │ CRITICAL pharmacogenetic enzyme                     │
│          │ TPMT deficiency → FATAL myelosuppression           │
├──────────┼─────────────────────────────────────────────────────┤
│ HNMT     │ Histamine                                           │
│          │ Primarily in brain                                  │
├──────────┼─────────────────────────────────────────────────────┤
│ AS3MT    │ Arsenic methylation                                 │
│          │ Detoxification pathway                              │
└──────────┴─────────────────────────────────────────────────────┘
```

### 7.3 TPMT - Critical Pharmacogenetics

```
TPMT AND THIOPURINE TOXICITY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6-Mercaptopurine (6-MP) metabolism:

                6-MP
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
  TPMT        HPRT         XO
    │            │            │
    ▼            ▼            ▼
  6-MeMP     6-TGN      6-Thiouric acid
  (inactive)  (active      (inactive)
              cytotoxic)
              
TPMT POLYMORPHISMS:
┌─────────────────────────────────────────────────────────────────┐
│ PHENOTYPE           │ FREQUENCY │ 6-TGN LEVELS │ TOXICITY      │
├─────────────────────┼───────────┼──────────────┼───────────────┤
│ High activity       │ ~89%      │ Normal       │ Standard dose │
├─────────────────────┼───────────┼──────────────┼───────────────┤
│ Intermediate (het)  │ ~10%      │ ↑            │ Reduce 30-50% │
├─────────────────────┼───────────┼──────────────┼───────────────┤
│ Deficient (hom)     │ ~0.3%     │ ↑↑↑          │ Reduce 90%!   │
│                     │ (1:300)   │              │ or fatal      │
└─────────────────────┴───────────┴──────────────┴───────────────┘

FDA REQUIREMENT:
  • TPMT testing recommended before starting thiopurines
  • Package insert includes dosing recommendations by genotype
  • CPIC guidelines available
```

---

## 8. Amino Acid Conjugation

### 8.1 Glycine Conjugation

```
GLYCINE CONJUGATION:

  R-COOH + Glycine ──► R-CO-NH-CH₂-COOH
  
Important for:
  • Benzoic acid → Hippuric acid
  • Salicylic acid → Salicyluric acid
  
Requires:
  • Acyl-CoA formation first
  • Then conjugation with glycine
  
Location: Mitochondria
```

### 8.2 Other Amino Acid Conjugations

```
TAURINE CONJUGATION:
  • Bile acids: Cholate → Taurocholate
  • Important for bile acid solubility

GLUTAMINE CONJUGATION:
  • Phenylacetic acid → Phenylacetylglutamine
  • Alternative to glycine in some species
```

---

## 9. Phase II Summary Table

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE II METABOLISM SUMMARY                               │
├───────────┬────────────┬─────────────┬──────────────┬───────────────────────┤
│ PATHWAY   │ LOCATION   │ CAPACITY    │ KEY DRUGS    │ CLINICAL NOTES        │
├───────────┼────────────┼─────────────┼──────────────┼───────────────────────┤
│ Glucuro-  │ ER         │ High        │ Morphine     │ Enterohepatic         │
│ nidation  │            │             │ Irinotecan   │ circulation possible  │
│           │            │             │ Bilirubin    │ UGT1A1*28 important   │
├───────────┼────────────┼─────────────┼──────────────┼───────────────────────┤
│ Sulfation │ Cytosol    │ Low         │ Acetaminophen│ Saturates at high     │
│           │            │ (limited    │ Minoxidil    │ doses                 │
│           │            │  PAPS)      │              │                       │
├───────────┼────────────┼─────────────┼──────────────┼───────────────────────┤
│ GSH       │ Cytosol    │ High        │ NAPQI        │ Depleted in overdose  │
│ conjugate │            │ (if GSH     │ Reactive     │ NAC replenishes       │
│           │            │  available) │ metabolites  │                       │
├───────────┼────────────┼─────────────┼──────────────┼───────────────────────┤
│ Acetyl-   │ Cytosol    │ Variable    │ Isoniazid    │ Fast/slow acetylator  │
│ ation     │            │ (genetic)   │ Hydralazine  │ polymorphism          │
├───────────┼────────────┼─────────────┼──────────────┼───────────────────────┤
│ Methyl-   │ Cytosol    │ Variable    │ 6-MP         │ TPMT deficiency fatal │
│ ation     │            │ (genetic)   │ L-DOPA       │ Testing recommended   │
└───────────┴────────────┴─────────────┴──────────────┴───────────────────────┘
```

---

## 10. Phase II and Liver Kp

```
PHASE II EFFECTS ON LIVER Kp:

1. METABOLITE TRAPPING:
   • Conjugates are charged (glucuronides: COO⁻, sulfates: SO₃⁻)
   • Can't passively diffuse out of hepatocyte
   • Must use TRANSPORTERS (MRP2, MRP3, BCRP)
   
   Parent drug (lipophilic) → Conjugate (trapped until effluxed)
   
   This creates HIGHER hepatocyte concentrations of conjugates

2. COMPETITION EFFECTS:
   • Multiple drugs competing for UGTs
   • Cofactor depletion at high doses
   • May affect parent drug Kp indirectly

3. PROTEIN BINDING:
   • Glucuronides generally LOW protein binding
   • Different Kp than parent compound

4. FEATURES FOR ML MODELS:
   • is_glucuronidation_substrate
   • is_sulfation_substrate  
   • contains_phenol (UGT/SULT substrate)
   • contains_carboxylic_acid (glycine conjugation)
   • contains_amine (NAT substrate)
```

---

**NEXT**: Part 5 - Hepatic Lipid Metabolism
