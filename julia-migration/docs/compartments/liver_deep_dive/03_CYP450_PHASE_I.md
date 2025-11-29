# THE CYP450 SYSTEM: Phase I Oxidative Metabolism

## 1. Overview: The Cytochrome P450 Superfamily

The CYP450 enzymes are **heme-containing monooxygenases** that catalyze:
- **Oxidation** (most common)
- Reduction
- Hydrolysis
- Isomerization

```
GENERAL CYP450 REACTION:

    R-H + O₂ + NADPH + H⁺ ──CYP450──► R-OH + H₂O + NADP⁺
    
    (substrate)  (oxygen)  (electron donor)    (hydroxylated product)

The enzyme inserts ONE oxygen atom into the substrate,
the other oxygen atom becomes water.
Hence: "MONOOXYGENASE"
```

### 1.1 Nomenclature

```
CYP 3 A 4
 │  │ │ │
 │  │ │ └── Individual enzyme (gene number)
 │  │ └──── Subfamily (>55% amino acid identity)
 │  └────── Family (>40% amino acid identity)
 └───────── Cytochrome P450

Examples:
  CYP1A2  - Family 1, Subfamily A, Gene 2
  CYP2D6  - Family 2, Subfamily D, Gene 6
  CYP3A4  - Family 3, Subfamily A, Gene 4
```

### 1.2 Location in the Hepatocyte

```
                    HEPATOCYTE
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │    ┌─────────────────────────────────────────┐  │
    │    │    SMOOTH ENDOPLASMIC RETICULUM (SER)   │  │
    │    │                                         │  │
    │    │   ╔════════════════════════════════╗   │  │
    │    │   ║        CYP450 ENZYMES          ║   │  │
    │    │   ║                                ║   │  │
    │    │   ║  ┌─────┐ ┌─────┐ ┌─────┐      ║   │  │
    │    │   ║  │CYP  │ │CYP  │ │CYP  │ ...  ║   │  │
    │    │   ║  │3A4  │ │2D6  │ │2C9  │      ║   │  │
    │    │   ║  └──┬──┘ └──┬──┘ └──┬──┘      ║   │  │
    │    │   ║     │       │       │         ║   │  │
    │    │   ║  NADPH-CYP450 REDUCTASE       ║   │  │
    │    │   ║  (electron transfer)          ║   │  │
    │    │   ║     │       │       │         ║   │  │
    │    │   ║     └───────┴───────┘         ║   │  │
    │    │   ║           NADPH               ║   │  │
    │    │   ╚════════════════════════════════╝   │  │
    │    │                                         │  │
    │    └─────────────────────────────────────────┘  │
    │                                                  │
    │    Also present in: Mitochondria (CYP11, CYP27) │
    │                                                  │
    └──────────────────────────────────────────────────┘

NOTE: CYP450 is membrane-bound, anchored in the SER membrane
      The active site faces the cytoplasm
      Zone 3 hepatocytes (centrilobular) have MORE CYP450!
```

---

## 2. The Major Drug-Metabolizing CYP450s

### 2.1 Relative Contributions

```
FRACTION OF DRUGS METABOLIZED BY EACH CYP450:

CYP3A4/5  ████████████████████████████████████████░░░░░  ~50%
CYP2D6    ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~25%
CYP2C9    ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~15%
CYP2C19   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~8%
CYP1A2    ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~5%
CYP2B6    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~3%
Others    █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~2%

HEPATIC EXPRESSION (% of total CYP):
CYP3A4    ████████████████████████████████░░░░░░░░░░░░  ~30%
CYP2C9    ██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░  ~20%
CYP2C8    ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~10%
CYP2E1    ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~10%
CYP1A2    ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~13%
CYP2D6    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~2%
CYP2A6    ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~4%
```

### 2.2 CYP3A4/5 - "The Master Metabolizer"

```
CYP3A4 (and CYP3A5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Location: Liver (Zone 3 > Zone 1), Intestine
Expression: ~30% of hepatic CYP, ~70% of intestinal CYP

Substrates (HUGE list - >50% of drugs!):
┌─────────────────────────────────────────────────────────────┐
│ Cardiovascular:                                             │
│   Statins: Atorvastatin, Simvastatin, Lovastatin           │
│   CCBs: Nifedipine, Amlodipine, Diltiazem, Verapamil      │
│   Others: Amiodarone, Quinidine, Lidocaine                 │
│                                                             │
│ Immunosuppressants:                                         │
│   Cyclosporine, Tacrolimus, Sirolimus, Everolimus          │
│                                                             │
│ HIV Antivirals:                                             │
│   Ritonavir, Saquinavir, Indinavir, Lopinavir              │
│                                                             │
│ Benzodiazepines:                                            │
│   Midazolam, Triazolam, Alprazolam (NOT lorazepam!)        │
│                                                             │
│ Opioids:                                                    │
│   Fentanyl, Alfentanil, Methadone, Oxycodone               │
│                                                             │
│ Anticancer:                                                 │
│   Docetaxel, Paclitaxel, Vincristine, Imatinib             │
│                                                             │
│ Steroids:                                                   │
│   Testosterone, Cortisol, Estradiol, Progesterone          │
│                                                             │
│ Macrolides:                                                 │
│   Erythromycin, Clarithromycin (NOT azithromycin!)         │
└─────────────────────────────────────────────────────────────┘

Inhibitors:
  STRONG: Ketoconazole, Itraconazole, Ritonavir, Clarithromycin
  MODERATE: Erythromycin, Diltiazem, Verapamil, Grapefruit juice
  WEAK: Cimetidine

Inducers:
  STRONG: Rifampin, Phenytoin, Carbamazepine, St. John's Wort
  MODERATE: Efavirenz, Phenobarbital

Active Site Characteristics:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  CYP3A4 has a LARGE, FLEXIBLE active site                  │
│                                                             │
│  Volume: ~1400 Å³ (largest of the CYPs)                    │
│                                                             │
│  Can accommodate:                                           │
│  • Small molecules (MW ~200)                                │
│  • Large molecules (MW >1200, e.g., cyclosporine)          │
│  • Multiple substrates simultaneously!                      │
│                                                             │
│  This explains its broad substrate specificity             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Intestinal First-Pass:
  • CYP3A4 in enterocytes = "Pre-hepatic" first-pass
  • Grapefruit juice inhibits intestinal CYP3A4 (not hepatic)
  • This increases oral bioavailability of CYP3A4 substrates
```

### 2.3 CYP2D6 - "The Polymorphic Enzyme"

```
CYP2D6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Location: Liver only (NOT in intestine)
Expression: Only ~2% of hepatic CYP (but metabolizes 25% of drugs!)

Substrates:
┌─────────────────────────────────────────────────────────────┐
│ Cardiovascular:                                             │
│   Beta-blockers: Metoprolol, Propranolol, Carvedilol       │
│   Antiarrhythmics: Flecainide, Propafenone                 │
│                                                             │
│ CNS:                                                        │
│   Antidepressants: Many TCAs, Venlafaxine, Fluoxetine      │
│   Antipsychotics: Haloperidol, Risperidone, Aripiprazole   │
│                                                             │
│ Opioids:                                                    │
│   Codeine → Morphine (PRODRUG ACTIVATION!)                 │
│   Tramadol → O-desmethyltramadol                           │
│   Hydrocodone → Hydromorphone                              │
│                                                             │
│ Antiemetics:                                                │
│   Ondansetron                                               │
│                                                             │
│ Antihistamines:                                             │
│   Diphenhydramine, Promethazine                            │
└─────────────────────────────────────────────────────────────┘

CRITICAL: CYP2D6 POLYMORPHISMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

>100 allelic variants identified!

Phenotypes:
┌─────────────────────────────────────────────────────────────┐
│ PHENOTYPE          │ FREQUENCY   │ ENZYME ACTIVITY         │
├────────────────────┼─────────────┼─────────────────────────┤
│ Poor Metabolizer   │ 5-10% White │ None (null alleles)     │
│ (PM)               │ 1-2% Asian  │                         │
│                    │             │ e.g., *4/*4, *5/*5      │
├────────────────────┼─────────────┼─────────────────────────┤
│ Intermediate       │ 10-15%      │ Reduced                 │
│ Metabolizer (IM)   │             │ e.g., *4/*10            │
├────────────────────┼─────────────┼─────────────────────────┤
│ Extensive          │ 70-80%      │ Normal                  │
│ Metabolizer (EM)   │             │ e.g., *1/*1             │
├────────────────────┼─────────────┼─────────────────────────┤
│ Ultrarapid         │ 1-3% White  │ High (gene duplication) │
│ Metabolizer (UM)   │ 20% Ethiopian│                        │
│                    │             │ e.g., *1/*1xN           │
└─────────────────────┼─────────────┼─────────────────────────┘

Clinical Impact - CODEINE Example:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Codeine ───CYP2D6───► Morphine (active analgesic)         │
│                                                             │
│  PM (5%):   No conversion → NO PAIN RELIEF                 │
│  UM (3%):   Rapid conversion → MORPHINE TOXICITY           │
│                                                             │
│  FDA BLACK BOX WARNING:                                     │
│  • Contraindicated in children <12 years                   │
│  • Avoid in breastfeeding mothers (infant deaths!)         │
│  • Consider CYP2D6 testing before prescribing              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

NOT Inducible:
  CYP2D6 is NOT INDUCIBLE by xenobiotics
  (Unlike CYP3A4, which is highly inducible)
```

### 2.4 CYP2C9 - "The Warfarin Enzyme"

```
CYP2C9
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Location: Liver, minor in intestine
Expression: ~20% of hepatic CYP

Substrates:
┌─────────────────────────────────────────────────────────────┐
│ Anticoagulants:                                             │
│   S-Warfarin (the active enantiomer!)                      │
│                                                             │
│ NSAIDs:                                                     │
│   Ibuprofen, Diclofenac, Celecoxib, Meloxicam, Piroxicam  │
│                                                             │
│ Oral Hypoglycemics:                                         │
│   Tolbutamide, Glipizide, Glimepiride                      │
│                                                             │
│ Antiepileptics:                                             │
│   Phenytoin                                                 │
│                                                             │
│ ARBs:                                                       │
│   Losartan → Active metabolite                             │
│   Irbesartan                                                │
└─────────────────────────────────────────────────────────────┘

Inhibitors:
  Fluconazole (moderate), Amiodarone, Sulfinpyrazone

CRITICAL: CYP2C9 POLYMORPHISMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Major Variants:
┌─────────────────────────────────────────────────────────────┐
│ ALLELE  │ FREQUENCY     │ ENZYME ACTIVITY  │ WARFARIN      │
├─────────┼───────────────┼──────────────────┼───────────────┤
│ *1      │ ~80% White    │ Normal (100%)    │ Standard dose │
├─────────┼───────────────┼──────────────────┼───────────────┤
│ *2      │ 8-13% White   │ Reduced (~70%)   │ ↓ dose 20%    │
│         │ <1% Asian     │                  │               │
├─────────┼───────────────┼──────────────────┼───────────────┤
│ *3      │ 6-10% White   │ Low (~5-10%)     │ ↓ dose 40%    │
│         │ <1% Asian     │                  │               │
└─────────┴───────────────┴──────────────────┴───────────────┘

WARFARIN DOSING: CYP2C9 + VKORC1
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  CYP2C9 genotype + VKORC1 genotype → Warfarin dose         │
│                                                             │
│  FDA label includes dosing table based on genotype         │
│  Pharmacogenomic dosing can reduce bleeding events         │
│                                                             │
│  Example:                                                   │
│  CYP2C9 *1/*1 + VKORC1 GG: 5-7 mg/day                     │
│  CYP2C9 *3/*3 + VKORC1 AA: 0.5-2 mg/day (10x less!)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.5 CYP2C19 - "The PPI Enzyme"

```
CYP2C19
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Location: Liver, minor in intestine
Expression: ~2-5% of hepatic CYP

Substrates:
┌─────────────────────────────────────────────────────────────┐
│ PPIs:                                                       │
│   Omeprazole, Esomeprazole, Lansoprazole, Pantoprazole    │
│                                                             │
│ Antiplatelet:                                               │
│   Clopidogrel → ACTIVE METABOLITE (prodrug!)               │
│                                                             │
│ Antidepressants:                                            │
│   Citalopram, Escitalopram, Amitriptyline                  │
│                                                             │
│ Antifungals:                                                │
│   Voriconazole                                              │
│                                                             │
│ Antiepileptics:                                             │
│   Phenytoin (minor), Diazepam                              │
└─────────────────────────────────────────────────────────────┘

CRITICAL: CYP2C19 POLYMORPHISMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────┐
│ PHENOTYPE    │ WHITE    │ ASIAN    │ ALLELES               │
├──────────────┼──────────┼──────────┼───────────────────────┤
│ PM           │ 2-5%     │ 15-25%   │ *2/*2, *2/*3, *3/*3   │
├──────────────┼──────────┼──────────┼───────────────────────┤
│ IM           │ 25-35%   │ 40-50%   │ *1/*2, *1/*3          │
├──────────────┼──────────┼──────────┼───────────────────────┤
│ EM           │ 35-50%   │ 35-45%   │ *1/*1                 │
├──────────────┼──────────┼──────────┼───────────────────────┤
│ UM           │ 5-30%    │ <5%      │ *17/*17, *1/*17       │
└──────────────┴──────────┴──────────┴───────────────────────┘

CLOPIDOGREL - The Critical Example:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Clopidogrel (prodrug) ──CYP2C19──► Active thiol metabolite│
│                                                             │
│  PM: Poor activation → HIGH CARDIOVASCULAR EVENT RISK      │
│  UM: Rapid activation → May have MORE bleeding             │
│                                                             │
│  FDA BLACK BOX WARNING (2010):                             │
│  "Effectiveness depends on CYP2C19 status"                 │
│  Consider alternative for PMs (prasugrel, ticagrelor)      │
│                                                             │
│  Meta-analysis: PM have 1.5-3× higher stent thrombosis    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

PPI Implications:
  PM: Higher PPI levels → Better H. pylori eradication!
  UM: Lower PPI levels → May need higher doses
```

### 2.6 CYP1A2 - "The Caffeine Enzyme"

```
CYP1A2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Location: Liver only (NOT intestine)
Expression: ~13% of hepatic CYP

Substrates:
┌─────────────────────────────────────────────────────────────┐
│ Xanthines:                                                  │
│   Caffeine (probe substrate)                               │
│   Theophylline                                              │
│                                                             │
│ Antipsychotics:                                             │
│   Clozapine, Olanzapine                                    │
│                                                             │
│ Antidepressants:                                            │
│   Duloxetine, Fluvoxamine, Amitriptyline                   │
│                                                             │
│ Others:                                                     │
│   Melatonin, Tizanidine, Ropinirole                        │
│                                                             │
│ PROCARCINOGENS (bioactivation):                            │
│   Aflatoxin B1, Heterocyclic amines (grilled meat)        │
│   Polycyclic aromatic hydrocarbons (smoke)                 │
└─────────────────────────────────────────────────────────────┘

Inducers (CYP1A2 is HIGHLY INDUCIBLE):
  • SMOKING (1.5-2× induction) - Most clinically important!
  • Chargrilled meat
  • Cruciferous vegetables (broccoli, Brussels sprouts)
  • Omeprazole (high doses)

Inhibitors:
  • Fluvoxamine (STRONG - 5-10× increase in substrate levels)
  • Ciprofloxacin, Norfloxacin
  • Oral contraceptives

SMOKING AND CYP1A2:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Polycyclic aromatic hydrocarbons in smoke → AhR → CYP1A2 │
│                                                             │
│  Effect: 1.5-2× INDUCTION of CYP1A2                        │
│                                                             │
│  Clinical Examples:                                         │
│                                                             │
│  CLOZAPINE:                                                 │
│  • Smokers need ~50% higher doses                          │
│  • SMOKING CESSATION → clozapine toxicity!                 │
│  • Must reduce dose when patient quits smoking             │
│                                                             │
│  THEOPHYLLINE:                                              │
│  • Smokers have faster clearance                           │
│  • Quit smoking → theophylline toxicity                    │
│                                                             │
│  Note: It's the SMOKE, not the nicotine                    │
│        Nicotine patches do NOT induce CYP1A2               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.7 CYP2E1 - "The Ethanol Enzyme"

```
CYP2E1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Location: Liver (Zone 3 - highest!), also lung, brain
Expression: ~10% of hepatic CYP

Substrates:
┌─────────────────────────────────────────────────────────────┐
│ Endogenous:                                                 │
│   Ethanol → Acetaldehyde                                   │
│   Fatty acids (ω-1 hydroxylation)                          │
│   Ketones (diabetic ketoacidosis)                          │
│                                                             │
│ Drugs:                                                      │
│   Acetaminophen → NAPQI (toxic metabolite!)                │
│   Chlorzoxazone (probe)                                    │
│   Isoniazid                                                 │
│   Anesthetics: Sevoflurane, Isoflurane, Enflurane         │
│                                                             │
│ Solvents/Toxins:                                           │
│   Benzene, Carbon tetrachloride, Vinyl chloride            │
│   Nitrosamines (tobacco-specific)                          │
└─────────────────────────────────────────────────────────────┘

Inducers:
  • ETHANOL (chronic use) - Protein stabilization
  • Isoniazid
  • Fasting/Diabetes/Ketosis

ACETAMINOPHEN TOXICITY:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Acetaminophen ──► 90% Conjugation (safe) → Excretion     │
│        │                                                    │
│        └──CYP2E1──► NAPQI (toxic!) ──GSH──► Safe excretion│
│                         │                                   │
│                         │ If GSH depleted:                  │
│                         ▼                                   │
│                    HEPATOCYTE DEATH                         │
│                                                             │
│  CYP2E1 INDUCTION (chronic alcohol, fasting):              │
│  • More NAPQI produced                                      │
│  • Lower threshold for toxicity                            │
│                                                             │
│  This is why alcoholics are at higher risk for             │
│  acetaminophen-induced liver failure!                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Reactive Oxygen Species:
  CYP2E1 is "leaky" - produces ROS during catalysis
  → Contributes to alcoholic liver disease
  → Zone 3 necrosis pattern (highest CYP2E1)
```

---

## 3. CYP450 Catalytic Cycle

```
THE CYP450 CATALYTIC CYCLE (Detailed):

        ┌────────────────────────────────────────────────────────┐
        │                                                        │
        │                    Fe³⁺ (resting)                     │
        │                         │                              │
        │                         │ 1. Substrate (RH) binds     │
        │                         ▼                              │
        │                    Fe³⁺-RH                            │
        │                         │                              │
        │                         │ 2. First electron            │
        │                         │    (from NADPH reductase)    │
        │                         ▼                              │
        │                    Fe²⁺-RH                            │
        │                         │                              │
        │                         │ 3. O₂ binds                  │
        │                         ▼                              │
        │                   Fe²⁺-O₂-RH                          │
        │                         │                              │
        │                         │ 4. Second electron           │
        │                         │    (from NADPH or cyt b5)    │
        │                         ▼                              │
        │                  Fe²⁺-O₂²⁻-RH                         │
        │                         │                              │
        │                         │ 5. Protonation              │
        │                         ▼                              │
        │                  Fe³⁺-OOH-RH                          │
        │                         │                              │
        │                         │ 6. O-O bond cleavage        │
        │                         │    (H₂O released)            │
        │                         ▼                              │
        │               [Fe⁴⁺=O]•⁺-RH  ◄── COMPOUND I           │
        │              (Ferryl-oxo porphyrin                    │
        │               radical cation)                         │
        │                         │                              │
        │                         │ 7. Hydrogen abstraction     │
        │                         │    (R-H → R•)               │
        │                         ▼                              │
        │                  Fe⁴⁺-OH + R•                         │
        │                         │                              │
        │                         │ 8. Oxygen rebound           │
        │                         │    (R• + OH → R-OH)         │
        │                         ▼                              │
        │                    Fe³⁺ + R-OH                        │
        │                         │                              │
        │                         │ 9. Product (R-OH) released  │
        │                         ▼                              │
        │                    Fe³⁺ (resting)                     │
        │                                                        │
        └────────────────────────────────────────────────────────┘

Key Points:
  • Requires 2 electrons (from NADPH via reductase)
  • Requires O₂ 
  • Compound I is the actual oxidizing species
  • Highly reactive - can oxidize almost any C-H bond
```

---

## 4. Reactions Catalyzed by CYP450

### 4.1 Types of Oxidation Reactions

```
1. ALIPHATIC HYDROXYLATION
   R-CH₃ ──► R-CH₂-OH
   Example: Tolbutamide → Hydroxytolbutamide

2. AROMATIC HYDROXYLATION
   Ph-H ──► Ph-OH
   Example: Phenytoin → p-Hydroxyphenytoin

3. N-DEALKYLATION
   R-N(CH₃)₂ ──► R-NH-CH₃ + HCHO
   Example: Diazepam → Nordiazepam

4. O-DEALKYLATION
   R-O-CH₃ ──► R-OH + HCHO
   Example: Codeine → Morphine (CYP2D6)

5. S-OXIDATION
   R-S-R' ──► R-SO-R' (sulfoxide)
   Example: Omeprazole → Omeprazole sulfone

6. N-OXIDATION
   R₃N ──► R₃N→O (N-oxide)
   Example: Nicotine → Nicotine N-oxide

7. EPOXIDATION
   R-CH=CH-R' ──► R-CH─CH-R' (epoxide)
                     ╲O╱
   Example: Benzo[a]pyrene → Diol epoxide (carcinogenic!)

8. DEHALOGENATION
   R-CHCl₂ ──► R-CHO + 2Cl⁻
   Example: Chloroform → Phosgene (toxic!)
```

---

## 5. CYP450 Induction and Inhibition

### 5.1 Induction Mechanisms

```
MECHANISM OF CYP450 INDUCTION:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  INDUCER (e.g., Rifampin)                                  │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐                                               │
│  │   PXR   │  Pregnane X Receptor                          │
│  │ (NR1I2) │  (Master regulator of drug metabolism)        │
│  └────┬────┘                                               │
│       │                                                     │
│       │ Ligand binding                                      │
│       ▼                                                     │
│  ┌─────────────────┐                                       │
│  │  PXR + RXR      │  Heterodimerizes with RXR             │
│  │  (active dimer) │                                       │
│  └────────┬────────┘                                       │
│           │                                                 │
│           │ Binds to DNA response elements                 │
│           ▼                                                 │
│  ┌───────────────────────────────────────────┐             │
│  │ ══DR-3══DR-4══ER-6══ GENE PROMOTERS ═════ │             │
│  │                                           │             │
│  │   CYP3A4, CYP2C9, MDR1, MRP2, UGT1A1    │             │
│  └────────────────────┬──────────────────────┘             │
│                       │                                     │
│                       │ Transcription ↑                     │
│                       ▼                                     │
│              ↑ mRNA → ↑ Protein → ↑ Enzyme Activity        │
│                                                             │
│  Timeline: Days to reach maximal induction                 │
│  Offset: Days to weeks after inducer stopped               │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Other Nuclear Receptors:
  • CAR (Constitutive Androstane Receptor): Phenobarbital
  • AhR (Aryl Hydrocarbon Receptor): Smoke, CYP1A2
  • VDR (Vitamin D Receptor): CYP24A1
```

### 5.2 Inhibition Mechanisms

```
TYPES OF CYP450 INHIBITION:

1. REVERSIBLE COMPETITIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Inhibitor competes with substrate for active site
   • Effect is immediate
   • Effect reverses when inhibitor cleared
   
   Example: Ketoconazole + CYP3A4 substrates
   
   Kinetics: ↑ Km, unchanged Vmax

2. REVERSIBLE NON-COMPETITIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Inhibitor binds outside active site
   • Reduces enzyme efficiency
   
   Kinetics: unchanged Km, ↓ Vmax

3. MECHANISM-BASED INHIBITION (MBI) - "Suicide Inhibition"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Inhibitor is metabolized to reactive intermediate
   • Intermediate COVALENTLY binds to enzyme
   • IRREVERSIBLE - enzyme must be resynthesized
   
   Examples:
   ┌─────────────────────────────────────────────────────────┐
   │ DRUG            │ CYP AFFECTED │ RECOVERY TIME         │
   ├─────────────────┼──────────────┼───────────────────────┤
   │ Erythromycin    │ CYP3A4       │ Days                  │
   │ Diltiazem       │ CYP3A4       │ Days                  │
   │ Ritonavir       │ CYP3A4       │ Days                  │
   │ Paroxetine      │ CYP2D6       │ Weeks (PM created!)   │
   │ Ticlopidine     │ CYP2C19      │ Days                  │
   └─────────────────┴──────────────┴───────────────────────┘
   
   Clinical Importance:
   • Effect persists even after inhibitor stopped
   • Must wait for new enzyme synthesis
   • Half-life of enzyme (~1-3 days for most CYPs)
```

---

## 6. Clinical Drug-Drug Interactions

```
MAJOR CYP-MEDIATED DDIS TO KNOW:

┌─────────────────────────────────────────────────────────────────┐
│ INTERACTION                  │ EFFECT        │ CLINICAL ACTION │
├──────────────────────────────┼───────────────┼─────────────────┤
│ Simvastatin + Ketoconazole  │ AUC ↑ 15×     │ CONTRAINDICATED │
│ Simvastatin + Grapefruit    │ AUC ↑ 3×      │ Avoid large amts│
│ Midazolam + Ketoconazole    │ AUC ↑ 16×     │ Reduce dose 75% │
│ Tacrolimus + Rifampin       │ AUC ↓ 90%     │ ↑ dose 3-5×     │
│ Warfarin + Rifampin         │ INR ↓↓        │ ↑ dose ~50%     │
│ Theophylline + Ciprofloxacin│ AUC ↑ 2×      │ Reduce dose 50% │
│ Clozapine + Fluvoxamine     │ AUC ↑ 5-10×   │ Reduce dose 75% │
│ Codeine + Paroxetine        │ No morphine   │ No pain relief! │
│ Clopidogrel + Omeprazole    │ Less active   │ Use pantoprazole│
└──────────────────────────────┴───────────────┴─────────────────┘
```

---

## 7. Implications for Liver Kp

```
CYP450 AND LIVER Kp:

1. HIGH CYP EXPRESSION → DRUG ACCUMULATION
   • Drug is taken up into hepatocytes
   • Metabolism creates concentration gradient
   • Drives MORE uptake = HIGHER Kp
   
   Kp_liver_high_CLint > Kp_liver_low_CLint

2. METABOLITE DISTRIBUTION
   • Metabolites may have different Kp than parent
   • Must consider BOTH in PBPK models
   
3. ZONE-SPECIFIC METABOLISM
   • Zone 3: High CYP → High extraction, high metabolite
   • Zone 1: Lower CYP → Less extraction
   
   This affects LOCAL drug concentrations within liver

4. DRUG-INDUCED CHANGES
   • Inducers ↑ CYP → ↑ extraction → may ↑ Kp_liver
   • Inhibitors ↓ CYP → ↓ extraction → may ↓ Kp_liver
```

---

**NEXT**: Part 4 - Phase II Metabolism: Conjugation Reactions
