# KIDNEY PHYSIOLOGY DEEP DIVE
## From Gross Anatomy to Molecular Mechanisms for PBPK Modeling

**Author:** Darwin PBPK Platform  
**Purpose:** Definitive reference for kidney drug distribution and elimination modeling  
**Level:** MD/PhD - Comprehensive mechanistic understanding

---

## TABLE OF CONTENTS

1. [Gross Anatomy](#1-gross-anatomy)
2. [Vascular Architecture](#2-vascular-architecture)
3. [The Nephron - Functional Unit](#3-the-nephron---functional-unit)
4. [Glomerular Filtration](#4-glomerular-filtration)
5. [Proximal Tubule - The Drug Processing Hub](#5-proximal-tubule---the-drug-processing-hub)
6. [Loop of Henle & Countercurrent System](#6-loop-of-henle--countercurrent-system)
7. [Distal Tubule & Collecting Duct](#7-distal-tubule--collecting-duct)
8. [Cellular Ultrastructure](#8-cellular-ultrastructure)
9. [Membrane Transporters](#9-membrane-transporters)
10. [Drug Partitioning Mechanisms](#10-drug-partitioning-mechanisms)
11. [Renal Drug Clearance](#11-renal-drug-clearance)
12. [SOTA Discoveries (2020-2024)](#12-sota-discoveries-2020-2024)
13. [PBPK Model Parameters](#13-pbpk-model-parameters)
14. [Clinical Implications](#14-clinical-implications)

---

## 1. GROSS ANATOMY

### 1.1 Location and Structure

The kidneys are paired retroperitoneal organs located at T12-L3:
- **Dimensions:** 11 × 6 × 3 cm (adult)
- **Weight:** 120-170g each (total ~300g, 0.4% body weight)
- **Position:** Right kidney slightly lower due to liver

### 1.2 External Features

```
                    ANTERIOR VIEW
                    
        Adrenal gland
              ↓
    ┌─────────────────┐
    │   Superior pole │
    │                 │
    │      Hilum  ────┼──→ Renal artery
    │       ↓        │    Renal vein
    │    ═══════     │    Ureter
    │                 │
    │   Inferior pole │
    └─────────────────┘
```

### 1.3 Internal Structure

**Three distinct zones:**

| Zone | Location | % of Mass | Blood Flow | Function |
|------|----------|-----------|------------|----------|
| **Cortex** | Outer | 70% | 90% of RBF | Filtration, Secretion |
| **Outer Medulla** | Middle | 20% | 8% | Concentration |
| **Inner Medulla** | Central | 10% | 2% | Final concentration |

**Key structures:**
- **Renal columns (of Bertin):** Cortical tissue between pyramids
- **Medullary pyramids:** 8-18 per kidney, apex = papilla
- **Minor/Major calyces:** Collect urine from papillae
- **Renal pelvis:** Funnel-shaped collecting area → ureter

### 1.4 PBPK Relevance

**Why does gross anatomy matter for drug distribution?**

1. **Cortex receives 90% of blood flow** → Primary drug exposure site
2. **Medulla is relatively hypoxic** → Different metabolic environment
3. **Corticomedullary gradient** → Drugs concentrate differently by zone
4. **Regional transporter expression varies** → OAT/OCT primarily cortical

---

## 2. VASCULAR ARCHITECTURE

### 2.1 The Unique Renal Circulation

**The kidney receives 20-25% of cardiac output (1.0-1.2 L/min) despite being only 0.4% of body mass!**

This translates to:
- **Blood flow per gram:** 4 mL/g/min (compare: liver 0.8, brain 0.5, muscle 0.04)
- **Purpose:** NOT for oxygen delivery (kidney extracts only 10-15% of O₂)
- **Real purpose:** FILTRATION - 180 L/day of plasma filtered!

### 2.2 Vascular Pathway

```
Renal artery (1 per kidney)
    ↓
Segmental arteries (5)
    ↓
Interlobar arteries (between pyramids)
    ↓
Arcuate arteries (at cortico-medullary junction)
    ↓
Interlobular arteries (radiate into cortex)
    ↓
┌─────────────────────────────────────┐
│     AFFERENT ARTERIOLE              │
│            ↓                        │
│     ╔═══════════════╗               │
│     ║  GLOMERULUS   ║  ← FILTRATION │
│     ╚═══════════════╝               │
│            ↓                        │
│     EFFERENT ARTERIOLE              │
│            ↓                        │
│  ┌─────────────────────┐            │
│  │ PERITUBULAR         │            │
│  │ CAPILLARIES         │ ← REABSORPTION/SECRETION
│  │ (cortical nephrons) │            │
│  │        OR           │            │
│  │ VASA RECTA          │            │
│  │ (juxtamedullary)    │            │
│  └─────────────────────┘            │
└─────────────────────────────────────┘
    ↓
Venous return → Renal vein → IVC
```

### 2.3 Two Capillary Beds in Series

**UNIQUE TO KIDNEY:** Two capillary networks in sequence

1. **Glomerular capillaries:** High pressure (45-60 mmHg), FILTRATION
2. **Peritubular capillaries:** Low pressure (10-15 mmHg), REABSORPTION

This allows:
- Filtered drug in tubular fluid
- SAME drug in peritubular blood
- **SECRETION:** Drug can move from blood → tubule
- **REABSORPTION:** Drug can move from tubule → blood

### 2.4 Vasa Recta - The Countercurrent Exchanger

For juxtamedullary nephrons (15% of nephrons):
- Long thin vessels parallel to Loop of Henle
- **Descending vasa recta:** Loses water, gains solutes
- **Ascending vasa recta:** Gains water, loses solutes
- Maintains medullary hypertonicity (up to 1200 mOsm/kg)

**Drug implications:**
- Lipophilic drugs concentrate in hyperosmolar medulla
- Long residence time for medullary drug exposure
- Nephrotoxicity patterns differ by zone

### 2.5 Renal Blood Flow Regulation

**Autoregulation maintains GFR despite BP changes (80-180 mmHg):**

| Mechanism | Mediator | Effect |
|-----------|----------|--------|
| Myogenic | Stretch-activated channels | Afferent constriction |
| Tubuloglomerular feedback | Macula densa → adenosine | Afferent constriction |
| Prostaglandins | PGE₂, PGI₂ | Afferent dilation |
| Angiotensin II | AT1 receptor | Efferent constriction |
| Nitric oxide | eNOS | Vasodilation |

**NSAID nephrotoxicity:** Block PGE₂/PGI₂ → unopposed vasoconstriction → ↓GFR

---

## 3. THE NEPHRON - FUNCTIONAL UNIT

### 3.1 Nephron Count and Types

- **~1 million nephrons per kidney** (range: 0.3-1.4 million)
- **NO regeneration after birth** - nephron loss is permanent
- **15% lost per decade after age 40**

**Two nephron populations:**

| Type | Location | % | Loop Length | Blood Supply |
|------|----------|---|-------------|--------------|
| **Cortical** | Outer cortex | 85% | Short | Peritubular capillaries |
| **Juxtamedullary** | Inner cortex | 15% | Long (to papilla) | Vasa recta |

### 3.2 Nephron Segments

```
                NEPHRON ANATOMY
                
         CORTEX
    ┌─────────────────────────────────────┐
    │                                     │
    │     ┌───────────────┐               │
    │     │  GLOMERULUS   │               │
    │     │   (Bowman's)  │               │
    │     └───────┬───────┘               │
    │             ↓                       │
    │  ╔═══════════════════════╗          │
    │  ║   PROXIMAL TUBULE    ║ S1-S3    │
    │  ║   (Convoluted→Straight)        ║ │
    │  ╚═══════════╤═══════════╝          │
    └──────────────┼──────────────────────┘
                   ↓                OUTER
    ┌──────────────┼──────────────────────┐
    │         DESCENDING                  │
    │         THIN LIMB                   │
    │              ↓                      │ MEDULLA
    └──────────────┼──────────────────────┘
                   ↓                INNER
    ┌──────────────┼──────────────────────┐
    │         HAIRPIN TURN                │
    │              ↓                      │ MEDULLA
    │         ASCENDING                   │
    │         THIN LIMB                   │
    └──────────────┼──────────────────────┘
                   ↓                OUTER
    ┌──────────────┼──────────────────────┐
    │  ╔═══════════╧═══════════╗          │
    │  ║  THICK ASCENDING      ║          │ MEDULLA
    │  ║  LIMB (TAL)           ║          │
    │  ╚═══════════╤═══════════╝          │
    └──────────────┼──────────────────────┘
                   ↓                CORTEX
    ┌──────────────┼──────────────────────┐
    │       MACULA DENSA                  │
    │       (at glomerulus)               │
    │              ↓                      │
    │  ╔═══════════════════════╗          │
    │  ║   DISTAL CONVOLUTED   ║          │
    │  ║   TUBULE (DCT)        ║          │
    │  ╚═══════════╤═══════════╝          │
    │              ↓                      │
    │       CONNECTING TUBULE             │
    │              ↓                      │
    │  ╔═══════════════════════╗          │
    │  ║   COLLECTING DUCT     ║          │
    │  ╚═══════════╤═══════════╝          │
    └──────────────┼──────────────────────┘
                   ↓
              TO PELVIS
```

---

## 4. GLOMERULAR FILTRATION

### 4.1 The Filtration Barrier

**Three-layer barrier with remarkable selectivity:**

```
BLOOD                        URINARY SPACE
  ↓                              ↓
═══════════════════════════════════════
  ║                              ║
  ║   1. ENDOTHELIUM             ║
  ║   (fenestrated, 70-100nm)    ║
  ║   - Negatively charged       ║
  ║   - Glycocalyx               ║
  ║                              ║
═══════════════════════════════════════
  ║                              ║
  ║   2. BASEMENT MEMBRANE       ║
  ║   (GBM, 300-350nm thick)     ║
  ║   - Type IV collagen         ║
  ║   - Laminin, nidogen         ║
  ║   - Heparan sulfate (-)      ║
  ║   - SIZE + CHARGE barrier    ║
  ║                              ║
═══════════════════════════════════════
  ║                              ║
  ║   3. PODOCYTES               ║
  ║   (visceral epithelium)      ║
  ║   - Foot processes           ║
  ║   - Slit diaphragm (4-11nm)  ║
  ║   - Nephrin, podocin         ║
  ║   - FINAL size barrier       ║
  ║                              ║
═══════════════════════════════════════
```

### 4.2 Filtration Selectivity

**Size selectivity:**

| Molecular Weight | Radius | Filterability |
|------------------|--------|---------------|
| < 7 kDa | < 1.8 nm | 100% (freely filtered) |
| 7-70 kDa | 1.8-4.4 nm | Variable (charge-dependent) |
| > 70 kDa | > 4.4 nm | ~0% (not filtered) |

**Charge selectivity:**
- GBM and podocyte glycocalyx are **negatively charged**
- **Cations filtered more easily than anions** of same size
- Albumin (66 kDa, anionic): < 0.1% filtered despite "borderline" size

### 4.3 Glomerular Filtration Rate (GFR)

**GFR = Kf × (ΔP - Δπ)**

Where:
- Kf = Filtration coefficient (permeability × surface area)
- ΔP = Hydrostatic pressure gradient (PGC - PBS)
- Δπ = Oncotic pressure gradient (πGC - πBS)

**Normal values:**
- GFR ≈ 120-125 mL/min (180 L/day)
- Filtration fraction = GFR/RPF ≈ 20%

**Starling forces:**

| Force | Value (mmHg) | Direction |
|-------|--------------|-----------|
| PGC (glomerular hydrostatic) | 45-50 | Favors filtration |
| PBS (Bowman's space hydrostatic) | 10 | Opposes filtration |
| πGC (glomerular oncotic) | 25-30 | Opposes filtration |
| πBS (Bowman's oncotic) | 0 | - |
| **Net filtration pressure** | ~10 | Filtration |

### 4.4 Drug Filtration

**Only UNBOUND drug filters!**

CLfiltration = fu × GFR

Examples:
| Drug | fu | CLfilt (mL/min) | % of CLrenal |
|------|-----|-----------------|--------------|
| Metformin | 0.99 | 118 | 20% (secretion dominant!) |
| Digoxin | 0.75 | 90 | 100% |
| Warfarin | 0.01 | 1.2 | <1% (hepatic metabolism) |
| Gentamicin | 0.90 | 108 | >95% |

---

## 5. PROXIMAL TUBULE - THE DRUG PROCESSING HUB

### 5.1 Anatomy

**The proximal tubule (PT) is THE critical segment for drug handling:**

- **Length:** 14 mm
- **Diameter:** 50-65 μm
- **Surface area:** Massively increased by brush border
- **Receives:** 65-70% of filtered load

**Three segments with distinct functions:**

| Segment | Location | Primary Function |
|---------|----------|------------------|
| **S1** | Early convoluted | High-capacity transport, most OAT/OCT |
| **S2** | Late convoluted | Continued transport |
| **S3** | Straight (pars recta) | Lower capacity, entry to medulla |

### 5.2 Brush Border Membrane

**The apical brush border is one of the most remarkable membrane specializations:**

- **Microvilli:** 1 μm tall, 0.1 μm diameter
- **Density:** 5,000-10,000 per cell
- **Surface amplification:** 30-40× increase
- **Total PT surface area:** 50-60 m² (tennis court!)

**Molecular composition:**
- **Glycocalyx:** Anionic (sialic acid, heparan sulfate)
- **Enzymes:** γ-glutamyl transferase, alkaline phosphatase
- **Transporters:** See Section 9

### 5.3 Mitochondrial Density

**PT cells are among the most metabolically active in the body:**

- **Mitochondria:** 30-40% of cell volume (compare: hepatocyte 20%)
- **ATP consumption:** Powers Na⁺/K⁺-ATPase and transporters
- **Oxygen consumption:** Kidney = 7% of body O₂ use (for 0.4% of mass)

**Clinical relevance:**
- Mitochondrial toxins (tenofovir, cisplatin) → PT necrosis
- Hypoxia → Acute tubular necrosis (ATN)
- CKD → Mitochondrial dysfunction

### 5.4 Reabsorption Functions

**The PT reabsorbs 65-70% of filtered solutes:**

| Substance | % Reabsorbed in PT | Mechanism |
|-----------|-------------------|-----------|
| Water | 65% | Osmotic (follows Na⁺) |
| Na⁺ | 65% | Na⁺/K⁺-ATPase, NHE3 |
| Cl⁻ | 55% | Paracellular, Cl⁻/formate |
| K⁺ | 65% | Paracellular |
| Glucose | 100% | SGLT2 (S1-S2), SGLT1 (S3) |
| Amino acids | 99% | Multiple transporters |
| HCO₃⁻ | 80% | NHE3, carbonic anhydrase |
| Phosphate | 80% | NaPi-IIa, NaPi-IIc |
| Urate | 90% | URAT1, OAT4 (reabsorption) |

---

## 6. LOOP OF HENLE & COUNTERCURRENT SYSTEM

### 6.1 Structure

| Segment | Cell Type | Permeability | Transport |
|---------|-----------|--------------|-----------|
| **Thin descending** | Squamous | High H₂O, Low solute | Passive |
| **Thin ascending** | Squamous | Low H₂O, High NaCl | Passive |
| **Thick ascending (TAL)** | Cuboidal | Impermeable H₂O | Active NaCl |

### 6.2 Countercurrent Multiplication

**Creates the medullary osmotic gradient (300 → 1200 mOsm/kg):**

1. TAL actively pumps NaCl into interstitium (NKCC2)
2. Thin descending loses water to hypertonic interstitium
3. Thin ascending loses NaCl passively
4. **Result:** Progressive concentration of tubular fluid

### 6.3 Drug Implications

- **Loop diuretics (furosemide):** Block NKCC2 in TAL
- **Concentrated medulla:** Lipophilic drugs accumulate
- **Urea recycling:** Affects some drug transport

---

## 7. DISTAL TUBULE & COLLECTING DUCT

### 7.1 Distal Convoluted Tubule (DCT)

- **Length:** 5 mm
- **Reabsorbs:** 5-10% of filtered Na⁺ (NCC transporter)
- **Thiazide target:** NCC = thiazide-sensitive Na⁺-Cl⁻ cotransporter

### 7.2 Collecting Duct (CD)

**Two cell types:**

| Cell Type | Function | Drug Relevance |
|-----------|----------|----------------|
| **Principal cells** | Na⁺ reabsorption (ENaC), K⁺ secretion | Aldosterone target |
| **Intercalated cells** | H⁺/HCO₃⁻ balance | Urine pH regulation! |

### 7.3 Urine pH and Drug Trapping

**CRITICAL FOR DRUG REABSORPTION:**

The collecting duct can vary urine pH from **4.5 to 8.0** — a 3,500-fold change in H⁺ concentration!

**Impact on ionizable drugs:**

For a weak acid (pKa 5.0):
| Urine pH | % Ionized | Reabsorption |
|----------|-----------|--------------|
| 4.5 | 24% | High (mostly non-ionized) |
| 6.0 | 91% | Low |
| 7.5 | 99.7% | Very low (trapped) |

For a weak base (pKa 9.0):
| Urine pH | % Ionized | Reabsorption |
|----------|-----------|--------------|
| 4.5 | 99.997% | Very low (trapped) |
| 6.0 | 99.9% | Very low |
| 7.5 | 97% | Low |

**Clinical applications:**
- **Salicylate overdose:** Alkalinize urine → trap acid → ↑ excretion
- **Amphetamine overdose:** Acidify urine → trap base → ↑ excretion

---

## 8. CELLULAR ULTRASTRUCTURE

### 8.1 Proximal Tubule Cell - The Drug Processing Unit

```
┌────────────────────────────────────────────────────────────────┐
│                    PROXIMAL TUBULE CELL                        │
│                                                                │
│  APICAL (Luminal)           │        BASOLATERAL (Blood)      │
│  ════════════════           │        ══════════════════       │
│                             │                                  │
│  ▲▲▲▲▲▲▲▲▲▲▲               │                                  │
│  BRUSH BORDER               │                                  │
│  (microvilli)               │                                  │
│                             │                                  │
│  ┌─────────────┐            │        ┌─────────────┐          │
│  │ Endosomes/  │            │        │ Peritubular │          │
│  │ Lysosomes   │            │        │ Capillary   │          │
│  │ pH 4.8-5.0  │            │        │             │          │
│  └─────────────┘            │        └─────────────┘          │
│        ↑                    │              ↑                   │
│   Megalin/                  │              │                   │
│   Cubilin                   │              │                   │
│   endocytosis               │              │                   │
│                             │              │                   │
│  ╔═══════════════╗          │        ╔═══════════════╗        │
│  ║ MITOCHONDRIA  ║          │        ║ Na⁺/K⁺-ATPase║        │
│  ║ 30-40% vol    ║          │        ║ (drives all  ║        │
│  ║ ATP synthesis ║          │        ║ transport)   ║        │
│  ╚═══════════════╝          │        ╚═══════════════╝        │
│                             │                                  │
│  Tight junctions ──────────────── (seal between cells)        │
│                             │                                  │
└────────────────────────────────────────────────────────────────┘
```

### 8.2 Lipid Composition

**The kidney has unique lipid composition affecting drug binding:**

| Lipid Class | Kidney (%) | Muscle (%) | Significance |
|-------------|------------|------------|--------------|
| **Phosphatidylserine (PS)** | 0.50 | 0.15 | HIGHEST in body! Binds cations |
| Total phospholipids | 2.4 | 0.7 | High membrane content |
| Neutral lipids | 1.3 | 1.0 | Similar |
| Cholesterol | 0.8 | 0.5 | Membrane fluidity |

**Why does high PS matter?**

PS is anionic (net negative charge) → Electrostatic binding of cationic drugs:
- Aminoglycosides (polyamino sugars, +4 to +5 charge)
- Quaternary ammonium compounds
- Basic drugs (protonated amines)

This explains why **Kp_kidney >> Kp_muscle** for basic drugs!

### 8.3 Lysosomes in PT Cells

**Lysosomal characteristics:**

| Parameter | PT Cells | Hepatocytes | Significance |
|-----------|----------|-------------|--------------|
| Volume fraction | 1.8% | 2.5% | Intermediate |
| pH | 4.5-5.0 | 4.8 | Slightly more acidic |
| Acid phosphatase | High | High | Active degradation |
| Megalin-mediated uptake | YES | No | Unique to PT! |

**Aminoglycoside pathway:**
1. Binds to brush border PS (electrostatic)
2. Endocytosed via megalin receptor
3. Delivered to lysosomes
4. Protonated at pH 5 → TRAPPED
5. Accumulates → phospholipidosis
6. Lysosomal rupture → cathepsin release → apoptosis

---

## 9. MEMBRANE TRANSPORTERS

### 9.1 The Transporter Landscape

**Proximal tubule has the highest transporter density in the body:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROXIMAL TUBULE CELL                          │
│                                                                  │
│  BLOOD                                           URINE           │
│  (Basolateral)                                   (Apical)        │
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  ORGANIC ANION TRANSPORTERS:                                    │
│  ─────────────────────────────                                  │
│  OAT1 (SLC22A6) ──────→                    ←────── OAT4 (SLC22A11)
│  OAT3 (SLC22A8) ──────→    ORGANIC         ←────── URAT1 (SLC22A12)
│                            ANIONS          ←────── MRP2 (ABCC2)  │
│                            (acids)         ←────── MRP4 (ABCC4)  │
│                                            ←────── BCRP (ABCG2)  │
│                                            ←────── NPT1 (SLC17A1)│
│                                                                  │
│  ORGANIC CATION TRANSPORTERS:                                   │
│  ─────────────────────────────                                  │
│  OCT2 (SLC22A2) ──────→                    ←────── MATE1 (SLC47A1)
│                            ORGANIC         ←────── MATE2-K       │
│  OCTN1 (SLC22A4)          CATIONS         ←────── P-gp (ABCB1)  │
│  OCTN2 (SLC22A5)          (bases)         ←────── OCT3? (unclear)
│                                                                  │
│  PEPTIDE TRANSPORTERS:                                          │
│  ────────────────────                                           │
│                             ←────── PEPT1 (SLC15A1)             │
│                             ←────── PEPT2 (SLC15A2)             │
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 OAT1 (SLC22A6) - The Anion Gatekeeper

**Expression:** Basolateral membrane of PT S1-S2

**Mechanism:** Tertiary active transport
1. Na⁺/K⁺-ATPase creates Na⁺ gradient
2. Na⁺-dicarboxylate cotransporter brings α-ketoglutarate in
3. OAT1 exchanges intracellular α-KG for extracellular organic anion

**Substrates:**
- PAH (prototype, used to measure RPF)
- Antivirals: tenofovir, adefovir, cidofovir
- Antibiotics: penicillins, cephalosporins
- Methotrexate
- NSAIDs (also inhibitors!)
- ACE inhibitors
- Urate

**Kinetics:**
| Parameter | Value |
|-----------|-------|
| Km (PAH) | 20-70 μM |
| Vmax | High capacity |
| Driving force | α-KG gradient |

**Clinical relevance:**
- **Tenofovir nephrotoxicity:** OAT1 uptake → mitochondrial toxicity
- **Probenecid interaction:** Inhibits OAT1 → ↓ penicillin secretion → ↑ levels

### 9.3 OAT3 (SLC22A8) - Broader Specificity

**Expression:** Basolateral membrane of PT

**Substrates (overlapping but distinct from OAT1):**
- Statins (pravastatin, rosuvastatin)
- Diuretics (furosemide, bumetanide)
- H2 blockers (cimetidine, ranitidine)
- NSAIDs
- Bile acids

**Clinical relevance:**
- Statin-NSAID interaction (competition for OAT3)

### 9.4 OCT2 (SLC22A2) - THE Cation Uptake Transporter

**Expression:** Basolateral membrane of PT (kidney-specific!)

**Mechanism:** Facilitated diffusion (electrogenic, potential-driven)

**Substrates:**
| Drug | Km (μM) | Consequence |
|------|---------|-------------|
| Metformin | 200-500 | CLrenal >> GFR |
| Cisplatin | 5-15 | Nephrotoxicity! |
| Cimetidine | 50-100 | DDI perpetrator |
| Oxaliplatin | Low | Less nephrotoxic |
| Memantine | ~100 | Accumulation |

**Polymorphisms:**
- OCT2 c.808G>T (p.A270S): ↓ function → ↓ metformin renal CL
- OCT2 c.596C>T: ↑ cisplatin nephrotoxicity

### 9.5 MATE1/MATE2-K - The Cation Efflux Pumps

**Expression:** Apical membrane of PT

**Mechanism:** H⁺/organic cation antiport (uses lumen acidification)

**Key concept: OCT2 + MATE = Coordinated Secretion**

```
Blood → [OCT2 uptake] → Cell → [MATE1/2-K efflux] → Urine

Without MATE: Drug ACCUMULATES in cell (toxicity!)
With MATE: Drug FLOWS THROUGH cell (secretion)
```

**This explains:**
- Metformin CLrenal = 400-600 mL/min (4-5× GFR)
- Cisplatin (poor MATE substrate) → accumulates → nephrotoxic
- Oxaliplatin (better MATE) → less accumulation → less nephrotoxic

### 9.6 P-glycoprotein (ABCB1)

**Expression:** Apical membrane of PT

**Function:** ATP-dependent efflux into urine

**Substrates:**
- Digoxin
- Cyclosporine
- Tacrolimus
- HIV protease inhibitors
- Many chemotherapy agents

**Contribution to renal clearance:**
- Usually modest compared to filtration
- Significant for P-gp substrates with low fu

### 9.7 Megalin/Cubilin - The Protein Scavengers

**Expression:** Apical membrane, clathrin-coated pits

**Function:** Receptor-mediated endocytosis of filtered proteins

**Ligands:**
- Albumin (recovers any that filters)
- Vitamin D binding protein
- Retinol binding protein
- Light chains
- **AMINOGLYCOSIDES!**

**This is why aminoglycosides accumulate:**
1. Aminoglycosides are polycationic
2. Bind to anionic brush border
3. Recognized by megalin
4. Endocytosed into lysosomes
5. Cannot be degraded → accumulate
6. Lysosomal membrane permeabilization → toxicity

---

## 10. DRUG PARTITIONING MECHANISMS

### 10.1 Kp_kidney Determinants

**The kidney:plasma partition coefficient depends on:**

1. **Water distribution** (f_ew + f_iw)
2. **Lipid partitioning** (neutral lipids, phospholipids)
3. **Acidic phospholipid binding** (PS, highest in body!)
4. **Lysosomal trapping** (for bases, pKa > 7)
5. **Transporter-mediated uptake** (OAT, OCT2)
6. **Protein binding** (albumin in extracellular space)

### 10.2 Why Kp_kidney is HIGH for Basic Drugs

**Three synergistic mechanisms:**

1. **Electrostatic binding to PS:**
   - PS = 0.5% of kidney mass (3× muscle)
   - Protonated amines bind to anionic PS heads
   - No membrane partitioning required!
   - Strongest for polycations (aminoglycosides)

2. **Lysosomal trapping:**
   - Lysosomal pH 4.8 vs cytosol pH 7.0
   - For base with pKa 8: 
     - In cytosol: 90% ionized
     - In lysosome: 99.98% ionized
     - Accumulation ratio: 100-200×
   - f_lysosome = 1.8% but contribution is large

3. **Transporter uptake (OCT2):**
   - For cationic OCT2 substrates
   - Creates concentration gradient blood → cell
   - If MATE low: accumulation
   - If MATE high: flow-through (↑ clearance, not Kp)

### 10.3 Aminoglycoside - The Extreme Case

**Why does gentamicin have Kp_kidney = 10-30?**

| Mechanism | Contribution |
|-----------|--------------|
| Water distribution | 0.75 |
| PS binding | 5-10 (polycationic, 5 charges) |
| Megalin endocytosis | 3-5 |
| Lysosomal trapping | 2-5 |
| **Total Kp** | **10-20** |

**Note:** logP = -3.1 (extremely hydrophilic), yet Kp >> 1!

This violates the typical "lipophilic = high Kp" rule because:
- Electrostatic binding doesn't require membrane partitioning
- Receptor-mediated endocytosis bypasses permeability requirement

### 10.4 Acid Drug Partitioning

**Acids (furosemide, penicillins) have different pattern:**

1. **Albumin binding in EW:** Some albumin in renal interstitium
2. **OAT-mediated uptake:** Creates cellular concentration
3. **No PS binding:** Anion-anion repulsion
4. **No lysosomal trapping:** Acids don't protonate at pH 4.8

**Result:** Kp typically 1-3 for OAT substrates

---

## 11. RENAL DRUG CLEARANCE

### 11.1 The Complete Equation

**CLrenal = fu × GFR × (1 + Secretion/Filtration) × (1 - Reabsorption)**

Or equivalently:

**CLrenal = CLfiltration + CLsecretion - CLreabsorption**

### 11.2 Filtration Clearance

CLfiltration = fu × GFR

| Drug | fu | GFR | CLfilt (mL/min) |
|------|-----|-----|-----------------|
| Inulin | 1.00 | 120 | 120 (reference) |
| Metformin | 0.99 | 120 | 119 |
| Digoxin | 0.75 | 120 | 90 |
| Warfarin | 0.01 | 120 | 1.2 |

### 11.3 Secretion Clearance

**Can exceed filtration! PAH used to measure renal plasma flow because:**
- PAH secretion is so efficient that extraction ratio → 90%
- CLpah ≈ Renal plasma flow (600-700 mL/min)

**Secretion ratio = CLsecretion / CLfiltration:**

| Drug | Transporter | Secretion Ratio | CLrenal |
|------|-------------|-----------------|---------|
| PAH | OAT1 | 5-6 | 600 mL/min |
| Metformin | OCT2/MATE | 3-4 | 500 mL/min |
| Penicillin G | OAT1/3 | 2-3 | 400 mL/min |
| Furosemide | OAT3 | 1-2 | ~120 mL/min |

### 11.4 Reabsorption

**Passive, depends on:**
1. Lipophilicity (logP)
2. Ionization (pKa, urine pH)
3. Urine flow rate (contact time)
4. Urine concentration

**Henderson-Hasselbalch in the tubule:**

For weak acid: 
Reabsorption ∝ [non-ionized] / [total]
            = 1 / (1 + 10^(pH - pKa))

For weak base:
Reabsorption ∝ 1 / (1 + 10^(pKa - pH))

### 11.5 Net Renal Clearance Examples

| Drug | Dominant Process | CLrenal | Notes |
|------|-----------------|---------|-------|
| Inulin | Filtration only | 120 | GFR marker |
| Metformin | Secretion | 500 | OCT2+MATE |
| Gentamicin | Filtration | 80 | Accumulates in tissue |
| Penicillin | Secretion | 400 | OAT1/3 |
| Propranolol | Reabsorption | <5 | Lipophilic, hepatic CL |
| Lithium | Reabsorption | 20 | Follows Na⁺ |

---

## 12. SOTA DISCOVERIES (2020-2024)

### 12.1 Single-Cell RNA Sequencing of Kidney (2021-2023)

**Reference:** Lake et al., Nature 2023; Muto et al., Science 2023

**Key findings:**
- Identified 51+ distinct cell types in human kidney
- PT has 3-4 distinct sub-populations with different transporter profiles
- S1 PT cells: High OAT1, OCT2
- S3 PT cells: Lower OAT1, more injury-susceptible
- Intercalated cells: Multiple subtypes affect urine pH

**PBPK implications:**
- Current models use "average" PT transporter expression
- Reality: Significant heterogeneity along nephron
- S3 segment may be more vulnerable to toxicity

### 12.2 Kidney Organoids for Drug Testing (2022-2024)

**Reference:** Freedman lab, Nature Communications 2023

**Advances:**
- Human iPSC-derived kidney organoids now express functional transporters
- OAT1, OAT3, OCT2 activity demonstrated
- Aminoglycoside uptake and toxicity can be modeled
- Personalized kidney organoids possible (patient-specific iPSC)

**Future PBPK:**
- In vitro Kp measurements in human organoids
- Patient-specific transporter activity
- Toxicity prediction

### 12.3 MATE Polymorphisms and Drug Response (2023)

**Reference:** Yonezawa et al., CPT 2023

**MATE1 rs2289669 (c.922-158G>A):**
- GG genotype: Normal MATE1 function
- AA genotype: ↓ MATE1 → ↓ metformin secretion → ↑ accumulation

**MATE2-K rs12943590:**
- Affects metformin renal clearance by 15-30%

**Implication:** Pharmacogenomic adjustment of renal clearance

### 12.4 Organic Anion Transporter-Mediated Drug-Drug Interactions (2024)

**Reference:** FDA Guidance Update 2024

**New understanding:**
- OAT1/3 DDIs are clinically significant for:
  - Tenofovir alafenamide (reduced with OAT3 inhibitors)
  - Methotrexate (toxicity with NSAIDs)
  - Pemetrexed (nephrotoxicity risk)

**Regulatory implication:** OAT1/3 substrate testing now recommended for renally cleared drugs

### 12.5 Tubular Secretion Scaling in CKD (2023-2024)

**Reference:** Hsueh et al., J Clin Pharmacol 2023

**Key insight:** In CKD, transporter expression changes non-linearly:

| CKD Stage | GFR (mL/min) | OAT1/3 Activity | OCT2 Activity |
|-----------|--------------|-----------------|---------------|
| 1 | >90 | 100% | 100% |
| 2 | 60-89 | 90% | 95% |
| 3a | 45-59 | 70% | 80% |
| 3b | 30-44 | 50% | 60% |
| 4 | 15-29 | 30% | 40% |
| 5 | <15 | 10% | 20% |

**PBPK implication:** Don't just scale CLfiltration by GFR — scale secretion too!

### 12.6 Uremic Toxins and Protein Binding (2023)

**Reference:** Nolin et al., JASN 2023

**Uremic toxins displace drugs from albumin:**
- Indoxyl sulfate (IS)
- p-Cresyl sulfate (pCS)
- Hippuric acid

**Effect:**
- ↑ fu in uremia → ↑ CLfiltration (partially compensates for ↓ GFR)
- But also ↑ tissue distribution → ↑ Vd
- Net effect: Variable, drug-specific

### 12.7 MicroRNA Regulation of Transporters (2022-2024)

**Reference:** Multiple Nature Communications papers

**miR-21:**
- Upregulated in kidney injury
- Suppresses OAT1 expression
- May explain reduced secretion in AKI

**miR-125:**
- Regulates MATE1 expression
- Could affect metformin handling

---

## 13. PBPK MODEL PARAMETERS

### 13.1 Tissue Composition (Rodgers & Rowland 2006, Updated)

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Volume (L) | V_kid | 0.31 | Anatomical |
| Blood flow (L/min) | Q_kid | 1.2 | Physiological |
| Neutral lipids | f_nl | 0.013 | R&R 2006 |
| Phospholipids | f_pl | 0.024 | R&R 2006 |
| Acidic phospholipids | f_apl | 0.005 | R&R 2006 |
| Extracellular water | f_ew | 0.273 | R&R 2006 |
| Intracellular water | f_iw | 0.483 | R&R 2006 |
| Lysosomal fraction | f_lyso | 0.018 | Schmitt 2021 |
| Lysosomal pH | pH_lyso | 4.8 | Literature |
| Cytosolic pH | pH_iw | 7.0 | Standard |

### 13.2 Transporter Abundances (Quantitative Proteomics)

| Transporter | pmol/mg protein | Km (μM) | Vmax relative |
|-------------|-----------------|---------|---------------|
| OAT1 | 4.0 ± 1.5 | 20-70 | High |
| OAT3 | 2.5 ± 1.0 | 10-50 | Moderate |
| OCT2 | 6.0 ± 2.0 | 200-500 | High |
| MATE1 | 2.5 ± 1.0 | 50-200 | Moderate |
| MATE2-K | 1.5 ± 0.5 | 50-150 | Moderate |
| P-gp | 0.8 ± 0.3 | Variable | Low |
| MRP2 | 1.5 ± 0.5 | Variable | Moderate |
| MRP4 | 2.0 ± 0.8 | Variable | Moderate |

### 13.3 Scaling Factors

**In vitro-in vivo extrapolation (IVIVE):**

| Scaling | Value | Notes |
|---------|-------|-------|
| Kidney weight | 310 g | Both kidneys |
| Proximal tubule fraction | 60% | Of kidney mass |
| S1-S2 fraction (high transport) | 70% | Of PT |
| Protein content | 40 mg/g kidney | Microsomal |
| Relative expression (cortex:medulla) | 10:1 | Transporters |

### 13.4 Age-Related Changes

| Age | GFR (% of young adult) | Transporter Activity | Kidney Volume |
|-----|------------------------|----------------------|---------------|
| 20-30 | 100% | 100% | 100% |
| 40-50 | 90% | 95% | 98% |
| 50-60 | 80% | 85% | 95% |
| 60-70 | 70% | 75% | 90% |
| 70-80 | 60% | 60% | 85% |
| >80 | 50% | 50% | 80% |

---

## 14. CLINICAL IMPLICATIONS

### 14.1 Drug Dosing in CKD

**When to adjust dose:**
1. fe (fraction excreted unchanged) > 0.3
2. Active metabolites renally cleared
3. Narrow therapeutic index
4. Transporter-mediated secretion

**How to adjust:**

For **filtration-dominant** drugs:
Dose_CKD = Dose_normal × (GFR_patient / 120)

For **secretion-dominant** drugs:
Need to scale both filtration AND secretion

### 14.2 Nephrotoxicity Prediction

**Risk factors for proximal tubule toxicity:**
1. High Kp_kidney (>5)
2. OCT2 substrate without MATE efflux
3. OAT1 substrate (accumulates in PT)
4. Aminoglycoside structure (megalin uptake)
5. Mitochondrial toxin (depletes ATP)

### 14.3 Drug-Drug Interactions

**High-risk combinations:**

| Perpetrator | Victim | Mechanism | Consequence |
|-------------|--------|-----------|-------------|
| NSAIDs | Methotrexate | OAT1/3 inhibition | MTX toxicity |
| Probenecid | Penicillin | OAT1/3 inhibition | ↑ penicillin levels (therapeutic!) |
| Cimetidine | Metformin | OCT2/MATE inhibition | ↑ metformin, lactic acidosis risk |
| Cyclosporine | Rosuvastatin | OAT3 inhibition | ↑ statin, myopathy |

### 14.4 Special Populations

**Pregnancy:**
- GFR increases 40-50% by 2nd trimester
- Increased renal clearance of hydrophilic drugs
- Dose adjustments may be needed

**Obesity:**
- GFR increases (hyperfiltration)
- But transporter expression may not scale
- Complex PK changes

**Neonates:**
- GFR very low at birth (25 mL/min/1.73m²)
- Transporters immature
- Very long elimination half-lives

---

## CONCLUSION

The kidney is a remarkably complex organ for drug handling. Our PBPK model must capture:

1. **Filtration:** fu × GFR (simple, well-understood)
2. **Secretion:** Transporter-mediated, can exceed filtration
3. **Reabsorption:** pH-dependent, lipophilicity-dependent
4. **Tissue partitioning:** High APL = high Kp for cations
5. **Lysosomal trapping:** Additional accumulation for bases
6. **Receptor-mediated uptake:** Aminoglycoside special case
7. **DDIs:** OAT, OCT2, MATE interactions
8. **Disease effects:** CKD reduces both filtration AND secretion
9. **Genetic variation:** Transporter polymorphisms

**Our competitive advantage:** Integrating all these mechanisms into a mechanistic, predictive model validated against clinical data.

---

*Last updated: 2024*  
*Darwin PBPK Platform - Where Physiology Meets Pharmacology*
