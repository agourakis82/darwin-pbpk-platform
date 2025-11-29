# BRAIN COMPARTMENT - Complete Physiology Deep Dive

## Darwin PBPK Platform - SOTA BBB Modeling

*From Socratic Discussion: Textbook Physiology to Disruptive Discoveries*

---

## Executive Summary

This document captures the complete brain physiology deep dive, implementing **11 novel modeling features** never before combined in a single PBPK brain model:

1. **Circadian P-gp Variation** (2x daily variation)
2. **Neuroinflammation/Cytokine Effects** (IL-6: up to 84% P-gp reduction!)
3. **Novel Meningitis Staging System** (5 stages including TB fibrotic)
4. **Pediatric BBB Maturation** (Brazilian priority: meningitis burden)
5. **COVID-19 BBB Dysfunction** (Long COVID: persistent changes)
6. **Glymphatic System** (Sleep-dependent clearance)
7. **White/Grey Matter Distribution** (Explains 4-week antidepressant delay)
8. **Intranasal Delivery Prediction** (Bypasses BBB)
9. **Dynamic Kp,uu Integration** (All factors in one calculation)
10. **Lithium Special Case** (Na+ channel entry, toxicity)
11. **Dexamethasone Paradox** (Heals BBB, reduces drug penetration)

---

## Part 1: Evolutionary Perspective

### Why Does the BBB Exist?

The BBB evolved as **toxin defense** for hunter-gatherers:

- Plant alkaloids (nicotine, caffeine, morphine)
- Environmental toxins
- Bacterial products

**Key insight**: The BBB doesn't "know" about modern drugs - it treats them like potential toxins!

This explains:
- Why P-gp substrates are often lipophilic alkaloid-like structures
- Why the BBB is so selective (paracellular permeability 10^-8 cm/s vs 10^-5 elsewhere)
- Why CNS drug development has such high failure rates

---

## Part 2: The Blood-Brain Barrier - Detailed Anatomy

### Tight Junctions (Zona Occludens)

```
┌─────────────────────────────────────────────────────────────┐
│           BRAIN CAPILLARY ENDOTHELIUM                       │
│                                                             │
│  Cell 1                    │            Cell 2              │
│                            │                                │
│  ════════════════════════════════════════════════════════  │
│        Claudin-5 ─────────┼────────  Claudin-5             │
│        Occludin ──────────┼────────  Occludin              │
│        JAMs ──────────────┼────────  JAMs                  │
│        ZO-1/2/3 ──────────┼────────  ZO-1/2/3              │
│  ════════════════════════════════════════════════════════  │
│                            │                                │
│  Cytoplasm                 │  Cytoplasm                     │
└─────────────────────────────────────────────────────────────┘
        Paracellular route: BLOCKED (10^-8 cm/s)
```

### The Neurovascular Unit (NVU)

The BBB is NOT just endothelium - it's a **dynamic system**:

```
                    ┌──────────────────┐
                    │  BLOOD (Lumen)   │
                    └────────┬─────────┘
                             │
              ╔══════════════╧══════════════╗
              ║   ENDOTHELIAL CELLS         ║
              ║   - Tight junctions         ║
              ║   - P-gp, BCRP (luminal)    ║
              ║   - LAT1, GLUT1 (influx)    ║
              ╚══════════════╤══════════════╝
                             │
              ┌──────────────┴──────────────┐
              │   PERICYTES (30% coverage)  │
              │   - Regulate diameter        │
              │   - Control permeability     │
              │   - Degenerate in AD!        │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │   BASEMENT MEMBRANE         │
              │   - Collagen IV, Laminin    │
              │   - Additional barrier       │
              └──────────────┬──────────────┘
                             │
        ╔════════════════════╧════════════════════╗
        ║   ASTROCYTE END-FEET (99% coverage)     ║
        ║   - AQP4 for water                      ║
        ║   - Secrete factors maintaining BBB     ║
        ║   - CONTROL P-gp EXPRESSION!            ║
        ╚════════════════════╤════════════════════╝
                             │
        ┌────────────────────┴────────────────────┐
        │        MICROGLIA (immune sensors)        │
        │   - Neuroinflammation mediators          │
        │   - Cytokine release                     │
        └────────────────────┬────────────────────┘
                             │
                    ┌────────┴────────┐
                    │  BRAIN TISSUE   │
                    └─────────────────┘
```

---

## Part 3: P-glycoprotein - The Gatekeeper

### P-gp Expression at BBB

- **5-20x higher** than intestine
- **Luminal (blood-facing)** membrane - pumps OUT of brain
- Broad substrate selectivity: lipophilic cations, planar molecules

### Classic Example: Loperamide

```
Loperamide (Imodium):
├── Structure: Opioid agonist (binds mu receptors)
├── logP: 4.8 (highly lipophilic)
├── MW: 477
├── P-gp: STRONG substrate
│
├── Normal doses: NO CNS effect (antidiarrheal only)
│   └── P-gp pumps it out before reaching receptors
│
├── P-gp inhibition (e.g., quinidine): Respiratory depression!
│
└── Overdose: CNS effects when P-gp saturated
```

### P-gp is NOT Static!

**Key Discovery from our Discussion**: P-gp activity varies by:

1. **Time of day** (Circadian: 2x variation)
2. **Inflammation** (Cytokines reduce function up to 84%)
3. **Chronic drug exposure** (Induction over weeks)
4. **Disease states** (Epilepsy: upregulated; Alzheimer's: downregulated)

---

## Part 4: Circadian P-gp Variation

### The Pattern

```
P-gp Activity (% of max)
100% ┤                  ████
 90% ┤                 █    █
 80% ┤                █      █
 70% ┤               █        ██
 60% ┤              █           █
 50% ┤   ████      █             █
 40% ┤  █    ████ █               █
     └──────────────────────────────────────
      2AM  6AM  10AM  2PM  6PM  10PM  2AM
      
      NADIR         PEAK          NADIR
```

### Clinical Implication: Quetiapine Morning Sedation

From our discussion:

> "If P-gp is at its NADIR at night (2-4am), and the patient takes quetiapine at 10pm...
> Could the 'morning sedation' actually be due to ENHANCED brain penetration during
> sleep hours, creating a larger CNS depot that persists into morning?"

**Answer**: YES! The same dose given at different times has **different brain exposure**.

### Implementation

```julia
@enum CircadianPhase begin
    MORNING_PEAK      # 6-10 AM: P-gp HIGH, brain penetration LOW
    MIDDAY            # 10 AM-2 PM: P-gp declining
    AFTERNOON         # 2-6 PM: P-gp moderate
    EVENING           # 6-10 PM: P-gp declining further
    NIGHT_NADIR       # 10 PM-2 AM: P-gp LOW, brain penetration HIGH
    LATE_NIGHT        # 2-6 AM: P-gp at lowest, then rising
end

function calculate_circadian_pgp_activity(phase::CircadianPhase)::Float64
    activity = if phase == MORNING_PEAK
        1.0     # 100% - Peak P-gp activity
    elseif phase == NIGHT_NADIR
        0.50    # 50% - Nadir (2x less than morning!)
    # ...
    end
    return activity
end
```

---

## Part 5: Neuroinflammation Effects

### Cytokine Impact on P-gp (Ronaldson 2012)

| Cytokine | P-gp Function Reduction |
|----------|-------------------------|
| IL-6     | Up to **84%** reduction |
| TNF-α    | 34-55% reduction        |
| IL-1β    | 36-42% reduction        |

### The ATP Depletion Paradox

```
INFLAMMATION
     │
     ▼
┌─────────────────────────────┐
│  P-gp mRNA may INCREASE     │  ← Gene expression UP
│  (defensive response)        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  BUT P-gp FUNCTION DECREASES │  ← Actual transport DOWN
│  - ATP depletion             │
│  - Membrane reorganization   │
│  - Post-translational changes│
└─────────────────────────────┘
```

### Clinical Example: Sepsis

In sepsis (IL-6 elevated 20-100x):
- P-gp function reduced to ~20-40% of normal
- Brain drug exposure increased 2.5-5x
- **Clinical implication**: Watch for CNS toxicity with normal doses

### Implementation

```julia
function calculate_inflammation_pgp_effect(;
    il6_fold::Float64 = 1.0,
    tnf_fold::Float64 = 1.0,
    il1b_fold::Float64 = 1.0,
    immune_status::ImmuneStatus = IMMUNOCOMPETENT
)
    pgp_function = 1.0
    
    # IL-6 effect (most potent!)
    if il6_fold > 1.0
        il6_effect = 1.0 / (1.0 + 0.1 * (il6_fold - 1.0))
        pgp_function *= max(il6_effect, 0.16)  # Floor at 16%
    end
    # ...
    return clamp(pgp_function, 0.1, 1.0)
end
```

---

## Part 6: Meningitis - Brazilian Priority

### Novel 5-Stage System

From our discussion on Brazilian clinical priorities:
> "COVID/Long-COVID and Post-Transplant...And as I live in Brazil, here we have 
> a lot of Tuberculosis and Meningitis (children and Adult)"

```julia
@enum MeningitisStage begin
    NO_MENINGITIS     # Normal BBB
    STAGE_0_PRE       # Bacteremia, prodrome, BBB intact
    STAGE_I_EARLY     # CSF protein 50-150, cells 10-500, BBB opening
    STAGE_II_ESTABLISHED  # CSF protein 150-500, significantly disrupted
    STAGE_III_SEVERE  # CSF protein >500, severely disrupted
    STAGE_IV_FIBROTIC # TB meningitis: fibrotic - PARADOX!
end
```

### The TB Fibrotic Stage Paradox

```
TB Meningitis Timeline:
                                             
INFLAMMATION        FIBROSIS         
     │                  │            
     ▼                  ▼            
┌─────────┐        ┌─────────┐      
│ Stage   │        │ Stage   │      
│ II-III  │   →    │   IV    │      
│         │        │         │      
│ BBB     │        │ BBB     │      
│ OPEN    │        │ CLOSED  │      
│         │        │ by      │      
│ Drugs   │        │ fibrosis│      
│ GET IN  │        │         │      
│         │        │ Drugs   │      
│         │        │ BLOCKED │      
└─────────┘        └─────────┘      

Patient "improving" clinically
BUT drug penetration DECREASING!
```

### CSF Drug Penetration Database

| Drug         | Inflamed | Non-inflamed | Dexa-sensitive |
|--------------|----------|--------------|----------------|
| **Ceftriaxone** | 6% | 1% | No |
| **Vancomycin** | 48% | 18% | **YES** |
| **Linezolid** | 75% | 75% | No |
| **Rifampicin** | **15%** | 3% | YES |
| **Isoniazid** | 90% | 85% | No |
| **Fluconazole** | 80% | 75% | No |

### Rifampicin Problem

> "WARNING: Rifampicin CSF levels often subtherapeutic. 
> Consider high-dose (30-35 mg/kg)."

Only 10-20% CSF penetration despite being cornerstone TB drug!

### Dexamethasone Paradox

Dexamethasone in bacterial meningitis:
- REDUCES inflammation (good)
- HEALS BBB faster (good for brain)
- REDUCES drug penetration **29%** (bad for treatment!)

**Vancomycin + Dexamethasone**: Must increase dose to maintain CSF levels

---

## Part 7: Pediatric BBB Maturation

### BBB Development Timeline

| Age Group | BBB Maturity | P-gp Expression | Permeability Factor |
|-----------|--------------|-----------------|---------------------|
| Preterm Neonate | 55% | 35% | **2.5x** |
| Term Neonate | 65% | 45% | **2.0x** |
| Infant (1-12 mo) | 80% | 65% | 1.5x |
| Toddler (1-3 yr) | 90% | 85% | 1.2x |
| Child (3-12 yr) | 98% | 95% | 1.05x |
| Adult | 100% | 100% | 1.0x |
| Elderly >65 | 85% | 80% | **1.3x** |

### Combined Effect: Neonatal Meningitis

```
Neonatal BBB (55% mature)
        │
        │  + Meningitis inflammation
        │
        ▼
┌─────────────────────────────────┐
│  MASSIVELY INCREASED            │
│  DRUG PENETRATION               │
│                                 │
│  Factor: 2.5 × 7-15 = 17-37x!  │
│                                 │
│  Clinical: Dose CAREFULLY       │
│  - Higher CSF levels expected   │
│  - Risk of toxicity             │
└─────────────────────────────────┘
```

---

## Part 8: White Matter vs Grey Matter

### Composition Difference

```
WHITE MATTER                    GREY MATTER
─────────────                   ───────────
49% lipid (dry weight)          36% lipid
70% of that is MYELIN           
                                
DRUG RESERVOIR                  EFFECT SITE
Slow equilibration              Fast access
                                to receptors
```

### Why Antidepressants Take 4 Weeks

```
Day 1: Dose given
        │
        ▼
Grey matter (effect site): Rapid initial uptake
        │
        │ But...
        ▼
White matter (myelin): SLOWLY absorbing drug
        │
        │ Over weeks...
        ▼
Equilibrium reached: Grey matter at steady state

Week 1: Effect site concentration still fluctuating
Week 2: Building reservoir in white matter
Week 3: Approaching steady state
Week 4: TRUE steady state - therapeutic effect
```

### Implementation

```julia
function calculate_white_grey_matter_distribution(;
    logP::Float64,
    is_base::Bool = false,
    pKa::Union{Float64, Nothing} = nothing
)
    # Very lipophilic drugs: up to 168 hours equilibration half-life
    # Time to steady state: 5 × t½ = up to 4+ weeks
    
    equilibration_hours = if logP < 1.0
        2.0   # Fast
    elseif logP < 2.0
        8.0
    elseif logP < 3.0
        24.0  # ~1 day
    elseif logP < 4.0
        72.0  # ~3 days
    else
        168.0 # ~1 week
    end
    
    time_to_steady_state_days = equilibration_hours * 5 / 24
    # ...
end
```

---

## Part 9: Glymphatic System

### Discovery (2012-2013)

The brain has a "lymphatic-like" system for waste clearance!

```
AWAKE                           ASLEEP
─────                           ──────

Interstitial space: SMALL       Interstitial space: EXPANDED 60%!
CSF-ISF exchange: Limited       CSF-ISF exchange: MASSIVE
Drug clearance: Slow            Drug clearance: FAST
Aβ clearance: Poor              Aβ clearance: Good


Poor sleep → Drug accumulation → Side effects
           → Aβ accumulation → Alzheimer's risk
```

### Sleep Quality Impact

| Sleep Quality | Glymphatic Clearance | Drug Accumulation Risk |
|---------------|---------------------|------------------------|
| Good (>80%) | 100% | Normal |
| Moderate (60-80%) | 75% | Moderate |
| Poor (40-60%) | 50% | **High** |
| Very poor (<40%) | 30% | **Very High** |

### Clinical Implication

> "Consider sleep hygiene optimization BEFORE increasing CNS drug doses"

---

## Part 10: COVID-19 BBB Effects

### Long COVID: Persistent BBB Dysfunction

From Nature Neuroscience 2024:

> Patients report: "Medications affect me differently now"
> "I'm sensitive to things I wasn't before"

### Phase-Dependent Effects

| Phase | TJ Integrity | P-gp Function | Permeability | Notes |
|-------|--------------|---------------|--------------|-------|
| Acute | 30% | 40% | **5x** | Cytokine storm |
| Post-acute | 50-90% | 60-90% | 1.5-3x | Recovering |
| Long COVID | 60-75% | 65-80% | **1.5-2x** | **Persistent!** |
| Recovered | 90-100% | 92-100% | 1.0-1.1x | May have residual |

### Model Explains Patient Reports

Patients with Long COVID experiencing "altered drug sensitivity":
- BBB permeability factor 1.5-2x
- Drugs reaching brain at HIGHER concentrations
- Same dose → More effect (or side effects)

---

## Part 11: Intranasal Delivery

### BBB Bypass Mechanism

```
INTRANASAL ADMINISTRATION
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
DIRECT        SYSTEMIC
PATHWAY       PATHWAY
    │             │
    │             ▼
    │         Absorption
    │             │
    │             ▼
    │         Blood → BBB → Brain
    │                  │
    │                  │ P-gp efflux!
    │                  │
    ▼                  ▼
BRAIN (direct)    BRAIN (systemic)
5-15 minutes      30-60 minutes
NO P-gp barrier   P-gp barrier

~20% of dose      ~80% of dose
```

### Why Esketamine (Spravato) Works So Fast

- Oral ketamine: Hours to days for effect
- Intranasal esketamine: **2 hours** for antidepressant effect
- Bypasses P-gp via direct nose-to-brain pathway

### P-gp Substrate Advantage

For P-gp substrates, intranasal provides:
- 20% direct brain delivery (no P-gp)
- Effective brain bioavailability much higher than oral
- Time to effect: **Minutes vs hours**

---

## Part 12: Kp vs Kp,uu - The Critical Distinction

### Definitions

```
Kp,brain = Total brain concentration / Total plasma concentration
         = [Drug]brain / [Drug]plasma
         = Affected by tissue binding!

Kp,uu    = UNBOUND brain / UNBOUND plasma
         = [Drug]brain,unbound / [Drug]plasma,unbound  
         = PHARMACOLOGICALLY RELEVANT
         = Determines receptor occupancy!
```

### Why This Matters

| Drug | Kp,brain | Kp,uu | Interpretation |
|------|----------|-------|----------------|
| Diazepam | 0.9 | 0.8 | Passive diffusion, high tissue binding |
| Haloperidol | **15** | 3.0 | Massive brain accumulation |
| Risperidone | **10** | **0.3** | P-gp efflux despite high Kp! |
| Loperamide | 0.05 | 0.02 | Strong P-gp - no CNS effect |
| Caffeine | 0.8 | 1.0 | Free equilibrium |

### Risperidone Paradox

- High Kp (10): Looks like good brain penetrator
- Low Kp,uu (0.3): Actually has poor unbound brain levels
- P-gp substrate: Drug bound to lipids but unbound drug is effluxed

---

## Part 13: Lithium - Special Case

### Unique Properties

- Li+ is a simple ion (atomic radius 76 pm)
- Enters brain via **Na+ channels** (substitutes for Na+)
- NOT subject to typical lipophilicity rules

### Brain Penetration

| Ratio | Value | Clinical Use |
|-------|-------|--------------|
| Brain:Plasma | 0.5-0.8 | Slower equilibration than plasma |
| CSF:Plasma | 0.4-0.5 | Monitor for toxicity |

### Toxicity Cascade

```
Plasma Level (mEq/L)    Clinical Status
──────────────────────────────────────────
< 0.6                   Subtherapeutic
0.6 - 1.2              THERAPEUTIC
1.2 - 1.5              High (monitor closely)
1.5 - 2.0              TOXIC: Confusion, tremor
> 2.0                  CRITICAL: Seizures, coma, death
```

### Risk Factors

- Dehydration → Lithium concentrates
- NSAIDs → Reduce renal clearance (+30%)
- ACE inhibitors → Reduce renal clearance (+20%)
- Elderly → More BBB permeable

### Depot Formulation - The Holy Grail

From our discussion:

> "Of course, it would be wonderful to have an antidepressant with a depot.
> For example, if you have a lithium depot, you would solve big problems."

Why it doesn't exist:
- Narrow therapeutic index
- Variable absorption would be dangerous
- BUT if we had it with **reversing mechanism** or **plasma ceiling**...

---

## Part 14: pH-Partition and Ion Trapping

### The Mechanism

```
PLASMA (pH 7.4)              BRAIN ISF (pH 7.3)
                             LYSOSOMES (pH 4.8!)
                             
Weak base (pKa 8.5):         
                             
In plasma:                   In lysosomes:
- 92% ionized                - 99.95% ionized!
- 8% neutral                 - 0.05% neutral
                             
Neutral form crosses BBB → Becomes ionized → TRAPPED!

Result: ACCUMULATION in acidic compartments
```

### Clinical Example: Haloperidol

- pKa: 8.3 (weak base)
- Accumulates in brain due to:
  1. Lipid binding (logP 4.3)
  2. Lysosomal trapping
  3. Acidic phospholipid binding
- Result: Kp,uu = 3.0 (3x higher unbound in brain than plasma!)

---

## Part 15: The Dynamic Kp,uu Model

### Integration of All Factors

Our master function calculates Kp,uu considering ALL modulating factors:

```julia
function calculate_dynamic_kpuu(;
    # Drug properties
    logP, fup, MW, TPSA, HBD, pKa, is_base, is_acid, is_pgp_substrate,
    # Dynamic factors
    circadian_phase,
    immune_status,
    il6_fold, tnf_fold,
    meningitis_stage,
    age_group,
    sleep_quality,
    covid_phase,
    days_on_treatment,
    on_dexamethasone
)
    # 1. Baseline Kp,uu
    # 2. × Circadian factor
    # 3. × Inflammation factor  
    # 4. × Meningitis factor
    # 5. × Age factor
    # 6. × COVID factor
    # 7. × P-gp induction factor
    
    return Kpuu_dynamic
end
```

### Example Calculation

Patient: 6-month-old infant with bacterial meningitis Stage II, on dexamethasone

```
Baseline Kp,uu for drug: 0.3

Factors:
- Age (Infant): 1.5x permeability
- Meningitis Stage II: 7x permeability
- Dexamethasone: 0.71x (reduces penetration)

Combined Kp,uu ≈ 0.3 × 7 × 1.5 × 0.71 = 2.2

Interpretation: Brain exposure ~7x higher than healthy adult!
Recommendation: Consider dose REDUCTION
```

---

## Part 16: What We Can Model That Was NEVER Modeled Before

From our discussion:

> "what are we missing?? what we can model that was NEVER modeled before?"

### Novel Features in Darwin PBPK Brain Model

1. **Circadian P-gp dynamics** → Chronopharmacology optimization
2. **Cytokine-quantified BBB effects** → Sepsis dosing
3. **Meningitis staging with CSF protein correlation**
4. **TB fibrotic stage paradox** → Explains treatment failures
5. **Dexamethasone effect quantification** → 29% penetration reduction
6. **Pediatric BBB maturation curves** → Age-specific dosing
7. **Long COVID BBB dysfunction** → Explains altered drug sensitivity
8. **Glymphatic/sleep integration** → Sleep quality affects drug levels
9. **White/grey matter kinetics** → Explains delayed onset
10. **Intranasal BBB bypass modeling** → Route optimization
11. **Dynamic Kp,uu** → One function integrating ALL factors

---

## Part 17: Clinical Decision Support

### Meningitis Treatment Algorithm

```
┌─────────────────────────────────────────────────────────────┐
│              MENINGITIS DRUG SELECTION                       │
└─────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     BACTERIAL         TB             FUNGAL
          │                │                │
          │                │                │
          ▼                ▼                ▼
    Ceftriaxone      INH (90%)        Fluconazole
    + Vancomycin     PZA (95%)        (80% CSF)
                     RIF (15%!) ← PROBLEM!
                     ETB (35%)
          │                │                
          │                │                
          ▼                ▼                
    On dexa?          TB Stage IV?      
    YES → Increase    YES → PARADOX:    
    vancomycin dose   BBB closing!      
    (29% reduction)   ↑ dose needed     
```

### Chronotherapy Recommendations

| Drug Type | Optimal Timing | Rationale |
|-----------|----------------|-----------|
| P-gp substrate needing CNS effect | Evening (8-10 PM) | P-gp nadir → more brain |
| P-gp substrate avoiding CNS effects | Morning (6-10 AM) | P-gp peak → less brain |
| Non-P-gp CNS drug | Any time | Timing less critical |
| Sedating drug with morning hangover | Consider split dosing | Avoid night depot |

---

## References

### Primary Literature

1. Ronaldson PT et al. PLoS One 2012 - Cytokine effects on P-gp
2. Greene C et al. Nature Neuroscience 2024 - Long COVID BBB
3. Nedergaard M et al. Science 2013 - Glymphatic discovery
4. Pardridge WM. Drug transport across BBB (various)
5. Hammarlund-Udenaes M. Kp,uu concept for CNS drugs

### CSF Penetration Data Sources

- Nau R et al. - Antibiotic CSF penetration review
- Lutsar I et al. - Meropenem meningitis
- van de Beek D - Dexamethasone in meningitis
- Thwaites GE - TB meningitis treatment
- Perfect JR - Cryptococcal meningitis

### Brazilian Clinical Context

- Priority pathogens: Meningococcus, TB, Cryptococcus
- Population: High HIV prevalence in some regions
- Children: High meningitis burden
- Transplant: Growing population, immunosuppression

---

## Implementation Files

| File | Purpose |
|------|---------|
| `src/DarwinPBPK/compartments/brain.jl` | Main brain compartment model |
| `docs/deep_dive/CSF_PENETRATION_DATABASE.md` | Literature CSF data |
| `docs/deep_dive/BRAIN_PHYSIOLOGY_COMPLETE.md` | This document |

---

*Darwin PBPK Platform - "From textbook physiology to SOTA disruptive discoveries"*

*Created from Socratic Discussion - Brain Compartment Deep Dive*
