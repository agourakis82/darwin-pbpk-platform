# CSF PENETRATION DATABASE
## Quantitative Data for BBB/Meningitis Modeling

**Purpose:** Literature-derived CSF penetration data for model validation  
**Clinical Focus:** Brazilian disease burden (Meningitis, TB, HIV, COVID)  
**Last Updated:** 2024

---

## TABLE OF CONTENTS

1. [Antibiotics - Bacterial Meningitis](#1-antibiotics---bacterial-meningitis)
2. [Anti-TB Drugs](#2-anti-tb-drugs)
3. [Antifungals - Cryptococcal Meningitis](#3-antifungals---cryptococcal-meningitis)
4. [Antivirals](#4-antivirals)
5. [Effect of Inflammation on CSF Penetration](#5-effect-of-inflammation-on-csf-penetration)
6. [Effect of Dexamethasone](#6-effect-of-dexamethasone)
7. [Pediatric-Specific Data](#7-pediatric-specific-data)
8. [P-glycoprotein and Cytokine Effects](#8-p-glycoprotein-and-cytokine-effects)
9. [COVID-19 BBB Disruption](#9-covid-19-bbb-disruption)
10. [Age-Related BBB Changes](#10-age-related-bbb-changes)
11. [References](#11-references)

---

## 1. ANTIBIOTICS - BACTERIAL MENINGITIS

### 1.1 Beta-Lactams

| Drug | MW | logP | Protein Binding | CSF/Plasma (Inflamed) | CSF/Plasma (Non-inflamed) | Source |
|------|-----|------|-----------------|----------------------|---------------------------|--------|
| **Ceftriaxone** | 554 | -1.7 | 85-95% | 1.8-24.6% (mean ~6%) | <1% | Onita 2024 |
| **Cefotaxime** | 455 | -0.4 | 30-40% | 5-30% | 1-5% | Literature |
| **Ceftazidime** | 547 | -1.6 | 10-17% | 10-30% | 5-10% | Literature |
| **Cefepime** | 481 | -0.1 | 20% | 10-30% | 5-10% | Literature |
| **Meropenem** | 383 | -0.6 | 2% | 9% (up to 39% inflamed) | 2-5% | Blassmann 2016 |
| **Penicillin G** | 334 | 1.8 | 60% | 5-10% | 1-2% | Literature |
| **Ampicillin** | 349 | 1.4 | 20% | 10-20% | 2-5% | Literature |

**Key Finding:** Meropenem CSF/serum ratio = 9% baseline, but variable up to 39% with inflammation. High interindividual variability observed.

### 1.2 Vancomycin

| Condition | CSF/Serum Ratio | Peak CSF (mg/L) | Trough CSF (mg/L) | Source |
|-----------|-----------------|-----------------|-------------------|--------|
| **With meningitis** | 48% (range 3-81%) | 5.7-19.0 (mean 11.1) | Variable | Multiple studies |
| **Without meningitis** | 18% (range 0-36%) | 0.89-2.42 (mean 3.45) | Variable | Nau 2018 |
| **With dexamethasone** | 29% REDUCTION | - | - | Ricard 2007 |
| **High dose (60mg/kg)** | Overcomes dexa effect | Adequate | Adequate | Pediatric studies |

**Critical Note:** Dexamethasone reduces vancomycin CSF penetration by 29%, but higher doses (60 mg/kg/day in children) can overcome this effect.

### 1.3 Linezolid (Excellent CNS Penetration)

| Parameter | Value | Notes | Source |
|-----------|-------|-------|--------|
| CSF/Plasma ratio | 66-80% | Consistent across studies | Multiple |
| Peak CSF concentration | 9.8 ± 5.6 µg/mL | After 600mg IV | Beer 2007 |
| Trough CSF concentration | 5.8 ± 4.2 µg/mL | - | Beer 2007 |
| CSF half-life | 18.3 ± 19.2 hours | Longer than plasma (5.9h) | Beer 2007 |
| AUC CSF/fAUC plasma | ~80% | Even without inflammation | Frasca 2024 |

**Key Finding:** Linezolid demonstrates excellent CSF penetration (66-80%) independent of meningeal inflammation. Critical for XDR-TB and Gram-positive CNS infections.

### 1.4 Fluoroquinolones

| Drug | CSF/Plasma (Inflamed) | CSF/Plasma (Non-inflamed) | Notes |
|------|----------------------|---------------------------|-------|
| **Levofloxacin** | 70-90% | 50-70% | Good penetration |
| **Moxifloxacin** | 70-90% | 50-80% | Excellent for TB meningitis |
| **Ciprofloxacin** | 20-40% | 10-20% | Lower than newer FQs |

### 1.5 Fosfomycin

| Parameter | Value | Source |
|-----------|-------|--------|
| CSF penetration rate | 46% | Fille 2023 |
| Notes | Excellent even WITHOUT inflammation | Small MW, no protein binding |

---

## 2. ANTI-TB DRUGS

### 2.1 First-Line Drugs

| Drug | MW | logP | Protein Binding | CSF/Plasma (Inflamed) | CSF/Plasma (Non-inflamed) | Key Issue |
|------|-----|------|-----------------|----------------------|---------------------------|-----------|
| **Isoniazid** | 137 | -0.7 | 0-10% | 80-90% | 80-90% | Excellent, inflammation-independent |
| **Rifampicin** | 823 | 3.7 | 80-90% | 10-20% | <5% | POOR - P-gp substrate! |
| **Pyrazinamide** | 123 | -0.6 | 5-10% | 90-100% | 90% | Excellent, inflammation-independent |
| **Ethambutol** | 204 | -0.3 | 20-30% | 20-50% | 10-20% | Moderate, inflammation-dependent |
| **Streptomycin** | 581 | -5.0 | 35% | <10% | ~0% | Poor - aminoglycoside |

### 2.2 Rifampicin - The Critical Problem

**From LASER-TBM and Vietnamese Studies:**

| Study | Population | Dose | CSF Concentration | Therapeutic? |
|-------|------------|------|-------------------|--------------|
| Vietnamese children | Pediatric | 10 mg/kg | Below MIC in most | NO |
| Indonesian children | Pediatric | Standard | Very low, suboptimal AUC | NO |
| LASER-TBM adults | HIV+ adults | 10 mg/kg | Low | NO |
| LASER-TBM adults | HIV+ adults | 35 mg/kg | Higher, approaching target | MAYBE |

**MIC for M. tuberculosis:** 0.1-0.5 mg/L  
**Problem:** Standard dosing achieves CSF levels BARELY at MIC, often subtherapeutic!

**Recommendation:** High-dose rifampicin (30-35 mg/kg) trials ongoing for TB meningitis.

### 2.3 Isoniazid and Pyrazinamide (South African LASER-TBM Data)

| Parameter | Isoniazid | Pyrazinamide |
|-----------|-----------|--------------|
| Pseudo-partition coefficient (CSF/Plasma) | 1.04 | 1.05 |
| Time to equilibration (half-life) | 3.87 hours | 0.66 hours |
| Affected by high-dose rifampicin? | NO | NO |

**Key Finding:** Both drugs achieve CSF concentrations essentially equal to plasma. This supports their critical role in TB meningitis treatment.

---

## 3. ANTIFUNGALS - CRYPTOCOCCAL MENINGITIS

### 3.1 Drug Comparison

| Drug | CSF Penetration | Fungicidal? | Notes |
|------|-----------------|-------------|-------|
| **Fluconazole** | 70-90% (excellent) | NO (static) | Oral, good tolerability |
| **Flucytosine** | 75-90% | YES | Requires combination |
| **Amphotericin B** | <5% | YES | Poor penetration but standard of care |
| **Liposomal AmB** | Better than conventional | YES | Preferred formulation |
| **Itraconazole** | Lower than fluconazole | NO | Less used for CNS |
| **Voriconazole** | 40-70% | YES | Good alternative |

### 3.2 Fluconazole Dosing Considerations

| Dose | CSF Concentration | Efficacy |
|------|-------------------|----------|
| 400 mg/day | Fungistatic only | Inadequate as monotherapy |
| 800-1200 mg/day | Higher, better activity | Used in resource-limited settings |

**Key Insight:** Fluconazole CSF concentrations increase linearly with dose. Higher doses needed for efficacy.

### 3.3 Treatment Regimens (WHO 2022)

| Regimen | Duration | CSF Clearance |
|---------|----------|---------------|
| AmB + Flucytosine | 1 week induction | Fastest |
| AmB + Fluconazole | 2 weeks induction | Moderate |
| Fluconazole + Flucytosine | 2 weeks induction | Slowest |

**ART Timing:** Defer 4-6 weeks after induction due to IRIS risk (54% with early ART vs 0% with delayed).

---

## 4. ANTIVIRALS

### 4.1 Common Antivirals

| Drug | CSF Penetration | Notes |
|------|-----------------|-------|
| **Acyclovir** | 30-50% | Good for HSV encephalitis |
| **Ganciclovir** | 30-70% | CMV, variable |
| **Oseltamivir** | Low (P-gp substrate) | May increase with BBB disruption |
| **Remdesivir** | Low | Poor CNS penetration |

---

## 5. EFFECT OF INFLAMMATION ON CSF PENETRATION

### 5.1 Quantitative Changes by Inflammation State

| Drug Class | No Inflammation | Mild Inflammation | Severe Inflammation | Factor Increase |
|------------|-----------------|-------------------|---------------------|-----------------|
| Beta-lactams | 1-5% | 5-15% | 15-30% | 3-10x |
| Vancomycin | 0-18% | 20-40% | 40-80% | 2-5x |
| Aminoglycosides | ~0% | <5% | 5-10% | Large (from zero) |
| Fluoroquinolones | 50-70% | 70-90% | 70-90% | 1.3-1.5x |
| Linezolid | 66-80% | 66-80% | 66-80% | 1x (unchanged!) |

### 5.2 CSF Protein as Marker of BBB Disruption

| CSF Protein (mg/dL) | BBB State | Expected Penetration Multiplier |
|---------------------|-----------|--------------------------------|
| <45 (normal) | Intact | 1.0x (baseline) |
| 45-100 | Mild disruption | 1.5-2x |
| 100-300 | Moderate disruption | 2-5x |
| 300-500 | Severe disruption | 5-10x |
| >500 | Severely disrupted | 10-20x |

---

## 6. EFFECT OF DEXAMETHASONE

### 6.1 Vancomycin Reduction

| Parameter | Without Dexamethasone | With Dexamethasone | Change |
|-----------|----------------------|--------------------| ------|
| CSF/Serum ratio | 48% | 34% | -29% |
| Clinical impact | Adequate | May be subtherapeutic | Dose increase needed |

**Solution:** Increase vancomycin dose to 60 mg/kg/day (pediatric) when using dexamethasone.

### 6.2 Other Drugs

| Drug | Effect of Dexamethasone |
|------|------------------------|
| Ceftriaxone | Minimal effect |
| Rifampicin | May reduce further (already poor) |
| Linezolid | Minimal effect |
| Meropenem | May reduce |

---

## 7. PEDIATRIC-SPECIFIC DATA

### 7.1 Age-Dependent BBB Maturity

| Age Group | BBB Maturity | P-gp Expression | Clinical Implication |
|-----------|--------------|-----------------|---------------------|
| Preterm neonate | 50-60% | 30-40% | Very permeable, drug sensitivity |
| Term neonate | 60-70% | 40-50% | More permeable than adult |
| Infant (1-12mo) | 75-85% | 60-70% | Still more permeable |
| Toddler (1-3yr) | 85-95% | 80-90% | Approaching adult |
| Child (>3yr) | 95-100% | 90-100% | Adult-like |

### 7.2 Pediatric Drug Considerations

| Drug | Pediatric CSF Penetration | Adult CSF Penetration | Ratio |
|------|--------------------------|----------------------|-------|
| Vancomycin | Higher (less mature BBB) | Lower | ~1.3x |
| Ceftriaxone | 1.8-24.6% | 1.8-24.6% | Similar |
| Linezolid | 66-80% | 66-80% | Similar |

### 7.3 Vietnamese TB Meningitis Study (Children)

- Age-dependent variation in rifampicin, isoniazid, and pyrazinamide PK
- Rifampicin CSF concentrations below MIC in nearly ALL children
- Higher doses warranted for pediatric TB meningitis

---

## 8. P-GLYCOPROTEIN AND CYTOKINE EFFECTS

### 8.1 Cytokine Effects on P-gp Function (Quantitative)

**From Guinea Pig Brain Endothelial Cell Studies:**

| Cytokine | Developmental Stage | P-gp Function Reduction | Significance |
|----------|--------------------|-----------------------|--------------|
| **IL-1β** | GD50 (early) | No change | NS |
| **IL-1β** | GD65 (mid) | 42% reduction | P<0.01 |
| **IL-1β** | PND14 (postnatal) | 36% reduction | P<0.01 |
| **IL-6** | GD50 | No change | NS |
| **IL-6** | GD65 | 65% reduction | P<0.01 |
| **IL-6** | PND14 | 84% reduction | P<0.05 |
| **TNF-α** | GD50 | No change | NS |
| **TNF-α** | GD65 | 34% reduction | P<0.01 |
| **TNF-α** | PND14 | 55% reduction | P<0.01 |

**Key Finding:** Pro-inflammatory cytokines INHIBIT P-gp function in a development-dependent manner. Mature BBB (postnatal) is MORE affected by cytokines than immature BBB.

### 8.2 Human Brain Microvascular Endothelial Cells

| Cytokine | Effect on P-gp Activity | Effect on P-gp mRNA | Effect on Protein |
|----------|------------------------|---------------------|-------------------|
| TNF-α (24h) | Decreased | Increased | No change |
| IFN-γ (24h) | Decreased | Increased | No change |
| TNF-α + IFN-γ | Markedly decreased | Markedly increased | No change |

**Paradox:** P-gp mRNA INCREASES but FUNCTION DECREASES. This suggests post-translational dysfunction (possibly ATP depletion in inflammation).

### 8.3 Sepsis Cytokine Levels

| Cytokine | Normal Plasma | Sepsis | Severe/Meningococcal |
|----------|---------------|--------|----------------------|
| **IL-6** | 10-75 ng/L | 1-2 µg/L | Up to 200 µg/L |
| **TNF-α** | Undetectable | Elevated | Very high |

---

## 9. COVID-19 BBB DISRUPTION

### 9.1 Mechanisms

| Mechanism | Effect on BBB | Consequence |
|-----------|---------------|-------------|
| ACE2 receptor infection | Endotheliitis | Direct damage |
| MMP-9 upregulation | Basement membrane degradation | Leakage |
| RhoA activation | Cytoskeleton disruption | Tight junction opening |
| IL-6 surge | Complement activation | Coagulopathy, permeability↑ |
| Microthrombi | Vascular occlusion | Regional hypoxia |

### 9.2 Clinical Phases

| Phase | BBB State | Drug Penetration | Clinical |
|-------|-----------|------------------|----------|
| Acute (Days 1-14) | Disrupted | Increased | Delirium, encephalopathy |
| Post-acute (Weeks 2-12) | Variable | Variable | Recovering or persistent |
| Long COVID (Months 3-24+) | Chronic dysfunction? | Altered | "Brain fog", drug sensitivity |

### 9.3 Long COVID BBB Findings (Nature Neuroscience 2024)

- Dynamic contrast-enhanced MRI demonstrates BBB disruption
- Sustained systemic inflammation correlates with cognitive impairment
- Coagulation dysregulation observed in brain fog patients
- BBB integrity may be clinically useful biomarker

---

## 10. AGE-RELATED BBB CHANGES

### 10.1 Developmental Maturation

| Stage | BBB Status | Key Events |
|-------|------------|------------|
| E12 (embryonic) | Angioblasts invade | Vascular plexus forms |
| E15-E18 | Barrier forming | Tight junctions developing |
| Postnatal | Maturing | P-gp expression increasing |
| Adult | Mature | Full barrier function |

### 10.2 Aging and Elderly

| Finding | Method | Clinical Relevance |
|---------|--------|-------------------|
| Increased [11C]verapamil retention | PET | P-gp function declining |
| Hippocampal BBB mild deterioration | MRI | Predisposes to neurodegeneration |
| Increased IgG extravasation post-TBI | Mouse studies | Aged brain more vulnerable |
| Loss of tight junction integrity | Histology | Increased permeability |

**Clinical Implication:** Elderly patients may have increased CNS drug exposure at equivalent doses.

---

## 11. REFERENCES

### Meningitis and Antibiotics
1. Onita T, et al. (2024) Cerebrospinal Pharmacokinetic Analysis of Ceftriaxone in Pediatric Bacterial Meningitis. J Clin Pharm Ther.
2. Blassmann U, et al. (2016) CSF penetration of meropenem in neurocritical care patients. Crit Care.
3. Beer R, et al. (2007) Serum and CSF concentrations of linezolid in neurosurgical patients. Antimicrob Agents Chemother.
4. Nau R, et al. (2018) Penetration of drugs through the BBB. J Antimicrob Chemother.

### TB Meningitis
5. Ruslami R, et al. (2022) PK and safety of isoniazid, rifampicin, pyrazinamide in children with TBM. PLoS One.
6. Thwaites GE, et al. (2016) Vietnamese children TBM PK study. BMC Infect Dis.
7. Cresswell FV, et al. (2025) Population PK of pyrazinamide and isoniazid in TBM. J Infect Dis.
8. Svensson EM, et al. (2019) Attainment of target rifampicin in CSF. Antimicrob Agents Chemother.

### Cryptococcal Meningitis
9. Day JN, et al. (2013) Combination antifungal therapy for cryptococcal meningitis. NEJM.
10. Molloy SF, et al. (2018) Antifungal combinations for CM in Africa. NEJM.

### COVID-19 and BBB
11. Erickson MA, et al. (2023) Alteration of BBB by COVID-19. Front Cell Neurosci.
12. Greene C, et al. (2024) BBB disruption in long COVID. Nature Neuroscience.
13. Kempuraj D, et al. (2024) COVID-19 and Long COVID: Disruption of NVU and BBB. Neuroscientist.

### P-gp and Cytokines
14. Ronaldson PT, et al. (2012) Pro-inflammatory cytokine regulation of P-gp in developing BBB. PLoS One.
15. Eum SY, et al. (2013) P-gp activity changes by IFN-γ and TNF-α in human BMECs. Arch Pharm Res.

### BBB Development and Aging
16. Obermeier B, et al. (2013) Development, maintenance and disruption of the BBB. Nat Med.
17. Sweeney MD, et al. (2019) BBB: From physiology to disease and back. Physiol Rev.

---

## SUMMARY: KEY NUMBERS FOR MODELING

### CSF Penetration Quick Reference

| Drug | Baseline | Inflamed | Inflammation-Independent? |
|------|----------|----------|---------------------------|
| Isoniazid | 80-90% | 80-90% | YES |
| Pyrazinamide | 90% | 90-100% | YES |
| Linezolid | 66-80% | 66-80% | YES |
| Fluconazole | 70-90% | 70-90% | YES |
| Moxifloxacin | 50-80% | 70-90% | Partially |
| Rifampicin | <5% | 10-20% | NO (always poor) |
| Vancomycin | 0-18% | 20-80% | NO (highly dependent) |
| Meropenem | 2-5% | 9-39% | NO |
| Ceftriaxone | <1% | 2-25% | NO |

### P-gp Modulation Quick Reference

| Condition | P-gp Function | Expected Drug Penetration Change |
|-----------|---------------|----------------------------------|
| Normal | 100% | Baseline |
| IL-6 elevated | 16-35% of normal | 3-6x increase for P-gp substrates |
| TNF-α elevated | 45-66% of normal | 1.5-2x increase |
| Sepsis | Highly variable | Unpredictable, monitor! |
| Dexamethasone | P-gp function restored | Return toward baseline |

---

*Darwin PBPK Platform - CSF Penetration Database*  
*For Brazilian and Global Clinical Applications*
