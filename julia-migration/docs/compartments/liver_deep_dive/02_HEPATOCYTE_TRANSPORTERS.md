# HEPATOCYTE MEMBRANE TRANSPORTERS: The Gatekeepers

## 1. Overview: The Hepatocyte as a Polarized Cell

Hepatocytes are **polarized epithelial cells** with distinct membrane domains:

```
              BLOOD (Sinusoid)
                    │
    ════════════════▼════════════════════════════════════
    │         SINUSOIDAL (BASOLATERAL) MEMBRANE         │
    │                                                    │
    │    ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐   │
    │    │OATP │  │OATP │  │ OCT │  │ OAT │  │NTCP │   │
    │    │ 1B1 │  │ 1B3 │  │  1  │  │  2  │  │     │   │
    │    └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘   │
    │       │        │        │        │        │       │
    │       ▼        ▼        ▼        ▼        ▼       │
    │    ═══════════════════════════════════════════    │
    │    │            HEPATOCYTE CYTOPLASM          │   │
    │    │                                          │   │
    │    │    Drug ──► Phase I (CYP450)            │   │
    │    │         └──► Phase II (Conjugation)     │   │
    │    │              └──► Metabolite            │   │
    │    │                                          │   │
    │    ═══════════════════════════════════════════    │
    │       │        │        │        │        │       │
    │       ▼        ▼        ▼        ▼        ▼       │
    │    ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐   │
    │    │BSEP │  │MRP2 │  │BCRP │  │MDR1 │  │MDR3 │   │
    │    │     │  │     │  │     │  │(Pgp)│  │     │   │
    │    └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘   │
    │       │        │        │        │        │       │
    │         CANALICULAR (APICAL) MEMBRANE             │
    ════════════════════════════════════════════════════
                    │
                    ▼
               BILE CANALICULUS
                    │
                    ▼
              BILE DUCTS ──► INTESTINE
```

---

## 2. Sinusoidal (Basolateral) Uptake Transporters

### 2.1 OATP Family (Organic Anion Transporting Polypeptides)

**The MAJOR uptake transporters for drugs into hepatocytes**

```
OATP1B1 (SLCO1B1) - "The Statin Transporter"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expression: Liver-specific (hepatocyte sinusoidal membrane)
Substrates: 
  • Statins (atorvastatin, rosuvastatin, pravastatin)
  • Methotrexate
  • Rifampin
  • Repaglinide
  • Bilirubin
  • Bile acids
  • Thyroid hormones

Inhibitors:
  • Cyclosporine (potent!)
  • Rifampin (also substrate - complex!)
  • Gemfibrozil glucuronide
  • Ritonavir

Clinical Relevance:
┌─────────────────────────────────────────────────────────────┐
│ OATP1B1 polymorphisms (SLCO1B1*5, *15, *17):               │
│                                                             │
│ • Reduced statin uptake → ↑ plasma levels                  │
│ • ↑ Risk of myopathy (especially simvastatin)              │
│ • FDA recommends genetic testing for simvastatin dosing    │
│                                                             │
│ c.521T>C (Val174Ala): ↓ function                           │
│   Wild-type: Normal statin metabolism                       │
│   Heterozygous: 1.5-2× higher statin AUC                   │
│   Homozygous: 3-4× higher statin AUC                       │
└─────────────────────────────────────────────────────────────┘
```

```
OATP1B3 (SLCO1B3) - "The Taxane Transporter"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expression: Liver-specific
Substrates:
  • Paclitaxel, Docetaxel
  • Digoxin
  • Telmisartan
  • Statins (overlap with 1B1)
  • CCK-8 (diagnostic probe)

Distinct from OATP1B1:
  • Prefers larger, more bulky substrates
  • Different polymorphism effects
```

```
OATP2B1 (SLCO2B1) - "The Fruit Juice Transporter"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expression: Liver, intestine, placenta
Substrates:
  • Fexofenadine
  • Statins (some)
  • Montelukast

Clinical Relevance:
  • Grapefruit, apple, orange juice INHIBIT OATP2B1
  • Reduces absorption of fexofenadine by 70%!
  • "Take with water, not fruit juice"
```

### 2.2 OCT Family (Organic Cation Transporters)

```
OCT1 (SLC22A1) - "The Metformin Transporter"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expression: Liver (sinusoidal), kidney, intestine
Substrates:
  • Metformin (PRIMARY hepatic uptake)
  • Oxaliplatin
  • Lamivudine
  • Morphine
  • Cimetidine
  • MPP+ (research probe)

Polymorphisms:
┌─────────────────────────────────────────────────────────────┐
│ OCT1 reduced function alleles (420del, R61C, G401S, etc.): │
│                                                             │
│ Population frequency: 9% Europeans are poor transporters   │
│                                                             │
│ Effect on Metformin:                                        │
│ • ↓ Hepatic uptake → ↓ glucose-lowering effect             │
│ • ↑ GI side effects (stays in gut longer)                  │
│ • May need higher dose or alternative therapy              │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 OAT Family (Organic Anion Transporters)

```
OAT2 (SLC22A7) - "The Hepatic Anion Transporter"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expression: Liver, kidney
Substrates:
  • cGMP (endogenous)
  • Nucleotide analogs (acyclovir, ganciclovir)
  • Salicylates
  • Diclofenac
  • Erythromycin

Note: OAT1 and OAT3 are primarily RENAL, not hepatic!
```

### 2.4 NTCP (Sodium Taurocholate Co-transporting Polypeptide)

```
NTCP (SLC10A1) - "The Bile Acid Recycler"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expression: Liver-specific (sinusoidal)
Function: Enterohepatic circulation of bile acids

Substrates:
  • Bile acids (primary function)
  • Rosuvastatin (minor)
  • Thyroid hormones

CRITICAL DISCOVERY (2012):
┌─────────────────────────────────────────────────────────────┐
│ NTCP is the ENTRY RECEPTOR for Hepatitis B and D viruses!  │
│                                                             │
│ • HBV binds to NTCP via preS1 domain                       │
│ • NTCP knockouts are resistant to HBV infection            │
│ • Myrcludex B (Bulevirtide): NTCP inhibitor for HBV/HDV    │
│                                                             │
│ Drug-Drug Interaction Risk:                                 │
│ • HBV antivirals may affect bile acid transport            │
│ • Bile acid sequestrants may affect HBV therapy            │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Canalicular (Apical) Efflux Transporters

### 3.1 P-glycoprotein (MDR1, ABCB1)

```
P-gp (ABCB1) - "The Master Efflux Pump"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expression: Liver (canalicular), intestine, kidney, BBB, placenta
Direction: Efflux INTO BILE

Substrates (extremely broad!):
┌─────────────────────────────────────────────────────────────┐
│ Cardiovascular: Digoxin, Verapamil, Diltiazem, Quinidine   │
│ Anticancer: Doxorubicin, Vinblastine, Paclitaxel           │
│ Immunosuppressants: Cyclosporine, Tacrolimus, Sirolimus    │
│ HIV: Ritonavir, Saquinavir, Indinavir                      │
│ Antibiotics: Erythromycin, Levofloxacin                    │
│ Others: Loperamide, Ondansetron, Fexofenadine              │
└─────────────────────────────────────────────────────────────┘

Inhibitors:
  • Verapamil, Quinidine, Cyclosporine
  • Ritonavir, Ketoconazole
  • Grapefruit juice (intestinal P-gp)

Inducers:
  • Rifampin (potent!)
  • St. John's Wort
  • Carbamazepine, Phenytoin

Mechanism:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│     CYTOPLASM                         BILE                  │
│         │                               │                   │
│    Drug─┤                               │                   │
│         │    ┌─────────────┐            │                   │
│         └───►│             │            │                   │
│              │    P-gp     │────────────┼───► Drug          │
│         ┌───►│   (ATP)     │            │                   │
│         │    └─────────────┘            │                   │
│    Drug─┤                               │                   │
│         │                               │                   │
│                                                             │
│   P-gp uses ATP to pump drugs AGAINST concentration        │
│   gradient from cytoplasm into bile                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 MRP2 (ABCC2)

```
MRP2 (ABCC2) - "The Conjugate Pump"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expression: Liver (canalicular), kidney, intestine
Function: Efflux of Phase II conjugates into bile

Substrates:
  • Glucuronide conjugates
  • Glutathione conjugates
  • Sulfate conjugates
  • Bilirubin glucuronide (critical!)
  • Methotrexate
  • Pravastatin

Dubin-Johnson Syndrome:
┌─────────────────────────────────────────────────────────────┐
│ MRP2 MUTATIONS cause Dubin-Johnson Syndrome:               │
│                                                             │
│ • Autosomal recessive                                       │
│ • Conjugated hyperbilirubinemia                            │
│ • Bilirubin glucuronide cannot be excreted                 │
│ • Benign condition (dark liver pigment)                    │
│                                                             │
│ Drug implications:                                          │
│ • Altered methotrexate elimination                         │
│ • May affect other glucuronide excretion                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 BCRP (ABCG2)

```
BCRP (ABCG2) - "The Breast Cancer Resistance Protein"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expression: Liver (canalicular), intestine, placenta, mammary
Function: Efflux of drugs and endogenous compounds into bile

Substrates:
  • Rosuvastatin (major!)
  • Sulfasalazine
  • Topotecan, Irinotecan
  • Nitrofurantoin
  • Uric acid
  • Porphyrins

Common Polymorphism:
┌─────────────────────────────────────────────────────────────┐
│ BCRP c.421C>A (Q141K):                                     │
│                                                             │
│ Frequency: 30% Asians, 10% Caucasians, 2% Africans         │
│                                                             │
│ Effect:                                                     │
│ • ↓ BCRP function → ↓ biliary efflux                       │
│ • ↑ Rosuvastatin AUC by 2-3 fold                          │
│ • ↑ Risk of statin-induced myopathy                        │
│                                                             │
│ Clinical: Consider lower rosuvastatin doses in Asians      │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 BSEP (ABCB11)

```
BSEP (ABCB11) - "The Bile Salt Export Pump"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expression: Liver-specific (canalicular)
Function: THE primary exporter of bile acids

Substrates:
  • Bile acids (primary function)
  • Pravastatin (minor)

CRITICAL FOR DRUG SAFETY:
┌─────────────────────────────────────────────────────────────┐
│ BSEP INHIBITION = DRUG-INDUCED LIVER INJURY (DILI)!        │
│                                                             │
│ Drugs that inhibit BSEP:                                    │
│ • Troglitazone (withdrawn - hepatotoxic)                   │
│ • Bosentan (boxed warning)                                 │
│ • Cyclosporine                                              │
│ • Ritonavir                                                 │
│ • Rifampin                                                  │
│                                                             │
│ Mechanism:                                                  │
│ • BSEP inhibition → bile acids accumulate in hepatocyte   │
│ • Bile acids are cytotoxic (detergent effect)              │
│ • Causes cholestatic liver injury                          │
│                                                             │
│ FDA now requires BSEP inhibition testing for new drugs!    │
└─────────────────────────────────────────────────────────────┘

Genetic BSEP Deficiency:
  • PFIC2 (Progressive Familial Intrahepatic Cholestasis)
  • Severe cholestasis, liver failure in infancy
  • Liver transplant often required
```

---

## 4. Sinusoidal Efflux Transporters (Back to Blood)

### 4.1 MRP3 (ABCC3) and MRP4 (ABCC4)

```
MRP3 and MRP4 - "The Safety Valves"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expression: Sinusoidal membrane (facing blood)
Direction: Efflux BACK INTO BLOOD (opposite to canalicular)

Purpose:
┌─────────────────────────────────────────────────────────────┐
│ When canalicular efflux is overwhelmed or inhibited,       │
│ MRP3/4 provide an ESCAPE ROUTE back to blood               │
│                                                             │
│                  Hepatocyte                                 │
│     ┌─────────────────────────────────────────┐            │
│     │                                         │            │
│ Blood │◄───MRP3/4───┤  Drug/Metabolite  ├───►│ Bile       │
│     │              │                    │MRP2│            │
│     │    ESCAPE    │                    │BCRP│            │
│     │     ROUTE    │                    │P-gp│            │
│     │              │                    │    │            │
│     └─────────────────────────────────────────┘            │
│                                                             │
│ This is why conjugated bilirubin appears in blood in       │
│ cholestasis - MRP3 exports it when MRP2 is blocked!        │
└─────────────────────────────────────────────────────────────┘

MRP3 Substrates:
  • Glucuronide conjugates
  • Bile acids
  • Methotrexate

MRP4 Substrates:
  • Nucleotide analogs (adefovir, tenofovir)
  • cAMP, cGMP
  • Bile acids
  • Prostaglandins
```

---

## 5. Integrated Transporter Map

```
                         BLOOD (Sinusoid)
                              │
    ══════════════════════════▼════════════════════════════════════
    │                    UPTAKE TRANSPORTERS                       │
    │                                                              │
    │   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐         │
    │   │OATP  │  │OATP  │  │ OCT1 │  │ OAT2 │  │ NTCP │         │
    │   │ 1B1  │  │ 1B3  │  │      │  │      │  │      │         │
    │   │Statins│ │Taxanes│ │Metfor│ │Anions│ │Bile  │         │
    │   └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘         │
    │      │         │         │         │         │              │
    │      └─────────┴────┬────┴─────────┴─────────┘              │
    │                     ▼                                        │
    │    ╔═══════════════════════════════════════════════╗        │
    │    ║              HEPATOCYTE CYTOPLASM             ║        │
    │    ║                                               ║        │
    │    ║   Drug ──► CYP450 ──► Phase I Metabolite     ║        │
    │    ║                           │                   ║        │
    │    ║                           ▼                   ║        │
    │    ║              UGT, SULT, GST, NAT             ║        │
    │    ║                           │                   ║        │
    │    ║                           ▼                   ║        │
    │    ║              Phase II Conjugate              ║        │
    │    ║                                               ║        │
    │    ╚═══════════════════════════════════════════════╝        │
    │                     │                                        │
    │      ┌──────────────┼──────────────┐                        │
    │      │              │              │                        │
    │      ▼              ▼              ▼                        │
    │   ┌──────┐      ┌──────┐      ┌──────┐                     │
    │   │ MRP3 │      │ MRP4 │      │      │ ──► To canaliculus  │
    │   │(back)│      │(back)│      │      │                     │
    │   └──┬───┘      └──┬───┘      └──────┘                     │
    │      │              │                                        │
    │      ▼              ▼                                        │
    │         BACK TO BLOOD                                        │
    │      (escape route)                                          │
    │                                                              │
    ════════════════════════════════════════════════════════════════
                              │
    ══════════════════════════▼════════════════════════════════════
    │                   EFFLUX TRANSPORTERS                        │
    │                                                              │
    │   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐         │
    │   │ BSEP │  │ MRP2 │  │ BCRP │  │ P-gp │  │ MDR3 │         │
    │   │      │  │      │  │      │  │(MDR1)│  │      │         │
    │   │Bile  │  │Conjug│  │Statins│ │Drugs │  │Phos- │         │
    │   │acids │  │-ates │  │Urate │  │      │  │lipid │         │
    │   └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘         │
    │      │         │         │         │         │              │
    │      └─────────┴────┬────┴─────────┴─────────┘              │
    │                     ▼                                        │
    ════════════════════════════════════════════════════════════════
                              │
                         BILE CANALICULUS
                              │
                              ▼
                         BILE DUCT
```

---

## 6. Transporter-Mediated Drug-Drug Interactions

### 6.1 Clinical Examples

```
EXAMPLE 1: Statin + Cyclosporine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cyclosporine inhibits:
  • OATP1B1 (↓ statin uptake into liver)
  • P-gp (↓ statin biliary efflux)
  • CYP3A4 (↓ statin metabolism for some)

Result:
  • Simvastatin AUC ↑ 8-fold
  • Atorvastatin AUC ↑ 15-fold!
  • Rosuvastatin AUC ↑ 7-fold

Clinical Action:
  • Use lowest statin doses
  • Prefer pravastatin or fluvastatin (less CYP3A4)
  • Monitor for myopathy
```

```
EXAMPLE 2: Metformin + Cimetidine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cimetidine inhibits:
  • OCT1 (↓ hepatic uptake)
  • OCT2 and MATE1/2 (↓ renal secretion)

Result:
  • Metformin AUC ↑ 50%
  • ↑ Plasma levels
  • ↓ Hepatic effect (paradoxically ↓ efficacy!)

Clinical Action:
  • Use alternative H2 blocker (ranitidine, famotidine)
  • Or use PPI instead
```

```
EXAMPLE 3: Digoxin + Quinidine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Quinidine inhibits:
  • P-gp (major effect)
  • Renal tubular secretion

Result:
  • Digoxin levels ↑ 2-3 fold
  • Risk of digoxin toxicity

Clinical Action:
  • Reduce digoxin dose by 50%
  • Monitor digoxin levels closely
```

---

## 7. Implications for Liver Kp Prediction

### 7.1 Transporter Effects on Kp

```
LIVER Kp IS NOT JUST ABOUT LIPOPHILICITY!

For highly transported drugs:

Kp_liver = f(passive_partition, 
             active_uptake, 
             active_efflux, 
             metabolism,
             binding)

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Kp_uu (unbound) = Uptake_CLint / (Passive + Efflux_CLint) │
│                                                             │
│  If Uptake >> Efflux: Kp_uu > 1 (concentrative uptake)     │
│  If Uptake << Efflux: Kp_uu < 1 (net efflux)               │
│  If Uptake ≈ Efflux: Kp_uu ≈ 1 (equilibrium)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Examples:
  • Statins: Kp_uu_liver >> 1 (OATP uptake, target is in liver)
  • Digoxin: Kp_uu_liver < 1 (P-gp efflux dominates)
  • Metformin: Kp_uu_liver > 1 (OCT1 uptake)
```

### 7.2 Transporter Features for ML Models

```python
# Recommended transporter-related features for Kp prediction

transporter_features = {
    # OATP1B1/1B3 (anionic drugs)
    'is_oatp_substrate': predict_oatp_substrate(smiles),
    'oatp_inhibitor_risk': predict_oatp_inhibition(smiles),
    
    # OCT1 (cationic drugs)
    'is_oct1_substrate': predict_oct1_substrate(smiles),
    'is_organic_cation': charge > 0 at pH 7.4,
    
    # P-gp
    'is_pgp_substrate': predict_pgp_substrate(smiles),
    'pgp_efflux_ratio': measured or predicted,
    
    # BCRP
    'is_bcrp_substrate': predict_bcrp_substrate(smiles),
    
    # Combined
    'net_hepatic_uptake': oatp_score + oct1_score - pgp_score - bcrp_score,
    'transporter_influence': high if any transporter substrate,
}
```

---

## 8. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│              HEPATOCYTE TRANSPORTERS KEY POINTS                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  UPTAKE (Sinusoidal → Cytoplasm):                              │
│  ─────────────────────────────────                              │
│  • OATP1B1/1B3: Statins, anions, bilirubin                     │
│  • OCT1: Metformin, cations                                     │
│  • NTCP: Bile acids (also HBV receptor!)                        │
│                                                                 │
│  EFFLUX (Cytoplasm → Bile):                                    │
│  ──────────────────────────                                     │
│  • P-gp: Broad specificity, lipophilic drugs                   │
│  • MRP2: Glucuronide conjugates                                 │
│  • BCRP: Rosuvastatin, sulfates                                 │
│  • BSEP: Bile acids (inhibition = DILI!)                       │
│                                                                 │
│  SAFETY VALVES (Cytoplasm → Blood):                            │
│  ──────────────────────────────────                             │
│  • MRP3/4: Escape route when canalicular blocked               │
│                                                                 │
│  CLINICAL IMPLICATIONS:                                         │
│  ──────────────────────                                         │
│  • Polymorphisms affect drug levels (SLCO1B1 for statins)      │
│  • DDIs at transporters (cyclosporine + statin)                │
│  • BSEP inhibition → cholestatic DILI                          │
│  • Kp_liver is NOT just passive partitioning                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**NEXT**: Part 3 - Hepatocyte Interior: CYP450 System and Phase I Metabolism
