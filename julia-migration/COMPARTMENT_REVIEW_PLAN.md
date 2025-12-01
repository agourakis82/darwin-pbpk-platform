# 🔬 Darwin PBPK - Comprehensive Compartment Review Plan

**Status:** Starting detailed review of each compartment  
**Goal:** Validate physiological accuracy, identify gaps, improve clinical relevance  
**Timeline:** Systematic review of all 14 compartments + special compartments

---

## 📋 COMPARTMENT REVIEW CHECKLIST

### **1. BLOOD/PLASMA COMPARTMENT** ⏳
**Current Status:** Basic implementation (5L volume, protein binding)

**To Review:**
- [ ] Plasma volume (5L for 70kg adult) - validate for different body weights
- [ ] Protein binding (albumin, α1-AGP, lipoproteins)
- [ ] pH effects (7.35-7.45) on ionizable drugs
- [ ] Temperature effects (37°C baseline)
- [ ] Red blood cell partitioning (for some drugs)
- [ ] Hematocrit effects (normal 40-50%)

**Clinical Relevance:**
- Pregnancy (plasma volume ↑ 50%)
- Anemia (↓ RBC, ↑ free drug)
- Sepsis (↓ albumin, ↑ free drug)
- Liver disease (↓ albumin synthesis)

---

### **2. LIVER COMPARTMENT** ⏳
**Current Status:** Advanced (CYP metabolism, OATP/OCT1 transporters, disease states)

**To Review:**
- [ ] Hepatic blood flow (90 L/h = 1.5 L/min)
- [ ] CYP enzyme expression (CYP3A4, 2D6, 2C9, etc.)
- [ ] Transporter expression (OATP1B1/1B3, OCT1, MDR1)
- [ ] Intrinsic clearance (CLint) calculation
- [ ] Hepatic extraction ratio (ER = CLint/(Q + CLint))
- [ ] Disease effects (cirrhosis, hepatitis, fatty liver)

**Clinical Relevance:**
- Cirrhosis (↓ flow, ↓ metabolism)
- Hepatitis (↓ CYP activity)
- NAFLD (↑ prevalence, effects unknown)
- Drug-drug interactions (CYP inhibition)

---

### **3. KIDNEY COMPARTMENT** ⏳
**Current Status:** Advanced (CKD stages, transporters, lysosomal trapping)

**To Review:**
- [ ] Renal blood flow (60 L/h = 1 L/min)
- [ ] GFR (120 mL/min for healthy adult)
- [ ] Tubular secretion (OAT1/3, OCT2, MATE)
- [ ] Tubular reabsorption (aquaporins, etc.)
- [ ] CKD scaling (eGFR-based)
- [ ] Dialysis effects (for ESRD patients)

**Clinical Relevance:**
- CKD stages (G1-G5)
- Acute kidney injury (AKI)
- Dialysis (removes water-soluble drugs)
- Age effects (↓ GFR with age)

---

### **4. BRAIN COMPARTMENT** ⏳
**Current Status:** Advanced (BBB, P-gp, regional distribution, disease states)

**To Review:**
- [ ] Blood-brain barrier (BBB) permeability
- [ ] P-glycoprotein (MDR1) efflux
- [ ] BCRP efflux transporter
- [ ] Regional distribution (grey/white matter)
- [ ] CSF dynamics (production, clearance)
- [ ] Disease effects (meningitis, inflammation, stroke)

**Clinical Relevance:**
- CNS infections (meningitis, encephalitis)
- Neuroinflammation (↑ BBB permeability)
- Stroke (BBB disruption)
- Neurodegenerative diseases

---

### **5. ADIPOSE COMPARTMENT** ⏳
**Current Status:** Advanced (lipid partitioning, perfusion-limited)

**To Review:**
- [ ] Adipose volume (highly variable: 10-40kg)
- [ ] Neutral lipid content (85%)
- [ ] Blood flow (0.03 × volume)
- [ ] Vegetable oil:water partition (not octanol:water!)
- [ ] Obesity effects (↑ volume, ↓ perfusion)
- [ ] Lipophilic drug accumulation

**Clinical Relevance:**
- Obesity (↑ adipose volume)
- Lipophilic drugs (accumulate in adipose)
- Long-term toxicity (adipose as reservoir)
- Weight loss (drug release from adipose)

---

## 🎯 NEXT STEPS

1. **Start with Blood Compartment** - Foundation for all others
2. **Review each compartment systematically**
3. **Validate against literature values**
4. **Test with known drugs**
5. **Document improvements**

---

**Last Updated:** 2025-11-30
