# 🩸 DISCUSSION 1: BLOOD/PLASMA COMPARTMENT

**Status:** Foundation compartment - CRITICAL for accuracy  
**Current Code:** `ode_solver.jl` lines 82-100

---

## 📊 PHYSIOLOGICAL PARAMETERS

### **Plasma Volume**
```
70kg adult male:    3.5 L (5% BW)
70kg adult female:  3.0 L (4.3% BW)
Scaling formula:    V_plasma = 0.05 × BW (males)
                    V_plasma = 0.043 × BW (females)
```

**Current Implementation:** Fixed 5.0 L ❌

**Issues:**
- No sex difference
- No body weight scaling
- Doesn't account for obesity/cachexia

### **Protein Binding**
```
Albumin:           35-50 g/L (60% plasma proteins)
α1-Acid Glycoprotein: 0.4-1.0 g/L (acute phase reactant)
Lipoproteins:      Variable
```

**Current Implementation:** None ❌

**Critical for:**
- Highly protein-bound drugs (fu < 0.1)
- Disease states (↓ albumin in liver disease)
- Drug-drug interactions (competition for binding)

### **pH Effects**
```
Normal plasma pH:   7.35-7.45
Acidosis:          pH < 7.35 (↑ ionization of weak acids)
Alkalosis:         pH > 7.45 (↑ ionization of weak bases)
```

**Current Implementation:** None ❌

**Critical for:**
- Weak acids (aspirin, warfarin)
- Weak bases (tricyclic antidepressants, quinidine)
- Sepsis/shock (acidosis)

---

## 🏥 CLINICAL CONDITIONS

### **Pregnancy**
- Plasma volume ↑ 50% (3.5L → 5.2L)
- Albumin ↓ 20%
- fu ↑ (↑ free drug)
- **Effect:** ↓ Cmax, ↑ Vd

### **Liver Disease**
- Albumin ↓ 30-50%
- α1-AGP ↑ (acute phase)
- fu ↑ (↑ free drug)
- **Effect:** ↑ free drug toxicity

### **Sepsis**
- Albumin ↓ 40%
- α1-AGP ↑ 200-300%
- Capillary leak (↑ interstitial fluid)
- **Effect:** Complex (depends on drug)

### **Renal Disease**
- Albumin ↓ (proteinuria)
- Uremic toxins ↑ (compete for binding)
- fu ↑ (↑ free drug)
- **Effect:** ↑ toxicity

### **Anemia**
- Hematocrit ↓ (< 35%)
- RBC partitioning ↓
- fu ↑ (↑ free drug)
- **Effect:** ↑ free drug

---

## 🎯 QUESTIONS FOR YOU

1. **Patient populations:** Do you need to model pediatric, elderly, pregnant, obese patients?
2. **Disease states:** Which are most important? (liver disease, sepsis, renal disease?)
3. **Drug types:** Highly protein-bound drugs? Weak acids/bases?
4. **Validation:** Do you have clinical data to validate against?

---

**Ready to discuss? What aspects are most important for your use case?**
