# 🩸 Blood/Plasma Compartment - Detailed Analysis

**Status:** NEEDS IMPROVEMENT  
**Current Implementation:** Basic (5L volume, no protein binding dynamics)  
**Clinical Relevance:** CRITICAL (foundation for all other compartments)

---

## 📊 CURRENT IMPLEMENTATION

```julia
"blood" => 5.0,  # L (fixed for 70kg adult)
```

**Issues:**
1. ❌ No body weight scaling
2. ❌ No protein binding model
3. ❌ No pH effects on ionizable drugs
4. ❌ No hematocrit effects
5. ❌ No disease state adjustments

---

## 🔬 PHYSIOLOGICAL PARAMETERS

### **Plasma Volume (Normal Adult)**
- **70kg male:** 3.5 L (5% body weight)
- **70kg female:** 3.0 L (4.3% body weight)
- **Scaling:** V_plasma = 0.05 × BW (kg)

### **Protein Binding**
- **Albumin:** 35-50 g/L (60% of plasma proteins)
- **α1-Acid Glycoprotein:** 0.4-1.0 g/L
- **Lipoproteins:** Variable
- **Effect:** fu (fraction unbound) = 0.01-0.99

### **pH Effects**
- **Plasma pH:** 7.35-7.45 (normal)
- **Acidosis:** pH 7.0-7.35 (↑ ionization of weak acids)
- **Alkalosis:** pH 7.45-7.8 (↑ ionization of weak bases)

### **Hematocrit**
- **Normal:** 40-50% (males), 35-45% (females)
- **Anemia:** <35% (↑ free drug)
- **Polycythemia:** >55% (↓ free drug)

---

## 🏥 CLINICAL CONDITIONS AFFECTING BLOOD COMPARTMENT

### **Pregnancy**
- Plasma volume ↑ 50% (3.5L → 5.2L)
- Albumin ↓ 20%
- fu ↑ (↑ free drug)

### **Liver Disease**
- Albumin ↓ 30-50%
- α1-AGP ↑ (acute phase)
- fu ↑ (↑ free drug)

### **Sepsis**
- Albumin ↓ 40%
- α1-AGP ↑ 200-300%
- fu ↑ (↑ free drug)

### **Renal Disease**
- Albumin ↓ (proteinuria)
- Uremic toxins ↑ (compete for binding)
- fu ↑ (↑ free drug)

---

## 🎯 IMPROVEMENTS NEEDED

1. **Body Weight Scaling**
   - V_plasma = 0.05 × BW (males)
   - V_plasma = 0.043 × BW (females)

2. **Protein Binding Model**
   - fu = f(logP, pKa, pH, protein_conc)
   - Account for albumin + α1-AGP

3. **Disease State Adjustments**
   - Pregnancy: V ↑ 50%, albumin ↓ 20%
   - Liver disease: albumin ↓ 30-50%
   - Sepsis: albumin ↓ 40%, α1-AGP ↑ 300%

4. **Hematocrit Effects**
   - RBC partitioning for some drugs
   - Anemia: fu ↑ 20-30%

---

**Next:** Review Liver Compartment
