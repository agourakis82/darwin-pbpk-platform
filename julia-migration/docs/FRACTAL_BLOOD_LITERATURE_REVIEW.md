# 🔬 Fractal Blood Dynamics - Literature Review

**Date:** 2025-11-30  
**Purpose:** Scientific foundation for multi-phase tubular reactor / fractal dynamics blood model

---

## 📚 KEY PAPERS FOUND

### **1. Network-driven anomalous transport (Nature Communications, 2021)**
- **URL:** https://www.nature.com/articles/s41467-021-27534-8
- **Key finding:** Blood microcirculation exhibits **anomalous transport** due to fractal network
- **Method:** Continuous Time Random Walk (CTRW) theory
- **Transit time:** Power-law distribution, NOT exponential
- **Clinical relevance:** Hypoxic regions, amyloid-β clearance (Alzheimer's)

### **2. Macheras - Fractal Approach to Drug Distribution (1996)**
- **Foundational paper** on fractal kinetics in pharmacokinetics
- **Key insight:** Rate "constants" are NOT constant - they are time-dependent
- **Formula:** k(t) = k₀ × t^(-h) where h = fractal exponent
- **Validation:** Calcium pharmacokinetics

### **3. Fractal Kinetic Implementation in Population PK (2023)**
- **URL:** https://www.mdpi.com/1999-4923/15/1/304
- **Method:** Fractal kinetics in NONMEM/Monolix
- **Application:** Population pharmacokinetics
- **Key finding:** Improved predictions for drugs with large distribution volume

### **4. Gastrointestinal Drug Absorption - Heterogeneity (2019)**
- **Key insight:** GI absorption follows fractal kinetics
- **Reason:** Heterogeneous mixing in intestinal lumen
- **Time-dependent:** Absorption rate decreases over time

---

## 🔬 KEY CONCEPTS

### **Continuous Time Random Walk (CTRW)**
- Particles (drugs) perform random walk through vascular network
- **Waiting time distribution:** ψ(t) ∝ t^(-1-β) (power-law)
- **Mean squared displacement:** ⟨x²⟩ ∝ t^β (anomalous diffusion)
- When β < 1: subdiffusion (drug trapped in slow regions)
- When β > 1: superdiffusion (drug channeled through fast paths)

### **Fractal Kinetics**
- Traditional: dC/dt = -k × C (constant k)
- Fractal: dC/dt = -k₀ × t^(-h) × C (time-dependent k)
- **Mittag-Leffler function:** Solution involves ML(t) not exp(t)
- **Power-law terminal phase:** C(t) ∝ t^(-α) at long times

### **Transit Time Distribution**
- Traditional PBPK: τ = V/Q (single value)
- Reality: P(τ) ∝ τ^(-α) (power-law distribution)
- **Heavy tail:** Long residence times in peripheral vessels
- **Clinical implication:** Deep tissue penetration delayed

---

## 📊 VALIDATION DATA SOURCES

### **Blood Transit Time:**
- Indicator dilution studies (dye/tracer)
- ICG (indocyanine green) clearance
- Contrast-enhanced imaging

### **RBC Partitioning:**
- BioIVT RBC partitioning assays
- Blood-to-plasma ratio (R) measurements
- In vivo vs in vitro comparison

### **Fractal Kinetics Validation:**
- Paclitaxel (power-law terminal phase)
- Calcium (Macheras 1996)
- Miberfradil (liver distribution)

---

## 🎯 IMPLEMENTATION APPROACH

### **Multi-Phase Tubular Reactor Model:**

1. **Plasma Phase:**
   - Advection-dispersion in vessel network
   - Protein binding dynamics
   - Free drug transport

2. **RBC Phase:**
   - Slower velocity near walls (Fåhræus effect)
   - Partitioning from plasma
   - Different transit time distribution

3. **Protein-Bound Phase:**
   - Albumin binding (slow on/off)
   - AGP binding (fast on/off)
   - Competition for binding sites

### **Fractal Network:**
- Murray's law for vessel branching
- Power-law transit time distribution
- CTRW for particle transport

### **Computational Approach:**
- Statistical moments (efficient)
- Full network simulation (accurate)
- Hybrid PFR-CSTR (practical)

---

## 🚀 NEXT STEPS

1. **Implement CTRW framework** for blood transport
2. **Add multi-phase dynamics** (plasma, RBC, bound)
3. **Validate against clinical data** (transit times, PK curves)
4. **Compare to traditional PBPK** (improvement metrics)

---

**Last Updated:** 2025-11-30
