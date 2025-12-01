# 🧬 Fractal Blood Dynamics - Theoretical Framework

**Model:** Multi-Phase Tubular Reactor with Fractal Network Dynamics  
**Paradigm Shift:** Blood as network of PFRs, NOT a well-stirred tank

---

## 📐 MATHEMATICAL FRAMEWORK

### **1. VASCULAR NETWORK TOPOLOGY**

#### **Murray's Law (Vessel Branching):**
```
r³_parent = Σ r³_children

Fractal dimension: D ≈ 2.7
Number of vessels: N(r) ∝ r^(-D)
```

#### **Vessel Parameters (Poiseuille Flow):**
```
Velocity:    v(r) = (ΔP × r²) / (8 × μ × L)
Flow rate:   Q(r) = π × r⁴ × ΔP / (8 × μ × L)
Resistance:  R(r) = 8 × μ × L / (π × r⁴)
```

### **2. TRANSIT TIME DISTRIBUTION**

#### **Traditional (Exponential):**
```
E(t) = (1/τ) × exp(-t/τ)

Mean: ⟨t⟩ = τ
Variance: σ² = τ²
```

#### **Fractal (Power-Law):**
```
E(t) = (α-1)/τ_min × (t/τ_min)^(-α) × H(t - τ_min)

Where:
- α = 1 + 1/D ≈ 1.37 (fractal exponent)
- τ_min = minimum transit time
- H(t) = Heaviside function
```

### **3. CONTINUOUS TIME RANDOM WALK (CTRW)**

#### **Waiting Time Distribution:**
```
ψ(t) = (β/τ) × (t/τ)^(-1-β) × H(t - τ_min)

Where β = anomalous diffusion exponent
- β < 1: subdiffusion (trapped)
- β = 1: normal diffusion
- β > 1: superdiffusion (channeled)
```

#### **Propagator (Green's Function):**
```
P(x,t) = ∫ ψ(t') × K(x,t') dt'

Where K(x,t) = advection-dispersion kernel
```

---

## 🔄 MULTI-PHASE TRANSPORT

### **Phase 1: Free Drug in Plasma**
```
∂C_free/∂t + v × ∂C_free/∂z = D × ∂²C_free/∂z² 
                              - k_bind × C_free × [Protein]
                              + k_off × C_bound
                              - k_RBC × (C_free - C_RBC/Kp)
```

### **Phase 2: Protein-Bound Drug**
```
∂C_bound/∂t + v × ∂C_bound/∂z = k_bind × C_free × [Protein] 
                                - k_off × C_bound

Equilibrium: fu = k_off / (k_off + k_bind × [Protein])
```

### **Phase 3: RBC-Partitioned Drug**
```
∂C_RBC/∂t + v_RBC × ∂C_RBC/∂z = k_RBC × (C_free - C_RBC/Kp_RBC)

Where:
- v_RBC = plasma velocity × (1 - 0.5 × Hct)  (Fåhræus effect)
- Kp_RBC = RBC:plasma partition coefficient
```

### **Total Concentration:**
```
C_total = C_free × (1-Hct) + C_bound × (1-Hct) + C_RBC × Hct

Blood:Plasma ratio: R = C_blood / C_plasma = 1 + Hct × (Kp_RBC - 1)
```

---

## 🌊 ADVECTION-DISPERSION-REACTION

### **General PDE (Single Vessel):**
```
∂C/∂t + v × ∂C/∂z = D × ∂²C/∂z² + R(C)

Where:
- v = velocity (Poiseuille profile)
- D = dispersion (Taylor dispersion)
- R(C) = reaction terms (binding, metabolism)
```

### **Taylor Dispersion Coefficient:**
```
D_eff = D_mol + (r² × v²) / (48 × D_mol)

For blood vessels: D_eff ≈ 10⁻⁸ to 10⁻⁶ m²/s
```

### **Boundary Conditions:**
- **Inlet:** C(0,t) = C_inlet(t)
- **Outlet:** ∂C/∂z(L,t) = 0 (no diffusion flux)
- **Branching:** Mass conservation at nodes

---

## 🔗 NETWORK COUPLING

### **Mass Balance at Branch Points:**
```
Q_parent × C_parent = Σ Q_child × C_child

Or equivalently:
C_mixed = Σ (Q_i × C_i) / Σ Q_i
```

### **Fractal Scaling of Network Properties:**
```
Total flow:      Q_total ∝ r^D
Total surface:   A_total ∝ r^(D-1)
Total volume:    V_total ∝ r^D
```

---

## 📊 OUTPUT: TRANSIT TIME DISTRIBUTION

### **Convolution Integral:**
```
C_outlet(t) = ∫₀ᵗ C_inlet(τ) × E(t-τ) dτ

Where E(t) = network transit time distribution (power-law)
```

### **Moments:**
```
Mean transit time:     ⟨τ⟩ = V/Q × correction_factor
Variance:              σ² = (V/Q)² × fractal_factor
Skewness:              γ > 0 (heavy right tail)
```

---

## 🎯 CLINICAL IMPLICATIONS

1. **First-pass dynamics:** Non-instantaneous mixing
2. **Recirculation:** Drug returns with delay distribution
3. **Deep compartment:** Power-law terminal phase
4. **Hypoxia prediction:** Critical regions from transit time
5. **Drug accumulation:** Different for lipophilic vs hydrophilic

---

**Next: Implementation in Julia**
