# Methods: DDI Prediction Model

## For Publication in Q1 Pharmacometrics Journals

---

## 2. Methods

### 2.1 DDI Prediction Framework

A mechanistic drug-drug interaction (DDI) prediction framework was developed within the Darwin PBPK platform, implementing FDA/EMA-recommended equations for reversible inhibition, mechanism-based inhibition (MBI), enzyme induction, and transporter-mediated interactions.

### 2.2 Reversible Inhibition

For competitive CYP inhibition, the change in victim drug exposure was calculated using the basic static model:

$$\text{AUC}_\text{ratio} = \frac{1}{f_m / (1 + [I]_u / K_i) + (1 - f_m)}$$

where $f_m$ is the fraction of victim drug clearance mediated by the inhibited enzyme, $[I]_u$ is the unbound inhibitor concentration at the enzyme site, and $K_i$ is the reversible inhibition constant.

For hepatic CYP enzymes, the inhibitor concentration was estimated as:

$$[I]_h = f_{u,p} \times (C_{\max} + \frac{F_a \cdot F_g \cdot k_a \cdot \text{Dose}}{Q_h})$$

where $f_{u,p}$ is the fraction unbound in plasma, $C_{\max}$ is the maximum systemic concentration, $F_a$ is fraction absorbed, $F_g$ is intestinal availability, $k_a$ is the absorption rate constant, and $Q_h$ is hepatic blood flow (90 L/h).

### 2.3 Mechanism-Based Inhibition

For time-dependent inhibitors that inactivate CYP enzymes, the steady-state inhibition ratio was calculated as:

$$R = 1 + \frac{k_{\text{inact}}}{k_{\text{deg}}} \times \frac{[I]}{K_I + [I]}$$

$$\text{AUC}_\text{ratio} = \frac{1}{f_m / R + (1 - f_m)}$$

where $k_{\text{inact}}$ is the maximum inactivation rate constant, $K_I$ is the inhibitor concentration producing half-maximal inactivation, and $k_{\text{deg}}$ is the enzyme degradation rate constant. For CYP3A4, $k_{\text{deg}}$ was set to 0.00048 min$^{-1}$ (half-life ~24h), consistent with literature estimates [1].

MBI predictions were calibrated against clinical data when available, using back-calculated effective $R$ values to account for uncertainties in in vitro to in vivo extrapolation.

### 2.4 Enzyme Induction

CYP induction effects were modeled using:

$$\text{Induction fold} = 1 + E_{\max} \times \frac{[I]_u}{EC_{50} + [I]_u}$$

$$\text{AUC}_\text{ratio} = \frac{1}{f_m \times \text{Induction fold} + (1 - f_m)}$$

where $E_{\max}$ is the maximum fold induction and $EC_{50}$ is the concentration producing half-maximal induction. For strong inducers (rifampin, carbamazepine, phenytoin), empirically calibrated values derived from clinical DDI studies were used preferentially over in vitro parameters due to known in vitro-in vivo disconnects [2].

### 2.5 Transporter-Mediated DDI

For drugs with significant hepatic uptake transporter involvement (e.g., OATP1B1 substrates), the transporter contribution was modeled as:

$$\text{AUC}_{\text{ratio,transporter}} = 1 + \frac{[I]_{\text{portal}}}{K_{i,\text{transporter}}}$$

where $[I]_{\text{portal}}$ is the inhibitor concentration in the portal vein, estimated as 5-10× systemic concentrations for orally administered drugs.

For dual-mechanism drugs (e.g., repaglinide with CYP2C8 and OATP1B1), total DDI was calculated as:

$$\text{AUC}_\text{ratio,total} = \text{AUC}_\text{ratio,CYP} \times \text{AUC}_\text{ratio,transporter}$$

### 2.6 Parameter Sources

| Parameter Type | Primary Source | N |
|---------------|----------------|---|
| Inhibitor $K_i$ | FDA DDI guidance, in vitro studies | 47 |
| Substrate $f_m$ | Clinical DDI studies | 37 |
| MBI $k_{\text{inact}}/K_I$ | Literature with clinical calibration | 25 |
| Induction $E_{\max}/EC_{50}$ | In vitro with empirical calibration | 23 |
| Transporter $K_i$ | FDA guidance | 32 |

### 2.7 External Validation

Model performance was assessed using 26 DDI pairs from independent clinical studies not used for model development or calibration. The validation set covered:
- 5 enzymes (CYP3A4, CYP2D6, CYP1A2, CYP2C8, CYP2C9)
- 3 mechanisms (reversible inhibition, MBI, induction)
- 1 transporter (OATP1B1)

### 2.8 Performance Metrics

Following Guest et al. [3], model performance was evaluated using:

1. **Percentage within X-fold:** Proportion of predictions within X-fold of observed values
2. **Average Fold Error (AFE):** Geometric mean of predicted/observed ratios (bias indicator)
   $$\text{AFE} = 10^{\frac{1}{n}\sum_{i=1}^{n}\log_{10}\left(\frac{\text{predicted}_i}{\text{observed}_i}\right)}$$
3. **Absolute Average Fold Error (AAFE):** Geometric mean of absolute fold errors (precision indicator)
   $$\text{AAFE} = 10^{\frac{1}{n}\sum_{i=1}^{n}\left|\log_{10}\left(\frac{\text{predicted}_i}{\text{observed}_i}\right)\right|}$$

Acceptance criteria per FDA/EMA guidance: ≥80% within 2-fold, AFE 0.5-2.0, AAFE <2.0.

### 2.9 Statistical Analysis

Pearson correlation coefficient (r) was calculated on log-transformed predicted vs. observed values. Root mean square error (RMSE) on log scale was used to assess overall prediction error.

---

## References

1. Yang J, et al. Cytochrome P450 turnover: regulation of synthesis and degradation, methods for determining rates, and implications for the prediction of drug interactions. Curr Drug Metab. 2008;9(5):384-94.

2. Sager JE, et al. Physiologically based pharmacokinetic (PBPK) modeling and simulation approaches: a systematic review of published models, applications, and model verification. Drug Metab Dispos. 2015;43(11):1823-37.

3. Guest EJ, et al. Critique of the two-fold measure of prediction success for ratios: application for the assessment of drug-drug interactions. Drug Metab Dispos. 2011;39(2):170-3.

4. FDA Guidance for Industry. In Vitro Drug Interaction Studies - Cytochrome P450 Enzyme- and Transporter-Mediated Drug Interactions. 2020.

5. EMA Guideline on the investigation of drug interactions. CPMP/EWP/560/95/Rev. 1 Corr. 2*. 2012.
