# ===========================================================================
# MEDLANG HEPATIC METABOLISM MODEL
# ===========================================================================
# Mechanistic model of hepatic drug metabolism with:
#
# ARCHITECTURE:
# - Fractal sinusoidal network (DLA-based, Df ≈ 1.7)
# - Zonation (periportal → pericentral O2 gradient)
# - Transporter-enzyme spatial distribution
#
# KINETICS:
# - Fractal Michaelis-Menten (anomalous diffusion in membranes)
# - Time-dependent inhibition (mechanism-based inactivation)
# - Enzyme induction (PXR/CAR/AhR nuclear receptors)
# - Competitive, non-competitive, uncompetitive inhibition
#
# TRANSPORTERS:
# - Uptake: OATP1B1, OATP1B3, OATP2B1, OCT1, NTCP
# - Efflux: P-gp, MRP2, BCRP, MATE1
#
# DISEASE STATES:
# - Cirrhosis (Child-Pugh A/B/C with fractal dimension collapse)
# - Fatty liver (NAFLD/NASH - zone-specific CYP changes)
# - Drug-induced liver injury (DILI)
#
# References:
# - PMC4778290 - DLA-based hepatic lobule models
# - PMC1571516 - Fractal analysis of hepatic sinusoids
# - PMC1304557 - Fractal Michaelis-Menten kinetics
# - Nature Digital Medicine 2024 - Virtual Hepatic Lobule
# - Frontiers Pharmacol 2024 - Cross-species CYP zonation
#
# Author: Dr. Demetrios Agourakis
# Date: November 2025
# ===========================================================================

module HepaticMetabolismModel

using ..MedLang
using SpecialFunctions: gamma

export HepaticParams, CYPEnzyme, HepaticTransporters
export FractalSinusoid, LiverZonation, Cirrhosis
export generate_hepatic_medlang, simulate_hepatic_clearance
export calculate_clh, calculate_extraction_ratio
export fractal_michaelis_menten, classical_michaelis_menten
export ddi_competitive, ddi_noncompetitive, ddi_mbi
export enzyme_induction_dynamics, calculate_net_ddi
export child_pugh_score, cirrhosis_state
export drug_hepatic_preset

# ===========================================================================
# FRACTAL ARCHITECTURE PARAMETERS
# ===========================================================================

"""
Hepatic sinusoidal fractal parameters.

The sinusoidal network follows diffusion-limited aggregation (DLA) pattern
with fractal dimension Df ≈ 1.7 in healthy liver.

References:
- PMC4778290 - DLA-based hepatic lobule models
- PMC1571516 - Fractal analysis in cirrhosis
"""
struct FractalSinusoid
    # Fractal dimensions
    Df::Float64                      # Mass fractal dimension (1.7 healthy, ↓ in cirrhosis)
    ds::Float64                      # Spectral dimension (4/3 for DLA = 1.33)

    # Derived parameters
    h::Float64                       # Fractal kinetic exponent = 1 - ds/2
    walk_dimension::Float64          # dw = 2*Df/ds (random walk on fractal)

    # Sinusoid geometry
    sinusoid_length_um::Float64      # Average sinusoid length
    sinusoid_diameter_um::Float64    # Average diameter

    # Flow heterogeneity
    transit_time_mean_s::Float64     # Mean transit time
    transit_time_cv::Float64         # Coefficient of variation (heterogeneity)
end

"""
Create fractal sinusoid parameters.

In healthy liver: Df ≈ 1.7, ds ≈ 1.33
In cirrhosis: Df ↓ (less complex, more linear sinusoids)
"""
function fractal_sinusoid(;
    Df::Float64 = 1.70,
    ds::Float64 = 1.33,
    sinusoid_length_um::Float64 = 250.0,
    sinusoid_diameter_um::Float64 = 8.0,
    transit_time_mean_s::Float64 = 8.0,
    transit_time_cv::Float64 = 0.5
)::FractalSinusoid

    # Derived fractal parameters
    h = 1.0 - ds / 2.0  # Fractal kinetic exponent
    dw = 2.0 * Df / ds  # Walk dimension

    return FractalSinusoid(
        Df, ds, h, dw,
        sinusoid_length_um, sinusoid_diameter_um,
        transit_time_mean_s, transit_time_cv
    )
end

export fractal_sinusoid

# ===========================================================================
# LIVER PHYSIOLOGY
# ===========================================================================

"""
Human liver physiological parameters.
"""
const LIVER_PHYSIOLOGY = (
    # Volumes
    V_liver_mL = 1500.0,             # Total liver volume
    V_hepatocyte_fraction = 0.78,    # Hepatocyte volume fraction
    V_sinusoid_fraction = 0.11,      # Sinusoidal volume fraction
    V_Disse_fraction = 0.05,         # Space of Disse
    V_bile_canaliculi = 0.02,        # Bile canalicular volume

    # Blood flow
    Q_hepatic_total_mL_min = 1450.0, # Total hepatic blood flow
    Q_portal_fraction = 0.75,        # Portal vein fraction
    Q_arterial_fraction = 0.25,      # Hepatic artery fraction

    # Oxygen gradient (drives zonation)
    pO2_portal_mmHg = 65.0,          # Periportal O2
    pO2_central_mmHg = 35.0,         # Pericentral O2

    # Protein binding
    albumin_plasma_g_L = 40.0,       # Normal plasma albumin
    AAG_plasma_g_L = 0.8,            # Alpha-1-acid glycoprotein

    # Lobule geometry
    lobule_diameter_mm = 1.0,        # Hexagonal lobule diameter
    lobules_per_liver = 1e6,         # Approximate number of lobules

    # Enzyme content
    CYP_total_pmol_mg = 300.0,       # Total CYP content per mg microsomal protein
    MPPGL_mg_g = 40.0,               # Microsomal protein per gram liver
)

# ===========================================================================
# ZONATION
# ===========================================================================

"""
Hepatic zonation parameters.

Zone 1 (Periportal): High O2, gluconeogenesis, OATP1B1
Zone 2 (Midzonal): Transitional
Zone 3 (Pericentral): Low O2, CYP3A4, lipogenesis, glycolysis
"""
struct LiverZonation
    # Enzyme distribution (fraction in each zone)
    CYP1A2_zone1::Float64
    CYP1A2_zone3::Float64

    CYP2C9_zone1::Float64
    CYP2C9_zone3::Float64

    CYP2C19_zone1::Float64
    CYP2C19_zone3::Float64

    CYP2D6_zone1::Float64
    CYP2D6_zone3::Float64

    CYP2E1_zone1::Float64
    CYP2E1_zone3::Float64

    CYP3A4_zone1::Float64
    CYP3A4_zone3::Float64

    # Transporter distribution
    OATP1B1_zone1::Float64
    OATP1B1_zone3::Float64

    OATP1B3_zone1::Float64
    OATP1B3_zone3::Float64

    # Metabolic zonation
    gluconeogenesis_zone1::Float64
    glycolysis_zone3::Float64
    lipogenesis_zone3::Float64
end

"""
Default zonation based on literature.

CYP3A4 is predominantly pericentral
OATP1B1 is predominantly periportal
"""
function default_zonation()::LiverZonation
    return LiverZonation(
        # CYP1A2: slightly pericentral
        0.35, 0.65,
        # CYP2C9: pericentral
        0.30, 0.70,
        # CYP2C19: pericentral
        0.30, 0.70,
        # CYP2D6: relatively uniform
        0.45, 0.55,
        # CYP2E1: strongly pericentral
        0.20, 0.80,
        # CYP3A4: strongly pericentral
        0.25, 0.75,
        # OATP1B1: periportal
        0.70, 0.30,
        # OATP1B3: less zonated
        0.55, 0.45,
        # Metabolic
        0.80, 0.75, 0.85
    )
end

export default_zonation

# ===========================================================================
# CYP ENZYMES
# ===========================================================================

"""
CYP enzyme parameters for drug metabolism.
"""
struct CYPEnzyme
    name::Symbol                     # :CYP3A4, :CYP2D6, etc.

    # Expression
    abundance_pmol_mg::Float64       # pmol/mg microsomal protein
    fraction_of_total::Float64       # Fraction of total CYP

    # Kinetics
    Km_uM::Float64                   # Michaelis constant
    Vmax_pmol_min_pmol_CYP::Float64  # Turnover number

    # Fractal correction
    use_fractal_kinetics::Bool       # Use fractal MM
    h_fractal::Float64               # Fractal exponent (0 = classical, 0.17 = DLA)

    # Induction/inhibition
    kdeg_h::Float64                  # Degradation rate constant (h⁻¹)
    half_life_h::Float64             # Protein half-life

    # Genetic polymorphism
    pm_frequency::Float64            # Poor metabolizer frequency
    um_frequency::Float64            # Ultra-rapid metabolizer frequency
end

"""
Default CYP enzyme parameters.

Based on Simcyp, GastroPlus, and literature values.
"""
function default_cyp_enzymes()::Dict{Symbol, CYPEnzyme}
    return Dict(
        :CYP3A4 => CYPEnzyme(
            :CYP3A4,
            137.0, 0.30,              # Abundance, fraction
            10.0, 10.0,               # Km, kcat
            true, 0.17,               # Fractal kinetics
            0.019, 36.0,              # kdeg, t½ (liver)
            0.0, 0.0                  # Polymorphism (mainly DDI, not genetic)
        ),
        :CYP2D6 => CYPEnzyme(
            :CYP2D6,
            10.0, 0.02,
            5.0, 8.0,
            true, 0.17,
            0.029, 24.0,
            0.07, 0.02                # 7% PM, 2% UM (Caucasian)
        ),
        :CYP2C9 => CYPEnzyme(
            :CYP2C9,
            60.0, 0.15,
            10.0, 6.0,
            true, 0.17,
            0.014, 50.0,
            0.03, 0.0                 # 3% PM (*3/*3)
        ),
        :CYP2C19 => CYPEnzyme(
            :CYP2C19,
            14.0, 0.04,
            15.0, 7.0,
            true, 0.17,
            0.019, 36.0,
            0.03, 0.05                # 3% PM (Caucasian), 5% UM
        ),
        :CYP1A2 => CYPEnzyme(
            :CYP1A2,
            45.0, 0.13,
            20.0, 5.0,
            true, 0.17,
            0.014, 50.0,
            0.0, 0.0                  # Inducible (smoking)
        ),
        :CYP2E1 => CYPEnzyme(
            :CYP2E1,
            50.0, 0.07,
            50.0, 12.0,
            true, 0.17,
            0.019, 36.0,
            0.0, 0.0                  # Induced in NASH
        ),
    )
end

export default_cyp_enzymes

# ===========================================================================
# HEPATIC TRANSPORTERS
# ===========================================================================

"""
Hepatic transporter parameters.
"""
struct HepaticTransporters
    # Basolateral uptake (sinusoidal membrane)
    oatp1b1_expression::Float64
    oatp1b1_km_uM::Float64
    oatp1b1_vmax_pmol_min_mg::Float64

    oatp1b3_expression::Float64
    oatp1b3_km_uM::Float64
    oatp1b3_vmax_pmol_min_mg::Float64

    oatp2b1_expression::Float64
    oatp2b1_km_uM::Float64

    oct1_expression::Float64
    oct1_km_uM::Float64

    ntcp_expression::Float64          # Bile acid transporter
    ntcp_km_uM::Float64

    # Canalicular efflux (apical membrane)
    pgp_expression::Float64
    pgp_km_uM::Float64

    mrp2_expression::Float64
    mrp2_km_uM::Float64

    bcrp_expression::Float64
    bcrp_km_uM::Float64

    # Basolateral efflux
    mrp3_expression::Float64
    mrp4_expression::Float64
end

"""
Default hepatic transporter parameters.
"""
function default_hepatic_transporters()::HepaticTransporters
    return HepaticTransporters(
        # OATP1B1 (major statin transporter)
        1.0, 5.0, 500.0,
        # OATP1B3
        0.8, 10.0, 300.0,
        # OATP2B1
        0.6, 20.0,
        # OCT1
        1.0, 100.0,
        # NTCP
        1.0, 15.0,
        # P-gp (canalicular)
        0.5, 10.0,
        # MRP2 (canalicular, organic anions)
        1.0, 50.0,
        # BCRP (canalicular)
        0.8, 5.0,
        # MRP3/4 (basolateral efflux)
        0.3, 0.5
    )
end

export default_hepatic_transporters

# ===========================================================================
# CIRRHOSIS / LIVER DISEASE
# ===========================================================================

"""
Child-Pugh score components and classification.
"""
struct ChildPughScore
    # Individual components (1-3 points each)
    bilirubin_points::Int             # <2, 2-3, >3 mg/dL
    albumin_points::Int               # >3.5, 2.8-3.5, <2.8 g/dL
    inr_points::Int                   # <1.7, 1.7-2.3, >2.3
    ascites_points::Int               # None, Mild, Moderate-Severe
    encephalopathy_points::Int        # None, Grade 1-2, Grade 3-4

    # Total and classification
    total_score::Int                  # 5-15
    class::Symbol                     # :A, :B, :C
end

"""
Calculate Child-Pugh score from clinical parameters.
"""
function child_pugh_score(;
    bilirubin_mg_dL::Float64 = 1.0,
    albumin_g_dL::Float64 = 4.0,
    inr::Float64 = 1.0,
    ascites::Symbol = :none,          # :none, :mild, :moderate_severe
    encephalopathy::Int = 0           # 0, 1-2, 3-4
)::ChildPughScore

    # Bilirubin points
    bili_pts = bilirubin_mg_dL < 2.0 ? 1 : (bilirubin_mg_dL <= 3.0 ? 2 : 3)

    # Albumin points
    alb_pts = albumin_g_dL > 3.5 ? 1 : (albumin_g_dL >= 2.8 ? 2 : 3)

    # INR points
    inr_pts = inr < 1.7 ? 1 : (inr <= 2.3 ? 2 : 3)

    # Ascites points
    asc_pts = ascites == :none ? 1 : (ascites == :mild ? 2 : 3)

    # Encephalopathy points
    enc_pts = encephalopathy == 0 ? 1 : (encephalopathy <= 2 ? 2 : 3)

    total = bili_pts + alb_pts + inr_pts + asc_pts + enc_pts

    # Classification
    class = total <= 6 ? :A : (total <= 9 ? :B : :C)

    return ChildPughScore(bili_pts, alb_pts, inr_pts, asc_pts, enc_pts, total, class)
end

"""
Cirrhosis disease state with fractal architecture changes.
"""
struct Cirrhosis
    # Classification
    child_pugh::ChildPughScore
    meld_score::Float64

    # Fractal architecture changes
    sinusoid_Df::Float64              # Fractal dimension (↓ in cirrhosis)

    # Physiological changes
    hepatic_blood_flow_fraction::Float64  # Reduced flow
    portal_shunt_fraction::Float64    # Porto-systemic shunting
    functional_liver_mass::Float64    # Remaining functional hepatocytes

    # Enzyme expression changes (fraction of normal)
    cyp3a4_expression::Float64
    cyp2d6_expression::Float64
    cyp2c9_expression::Float64
    cyp2c19_expression::Float64
    cyp1a2_expression::Float64
    cyp2e1_expression::Float64        # May increase in NASH

    # Transporter expression changes
    oatp1b1_expression::Float64
    oatp1b3_expression::Float64
    pgp_expression::Float64
    mrp2_expression::Float64

    # Protein binding changes
    albumin_fraction::Float64         # ↓ albumin → ↑fu for albumin-bound drugs
    aag_fraction::Float64             # Variable (acute phase reactant)
end

"""
Create cirrhosis state from Child-Pugh class.

Incorporates:
1. Fractal dimension collapse (sinusoid simplification)
2. Portal shunting
3. Enzyme/transporter expression changes
4. Protein binding changes
"""
function cirrhosis_state(child_pugh_class::Symbol)::Cirrhosis

    # Parameters by Child-Pugh class
    params = Dict(
        :A => (
            score = child_pugh_score(bilirubin_mg_dL=1.8, albumin_g_dL=3.8, inr=1.3),
            meld = 8.0,
            Df = 1.55,                # Mild fractal loss
            flow = 0.85,
            shunt = 0.10,
            mass = 0.80,
            cyp3a4 = 0.70, cyp2d6 = 0.80, cyp2c9 = 0.75, cyp2c19 = 0.75,
            cyp1a2 = 0.70, cyp2e1 = 1.0,
            oatp1b1 = 0.70, oatp1b3 = 0.75, pgp = 0.80, mrp2 = 0.70,
            albumin = 0.90, aag = 1.0
        ),
        :B => (
            score = child_pugh_score(bilirubin_mg_dL=2.5, albumin_g_dL=3.0, inr=1.8,
                                     ascites=:mild, encephalopathy=1),
            meld = 15.0,
            Df = 1.45,                # Moderate fractal loss
            flow = 0.65,
            shunt = 0.30,
            mass = 0.55,
            cyp3a4 = 0.45, cyp2d6 = 0.60, cyp2c9 = 0.50, cyp2c19 = 0.50,
            cyp1a2 = 0.45, cyp2e1 = 0.90,
            oatp1b1 = 0.45, oatp1b3 = 0.50, pgp = 0.60, mrp2 = 0.45,
            albumin = 0.70, aag = 1.1
        ),
        :C => (
            score = child_pugh_score(bilirubin_mg_dL=4.0, albumin_g_dL=2.5, inr=2.5,
                                     ascites=:moderate_severe, encephalopathy=3),
            meld = 25.0,
            Df = 1.35,                # Severe fractal loss
            flow = 0.45,
            shunt = 0.50,
            mass = 0.30,
            cyp3a4 = 0.25, cyp2d6 = 0.40, cyp2c9 = 0.30, cyp2c19 = 0.30,
            cyp1a2 = 0.25, cyp2e1 = 0.70,
            oatp1b1 = 0.25, oatp1b3 = 0.30, pgp = 0.40, mrp2 = 0.25,
            albumin = 0.55, aag = 1.2
        )
    )

    p = get(params, child_pugh_class, params[:A])

    return Cirrhosis(
        p.score, p.meld, p.Df,
        p.flow, p.shunt, p.mass,
        p.cyp3a4, p.cyp2d6, p.cyp2c9, p.cyp2c19, p.cyp1a2, p.cyp2e1,
        p.oatp1b1, p.oatp1b3, p.pgp, p.mrp2,
        p.albumin, p.aag
    )
end

export child_pugh_score, cirrhosis_state

# ===========================================================================
# FRACTAL MICHAELIS-MENTEN KINETICS
# ===========================================================================

"""
Classical Michaelis-Menten kinetics (well-stirred assumption).

v = Vmax × [S] / (Km + [S])
"""
function classical_michaelis_menten(
    S::Float64,      # Substrate concentration (µM)
    Vmax::Float64,   # Maximum velocity
    Km::Float64      # Michaelis constant (µM)
)::Float64
    return Vmax * S / (Km + S)
end

"""
Fractal Michaelis-Menten kinetics for heterogeneous media.

In spatially constrained environments (2D membranes, fractal sinusoids),
diffusion is anomalous and reaction kinetics deviate from classical MM.

v = Vmax × [S]^(1-h) / (Km' + [S]^(1-h))

Where:
- h = fractal kinetic exponent = 1 - ds/2
- ds = spectral dimension (~1.33 for DLA)
- Km' = modified Michaelis constant

For h → 0 (homogeneous): reduces to classical MM
For h = 0.17 (DLA): fractal kinetics

References:
- PMC1304557 - Fractal MM kinetics for mibefradil
"""
function fractal_michaelis_menten(
    S::Float64,      # Substrate concentration (µM)
    Vmax::Float64,   # Maximum velocity
    Km::Float64,     # Michaelis constant (µM)
    h::Float64       # Fractal exponent (0.17 for DLA, 0 for classical)
)::Float64

    if h ≈ 0.0
        return classical_michaelis_menten(S, Vmax, Km)
    end

    # Fractal modification
    S_eff = S^(1 - h)
    Km_eff = Km^(1 - h)

    return Vmax * S_eff / (Km_eff + S_eff)
end

"""
Time-dependent fractal kinetics.

In fractal media, rate "constants" become time-dependent:
k(t) = k₀ × t^(-h)

This captures the initial fast reaction followed by slowing
as reactants become trapped in fractal geometry.
"""
function fractal_rate_time_dependent(
    k0::Float64,     # Initial rate constant
    t::Float64,      # Time
    h::Float64       # Fractal exponent
)::Float64
    if t ≤ 0.0 || h ≤ 0.0
        return k0
    end
    return k0 * t^(-h)
end

export classical_michaelis_menten, fractal_michaelis_menten

# ===========================================================================
# DRUG-DRUG INTERACTIONS
# ===========================================================================

"""
Competitive inhibition (Ki affects apparent Km).

v = Vmax × [S] / (Km × (1 + [I]/Ki) + [S])
"""
function ddi_competitive(
    S::Float64,      # Substrate concentration
    I::Float64,      # Inhibitor concentration
    Vmax::Float64,
    Km::Float64,
    Ki::Float64      # Inhibition constant
)::Float64
    Km_app = Km * (1 + I / Ki)
    return Vmax * S / (Km_app + S)
end

"""
Non-competitive inhibition (affects Vmax).

v = Vmax / (1 + [I]/Ki) × [S] / (Km + [S])
"""
function ddi_noncompetitive(
    S::Float64,
    I::Float64,
    Vmax::Float64,
    Km::Float64,
    Ki::Float64
)::Float64
    Vmax_app = Vmax / (1 + I / Ki)
    return Vmax_app * S / (Km + S)
end

"""
Uncompetitive inhibition (binds ES complex).

v = Vmax × [S] / (Km + [S] × (1 + [I]/Ki))
"""
function ddi_uncompetitive(
    S::Float64,
    I::Float64,
    Vmax::Float64,
    Km::Float64,
    Ki::Float64
)::Float64
    return Vmax * S / (Km + S * (1 + I / Ki))
end

"""
Mechanism-based inactivation (time-dependent, irreversible).

MBI follows:
λ = kinact × [I] / (KI + [I])

Remaining enzyme:
E(t) = E₀ × exp(-λ × t)

AUC ratio:
AUCR = 1 / (1 - (fm × kinact × [I]/(KI + [I]) / (kdeg + kinact × [I]/(KI + [I]))))

References:
- ScienceDirect 2024 - Time-dependent CYP3A4 inhibition
"""
function ddi_mbi(
    I::Float64,              # Inactivator concentration
    kinact::Float64,         # Maximum inactivation rate (min⁻¹)
    KI::Float64,             # Concentration for half-maximal inactivation (µM)
    kdeg::Float64,           # Enzyme degradation rate (min⁻¹)
    fm::Float64 = 1.0        # Fraction metabolized by affected CYP
)::Float64

    # Inactivation rate
    lambda = kinact * I / (KI + I)

    # AUC ratio (steady-state approximation)
    auc_ratio = 1.0 / (1.0 - fm * lambda / (kdeg + lambda))

    return auc_ratio
end

export ddi_competitive, ddi_noncompetitive, ddi_uncompetitive, ddi_mbi

# ===========================================================================
# ENZYME INDUCTION
# ===========================================================================

"""
Enzyme induction dynamics.

Induction via PXR/CAR/AhR follows:
1. Inducer binds nuclear receptor
2. Receptor translocates to nucleus
3. mRNA transcription increases (hours)
4. Protein synthesis increases (days)
5. Reaches new steady-state (1-2 weeks)

E_ss = E₀ × (1 + Emax × [I] / (EC50 + [I]))

Time to new steady-state ≈ 4-5 × t½_protein

References:
- Springer 2023 - Induction onset/offset CYP3A4
"""
struct InductionParams
    inducer_name::String
    target_cyp::Symbol

    # Potency
    Emax::Float64                    # Maximum fold-induction
    EC50_uM::Float64                 # Concentration for half-max induction

    # Kinetics
    nuclear_receptor::Symbol         # :PXR, :CAR, :AhR
    mRNA_onset_h::Float64            # Time to mRNA increase
    protein_onset_days::Float64      # Time to protein increase
    full_induction_days::Float64     # Time to steady-state
    offset_days::Float64             # Time to return to baseline
end

"""
Calculate induced enzyme level at time t after starting inducer.
"""
function enzyme_induction_dynamics(
    t_days::Float64,         # Time since starting inducer
    I::Float64,              # Inducer concentration (µM)
    params::InductionParams,
    kdeg_h::Float64          # Enzyme degradation rate (h⁻¹)
)::Float64

    # Target fold-induction at steady state
    fold_ss = 1.0 + params.Emax * I / (params.EC50_uM + I)

    # Time constant for approach to steady state (based on protein turnover)
    tau = 1.0 / kdeg_h / 24.0  # days

    # Lag phase (mRNA/translation delay)
    if t_days < params.protein_onset_days
        return 1.0  # No change yet
    end

    # Exponential approach to new steady state
    t_effective = t_days - params.protein_onset_days
    fold_current = 1.0 + (fold_ss - 1.0) * (1.0 - exp(-t_effective / tau))

    return fold_current
end

"""
Common inducer parameters.
"""
function inducer_preset(name::Symbol)::InductionParams
    presets = Dict(
        :rifampicin => InductionParams(
            "Rifampicin", :CYP3A4,
            12.0, 0.3,               # Emax 12-fold, EC50 0.3 µM
            :PXR,
            6.0, 2.0, 14.0, 7.0      # mRNA 6h, protein 2d, full 14d, offset 7d
        ),
        :carbamazepine => InductionParams(
            "Carbamazepine", :CYP3A4,
            3.0, 30.0,
            :CAR,
            12.0, 3.0, 7.0, 5.0
        ),
        :phenytoin => InductionParams(
            "Phenytoin", :CYP3A4,
            4.0, 20.0,
            :CAR,
            12.0, 3.0, 7.0, 5.0
        ),
        :smoking => InductionParams(
            "Smoking (PAH)", :CYP1A2,
            2.5, 1.0,                # Approximation
            :AhR,
            6.0, 1.0, 3.0, 3.0
        ),
        :st_johns_wort => InductionParams(
            "St. John's Wort", :CYP3A4,
            1.5, 0.1,
            :PXR,
            6.0, 2.0, 14.0, 7.0
        ),
    )
    return get(presets, name, presets[:rifampicin])
end

export enzyme_induction_dynamics, inducer_preset

# ===========================================================================
# NET DDI CALCULATION
# ===========================================================================

"""
Calculate net DDI effect combining inhibition and induction.

For perpetrators that both inhibit and induce (e.g., ritonavir):
- Acute: Inhibition dominates
- Chronic: Induction may offset inhibition

AUC_ratio = f(inhibition) × f(induction)
"""
function calculate_net_ddi(
    substrate_fm::Float64,           # Fraction metabolized by affected CYP
    inhibitor_conc_uM::Float64,
    Ki_uM::Float64,                  # Competitive inhibition
    kinact_min::Float64,             # MBI rate (0 if none)
    KI_uM::Float64,                  # MBI KI
    inducer_conc_uM::Float64,
    Emax::Float64,                   # Induction Emax
    EC50_uM::Float64,                # Induction EC50
    kdeg_min::Float64,               # CYP degradation rate
    t_days::Float64                  # Time on perpetrator
)::Dict{String, Float64}

    # Competitive inhibition factor
    inhib_factor = 1.0 + inhibitor_conc_uM / Ki_uM

    # MBI factor (time-dependent)
    if kinact_min > 0 && KI_uM > 0
        lambda = kinact_min * inhibitor_conc_uM / (KI_uM + inhibitor_conc_uM)
        mbi_factor = (kdeg_min + lambda) / kdeg_min
    else
        mbi_factor = 1.0
    end

    # Induction factor (time-dependent onset)
    if Emax > 0 && EC50_uM > 0
        fold_induction = 1.0 + Emax * inducer_conc_uM / (EC50_uM + inducer_conc_uM)
        # Onset kinetics (simplified)
        if t_days < 3
            fold_induction = 1.0 + (fold_induction - 1.0) * t_days / 3.0
        end
    else
        fold_induction = 1.0
    end

    # Net effect on clearance
    cl_ratio = fold_induction / (inhib_factor * mbi_factor)

    # AUC ratio (inverse of clearance ratio for affected pathway)
    auc_ratio_affected = 1.0 / cl_ratio

    # Overall AUC ratio considering fm
    auc_ratio_total = 1.0 / (substrate_fm / cl_ratio + (1.0 - substrate_fm))

    return Dict{String, Float64}(
        "inhibition_factor" => inhib_factor,
        "mbi_factor" => mbi_factor,
        "induction_factor" => fold_induction,
        "cl_ratio" => cl_ratio,
        "auc_ratio_affected_pathway" => auc_ratio_affected,
        "auc_ratio_total" => auc_ratio_total
    )
end

export calculate_net_ddi

# ===========================================================================
# HEPATIC DRUG PARAMETERS
# ===========================================================================

"""
Complete hepatic metabolism parameters for a drug.
"""
struct HepaticParams
    # Drug identification
    drug_name::String

    # Physicochemistry
    MW::Float64
    logP::Float64
    fu_plasma::Float64               # Unbound fraction in plasma
    fu_mic::Float64                  # Unbound fraction in microsomes
    blood_plasma_ratio::Float64      # B/P ratio

    # Primary metabolizing enzymes
    primary_cyp::Symbol              # Main CYP (e.g., :CYP3A4)
    fm_primary::Float64              # Fraction metabolized by primary CYP

    secondary_cyp::Union{Symbol,Nothing}
    fm_secondary::Float64

    # CYP kinetics
    Km_uM::Float64
    Vmax_pmol_min_mg::Float64
    CLint_uL_min_mg::Float64         # Intrinsic clearance

    # Transporter involvement
    is_oatp1b1_substrate::Bool
    oatp1b1_km_uM::Float64
    is_oatp1b3_substrate::Bool
    is_pgp_substrate::Bool
    is_mrp2_substrate::Bool
    is_bcrp_substrate::Bool

    # DDI parameters (as perpetrator)
    is_cyp3a4_inhibitor::Bool
    cyp3a4_ki_uM::Float64
    is_cyp3a4_inducer::Bool
    cyp3a4_induction_emax::Float64

    # Extraction ratio
    Eh_predicted::Float64            # Predicted hepatic extraction
end

# ===========================================================================
# HEPATIC CLEARANCE CALCULATIONS
# ===========================================================================

"""
Calculate hepatic clearance with fractal kinetics and zonation.

Uses extended well-stirred model with:
1. Fractal kinetic correction
2. Zonation-weighted enzyme activity
3. Transporter-enzyme interplay
4. Disease state modifications

CLh = Qh × fu × CLint / (Qh + fu × CLint)  [Well-stirred]

With fractal correction:
CLint_fractal = CLint × (Df/Df_ref)^α × transit_time_correction
"""
function calculate_clh(
    params::HepaticParams,
    cyp_enzymes::Dict{Symbol, CYPEnzyme},
    transporters::HepaticTransporters;
    fractal::FractalSinusoid = fractal_sinusoid(),
    zonation::LiverZonation = default_zonation(),
    cirrhosis::Union{Cirrhosis,Nothing} = nothing,
    inhibitor_conc_uM::Float64 = 0.0,
    inhibitor_ki_uM::Float64 = Inf,
    inducer_fold::Float64 = 1.0
)::Dict{String, Float64}

    # Base parameters
    Qh = LIVER_PHYSIOLOGY.Q_hepatic_total_mL_min
    fu = params.fu_plasma

    # Apply cirrhosis modifications
    if cirrhosis !== nothing
        Qh *= cirrhosis.hepatic_blood_flow_fraction
        fu = fu / cirrhosis.albumin_fraction  # ↓albumin → ↑fu
        fu = min(fu, 1.0)
    end

    # Get primary CYP
    cyp = get(cyp_enzymes, params.primary_cyp, cyp_enzymes[:CYP3A4])

    # Intrinsic clearance
    CLint = params.CLint_uL_min_mg * LIVER_PHYSIOLOGY.MPPGL_mg_g *
            LIVER_PHYSIOLOGY.V_liver_mL / 1000  # mL/min

    # Fractal correction
    # Higher Df = better sinusoid access = more efficient clearance
    Df_ref = 1.70  # Reference healthy Df
    Df_actual = cirrhosis !== nothing ? cirrhosis.sinusoid_Df : fractal.Df

    # Fractal scaling exponent (empirical, ~0.5-1.0)
    fractal_scaling = (Df_actual / Df_ref)^0.8

    # Apply fractal kinetics to CLint
    if cyp.use_fractal_kinetics
        # Fractal MM gives lower effective CLint at same Vmax/Km
        # due to anomalous diffusion
        fractal_kinetic_factor = 1.0 - cyp.h_fractal * 0.3  # ~5% reduction for DLA
        CLint *= fractal_scaling * fractal_kinetic_factor
    else
        CLint *= fractal_scaling
    end

    # Enzyme expression in disease
    if cirrhosis !== nothing
        cyp_expression = if params.primary_cyp == :CYP3A4
            cirrhosis.cyp3a4_expression
        elseif params.primary_cyp == :CYP2D6
            cirrhosis.cyp2d6_expression
        elseif params.primary_cyp == :CYP2C9
            cirrhosis.cyp2c9_expression
        else
            0.7  # Default reduction
        end
        CLint *= cyp_expression
    end

    # Transporter limitation (if OATP substrate)
    transporter_limited = false
    if params.is_oatp1b1_substrate
        # OATP1B1 may be rate-limiting
        oatp_clint = transporters.oatp1b1_vmax_pmol_min_mg / params.oatp1b1_km_uM *
                     LIVER_PHYSIOLOGY.MPPGL_mg_g * LIVER_PHYSIOLOGY.V_liver_mL / 1000

        if cirrhosis !== nothing
            oatp_clint *= cirrhosis.oatp1b1_expression
        end

        # Rate-limiting step
        if oatp_clint < CLint
            transporter_limited = true
            CLint = oatp_clint  # Uptake-limited
        end
    end

    # DDI: Inhibition
    if inhibitor_conc_uM > 0 && inhibitor_ki_uM < Inf
        CLint /= (1 + inhibitor_conc_uM / inhibitor_ki_uM)
    end

    # DDI: Induction
    CLint *= inducer_fold

    # Portal shunting in cirrhosis
    shunt_fraction = cirrhosis !== nothing ? cirrhosis.portal_shunt_fraction : 0.0

    # Well-stirred model with shunting
    # CLh = Qh × (1-shunt) × fu × CLint / (Qh × (1-shunt) + fu × CLint)
    Qh_effective = Qh * (1 - shunt_fraction)

    if Qh_effective + fu * CLint > 0
        CLh = Qh_effective * fu * CLint / (Qh_effective + fu * CLint)
    else
        CLh = 0.0
    end

    # Add shunted fraction (bypasses liver entirely)
    # Shunted drug has CLh = 0 for that fraction
    # Effective CLh is reduced by shunting

    # Extraction ratio
    Eh = Qh_effective > 0 ? CLh / Qh_effective : 0.0

    # Bioavailability (oral)
    Fh = 1.0 - Eh

    return Dict{String, Float64}(
        "CLint_mL_min" => CLint,
        "CLh_mL_min" => CLh,
        "Eh" => Eh,
        "Fh" => Fh,
        "Qh_effective_mL_min" => Qh_effective,
        "fu_effective" => fu,
        "fractal_Df" => Df_actual,
        "fractal_scaling" => fractal_scaling,
        "portal_shunt_fraction" => shunt_fraction,
        "transporter_limited" => transporter_limited ? 1.0 : 0.0
    )
end

"""
Calculate extraction ratio classification.
"""
function calculate_extraction_ratio(Eh::Float64)::Symbol
    if Eh < 0.3
        return :low          # Capacity-limited (fu and CLint matter)
    elseif Eh > 0.7
        return :high         # Flow-limited (Qh matters)
    else
        return :intermediate
    end
end

export calculate_clh, calculate_extraction_ratio

# ===========================================================================
# MEDLANG CODE GENERATION
# ===========================================================================

"""
Generate MedLang DSL code for complete hepatic metabolism model.
"""
function generate_hepatic_medlang(
    params::HepaticParams;
    cirrhosis::Union{Cirrhosis,Nothing} = nothing,
    ddi_inhibitor::Union{String,Nothing} = nothing,
    ddi_inducer::Union{String,Nothing} = nothing
)::String

    buf = IOBuffer()

    # Calculate clearance
    cyp_enzymes = default_cyp_enzymes()
    transporters = default_hepatic_transporters()
    fractal = fractal_sinusoid()
    zonation = default_zonation()

    cl_results = calculate_clh(params, cyp_enzymes, transporters;
                               fractal=fractal, cirrhosis=cirrhosis)

    Eh_class = calculate_extraction_ratio(cl_results["Eh"])

    # Zonation info for primary CYP
    zone1_frac, zone3_frac = if params.primary_cyp == :CYP3A4
        zonation.CYP3A4_zone1, zonation.CYP3A4_zone3
    elseif params.primary_cyp == :CYP2D6
        zonation.CYP2D6_zone1, zonation.CYP2D6_zone3
    else
        0.5, 0.5
    end

    disease_section = ""
    if cirrhosis !== nothing
        disease_section = """
    // ================================================================
    // CIRRHOSIS - Child-Pugh $(cirrhosis.child_pugh.class)
    // ================================================================
    disease_state Cirrhosis {
        child_pugh_class: $(cirrhosis.child_pugh.class)
        child_pugh_score: $(cirrhosis.child_pugh.total_score)
        meld_score: $(round(cirrhosis.meld_score, digits=1))

        // FRACTAL ARCHITECTURE COLLAPSE
        // Normal sinusoid Df ≈ 1.70, cirrhosis causes simplification
        sinusoid_fractal_dimension: $(round(cirrhosis.sinusoid_Df, digits=2))

        // Physiological changes
        hepatic_blood_flow: $(round(cirrhosis.hepatic_blood_flow_fraction * 100, digits=0))%
        portal_shunt_fraction: $(round(cirrhosis.portal_shunt_fraction * 100, digits=0))%
        functional_liver_mass: $(round(cirrhosis.functional_liver_mass * 100, digits=0))%

        // Enzyme expression changes
        CYP3A4_expression: $(round(cirrhosis.cyp3a4_expression * 100, digits=0))%
        CYP2D6_expression: $(round(cirrhosis.cyp2d6_expression * 100, digits=0))%
        CYP2C9_expression: $(round(cirrhosis.cyp2c9_expression * 100, digits=0))%

        // Transporter expression changes
        OATP1B1_expression: $(round(cirrhosis.oatp1b1_expression * 100, digits=0))%
        OATP1B3_expression: $(round(cirrhosis.oatp1b3_expression * 100, digits=0))%

        // Protein binding changes
        albumin_fraction: $(round(cirrhosis.albumin_fraction * 100, digits=0))%
        // Note: ↓albumin → ↑fu for albumin-bound drugs
    }
"""
    end

    println(buf, """
model $(params.drug_name)_Hepatic_PBPK {
    // ================================================================
    // HEPATIC METABOLISM MODEL WITH FRACTAL ARCHITECTURE
    // Generated by Darwin PBPK Platform - MedLang DSL
    // ================================================================
    // Drug: $(params.drug_name)
    // MW: $(params.MW) Da
    // logP: $(params.logP)
    // fu,plasma: $(params.fu_plasma)
    // B/P ratio: $(params.blood_plasma_ratio)
    //
    // Primary CYP: $(params.primary_cyp) (fm = $(params.fm_primary))
    // Km: $(params.Km_uM) µM
    // CLint: $(params.CLint_uL_min_mg) µL/min/mg
    //
    // Extraction ratio: $(round(cl_results["Eh"], digits=2)) ($(Eh_class))
    // CLh: $(round(cl_results["CLh_mL_min"], digits=1)) mL/min
    // Fh: $(round(cl_results["Fh"], digits=2))
    // ================================================================

$disease_section
    // ================================================================
    // FRACTAL SINUSOIDAL ARCHITECTURE
    // ================================================================
    // The hepatic sinusoid network follows DLA (diffusion-limited
    // aggregation) pattern with fractal dimension Df ≈ 1.70 in health.
    //
    // CLINICAL SIGNIFICANCE:
    // - Df determines sinusoid branching complexity
    // - Higher Df = more hepatocyte contact = better extraction
    // - Cirrhosis REDUCES Df (simpler, more linear sinusoids)
    // - This explains unpredictable clearance in liver disease!
    //
    architecture sinusoid {
        fractal_dimension: $(round(cl_results["fractal_Df"], digits=2))
        spectral_dimension: 1.33  // ds = 4/3 for DLA
        fractal_exponent_h: 0.17  // h = 1 - ds/2

        // Geometry
        sinusoid_length: 250_um
        sinusoid_diameter: 8_um

        // Flow heterogeneity (NOT well-stirred!)
        transit_time_mean: 8_s
        transit_time_cv: 0.5  // High variability

        // Fractal scaling of clearance
        fractal_scaling_factor: $(round(cl_results["fractal_scaling"], digits=3))
    }

    // ================================================================
    // HEPATIC ZONATION
    // ================================================================
    // Zone 1 (Periportal): High O2, OATP1B1, gluconeogenesis
    // Zone 3 (Pericentral): Low O2, CYP3A4, lipogenesis
    //
    zonation {
        O2_gradient {
            portal: 65_mmHg
            central: 35_mmHg
        }

        // $(params.primary_cyp) distribution
        $(params.primary_cyp)_zone1: $(round(zone1_frac * 100, digits=0))%
        $(params.primary_cyp)_zone3: $(round(zone3_frac * 100, digits=0))%

        // OATP1B1 distribution (periportal dominant)
        OATP1B1_zone1: 70%
        OATP1B1_zone3: 30%

        // SPATIAL MISMATCH for OATP1B1-CYP3A4 substrates!
        // Drug enters periportally (OATP) but is metabolized pericentrally (CYP)
        // Must traverse sinusoid → transit time matters
    }

    // ================================================================
    // FRACTAL MICHAELIS-MENTEN KINETICS
    // ================================================================
    // Classical MM assumes homogeneous, well-stirred compartment.
    // Reality: CYP enzymes on ER membrane (2D), sinusoid is fractal.
    // Anomalous diffusion → fractal kinetics
    //
    // Classical:  v = Vmax × [S] / (Km + [S])
    // Fractal:    v = Vmax × [S]^(1-h) / (Km' + [S]^(1-h))
    //
    kinetics $(params.primary_cyp) {
        type: fractal_michaelis_menten

        Vmax: $(params.Vmax_pmol_min_mg)_pmol/min/mg
        Km: $(params.Km_uM)_uM
        fractal_exponent_h: 0.17

        // Effect of fractal kinetics:
        // - Slightly lower effective clearance than classical MM
        // - Time-dependent rate "constant"
        // - Captures membrane-bound enzyme behavior
    }

    // ================================================================
    // HEPATIC TRANSPORTERS
    // ================================================================

    // Sinusoidal (basolateral) uptake transporters
    transporters_sinusoidal {
        OATP1B1: {
            substrate: $(params.is_oatp1b1_substrate),
            Km: $(params.oatp1b1_km_uM)_uM,
            zone: periportal,
            rate_limiting: $(cl_results["transporter_limited"] > 0)
        }
        OATP1B3: {
            substrate: $(params.is_oatp1b3_substrate)
        }
        OCT1: {
            substrate: false  // For cations
        }
    }

    // Canalicular (apical) efflux transporters
    transporters_canalicular {
        PGP: {
            substrate: $(params.is_pgp_substrate),
            direction: bile
        }
        MRP2: {
            substrate: $(params.is_mrp2_substrate),
            direction: bile,
            substrates: [glucuronides, glutathione_conjugates]
        }
        BCRP: {
            substrate: $(params.is_bcrp_substrate),
            direction: bile
        }
    }

    // ================================================================
    // TRANSPORTER-ENZYME INTERPLAY
    // ================================================================
    // For OATP1B1-CYP3A4 substrates (e.g., statins):
    //
    // Blood → [OATP1B1] → Hepatocyte → [CYP3A4] → Metabolite → [MRP2] → Bile
    //           Zone 1                    Zone 3
    //           (periportal)              (pericentral)
    //
    // Rate-limiting step depends on:
    // - Fractal Df (sinusoid access)
    // - Relative OATP vs CYP expression
    // - Disease state
    //
    interplay {
        rate_limiting_step: $(cl_results["transporter_limited"] > 0 ? "OATP1B1_uptake" : "CYP_metabolism")

        // In cirrhosis:
        // - OATP1B1 expression ↓↓
        // - CYP expression ↓
        // - Df ↓ (less sinusoid access)
        // - Portal shunting (bypass liver)
        // → Complex, unpredictable changes!
    }

    // ================================================================
    // CLEARANCE CALCULATIONS
    // ================================================================

    clearance hepatic {
        // Well-stirred model with fractal correction
        // CLh = Qh × fu × CLint_fractal / (Qh + fu × CLint_fractal)

        Qh: $(round(LIVER_PHYSIOLOGY.Q_hepatic_total_mL_min, digits=0))_mL/min
        fu_plasma: $(params.fu_plasma)
        CLint: $(round(cl_results["CLint_mL_min"], digits=1))_mL/min

        CLh: $(round(cl_results["CLh_mL_min"], digits=1))_mL/min
        Eh: $(round(cl_results["Eh"], digits=3))
        extraction_class: $(Eh_class)

        // Hepatic bioavailability
        Fh: $(round(cl_results["Fh"], digits=3))

        // Portal shunting (if cirrhosis)
        portal_shunt: $(round(cl_results["portal_shunt_fraction"] * 100, digits=0))%
    }

    // ================================================================
    // DDI MECHANISMS
    // ================================================================

    ddi_as_victim {
        primary_cyp: $(params.primary_cyp)
        fm: $(params.fm_primary)

        // Competitive inhibition: ↑Km_apparent
        // Non-competitive: ↓Vmax
        // MBI (mechanism-based): time-dependent, irreversible
        // Induction: delayed onset (1-2 weeks), ↑Vmax
    }

    ddi_as_perpetrator {
        CYP3A4_inhibitor: $(params.is_cyp3a4_inhibitor)
        CYP3A4_Ki: $(params.cyp3a4_ki_uM)_uM
        CYP3A4_inducer: $(params.is_cyp3a4_inducer)
        CYP3A4_induction_Emax: $(params.cyp3a4_induction_emax)
    }

    // ================================================================
    // STATE VARIABLES
    // ================================================================
    state C_plasma: Concentration = 0.0_uM
    state C_liver: Concentration = 0.0_uM
    state C_bile: Concentration = 0.0_uM
    state A_metabolite: Amount = 0.0_mg

    // ================================================================
    // PARAMETERS
    // ================================================================
    param Qh: Real = $(round(cl_results["Qh_effective_mL_min"], digits=1))_mL/min
    param fu: Real = $(round(cl_results["fu_effective"], digits=3))
    param CLint: Real = $(round(cl_results["CLint_mL_min"], digits=1))_mL/min
    param Eh: Real = $(round(cl_results["Eh"], digits=3))
    param Df: Real = $(round(cl_results["fractal_Df"], digits=2))

    // ================================================================
    // ODE EQUATIONS (simplified)
    // ================================================================

    // Hepatic uptake and metabolism
    ode dC_liver/dt = (
        Qh * (C_plasma - C_liver / Kp_liver)  // Blood flow
        - CLint * fu * C_liver                 // Metabolism
    ) / V_liver

    // Using fractal MM for metabolism rate:
    // rate = Vmax * [S]^(1-h) / (Km^(1-h) + [S]^(1-h))

    // ================================================================
    // OBSERVABLES
    // ================================================================
    observable CLh = Qh * fu * CLint / (Qh + fu * CLint)
    observable Eh_observed = CLh / Qh
    observable Fh_observed = 1 - Eh_observed
    observable half_life_h = 0.693 * Vd / CLh * 60
}
""")

    return String(take!(buf))
end

export generate_hepatic_medlang

# ===========================================================================
# SIMULATION
# ===========================================================================

"""
Simulate hepatic clearance with fractal kinetics.
"""
function simulate_hepatic_clearance(
    params::HepaticParams,
    dose_mg::Float64;
    t_max_h::Float64 = 24.0,
    dt_min::Float64 = 1.0,
    cirrhosis::Union{Cirrhosis,Nothing} = nothing,
    Vd_L::Float64 = 70.0,
    CL_other_mL_min::Float64 = 0.0
)::Dict{String, Any}

    # Calculate hepatic clearance
    cyp_enzymes = default_cyp_enzymes()
    transporters = default_hepatic_transporters()
    cl_results = calculate_clh(params, cyp_enzymes, transporters; cirrhosis=cirrhosis)

    CLh = cl_results["CLh_mL_min"]
    CL_total = CLh + CL_other_mL_min
    ke = CL_total / (Vd_L * 1000)  # min⁻¹

    # Initial concentration (IV bolus)
    C0 = (dose_mg * 1000 / params.MW) / (Vd_L * 1000)  # µM

    # Time series
    n_steps = Int(ceil(t_max_h * 60 / dt_min))
    times = Float64[]
    plasma_conc = Float64[]
    liver_conc = Float64[]
    metabolite_amount = Float64[]

    C_plasma = C0
    C_liver = 0.0
    A_met = 0.0

    Kp_liver = 5.0  # Approximate liver partition
    V_liver = LIVER_PHYSIOLOGY.V_liver_mL / 1000  # L

    for step in 1:n_steps
        t_min = step * dt_min

        # Liver uptake and metabolism (simplified)
        # Using well-stirred approximation
        dC_plasma = -ke * C_plasma * dt_min
        C_plasma += dC_plasma
        C_plasma = max(C_plasma, 0.0)

        # Liver concentration (approximate equilibrium)
        C_liver = C_plasma * Kp_liver * (1 - cl_results["Eh"])

        # Metabolite formation
        rate_met = CLh * C_plasma * params.fu_plasma * params.MW / 1000  # mg/min
        A_met += rate_met * dt_min

        push!(times, t_min / 60.0)
        push!(plasma_conc, C_plasma)
        push!(liver_conc, C_liver)
        push!(metabolite_amount, A_met)
    end

    # Calculate PK parameters
    half_life = 0.693 / ke / 60  # hours

    return Dict{String, Any}(
        "time_h" => times,
        "C_plasma_uM" => plasma_conc,
        "C_liver_uM" => liver_conc,
        "A_metabolite_mg" => metabolite_amount,
        "CLh_mL_min" => CLh,
        "CL_total_mL_min" => CL_total,
        "Eh" => cl_results["Eh"],
        "Fh" => cl_results["Fh"],
        "half_life_h" => half_life,
        "fractal_Df" => cl_results["fractal_Df"],
        "clearance_results" => cl_results,
        "params" => params
    )
end

export simulate_hepatic_clearance

# ===========================================================================
# DRUG PRESETS
# ===========================================================================

"""
Create HepaticParams for known drugs.
"""
function drug_hepatic_preset(name::Symbol)::HepaticParams
    presets = Dict(
        # Midazolam: CYP3A4 probe, high extraction
        :midazolam => HepaticParams(
            "Midazolam",
            325.8, 3.9, 0.03, 0.04, 0.55,
            :CYP3A4, 0.95,
            :CYP3A5, 0.05,
            4.0, 500.0, 125.0,
            false, 0.0, false, false, false, false,
            false, Inf, false, 0.0,
            0.5
        ),

        # Atorvastatin: OATP1B1 + CYP3A4 substrate
        :atorvastatin => HepaticParams(
            "Atorvastatin",
            558.6, 4.5, 0.02, 0.1, 0.65,
            :CYP3A4, 0.70,
            :CYP2C8, 0.10,
            15.0, 200.0, 50.0,
            true, 5.0, true, false, false, true,
            false, Inf, false, 0.0,
            0.3
        ),

        # Simvastatin: OATP1B1 + CYP3A4, lactone prodrug
        :simvastatin => HepaticParams(
            "Simvastatin",
            418.6, 4.7, 0.05, 0.1, 0.58,
            :CYP3A4, 0.85,
            nothing, 0.0,
            8.0, 300.0, 75.0,
            true, 8.0, true, false, false, false,
            false, Inf, false, 0.0,
            0.6
        ),

        # Caffeine: CYP1A2 probe, low extraction
        :caffeine => HepaticParams(
            "Caffeine",
            194.2, -0.1, 0.65, 0.9, 1.0,
            :CYP1A2, 0.95,
            nothing, 0.0,
            500.0, 100.0, 0.5,
            false, 0.0, false, false, false, false,
            false, Inf, false, 0.0,
            0.03
        ),

        # Dextromethorphan: CYP2D6 probe
        :dextromethorphan => HepaticParams(
            "Dextromethorphan",
            271.4, 3.5, 0.35, 0.5, 1.0,
            :CYP2D6, 0.80,
            :CYP3A4, 0.15,
            5.0, 200.0, 80.0,
            false, 0.0, false, true, false, false,
            false, Inf, false, 0.0,
            0.4
        ),

        # Warfarin: CYP2C9 substrate, low extraction
        :warfarin => HepaticParams(
            "Warfarin (S)",
            308.3, 2.7, 0.01, 0.05, 0.55,
            :CYP2C9, 0.85,
            :CYP3A4, 0.10,
            3.0, 50.0, 0.2,
            false, 0.0, false, false, false, false,
            false, Inf, false, 0.0,
            0.01
        ),

        # Ketoconazole: CYP3A4 inhibitor
        :ketoconazole => HepaticParams(
            "Ketoconazole",
            531.4, 4.3, 0.01, 0.05, 0.6,
            :CYP3A4, 0.90,
            nothing, 0.0,
            0.5, 100.0, 200.0,
            false, 0.0, false, true, false, false,
            true, 0.015, false, 0.0,  # Strong CYP3A4 inhibitor
            0.7
        ),

        # Rifampicin: CYP3A4 inducer
        :rifampicin => HepaticParams(
            "Rifampicin",
            822.9, 2.7, 0.20, 0.3, 0.9,
            :CYP3A4, 0.50,
            :CYP2C8, 0.20,
            30.0, 150.0, 20.0,
            true, 2.0, true, false, false, false,
            false, Inf, true, 12.0,   # Strong CYP3A4 inducer
            0.3
        ),
    )

    return get(presets, name, presets[:midazolam])
end

export drug_hepatic_preset

end # module
