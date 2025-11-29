# =============================================================================
# GLP-1 AGONIST & BARIATRIC SURGERY PBPK MODEL - MedLang v1.0
# =============================================================================
# Darwin PBPK Platform - Publication-Ready Mechanistic Model
#
# Key Mechanisms:
# 1. GLP-1 receptor agonist effects on GI physiology
#    - Delayed gastric emptying (dose-dependent)
#    - Reduced gut motility
#    - Altered intestinal transit times
#    - Effects on bile secretion
#
# 2. Bariatric Surgery Modifications
#    - Sleeve gastrectomy (VSG): reduced gastric volume, accelerated emptying
#    - Roux-en-Y gastric bypass (RYGB): bypass of duodenum, altered pH
#    - Biliopancreatic diversion (BPD): malabsorption
#    - Gastric banding: restricted intake
#
# 3. Drug-specific considerations
#    - pH-dependent solubility changes
#    - Altered dissolution kinetics
#    - Modified transporter exposure
#    - Changed first-pass metabolism
#
# Literature Basis:
# - Marathe et al. (2011) - Effect of GLP-1 on gastric emptying
# - Padwal et al. (2010) - Drug absorption after bariatric surgery
# - Gesquiere et al. (2015) - Medication management post-bariatric
# - Troke et al. (2014) - GI adaptations after RYGB
# - Horowitz et al. (2012) - GLP-1 and gastric emptying
#
# Author: Dr. Demetrios Agourakis
# Date: November 2025
# =============================================================================

module GLP1BariatricModel

using DifferentialEquations
using Statistics: mean

export GLP1AgonistParams, GLP1Effect, GLP1Receptor
export BariatricSurgery, SurgeryType, PostSurgeryPhysiology
export GIPhysiologyModified, DrugAbsorptionModifier
export calculate_glp1_effect, gastric_emptying_delay
export transit_time_modification, dissolution_rate_modifier
export calculate_fa_bariatric, bioavailability_change
export simulate_oral_with_glp1, simulate_oral_post_bariatric
export glp1_agonist_preset, surgery_preset
export create_modified_gi_params, validate_glp1_model

# =============================================================================
# ENUMERATIONS
# =============================================================================

"""
Types of bariatric surgery procedures.
"""
@enum SurgeryType begin
    NO_SURGERY              # Control/normal
    SLEEVE_GASTRECTOMY      # VSG - vertical sleeve gastrectomy
    ROUX_EN_Y_BYPASS        # RYGB - Roux-en-Y gastric bypass
    BILIOPANCREATIC_DIVERSION  # BPD/DS
    GASTRIC_BANDING         # Adjustable gastric band (AGB)
    MINI_GASTRIC_BYPASS     # One anastomosis/mini gastric bypass
end

# =============================================================================
# GLP-1 RECEPTOR AND AGONIST STRUCTURES
# =============================================================================

"""
    GLP1Receptor

GLP-1 receptor expression and sensitivity in GI tract.
"""
struct GLP1Receptor
    # Regional expression (relative to reference)
    stomach_expression::Float64     # Pyloric region
    duodenum_expression::Float64
    jejunum_expression::Float64
    ileum_expression::Float64       # L-cells (endogenous GLP-1)
    colon_expression::Float64

    # Receptor sensitivity
    ec50_pM::Float64               # EC50 for receptor activation
    emax::Float64                  # Maximum effect
    hill_coefficient::Float64       # Hill coefficient for cooperativity
end

"""
    GLP1AgonistParams

Parameters for GLP-1 receptor agonists.
"""
struct GLP1AgonistParams
    name::String

    # Pharmacokinetics
    molecular_weight::Float64
    half_life_h::Float64           # Elimination half-life
    tmax_h::Float64                # Time to peak concentration
    bioavailability::Float64       # SC bioavailability

    # Receptor binding
    kd_pM::Float64                 # Dissociation constant
    relative_potency::Float64      # Relative to native GLP-1

    # Dosing
    typical_dose_mg::Float64
    dosing_frequency::Symbol       # :daily, :weekly

    # Effect parameters
    gastric_emptying_delay_factor::Float64  # Max delay factor
    effect_onset_h::Float64        # Time to onset of GI effects
    effect_duration_h::Float64     # Duration of GI effects

    # Additional effects
    reduces_appetite::Bool
    affects_bile_secretion::Bool
    nausea_propensity::Float64     # 0-1, dose-dependent nausea
end

"""
    GLP1Effect

Calculated effects of GLP-1 agonist at a given concentration.
"""
struct GLP1Effect
    # Gastric effects
    gastric_emptying_t50_factor::Float64  # Multiplier for T50 (>1 = slower)
    pyloric_tone_increase::Float64        # Fraction increase

    # Intestinal effects
    small_bowel_transit_factor::Float64   # Multiplier for transit time
    colon_transit_factor::Float64

    # Secretion effects
    bile_secretion_factor::Float64        # Effect on bile (reduced)
    gastric_acid_factor::Float64          # Effect on acid secretion

    # Derived absorption effects
    ka_reduction_factor::Float64          # Effect on absorption rate
    dissolution_time_factor::Float64      # More time for dissolution
end

# =============================================================================
# BARIATRIC SURGERY STRUCTURES
# =============================================================================

"""
    BariatricSurgery

Parameters describing bariatric surgery type and modifications.
"""
struct BariatricSurgery
    surgery_type::SurgeryType
    time_since_surgery_months::Float64

    # Anatomical changes
    gastric_volume_mL::Float64           # Residual gastric volume
    gastric_pouch_pH::Float64            # Often higher post-surgery
    duodenum_bypassed::Bool              # RYGB, BPD
    biliopancreatic_limb_cm::Float64     # Length if bypass
    alimentary_limb_cm::Float64          # Roux limb length
    common_channel_cm::Float64           # For absorption

    # Functional changes
    gastric_emptying_rate::Float64       # Relative to normal (often faster)
    intestinal_transit_factor::Float64   # Often accelerated
    bile_mixing_delay_min::Float64       # Time to bile contact

    # Metabolic changes
    gut_hormone_amplification::Float64   # Enhanced GLP-1/PYY response
    dumping_syndrome_risk::Float64       # 0-1
end

"""
    PostSurgeryPhysiology

Time-dependent physiological adaptations after bariatric surgery.
"""
struct PostSurgeryPhysiology
    # Acute phase (0-3 months)
    acute_inflammation::Float64
    gut_adaptation_factor::Float64

    # Chronic phase (>12 months)
    intestinal_hypertrophy::Float64      # Villous adaptation
    transporter_upregulation::Float64    # Compensatory increase

    # pH changes
    gastric_pH::Float64                  # Often 4-6 vs normal 1-3
    duodenal_pH::Float64                 # If not bypassed

    # Surface area changes
    effective_absorption_area::Float64   # Fraction of normal

    # Bile and enzyme changes
    bile_concentration_factor::Float64
    pancreatic_enzyme_factor::Float64
end

"""
    GIPhysiologyModified

Modified GI physiology accounting for GLP-1 and/or bariatric changes.
"""
struct GIPhysiologyModified
    # Volumes (mL)
    stomach_volume::Float64
    duodenum_volume::Float64
    jejunum_volume::Float64
    ileum_volume::Float64
    colon_volume::Float64

    # pH values
    stomach_pH::Float64
    duodenum_pH::Float64
    jejunum_pH::Float64
    ileum_pH::Float64

    # Transit times (min)
    gastric_emptying_t50::Float64
    duodenum_transit::Float64
    jejunum_transit::Float64
    ileum_transit::Float64
    colon_transit::Float64

    # Surface areas (cm²) - may be reduced post-surgery
    duodenum_sa::Float64
    jejunum_sa::Float64
    ileum_sa::Float64
    colon_sa::Float64

    # Bile salt concentrations (mM)
    duodenum_bile::Float64
    jejunum_bile::Float64
    ileum_bile::Float64

    # Flags
    duodenum_functional::Bool    # False if bypassed
    bile_delay_min::Float64      # Delay to bile mixing
end

"""
    DrugAbsorptionModifier

Drug-specific absorption modifications based on GLP-1/bariatric status.
"""
struct DrugAbsorptionModifier
    drug_name::String

    # Solubility effects
    ph_sensitive::Bool
    optimal_ph::Float64
    solubility_factor_at_modified_ph::Float64

    # Dissolution effects
    dissolution_rate_factor::Float64

    # Permeability effects
    ka_factor::Float64                   # Absorption rate modifier

    # First-pass effects
    fg_factor::Float64                   # Gut availability modifier
    fh_factor::Float64                   # Hepatic availability modifier

    # Transporter effects
    transporter_exposure_factor::Float64 # If duodenum bypassed

    # Overall bioavailability change
    f_total_factor::Float64

    # Clinical relevance
    dose_adjustment_needed::Bool
    recommended_adjustment::Float64      # e.g., 0.5 = reduce by 50%
end

# =============================================================================
# GLP-1 EFFECT CALCULATIONS
# =============================================================================

"""
    calculate_glp1_effect(agonist, concentration_pM, receptor)

Calculate GLP-1 agonist effects on GI physiology.

Uses Emax model:
    Effect = Emax × C^n / (EC50^n + C^n)
"""
function calculate_glp1_effect(
    agonist::GLP1AgonistParams,
    concentration_pM::Float64,
    receptor::GLP1Receptor = default_glp1_receptor()
)::GLP1Effect
    # Effective concentration adjusted for potency
    c_eff = concentration_pM * agonist.relative_potency

    # Emax model for gastric emptying delay
    ec50 = receptor.ec50_pM
    emax = receptor.emax
    n = receptor.hill_coefficient

    effect_fraction = emax * c_eff^n / (ec50^n + c_eff^n)

    # Gastric emptying delay (T50 multiplier)
    # At max effect, T50 can be 2-4x longer
    max_delay = agonist.gastric_emptying_delay_factor
    t50_factor = 1.0 + (max_delay - 1.0) * effect_fraction

    # Pyloric tone increase (contributes to delayed emptying)
    pyloric_increase = 0.5 * effect_fraction

    # Small bowel transit (modest slowing)
    sbt_factor = 1.0 + 0.3 * effect_fraction

    # Colon transit (minimal effect)
    colon_factor = 1.0 + 0.1 * effect_fraction

    # Bile secretion (reduced)
    bile_factor = 1.0 - 0.2 * effect_fraction

    # Gastric acid (reduced)
    acid_factor = 1.0 - 0.3 * effect_fraction

    # Derived effects on absorption
    ka_reduction = 1.0 / t50_factor  # Slower emptying = slower ka
    dissolution_benefit = t50_factor  # More time in stomach for dissolution

    return GLP1Effect(
        t50_factor,
        pyloric_increase,
        sbt_factor,
        colon_factor,
        bile_factor,
        acid_factor,
        ka_reduction,
        dissolution_benefit
    )
end

"""
    gastric_emptying_delay(agonist, dose_mg, time_since_dose_h)

Calculate time-dependent gastric emptying delay.
"""
function gastric_emptying_delay(
    agonist::GLP1AgonistParams,
    dose_mg::Float64,
    time_since_dose_h::Float64
)::Float64
    # Simple PK model for GLP-1 agonist concentration
    if time_since_dose_h < agonist.effect_onset_h
        return 1.0  # No effect yet
    end

    # Time after onset
    t_eff = time_since_dose_h - agonist.effect_onset_h

    # Concentration profile (simplified one-compartment)
    ke = log(2) / agonist.half_life_h
    ka_sc = 0.5  # SC absorption rate

    # Two-compartment absorption-elimination
    c_rel = (exp(-ke * t_eff) - exp(-ka_sc * t_eff)) / (ka_sc - ke)
    c_rel = max(0.0, c_rel)

    # Normalize to typical peak
    c_peak = (exp(-ke * agonist.tmax_h) - exp(-ka_sc * agonist.tmax_h)) / (ka_sc - ke)
    if c_peak > 0
        c_normalized = c_rel / c_peak
    else
        c_normalized = 0.0
    end

    # Dose-dependent effect (assume linear up to typical dose)
    dose_factor = min(1.5, dose_mg / agonist.typical_dose_mg)

    # Effect on T50
    max_delay = agonist.gastric_emptying_delay_factor
    delay_factor = 1.0 + (max_delay - 1.0) * c_normalized * dose_factor

    return delay_factor
end

"""
    transit_time_modification(glp1_effect, surgery)

Calculate modified transit times combining GLP-1 and surgical effects.
"""
function transit_time_modification(
    glp1_effect::Union{GLP1Effect, Nothing},
    surgery::Union{BariatricSurgery, Nothing}
)::NamedTuple
    # Base transit times (min)
    base_gastric_t50 = 15.0
    base_duodenum = 10.0
    base_jejunum = 90.0
    base_ileum = 120.0
    base_colon = 720.0

    # GLP-1 effects (slow things down)
    if glp1_effect !== nothing
        gastric_factor = glp1_effect.gastric_emptying_t50_factor
        sb_factor = glp1_effect.small_bowel_transit_factor
        colon_factor = glp1_effect.colon_transit_factor
    else
        gastric_factor = 1.0
        sb_factor = 1.0
        colon_factor = 1.0
    end

    # Surgery effects (often speed things up, especially RYGB)
    if surgery !== nothing && surgery.surgery_type != NO_SURGERY
        # Gastric emptying often faster post-surgery (smaller pouch, no pylorus control)
        gastric_surgery_factor = surgery.gastric_emptying_rate
        transit_surgery_factor = surgery.intestinal_transit_factor
    else
        gastric_surgery_factor = 1.0
        transit_surgery_factor = 1.0
    end

    # Combined effects
    # Note: GLP-1 agonists can counteract rapid emptying post-surgery
    gastric_t50 = base_gastric_t50 * gastric_factor * gastric_surgery_factor
    duodenum_transit = base_duodenum * sb_factor * transit_surgery_factor
    jejunum_transit = base_jejunum * sb_factor * transit_surgery_factor
    ileum_transit = base_ileum * sb_factor * transit_surgery_factor
    colon_transit = base_colon * colon_factor

    return (
        gastric_t50 = gastric_t50,
        duodenum = duodenum_transit,
        jejunum = jejunum_transit,
        ileum = ileum_transit,
        colon = colon_transit,
        total_small_bowel = duodenum_transit + jejunum_transit + ileum_transit
    )
end

# =============================================================================
# BARIATRIC SURGERY EFFECTS
# =============================================================================

"""
    calculate_surgery_physiology(surgery)

Calculate post-surgery GI physiology modifications.
"""
function calculate_surgery_physiology(
    surgery::BariatricSurgery
)::PostSurgeryPhysiology
    t_months = surgery.time_since_surgery_months

    # Time-dependent adaptation
    # Acute phase inflammation (first 3 months)
    if t_months < 3
        acute = 1.0 - t_months / 3
        adaptation = t_months / 3 * 0.3
    else
        acute = 0.0
        adaptation = 0.3 + min(0.7, (t_months - 3) / 12 * 0.7)
    end

    # Intestinal hypertrophy (compensatory)
    if t_months > 6
        hypertrophy = min(1.5, 1.0 + (t_months - 6) / 24)
    else
        hypertrophy = 1.0
    end

    # Transporter upregulation
    transporter_upreg = 1.0 + adaptation * 0.5

    # pH changes by surgery type
    if surgery.surgery_type == SLEEVE_GASTRECTOMY
        gastric_ph = 3.0 + 1.5 * (1 - adaptation)  # Higher acutely, normalizes
        duodenal_ph = 6.0
        absorption_area = 1.0  # No bypass
        bile_factor = 1.0
        enzyme_factor = 1.0
    elseif surgery.surgery_type == ROUX_EN_Y_BYPASS
        gastric_ph = 5.0  # Small pouch, less acid
        duodenal_ph = 6.5  # Bypassed
        absorption_area = 0.7  # Reduced due to bypass
        bile_factor = 0.6  # Delayed mixing
        enzyme_factor = 0.7
    elseif surgery.surgery_type == BILIOPANCREATIC_DIVERSION
        gastric_ph = 4.5
        duodenal_ph = 7.0
        absorption_area = 0.4  # Significant malabsorption
        bile_factor = 0.3
        enzyme_factor = 0.3
    elseif surgery.surgery_type == GASTRIC_BANDING
        gastric_ph = 2.5  # Near normal
        duodenal_ph = 6.0
        absorption_area = 1.0
        bile_factor = 1.0
        enzyme_factor = 1.0
    else
        gastric_ph = 2.0
        duodenal_ph = 6.0
        absorption_area = 1.0
        bile_factor = 1.0
        enzyme_factor = 1.0
    end

    return PostSurgeryPhysiology(
        acute,
        adaptation,
        hypertrophy,
        transporter_upreg,
        gastric_ph,
        duodenal_ph,
        absorption_area,
        bile_factor,
        enzyme_factor
    )
end

"""
    dissolution_rate_modifier(drug_pka, drug_charge, modified_ph, normal_ph)

Calculate dissolution rate change based on pH modification.
"""
function dissolution_rate_modifier(
    drug_pka::Union{Float64, Nothing},
    drug_charge::Symbol,
    modified_ph::Float64,
    normal_ph::Float64 = 2.0
)::Float64
    if drug_pka === nothing
        return 1.0  # Neutral drug, no pH effect
    end

    # Henderson-Hasselbalch for ionization
    if drug_charge == :acid
        # Acids more soluble at high pH (ionized)
        ionized_normal = 1.0 / (1.0 + 10^(drug_pka - normal_ph))
        ionized_modified = 1.0 / (1.0 + 10^(drug_pka - modified_ph))
    elseif drug_charge == :base
        # Bases more soluble at low pH (ionized)
        ionized_normal = 1.0 / (1.0 + 10^(normal_ph - drug_pka))
        ionized_modified = 1.0 / (1.0 + 10^(modified_ph - drug_pka))
    else
        return 1.0
    end

    # Dissolution rate proportional to solubility
    # Ionized form typically 10-1000x more soluble
    sol_factor = 100.0  # Assume 100x for ionized

    sol_normal = 1.0 + sol_factor * ionized_normal
    sol_modified = 1.0 + sol_factor * ionized_modified

    return sol_modified / sol_normal
end

"""
    calculate_fa_bariatric(drug, surgery, physiology)

Calculate fraction absorbed (Fa) after bariatric surgery.
"""
function calculate_fa_bariatric(
    drug_solubility_mg_mL::Float64,
    drug_pka::Union{Float64, Nothing},
    drug_charge::Symbol,
    drug_permeability::Float64,  # Peff in cm/s
    surgery::BariatricSurgery,
    physiology::PostSurgeryPhysiology
)::NamedTuple
    # Base Fa from permeability (assuming normal anatomy)
    # Fa = 1 - exp(-2 × Peff × transit_time × SA/V)

    # Dissolution factor
    diss_factor = dissolution_rate_modifier(
        drug_pka, drug_charge,
        physiology.gastric_pH, 2.0
    )

    # Solubility-limited absorption
    if drug_solubility_mg_mL < 0.1  # Poorly soluble
        sol_limitation = drug_solubility_mg_mL / 0.1 * diss_factor
    else
        sol_limitation = 1.0
    end

    # Permeability-limited absorption with modified surface area
    perm_factor = physiology.effective_absorption_area

    # Transporter upregulation can compensate
    if drug_permeability < 1e-5  # Transporter-dependent
        perm_factor *= physiology.transporter_upregulation
    end

    # Duodenum bypass effect
    if surgery.duodenum_bypassed
        # Loss of duodenal absorption (important for some drugs)
        duodenum_contribution = 0.15  # ~15% of absorption normally
        bypass_loss = duodenum_contribution
    else
        bypass_loss = 0.0
    end

    # Bile salt effect on lipophilic drugs
    bile_effect = physiology.bile_concentration_factor

    # Calculate final Fa
    fa_base = 0.9  # Assume good baseline permeability
    fa_modified = fa_base * sol_limitation * perm_factor * (1 - bypass_loss)

    return (
        fa = clamp(fa_modified, 0.05, 1.0),
        dissolution_factor = diss_factor,
        solubility_limitation = sol_limitation,
        permeability_factor = perm_factor,
        bypass_loss = bypass_loss,
        bile_effect = bile_effect
    )
end

"""
    bioavailability_change(drug_name, surgery_type, time_months)

Get literature-based bioavailability change for specific drug/surgery combinations.
"""
function bioavailability_change(
    drug_name::Symbol,
    surgery_type::SurgeryType,
    time_months::Float64 = 12.0
)::NamedTuple
    # Literature-based changes for common drugs
    # Data from Padwal et al., Gesquiere et al., and other reviews

    changes = Dict(
        # Metformin: Increased absorption post-RYGB (enhanced GLP-1)
        (:metformin, ROUX_EN_Y_BYPASS) => (factor=1.5, mechanism="enhanced_glp1_response"),
        (:metformin, SLEEVE_GASTRECTOMY) => (factor=1.2, mechanism="faster_emptying"),

        # Levothyroxine: Decreased (pH-dependent, needs acid)
        (:levothyroxine, ROUX_EN_Y_BYPASS) => (factor=0.6, mechanism="reduced_acid"),
        (:levothyroxine, SLEEVE_GASTRECTOMY) => (factor=0.8, mechanism="reduced_acid"),

        # Atorvastatin: Variable (CYP3A4 changes)
        (:atorvastatin, ROUX_EN_Y_BYPASS) => (factor=1.3, mechanism="reduced_first_pass"),

        # Sertraline: Increased (lipophilic, bile effects)
        (:sertraline, ROUX_EN_Y_BYPASS) => (factor=1.4, mechanism="altered_bile_mixing"),

        # Tacrolimus: Highly variable, often increased
        (:tacrolimus, ROUX_EN_Y_BYPASS) => (factor=2.0, mechanism="reduced_cyp3a4_pgp"),

        # Cyclosporine: Decreased (needs bile for absorption)
        (:cyclosporine, ROUX_EN_Y_BYPASS) => (factor=0.5, mechanism="bile_dependent"),

        # Omeprazole: Decreased (acid-dependent release)
        (:omeprazole, ROUX_EN_Y_BYPASS) => (factor=0.7, mechanism="ph_dependent"),

        # Alendronate: Decreased (needs fasting, acid)
        (:alendronate, ROUX_EN_Y_BYPASS) => (factor=0.4, mechanism="acid_dependent"),

        # Duloxetine: Decreased (enteric coated)
        (:duloxetine, ROUX_EN_Y_BYPASS) => (factor=0.6, mechanism="enteric_coating"),

        # Acetaminophen: Often faster Tmax, similar AUC
        (:acetaminophen, ROUX_EN_Y_BYPASS) => (factor=1.0, mechanism="faster_absorption"),
        (:acetaminophen, SLEEVE_GASTRECTOMY) => (factor=1.0, mechanism="faster_absorption"),
    )

    key = (drug_name, surgery_type)
    if haskey(changes, key)
        result = changes[key]
        return (
            f_factor = result.factor,
            mechanism = result.mechanism,
            evidence = "literature",
            time_dependent = time_months < 6
        )
    else
        return (
            f_factor = 1.0,
            mechanism = "unknown",
            evidence = "assumed",
            time_dependent = false
        )
    end
end

# =============================================================================
# ODE SYSTEM FOR MODIFIED GI ABSORPTION
# =============================================================================

"""
    modified_gi_ode!(du, u, p, t)

ODE system for oral absorption with GLP-1 and/or bariatric modifications.

State variables:
1. A_stomach_solid - Solid drug in stomach
2. A_stomach_dissolved - Dissolved drug in stomach
3. A_duodenum_solid - Solid in duodenum (if not bypassed)
4. A_duodenum_dissolved - Dissolved in duodenum
5. A_jejunum_solid
6. A_jejunum_dissolved
7. A_ileum_solid
8. A_ileum_dissolved
9. A_colon_solid
10. A_colon_dissolved
11. A_portal - Amount in portal vein
12. A_systemic - Systemically available amount
"""
function modified_gi_ode!(du, u, p, t)
    # Unpack parameters
    gi = p.gi_physiology
    drug = p.drug_params
    Fg = p.fg
    Fh = p.fh

    # State variables
    A_sto_s, A_sto_d = u[1], u[2]
    A_duo_s, A_duo_d = u[3], u[4]
    A_jej_s, A_jej_d = u[5], u[6]
    A_ile_s, A_ile_d = u[7], u[8]
    A_col_s, A_col_d = u[9], u[10]
    A_portal = u[11]
    A_systemic = u[12]

    # Dissolution rate constants (1/min)
    k_diss_sto = drug.dissolution_rate * gi.stomach_pH / 2.0  # pH effect
    k_diss_duo = drug.dissolution_rate * (gi.duodenum_bile / 5.0 + 0.5)  # Bile effect
    k_diss_jej = drug.dissolution_rate * (gi.jejunum_bile / 5.0 + 0.5)
    k_diss_ile = drug.dissolution_rate
    k_diss_col = drug.dissolution_rate * 0.3  # Slower in colon

    # Transit rate constants (1/min)
    k_ge = log(2) / gi.gastric_emptying_t50  # Gastric emptying
    k_duo = 1.0 / gi.duodenum_transit
    k_jej = 1.0 / gi.jejunum_transit
    k_ile = 1.0 / gi.ileum_transit
    k_col = 1.0 / gi.colon_transit

    # Absorption rate constants (1/min)
    # Scaled by surface area relative to normal
    ka_duo = drug.ka_base * gi.duodenum_sa / 2000.0
    ka_jej = drug.ka_base * gi.jejunum_sa / 18000.0
    ka_ile = drug.ka_base * gi.ileum_sa / 12000.0
    ka_col = drug.ka_base * 0.1  # Minimal colon absorption

    # If duodenum bypassed
    if !gi.duodenum_functional
        ka_duo = 0.0
        k_duo = 0.0
        # Direct transit from stomach to jejunum after bile mixing delay
        k_ge_to_jej = k_ge
    else
        k_ge_to_jej = 0.0
    end

    # Portal to liver clearance
    k_portal = 1.0  # Fast portal blood flow

    # === ODEs ===

    # Stomach solid
    du[1] = -k_diss_sto * A_sto_s - k_ge * A_sto_s

    # Stomach dissolved
    du[2] = k_diss_sto * A_sto_s - k_ge * A_sto_d

    # Duodenum solid
    if gi.duodenum_functional
        du[3] = k_ge * A_sto_s - k_diss_duo * A_duo_s - k_duo * A_duo_s
    else
        du[3] = 0.0  # Bypassed
    end

    # Duodenum dissolved
    if gi.duodenum_functional
        du[4] = k_ge * A_sto_d + k_diss_duo * A_duo_s - ka_duo * A_duo_d - k_duo * A_duo_d
    else
        du[4] = 0.0
    end

    # Jejunum solid
    if gi.duodenum_functional
        du[5] = k_duo * A_duo_s - k_diss_jej * A_jej_s - k_jej * A_jej_s
    else
        # Direct from stomach (RYGB anatomy)
        du[5] = k_ge * A_sto_s - k_diss_jej * A_jej_s - k_jej * A_jej_s
    end

    # Jejunum dissolved
    if gi.duodenum_functional
        du[6] = k_duo * A_duo_d + k_diss_jej * A_jej_s - ka_jej * A_jej_d - k_jej * A_jej_d
    else
        du[6] = k_ge * A_sto_d + k_diss_jej * A_jej_s - ka_jej * A_jej_d - k_jej * A_jej_d
    end

    # Ileum solid
    du[7] = k_jej * A_jej_s - k_diss_ile * A_ile_s - k_ile * A_ile_s

    # Ileum dissolved
    du[8] = k_jej * A_jej_d + k_diss_ile * A_ile_s - ka_ile * A_ile_d - k_ile * A_ile_d

    # Colon solid
    du[9] = k_ile * A_ile_s - k_diss_col * A_col_s - k_col * A_col_s

    # Colon dissolved
    du[10] = k_ile * A_ile_d + k_diss_col * A_col_s - ka_col * A_col_d - k_col * A_col_d

    # Portal vein (sum of absorbed × Fg)
    total_absorbed = ka_duo * A_duo_d + ka_jej * A_jej_d + ka_ile * A_ile_d + ka_col * A_col_d
    du[11] = Fg * total_absorbed - k_portal * A_portal

    # Systemic (portal × Fh)
    du[12] = Fh * k_portal * A_portal

    return nothing
end

# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

"""
    simulate_oral_with_glp1(drug, dose_mg, glp1_agonist, glp1_dose; kwargs...)

Simulate oral drug absorption with concurrent GLP-1 agonist therapy.
"""
function simulate_oral_with_glp1(
    drug_ka::Float64,
    drug_dissolution::Float64,
    drug_pka::Union{Float64, Nothing},
    drug_charge::Symbol,
    dose_mg::Float64,
    glp1::GLP1AgonistParams,
    glp1_dose_mg::Float64,
    time_since_glp1_h::Float64;
    tspan::Tuple{Float64, Float64} = (0.0, 24.0),
    fg::Float64 = 0.9,
    fh::Float64 = 0.8
)
    # Calculate GLP-1 effect at time of dosing
    delay_factor = gastric_emptying_delay(glp1, glp1_dose_mg, time_since_glp1_h)

    # Modify GI physiology
    gi = GIPhysiologyModified(
        250.0,   # stomach_volume
        50.0, 100.0, 80.0, 200.0,  # intestinal volumes
        2.0, 6.0, 6.5, 7.2,        # pH values
        15.0 * delay_factor,       # Modified gastric emptying T50
        10.0 * 1.2, 90.0 * 1.2, 120.0 * 1.2, 720.0,  # Transit times
        2000.0, 18000.0, 12000.0, 1500.0,  # Surface areas
        8.0, 6.0, 2.0,             # Bile concentrations
        true,                       # Duodenum functional
        0.0                         # No bile delay
    )

    # Drug parameters
    drug_params = (
        ka_base = drug_ka / delay_factor,  # Adjusted for slower emptying
        dissolution_rate = drug_dissolution,
        pka = drug_pka,
        charge = drug_charge
    )

    # Parameters for ODE
    p = (
        gi_physiology = gi,
        drug_params = drug_params,
        fg = fg,
        fh = fh
    )

    # Initial conditions (all drug in stomach as solid)
    u0 = zeros(12)
    u0[1] = dose_mg  # Stomach solid

    # Solve ODE
    prob = ODEProblem(modified_gi_ode!, u0, tspan .* 60.0, p)  # Convert to minutes
    sol = solve(prob, Tsit5(), saveat=1.0)

    # Extract results
    times_h = sol.t ./ 60.0
    A_systemic = [s[12] for s in sol.u]
    A_gut_total = [sum(s[1:10]) for s in sol.u]

    # PK metrics
    F_observed = A_systemic[end] / dose_mg

    return (
        times = times_h,
        A_systemic = A_systemic,
        A_gut = A_gut_total,
        F = F_observed,
        delay_factor = delay_factor,
        glp1_effect = "Gastric emptying delayed $(round(delay_factor, digits=2))x"
    )
end

"""
    simulate_oral_post_bariatric(drug, dose_mg, surgery; kwargs...)

Simulate oral drug absorption after bariatric surgery.
"""
function simulate_oral_post_bariatric(
    drug_ka::Float64,
    drug_dissolution::Float64,
    drug_pka::Union{Float64, Nothing},
    drug_charge::Symbol,
    dose_mg::Float64,
    surgery::BariatricSurgery;
    tspan::Tuple{Float64, Float64} = (0.0, 24.0),
    fg::Float64 = 0.9,
    fh::Float64 = 0.8
)
    # Calculate surgery-specific physiology
    physiology = calculate_surgery_physiology(surgery)

    # Build modified GI structure
    if surgery.surgery_type == ROUX_EN_Y_BYPASS
        # RYGB: small pouch, bypassed duodenum, rapid emptying
        gi = GIPhysiologyModified(
            30.0,    # Tiny gastric pouch
            0.0, surgery.alimentary_limb_cm * 0.5, 80.0, 200.0,
            physiology.gastric_pH, 6.5, 6.5, 7.2,
            5.0 * surgery.gastric_emptying_rate,  # Often faster
            0.0,     # Duodenum bypassed
            60.0 * surgery.intestinal_transit_factor,
            100.0 * surgery.intestinal_transit_factor,
            720.0,
            0.0,     # No duodenal SA
            12000.0 * physiology.effective_absorption_area,
            10000.0 * physiology.effective_absorption_area,
            1500.0,
            0.0,     # Bile mixing delayed
            3.0 * physiology.bile_concentration_factor,
            1.0 * physiology.bile_concentration_factor,
            false,   # Duodenum NOT functional
            surgery.bile_mixing_delay_min
        )
    elseif surgery.surgery_type == SLEEVE_GASTRECTOMY
        # VSG: reduced stomach, normal anatomy otherwise
        gi = GIPhysiologyModified(
            surgery.gastric_volume_mL,
            50.0, 100.0, 80.0, 200.0,
            physiology.gastric_pH, 6.0, 6.5, 7.2,
            10.0 * surgery.gastric_emptying_rate,
            10.0, 90.0, 120.0, 720.0,
            2000.0, 18000.0, 12000.0, 1500.0,
            8.0, 6.0, 2.0,
            true,
            0.0
        )
    else
        # Default (no surgery or other)
        gi = GIPhysiologyModified(
            250.0, 50.0, 100.0, 80.0, 200.0,
            2.0, 6.0, 6.5, 7.2,
            15.0, 10.0, 90.0, 120.0, 720.0,
            2000.0, 18000.0, 12000.0, 1500.0,
            8.0, 6.0, 2.0,
            true, 0.0
        )
    end

    # Adjust Fg/Fh based on surgery
    # RYGB often reduces first-pass due to altered CYP3A4 exposure
    if surgery.surgery_type == ROUX_EN_Y_BYPASS
        fg_adj = fg * 1.2  # Less gut metabolism
        fh_adj = fh  # Hepatic usually unchanged
    else
        fg_adj = fg
        fh_adj = fh
    end

    # Drug parameters
    drug_params = (
        ka_base = drug_ka * surgery.gastric_emptying_rate,
        dissolution_rate = drug_dissolution * dissolution_rate_modifier(
            drug_pka, drug_charge, physiology.gastric_pH
        ),
        pka = drug_pka,
        charge = drug_charge
    )

    p = (
        gi_physiology = gi,
        drug_params = drug_params,
        fg = clamp(fg_adj, 0.1, 1.0),
        fh = fh_adj
    )

    u0 = zeros(12)
    u0[1] = dose_mg

    prob = ODEProblem(modified_gi_ode!, u0, tspan .* 60.0, p)
    sol = solve(prob, Tsit5(), saveat=1.0)

    times_h = sol.t ./ 60.0
    A_systemic = [s[12] for s in sol.u]

    # Find Tmax and Cmax (from rate)
    rates = diff(A_systemic)
    if length(rates) > 0
        tmax_idx = argmax(rates) + 1
        tmax = times_h[min(tmax_idx, length(times_h))]
    else
        tmax = 0.0
    end

    F_observed = A_systemic[end] / dose_mg

    return (
        times = times_h,
        A_systemic = A_systemic,
        F = F_observed,
        tmax = tmax,
        surgery = surgery.surgery_type,
        physiology = physiology
    )
end

# =============================================================================
# PRESETS
# =============================================================================

"""
Default GLP-1 receptor expression profile.
"""
function default_glp1_receptor()::GLP1Receptor
    return GLP1Receptor(
        0.8,    # stomach (pylorus)
        1.0,    # duodenum
        1.0,    # jejunum
        2.0,    # ileum (L-cells, high expression)
        1.5,    # colon
        50.0,   # EC50 pM
        1.0,    # Emax
        1.5     # Hill coefficient
    )
end

"""
GLP-1 agonist presets from clinical data.
"""
function glp1_agonist_preset(name::Symbol)::GLP1AgonistParams
    presets = Dict(
        :semaglutide_oral => GLP1AgonistParams(
            "Semaglutide (oral)",
            4113.6,   # MW
            168.0,    # t1/2 (7 days = 168 h)
            1.0,      # Tmax
            0.01,     # Very low oral F
            0.1,      # Kd pM
            5.0,      # High potency
            14.0,     # Typical dose mg
            :daily,
            2.5,      # Strong gastric delay
            0.5,      # Effect onset
            24.0,     # Effect duration
            true, true, 0.3
        ),

        :semaglutide_sc => GLP1AgonistParams(
            "Semaglutide (SC)",
            4113.6,
            168.0,    # Weekly dosing
            24.0,     # Tmax 1 day
            0.89,     # High SC bioavailability
            0.1,
            5.0,
            2.4,      # Typical dose mg
            :weekly,
            3.0,      # Strong effect
            2.0,
            168.0,    # Week-long effect
            true, true, 0.4
        ),

        :tirzepatide => GLP1AgonistParams(
            "Tirzepatide",
            4813.5,   # Dual GIP/GLP-1
            120.0,    # t1/2 ~5 days
            24.0,
            0.80,
            0.05,     # Very high affinity
            8.0,      # Higher potency (dual action)
            15.0,     # Max dose
            :weekly,
            3.5,      # Very strong delay
            2.0,
            120.0,
            true, true, 0.5  # More nausea
        ),

        :liraglutide => GLP1AgonistParams(
            "Liraglutide",
            3751.2,
            13.0,     # t1/2 ~13 h
            10.0,
            0.55,
            1.0,
            3.0,
            1.8,      # Victoza dose
            :daily,
            2.0,
            0.5,
            18.0,
            true, true, 0.3
        ),

        :dulaglutide => GLP1AgonistParams(
            "Dulaglutide",
            63000.0,  # Fc fusion, large
            120.0,    # ~5 days
            48.0,
            0.65,
            2.0,
            2.5,
            1.5,
            :weekly,
            2.5,
            4.0,
            120.0,
            true, false, 0.25
        ),

        :exenatide_er => GLP1AgonistParams(
            "Exenatide ER",
            4186.6,
            96.0,     # Extended release
            48.0,
            0.75,
            5.0,
            1.5,
            2.0,
            :weekly,
            2.0,
            6.0,
            96.0,
            true, true, 0.35
        ),
    )

    if !haskey(presets, name)
        available = join(keys(presets), ", ")
        error("Unknown GLP-1 agonist: $name. Available: $available")
    end

    return presets[name]
end

"""
Bariatric surgery presets.
"""
function surgery_preset(
    surgery_type::SurgeryType,
    time_months::Float64 = 12.0
)::BariatricSurgery
    if surgery_type == SLEEVE_GASTRECTOMY
        return BariatricSurgery(
            SLEEVE_GASTRECTOMY,
            time_months,
            100.0,    # Reduced gastric volume
            3.5,      # Slightly higher pH
            false,    # Duodenum not bypassed
            0.0, 0.0, 0.0,
            0.7,      # Faster emptying
            1.0,      # Normal transit
            0.0,      # No bile delay
            1.5,      # Enhanced gut hormones
            0.1       # Low dumping risk
        )
    elseif surgery_type == ROUX_EN_Y_BYPASS
        return BariatricSurgery(
            ROUX_EN_Y_BYPASS,
            time_months,
            30.0,     # Very small pouch
            5.0,      # Higher pH
            true,     # Duodenum bypassed!
            50.0,     # Biliopancreatic limb
            100.0,    # Alimentary limb
            400.0,    # Common channel
            0.5,      # Very fast emptying
            0.8,      # Faster transit
            30.0,     # 30 min to bile mixing
            3.0,      # Strongly enhanced GLP-1
            0.4       # High dumping risk
        )
    elseif surgery_type == BILIOPANCREATIC_DIVERSION
        return BariatricSurgery(
            BILIOPANCREATIC_DIVERSION,
            time_months,
            200.0,
            4.0,
            true,
            200.0,
            200.0,
            100.0,    # Short common channel
            0.6,
            0.7,
            60.0,
            2.5,
            0.5
        )
    elseif surgery_type == GASTRIC_BANDING
        return BariatricSurgery(
            GASTRIC_BANDING,
            time_months,
            30.0,     # Small pouch above band
            2.5,      # Near normal pH
            false,
            0.0, 0.0, 0.0,
            0.3,      # Slow due to restriction
            1.0,
            0.0,
            1.2,
            0.05
        )
    else
        return BariatricSurgery(
            NO_SURGERY,
            0.0,
            250.0, 2.0,
            false,
            0.0, 0.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 0.0
        )
    end
end

"""
Create modified GI parameters combining GLP-1 and surgery effects.
"""
function create_modified_gi_params(;
    glp1::Union{GLP1AgonistParams, Nothing} = nothing,
    glp1_dose_mg::Float64 = 0.0,
    time_since_glp1_h::Float64 = 0.0,
    surgery::Union{BariatricSurgery, Nothing} = nothing
)::GIPhysiologyModified
    # Start with normal physiology
    base = GIPhysiologyModified(
        250.0, 50.0, 100.0, 80.0, 200.0,
        2.0, 6.0, 6.5, 7.2,
        15.0, 10.0, 90.0, 120.0, 720.0,
        2000.0, 18000.0, 12000.0, 1500.0,
        8.0, 6.0, 2.0,
        true, 0.0
    )

    # Apply GLP-1 effects
    if glp1 !== nothing && glp1_dose_mg > 0
        delay = gastric_emptying_delay(glp1, glp1_dose_mg, time_since_glp1_h)
        # Modify gastric emptying primarily
        base = GIPhysiologyModified(
            base.stomach_volume,
            base.duodenum_volume, base.jejunum_volume, base.ileum_volume, base.colon_volume,
            base.stomach_pH, base.duodenum_pH, base.jejunum_pH, base.ileum_pH,
            base.gastric_emptying_t50 * delay,  # Delayed!
            base.duodenum_transit * 1.2,
            base.jejunum_transit * 1.2,
            base.ileum_transit * 1.2,
            base.colon_transit,
            base.duodenum_sa, base.jejunum_sa, base.ileum_sa, base.colon_sa,
            base.duodenum_bile, base.jejunum_bile, base.ileum_bile,
            base.duodenum_functional,
            base.bile_delay_min
        )
    end

    # Apply surgery effects
    if surgery !== nothing && surgery.surgery_type != NO_SURGERY
        physiology = calculate_surgery_physiology(surgery)

        base = GIPhysiologyModified(
            surgery.gastric_volume_mL,
            surgery.duodenum_bypassed ? 0.0 : 50.0,
            100.0, 80.0, 200.0,
            physiology.gastric_pH,
            physiology.duodenal_pH,
            6.5, 7.2,
            base.gastric_emptying_t50 * surgery.gastric_emptying_rate,
            surgery.duodenum_bypassed ? 0.0 : 10.0,
            90.0 * surgery.intestinal_transit_factor,
            120.0 * surgery.intestinal_transit_factor,
            720.0,
            surgery.duodenum_bypassed ? 0.0 : 2000.0,
            18000.0 * physiology.effective_absorption_area,
            12000.0 * physiology.effective_absorption_area,
            1500.0,
            physiology.bile_concentration_factor * 8.0,
            physiology.bile_concentration_factor * 6.0,
            physiology.bile_concentration_factor * 2.0,
            !surgery.duodenum_bypassed,
            surgery.bile_mixing_delay_min
        )
    end

    return base
end

"""
    validate_glp1_model()

Validate model against literature benchmarks.
"""
function validate_glp1_model()
    results = Dict{String, Any}()

    # Test 1: Semaglutide gastric emptying delay
    sema = glp1_agonist_preset(:semaglutide_sc)
    delay_4h = gastric_emptying_delay(sema, 1.0, 4.0)
    delay_24h = gastric_emptying_delay(sema, 1.0, 24.0)
    delay_168h = gastric_emptying_delay(sema, 1.0, 168.0)

    results["semaglutide_delay"] = (
        at_4h = delay_4h,
        at_24h = delay_24h,
        at_168h = delay_168h,
        literature = "T50 increased 2-3x at steady state"
    )

    # Test 2: RYGB bioavailability changes
    surgery_rygb = surgery_preset(ROUX_EN_Y_BYPASS, 12.0)
    physiology_rygb = calculate_surgery_physiology(surgery_rygb)

    results["rygb_physiology"] = (
        gastric_pH = physiology_rygb.gastric_pH,
        effective_area = physiology_rygb.effective_absorption_area,
        bile_factor = physiology_rygb.bile_concentration_factor
    )

    # Test 3: Drug-specific changes
    metformin_change = bioavailability_change(:metformin, ROUX_EN_Y_BYPASS)
    levo_change = bioavailability_change(:levothyroxine, ROUX_EN_Y_BYPASS)

    results["drug_specific"] = (
        metformin = metformin_change,
        levothyroxine = levo_change
    )

    # Test 4: Combined GLP-1 + RYGB (common scenario)
    gi_combined = create_modified_gi_params(
        glp1 = sema,
        glp1_dose_mg = 1.0,
        time_since_glp1_h = 24.0,
        surgery = surgery_rygb
    )

    results["combined_effects"] = (
        gastric_emptying_t50 = gi_combined.gastric_emptying_t50,
        duodenum_functional = gi_combined.duodenum_functional,
        jejunum_transit = gi_combined.jejunum_transit
    )

    return results
end

end # module GLP1BariatricModel
