"""
GI Detailed Compartment Module - 7-Segment Gastrointestinal Tract Model

Implements PK-Sim standard 7-segment GI tract with physiological accuracy:
1. Stomach - pH-dependent dissolution, gastric emptying
2. Duodenum - High blood flow, bile/pancreatic secretions
3. Jejunum Upper - Largest surface area, active transport
4. Jejunum Lower - Continued absorption
5. Ileum Upper - Bile acid reabsorption
6. Ileum Lower - GALT (immune surveillance)
7. Colon - Microbiota metabolism, slow transit

Features:
- pH-dependent ionization (Henderson-Hasselbalch)
- Transporter expression (P-gp, BCRP, OATP, PEPT1)
- CYP3A4/A5 intestinal metabolism
- UGT1A1 glucuronidation
- Regional blood flow and permeability
- Fed vs. fasted state
- Cascade transit model

References:
- Willmann S et al. J Med Chem 2004;47:4022-4031 (PK-Sim)
- Amidon GL et al. Pharm Res 1995;12:413-420 (BCS)
- Yu LX et al. AAPS PharmSci 2002;4:E33 (BCS)
- Lennernäs H et al. J Pharm Sci 1997;86:403-410 (Regional permeability)
- DeSesso JM, Jacobson CF. Food Chem Toxicol 2001;39:209-228 (GI physiology)

Author: Dr. Sounio Agourakis + AI Assistant
Date: December 2025
"""

module GIDetailed

# Note: PatientProfile is not required for GI tract model
# GI model is self-contained with physiological parameters

export GISegment, GITract, DrugGIProperties
export create_gi_tract, calculate_gi_absorption, simulate_gi_transit
export calculate_ionization_fraction, calculate_permeability
export GI_SEGMENTS, SEGMENT_NAMES

# Constants
const SEGMENT_NAMES = [:stomach, :duodenum, :jejunum_upper, :jejunum_lower,
                       :ileum_upper, :ileum_lower, :colon]
const NUM_GI_SEGMENTS = length(SEGMENT_NAMES)

"""
GISegment - Individual gastrointestinal segment with physiological properties

Fields:
- name::Symbol - Segment identifier
- volume_fasting_mL::Float64 - Luminal volume in fasted state
- volume_fed_mL::Float64 - Luminal volume in fed state
- ph_fasting::Float64 - pH in fasted state
- ph_fed::Float64 - pH in fed state
- blood_flow_mL_min::Float64 - Regional blood flow
- surface_area_m2::Float64 - Absorptive surface area (includes villi/microvilli)
- transit_time_min::Float64 - Mean transit time through segment
- length_cm::Float64 - Anatomical length
- radius_cm::Float64 - Average radius
- permeability_cm_s::Float64 - Baseline effective permeability (Peff)
- transporter_expression::Dict{Symbol, Float64} - Transporter abundance (0-1)
- cyp_expression::Dict{Symbol, Float64} - CYP enzyme expression (0-1)
- ugt_expression::Dict{Symbol, Float64} - UGT enzyme expression (0-1)
- bile_salts::Bool - Presence of bile salts
- lipase_activity::Float64 - Digestive lipase activity (0-1)
- microbiota_density::Float64 - Bacterial density (CFU/mL, log10)

References:
- Willmann S et al. J Med Chem 2004 (PK-Sim physiological parameters)
- Lennernäs H et al. J Pharm Sci 1997 (Regional permeability)
- Tubic-Grozdanis M et al. Eur J Pharm Sci 2008;33:36-58 (Bile salts)
"""
mutable struct GISegment
    name::Symbol
    volume_fasting_mL::Float64
    volume_fed_mL::Float64
    ph_fasting::Float64
    ph_fed::Float64
    blood_flow_mL_min::Float64
    surface_area_m2::Float64
    transit_time_min::Float64
    length_cm::Float64
    radius_cm::Float64
    permeability_cm_s::Float64
    transporter_expression::Dict{Symbol, Float64}
    cyp_expression::Dict{Symbol, Float64}
    ugt_expression::Dict{Symbol, Float64}
    bile_salts::Bool
    lipase_activity::Float64
    microbiota_density::Float64
end

"""
GITract - Complete 7-segment gastrointestinal tract model

Fields:
- segments::Vector{GISegment} - All 7 segments in order
- fed_state::Bool - Current nutritional state
- total_surface_area_m2::Float64 - Sum of all segment surface areas
"""
mutable struct GITract
    segments::Vector{GISegment}
    fed_state::Bool
    total_surface_area_m2::Float64
end

"""
DrugGIProperties - Drug-specific gastrointestinal properties

Fields:
- pka_acid::Union{Float64, Nothing} - Acidic pKa (if applicable)
- pka_base::Union{Float64, Nothing} - Basic pKa (if applicable)
- log_p::Float64 - Octanol-water partition coefficient
- molecular_weight::Float64 - Molecular weight (Da)
- hbd::Int - Hydrogen bond donors
- hba::Int - Hydrogen bond acceptors
- psa::Float64 - Polar surface area (Ų)
- fu_gut::Float64 - Fraction unbound in gut (0-1)
- solubility_mg_mL::Float64 - Aqueous solubility at pH 6.5
- dissolution_rate::Float64 - Dissolution rate constant (1/min)
- pgp_substrate::Bool - P-glycoprotein substrate
- bcrp_substrate::Bool - BCRP substrate
- oatp_substrate::Bool - OATP substrate
- pept1_substrate::Bool - PEPT1 substrate
- cyp3a4_substrate::Bool - CYP3A4 substrate
- ugt1a1_substrate::Bool - UGT1A1 substrate
"""
struct DrugGIProperties
    pka_acid::Union{Float64, Nothing}
    pka_base::Union{Float64, Nothing}
    log_p::Float64
    molecular_weight::Float64
    hbd::Int
    hba::Int
    psa::Float64
    fu_gut::Float64
    solubility_mg_mL::Float64
    dissolution_rate::Float64
    pgp_substrate::Bool
    bcrp_substrate::Bool
    oatp_substrate::Bool
    pept1_substrate::Bool
    cyp3a4_substrate::Bool
    ugt1a1_substrate::Bool
end

"""
Create default GI tract with PK-Sim physiological parameters.

References:
- Willmann S et al. J Med Chem 2004;47:4022-4031
- PK-Sim v11 documentation (Bayer Technology Services)
- DeSesso JM, Jacobson CF. Food Chem Toxicol 2001;39:209-228

Returns:
    GITract with all 7 segments initialized
"""
function create_gi_tract(;fed_state::Bool=false)::GITract

    # 1. STOMACH
    # - Acidic environment (pH 1.5-3.5 fasted, 4.5-5.5 fed)
    # - Gastric emptying controlled by pylorus
    # - Limited absorption except lipophilic weak bases
    # - Fed state: increased volume, higher pH, delayed emptying
    stomach = GISegment(
        :stomach,
        50.0,      # volume_fasting_mL (minimal residual volume)
        500.0,     # volume_fed_mL (after standard meal)
        2.0,       # ph_fasting (highly acidic)
        5.0,       # ph_fed (buffered by food)
        100.0,     # blood_flow_mL_min (moderate)
        0.1,       # surface_area_m2 (limited absorptive area)
        30.0,      # transit_time_min (fasted) - fed: 60-120 min
        20.0,      # length_cm
        5.0,       # radius_cm
        1e-6,      # permeability_cm_s (low, thick mucus)
        Dict(:pgp => 0.3, :bcrp => 0.2, :oatp => 0.1, :pept1 => 0.2),
        Dict(:cyp3a4 => 0.1, :cyp3a5 => 0.05),
        Dict(:ugt1a1 => 0.05),
        false,     # bile_salts (absent)
        0.3,       # lipase_activity (gastric lipase present)
        3.0        # microbiota_density (log10 CFU/mL, low)
    )

    # 2. DUODENUM
    # - pH 6.0 (buffered by bicarbonate)
    # - High blood flow (portal vein drainage)
    # - Bile and pancreatic secretions
    # - Brunner's glands secrete alkaline mucus
    duodenum = GISegment(
        :duodenum,
        50.0,      # volume_fasting_mL
        75.0,      # volume_fed_mL
        6.0,       # ph_fasting
        6.2,       # ph_fed
        600.0,     # blood_flow_mL_min (highest in small intestine)
        0.2,       # surface_area_m2
        15.0,      # transit_time_min (rapid transit)
        25.0,      # length_cm (~10 inches)
        2.0,       # radius_cm
        2e-4,      # permeability_cm_s (high)
        Dict(:pgp => 0.5, :bcrp => 0.4, :oatp => 0.6, :pept1 => 0.7),
        Dict(:cyp3a4 => 0.7, :cyp3a5 => 0.4),  # High metabolic activity
        Dict(:ugt1a1 => 0.5),
        true,      # bile_salts (ampulla of Vater)
        1.0,       # lipase_activity (pancreatic lipase peak)
        5.0        # microbiota_density (increasing)
    )

    # 3. JEJUNUM UPPER
    # - pH 6.5-7.0
    # - Largest surface area (long villi)
    # - Major site of nutrient/drug absorption
    # - High expression of transporters
    jejunum_upper = GISegment(
        :jejunum_upper,
        100.0,     # volume_fasting_mL
        150.0,     # volume_fed_mL
        6.5,       # ph_fasting
        6.8,       # ph_fed
        500.0,     # blood_flow_mL_min
        50.0,      # surface_area_m2 (HUGE due to villi/microvilli)
        30.0,      # transit_time_min
        100.0,     # length_cm (first half of jejunum)
        1.5,       # radius_cm
        3e-4,      # permeability_cm_s (highest permeability)
        Dict(:pgp => 0.8, :bcrp => 0.6, :oatp => 0.9, :pept1 => 1.0),  # Peak transporter expression
        Dict(:cyp3a4 => 0.9, :cyp3a5 => 0.6),
        Dict(:ugt1a1 => 0.7),
        true,      # bile_salts (still present)
        0.8,       # lipase_activity
        6.0        # microbiota_density
    )

    # 4. JEJUNUM LOWER
    # - pH 7.0
    # - Continued high absorption
    # - Gradually decreasing surface area
    jejunum_lower = GISegment(
        :jejunum_lower,
        100.0,     # volume_fasting_mL
        150.0,     # volume_fed_mL
        7.0,       # ph_fasting
        7.0,       # ph_fed
        400.0,     # blood_flow_mL_min
        40.0,      # surface_area_m2
        30.0,      # transit_time_min
        100.0,     # length_cm (second half of jejunum)
        1.5,       # radius_cm
        2.5e-4,    # permeability_cm_s
        Dict(:pgp => 0.7, :bcrp => 0.5, :oatp => 0.8, :pept1 => 0.9),
        Dict(:cyp3a4 => 0.8, :cyp3a5 => 0.5),
        Dict(:ugt1a1 => 0.6),
        true,      # bile_salts
        0.6,       # lipase_activity
        7.0        # microbiota_density
    )

    # 5. ILEUM UPPER
    # - pH 7.0-7.5
    # - Bile acid reabsorption (enterohepatic circulation)
    # - Vitamin B12 absorption (intrinsic factor)
    # - Peyer's patches (immune surveillance)
    ileum_upper = GISegment(
        :ileum_upper,
        100.0,     # volume_fasting_mL
        150.0,     # volume_fed_mL
        7.2,       # ph_fasting
        7.2,       # ph_fed
        300.0,     # blood_flow_mL_min
        30.0,      # surface_area_m2 (shorter villi)
        40.0,      # transit_time_min
        100.0,     # length_cm
        1.5,       # radius_cm
        2e-4,      # permeability_cm_s
        Dict(:pgp => 0.6, :bcrp => 0.4, :oatp => 0.7, :pept1 => 0.7),
        Dict(:cyp3a4 => 0.6, :cyp3a5 => 0.3),
        Dict(:ugt1a1 => 0.4),
        true,      # bile_salts (reabsorption site)
        0.4,       # lipase_activity
        8.0        # microbiota_density (increasing)
    )

    # 6. ILEUM LOWER
    # - pH 7.5
    # - Terminal ileum (ileocecal valve)
    # - Reduced absorption
    # - Transition to colon microbiota
    ileum_lower = GISegment(
        :ileum_lower,
        100.0,     # volume_fasting_mL
        150.0,     # volume_fed_mL
        7.5,       # ph_fasting
        7.5,       # ph_fed
        250.0,     # blood_flow_mL_min
        20.0,      # surface_area_m2
        40.0,      # transit_time_min
        100.0,     # length_cm
        1.5,       # radius_cm
        1.5e-4,    # permeability_cm_s
        Dict(:pgp => 0.5, :bcrp => 0.3, :oatp => 0.5, :pept1 => 0.5),
        Dict(:cyp3a4 => 0.4, :cyp3a5 => 0.2),
        Dict(:ugt1a1 => 0.3),
        false,     # bile_salts (mostly reabsorbed)
        0.2,       # lipase_activity
        9.0        # microbiota_density (high)
    )

    # 7. COLON
    # - pH 6.5-7.5 (slightly acidic due to SCFA production)
    # - Water/electrolyte reabsorption
    # - Microbiota metabolism (reduction, hydrolysis)
    # - Slow transit (12-24h total)
    # - Limited drug absorption (lipophilic drugs only)
    colon = GISegment(
        :colon,
        200.0,     # volume_fasting_mL
        300.0,     # volume_fed_mL
        6.8,       # ph_fasting (SCFA production)
        6.8,       # ph_fed
        150.0,     # blood_flow_mL_min (lower than small intestine)
        2.0,       # surface_area_m2 (no villi)
        720.0,     # transit_time_min (12h average)
        150.0,     # length_cm
        3.0,       # radius_cm (wider lumen)
        1e-5,      # permeability_cm_s (very low)
        Dict(:pgp => 0.4, :bcrp => 0.3, :oatp => 0.2, :pept1 => 0.1),
        Dict(:cyp3a4 => 0.1, :cyp3a5 => 0.05),
        Dict(:ugt1a1 => 0.1),
        false,     # bile_salts (absent)
        0.0,       # lipase_activity (negligible)
        11.0       # microbiota_density (log10 CFU/mL, very high 10^11)
    )

    segments = [stomach, duodenum, jejunum_upper, jejunum_lower,
                ileum_upper, ileum_lower, colon]

    total_sa = sum(s.surface_area_m2 for s in segments)

    return GITract(segments, fed_state, total_sa)
end

"""
Calculate ionization fraction using Henderson-Hasselbalch equation.

For acids:  f_ionized = 1 / (1 + 10^(pKa - pH))
For bases:  f_ionized = 1 / (1 + 10^(pH - pKa))

Unionized fraction determines membrane permeability (pH-partition hypothesis).

Args:
    ph::Float64 - Local pH
    pka_acid::Union{Float64, Nothing} - Acidic pKa
    pka_base::Union{Float64, Nothing} - Basic pKa

Returns:
    Tuple{Float64, Float64} - (f_unionized, f_ionized)

References:
- Shore PA et al. J Pharmacol Exp Ther 1957;119:361-369 (pH partition)
- Avdeef A. Curr Top Med Chem 2001;1:277-351 (PAMPA)
"""
function calculate_ionization_fraction(
    ph::Float64,
    pka_acid::Union{Float64, Nothing},
    pka_base::Union{Float64, Nothing}
)::Tuple{Float64, Float64}

    f_unionized = 1.0

    # For acids: HA ⇌ H+ + A-
    if pka_acid !== nothing
        # f_unionized_acid = 1 / (1 + 10^(pH - pKa))
        f_unionized *= 1.0 / (1.0 + 10.0^(ph - pka_acid))
    end

    # For bases: BH+ ⇌ B + H+
    if pka_base !== nothing
        # f_unionized_base = 1 / (1 + 10^(pKa - pH))
        f_unionized *= 1.0 / (1.0 + 10.0^(pka_base - ph))
    end

    f_ionized = 1.0 - f_unionized

    return (f_unionized, f_ionized)
end

"""
Calculate effective permeability coefficient (Peff) with pH, transporter, and metabolism corrections.

Peff = Peff_baseline × f_unionized × (1 + Σ transporter_contributions) × (1 - E_gut)

Where:
- f_unionized: Henderson-Hasselbalch pH correction
- transporter_contributions: P-gp (efflux, negative), OATP/PEPT1 (influx, positive)
- E_gut: Intestinal extraction ratio (CYP3A4, UGT1A1)

Args:
    segment::GISegment - GI segment
    drug::DrugGIProperties - Drug properties
    ph::Float64 - Actual pH in segment

Returns:
    Float64 - Effective permeability (cm/s)

References:
- Lennernäs H. J Pharm Sci 1997;86:403-410
- Amidon GL et al. Pharm Res 1988;5:651-654
- Tachibana T et al. J Pharm Sci 2010;99:2493-2501
"""
function calculate_permeability(
    segment::GISegment,
    drug::DrugGIProperties,
    ph::Float64
)::Float64

    # Base permeability
    peff = segment.permeability_cm_s

    # pH-dependent ionization (only unionized crosses membrane)
    f_unionized, _ = calculate_ionization_fraction(ph, drug.pka_acid, drug.pka_base)
    peff *= f_unionized

    # Transporter effects
    transporter_factor = 1.0

    # P-gp efflux (reduces net absorption)
    if drug.pgp_substrate
        pgp_expr = get(segment.transporter_expression, :pgp, 0.0)
        transporter_factor *= (1.0 - 0.5 * pgp_expr)  # Max 50% reduction
    end

    # BCRP efflux
    if drug.bcrp_substrate
        bcrp_expr = get(segment.transporter_expression, :bcrp, 0.0)
        transporter_factor *= (1.0 - 0.3 * bcrp_expr)  # Max 30% reduction
    end

    # OATP influx (enhances absorption)
    if drug.oatp_substrate
        oatp_expr = get(segment.transporter_expression, :oatp, 0.0)
        transporter_factor *= (1.0 + 0.5 * oatp_expr)  # Max 50% enhancement
    end

    # PEPT1 influx (peptide/peptidomimetic drugs)
    if drug.pept1_substrate
        pept1_expr = get(segment.transporter_expression, :pept1, 0.0)
        transporter_factor *= (1.0 + 0.7 * pept1_expr)  # Max 70% enhancement
    end

    peff *= transporter_factor

    # Intestinal metabolism (CYP3A4, UGT1A1)
    # E_gut = CL_int / (Q + CL_int)  (well-stirred model)
    # Simplified: E_gut ≈ expression level × substrate susceptibility
    e_gut = 0.0

    if drug.cyp3a4_substrate
        cyp3a4_expr = get(segment.cyp_expression, :cyp3a4, 0.0)
        e_gut += 0.3 * cyp3a4_expr  # Max 30% extraction
    end

    if drug.ugt1a1_substrate
        ugt1a1_expr = get(segment.ugt_expression, :ugt1a1, 0.0)
        e_gut += 0.2 * ugt1a1_expr  # Max 20% extraction
    end

    e_gut = min(e_gut, 0.8)  # Cap at 80% extraction

    # Apply first-pass effect
    fg = 1.0 - e_gut  # Fraction escaping gut wall metabolism
    peff *= fg

    return max(peff, 1e-10)  # Numerical stability
end

"""
Calculate absorption rate from a GI segment.

Rate = Peff × SA × C_lumen × (1 - R_efflux)

Where:
- Peff: Effective permeability (pH, transporter, metabolism corrected)
- SA: Surface area (m²)
- C_lumen: Concentration in lumen (mg/mL)
- R_efflux: Efflux ratio (for P-gp substrates)

Args:
    segment::GISegment - GI segment
    drug::DrugGIProperties - Drug properties
    amount_mg::Float64 - Amount in segment lumen (mg)
    fed_state::Bool - Nutritional state

Returns:
    Float64 - Absorption rate (mg/min)

References:
- Amidon GL et al. Pharm Res 1995;12:413-420
- Yu LX et al. AAPS PharmSci 2002;4:E33
"""
function calculate_gi_absorption(
    segment::GISegment,
    drug::DrugGIProperties,
    amount_mg::Float64,
    fed_state::Bool
)::Float64

    # Get volume and pH based on fed state
    volume_mL = fed_state ? segment.volume_fed_mL : segment.volume_fasting_mL
    ph = fed_state ? segment.ph_fed : segment.ph_fasting

    # Concentration in lumen
    c_lumen = amount_mg / volume_mL  # mg/mL

    # Effective permeability
    peff = calculate_permeability(segment, drug, ph)  # cm/s

    # Surface area
    sa = segment.surface_area_m2 * 1e4  # Convert m² to cm²

    # Absorption rate (cm/s × cm² × mg/cm³ = mg/s)
    # Convert mg/mL to mg/cm³ (already same unit)
    rate_mg_per_s = peff * sa * c_lumen

    # Convert to mg/min
    rate_mg_per_min = rate_mg_per_s * 60.0

    # Bile salt enhancement (for lipophilic drugs in duodenum/upper jejunum)
    if segment.bile_salts && drug.log_p > 2.0
        bile_enhancement = 1.0 + 0.3 * segment.lipase_activity  # Up to 30% enhancement
        rate_mg_per_min *= bile_enhancement
    end

    return max(rate_mg_per_min, 0.0)
end

"""
Simulate gastrointestinal transit using cascade model.

Models sequential transit through 7 GI segments:
- First-order transit between segments
- Absorption from each segment
- Dissolution (if needed)
- First-pass metabolism

Args:
    gi_tract::GITract - GI tract model
    drug::DrugGIProperties - Drug properties
    dose_mg::Float64 - Oral dose
    time_points::Vector{Float64} - Time points (min)

Returns:
    Dict with:
        - "time": Time points (min)
        - "stomach", "duodenum", etc.: Amount in each segment (mg)
        - "absorbed_total": Cumulative absorbed (mg)
        - "absorption_rate": Rate at each timepoint (mg/min)

References:
- Yu LX, Amidon GL. Int J Pharm 1999;186:119-125 (Compartmental absorption)
- Grass GM. Adv Drug Deliv Rev 1997;23:199-219 (Simulation models)
"""
function simulate_gi_transit(
    gi_tract::GITract,
    drug::DrugGIProperties,
    dose_mg::Float64,
    time_points::Vector{Float64}
)::Dict{String, Vector{Float64}}

    n_times = length(time_points)
    dt = time_points[2] - time_points[1]  # Assume uniform spacing

    # Initialize amounts in each segment
    amounts = zeros(Float64, NUM_GI_SEGMENTS, n_times)
    amounts[1, 1] = dose_mg  # All dose starts in stomach

    absorbed_total = zeros(Float64, n_times)
    absorption_rate = zeros(Float64, n_times)

    # Time evolution
    for t_idx in 2:n_times
        for seg_idx in 1:NUM_GI_SEGMENTS
            segment = gi_tract.segments[seg_idx]

            # Current amount
            amount = amounts[seg_idx, t_idx-1]

            # Absorption from this segment
            abs_rate = calculate_gi_absorption(segment, drug, amount, gi_tract.fed_state)
            absorbed = abs_rate * dt
            absorbed = min(absorbed, amount)  # Can't absorb more than present

            # Transit to next segment
            ktr = 1.0 / segment.transit_time_min  # First-order transit rate
            transited = ktr * amount * dt
            transited = min(transited, amount - absorbed)

            # Update amount in current segment
            amounts[seg_idx, t_idx] = amount - absorbed - transited

            # Transfer to next segment (if not colon)
            if seg_idx < NUM_GI_SEGMENTS
                amounts[seg_idx+1, t_idx] += transited
            end

            # Accumulate absorbed amount
            absorption_rate[t_idx] += abs_rate
            absorbed_total[t_idx] = absorbed_total[t_idx-1] + absorbed
        end
    end

    # Build results dictionary
    results = Dict{String, Vector{Float64}}()
    results["time"] = time_points

    for (idx, name) in enumerate(SEGMENT_NAMES)
        results[String(name)] = amounts[idx, :]
    end

    results["absorbed_total"] = absorbed_total
    results["absorption_rate"] = absorption_rate
    results["bioavailability"] = absorbed_total[end] / dose_mg

    return results
end

"""
Calculate Biopharmaceutics Classification System (BCS) class.

BCS Class:
- Class I: High solubility, High permeability (>90% absorbed, dissolution not rate-limiting)
- Class II: Low solubility, High permeability (dissolution rate-limiting)
- Class III: High solubility, Low permeability (permeability rate-limiting)
- Class IV: Low solubility, Low permeability (poor bioavailability)

Criteria (FDA 2015):
- High solubility: Dose soluble in ≤250 mL over pH 1-7.5
- High permeability: ≥90% absorbed or Peff > 1.5×10⁻⁴ cm/s

Args:
    drug::DrugGIProperties - Drug properties
    dose_mg::Float64 - Dose strength

Returns:
    Symbol - :class_i, :class_ii, :class_iii, or :class_iv

References:
- Amidon GL et al. Pharm Res 1995;12:413-420
- FDA Guidance 2015: Waiver of In Vivo Bioavailability and Bioequivalence Studies
- Wu CY, Benet LZ. Pharm Res 2005;22:11-23
"""
function calculate_bcs_class(drug::DrugGIProperties, dose_mg::Float64)::Symbol
    # High solubility: dose soluble in 250 mL
    dose_solubility = drug.solubility_mg_mL * 250.0  # mg
    high_solubility = dose_mg <= dose_solubility

    # High permeability: Peff > 1.5e-4 cm/s (reference: metoprolol)
    # Estimate Peff from log P (crude but useful)
    estimated_peff = 1e-6 * 10^(0.5 * drug.log_p)  # Empirical correlation
    high_permeability = estimated_peff > 1.5e-4

    # BCS classification
    if high_solubility && high_permeability
        return :class_i
    elseif !high_solubility && high_permeability
        return :class_ii
    elseif high_solubility && !high_permeability
        return :class_iii
    else
        return :class_iv
    end
end

"""
Estimate intestinal first-pass extraction ratio (Fg).

Fg = 1 / (1 + (CL_int_gut × F_u_gut) / Q_gut)

Where:
- CL_int_gut: Intrinsic metabolic clearance in gut wall
- F_u_gut: Fraction unbound in gut
- Q_gut: Intestinal blood flow

Args:
    gi_tract::GITract - GI tract model
    drug::DrugGIProperties - Drug properties

Returns:
    Float64 - Fraction escaping gut metabolism (0-1)

References:
- Paine MF et al. J Pharmacol Exp Ther 1996;279:166-171 (CYP3A4 in gut)
- Benet LZ, Cummins CL. Adv Drug Deliv Rev 2001;50:S3-S11
"""
function calculate_fg(gi_tract::GITract, drug::DrugGIProperties)::Float64
    # Total gut blood flow (mL/min)
    q_gut = sum(s.blood_flow_mL_min for s in gi_tract.segments[2:6])  # Small intestine

    # Estimate intrinsic clearance (very drug-dependent, placeholder)
    cl_int_gut = 0.0  # mL/min

    if drug.cyp3a4_substrate
        # High CYP3A4 substrate: estimate from expression
        cyp3a4_expr = mean(s.cyp_expression[:cyp3a4] for s in gi_tract.segments[2:6])
        cl_int_gut += 500.0 * cyp3a4_expr  # Placeholder: max 500 mL/min
    end

    if drug.ugt1a1_substrate
        ugt1a1_expr = mean(s.ugt_expression[:ugt1a1] for s in gi_tract.segments[2:6])
        cl_int_gut += 200.0 * ugt1a1_expr
    end

    # Well-stirred model
    if cl_int_gut > 0.0
        fg = q_gut / (q_gut + cl_int_gut * drug.fu_gut)
    else
        fg = 1.0  # No gut metabolism
    end

    return clamp(fg, 0.01, 1.0)
end

"""
Create example drug properties for common drugs.
"""
function example_drug_metoprolol()::DrugGIProperties
    # Metoprolol: BCS Class I, high permeability reference
    DrugGIProperties(
        nothing,        # pka_acid
        9.7,            # pka_base (weak base)
        1.88,           # log_p
        267.4,          # molecular_weight
        2,              # hbd
        4,              # hba
        50.7,           # psa
        0.9,            # fu_gut
        16.9,           # solubility_mg_mL (highly soluble)
        0.5,            # dissolution_rate
        false,          # pgp_substrate
        false,          # bcrp_substrate
        false,          # oatp_substrate
        false,          # pept1_substrate
        true,           # cyp3a4_substrate (minor, CYP2D6 major)
        false           # ugt1a1_substrate
    )
end

function example_drug_atorvastatin()::DrugGIProperties
    # Atorvastatin: BCS Class II, low solubility, high permeability
    DrugGIProperties(
        4.33,           # pka_acid (carboxylic acid)
        nothing,        # pka_base
        5.7,            # log_p (lipophilic)
        558.6,          # molecular_weight
        3,              # hbd
        7,              # hba
        111.8,          # psa
        0.02,           # fu_gut (highly protein bound)
        0.04,           # solubility_mg_mL (poor aqueous solubility)
        0.1,            # dissolution_rate (slow)
        true,           # pgp_substrate
        true,           # bcrp_substrate
        true,           # oatp_substrate (OATP1B1)
        false,          # pept1_substrate
        true,           # cyp3a4_substrate
        true            # ugt1a1_substrate
    )
end

function example_drug_lisinopril()::DrugGIProperties
    # Lisinopril: BCS Class III, high solubility, low permeability
    DrugGIProperties(
        2.5,            # pka_acid (carboxylic acid)
        6.7,            # pka_base (lysine NH3+)
        -1.8,           # log_p (hydrophilic)
        405.5,          # molecular_weight
        4,              # hbd
        7,              # hba
        146.6,          # psa (high)
        1.0,            # fu_gut (no protein binding)
        25.0,           # solubility_mg_mL (high solubility)
        1.0,            # dissolution_rate (rapid)
        false,          # pgp_substrate
        false,          # bcrp_substrate
        false,          # oatp_substrate
        true,           # pept1_substrate (peptidomimetic)
        false,          # cyp3a4_substrate (not metabolized)
        false           # ugt1a1_substrate
    )
end

export create_gi_tract, calculate_ionization_fraction, calculate_permeability
export calculate_gi_absorption, simulate_gi_transit, calculate_bcs_class, calculate_fg
export example_drug_metoprolol, example_drug_atorvastatin, example_drug_lisinopril
export GI_SEGMENTS

end # module GIDetailed
