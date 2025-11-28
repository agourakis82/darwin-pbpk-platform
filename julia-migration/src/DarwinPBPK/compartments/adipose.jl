# ADIPOSE TISSUE COMPARTMENT MODEL
# ================================
#
# Adipose tissue is UNIQUE among all compartments:
# - 85% neutral lipids (triglycerides) - NOT phospholipids
# - Very low water content (1.7% intracellular)
# - Drug partitioning follows OLIVE OIL:WATER, not octanol:water
# - Highly variable between individuals (10-40kg in adults)
# - Perfusion-limited for most drugs (low blood flow)
#
# Key insight: Lipophilic drugs accumulate massively in adipose,
# but slowly (low perfusion). This creates a "deep compartment"
# in multi-compartment PK models.

module AdiposeCompartment

export AdiposeProperties, calculate_kp_adipose, calculate_adipose_contribution

"""
Adipose tissue physiological properties

Reference values for 70kg adult:
- Volume: 12-15 L (can be 5-50L depending on body composition)
- Blood flow: 0.03 L/min/kg tissue (very low!)
- Composition: 85% triglycerides, 1.7% water, 0.16% phospholipids
"""
struct AdiposeProperties
    volume_L::Float64           # Total adipose volume
    blood_flow_L_min::Float64   # Blood flow to adipose
    f_neutral_lipid::Float64    # Fraction neutral lipids (triglycerides)
    f_phospholipid::Float64     # Fraction phospholipids
    f_water_iw::Float64         # Fraction intracellular water
    f_water_ew::Float64         # Fraction extracellular water
end

# Default for 70kg adult with normal BMI
const DEFAULT_ADIPOSE = AdiposeProperties(
    12.0,    # volume (L) - highly variable!
    0.36,    # blood flow (L/min) = 0.03 × 12
    0.853,   # neutral lipids
    0.0016,  # phospholipids
    0.017,   # intracellular water
    0.135    # extracellular water
)

"""
Calculate adipose:plasma partition coefficient

CRITICAL: Adipose uses VEGETABLE OIL (olive oil):water partition,
NOT octanol:water like other tissues!

The key equation (Poulin & Theil):
  Kp_adipose = (D_vo:w × f_nl + 0.3×P×f_pl + 0.7×f_pl + f_w) × fup

Where:
  D_vo:w = vegetable oil:water distribution coefficient
  D_vo:w ≈ 10^(1.115 × logP - 1.35) for neutral compounds

For ionized compounds:
  D_vo:w = P_vo:w / (1 + ionization_factor)
"""
function calculate_kp_adipose(;
    logP::Float64,
    logD::Float64 = logP,  # Use logD if available (pH 7.4)
    fup::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    adipose::AdiposeProperties = DEFAULT_ADIPOSE
)
    # IMPORTANT: Use olive oil:water partition for adipose
    # Empirical correlation from Poulin & Theil
    logP_ow = 1.115 * logP - 1.35  # olive oil:water from octanol:water
    P_ow = 10^logP_ow

    # For comparison, octanol:water
    P = 10^logP
    D = 10^logD  # Distribution coefficient at pH 7.4

    # Handle ionization
    ionization_factor = 0.0
    if !isnothing(pKa)
        if is_base
            # For bases, ionization reduces lipid partitioning
            ionization_factor = 10^(pKa - 7.4)
        elseif is_acid
            ionization_factor = 10^(7.4 - pKa)
        end
    end

    # Effective partition coefficient for ionized drugs
    # Ionized forms don't partition well into lipids
    D_ow = P_ow / (1 + ionization_factor)

    # Adipose Kp calculation
    # Key insight: Neutral lipids dominate (85%!), phospholipids are negligible
    f_nl = adipose.f_neutral_lipid
    f_pl = adipose.f_phospholipid
    f_w = adipose.f_water_iw + adipose.f_water_ew

    # Partition into each component
    # Neutral lipids: D_ow (olive oil equivalent)
    # Phospholipids: 0.3P + 0.7 (Rodgers-Rowland)
    # Water: 1.0

    Kpu = D_ow * f_nl + (0.3 * D + 0.7) * f_pl + f_w
    Kp = Kpu * fup

    return max(Kp, 0.01)
end

"""
Calculate adipose contribution to Vdss

Returns: (Kp, contribution_L)
where contribution_L = Kp × Volume
"""
function calculate_adipose_contribution(;
    logP::Float64,
    logD::Float64 = logP,
    fup::Float64,
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false,
    is_acid::Bool = false,
    adipose_volume::Float64 = 12.0  # L, highly variable!
)
    adipose = AdiposeProperties(
        adipose_volume,
        0.03 * adipose_volume,  # blood flow scales with volume
        0.853, 0.0016, 0.017, 0.135
    )

    Kp = calculate_kp_adipose(
        logP=logP, logD=logD, fup=fup,
        pKa=pKa, is_base=is_base, is_acid=is_acid,
        adipose=adipose
    )

    contribution = Kp * adipose_volume

    return (Kp=Kp, contribution_L=contribution, volume=adipose_volume)
end

"""
Estimate fraction unbound in adipose tissue (fut_adipose)

This is NOT the same as fut in other tissues!

For adipose:
  fut_adipose ≈ 1 / (1 + D_ow × f_nl / f_w)

Highly lipophilic drugs are essentially "trapped" in fat.
"""
function calculate_fut_adipose(logP::Float64;
    pKa::Union{Float64, Nothing} = nothing,
    is_base::Bool = false)

    logP_ow = 1.115 * logP - 1.35
    P_ow = 10^logP_ow

    # Ionization adjustment
    if !isnothing(pKa) && is_base
        ionization = 10^(pKa - 7.4)
        P_ow = P_ow / (1 + ionization)
    end

    f_nl = 0.853
    f_w = 0.017 + 0.135  # Total water

    # Unbound fraction = water / (water + lipid_partitioning)
    fut = f_w / (f_w + P_ow * f_nl)

    return clamp(fut, 0.0001, 0.99)
end

"""
Special considerations for adipose tissue:

1. PERFUSION-LIMITED DISTRIBUTION
   - Blood flow is only 0.03 L/min/kg
   - Equilibration takes hours to days
   - Creates "deep compartment" in PK

2. VARIABILITY
   - Obese: 30-50 L adipose
   - Lean: 5-10 L adipose
   - This MASSIVELY affects Vdss for lipophilic drugs!

3. TEMPERATURE EFFECTS
   - Cold → vasoconstriction → even lower perfusion
   - Important for anesthetics

4. DRUG RELEASE
   - Lipophilic drugs slowly release from adipose
   - Causes prolonged terminal half-life
   - Examples: THC, amiodarone, chloroquine
"""

# Example drugs with extreme adipose accumulation:
const ADIPOSE_ACCUMULATORS = Dict(
    "amiodarone" => (logP=7.6, Kp_adipose=300.0),
    "chloroquine" => (logP=4.6, Kp_adipose=50.0),
    "thiopental" => (logP=2.9, Kp_adipose=10.0),
    "diazepam" => (logP=2.8, Kp_adipose=8.0),
    "THC" => (logP=6.4, Kp_adipose=100.0),
)

end # module
