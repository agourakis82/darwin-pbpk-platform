# ===========================================================================
# ML-MEDLANG INTEGRATION
# ===========================================================================
# Bridges ML transporter predictions with MedLang DSL for oral absorption.
#
# This module generates MedLang model definitions from:
# 1. ML-predicted transporter substrates
# 2. ML-estimated Km values
# 3. Derived first-pass metabolism parameters
#
# The generated MedLang can then use the full DSL infrastructure:
# - compile_model() for PBPKParams
# - simulate_oral() for concentration-time profiles
# - validate_model() for model checking
#
# Author: Dr. Sounio Agourakis
# Date: November 2025
# ===========================================================================

"""
ML-MedLang Integration for Transporter-Based Oral Absorption Modeling

Converts ML transporter predictions into MedLang DSL code that can be
compiled and simulated using the full MedLang infrastructure.
"""
module MLMedLangIntegration

using ..MedLang
using ..MedLang.MedLangParser: ROUTE_ORAL
using ..MedLang.MedLangTranspiler: OralAbsorptionParams

# ===========================================================================
# TRANSPORTER → ABSORPTION PARAMETER MAPPING
# ===========================================================================

"""
Transporter kinetics database for Ka estimation.

Each transporter has:
- vmax_factor: Relative expression in intestine (vs reference)
- ka_boost: Multiplication factor for Ka when this transporter is active
- regional_weight: Jejunum > Ileum > Duodenum weighting
"""
const TRANSPORTER_KA_EFFECTS = Dict{Symbol, NamedTuple}(
    # Uptake transporters (increase Ka)
    :PEPT1 => (vmax_factor = 1.0, ka_boost = 1.8, fg_effect = 0.0),
    :OCT1 => (vmax_factor = 0.3, ka_boost = 1.2, fg_effect = 0.0),
    :OCT3 => (vmax_factor = 0.2, ka_boost = 1.1, fg_effect = 0.0),
    :OATP2B1 => (vmax_factor = 0.6, ka_boost = 1.5, fg_effect = -0.15),  # Hepatic uptake reduces Fh
    :ENT1 => (vmax_factor = 0.4, ka_boost = 1.3, fg_effect = 0.0),
    :ENT2 => (vmax_factor = 0.3, ka_boost = 1.2, fg_effect = 0.0),
    :MCT1 => (vmax_factor = 0.5, ka_boost = 1.4, fg_effect = 0.0),
    :LAT1 => (vmax_factor = 0.3, ka_boost = 1.3, fg_effect = 0.0),
    :LAT2 => (vmax_factor = 0.4, ka_boost = 1.4, fg_effect = 0.0),
    :ASBT => (vmax_factor = 0.2, ka_boost = 1.2, fg_effect = 0.0),  # Ileum-specific

    # Efflux transporters (decrease Fg)
    :PGP => (vmax_factor = 1.0, ka_boost = 0.7, fg_effect = -0.25),
    :BCRP => (vmax_factor = 0.8, ka_boost = 0.8, fg_effect = -0.15),
    :MRP2 => (vmax_factor = 0.5, ka_boost = 0.9, fg_effect = -0.10),
)

"""
CYP enzyme expression in gut wall for Fg estimation.
"""
const GUT_WALL_CYP = Dict{Symbol, Float64}(
    :CYP3A4 => 0.80,   # Dominant gut CYP
    :CYP3A5 => 0.10,
    :CYP2C9 => 0.05,
    :CYP2C19 => 0.03,
    :CYP2D6 => 0.02,
)

# ===========================================================================
# ABSORPTION PARAMETER ESTIMATION
# ===========================================================================

"""
    estimate_ka_from_transporters(transporters::Vector{Symbol}, km_values::Dict) -> Float64

Estimate absorption rate constant from predicted transporter substrates.

Uses a multi-transporter model that accounts for:
- Transporter expression levels in intestine
- Substrate affinity (Km-based weighting)
- Regional expression patterns
"""
function estimate_ka_from_transporters(
    transporters::Vector{Symbol},
    km_values::Dict{Symbol, Float64};
    baseline_ka::Float64 = 1.0
)::Float64
    if isempty(transporters)
        return baseline_ka
    end

    ka_multiplier = 1.0

    for transporter in transporters
        if !haskey(TRANSPORTER_KA_EFFECTS, transporter)
            continue
        end

        effect = TRANSPORTER_KA_EFFECTS[transporter]
        km = get(km_values, transporter, 100.0)  # Default 100 μM

        # Higher affinity (lower Km) = stronger effect
        affinity_factor = 100.0 / (km + 100.0)  # Normalize to 100 μM

        # Combine vmax and affinity effects
        contribution = effect.ka_boost * effect.vmax_factor * affinity_factor

        # Multiplicative model
        ka_multiplier *= (1.0 + (contribution - 1.0) * 0.5)
    end

    return baseline_ka * ka_multiplier
end

"""
    estimate_fg_from_transporters(transporters::Vector{Symbol}; cyp_substrate::Symbol=:CYP3A4) -> Float64

Estimate gut availability (Fg) from transporters and CYP metabolism.

Fg = 1 - (gut wall extraction)

Accounts for:
- P-gp/BCRP efflux reducing absorption
- CYP3A4 gut wall metabolism
- OATP uptake into enterocytes
"""
function estimate_fg_from_transporters(
    transporters::Vector{Symbol};
    cyp_substrate::Symbol = :none,
    fu_gut::Float64 = 1.0  # Unbound fraction in gut
)::Float64
    base_fg = 1.0

    # Efflux transporter effects
    for transporter in transporters
        if haskey(TRANSPORTER_KA_EFFECTS, transporter)
            effect = TRANSPORTER_KA_EFFECTS[transporter]
            base_fg += effect.fg_effect
        end
    end

    # CYP metabolism in gut wall
    if cyp_substrate != :none && haskey(GUT_WALL_CYP, cyp_substrate)
        cyp_expression = GUT_WALL_CYP[cyp_substrate]
        # Simplified extraction: E_gut = fu * CLint_gut / (Q_gut + fu * CLint_gut)
        # For typical CYP3A4 substrate, assume CLint contributes to ~20-50% extraction
        cyp_extraction = cyp_expression * 0.3 * fu_gut
        base_fg *= (1.0 - cyp_extraction)
    end

    return clamp(base_fg, 0.1, 1.0)
end

"""
    estimate_fh_from_clearance(cl_hepatic::Float64; qh::Float64=90.0) -> Float64

Estimate hepatic availability from hepatic clearance.

Fh = 1 - ERh = 1 - (CLh / Qh)

Where:
- CLh = hepatic clearance (L/h)
- Qh = hepatic blood flow (~90 L/h for 70kg adult)
"""
function estimate_fh_from_clearance(
    cl_hepatic::Float64;
    qh::Float64 = 90.0,
    fu::Float64 = 1.0  # Unbound fraction in plasma
)::Float64
    # Well-stirred model
    er_h = fu * cl_hepatic / (qh + fu * cl_hepatic)
    return clamp(1.0 - er_h, 0.05, 1.0)
end

# ===========================================================================
# MEDLANG CODE GENERATION
# ===========================================================================

"""
    generate_medlang_absorption(ml_result::NamedTuple; drug_name::String="Drug") -> String

Generate MedLang absorption and firstpass blocks from ML predictions.

Returns a string that can be inserted into a MedLang model definition.
"""
function generate_medlang_absorption(
    ml_result::NamedTuple;
    drug_name::String = "Drug",
    cl_hepatic::Float64 = 10.0,
    cyp_substrate::Symbol = :none,
    lag_time::Float64 = 0.25
)::String
    # Estimate absorption parameters from ML predictions
    ka = estimate_ka_from_transporters(
        ml_result.uptake_transporters,
        ml_result.carrier_km_values
    )

    # Get all transporters (uptake + efflux)
    all_transporters = [ml_result.uptake_transporters...]
    if ml_result.is_pgp_substrate
        push!(all_transporters, :PGP)
    end

    # Estimate first-pass metabolism
    fg = estimate_fg_from_transporters(all_transporters; cyp_substrate=cyp_substrate)
    fh = estimate_fh_from_clearance(cl_hepatic)

    # Calculate effective F
    f_eff = fg * fh

    # Generate MedLang code
    medlang = """
    // ML-predicted oral absorption for $drug_name
    // Predicted transporters: $(join(ml_result.uptake_transporters, ", "))
    // P-gp substrate: $(ml_result.is_pgp_substrate)

    route: oral

    absorption {
        Ka: $(round(ka, digits=2)),     // ML-estimated from transporter kinetics
        F: 0.95,                         // Assume high fraction absorbed
        lag: $lag_time                   // Gastric emptying lag
    }

    firstpass {
        Fg: $(round(fg, digits=2)),     // Gut availability (transporter + CYP3A4)
        Fh: $(round(fh, digits=2))      // Hepatic availability
    }
"""
    return medlang
end

"""
    generate_full_medlang_model(smiles::String; kwargs...) -> String

Generate a complete MedLang model from SMILES using ML predictions.

This is the main entry point for ML-driven model generation.
"""
function generate_full_medlang_model(
    ml_result::NamedTuple;
    drug_name::String = "MLDrug",
    mw::Float64 = 400.0,
    cl_hepatic::Float64 = 10.0,
    cl_renal::Float64 = 0.1,
    vd::Float64 = 70.0,
    fu::Float64 = 0.1,
    cyp_substrate::Symbol = :CYP3A4,
    bbb_permeable::Bool = false
)::String
    # Generate absorption block
    absorption_block = generate_medlang_absorption(
        ml_result;
        drug_name = drug_name,
        cl_hepatic = cl_hepatic,
        cyp_substrate = cyp_substrate
    )

    # Estimate Kp values from physicochemical properties
    logp_estimate = (mw > 300 ? 2.5 : 1.5)  # Rough estimate
    kp_adipose = 10^(0.7 * logp_estimate - 0.5)
    kp_liver = 2.0 + fu * 1.5
    kp_brain = bbb_permeable ? 0.8 : 0.1

    # Build complete MedLang model
    model = """
model $(drug_name)_PBPK {
    // Generated from ML transporter predictions
    // MW: $mw Da
    // Predicted uptake transporters: $(join(ml_result.uptake_transporters, ", "))
    // P-gp substrate: $(ml_result.is_pgp_substrate) (ER=$(round(ml_result.pgp_efflux_ratio, digits=1)))

$absorption_block

    // Clearance mechanisms
    clearance hepatic: $(cl_hepatic)_L/h
    clearance renal: $(cl_renal)_L/h

    // 14-compartment PBPK model
    organ blood { V: 5.0_L, Q: 0.0_L/h, Kp: 1.0 }
    organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: $(round(kp_liver, digits=1)) }
    organ kidney { V: 0.31_L, Q: 60.0_L/h, Kp: 2.0 }
    organ brain { V: 1.4_L, Q: 50.0_L/h, Kp: $(round(kp_brain, digits=1)) }
    organ heart { V: 0.33_L, Q: 20.0_L/h, Kp: 2.5 }
    organ lung { V: 0.5_L, Q: 300.0_L/h, Kp: 1.8 }
    organ muscle { V: 30.0_L, Q: 75.0_L/h, Kp: 1.5 }
    organ adipose { V: 15.0_L, Q: 12.0_L/h, Kp: $(round(kp_adipose, digits=1)) }
    organ gut { V: 1.1_L, Q: 45.0_L/h, Kp: 2.0 }
    organ skin { V: 3.3_L, Q: 10.0_L/h, Kp: 1.2 }
    organ bone { V: 10.0_L, Q: 5.0_L/h, Kp: 0.5 }
    organ spleen { V: 0.18_L, Q: 15.0_L/h, Kp: 2.2 }
    organ pancreas { V: 0.1_L, Q: 5.0_L/h, Kp: 1.8 }
    organ other { V: 5.0_L, Q: 20.0_L/h, Kp: 1.5 }
}
"""
    return model
end

# ===========================================================================
# SIMULATION WITH ML PREDICTIONS
# ===========================================================================

"""
    simulate_ml_oral(smiles::String, dose::Float64; kwargs...) -> Dict

End-to-end simulation using ML transporter predictions.

1. Predict transporters from SMILES
2. Generate MedLang model
3. Compile and simulate using MedLang DSL

This uses the FULL MedLang infrastructure!
"""
function simulate_ml_oral(
    ml_result::NamedTuple,
    dose::Float64;
    drug_name::String = "MLDrug",
    cl_hepatic::Float64 = 10.0,
    cl_renal::Float64 = 0.1,
    cyp_substrate::Symbol = :CYP3A4,
    t_max::Float64 = 24.0,
    num_points::Int = 100
)
    # Generate MedLang model
    medlang_source = generate_full_medlang_model(
        ml_result;
        drug_name = drug_name,
        cl_hepatic = cl_hepatic,
        cl_renal = cl_renal,
        cyp_substrate = cyp_substrate
    )

    # Simulate using MedLang DSL
    results = MedLang.simulate_oral(
        medlang_source,
        dose;
        t_max = t_max,
        num_points = num_points
    )

    # Add ML metadata to results
    results["ml_predictions"] = ml_result
    results["medlang_source"] = medlang_source

    return results
end

# ===========================================================================
# TRANSPORTER ANNOTATION BLOCK
# ===========================================================================

"""
    generate_transporter_annotation(ml_result::NamedTuple) -> String

Generate MedLang comment block with detailed transporter annotations.

This can be added to any MedLang model for documentation.
"""
function generate_transporter_annotation(ml_result::NamedTuple)::String
    lines = String[]
    push!(lines, "// ==== ML Transporter Predictions ====")

    # Uptake transporters
    if !isempty(ml_result.uptake_transporters)
        push!(lines, "// Uptake transporters:")
        for t in ml_result.uptake_transporters
            km = get(ml_result.carrier_km_values, t, NaN)
            push!(lines, "//   - $t: Km = $(round(km, digits=0)) μM")
        end
    else
        push!(lines, "// No uptake transporters predicted (passive diffusion)")
    end

    # Efflux
    if ml_result.is_pgp_substrate
        push!(lines, "// Efflux:")
        push!(lines, "//   - P-gp substrate: ER = $(round(ml_result.pgp_efflux_ratio, digits=1))")
    end

    push!(lines, "// =====================================")

    return join(lines, "\n")
end

# ===========================================================================
# EXPORTS
# ===========================================================================

export estimate_ka_from_transporters, estimate_fg_from_transporters, estimate_fh_from_clearance
export generate_medlang_absorption, generate_full_medlang_model
export simulate_ml_oral, generate_transporter_annotation
export TRANSPORTER_KA_EFFECTS, GUT_WALL_CYP

end # module
