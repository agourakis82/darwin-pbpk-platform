"""
MedLang DSL Oral Absorption Validation Script

Enhanced validation with oral absorption and first-pass metabolism.
Tests against 572 drugs from ULTIMATE_DATASET using:
- Oral absorption (Ka)
- First-pass metabolism (Fg, Fh)
- Bioavailability corrections

Author: Dr. Sounio Agourakis
Date: November 2025
"""

using Printf
using Statistics
using Dates

# Include DarwinPBPK
include(joinpath(@__DIR__, "..", "..", "src", "DarwinPBPK.jl"))
using .DarwinPBPK
using .DarwinPBPK.MedLang
using .DarwinPBPK.ODEPBPKSolver

# JSON parsing (simple implementation to avoid dependency)
function parse_json_file(filepath::String)
    content = read(filepath, String)
    return parse_json(content)
end

function parse_json(s::String)
    s = strip(s)
    if startswith(s, '[')
        return parse_json_array(s)
    elseif startswith(s, '{')
        return parse_json_object(s)
    elseif startswith(s, '"')
        return parse_json_string(s)
    elseif s == "null"
        return nothing
    elseif s == "true"
        return true
    elseif s == "false"
        return false
    else
        # Number
        return occursin('.', s) ? parse(Float64, s) : parse(Int, s)
    end
end

function parse_json_string(s::String)
    m = match(r"^\"([^\"]*)\"", s)
    return m !== nothing ? m.captures[1] : ""
end

function parse_json_array(s::String)
    result = Any[]
    s = strip(s[2:end])
    depth = 0
    current = ""

    for c in s
        if c == '[' || c == '{'
            depth += 1
            current *= c
        elseif c == ']' || c == '}'
            if depth == 0 && c == ']'
                if !isempty(strip(current))
                    push!(result, parse_json(strip(current)))
                end
                break
            end
            depth -= 1
            current *= c
        elseif c == ',' && depth == 0
            if !isempty(strip(current))
                push!(result, parse_json(strip(current)))
            end
            current = ""
        else
            current *= c
        end
    end

    return result
end

function parse_json_object(s::String)
    result = Dict{String, Any}()
    s = strip(s[2:end])
    depth = 0
    current = ""
    key = ""
    in_key = true

    for c in s
        if c == '[' || c == '{'
            depth += 1
            current *= c
        elseif c == ']' || c == '}'
            if depth == 0 && c == '}'
                if !isempty(key) && !isempty(strip(current))
                    result[key] = parse_json(strip(current))
                end
                break
            end
            depth -= 1
            current *= c
        elseif c == ':' && depth == 0 && in_key
            key = strip(current)
            if startswith(key, '"') && endswith(key, '"')
                key = key[2:end-1]
            end
            current = ""
            in_key = false
        elseif c == ',' && depth == 0
            if !isempty(key) && !isempty(strip(current))
                result[key] = parse_json(strip(current))
            end
            current = ""
            key = ""
            in_key = true
        else
            current *= c
        end
    end

    return result
end

function load_dataset(filepath::String)
    try
        @eval using JSON
        return JSON.parsefile(filepath)
    catch
        return parse_json_file(filepath)
    end
end

"""
Generate MedLang model definition with oral absorption from drug PK parameters.
"""
function generate_oral_medlang_model(drug)::String
    name = get(drug, "drug_name", "Unknown")
    safe_name = replace(name, r"[^a-zA-Z0-9_]" => "_")

    # Extract parameters with defaults
    cl = get(drug, "CL_lit", 10.0)
    vd = get(drug, "Vd_lit", 50.0)
    f = get(drug, "bioavailability", 0.8)
    fu = get(drug, "fu", 0.5)
    half_life = get(drug, "half_life", 4.0)

    # Handle missing/null values
    cl = cl === nothing ? 10.0 : Float64(cl)
    vd = vd === nothing ? 50.0 : Float64(vd)
    f = f === nothing ? 0.8 : Float64(f)
    fu = fu === nothing ? 0.5 : Float64(fu)
    half_life = half_life === nothing ? 4.0 : Float64(half_life)

    # Ensure positive values
    cl = max(cl, 0.1)
    vd = max(vd, 1.0)
    f = clamp(f, 0.01, 1.0)
    fu = clamp(fu, 0.001, 1.0)

    # Calculate Ka from Tmax using the relationship: Tmax = ln(Ka/Ke)/(Ka-Ke)
    # Simplified: Ka ≈ 2.5/Tmax for typical absorption profiles
    tmax = get(drug, "tmax_obs", 1.0)
    tmax = tmax === nothing ? 1.0 : Float64(tmax)
    tmax = max(tmax, 0.1)
    ka = 2.5 / tmax
    # Clamp Ka to physiological range (0.1 - 10 1/h)
    ka = clamp(ka, 0.1, 10.0)

    # Hepatic blood flow (L/h) - standard value
    q_hepatic = 90.0

    # Calculate hepatic extraction ratio and Fh
    # ERH = CL_hepatic / Q_hepatic
    cl_hepatic = cl * (1.0 - fu * 0.3)  # Estimate hepatic CL
    cl_hepatic = max(0.01, cl_hepatic)

    erh = min(0.95, cl_hepatic / q_hepatic)  # Cap at 0.95
    fh = 1.0 - erh
    fh = max(0.05, fh)  # Minimum 5% escapes first-pass

    # Estimate Fg based on drug properties
    # For high Vd drugs, assume some gut metabolism
    # CYP3A4 substrates typically have Fg = 0.4-0.9
    fg = if vd > 200.0
        0.6  # High Vd often correlates with CYP3A4 metabolism
    elseif vd > 100.0
        0.75
    else
        0.9  # Lower Vd, less gut metabolism
    end

    # Adjust F to account for Fa (fraction absorbed)
    # F_observed = Fa * Fg * Fh
    # So Fa = F_observed / (Fg * Fh)
    fa = min(1.0, f / (fg * fh))
    fa = max(0.1, fa)  # Minimum 10% absorbed

    # Recalculate effective bioavailability
    f_eff = fa * fg * fh

    # Estimate lag time based on formulation (assume immediate release)
    lag = 0.0

    # Clearances
    cl_renal = cl * fu * 0.3

    # Estimate partition coefficients from Vd
    base_kp = max(0.1, vd / 50.0)
    safe_kp(x) = max(0.01, round(x, digits=3))

    # Ensure minimum clearances
    cl_renal = max(0.0, cl_renal)

    return """
model $(safe_name)_OralPBPK {
    // Drug: $name
    // Source: Literature PK parameters with oral absorption

    // Route of administration
    route: oral

    // Oral absorption parameters
    absorption {
        Ka: $(round(ka, digits=3)),
        F: $(round(fa, digits=3)),
        lag: $(round(lag, digits=2))
    }

    // First-pass metabolism
    firstpass {
        Fg: $(round(fg, digits=3)),
        Fh: $(round(fh, digits=3))
    }

    // Clearance mechanisms
    clearance hepatic: $(round(cl_hepatic, digits=3))_L/h
    clearance renal: $(round(cl_renal, digits=3))_L/h

    // Organ definitions with estimated Kp values
    organ blood { V: 5.0_L, Q: 0.0_L/h, Kp: 1.0 }
    organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: $(safe_kp(base_kp * 1.2)) }
    organ kidney { V: 0.31_L, Q: 60.0_L/h, Kp: $(safe_kp(base_kp * 0.8)) }
    organ brain { V: 1.4_L, Q: 50.0_L/h, Kp: $(safe_kp(base_kp * 0.3 * max(fu, 0.01))) }
    organ heart { V: 0.33_L, Q: 20.0_L/h, Kp: $(safe_kp(base_kp * 0.9)) }
    organ lung { V: 0.5_L, Q: 300.0_L/h, Kp: $(safe_kp(base_kp * 0.7)) }
    organ muscle { V: 30.0_L, Q: 75.0_L/h, Kp: $(safe_kp(base_kp * 0.6)) }
    organ adipose { V: 15.0_L, Q: 12.0_L/h, Kp: $(safe_kp(base_kp * 1.5)) }
    organ gut { V: 1.1_L, Q: 45.0_L/h, Kp: $(safe_kp(base_kp * 0.8)) }
    organ skin { V: 3.3_L, Q: 10.0_L/h, Kp: $(safe_kp(base_kp * 0.5)) }
    organ bone { V: 10.0_L, Q: 5.0_L/h, Kp: $(safe_kp(base_kp * 0.2)) }
    organ spleen { V: 0.18_L, Q: 15.0_L/h, Kp: $(safe_kp(base_kp * 0.9)) }
    organ pancreas { V: 0.1_L, Q: 5.0_L/h, Kp: $(safe_kp(base_kp * 0.7)) }
    organ other { V: 5.0_L, Q: 20.0_L/h, Kp: $(safe_kp(base_kp * 0.5)) }
}
"""
end

"""
Simulate oral absorption using FULL ODE integration (15-compartment PBPK).

Uses the ODEPBPKSolver.simulate_oral() function with proper gut lumen compartment
for accurate multi-compartment PK dynamics.
"""
function simulate_oral_absorption(params::ODEPBPKSolver.PBPKParams,
                                   oral_params::MedLang.OralAbsorptionParams,
                                   dose::Float64;
                                   t_max::Float64=24.0,
                                   num_points::Int=200)
    # Convert MedLang.OralAbsorptionParams to ODEPBPKSolver.OralParams
    ode_oral_params = ODEPBPKSolver.OralParams(
        oral_params.ka,
        oral_params.f,    # Fa (fraction absorbed)
        oral_params.fg,   # Fg (gut availability)
        oral_params.fh,   # Fh (hepatic availability)
        oral_params.lag   # Lag time
    )

    # Use full ODE integration from ODEPBPKSolver
    results = ODEPBPKSolver.simulate_oral(
        params,
        ode_oral_params,
        dose;
        t_max=t_max,
        num_points=num_points
    )

    # Ensure half_life key exists (may be NaN)
    if !haskey(results, "half_life")
        results["half_life"] = NaN
    end

    return results
end

"""
Calculate fold error (predicted/observed).
"""
function fold_error(pred::Float64, obs::Float64)::Float64
    if obs <= 0 || pred <= 0 || isnan(pred) || isnan(obs)
        return NaN
    end
    return pred > obs ? pred / obs : obs / pred
end

"""
Run validation for a single drug with oral absorption.
"""
function validate_drug_oral(drug)
    name = get(drug, "drug_name", "Unknown")

    try
        # Generate MedLang model with oral absorption
        medlang_source = generate_oral_medlang_model(drug)

        # Get extended params
        extended = transpile_to_extended_params(medlang_source)

        # Get dose
        dose = get(drug, "dose", 100.0)
        dose = dose === nothing ? 100.0 : Float64(dose)

        # Determine simulation time based on half-life
        half_life = get(drug, "half_life", 4.0)
        half_life = half_life === nothing ? 4.0 : Float64(half_life)
        t_max = max(24.0, 5 * half_life)

        # Run oral absorption simulation
        results = simulate_oral_absorption(
            extended.pbpk_params,
            extended.oral_params,
            dose;
            t_max=t_max,
            num_points=200
        )

        # Get predicted values
        cmax_pred = results["cmax"]
        tmax_pred = results["tmax"]
        auc_pred = results["auc"]
        halflife_pred = get(results, "half_life", NaN)

        # Get observed values
        cmax_obs_raw = get(drug, "cmax_obs", nothing)
        cmax_obs = nothing
        cmax_units = "mg/L"
        if cmax_obs_raw !== nothing
            if !isa(cmax_obs_raw, Number)
                cmax_obs = get(cmax_obs_raw, "value", nothing)
                cmax_units = get(cmax_obs_raw, "units", "mg/L")
            else
                cmax_obs = cmax_obs_raw
            end
            if cmax_obs !== nothing && contains(lowercase(string(cmax_units)), "ng")
                cmax_obs = cmax_obs / 1000.0
            end
        end

        auc_obs_raw = get(drug, "auc_obs", nothing)
        auc_obs = nothing
        auc_units = "mg*h/L"
        if auc_obs_raw !== nothing
            if !isa(auc_obs_raw, Number)
                auc_obs = get(auc_obs_raw, "value", nothing)
                auc_units = get(auc_obs_raw, "units", "mg*h/L")
            else
                auc_obs = auc_obs_raw
            end
            if auc_obs !== nothing && contains(lowercase(string(auc_units)), "ng")
                auc_obs = auc_obs / 1000.0
            end
        end

        tmax_obs = get(drug, "tmax_obs", nothing)
        if tmax_obs !== nothing && !isa(tmax_obs, Number)
            tmax_obs = get(tmax_obs, "value", nothing)
        end

        half_life_obs = get(drug, "half_life", nothing)
        if half_life_obs !== nothing && !isa(half_life_obs, Number)
            half_life_obs = get(half_life_obs, "value", nothing)
        end

        # Calculate fold errors
        fe_cmax = (cmax_obs !== nothing && cmax_obs > 0) ? fold_error(cmax_pred, Float64(cmax_obs)) : NaN
        fe_auc = (auc_obs !== nothing && auc_obs > 0) ? fold_error(auc_pred, Float64(auc_obs)) : NaN
        fe_tmax = (tmax_obs !== nothing && tmax_obs > 0) ? fold_error(tmax_pred, Float64(tmax_obs)) : NaN
        fe_halflife = (half_life_obs !== nothing && half_life_obs > 0) ? fold_error(halflife_pred, Float64(half_life_obs)) : NaN

        # Get oral params for reporting
        f_eff = MedLang.effective_bioavailability(extended.oral_params)

        return (
            name = name,
            success = true,
            cmax_pred = cmax_pred,
            cmax_obs = cmax_obs,
            fe_cmax = fe_cmax,
            auc_pred = auc_pred,
            auc_obs = auc_obs,
            fe_auc = fe_auc,
            tmax_pred = tmax_pred,
            tmax_obs = tmax_obs,
            fe_tmax = fe_tmax,
            halflife_pred = halflife_pred,
            halflife_obs = half_life_obs,
            fe_halflife = fe_halflife,
            ka = extended.oral_params.ka,
            f_eff = f_eff,
            error = nothing
        )

    catch e
        @warn "Drug $name failed: $e"
        return (
            name = name,
            success = false,
            cmax_pred = NaN,
            cmax_obs = nothing,
            fe_cmax = NaN,
            auc_pred = NaN,
            auc_obs = nothing,
            fe_auc = NaN,
            tmax_pred = NaN,
            tmax_obs = nothing,
            fe_tmax = NaN,
            halflife_pred = NaN,
            halflife_obs = nothing,
            fe_halflife = NaN,
            ka = NaN,
            f_eff = NaN,
            error = string(e)
        )
    end
end

"""
Calculate geometric mean fold error.
"""
function gmfe(fold_errors::Vector{Float64})::Float64
    valid_fe = filter(x -> !isnan(x) && x > 0, fold_errors)
    if isempty(valid_fe)
        return NaN
    end
    return exp(mean(log.(valid_fe)))
end

"""
Calculate percentage within X-fold.
"""
function percent_within_fold(fold_errors::Vector{Float64}, fold::Float64)::Float64
    valid_fe = filter(x -> !isnan(x) && x > 0, fold_errors)
    if isempty(valid_fe)
        return 0.0
    end
    within = count(fe -> fe <= fold, valid_fe)
    return 100.0 * within / length(valid_fe)
end

"""
Main validation function with oral absorption.
"""
function run_oral_validation(dataset_path::String; max_drugs::Int=0)
    println("=" ^ 70)
    println("MedLang DSL ORAL ABSORPTION Validation")
    println("Enhanced Model with First-Pass Metabolism")
    println("=" ^ 70)
    println()

    # Load dataset
    println("Loading dataset: $dataset_path")
    drugs = load_dataset(dataset_path)
    println("Total drugs in dataset: $(length(drugs))")

    if max_drugs > 0 && max_drugs < length(drugs)
        drugs = drugs[1:max_drugs]
        println("Limiting to first $max_drugs drugs for testing")
    end
    println()

    # Run validation for each drug
    results = []
    successful = 0
    failed = 0

    println("Running oral absorption simulations...")
    println("-" ^ 70)

    for (i, drug) in enumerate(drugs)
        result = validate_drug_oral(drug)
        push!(results, result)

        if result.success
            successful += 1
            status = "OK"
        else
            failed += 1
            status = "XX"
        end

        if i % 50 == 0 || i == length(drugs)
            @printf("  [%3d/%3d] %s\n", i, length(drugs), status)
        end
    end

    println()
    println("=" ^ 70)
    println("ORAL ABSORPTION VALIDATION RESULTS")
    println("=" ^ 70)
    println()

    # Summary statistics
    println("Simulation Success Rate:")
    @printf("  Successful: %d / %d (%.1f%%)\n", successful, length(drugs), 100.0 * successful / length(drugs))
    @printf("  Failed: %d\n", failed)
    println()

    # Collect fold errors
    fe_cmax_all = [r.fe_cmax for r in results if r.success]
    fe_auc_all = [r.fe_auc for r in results if r.success]
    fe_tmax_all = [r.fe_tmax for r in results if r.success]
    fe_halflife_all = [r.fe_halflife for r in results if r.success]

    println("FDA/EMA Regulatory Validation Metrics:")
    println("-" ^ 50)

    # Cmax metrics
    println("\nCmax Predictions (with oral absorption):")
    valid_cmax = filter(!isnan, fe_cmax_all)
    if !isempty(valid_cmax)
        @printf("  N valid: %d\n", length(valid_cmax))
        @printf("  GMFE: %.3f\n", gmfe(valid_cmax))
        @printf("  Within 1.25-fold: %.1f%%\n", percent_within_fold(valid_cmax, 1.25))
        @printf("  Within 1.5-fold:  %.1f%%\n", percent_within_fold(valid_cmax, 1.5))
        @printf("  Within 2.0-fold:  %.1f%%\n", percent_within_fold(valid_cmax, 2.0))
        @printf("  Within 3.0-fold:  %.1f%%\n", percent_within_fold(valid_cmax, 3.0))
    else
        println("  No valid predictions")
    end

    # AUC metrics
    println("\nAUC Predictions (with oral absorption):")
    valid_auc = filter(!isnan, fe_auc_all)
    if !isempty(valid_auc)
        @printf("  N valid: %d\n", length(valid_auc))
        @printf("  GMFE: %.3f\n", gmfe(valid_auc))
        @printf("  Within 1.25-fold: %.1f%%\n", percent_within_fold(valid_auc, 1.25))
        @printf("  Within 1.5-fold:  %.1f%%\n", percent_within_fold(valid_auc, 1.5))
        @printf("  Within 2.0-fold:  %.1f%%\n", percent_within_fold(valid_auc, 2.0))
        @printf("  Within 3.0-fold:  %.1f%%\n", percent_within_fold(valid_auc, 3.0))
    else
        println("  No valid predictions")
    end

    # Tmax metrics
    println("\nTmax Predictions:")
    valid_tmax = filter(!isnan, fe_tmax_all)
    if !isempty(valid_tmax)
        @printf("  N valid: %d\n", length(valid_tmax))
        @printf("  GMFE: %.3f\n", gmfe(valid_tmax))
        @printf("  Within 1.5-fold:  %.1f%%\n", percent_within_fold(valid_tmax, 1.5))
        @printf("  Within 2.0-fold:  %.1f%%\n", percent_within_fold(valid_tmax, 2.0))
    else
        println("  No valid predictions")
    end

    # Half-life metrics
    println("\nHalf-life Predictions:")
    valid_hl = filter(!isnan, fe_halflife_all)
    if !isempty(valid_hl)
        @printf("  N valid: %d\n", length(valid_hl))
        @printf("  GMFE: %.3f\n", gmfe(valid_hl))
        @printf("  Within 1.5-fold:  %.1f%%\n", percent_within_fold(valid_hl, 1.5))
        @printf("  Within 2.0-fold:  %.1f%%\n", percent_within_fold(valid_hl, 2.0))
    else
        println("  No valid predictions")
    end

    println()
    println("=" ^ 70)
    println("Improvement Analysis:")
    println("=" ^ 70)
    println()
    println("Previous IV model results:")
    println("  - Cmax: GMFE 11.73, 4.4% within 2-fold")
    println("  - AUC:  GMFE 13.77, 43.4% within 2-fold")
    println()
    println("Current oral absorption model results:")
    if !isempty(valid_cmax)
        @printf("  - Cmax: GMFE %.2f, %.1f%% within 2-fold\n", gmfe(valid_cmax), percent_within_fold(valid_cmax, 2.0))
    end
    if !isempty(valid_auc)
        @printf("  - AUC:  GMFE %.2f, %.1f%% within 2-fold\n", gmfe(valid_auc), percent_within_fold(valid_auc, 2.0))
    end

    println()
    println("=" ^ 70)
    println("FDA Acceptance Criteria (for reference):")
    println("  - GMFE < 2.0 for Cmax and AUC")
    println("  - >50% predictions within 2-fold")
    println("=" ^ 70)

    # Sample predictions
    println()
    println("Sample Predictions (first 15 successful with valid Cmax):")
    println("-" ^ 90)
    @printf("%-20s %8s %8s %6s %8s %8s %6s\n", "Drug", "Cmax_P", "Cmax_O", "FE", "AUC_P", "AUC_O", "FE")
    println("-" ^ 90)

    count = 0
    for r in results
        if r.success && !isnan(r.fe_cmax) && count < 15
            cmax_obs_str = r.cmax_obs !== nothing ? @sprintf("%.2f", r.cmax_obs) : "N/A"
            auc_obs_str = r.auc_obs !== nothing ? @sprintf("%.1f", r.auc_obs) : "N/A"
            auc_fe_str = !isnan(r.fe_auc) ? @sprintf("%.2f", r.fe_auc) : "N/A"
            @printf("%-20s %8.2f %8s %6.2f %8.1f %8s %6s\n",
                    r.name[1:min(20, length(r.name))],
                    r.cmax_pred,
                    cmax_obs_str,
                    r.fe_cmax,
                    r.auc_pred,
                    auc_obs_str,
                    auc_fe_str)
            count += 1
        end
    end

    println()
    println("Validation completed at: ", Dates.now())

    return results
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    dataset_path = "/mnt/f/DARWIN_VALIDATION/datasets/ULTIMATE_DATASET_v1_normalized_with_smiles.json"

    if length(ARGS) >= 1
        dataset_path = ARGS[1]
    end

    results = run_oral_validation(dataset_path; max_drugs=0)
end
