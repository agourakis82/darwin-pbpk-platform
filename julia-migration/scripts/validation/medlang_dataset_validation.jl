"""
MedLang DSL Dataset Validation Script

First Real Implementation Validation of MedLang DSL
Tests against 572 drugs from ULTIMATE_DATASET

This script:
1. Loads drug PK data from JSON dataset
2. Generates MedLang model definitions dynamically
3. Compiles to PBPKParams and simulates
4. Compares predicted vs observed PK parameters
5. Computes FDA/EMA regulatory validation metrics

Author: Dr. Demetrios Agourakis
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
    # Simple string parsing (assumes no escaped quotes in data)
    m = match(r"^\"([^\"]*)\"", s)
    return m !== nothing ? m.captures[1] : ""
end

function parse_json_array(s::String)
    result = Any[]
    s = strip(s[2:end])  # Remove opening [
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
    s = strip(s[2:end])  # Remove opening {
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

# Use Julia's JSON if available, otherwise fall back to simple parser
function load_dataset(filepath::String)
    try
        # Try to use JSON.jl if available
        @eval using JSON
        return JSON.parsefile(filepath)
    catch
        # Fall back to simple parser
        return parse_json_file(filepath)
    end
end

"""
Generate MedLang model definition from drug PK parameters.
"""
function generate_medlang_model(drug)::String
    name = get(drug, "drug_name", "Unknown")
    # Sanitize name for identifier
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

    # Calculate Ka from Tmax (approximation: Ka ≈ 2.5/Tmax for 1-cpt oral)
    tmax = get(drug, "tmax_obs", 1.0)
    tmax = tmax === nothing ? 1.0 : Float64(tmax)
    tmax = max(tmax, 0.1)
    ka = 2.5 / tmax

    # Estimate hepatic vs renal clearance based on fu
    # Higher fu → more renal clearance
    cl_hepatic = cl * (1.0 - fu * 0.3)
    cl_renal = cl * fu * 0.3

    # Estimate partition coefficients from Vd
    # Higher Vd → more tissue distribution
    base_kp = max(0.1, vd / 50.0)  # Normalize to typical Vd, ensure min 0.1

    # Helper function to ensure minimum Kp
    safe_kp(x) = max(0.01, round(x, digits=3))

    # Ensure minimum clearances
    cl_hepatic = max(0.01, cl_hepatic)
    cl_renal = max(0.0, cl_renal)

    return """
model $(safe_name)_PBPK {
    // Drug: $name
    // Source: Literature PK parameters

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
Calculate PK parameters from simulation results.
"""
function calculate_pk_params(results::Dict, dose::Float64)
    time = results["time"]
    conc = results["blood"]

    # Cmax
    cmax = maximum(conc)
    cmax_idx = argmax(conc)
    tmax = time[cmax_idx]

    # AUC (trapezoidal rule)
    auc = 0.0
    for i in 2:length(time)
        dt = time[i] - time[i-1]
        auc += 0.5 * (conc[i] + conc[i-1]) * dt
    end

    # Terminal half-life (from last 3 points in log domain)
    if length(conc) >= 3 && conc[end] > 0
        # Find terminal phase (last 25% of data where conc > 0)
        n_terminal = max(3, length(conc) ÷ 4)
        terminal_idx = (length(conc) - n_terminal + 1):length(conc)

        # Filter positive concentrations
        valid_idx = [i for i in terminal_idx if conc[i] > 1e-10]

        if length(valid_idx) >= 2
            t_term = time[valid_idx]
            c_term = log.(conc[valid_idx])

            # Linear regression for slope
            n = length(t_term)
            sum_t = sum(t_term)
            sum_c = sum(c_term)
            sum_tc = sum(t_term .* c_term)
            sum_t2 = sum(t_term .^ 2)

            slope = (n * sum_tc - sum_t * sum_c) / (n * sum_t2 - sum_t^2)
            half_life = -log(2) / slope
        else
            half_life = NaN
        end
    else
        half_life = NaN
    end

    return (cmax=cmax, tmax=tmax, auc=auc, half_life=half_life)
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
Run validation for a single drug.
"""
function validate_drug(drug)
    name = get(drug, "drug_name", "Unknown")

    try
        # Generate MedLang model
        medlang_source = generate_medlang_model(drug)

        # Compile to PBPKParams
        params = compile_model(medlang_source)

        # Get dose and simulate
        dose = get(drug, "dose", 100.0)
        dose = dose === nothing ? 100.0 : Float64(dose)

        # Determine simulation time based on half-life
        half_life = get(drug, "half_life", 4.0)
        half_life = half_life === nothing ? 4.0 : Float64(half_life)
        t_max = max(24.0, 5 * half_life)

        # Run simulation
        results = ODEPBPKSolver.simulate(params, dose; t_max=t_max, num_points=200)

        # Calculate predicted PK parameters
        pk_pred = calculate_pk_params(results, dose)

        # Get observed values (handle both direct values and {value, units} objects)
        # Convert ng/mL to mg/L (divide by 1000) for comparison with PBPK output
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
            # Convert ng/mL to mg/L
            if cmax_obs !== nothing && contains(lowercase(string(cmax_units)), "ng")
                cmax_obs = cmax_obs / 1000.0  # ng/mL → mg/L (or µg/L)
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
            # Convert ng*h/mL to mg*h/L
            if auc_obs !== nothing && contains(lowercase(string(auc_units)), "ng")
                auc_obs = auc_obs / 1000.0  # ng*h/mL → mg*h/L (or µg*h/L)
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

        # Calculate fold errors (convert to Float64 safely)
        fe_cmax = (cmax_obs !== nothing && cmax_obs > 0) ? fold_error(pk_pred.cmax, Float64(cmax_obs)) : NaN
        fe_auc = (auc_obs !== nothing && auc_obs > 0) ? fold_error(pk_pred.auc, Float64(auc_obs)) : NaN
        fe_tmax = (tmax_obs !== nothing && tmax_obs > 0) ? fold_error(pk_pred.tmax, Float64(tmax_obs)) : NaN
        fe_halflife = (half_life_obs !== nothing && half_life_obs > 0) ? fold_error(pk_pred.half_life, Float64(half_life_obs)) : NaN

        return (
            name = name,
            success = true,
            cmax_pred = pk_pred.cmax,
            cmax_obs = cmax_obs,
            fe_cmax = fe_cmax,
            auc_pred = pk_pred.auc,
            auc_obs = auc_obs,
            fe_auc = fe_auc,
            tmax_pred = pk_pred.tmax,
            tmax_obs = tmax_obs,
            fe_tmax = fe_tmax,
            halflife_pred = pk_pred.half_life,
            halflife_obs = half_life_obs,
            fe_halflife = fe_halflife,
            error = nothing
        )

    catch e
        # Print error for debugging
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
Main validation function.
"""
function run_validation(dataset_path::String; max_drugs::Int=0)
    println("=" ^ 70)
    println("MedLang DSL Dataset Validation")
    println("First Real Implementation Test")
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

    println("Running simulations...")
    println("-" ^ 70)

    for (i, drug) in enumerate(drugs)
        result = validate_drug(drug)
        push!(results, result)

        if result.success
            successful += 1
            status = "✓"
        else
            failed += 1
            status = "✗"
        end

        # Progress indicator every 10 drugs
        if i % 10 == 0 || i == length(drugs)
            @printf("  [%3d/%3d] %s %s\n", i, length(drugs), status, result.name)
        end
    end

    println()
    println("=" ^ 70)
    println("VALIDATION RESULTS SUMMARY")
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
    println("\nCmax Predictions:")
    valid_cmax = filter(!isnan, fe_cmax_all)
    if !isempty(valid_cmax)
        @printf("  N valid: %d\n", length(valid_cmax))
        @printf("  GMFE: %.3f\n", gmfe(valid_cmax))
        @printf("  Within 1.25-fold: %.1f%%\n", percent_within_fold(valid_cmax, 1.25))
        @printf("  Within 1.5-fold:  %.1f%%\n", percent_within_fold(valid_cmax, 1.5))
        @printf("  Within 2.0-fold:  %.1f%%\n", percent_within_fold(valid_cmax, 2.0))
    else
        println("  No valid predictions")
    end

    # AUC metrics
    println("\nAUC Predictions:")
    valid_auc = filter(!isnan, fe_auc_all)
    if !isempty(valid_auc)
        @printf("  N valid: %d\n", length(valid_auc))
        @printf("  GMFE: %.3f\n", gmfe(valid_auc))
        @printf("  Within 1.25-fold: %.1f%%\n", percent_within_fold(valid_auc, 1.25))
        @printf("  Within 1.5-fold:  %.1f%%\n", percent_within_fold(valid_auc, 1.5))
        @printf("  Within 2.0-fold:  %.1f%%\n", percent_within_fold(valid_auc, 2.0))
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
    println("FDA Acceptance Criteria (for reference):")
    println("  - GMFE < 2.0 for Cmax and AUC")
    println("  - >50% predictions within 2-fold")
    println("=" ^ 70)

    # Show some example predictions
    println()
    println("Sample Predictions (first 10 successful):")
    println("-" ^ 70)
    @printf("%-20s %10s %10s %8s\n", "Drug", "Pred Cmax", "Obs Cmax", "FE")
    println("-" ^ 70)

    count = 0
    for r in results
        if r.success && !isnan(r.fe_cmax) && count < 10
            obs_str = r.cmax_obs !== nothing ? @sprintf("%.2f", r.cmax_obs) : "N/A"
            @printf("%-20s %10.2f %10s %8.2f\n",
                    r.name[1:min(20, length(r.name))],
                    r.cmax_pred,
                    obs_str,
                    r.fe_cmax)
            count += 1
        end
    end

    println()
    println("Validation completed at: ", Dates.now())

    return results
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    # Default dataset path
    dataset_path = "/mnt/f/DARWIN_VALIDATION/datasets/ULTIMATE_DATASET_v1_normalized_with_smiles.json"

    # Check command line arguments
    if length(ARGS) >= 1
        dataset_path = ARGS[1]
    end

    # Run full validation on all drugs
    results = run_validation(dataset_path; max_drugs=0)
end
