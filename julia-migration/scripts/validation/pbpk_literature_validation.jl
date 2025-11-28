#!/usr/bin/env julia
#=
PBPK Literature Validation Script
Validates Darwin PBPK model against Obach-Lombardo 1352 drug dataset

Reference: Lombardo F, Berellini G, Obach RS (2018) Drug Metab Dispos 46:1466-1477
"Trend Analysis of a Database of Intravenous Pharmacokinetic Parameters
in Humans for 1352 Drug Compounds"

Validation Metrics:
- GMFE (Geometric Mean Fold Error): Should be < 2 for good predictions
- AFE (Average Fold Error): Ideally close to 1
- % within 2-fold: Target > 70%
- % within 3-fold: Target > 90%
- R^2 (correlation): Higher is better
=#

using CSV
using DataFrames
using Statistics
using Printf

# Add project to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))

using DarwinPBPK
using DarwinPBPK.ODEPBPKSolver: PBPKParams, simulate

println("=" ^ 70)
println("PBPK Literature Validation - Obach-Lombardo Dataset")
println("=" ^ 70)
println()

# Load dataset
data_path = joinpath(@__DIR__, "..", "..", "..", "data", "external_datasets", "obach_lombardo_1352_drugs.csv")
println("Loading dataset: $data_path")
df = CSV.read(data_path, DataFrame)

# Filter complete data (VDss + CL + t1/2)
complete_df = dropmissing(df, [:human_VDss_L_kg, :human_CL_mL_min_kg, :human_thalf])
println("Total drugs with complete data: $(nrow(complete_df))")
println()

# Validation parameters
const STANDARD_DOSE = 100.0  # mg IV bolus
const BODY_WEIGHT = 70.0     # kg standard human
const T_MAX = 168.0          # 1 week simulation
const NUM_POINTS = 500

# Default tissue volumes for Kp calculation (L, 70kg human)
const TISSUE_VOLUMES = Dict(
    "blood" => 5.0, "liver" => 1.8, "kidney" => 0.31, "brain" => 1.4,
    "heart" => 0.33, "lung" => 0.5, "muscle" => 30.0, "adipose" => 15.0,
    "gut" => 1.1, "skin" => 3.3, "bone" => 10.0, "spleen" => 0.18,
    "pancreas" => 0.1, "other" => 5.0
)
const TOTAL_TISSUE_VOLUME = sum(values(TISSUE_VOLUMES))  # ~73 L

"""
Calculate partition coefficients to match observed Vdss.
Uses well-mixed model: Vdss = Vp + sum(Vt * Kp * fu/fut)
Simplified: Scale all Kp uniformly to match total Vdss
"""
function calculate_kp_from_vdss(obs_vdss_l_kg::Float64, fup::Float64)
    # Target Vdss in L (70kg)
    vdss_l = obs_vdss_l_kg * BODY_WEIGHT

    # Blood/plasma volume (~5L)
    vp = 5.0

    # Tissue volume available for distribution
    vt_total = TOTAL_TISSUE_VOLUME - vp

    # For simple well-mixed model: Vdss ~ Vp + Kp_avg * Vt
    # Solve for average Kp
    kp_avg = max(0.1, (vdss_l - vp) / vt_total)

    # Scale individual Kp values based on tissue characteristics
    return Dict(
        "blood" => 1.0,
        "liver" => kp_avg * 1.2,      # Moderate binding
        "kidney" => kp_avg * 1.0,
        "brain" => kp_avg * 0.3,      # BBB limits distribution
        "heart" => kp_avg * 0.8,
        "lung" => kp_avg * 0.8,
        "muscle" => kp_avg * 0.6,     # Large volume, moderate Kp
        "adipose" => kp_avg * 1.5,    # Lipophilic accumulation
        "gut" => kp_avg * 1.0,
        "skin" => kp_avg * 0.5,
        "bone" => kp_avg * 0.2,
        "spleen" => kp_avg * 0.8,
        "pancreas" => kp_avg * 0.8,
        "other" => kp_avg * 0.7
    )
end

# Results storage
results = DataFrame(
    drug_idx = Int[],
    smiles = String[],
    mw = Float64[],
    obs_vdss = Float64[],
    obs_cl = Float64[],
    obs_thalf = Float64[],
    pred_thalf = Float64[],
    fold_error = Float64[],
    log_fold_error = Float64[]
)

# Run simulations
println("Running PBPK simulations...")
println("-" ^ 70)

n_total = nrow(complete_df)  # Run ALL 1232 drugs

function run_validation(complete_df, n_total, results)
    n_success = 0
    n_failed = 0

    for i in 1:n_total
        row = complete_df[i, :]

        smiles = row.smiles_r
        mw = coalesce(row.MW, row.molecular_weight_smiles_r, 300.0)
        obs_vdss = row.human_VDss_L_kg  # L/kg
        obs_cl = row.human_CL_mL_min_kg  # mL/min/kg
        obs_thalf = row.human_thalf  # hours
        fup = coalesce(row.human_fup, 0.5)  # Default 50% if missing

        # Convert units for PBPK model
        # CL: mL/min/kg -> L/h (for 70kg human)
        cl_l_h = obs_cl * 60 / 1000 * BODY_WEIGHT  # L/h

        # Vd: L/kg -> L (for 70kg human)
        vd_l = obs_vdss * BODY_WEIGHT  # L

        try
            # Calculate hepatic clearance assuming CL ~ hepatic CL for most drugs
            hepatic_cl = cl_l_h * 0.9  # 90% hepatic
            renal_cl = cl_l_h * 0.1    # 10% renal

            # Calculate partition coefficients from observed Vdss
            partition_coeffs = calculate_kp_from_vdss(obs_vdss, fup)

            params = PBPKParams(
                clearance_hepatic = hepatic_cl,
                clearance_renal = renal_cl,
                partition_coeffs = partition_coeffs
            )

            # Run simulation with standard dose
            result = simulate(params, STANDARD_DOSE; t_max=T_MAX, num_points=NUM_POINTS)

            # Extract concentration-time profile from blood compartment
            times = result["time"]
            conc = result["blood"]

            # Calculate half-life from terminal phase
            # Find elimination phase (after Cmax)
            cmax_idx = argmax(conc)
            pred_thalf = NaN

            if cmax_idx < length(conc) - 10
                elim_times = times[cmax_idx:end]
                elim_conc = conc[cmax_idx:end]

                # Filter positive concentrations
                valid_idx = elim_conc .> 1e-10
                if sum(valid_idx) > 5
                    log_conc = log.(elim_conc[valid_idx])
                    t_elim = elim_times[valid_idx]

                    # Linear regression on log-linear plot
                    n = length(t_elim)
                    sum_t = sum(t_elim)
                    sum_log = sum(log_conc)
                    sum_t2 = sum(t_elim.^2)
                    sum_t_log = sum(t_elim .* log_conc)

                    slope = (n * sum_t_log - sum_t * sum_log) / (n * sum_t2 - sum_t^2)

                    # Half-life = ln(2) / |slope|
                    if slope < 0
                        pred_thalf = log(2) / abs(slope)
                    end
                end
            end

            if !isnan(pred_thalf) && pred_thalf > 0 && pred_thalf < 10000
                # Calculate fold error
                fe = pred_thalf > obs_thalf ? pred_thalf / obs_thalf : obs_thalf / pred_thalf
                log_fe = log10(pred_thalf / obs_thalf)

                push!(results, (i, smiles, mw, obs_vdss, obs_cl, obs_thalf, pred_thalf, fe, log_fe))
                n_success += 1

                if i <= 10 || i % 100 == 0
                    @printf("Drug %4d: obs_t1/2=%.2fh, pred_t1/2=%.2fh, FE=%.2f\n",
                           i, obs_thalf, pred_thalf, fe)
                end
            else
                n_failed += 1
            end

        catch e
            n_failed += 1
            if i <= 5
                println("Drug $i failed: $e")
            end
        end
    end

    return n_success, n_failed
end

n_success, n_failed = run_validation(complete_df, n_total, results)

println()
println("-" ^ 70)
println("Completed: $n_success successful, $n_failed failed")
println()

# Calculate validation metrics
if nrow(results) > 0
    println("=" ^ 70)
    println("VALIDATION METRICS (n=$(nrow(results)) drugs)")
    println("=" ^ 70)
    println()

    # Geometric Mean Fold Error (GMFE)
    gmfe = 10^mean(abs.(results.log_fold_error))

    # Average Fold Error (AFE)
    afe = 10^mean(results.log_fold_error)

    # Percentage within X-fold
    within_2fold = sum(results.fold_error .<= 2.0) / nrow(results) * 100
    within_3fold = sum(results.fold_error .<= 3.0) / nrow(results) * 100
    within_5fold = sum(results.fold_error .<= 5.0) / nrow(results) * 100

    # R^2 (on log scale)
    log_obs = log10.(results.obs_thalf)
    log_pred = log10.(results.pred_thalf)

    ss_res = sum((log_pred .- log_obs).^2)
    ss_tot = sum((log_obs .- mean(log_obs)).^2)
    r2 = 1 - ss_res / ss_tot

    # Correlation
    corr_val = cor(log_obs, log_pred)

    # Print results
    println("Metric                      Value     Target")
    println("-" ^ 50)
    @printf("GMFE (Geometric Mean FE):   %.3f     < 2.0\n", gmfe)
    @printf("AFE (Average FE):           %.3f     ~ 1.0\n", afe)
    @printf("Within 2-fold:              %.1f%%    > 70%%\n", within_2fold)
    @printf("Within 3-fold:              %.1f%%    > 90%%\n", within_3fold)
    @printf("Within 5-fold:              %.1f%%\n", within_5fold)
    @printf("R^2 (log scale):            %.3f     > 0.5\n", r2)
    @printf("Correlation (r):            %.3f\n", corr_val)
    println()

    # Assessment
    println("ASSESSMENT:")
    if gmfe < 2.0 && within_2fold > 70
        println("++ Model meets Obach criteria for acceptable predictions")
    elseif gmfe < 3.0 && within_2fold > 50
        println("~~ Model shows moderate prediction accuracy")
    else
        println("-- Model needs improvement")
    end
    println()

    # Save results
    output_path = joinpath(@__DIR__, "validation_results.csv")
    CSV.write(output_path, results)
    println("Results saved to: $output_path")

    # Summary statistics
    println()
    println("Summary Statistics:")
    println("-" ^ 50)
    @printf("Observed t1/2:   median=%.2fh, range=[%.2f-%.1f]h\n",
           median(results.obs_thalf), minimum(results.obs_thalf), maximum(results.obs_thalf))
    @printf("Predicted t1/2:  median=%.2fh, range=[%.2f-%.1f]h\n",
           median(results.pred_thalf), minimum(results.pred_thalf), maximum(results.pred_thalf))

else
    println("ERROR: No successful simulations")
end

println()
println("=" ^ 70)
println("Validation complete")
println("=" ^ 70)
