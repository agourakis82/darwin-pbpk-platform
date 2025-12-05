"""
Sensitivity Analysis Demo Script

Demonstrates usage of the sensitivity analysis module with:
1. Simple pharmacokinetic model
2. Coagulation model sensitivity
3. Visualization of results

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

# Add project to path if running as script
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
end

using DarwinPBPK
using Printf
using Statistics

println("="^80)
println("Darwin PBPK Platform - Sensitivity Analysis Demo")
println("="^80)
println()

# ============================================================================
# DEMO 1: Simple PK Model - One-Compartment Clearance
# ============================================================================

println("DEMO 1: One-Compartment PK Model Sensitivity")
println("-"^80)

"""
One-compartment PK model with first-order elimination
C(t) = (Dose/Vd) * exp(-CL/Vd * t)
"""
function one_compartment_pk(params::Dict{String, Float64})
    # Extract parameters
    dose = params["dose"]           # mg
    Vd = params["Vd"]              # L
    CL = params["CL"]              # L/h
    t_sample = params["t_sample"]  # hours

    # Calculate concentration at sampling time
    C0 = dose / Vd
    ke = CL / Vd
    C_sample = C0 * exp(-ke * t_sample)

    # Calculate PK metrics
    AUC = dose / CL
    t_half = log(2) / ke

    return Dict(
        "C_sample" => C_sample,     # Concentration at t_sample (mg/L)
        "AUC" => AUC,               # Area under curve (mg·h/L)
        "t_half" => t_half,         # Half-life (hours)
        "Cmax" => C0                # Maximum concentration (mg/L)
    )
end

# Define parameter ranges (±30% variability)
pk_params = [
    ParameterRange("dose", 100.0, 70.0, 130.0),
    ParameterRange("Vd", 50.0, 35.0, 65.0),
    ParameterRange("CL", 5.0, 3.5, 6.5),
    ParameterRange("t_sample", 4.0, 4.0, 4.0)  # Fixed sampling time
]

pk_outputs = ["C_sample", "AUC", "t_half", "Cmax"]

# Run OAT sensitivity
println("\n1a. One-At-a-Time (OAT) Sensitivity...")
oat_result = one_at_a_time_sensitivity(one_compartment_pk, pk_params, pk_outputs)

println("\nOAT Sensitivity Results:")
println("─"^80)
for output in pk_outputs
    println("\n$output:")
    rankings = oat_result.rankings[output]
    for (i, (param, sens)) in enumerate(rankings)
        @printf("  %d. %-12s: %+.4f\n", i, param, sens)
    end
end

# Run Sobol sensitivity
println("\n1b. Sobol Global Sensitivity (variance-based)...")
sobol_result = sobol_sensitivity(one_compartment_pk, pk_params[1:3], pk_outputs, n_samples=512)

println("\nSobol Indices (First-Order and Total):")
println("─"^80)
for output in pk_outputs
    println("\n$output:")
    S1 = sobol_result.indices["S1"][output]
    ST = sobol_result.indices["ST"][output]

    println("  Parameter    S₁ (main)   Sᴛ (total)  Interaction")
    println("  " * "─"^60)
    for param in sobol_result.parameters
        interaction = ST[param] - S1[param]
        @printf("  %-12s %8.4f    %8.4f    %8.4f\n", param, S1[param], ST[param], interaction)
    end
end

# Identify influential parameters
influential = identify_influential_parameters(sobol_result, 0.1)
println("\nInfluential Parameters (|S| > 0.1):")
for (out, params) in influential
    println("  $out: $(join(params, ", "))")
end

# ============================================================================
# DEMO 2: Coagulation Model Sensitivity
# ============================================================================

println("\n\n" * "="^80)
println("DEMO 2: Coagulation Cascade Sensitivity")
println("-"^80)

# Get default coagulation parameters
coag_params = default_coagulation_parameters()
coag_outputs = ["peak_thrombin", "lag_time", "etp"]

println("\nAnalyzing sensitivity of thrombin generation to coagulation factors...")
println("Parameters: $(join([p.name for p in coag_params], ", "))")
println("Outputs: $(join(coag_outputs, ", "))")

# Run OAT sensitivity
println("\n2a. OAT Sensitivity Analysis...")
coag_oat = one_at_a_time_sensitivity(
    coagulation_sensitivity_wrapper,
    coag_params,
    coag_outputs,
    perturbation=0.05
)

println("\nCoagulation Factor Sensitivities:")
println("─"^80)
for output in coag_outputs
    println("\n$output:")
    rankings = coag_oat.rankings[output]
    for (i, (param, sens)) in enumerate(rankings[1:min(5, length(rankings))])
        impact = sens > 0 ? "↑ increases" : "↓ decreases"
        @printf("  %d. %-8s: %+.4f  (%s output)\n", i, param, sens, impact)
    end
end

# Morris screening for efficient global sensitivity
println("\n2b. Morris Screening (Elementary Effects)...")
coag_morris = morris_screening(
    coagulation_sensitivity_wrapper,
    coag_params,
    coag_outputs,
    n_trajectories=8,
    n_levels=4
)

println("\nMorris Screening Results (μ* = importance, σ = interaction/nonlinearity):")
println("─"^80)
for output in coag_outputs
    println("\n$output:")
    μ_star = coag_morris.indices["μ_star"][output]
    σ = coag_morris.indices["σ"][output]

    # Sort by μ*
    sorted_params = sort(collect(zip(keys(μ_star), values(μ_star),
                                     [σ[k] for k in keys(μ_star)])),
                        by=x->x[2], rev=true)

    println("  Parameter     μ*        σ      σ/μ*")
    println("  " * "─"^50)
    for (param, mu, sig) in sorted_params[1:min(5, length(sorted_params))]
        ratio = mu > 1e-10 ? sig/mu : 0.0
        @printf("  %-10s %8.2f  %8.2f  %6.3f\n", param, mu, sig, ratio)
    end
end

# ============================================================================
# DEMO 3: PRCC Analysis for Coagulation
# ============================================================================

println("\n\n" * "="^80)
println("DEMO 3: PRCC Analysis (Partial Rank Correlation)")
println("-"^80)

println("\n3a. Running PRCC analysis...")
coag_prcc = prcc_analysis(
    coagulation_sensitivity_wrapper,
    coag_params,
    coag_outputs,
    n_samples=500,
    seed=42
)

println("\nPRCC Values (monotonic relationship with output):")
println("─"^80)
for output in coag_outputs
    println("\n$output:")
    prcc_vals = coag_prcc.indices["PRCC"][output]

    # Sort by absolute PRCC
    sorted_prcc = sort(collect(prcc_vals), by=x->abs(x[2]), rev=true)

    println("  Parameter     PRCC    Interpretation")
    println("  " * "─"^60)
    for (param, prcc) in sorted_prcc[1:min(5, length(sorted_prcc))]
        if abs(prcc) > 0.5
            strength = "Strong"
        elseif abs(prcc) > 0.3
            strength = "Moderate"
        elseif abs(prcc) > 0.1
            strength = "Weak"
        else
            strength = "Negligible"
        end

        direction = prcc > 0 ? "positive" : "negative"
        @printf("  %-10s %+7.4f   %s %s correlation\n",
                param, prcc, strength, direction)
    end
end

# ============================================================================
# DEMO 4: Method Comparison
# ============================================================================

println("\n\n" * "="^80)
println("DEMO 4: Sensitivity Method Comparison")
println("-"^80)

println("\nRanking comparison for peak_thrombin:")
println("─"^80)

# Compare rankings from different methods
output = "peak_thrombin"

println("\n%-12s %12s %12s %12s" % ("Parameter", "OAT", "Morris μ*", "PRCC"))
println("─"^60)

for param in [p.name for p in coag_params]
    oat_val = coag_oat.sensitivities[output][param]
    morris_val = coag_morris.indices["μ_star"][output][param]
    prcc_val = coag_prcc.indices["PRCC"][output][param]

    @printf("%-12s %+11.4f %11.2f %+11.4f\n",
            param, oat_val, morris_val, prcc_val)
end

println("\nKey Insights:")
println("  • OAT: Local sensitivity (1% perturbation around nominal)")
println("  • Morris μ*: Global screening (mean absolute elementary effect)")
println("  • PRCC: Monotonic correlation across full parameter space")
println("  • High agreement → robust parameter importance")
println("  • Disagreement → nonlinear effects or interactions")

# ============================================================================
# DEMO 5: Tornado Plot Data
# ============================================================================

println("\n\n" * "="^80)
println("DEMO 5: Tornado Plot Data (for visualization)")
println("-"^80)

tornado = sensitivity_tornado_plot_data(coag_oat, "peak_thrombin")

println("\nTornado diagram data (sorted by |sensitivity|):")
println("─"^80)
println("Parameter    Sensitivity   " * " "^10 * "Visual Bar")
println("─"^80)

max_sens = maximum(abs(x[2]) for x in tornado)
for (param, sens) in tornado
    bar_length = Int(round(abs(sens) / max_sens * 40))
    bar = sens > 0 ? "█"^bar_length : "▓"^bar_length
    direction = sens > 0 ? "→" : "←"
    @printf("%-10s %+9.4f   %s %s\n", param, sens, direction, bar)
end

# ============================================================================
# DEMO 6: Heatmap Data
# ============================================================================

println("\n\n" * "="^80)
println("DEMO 6: Sensitivity Heatmap (Parameters × Outputs)")
println("-"^80)

heatmap = sensitivity_heatmap_data(coag_oat)

println("\nSensitivity matrix:")
println("─"^80)

# Print header
@printf("%-12s", "Parameter")
for out in heatmap.outputs
    @printf(" %12s", out)
end
println()
println("─"^(12 + 13*length(heatmap.outputs)))

# Print matrix
for (i, param) in enumerate(heatmap.parameters)
    @printf("%-12s", param)
    for j in 1:length(heatmap.outputs)
        @printf(" %+11.4f", heatmap.matrix[i, j])
    end
    println()
end

# ============================================================================
# Summary
# ============================================================================

println("\n\n" * "="^80)
println("SUMMARY: Sensitivity Analysis Recommendations")
println("="^80)

println("""
Method Selection Guidelines:

1. ONE-AT-A-TIME (OAT)
   ✓ Fast, simple interpretation
   ✓ Good for initial screening
   ✗ Local (only near nominal values)
   ✗ Misses interactions
   → Use for: Quick preliminary analysis, linear models

2. SOBOL INDICES
   ✓ Variance decomposition (most rigorous)
   ✓ Quantifies interactions (ST - S1)
   ✓ Global across full parameter space
   ✗ Expensive (N×(2p+2) evaluations)
   → Use for: Final analysis, regulatory submissions, nonlinear models

3. MORRIS SCREENING
   ✓ Efficient global screening
   ✓ Identifies important parameters quickly
   ✓ Detects nonlinearity (via σ)
   ✗ Less precise than Sobol
   → Use for: Large parameter sets, initial global screening

4. PRCC (Partial Rank Correlation)
   ✓ Handles monotonic nonlinearity
   ✓ Controls for other parameters
   ✓ Good for biological models
   ✗ Assumes monotonicity
   → Use for: Dose-response, biological systems

Typical Workflow:
1. OAT for initial parameter ranking
2. Morris screening to identify top 5-10 parameters globally
3. Sobol indices for rigorous variance decomposition of key parameters
4. PRCC for monotonic relationship quantification

Computational Cost (for p=10 parameters):
- OAT: 11 evaluations
- Morris (r=10): 110 evaluations
- PRCC (N=1000): 1,000 evaluations
- Sobol (N=1024): 22,528 evaluations
""")

println("="^80)
println("Demo complete!")
println("="^80)
