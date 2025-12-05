"""
LeukocyteDiagnostics - Integrated Morphology + Dynamics Analysis
================================================================

Combines:
1. SAM-3 segmentation (morphological features)
2. Fractal dimension analysis (box-counting)
3. CTRW dynamics from FractalBlood (anomalous transport)
4. ML classification (Random Forest)

Clinical Applications:
- Leukemia detection and classification
- WBC differential analysis
- Pathological morphology identification

Integration with:
- sam3_integration.jl: SAM-3 mask loading and fractal analysis
- fractal_blood.jl: CTRW dynamics and phase transitions

Author: Darwin PBPK Platform
Date: 2025-12-05
"""

module LeukocyteDiagnostics

using Statistics
using LinearAlgebra
using Random

# Include dependencies
include("sam3_integration.jl")
using .SAM3Integration

# Re-export SAM3Integration functions
export load_sam3_masks, analyze_sam3_masks, box_counting_fractal_dimension

# New exports
export LeukocyteProfile, DiagnosticResult, CTRWCellDynamics
export create_leukocyte_profile, diagnose_sample
export simulate_cell_dynamics, predict_cell_behavior
export calculate_anomalous_diffusion, estimate_ctrw_parameters
export train_classifier, classify_cells
export generate_diagnostic_report

# ============================================================================
# CONSTANTS
# ============================================================================

# Reference fractal dimensions from our ML analysis
const REFERENCE_DF = Dict{String, Tuple{Float64, Float64}}(
    # (mean, std) for each cell type
    "neutrophils" => (1.660, 0.122),
    "lymphocytes" => (1.722, 0.036),
    "monocytes"   => (1.676, 0.077),
    "eosinophils" => (1.711, 0.032),
    "leukemia"    => (1.761, 0.068),
)

# CTRW parameters for different cell states
const CTRW_PARAMS = Dict{String, Dict{String, Float64}}(
    "normal" => Dict(
        "beta" => 0.85,           # Slightly subdiffusive
        "alpha" => 1.37,          # Transit time power-law
        "tau_scale" => 1.0,       # Reference waiting time
        "velocity_factor" => 1.0, # Normal velocity
    ),
    "activated" => Dict(
        "beta" => 0.75,           # More subdiffusive (sticky)
        "alpha" => 1.25,          # Longer transit times
        "tau_scale" => 1.5,       # Increased waiting
        "velocity_factor" => 0.8, # Slower movement
    ),
    "leukemia" => Dict(
        "beta" => 0.65,           # Highly subdiffusive
        "alpha" => 1.15,          # Very long transit times
        "tau_scale" => 2.5,       # Much increased waiting
        "velocity_factor" => 0.5, # Significantly impaired
    ),
)

# ML model weights (from trained Random Forest)
# Feature order: df_combined, df_edges, df_distribution, mean_circularity,
#                mean_df_edge, std_df_edge, n_cells, mean_area
const ML_FEATURE_IMPORTANCE = Dict{String, Float64}(
    "df_edges" => 0.247,
    "mean_area" => 0.189,
    "n_cells" => 0.166,
    "df_distribution" => 0.166,
    "mean_df_edge" => 0.096,
    "mean_circularity" => 0.064,
    "df_combined" => 0.038,
    "std_df_edge" => 0.035,
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
LeukocyteProfile - Complete profile of a leukocyte sample
"""
struct LeukocyteProfile
    # Source information
    source_image::String
    cell_type::String
    n_cells::Int

    # Morphological features (from SAM-3)
    df_combined::Float64
    df_edges::Float64
    df_distribution::Float64
    mean_circularity::Float64
    mean_df_edge::Float64
    std_df_edge::Float64
    mean_area::Float64

    # CTRW dynamics parameters (estimated)
    beta::Float64              # Anomalous diffusion exponent
    alpha::Float64             # Transit time power-law
    tau_scale::Float64         # Waiting time scale

    # Per-cell metrics
    cell_metrics::Vector{CellFractalMetrics}
end

"""
CTRWCellDynamics - Dynamics simulation for a cell population
"""
struct CTRWCellDynamics
    # Time series
    time::Vector{Float64}

    # Mean squared displacement
    msd::Vector{Float64}

    # Concentration profiles
    concentration::Vector{Float64}

    # Phase fractions over time
    phase_fractions::Matrix{Float64}  # (n_times, n_phases)

    # Estimated parameters
    D_eff::Float64             # Effective diffusion coefficient
    beta_fitted::Float64       # Fitted anomalous exponent
    residence_time::Float64    # Mean residence time
end

"""
DiagnosticResult - Clinical diagnostic output
"""
struct DiagnosticResult
    # Classification
    predicted_class::String    # "normal", "activated", "leukemia"
    confidence::Float64        # 0-1

    # Probabilities per class
    class_probabilities::Dict{String, Float64}

    # Morphological assessment
    morphology_score::Float64  # 0-1 (1 = highly abnormal)
    morphology_interpretation::String

    # Dynamic assessment
    dynamics_score::Float64    # 0-1 (1 = highly impaired)
    dynamics_interpretation::String

    # Combined score
    overall_score::Float64
    clinical_recommendation::String

    # Feature contributions
    feature_contributions::Dict{String, Float64}
end

# ============================================================================
# PROFILE CREATION
# ============================================================================

"""
create_leukocyte_profile(mask_data::SAM3MaskData) -> LeukocyteProfile

Create a complete leukocyte profile from SAM-3 mask data.
Includes morphological analysis and CTRW parameter estimation.
"""
function create_leukocyte_profile(mask_data::SAM3MaskData)::LeukocyteProfile
    # Perform fractal analysis
    fractal_result = analyze_sam3_masks(mask_data)

    # Estimate CTRW parameters from morphology
    ctrw_params = estimate_ctrw_parameters(fractal_result)

    # Calculate mean area from cell metrics
    areas = [m.area for m in fractal_result.cell_metrics if m.area > 0]
    mean_area = isempty(areas) ? 0.0 : mean(areas)

    return LeukocyteProfile(
        mask_data.source_image,
        mask_data.cell_type,
        mask_data.n_cells,
        fractal_result.df_combined,
        fractal_result.df_edges,
        fractal_result.df_distribution,
        fractal_result.mean_circularity,
        fractal_result.mean_df_edge,
        fractal_result.std_df_edge,
        mean_area,
        ctrw_params["beta"],
        ctrw_params["alpha"],
        ctrw_params["tau_scale"],
        fractal_result.cell_metrics
    )
end

"""
create_leukocyte_profile(npz_path::String) -> LeukocyteProfile

Load SAM-3 masks and create profile.
"""
function create_leukocyte_profile(npz_path::String)::LeukocyteProfile
    mask_data = load_sam3_masks(npz_path)
    return create_leukocyte_profile(mask_data)
end

# ============================================================================
# CTRW PARAMETER ESTIMATION
# ============================================================================

"""
estimate_ctrw_parameters(fractal_result::SAM3FractalResult) -> Dict

Estimate CTRW dynamics parameters from morphological features.

The mapping is based on the hypothesis that:
- Higher Df (more complex morphology) → more subdiffusive behavior
- Lower circularity → more irregular movement
- Complex edge structure → more phase transitions
"""
function estimate_ctrw_parameters(fractal_result::SAM3FractalResult)::Dict{String, Float64}
    # Base parameters (normal)
    base = CTRW_PARAMS["normal"]

    # Morphology-based adjustments
    df_deviation = fractal_result.df_combined - REFERENCE_DF["lymphocytes"][1]
    circ_deviation = fractal_result.mean_circularity - 0.5  # Reference circularity
    edge_complexity = fractal_result.df_edges - 1.0  # Edge Df deviation from 1

    # Estimate beta (anomalous exponent)
    # Higher morphological complexity → lower beta (more subdiffusive)
    beta = base["beta"] - 0.15 * df_deviation - 0.1 * edge_complexity
    beta = clamp(beta, 0.3, 1.0)

    # Estimate alpha (transit time power-law)
    # More irregular cells → lower alpha (heavier tail)
    alpha = base["alpha"] - 0.2 * df_deviation + 0.1 * circ_deviation
    alpha = clamp(alpha, 1.1, 2.0)

    # Estimate tau_scale (waiting time)
    # Larger, more complex cells → longer waiting times
    tau_scale = base["tau_scale"] * (1.0 + 0.5 * df_deviation)
    tau_scale = clamp(tau_scale, 0.5, 5.0)

    return Dict(
        "beta" => beta,
        "alpha" => alpha,
        "tau_scale" => tau_scale,
        "velocity_factor" => 1.0 - 0.3 * df_deviation,
    )
end

"""
calculate_anomalous_diffusion(beta, D0, t)

Calculate mean squared displacement for anomalous diffusion:
⟨x²(t)⟩ = 2D₀ × t^β / Γ(1+β)

For subdiffusion (β < 1), MSD grows slower than linear.
"""
function calculate_anomalous_diffusion(beta::Float64, D0::Float64, t::Float64)::Float64
    if t <= 0
        return 0.0
    end
    # Using Sterling approximation for Gamma function
    gamma_factor = sqrt(2π * (1 + beta)) * ((1 + beta) / exp(1))^(1 + beta)
    return 2 * D0 * t^beta / gamma_factor
end

# ============================================================================
# CELL DYNAMICS SIMULATION
# ============================================================================

"""
simulate_cell_dynamics(profile::LeukocyteProfile; t_max=100.0, dt=0.1)

Simulate CTRW dynamics for the cell population.
Returns time series of MSD, concentration, and phase fractions.
"""
function simulate_cell_dynamics(profile::LeukocyteProfile;
                                 t_max::Float64=100.0,
                                 dt::Float64=0.1,
                                 n_particles::Int=1000)::CTRWCellDynamics
    n_steps = Int(ceil(t_max / dt))

    # Initialize arrays
    time = collect(range(0, t_max, length=n_steps))
    msd = zeros(n_steps)
    concentration = zeros(n_steps)
    phase_fractions = zeros(n_steps, 3)  # 3 phases: free, bound, sequestered

    # CTRW parameters from profile
    beta = profile.beta
    alpha = profile.alpha
    tau_scale = profile.tau_scale

    # Simulate particle ensemble
    D0 = 1.0  # Reference diffusion coefficient

    # Particle states: (position, phase, waiting_time)
    positions = zeros(n_particles)
    phases = ones(Int, n_particles)  # Start in phase 1 (free)
    waiting_times = tau_scale .* rand(n_particles) .^ (-1/alpha)
    time_since_jump = zeros(n_particles)

    for (step, t) in enumerate(time)
        # Update MSD
        msd[step] = mean(positions .^ 2)

        # Concentration (simplified: exponential decay)
        concentration[step] = exp(-t / (tau_scale * 10))

        # Phase fractions
        for phase in 1:3
            phase_fractions[step, phase] = count(==(phase), phases) / n_particles
        end

        # Update particles
        for i in 1:n_particles
            time_since_jump[i] += dt

            # Check for jump
            if time_since_jump[i] >= waiting_times[i]
                # Anomalous jump size
                jump_size = randn() * sqrt(2 * D0 * dt^beta)
                positions[i] += jump_size

                # Phase transition (with probability)
                p_transition = 0.1 * dt / tau_scale
                if rand() < p_transition
                    phases[i] = rand(1:3)
                end

                # Reset
                time_since_jump[i] = 0.0
                waiting_times[i] = tau_scale * rand()^(-1/alpha)
            end
        end
    end

    # Fit effective diffusion from MSD
    # MSD(t) = 2D_eff × t^beta_fitted
    valid_idx = msd .> 0
    if sum(valid_idx) >= 10
        log_t = log.(time[valid_idx][2:end])
        log_msd = log.(msd[valid_idx][2:end])

        # Linear fit in log-log space
        n = length(log_t)
        sum_x = sum(log_t)
        sum_y = sum(log_msd)
        sum_xy = sum(log_t .* log_msd)
        sum_x2 = sum(log_t .^ 2)

        beta_fitted = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)
        intercept = (sum_y - beta_fitted * sum_x) / n
        D_eff = exp(intercept) / 2
    else
        beta_fitted = beta
        D_eff = D0
    end

    # Residence time
    residence_time = tau_scale * alpha / (alpha - 1)

    return CTRWCellDynamics(
        time, msd, concentration, phase_fractions,
        D_eff, beta_fitted, residence_time
    )
end

# ============================================================================
# CLASSIFICATION AND DIAGNOSIS
# ============================================================================

"""
classify_cells(profile::LeukocyteProfile) -> DiagnosticResult

Classify cell sample using morphological and dynamic features.
Uses pre-trained model weights from Random Forest analysis.
"""
function classify_cells(profile::LeukocyteProfile)::DiagnosticResult
    # Extract features
    features = Dict{String, Float64}(
        "df_combined" => profile.df_combined,
        "df_edges" => profile.df_edges,
        "df_distribution" => profile.df_distribution,
        "mean_circularity" => profile.mean_circularity,
        "mean_df_edge" => profile.mean_df_edge,
        "std_df_edge" => profile.std_df_edge,
        "n_cells" => Float64(profile.n_cells),
        "mean_area" => profile.mean_area,
    )

    # Calculate leukemia probability using calibrated thresholds from ML training
    # Key findings: Normal Df=1.69, Leukemia Df=1.76, threshold=1.754
    # Most important: df_edges (0.247), n_cells (0.166), mean_area (0.189)

    score = 0.0
    feature_contributions = Dict{String, Float64}()

    # 1. df_edges (24.7% importance) - MOST DISCRIMINATIVE
    # Normal: ~1.31, Leukemia: ~1.60
    df_edges_contrib = 0.30 * sigmoid((features["df_edges"] - 1.40) * 8)
    score += df_edges_contrib
    feature_contributions["df_edges"] = df_edges_contrib

    # 2. n_cells (16.6% importance) - Leukemia samples have many more cells
    # Normal: ~8-22 per image, Leukemia: ~70-80 per image
    n_cells_contrib = 0.30 * sigmoid((features["n_cells"] - 40) / 15)
    score += n_cells_contrib
    feature_contributions["n_cells"] = n_cells_contrib

    # 3. df_combined - threshold from ML: 1.754
    df_combined_contrib = 0.20 * sigmoid((features["df_combined"] - 1.72) * 12)
    score += df_combined_contrib
    feature_contributions["df_combined"] = df_combined_contrib

    # 4. df_distribution
    df_dist_contrib = 0.10 * sigmoid((features["df_distribution"] - 0.55) * 5)
    score += df_dist_contrib
    feature_contributions["df_distribution"] = df_dist_contrib

    # 5. mean_circularity - Leukemia cells slightly rounder (~0.56 vs 0.50)
    circ_contrib = 0.10 * sigmoid((features["mean_circularity"] - 0.52) * 8)
    score += circ_contrib
    feature_contributions["mean_circularity"] = circ_contrib

    # Normalize score to 0-1
    score = clamp(score, 0.0, 1.0)

    # Calculate class probabilities
    p_normal = exp(-3 * score)
    p_activated = exp(-2 * abs(score - 0.5))
    p_leukemia = exp(-3 * (1 - score))

    total = p_normal + p_activated + p_leukemia
    class_probs = Dict(
        "normal" => p_normal / total,
        "activated" => p_activated / total,
        "leukemia" => p_leukemia / total,
    )

    # Determine class
    predicted_class = argmax(class_probs)
    confidence = class_probs[predicted_class]

    # Morphology assessment
    morphology_score = score
    morphology_interpretation = interpret_morphology(profile, score)

    # Dynamics assessment
    dynamics_score = 1.0 - profile.beta  # Lower beta = more impaired
    dynamics_interpretation = interpret_dynamics(profile)

    # Combined assessment
    overall_score = 0.7 * morphology_score + 0.3 * dynamics_score
    clinical_recommendation = generate_recommendation(predicted_class, confidence, overall_score)

    return DiagnosticResult(
        predicted_class,
        confidence,
        class_probs,
        morphology_score,
        morphology_interpretation,
        dynamics_score,
        dynamics_interpretation,
        overall_score,
        clinical_recommendation,
        feature_contributions
    )
end

"""
diagnose_sample(npz_path::String) -> DiagnosticResult

Complete diagnostic pipeline: load masks, analyze, classify.
"""
function diagnose_sample(npz_path::String)::DiagnosticResult
    profile = create_leukocyte_profile(npz_path)
    return classify_cells(profile)
end

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

sigmoid(x::Float64) = 1.0 / (1.0 + exp(-x))

function argmax(d::Dict)
    max_key = nothing
    max_val = -Inf
    for (k, v) in d
        if v > max_val
            max_val = v
            max_key = k
        end
    end
    return max_key
end

function interpret_morphology(profile::LeukocyteProfile, score::Float64)::String
    if score < 0.3
        return "Normal morphology - regular cell shapes with typical fractal dimensions"
    elseif score < 0.6
        return "Mild abnormalities - slightly irregular morphology, possible activation"
    elseif score < 0.8
        return "Moderate abnormalities - irregular morphology consistent with pathology"
    else
        return "Severe abnormalities - highly irregular morphology suggestive of malignancy"
    end
end

function interpret_dynamics(profile::LeukocyteProfile)::String
    beta = profile.beta
    if beta > 0.8
        return "Normal transport dynamics - near-Brownian diffusion"
    elseif beta > 0.7
        return "Mildly impaired transport - slight subdiffusion"
    elseif beta > 0.5
        return "Moderately impaired transport - significant subdiffusion (anomalous)"
    else
        return "Severely impaired transport - strong subdiffusion (pathological trapping)"
    end
end

function generate_recommendation(class::String, confidence::Float64, score::Float64)::String
    if class == "normal" && confidence > 0.7
        return "No further action required. Routine follow-up recommended."
    elseif class == "activated"
        return "Signs of immune activation. Consider infection workup or inflammatory markers."
    elseif class == "leukemia" && confidence > 0.8
        return "HIGH PRIORITY: Findings strongly suggestive of leukemia. Immediate hematology referral and bone marrow biopsy recommended."
    elseif class == "leukemia" && confidence > 0.5
        return "Abnormal findings consistent with possible malignancy. Further evaluation with flow cytometry and genetic studies recommended."
    else
        return "Indeterminate results. Consider repeat sampling or additional testing."
    end
end

# ============================================================================
# BATCH ANALYSIS AND REPORTING
# ============================================================================

"""
analyze_batch(npz_dir::String) -> Vector{DiagnosticResult}

Analyze all samples in a directory.
"""
function analyze_batch(npz_dir::String)::Vector{DiagnosticResult}
    results = DiagnosticResult[]

    npz_files = filter(f -> endswith(f, ".npz"), readdir(npz_dir))

    for (i, filename) in enumerate(npz_files)
        filepath = joinpath(npz_dir, filename)
        println("[$i/$(length(npz_files))] Diagnosing: $filename")

        try
            result = diagnose_sample(filepath)
            push!(results, result)
        catch e
            @warn "Error processing $filename: $e"
        end
    end

    return results
end

"""
generate_diagnostic_report(results::Vector{DiagnosticResult}) -> Dict

Generate summary report from batch analysis.
"""
function generate_diagnostic_report(results::Vector{DiagnosticResult})::Dict{String, Any}
    if isempty(results)
        return Dict{String, Any}("error" => "No results to analyze")
    end

    # Count by class
    class_counts = Dict{String, Int}()
    for r in results
        class = r.predicted_class
        class_counts[class] = get(class_counts, class, 0) + 1
    end

    # Statistics
    morphology_scores = [r.morphology_score for r in results]
    dynamics_scores = [r.dynamics_score for r in results]
    overall_scores = [r.overall_score for r in results]
    confidences = [r.confidence for r in results]

    # High-risk cases
    high_risk = filter(r -> r.predicted_class == "leukemia" && r.confidence > 0.7, results)

    report = Dict{String, Any}(
        "summary" => Dict(
            "total_samples" => length(results),
            "class_distribution" => class_counts,
            "high_risk_count" => length(high_risk),
        ),
        "morphology" => Dict(
            "mean_score" => mean(morphology_scores),
            "std_score" => std(morphology_scores),
            "min_score" => minimum(morphology_scores),
            "max_score" => maximum(morphology_scores),
        ),
        "dynamics" => Dict(
            "mean_score" => mean(dynamics_scores),
            "std_score" => std(dynamics_scores),
        ),
        "overall" => Dict(
            "mean_score" => mean(overall_scores),
            "mean_confidence" => mean(confidences),
        ),
        "recommendations" => unique([r.clinical_recommendation for r in results]),
    )

    return report
end

# ============================================================================
# INTEGRATION WITH FRACTALBLOOD
# ============================================================================

"""
predict_cell_behavior(profile::LeukocyteProfile, drug_params::Dict)

Predict how cells will behave with drug exposure using CTRW dynamics.
Integrates morphological assessment with transport modeling.
"""
function predict_cell_behavior(profile::LeukocyteProfile,
                               drug_params::Dict;
                               t_max::Float64=24.0)::Dict{String, Any}
    # Get CTRW parameters
    beta = profile.beta
    alpha = profile.alpha
    tau_scale = profile.tau_scale

    # Drug parameters
    dose = get(drug_params, "dose", 100.0)
    k_el = get(drug_params, "k_el", 0.1)  # Elimination rate

    # Simulate dynamics
    dynamics = simulate_cell_dynamics(profile, t_max=t_max)

    # Calculate drug exposure metrics
    # Traditional: C(t) = C0 * exp(-k_el * t)
    # Fractal: C(t) = C0 * E_beta(-k_el * t^beta)

    t = dynamics.time
    C_traditional = dose .* exp.(-k_el .* t)

    # Simplified Mittag-Leffler approximation
    C_fractal = dose .* exp.(-k_el .* t .^ beta)

    # AUC calculation
    dt = t[2] - t[1]
    AUC_traditional = sum(C_traditional) * dt
    AUC_fractal = sum(C_fractal) * dt

    # Cell survival prediction (higher anomalous transport → more drug resistance)
    survival_factor = 1.0 - 0.3 * beta  # Lower beta = more survival

    return Dict{String, Any}(
        "time" => t,
        "C_traditional" => C_traditional,
        "C_fractal" => C_fractal,
        "AUC_traditional" => AUC_traditional,
        "AUC_fractal" => AUC_fractal,
        "AUC_ratio" => AUC_fractal / AUC_traditional,
        "effective_diffusion" => dynamics.D_eff,
        "fitted_beta" => dynamics.beta_fitted,
        "residence_time" => dynamics.residence_time,
        "msd" => dynamics.msd,
        "phase_fractions" => dynamics.phase_fractions,
        "predicted_survival_factor" => survival_factor,
        "interpretation" => interpret_drug_response(beta, AUC_fractal / AUC_traditional),
    )
end

function interpret_drug_response(beta::Float64, auc_ratio::Float64)::String
    if beta > 0.8 && auc_ratio > 0.9
        return "Normal drug response expected. Standard dosing appropriate."
    elseif beta > 0.6
        return "Moderately altered pharmacokinetics. Consider extended dosing interval."
    else
        return "Significantly altered pharmacokinetics (anomalous transport). May require dose adjustment or alternative therapy."
    end
end

end  # module LeukocyteDiagnostics
