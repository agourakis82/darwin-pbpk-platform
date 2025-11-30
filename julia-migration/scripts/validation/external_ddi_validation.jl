#!/usr/bin/env julia
# =============================================================================
# EXTERNAL DDI VALIDATION STUDY
# =============================================================================
# Rigorous validation against independent clinical DDI data
# Following FDA/EMA guidance for PBPK model qualification
#
# SCIENTIFIC APPROACH:
# 1. Use DDI pairs NOT used for model development (external validation)
# 2. Calculate prediction intervals, not just point estimates
# 3. Apply Guest et al. criteria (within 2-fold, AFE, AAFE)
# 4. Identify systematic biases and model limitations
# 5. Compare vs published Simcyp/GastroPlus benchmarks
#
# Darwin PBPK Platform v2.10.0
# =============================================================================

using Statistics
using Printf

push!(LOAD_PATH, joinpath(@__DIR__, "../../src"))

println("=" ^ 70)
println("EXTERNAL DDI VALIDATION STUDY")
println("Following FDA/EMA PBPK Guidance")
println("=" ^ 70)
println()

# Include modules
include("../../src/DarwinPBPK/medlang/ddi_prediction.jl")
using .DDIPrediction

# =============================================================================
# EXTERNAL VALIDATION DATASET
# =============================================================================
# These DDI pairs were NOT used for model calibration
# Sources: FDA DDI guidance, published clinical studies, UW DIDB

const EXTERNAL_VALIDATION_SET = [
    # -------------------------------------------------------------------------
    # CYP3A4 INHIBITION - Strong inhibitors
    # -------------------------------------------------------------------------
    (
        perpetrator = :itraconazole,
        victim = :midazolam,
        observed_auc_ratio = 10.8,
        observed_range = (6.0, 18.0),
        route = :oral,
        reference = "Olkkola 1996",
        mechanism = :reversible,
        notes = "Strong CYP3A4 inhibitor"
    ),
    (
        perpetrator = :ketoconazole,
        victim = :triazolam,
        observed_auc_ratio = 22.0,
        observed_range = (15.0, 30.0),
        route = :oral,
        reference = "Varhe 1994",
        mechanism = :reversible,
        notes = "Strong CYP3A4 inhibitor"
    ),
    (
        perpetrator = :ritonavir,
        victim = :midazolam,
        observed_auc_ratio = 28.0,
        observed_range = (20.0, 40.0),
        route = :oral,
        reference = "Greenblatt 2003",
        mechanism = :mbi,
        notes = "Potent MBI"
    ),
    (
        perpetrator = :clarithromycin,
        victim = :midazolam,
        observed_auc_ratio = 6.3,  # Meta-analysis: Greenblatt 2015, RAUC 6.5±10.9
        observed_range = (4.0, 10.0),
        route = :oral,
        reference = "Greenblatt 2015 meta-analysis; PBPK modeling PMC6202474",
        mechanism = :mbi,
        notes = "Moderate MBI, oral midazolam"
    ),
    (
        perpetrator = :erythromycin,
        victim = :midazolam,
        observed_auc_ratio = 4.4,
        observed_range = (3.0, 6.0),
        route = :oral,
        reference = "Zimmermann 1996",
        mechanism = :mbi,
        notes = "Moderate MBI"
    ),
    (
        perpetrator = :diltiazem,
        victim = :midazolam,
        observed_auc_ratio = 3.7,
        observed_range = (2.5, 5.0),
        route = :oral,
        reference = "Backman 1994",
        mechanism = :mbi,
        notes = "Moderate inhibitor with MBI"
    ),
    (
        perpetrator = :verapamil,
        victim = :midazolam,
        observed_auc_ratio = 2.9,
        observed_range = (2.0, 4.0),
        route = :oral,
        reference = "Backman 1994",
        mechanism = :reversible,
        notes = "Moderate inhibitor"
    ),
    (
        perpetrator = :fluconazole,
        victim = :midazolam,
        observed_auc_ratio = 3.6,
        observed_range = (2.5, 5.0),
        route = :oral,
        reference = "Olkkola 1996",
        mechanism = :reversible,
        notes = "Moderate CYP3A4 inhibitor"
    ),

    # -------------------------------------------------------------------------
    # CYP3A4 INHIBITION - Statins (high first-pass)
    # -------------------------------------------------------------------------
    (
        perpetrator = :itraconazole,
        victim = :simvastatin,
        observed_auc_ratio = 19.0,
        observed_range = (10.0, 30.0),
        route = :oral,
        reference = "Neuvonen 1998",
        mechanism = :reversible,
        notes = "High first-pass drug"
    ),
    (
        perpetrator = :itraconazole,
        victim = :atorvastatin,
        observed_auc_ratio = 3.3,
        observed_range = (2.0, 5.0),
        route = :oral,
        reference = "Kantola 1998",
        mechanism = :reversible,
        notes = "OATP1B1 also involved"
    ),

    # -------------------------------------------------------------------------
    # CYP2D6 INHIBITION
    # -------------------------------------------------------------------------
    (
        perpetrator = :quinidine,
        victim = :dextromethorphan,
        observed_auc_ratio = 26.0,
        observed_range = (15.0, 40.0),
        route = :oral,
        reference = "Schadel 1995",
        mechanism = :reversible,
        notes = "Potent CYP2D6 inhibitor"
    ),
    (
        perpetrator = :paroxetine,
        victim = :dextromethorphan,
        observed_auc_ratio = 9.0,
        observed_range = (5.0, 15.0),
        route = :oral,
        reference = "Liston 2002",
        mechanism = :mbi,
        notes = "MBI component"
    ),
    (
        perpetrator = :fluoxetine,
        victim = :dextromethorphan,
        observed_auc_ratio = 8.0,
        observed_range = (4.0, 12.0),
        route = :oral,
        reference = "Liston 2002",
        mechanism = :reversible,
        notes = "Moderate-strong inhibitor"
    ),
    (
        perpetrator = :bupropion,
        victim = :dextromethorphan,
        observed_auc_ratio = 5.0,
        observed_range = (3.0, 8.0),
        route = :oral,
        reference = "Kotlyar 2005",
        mechanism = :reversible,
        notes = "Moderate inhibitor"
    ),
    (
        perpetrator = :quinidine,
        victim = :metoprolol,
        observed_auc_ratio = 3.2,
        observed_range = (2.0, 5.0),
        route = :oral,
        reference = "Leemann 1986",
        mechanism = :reversible,
        notes = "Metoprolol is less fm_2d6"
    ),

    # -------------------------------------------------------------------------
    # CYP1A2 INHIBITION
    # -------------------------------------------------------------------------
    (
        perpetrator = :fluvoxamine,
        victim = :theophylline,
        observed_auc_ratio = 2.8,
        observed_range = (2.0, 4.0),
        route = :oral,
        reference = "Rasmussen 1995",
        mechanism = :reversible,
        notes = "Strong CYP1A2 inhibitor"
    ),
    (
        perpetrator = :ciprofloxacin,
        victim = :theophylline,
        observed_auc_ratio = 1.8,
        observed_range = (1.3, 2.5),
        route = :oral,
        reference = "Wijnands 1986",
        mechanism = :reversible,
        notes = "Moderate inhibitor"
    ),
    (
        perpetrator = :fluvoxamine,
        victim = :caffeine,
        observed_auc_ratio = 5.0,
        observed_range = (3.0, 8.0),
        route = :oral,
        reference = "Jeppesen 1996",
        mechanism = :reversible,
        notes = "Caffeine is CYP1A2 probe"
    ),

    # -------------------------------------------------------------------------
    # CYP2C9 INHIBITION
    # -------------------------------------------------------------------------
    (
        perpetrator = :fluconazole,
        victim = :warfarin,
        observed_auc_ratio = 2.3,
        observed_range = (1.5, 3.0),
        route = :oral,
        reference = "Black 1996",
        mechanism = :reversible,
        notes = "S-warfarin only"
    ),
    (
        perpetrator = :amiodarone,
        victim = :warfarin,
        observed_auc_ratio = 1.5,
        observed_range = (1.2, 2.0),
        route = :oral,
        reference = "Heimark 1992",
        mechanism = :reversible,
        notes = "Weak inhibitor"
    ),

    # -------------------------------------------------------------------------
    # CYP2C8 INHIBITION
    # -------------------------------------------------------------------------
    (
        perpetrator = :gemfibrozil,
        victim = :repaglinide,
        observed_auc_ratio = 8.1,
        observed_range = (5.0, 12.0),
        route = :oral,
        reference = "Niemi 2003",
        mechanism = :mbi,
        notes = "CYP2C8 MBI + OATP1B1"
    ),
    (
        perpetrator = :gemfibrozil,
        victim = :rosiglitazone,
        observed_auc_ratio = 2.3,
        observed_range = (1.5, 3.0),
        route = :oral,
        reference = "Niemi 2003",
        mechanism = :mbi,
        notes = "CYP2C8 substrate"
    ),

    # -------------------------------------------------------------------------
    # CYP3A4 INDUCTION
    # -------------------------------------------------------------------------
    (
        perpetrator = :rifampin,
        victim = :midazolam,
        observed_auc_ratio = 0.04,
        observed_range = (0.02, 0.08),
        route = :oral,
        reference = "Backman 1996",
        mechanism = :induction,
        notes = "Strong inducer - 96% decrease"
    ),
    (
        perpetrator = :rifampin,
        victim = :simvastatin,
        observed_auc_ratio = 0.13,
        observed_range = (0.05, 0.20),
        route = :oral,
        reference = "Kyrklund 2000",
        mechanism = :induction,
        notes = "Strong inducer"
    ),
    (
        perpetrator = :carbamazepine,
        victim = :midazolam,
        observed_auc_ratio = 0.10,
        observed_range = (0.05, 0.20),
        route = :oral,
        reference = "Backman 1996",
        mechanism = :induction,
        notes = "Strong inducer"
    ),
    (
        perpetrator = :phenytoin,
        victim = :midazolam,
        observed_auc_ratio = 0.06,
        observed_range = (0.03, 0.12),
        route = :oral,
        reference = "Backman 1996",
        mechanism = :induction,
        notes = "Strong inducer"
    ),
]

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

"""
Drugs that require comprehensive (CYP + transporter) prediction.
"""
const DUAL_MECHANISM_PAIRS = Set([
    (:gemfibrozil, :repaglinide),  # CYP2C8 + OATP1B1
    (:cyclosporine, :rosuvastatin),  # CYP3A4 + OATP1B1/BCRP
    (:rifampin, :repaglinide),  # Induction + OATP1B1 inhibition (acute)
])

"""
Run prediction for a single DDI pair.
Uses comprehensive prediction for known dual-mechanism pairs.
"""
function predict_single_ddi(pair::NamedTuple)::NamedTuple
    # Check if this is a dual-mechanism pair
    if (pair.perpetrator, pair.victim) in DUAL_MECHANISM_PAIRS
        comp = predict_ddi_comprehensive(pair.perpetrator, pair.victim)
        predicted = comp.result
    else
        predicted = predict_ddi(pair.perpetrator, pair.victim)
    end

    return (
        perpetrator = pair.perpetrator,
        victim = pair.victim,
        observed = pair.observed_auc_ratio,
        observed_lo = pair.observed_range[1],
        observed_hi = pair.observed_range[2],
        predicted = predicted.auc_ratio,
        mechanism = pair.mechanism,
        reference = pair.reference,
        notes = pair.notes
    )
end

"""
Calculate fold error.
"""
function fold_error(predicted::Float64, observed::Float64)::Float64
    if predicted >= observed
        return predicted / observed
    else
        return observed / predicted
    end
end

"""
Check if prediction is within X-fold of observed.
"""
function within_fold(predicted::Float64, observed::Float64, fold::Float64)::Bool
    return (predicted / observed <= fold) && (observed / predicted <= fold)
end

"""
Calculate geometric mean fold error (GMFE / AFE).
"""
function calculate_afe(predicted::Vector{Float64}, observed::Vector{Float64})::Float64
    n = length(predicted)
    log_ratios = log10.(predicted ./ observed)
    return 10^(sum(log_ratios) / n)
end

"""
Calculate absolute average fold error (AAFE).
"""
function calculate_aafe(predicted::Vector{Float64}, observed::Vector{Float64})::Float64
    n = length(predicted)
    abs_log_ratios = abs.(log10.(predicted ./ observed))
    return 10^(sum(abs_log_ratios) / n)
end

"""
Calculate root mean square error of log-transformed values.
"""
function calculate_rmse_log(predicted::Vector{Float64}, observed::Vector{Float64})::Float64
    log_errors = log10.(predicted) .- log10.(observed)
    return sqrt(sum(log_errors.^2) / length(log_errors))
end

"""
Calculate correlation coefficient.
"""
function calculate_correlation(predicted::Vector{Float64}, observed::Vector{Float64})::Float64
    log_pred = log10.(predicted)
    log_obs = log10.(observed)

    mean_pred = mean(log_pred)
    mean_obs = mean(log_obs)

    num = sum((log_pred .- mean_pred) .* (log_obs .- mean_obs))
    denom = sqrt(sum((log_pred .- mean_pred).^2) * sum((log_obs .- mean_obs).^2))

    return num / denom
end

# =============================================================================
# RUN EXTERNAL VALIDATION
# =============================================================================

println("Running predictions on external validation set...")
println("N = $(length(EXTERNAL_VALIDATION_SET)) DDI pairs")
println()

results = []
for pair in EXTERNAL_VALIDATION_SET
    try
        result = predict_single_ddi(pair)
        push!(results, result)
    catch e
        println("  Warning: Failed to predict $(pair.perpetrator) + $(pair.victim): $e")
    end
end

println("Successfully predicted $(length(results)) / $(length(EXTERNAL_VALIDATION_SET)) pairs")
println()

# Extract vectors
observed = [r.observed for r in results]
predicted = [r.predicted for r in results]

# =============================================================================
# CALCULATE PERFORMANCE METRICS
# =============================================================================

println("=" ^ 70)
println("PERFORMANCE METRICS (Guest et al. criteria)")
println("=" ^ 70)
println()

# Within X-fold
within_1_5 = sum(within_fold.(predicted, observed, 1.5))
within_2_0 = sum(within_fold.(predicted, observed, 2.0))
within_3_0 = sum(within_fold.(predicted, observed, 3.0))

n = length(results)

println("ACCURACY:")
@printf("  Within 1.5-fold: %d/%d (%.1f%%)\n", within_1_5, n, 100*within_1_5/n)
@printf("  Within 2.0-fold: %d/%d (%.1f%%)\n", within_2_0, n, 100*within_2_0/n)
@printf("  Within 3.0-fold: %d/%d (%.1f%%)\n", within_3_0, n, 100*within_3_0/n)
println()

# Bias metrics
afe = calculate_afe(predicted, observed)
aafe = calculate_aafe(predicted, observed)
rmse_log = calculate_rmse_log(predicted, observed)
r = calculate_correlation(predicted, observed)

println("BIAS AND PRECISION:")
@printf("  AFE (Average Fold Error):     %.2f (ideal = 1.0)\n", afe)
@printf("  AAFE (Absolute AFE):          %.2f (ideal = 1.0, acceptable < 2.0)\n", aafe)
@printf("  RMSE (log scale):             %.2f\n", rmse_log)
@printf("  Correlation (r):              %.3f\n", r)
println()

# Regulatory acceptance
println("REGULATORY ACCEPTANCE (FDA/EMA criteria):")
println("  Within 2-fold: $(within_2_0/n >= 0.80 ? "PASS" : "FAIL") (need ≥80%)")
println("  AFE 0.5-2.0:   $(0.5 <= afe <= 2.0 ? "PASS" : "FAIL")")
println("  AAFE < 2.0:    $(aafe < 2.0 ? "PASS" : "FAIL")")
println()

# =============================================================================
# ANALYSIS BY MECHANISM
# =============================================================================

println("=" ^ 70)
println("ANALYSIS BY DDI MECHANISM")
println("=" ^ 70)
println()

for mech in [:reversible, :mbi, :induction]
    idx = findall(r -> r.mechanism == mech, results)
    if isempty(idx)
        continue
    end

    obs_mech = observed[idx]
    pred_mech = predicted[idx]
    n_mech = length(idx)

    within_2_mech = sum(within_fold.(pred_mech, obs_mech, 2.0))
    afe_mech = calculate_afe(pred_mech, obs_mech)
    aafe_mech = calculate_aafe(pred_mech, obs_mech)

    println("$(uppercase(string(mech))):")
    @printf("  N = %d\n", n_mech)
    @printf("  Within 2-fold: %d/%d (%.1f%%)\n", within_2_mech, n_mech, 100*within_2_mech/n_mech)
    @printf("  AFE:  %.2f\n", afe_mech)
    @printf("  AAFE: %.2f\n", aafe_mech)
    println()
end

# =============================================================================
# ANALYSIS BY ENZYME
# =============================================================================

println("=" ^ 70)
println("ANALYSIS BY ENZYME")
println("=" ^ 70)
println()

# Infer enzyme from perpetrator
function get_primary_enzyme(perp::Symbol)::Symbol
    cyp3a4_inhibitors = [:itraconazole, :ketoconazole, :ritonavir, :clarithromycin,
                         :erythromycin, :diltiazem, :verapamil, :fluconazole,
                         :rifampin, :carbamazepine, :phenytoin]
    cyp2d6_inhibitors = [:quinidine, :paroxetine, :fluoxetine, :bupropion]
    cyp1a2_inhibitors = [:fluvoxamine, :ciprofloxacin]
    cyp2c9_inhibitors = [:fluconazole, :amiodarone]
    cyp2c8_inhibitors = [:gemfibrozil]

    if perp in cyp3a4_inhibitors
        return :CYP3A4
    elseif perp in cyp2d6_inhibitors
        return :CYP2D6
    elseif perp in cyp1a2_inhibitors
        return :CYP1A2
    elseif perp in cyp2c9_inhibitors
        return :CYP2C9
    elseif perp in cyp2c8_inhibitors
        return :CYP2C8
    else
        return :unknown
    end
end

for enzyme in [:CYP3A4, :CYP2D6, :CYP1A2, :CYP2C9, :CYP2C8]
    idx = findall(r -> get_primary_enzyme(r.perpetrator) == enzyme, results)
    if isempty(idx)
        continue
    end

    obs_enz = observed[idx]
    pred_enz = predicted[idx]
    n_enz = length(idx)

    within_2_enz = sum(within_fold.(pred_enz, obs_enz, 2.0))
    afe_enz = calculate_afe(pred_enz, obs_enz)

    @printf("%-8s N=%2d  Within 2-fold: %2d/%2d (%5.1f%%)  AFE: %.2f\n",
            string(enzyme), n_enz, within_2_enz, n_enz, 100*within_2_enz/n_enz, afe_enz)
end
println()

# =============================================================================
# DETAILED RESULTS TABLE
# =============================================================================

println("=" ^ 70)
println("DETAILED RESULTS")
println("=" ^ 70)
println()

println("Perpetrator          Victim              Obs    Pred   Fold   Status")
println("-" ^ 70)

for r in results
    fe = fold_error(r.predicted, r.observed)
    status = within_fold(r.predicted, r.observed, 2.0) ? "OK" : "MISS"

    # Color-code by status (for terminal)
    @printf("%-20s %-18s %6.2f %6.2f %5.2f  %s\n",
            string(r.perpetrator), string(r.victim),
            r.observed, r.predicted, fe, status)
end
println()

# =============================================================================
# IDENTIFY SYSTEMATIC BIASES
# =============================================================================

println("=" ^ 70)
println("SYSTEMATIC BIAS ANALYSIS")
println("=" ^ 70)
println()

# Overpredictions
overpred = filter(r -> r.predicted > r.observed * 2.0, results)
println("OVERPREDICTIONS (predicted > 2× observed):")
if isempty(overpred)
    println("  None")
else
    for r in overpred
        @printf("  %s + %s: predicted %.1fx, observed %.1fx\n",
                string(r.perpetrator), string(r.victim), r.predicted, r.observed)
    end
end
println()

# Underpredictions
underpred = filter(r -> r.predicted < r.observed / 2.0, results)
println("UNDERPREDICTIONS (predicted < observed/2):")
if isempty(underpred)
    println("  None")
else
    for r in underpred
        @printf("  %s + %s: predicted %.1fx, observed %.1fx\n",
                string(r.perpetrator), string(r.victim), r.predicted, r.observed)
    end
end
println()

# =============================================================================
# COMPARISON VS PUBLISHED BENCHMARKS
# =============================================================================

println("=" ^ 70)
println("COMPARISON VS PUBLISHED BENCHMARKS")
println("=" ^ 70)
println()

println("Method                        Within 2-fold    AFE      AAFE")
println("-" ^ 60)
@printf("Darwin PBPK (this work)       %5.1f%%          %.2f     %.2f\n",
        100*within_2_0/n, afe, aafe)
println("Simcyp (typical)              75-85%           ~1.0     ~1.5")
println("GastroPlus (typical)          70-80%           ~1.1     ~1.6")
println("Static R-model (basic)        50-60%           ~0.8     ~2.0")
println()

# =============================================================================
# CONCLUSIONS
# =============================================================================

println("=" ^ 70)
println("CONCLUSIONS")
println("=" ^ 70)
println()

if within_2_0/n >= 0.80 && aafe < 2.0
    println("MODEL QUALIFIED for DDI prediction per FDA/EMA criteria")
    println()
    println("Key findings:")
    println("  - $(round(Int, 100*within_2_0/n))% of predictions within 2-fold of observed")
    println("  - AFE = $(round(afe, digits=2)) indicates $(afe > 1 ? "slight overprediction" : "slight underprediction") bias")
    println("  - AAFE = $(round(aafe, digits=2)) indicates good precision")
else
    println("MODEL REQUIRES REFINEMENT")
    println()
    println("Issues identified:")
    if within_2_0/n < 0.80
        println("  - Within 2-fold accuracy below 80% threshold")
    end
    if aafe >= 2.0
        println("  - AAFE >= 2.0 indicates poor precision")
    end
end
println()

println("Limitations:")
println("  - Transporter-mediated DDIs may require additional parameters")
println("  - MBI predictions depend on accurate kinact/KI values")
println("  - Induction onset timing not captured in static model")
println()
