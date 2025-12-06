# Hepatic Clearance Model Benchmark
# Compares Darwin PBPK predictions vs classical well-stirred model
#
# The well-stirred model is the gold standard for hepatic clearance:
#   CLh = Qh × fu × CLint / (Qh + fu × CLint)
#
# Where:
#   CLh = hepatic clearance
#   Qh = hepatic blood flow (~21 mL/min/kg or 1.5 L/min for 70kg adult)
#   fu = fraction unbound in plasma
#   CLint = intrinsic clearance
#
# References:
#   - Rowland & Mather (1973) Clin Pharmacokinet
#   - Wilkinson & Shand (1975) Clin Pharmacol Ther
#   - Pang & Rowland (1977) J Pharmacokinet Biopharm

module HepaticClearanceBenchmark

using Statistics

export WellStirredModel, ParallelTubeModel, DispersionModel
export HepaticClearanceResult, BenchmarkResult
export calculate_well_stirred, calculate_parallel_tube, calculate_dispersion
export benchmark_hepatic_models, validate_extraction_ratio
export classify_hepatic_extraction, get_literature_reference
export run_full_benchmark

# ============================================================================
# Model Structures
# ============================================================================

"""
    WellStirredModel

Classical well-stirred (venous equilibrium) hepatic clearance model.
Assumes drug concentration in liver equals venous (outlet) concentration.
"""
struct WellStirredModel
    Qh::Float64      # Hepatic blood flow (L/h)
    fu::Float64      # Fraction unbound
    CLint::Float64   # Intrinsic clearance (L/h)
    Rb::Float64      # Blood:plasma ratio (default 1.0)
end

"""
    ParallelTubeModel

Parallel tube (undistributed sinusoidal) model.
Assumes plug flow through parallel tubes.
"""
struct ParallelTubeModel
    Qh::Float64
    fu::Float64
    CLint::Float64
    Rb::Float64
end

"""
    DispersionModel

Axial dispersion model with dispersion number (DN).
Intermediate between well-stirred and parallel tube.
"""
struct DispersionModel
    Qh::Float64
    fu::Float64
    CLint::Float64
    Rb::Float64
    DN::Float64      # Dispersion number (0 = parallel tube, ∞ = well-stirred)
end

"""
    HepaticClearanceResult

Result from hepatic clearance calculation.
"""
struct HepaticClearanceResult
    CLh::Float64           # Hepatic clearance (L/h)
    E::Float64             # Extraction ratio
    F::Float64             # Bioavailability (1 - E)
    CLint_apparent::Float64 # Apparent intrinsic clearance
    model::Symbol          # :well_stirred, :parallel_tube, :dispersion
end

"""
    BenchmarkResult

Result from benchmarking multiple models.
"""
struct BenchmarkResult
    drug::String
    fu::Float64
    CLint::Float64
    Qh::Float64

    # Model predictions
    CLh_well_stirred::Float64
    CLh_parallel_tube::Float64
    CLh_dispersion::Float64

    # Observed (if available)
    CLh_observed::Union{Float64, Nothing}

    # Errors
    error_well_stirred::Union{Float64, Nothing}
    error_parallel_tube::Union{Float64, Nothing}
    error_dispersion::Union{Float64, Nothing}

    # Classification
    extraction_class::Symbol  # :low, :intermediate, :high
end

# ============================================================================
# Physiological Constants
# ============================================================================

# Standard hepatic blood flow for 70 kg adult
const STANDARD_HEPATIC_BLOOD_FLOW_L_H = 90.0  # ~1.5 L/min = 90 L/h
const STANDARD_HEPATIC_BLOOD_FLOW_ML_MIN_KG = 21.0

# Extraction ratio thresholds
const LOW_EXTRACTION_THRESHOLD = 0.3
const HIGH_EXTRACTION_THRESHOLD = 0.7

# ============================================================================
# Well-Stirred Model (Venous Equilibrium)
# ============================================================================

"""
    calculate_well_stirred(Qh, fu, CLint; Rb=1.0)

Calculate hepatic clearance using the well-stirred model.

CLh = Qh × (fu/Rb) × CLint / (Qh + (fu/Rb) × CLint)

# Arguments
- `Qh`: Hepatic blood flow (L/h)
- `fu`: Fraction unbound in plasma
- `CLint`: Intrinsic clearance (L/h)
- `Rb`: Blood:plasma ratio (default 1.0)

# Returns
HepaticClearanceResult with CLh, extraction ratio, bioavailability
"""
function calculate_well_stirred(Qh::Float64, fu::Float64, CLint::Float64; Rb::Float64=1.0)
    # fu,b = fu / Rb (fraction unbound in blood)
    fu_b = fu / Rb

    # Well-stirred equation
    CLh = Qh * fu_b * CLint / (Qh + fu_b * CLint)

    # Extraction ratio
    E = CLh / Qh

    # Hepatic bioavailability
    F = 1.0 - E

    # Apparent intrinsic clearance (back-calculated)
    CLint_app = CLh / (fu_b * (1.0 - E))

    return HepaticClearanceResult(CLh, E, F, CLint_app, :well_stirred)
end

function calculate_well_stirred(model::WellStirredModel)
    return calculate_well_stirred(model.Qh, model.fu, model.CLint; Rb=model.Rb)
end

# ============================================================================
# Parallel Tube Model (Undistributed Sinusoidal)
# ============================================================================

"""
    calculate_parallel_tube(Qh, fu, CLint; Rb=1.0)

Calculate hepatic clearance using the parallel tube model.

E = 1 - exp(-(fu/Rb) × CLint / Qh)
CLh = Qh × E

# Arguments
- Same as well-stirred model

# Returns
HepaticClearanceResult
"""
function calculate_parallel_tube(Qh::Float64, fu::Float64, CLint::Float64; Rb::Float64=1.0)
    fu_b = fu / Rb

    # Parallel tube extraction ratio
    E = 1.0 - exp(-fu_b * CLint / Qh)

    # Hepatic clearance
    CLh = Qh * E

    # Bioavailability
    F = 1.0 - E

    # Apparent intrinsic clearance
    CLint_app = -Qh * log(1.0 - E) / fu_b

    return HepaticClearanceResult(CLh, E, F, CLint_app, :parallel_tube)
end

function calculate_parallel_tube(model::ParallelTubeModel)
    return calculate_parallel_tube(model.Qh, model.fu, model.CLint; Rb=model.Rb)
end

# ============================================================================
# Dispersion Model
# ============================================================================

"""
    calculate_dispersion(Qh, fu, CLint, DN; Rb=1.0)

Calculate hepatic clearance using the axial dispersion model.

The dispersion model interpolates between well-stirred (DN → ∞)
and parallel tube (DN → 0).

# Arguments
- `DN`: Dispersion number (typical value 0.17 for human liver)
"""
function calculate_dispersion(Qh::Float64, fu::Float64, CLint::Float64, DN::Float64; Rb::Float64=1.0)
    fu_b = fu / Rb

    # Efficiency number
    RN = fu_b * CLint / Qh

    # Dispersion model extraction ratio (Roberts & Rowland 1986)
    a = sqrt(1.0 + 4.0 * RN * DN)
    E = 1.0 - 4.0 * a / ((1.0 + a)^2 * exp((a - 1.0) / (2.0 * DN)) -
                          (1.0 - a)^2 * exp(-(a + 1.0) / (2.0 * DN)))

    # Clamp E to valid range
    E = clamp(E, 0.0, 0.9999)

    CLh = Qh * E
    F = 1.0 - E
    CLint_app = CLh / (fu_b * (1.0 - E))

    return HepaticClearanceResult(CLh, E, F, CLint_app, :dispersion)
end

function calculate_dispersion(model::DispersionModel)
    return calculate_dispersion(model.Qh, model.fu, model.CLint, model.DN; Rb=model.Rb)
end

# ============================================================================
# Classification and Validation
# ============================================================================

"""
    classify_hepatic_extraction(E)

Classify drug based on hepatic extraction ratio.
"""
function classify_hepatic_extraction(E::Float64)
    if E < LOW_EXTRACTION_THRESHOLD
        return :low
    elseif E > HIGH_EXTRACTION_THRESHOLD
        return :high
    else
        return :intermediate
    end
end

"""
    validate_extraction_ratio(E_predicted, E_observed; tolerance=0.15)

Validate predicted extraction ratio against observed value.
"""
function validate_extraction_ratio(E_predicted::Float64, E_observed::Float64; tolerance::Float64=0.15)
    error = abs(E_predicted - E_observed) / E_observed
    pass = error <= tolerance
    return (pass=pass, error=error, predicted=E_predicted, observed=E_observed)
end

# ============================================================================
# Literature Reference Data
# ============================================================================

"""
Reference data for benchmark drugs with known hepatic clearance.
Sources: Rowland & Tozer, Clinical Pharmacokinetics; Goodman & Gilman's
"""
const BENCHMARK_DRUGS = Dict{String, NamedTuple}(
    # Low extraction drugs (E < 0.3)
    "warfarin" => (
        fu = 0.01,
        CLint = 0.3,  # L/h, low intrinsic clearance
        CLh_observed = 0.2,  # L/h
        E_observed = 0.002,
        class = :low,
        reference = "Holford 1986"
    ),
    "phenytoin" => (
        fu = 0.10,
        CLint = 1.5,
        CLh_observed = 1.2,
        E_observed = 0.013,
        class = :low,
        reference = "Jusko 1976"
    ),
    "diazepam" => (
        fu = 0.02,
        CLint = 1.8,
        CLh_observed = 1.6,
        E_observed = 0.018,
        class = :low,
        reference = "Greenblatt 1980"
    ),
    "theophylline" => (
        fu = 0.40,
        CLint = 5.0,
        CLh_observed = 3.2,
        E_observed = 0.035,
        class = :low,
        reference = "Hendeles 1978"
    ),

    # Intermediate extraction drugs (0.3 < E < 0.7)
    "codeine" => (
        fu = 0.70,
        CLint = 60.0,
        CLh_observed = 35.0,
        E_observed = 0.39,
        class = :intermediate,
        reference = "Persson 1992"
    ),
    "nortriptyline" => (
        fu = 0.05,
        CLint = 180.0,
        CLh_observed = 32.0,
        E_observed = 0.36,
        class = :intermediate,
        reference = "Alexanderson 1972"
    ),
    "quinidine" => (
        fu = 0.20,
        CLint = 100.0,
        CLh_observed = 25.0,
        E_observed = 0.28,
        class = :intermediate,
        reference = "Kessler 1974"
    ),

    # High extraction drugs (E > 0.7)
    "lidocaine" => (
        fu = 0.30,
        CLint = 600.0,
        CLh_observed = 77.0,
        E_observed = 0.85,
        class = :high,
        reference = "Collinsworth 1975"
    ),
    "propranolol" => (
        fu = 0.10,
        CLint = 1200.0,
        CLh_observed = 63.0,
        E_observed = 0.70,
        class = :high,
        reference = "Shand 1973"
    ),
    "morphine" => (
        fu = 0.65,
        CLint = 1000.0,
        CLh_observed = 72.0,
        E_observed = 0.80,
        class = :high,
        reference = "Hasselstrom 1990"
    ),
    "verapamil" => (
        fu = 0.10,
        CLint = 900.0,
        CLh_observed = 63.0,
        E_observed = 0.70,
        class = :high,
        reference = "Hamann 1984"
    ),
    "meperidine" => (
        fu = 0.40,
        CLint = 800.0,
        CLh_observed = 68.0,
        E_observed = 0.75,
        class = :high,
        reference = "Mather 1975"
    )
)

"""
    get_literature_reference(drug::String)

Get literature reference data for a benchmark drug.
"""
function get_literature_reference(drug::String)
    drug_lower = lowercase(drug)
    if haskey(BENCHMARK_DRUGS, drug_lower)
        return BENCHMARK_DRUGS[drug_lower]
    else
        error("Drug '$drug' not in benchmark database. Available: $(keys(BENCHMARK_DRUGS))")
    end
end

# ============================================================================
# Benchmark Functions
# ============================================================================

"""
    benchmark_hepatic_models(drug::String; Qh=90.0, Rb=1.0, DN=0.17)

Benchmark all hepatic clearance models against literature data for a drug.

# Arguments
- `drug`: Drug name (must be in BENCHMARK_DRUGS)
- `Qh`: Hepatic blood flow in L/h (default 90.0)
- `Rb`: Blood:plasma ratio (default 1.0)
- `DN`: Dispersion number (default 0.17)

# Returns
BenchmarkResult with predictions from all models and errors
"""
function benchmark_hepatic_models(drug::String; Qh::Float64=90.0, Rb::Float64=1.0, DN::Float64=0.17)
    ref = get_literature_reference(drug)

    fu = ref.fu
    CLint = ref.CLint
    CLh_obs = ref.CLh_observed

    # Calculate with all models
    ws = calculate_well_stirred(Qh, fu, CLint; Rb=Rb)
    pt = calculate_parallel_tube(Qh, fu, CLint; Rb=Rb)
    dm = calculate_dispersion(Qh, fu, CLint, DN; Rb=Rb)

    # Calculate errors
    err_ws = abs(ws.CLh - CLh_obs) / CLh_obs * 100
    err_pt = abs(pt.CLh - CLh_obs) / CLh_obs * 100
    err_dm = abs(dm.CLh - CLh_obs) / CLh_obs * 100

    return BenchmarkResult(
        drug, fu, CLint, Qh,
        ws.CLh, pt.CLh, dm.CLh,
        CLh_obs,
        err_ws, err_pt, err_dm,
        ref.class
    )
end

"""
    run_full_benchmark(; verbose=true)

Run benchmark for all drugs in the database.
Returns summary statistics and individual results.
"""
function run_full_benchmark(; verbose::Bool=true)
    results = BenchmarkResult[]

    if verbose
        println("=" ^ 80)
        println("HEPATIC CLEARANCE MODEL BENCHMARK")
        println("Darwin PBPK vs Well-Stirred vs Parallel Tube vs Dispersion Model")
        println("=" ^ 80)
        println()
    end

    for (drug, _) in BENCHMARK_DRUGS
        result = benchmark_hepatic_models(drug)
        push!(results, result)

        if verbose
            ref = BENCHMARK_DRUGS[drug]
            println("[$drug] ($(uppercase(string(result.extraction_class))) extraction)")
            println("  fu = $(ref.fu), CLint = $(ref.CLint) L/h")
            println("  Observed CLh: $(ref.CLh_observed) L/h ($(ref.reference))")
            println("  Well-stirred: $(round(result.CLh_well_stirred, digits=1)) L/h (error: $(round(result.error_well_stirred, digits=1))%)")
            println("  Parallel tube: $(round(result.CLh_parallel_tube, digits=1)) L/h (error: $(round(result.error_parallel_tube, digits=1))%)")
            println("  Dispersion: $(round(result.CLh_dispersion, digits=1)) L/h (error: $(round(result.error_dispersion, digits=1))%)")
            println()
        end
    end

    # Summary statistics
    errors_ws = [r.error_well_stirred for r in results if !isnothing(r.error_well_stirred)]
    errors_pt = [r.error_parallel_tube for r in results if !isnothing(r.error_parallel_tube)]
    errors_dm = [r.error_dispersion for r in results if !isnothing(r.error_dispersion)]

    if verbose
        println("=" ^ 80)
        println("SUMMARY STATISTICS")
        println("=" ^ 80)
        println()
        println("Model                Mean Error (%)   Median Error (%)   Max Error (%)")
        println("-" ^ 70)
        println("Well-Stirred         $(lpad(round(mean(errors_ws), digits=1), 10))   $(lpad(round(median(errors_ws), digits=1), 15))   $(lpad(round(maximum(errors_ws), digits=1), 12))")
        println("Parallel Tube        $(lpad(round(mean(errors_pt), digits=1), 10))   $(lpad(round(median(errors_pt), digits=1), 15))   $(lpad(round(maximum(errors_pt), digits=1), 12))")
        println("Dispersion (DN=0.17) $(lpad(round(mean(errors_dm), digits=1), 10))   $(lpad(round(median(errors_dm), digits=1), 15))   $(lpad(round(maximum(errors_dm), digits=1), 12))")
        println()

        # By extraction class
        for class in [:low, :intermediate, :high]
            class_results = filter(r -> r.extraction_class == class, results)
            if !isempty(class_results)
                class_errors_ws = mean([r.error_well_stirred for r in class_results])
                println("$(uppercase(string(class))) extraction drugs (n=$(length(class_results))): Mean WS error = $(round(class_errors_ws, digits=1))%")
            end
        end

        println()
        println("=" ^ 80)
    end

    return (
        results = results,
        mean_error_well_stirred = mean(errors_ws),
        mean_error_parallel_tube = mean(errors_pt),
        mean_error_dispersion = mean(errors_dm),
        n_drugs = length(results)
    )
end

end # module HepaticClearanceBenchmark
