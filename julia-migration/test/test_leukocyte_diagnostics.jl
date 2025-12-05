#!/usr/bin/env julia
"""
Test Suite: LeukocyteDiagnostics Integration
=============================================

Tests the complete integrated pipeline:
1. SAM-3 mask loading
2. Fractal analysis
3. CTRW parameter estimation
4. ML classification
5. Drug response prediction
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src", "DarwinPBPK", "image_analysis"))

include(joinpath(@__DIR__, "..", "src", "DarwinPBPK", "image_analysis", "leukocyte_diagnostics.jl"))
using .LeukocyteDiagnostics

function test_normal_sample()
    println("\n" * "=" ^ 70)
    println("TEST 1: Normal WBC Sample Analysis")
    println("=" ^ 70)

    # Use neutrophil sample from ML classifier results
    masks_dir = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", "results", "ml_classifier", "masks", "neutrophils")
    npz_files = filter(f -> endswith(f, ".npz"), readdir(masks_dir))

    if isempty(npz_files)
        println("No NPZ files found. Skipping test.")
        return nothing
    end

    npz_path = joinpath(masks_dir, npz_files[1])
    println("Loading: $(basename(npz_path))")

    # Create profile
    profile = create_leukocyte_profile(npz_path)

    println("\n--- Morphological Features ---")
    println("  Cell type: $(profile.cell_type)")
    println("  N cells: $(profile.n_cells)")
    println("  Df combined: $(round(profile.df_combined, digits=3))")
    println("  Df edges: $(round(profile.df_edges, digits=3))")
    println("  Mean circularity: $(round(profile.mean_circularity, digits=3))")

    println("\n--- CTRW Parameters (Estimated) ---")
    println("  Beta (anomalous exponent): $(round(profile.beta, digits=3))")
    println("  Alpha (transit power-law): $(round(profile.alpha, digits=3))")
    println("  Tau scale: $(round(profile.tau_scale, digits=3))")

    # Classify
    result = classify_cells(profile)

    println("\n--- Diagnostic Result ---")
    println("  Predicted class: $(result.predicted_class)")
    println("  Confidence: $(round(result.confidence * 100, digits=1))%")
    println("  Morphology score: $(round(result.morphology_score, digits=3))")
    println("  Dynamics score: $(round(result.dynamics_score, digits=3))")
    println("  Overall score: $(round(result.overall_score, digits=3))")
    println("\n  Morphology: $(result.morphology_interpretation)")
    println("  Dynamics: $(result.dynamics_interpretation)")
    println("\n  RECOMMENDATION: $(result.clinical_recommendation)")

    return result
end

function test_leukemia_sample()
    println("\n" * "=" ^ 70)
    println("TEST 2: Leukemia Sample Analysis")
    println("=" ^ 70)

    # Use leukemia sample from ML classifier results
    masks_dir = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", "results", "ml_classifier", "masks", "leukemia_pre")
    npz_files = filter(f -> endswith(f, ".npz"), readdir(masks_dir))

    if isempty(npz_files)
        println("No NPZ files found. Skipping test.")
        return nothing
    end

    npz_path = joinpath(masks_dir, npz_files[1])
    println("Loading: $(basename(npz_path))")

    # Create profile
    profile = create_leukocyte_profile(npz_path)

    println("\n--- Morphological Features ---")
    println("  Cell type: $(profile.cell_type)")
    println("  N cells: $(profile.n_cells)")
    println("  Df combined: $(round(profile.df_combined, digits=3))")
    println("  Df edges: $(round(profile.df_edges, digits=3))")
    println("  Mean circularity: $(round(profile.mean_circularity, digits=3))")

    println("\n--- CTRW Parameters (Estimated) ---")
    println("  Beta (anomalous exponent): $(round(profile.beta, digits=3))")
    println("  Alpha (transit power-law): $(round(profile.alpha, digits=3))")
    println("  Tau scale: $(round(profile.tau_scale, digits=3))")

    # Classify
    result = classify_cells(profile)

    println("\n--- Diagnostic Result ---")
    println("  Predicted class: $(result.predicted_class)")
    println("  Confidence: $(round(result.confidence * 100, digits=1))%")
    println("  Morphology score: $(round(result.morphology_score, digits=3))")
    println("  Dynamics score: $(round(result.dynamics_score, digits=3))")
    println("  Overall score: $(round(result.overall_score, digits=3))")
    println("\n  Morphology: $(result.morphology_interpretation)")
    println("  Dynamics: $(result.dynamics_interpretation)")
    println("\n  RECOMMENDATION: $(result.clinical_recommendation)")

    return result
end

function test_ctrw_simulation()
    println("\n" * "=" ^ 70)
    println("TEST 3: CTRW Dynamics Simulation")
    println("=" ^ 70)

    # Load a sample
    masks_dir = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", "results", "ml_classifier", "masks", "lymphocytes")
    npz_files = filter(f -> endswith(f, ".npz"), readdir(masks_dir))

    if isempty(npz_files)
        println("No NPZ files found. Skipping test.")
        return nothing
    end

    npz_path = joinpath(masks_dir, npz_files[1])
    profile = create_leukocyte_profile(npz_path)

    println("Simulating CTRW dynamics for $(profile.n_cells) cells...")
    println("  Beta = $(round(profile.beta, digits=3))")

    # Simulate
    dynamics = simulate_cell_dynamics(profile, t_max=50.0, dt=0.5, n_particles=500)

    println("\n--- Dynamics Results ---")
    println("  Effective diffusion: $(round(dynamics.D_eff, digits=4))")
    println("  Fitted beta: $(round(dynamics.beta_fitted, digits=3))")
    println("  Mean residence time: $(round(dynamics.residence_time, digits=2)) time units")

    println("\n  MSD at t=1: $(round(dynamics.msd[3], digits=4))")
    println("  MSD at t=10: $(round(dynamics.msd[21], digits=4))")
    println("  MSD at t=50: $(round(dynamics.msd[end], digits=4))")

    # Check subdiffusive behavior
    msd_ratio = dynamics.msd[end] / dynamics.msd[21]
    expected_ratio = (50.0 / 10.0) ^ dynamics.beta_fitted
    println("\n  MSD ratio (t=50/t=10): $(round(msd_ratio, digits=2))")
    println("  Expected (t^beta): $(round(expected_ratio, digits=2))")

    return dynamics
end

function test_drug_response()
    println("\n" * "=" ^ 70)
    println("TEST 4: Drug Response Prediction")
    println("=" ^ 70)

    # Compare normal vs leukemia response
    masks_normal = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", "results", "ml_classifier", "masks", "lymphocytes")
    masks_leukemia = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", "results", "ml_classifier", "masks", "leukemia_pre")

    npz_normal = filter(f -> endswith(f, ".npz"), readdir(masks_normal))
    npz_leukemia = filter(f -> endswith(f, ".npz"), readdir(masks_leukemia))

    if isempty(npz_normal) || isempty(npz_leukemia)
        println("NPZ files not found. Skipping test.")
        return nothing
    end

    profile_normal = create_leukocyte_profile(joinpath(masks_normal, npz_normal[1]))
    profile_leukemia = create_leukocyte_profile(joinpath(masks_leukemia, npz_leukemia[1]))

    drug_params = Dict(
        "dose" => 100.0,
        "k_el" => 0.1,
    )

    println("Drug: Dose=100, k_el=0.1")

    println("\n--- Normal Cells ---")
    response_normal = predict_cell_behavior(profile_normal, drug_params, t_max=24.0)
    println("  AUC (traditional): $(round(response_normal["AUC_traditional"], digits=1))")
    println("  AUC (fractal): $(round(response_normal["AUC_fractal"], digits=1))")
    println("  AUC ratio: $(round(response_normal["AUC_ratio"], digits=3))")
    println("  Predicted survival factor: $(round(response_normal["predicted_survival_factor"], digits=3))")
    println("  $(response_normal["interpretation"])")

    println("\n--- Leukemia Cells ---")
    response_leukemia = predict_cell_behavior(profile_leukemia, drug_params, t_max=24.0)
    println("  AUC (traditional): $(round(response_leukemia["AUC_traditional"], digits=1))")
    println("  AUC (fractal): $(round(response_leukemia["AUC_fractal"], digits=1))")
    println("  AUC ratio: $(round(response_leukemia["AUC_ratio"], digits=3))")
    println("  Predicted survival factor: $(round(response_leukemia["predicted_survival_factor"], digits=3))")
    println("  $(response_leukemia["interpretation"])")

    # Compare
    println("\n--- Comparison ---")
    println("  Leukemia cells have $(round((1 - response_leukemia["AUC_ratio"]/response_normal["AUC_ratio"]) * 100, digits=1))% lower drug exposure")
    println("  Leukemia cells have $(round((response_leukemia["predicted_survival_factor"]/response_normal["predicted_survival_factor"] - 1) * 100, digits=1))% higher survival")

    return (normal=response_normal, leukemia=response_leukemia)
end

function test_batch_analysis()
    println("\n" * "=" ^ 70)
    println("TEST 5: Batch Analysis Report")
    println("=" ^ 70)

    # Analyze multiple cell types
    base_dir = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", "results", "ml_classifier", "masks")

    all_results = DiagnosticResult[]

    for cell_type in ["neutrophils", "lymphocytes", "monocytes", "leukemia_pre"]
        masks_dir = joinpath(base_dir, cell_type)
        if !isdir(masks_dir)
            continue
        end

        npz_files = filter(f -> endswith(f, ".npz"), readdir(masks_dir))

        for (i, npz_file) in enumerate(npz_files[1:min(5, length(npz_files))])  # First 5
            try
                result = diagnose_sample(joinpath(masks_dir, npz_file))
                push!(all_results, result)
            catch e
                @warn "Error: $e"
            end
        end

        println("Processed $(min(5, length(npz_files))) samples from $cell_type")
    end

    # Generate report
    report = generate_diagnostic_report(all_results)

    println("\n--- Batch Report ---")
    println("  Total samples: $(report["summary"]["total_samples"])")
    println("  Class distribution: $(report["summary"]["class_distribution"])")
    println("  High-risk cases: $(report["summary"]["high_risk_count"])")
    println("  Mean morphology score: $(round(report["morphology"]["mean_score"], digits=3))")
    println("  Mean dynamics score: $(round(report["dynamics"]["mean_score"], digits=3))")
    println("  Mean confidence: $(round(report["overall"]["mean_confidence"] * 100, digits=1))%")

    return report
end

function main()
    println("=" ^ 70)
    println("LEUKOCYTE DIAGNOSTICS INTEGRATION TEST")
    println("Combining SAM-3 Morphology + CTRW Dynamics + ML Classification")
    println("=" ^ 70)

    # Run all tests
    result_normal = test_normal_sample()
    result_leukemia = test_leukemia_sample()
    dynamics = test_ctrw_simulation()
    drug_response = test_drug_response()
    batch_report = test_batch_analysis()

    # Summary
    println("\n" * "=" ^ 70)
    println("INTEGRATION TEST SUMMARY")
    println("=" ^ 70)

    tests_passed = 0

    if result_normal !== nothing && result_normal.predicted_class in ["normal", "activated"]
        println("[PASS] Normal sample classified correctly")
        tests_passed += 1
    else
        println("[FAIL] Normal sample classification")
    end

    if result_leukemia !== nothing && result_leukemia.predicted_class == "leukemia"
        println("[PASS] Leukemia sample classified correctly")
        tests_passed += 1
    else
        println("[FAIL] Leukemia sample classification")
    end

    if dynamics !== nothing && dynamics.beta_fitted > 0 && dynamics.beta_fitted < 1
        println("[PASS] CTRW simulation shows subdiffusive behavior")
        tests_passed += 1
    else
        println("[FAIL] CTRW simulation")
    end

    if drug_response !== nothing && drug_response.leukemia["AUC_ratio"] != drug_response.normal["AUC_ratio"]
        println("[PASS] Drug response prediction shows altered PK in leukemia")
        tests_passed += 1
    else
        println("[FAIL] Drug response prediction")
    end

    if batch_report !== nothing && batch_report["summary"]["total_samples"] > 0
        println("[PASS] Batch analysis completed")
        tests_passed += 1
    else
        println("[FAIL] Batch analysis")
    end

    println("\n" * "-" ^ 70)
    println("Tests passed: $tests_passed / 5")
    println("=" ^ 70)

    return tests_passed == 5
end

# Run tests
success = main()
exit(success ? 0 : 1)
