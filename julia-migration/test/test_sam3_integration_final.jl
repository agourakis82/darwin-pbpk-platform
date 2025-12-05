#!/usr/bin/env julia
"""
SAM-3 + Julia Fractal Integration Test
======================================

Tests the complete pipeline:
1. Load SAM-3 masks exported from Python
2. Perform fractal dimension analysis
3. Generate morphological report
"""

# Add the image_analysis path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src", "DarwinPBPK", "image_analysis"))

include(joinpath(@__DIR__, "..", "src", "DarwinPBPK", "image_analysis", "sam3_integration.jl"))
using .SAM3Integration

function main()
    println("=" ^ 60)
    println("SAM-3 + Julia Fractal Integration Test")
    println("=" ^ 60)

    masks_dir = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", "results", "sam3_masks_julia")

    # Find first NPZ file
    npz_files = filter(f -> endswith(f, ".npz"), readdir(masks_dir))

    if isempty(npz_files)
        println("ERROR: No NPZ files found in $masks_dir")
        println("Run: python export_sam3_masks.py first")
        return
    end

    npz_path = joinpath(masks_dir, npz_files[1])

    # 1. Load masks
    println("\n1. Loading SAM-3 masks from: $(basename(npz_path))")
    mask_data = load_sam3_masks(npz_path)

    println("   - Loaded $(mask_data.n_cells) cells")
    println("   - Cell type: $(mask_data.cell_type)")
    println("   - Image shape: $(mask_data.image_shape)")
    println("   - Prompt used: $(mask_data.prompt_used)")

    # 2. Analyze fractal dimensions
    println("\n2. Analyzing fractal dimensions...")
    result = analyze_sam3_masks(mask_data)

    println("   - Combined mask Df: $(round(result.df_combined, digits=3)) (R2=$(round(result.r_squared_combined, digits=3)))")
    println("   - Edge mask Df: $(round(result.df_edges, digits=3)) (R2=$(round(result.r_squared_edges, digits=3)))")
    println("   - Cell distribution Df: $(round(result.df_distribution, digits=3))")
    println("   - Mean cell edge Df: $(round(result.mean_df_edge, digits=3)) +/- $(round(result.std_df_edge, digits=3))")
    println("   - Mean circularity: $(round(result.mean_circularity, digits=3))")

    # 3. Batch analysis
    println("\n3. Batch analysis of all exported images...")
    results = analyze_batch_sam3(masks_dir)
    println("   - Analyzed $(length(results)) images")

    total_cells = sum(r.n_cells for r in results)
    println("   - Total cells: $total_cells")

    # 4. Generate report
    println("\n4. Generating fractal report...")
    report = generate_fractal_report(results)

    summary = report["summary"]
    fractal = report["fractal_dimensions"]

    println("   - Total cells analyzed: $(summary["total_cells"])")
    println("   - Mean combined Df: $(round(fractal["combined_mask"]["mean"], digits=3))")
    println("   - Mean edge Df: $(round(fractal["edge_mask"]["mean"], digits=3))")
    println("   - Mean cell distribution Df: $(round(fractal["cell_distribution"]["mean"], digits=3))")

    println("\n" * "=" ^ 60)
    println("INTEGRATION TEST: SUCCESS!")
    println("=" ^ 60)

    println("\nInterpretation:")
    println(report["interpretation"])

    # Per-cell statistics
    println("\n" * "-" ^ 60)
    println("Per-Cell Fractal Metrics (first 5 cells):")
    println("-" ^ 60)

    for (i, cell) in enumerate(result.cell_metrics[1:min(5, length(result.cell_metrics))])
        println("Cell $(cell.cell_id): Df_edge=$(round(cell.df_edge, digits=2)), " *
                "Df_area=$(round(cell.df_area, digits=2)), " *
                "area=$(cell.area)px, " *
                "circularity=$(round(cell.circularity, digits=2))")
    end
end

main()
