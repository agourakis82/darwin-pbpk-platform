"""
Test SAM-3 Integration with Julia Fractal Analysis
"""

println("Testing SAM-3 Integration...")

# Load the module
include("../src/DarwinPBPK/image_analysis/sam3_integration.jl")
using .SAM3Integration

# Path to exported masks
mask_dir = joinpath(@__DIR__, "../../analysis/fractal_poc/results/sam3_masks/lymphocytes")
npz_files = filter(f -> endswith(f, ".npz"), readdir(mask_dir))

if isempty(npz_files)
    error("No NPZ files found in $mask_dir")
end

npz_file = joinpath(mask_dir, npz_files[1])

println("\n1. Loading SAM-3 masks from: $(basename(npz_file))")
mask_data = load_sam3_masks(npz_file)

println("\n2. Mask Data Summary:")
println("   - N cells: $(mask_data.n_cells)")
println("   - Cell type: $(mask_data.cell_type)")
println("   - Image shape: $(mask_data.image_shape)")
println("   - Prompt used: $(mask_data.prompt_used)")
println("   - Scores: $(mask_data.scores[1:min(3, length(mask_data.scores))])...")

println("\n3. Performing fractal analysis...")
result = analyze_sam3_masks(mask_data)

println("\n4. Fractal Analysis Results:")
println("   Combined mask Df: $(round(result.df_combined, digits=3))")
println("   Edge mask Df: $(round(result.df_edges, digits=3))")
println("   Distribution Df: $(round(result.df_distribution, digits=3))")
println("   Mean cell edge Df: $(round(result.mean_df_edge, digits=3)) +/- $(round(result.std_df_edge, digits=3))")
println("   Mean circularity: $(round(result.mean_circularity, digits=3))")

println("\n5. Per-cell metrics (first 5):")
for (i, cm) in enumerate(result.cell_metrics[1:min(5, length(result.cell_metrics))])
    df_e = isnan(cm.df_edge) ? "NaN" : string(round(cm.df_edge, digits=2))
    circ = isnan(cm.circularity) ? "NaN" : string(round(cm.circularity, digits=2))
    println("   Cell $i: Df_edge=$df_e, area=$(cm.area), circ=$circ")
end

# Batch analysis
println("\n6. Batch Analysis (all files):")
results = analyze_batch_sam3(mask_dir)
println("   Analyzed $(length(results)) images")

# Generate report
println("\n7. Generating Fractal Report...")
report = generate_fractal_report(results)

println("\n8. Report Summary:")
println("   Total cells: $(report["summary"]["total_cells"])")
println("   Mean Df (combined): $(round(report["fractal_dimensions"]["combined_mask"]["mean"], digits=3))")
println("   Mean Df (edges): $(round(report["fractal_dimensions"]["edge_mask"]["mean"], digits=3))")
println("   Mean circularity: $(round(report["morphology"]["mean_circularity"], digits=3))")

println("\n9. Interpretation:")
println(report["interpretation"])

println("\n" * "="^60)
println("SUCCESS! SAM-3 + Julia Fractal Integration Working!")
println("="^60)
