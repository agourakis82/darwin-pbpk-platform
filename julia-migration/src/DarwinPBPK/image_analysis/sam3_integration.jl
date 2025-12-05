"""
SAM-3 Integration for Julia Fractal Analysis
=============================================

Reads SAM-3 segmentation masks exported from Python and performs
fractal dimension analysis on the cell morphology.

Integration with:
- export_sam3_masks.py: Exports masks to NPZ format
- LeukocyteFractalAnalysis: Box-counting fractal dimension

Author: Darwin PBPK Platform
Date: 2025-12-04
"""

module SAM3Integration

using NPZ
using JSON3
using Statistics
using LinearAlgebra

# Box-counting algorithm (standalone, no external image dependencies)
# Based on LeukocyteFractalAnalysis but without Images.jl dependency

export SAM3MaskData, SAM3FractalResult, CellFractalMetrics
export box_counting_fractal_dimension

# ============================================================================
# BOX-COUNTING ALGORITHM (Standalone)
# ============================================================================

"""
box_counting_fractal_dimension(binary_image; min_box_size=2, max_box_size=nothing, n_sizes=10)

Perform box-counting algorithm to calculate fractal dimension.
Returns: (df, r², box_sizes, box_counts)
"""
function box_counting_fractal_dimension(
    binary_image::BitMatrix;
    min_box_size::Int=2,
    max_box_size::Union{Int, Nothing}=nothing,
    n_sizes::Int=10
)::Tuple{Float64, Float64, Vector{Int}, Vector{Int}}

    height, width = size(binary_image)

    if !any(binary_image)
        return (NaN, 0.0, Int[], Int[])
    end

    if max_box_size === nothing
        max_box_size = min(height, width) ÷ 4
    end
    max_box_size = max(max_box_size, min_box_size + 1)

    log_min = log10(min_box_size)
    log_max = log10(max_box_size)
    log_sizes = range(log_min, log_max, length=n_sizes)
    box_sizes = unique(round.(Int, 10.0.^log_sizes))

    box_counts = Int[]

    for box_size in box_sizes
        count = 0
        for i in 1:box_size:height
            for j in 1:box_size:width
                i_end = min(i + box_size - 1, height)
                j_end = min(j + box_size - 1, width)
                box = binary_image[i:i_end, j:j_end]
                if any(box)
                    count += 1
                end
            end
        end
        push!(box_counts, count)
    end

    # Filter zero counts
    valid_idx = box_counts .> 0
    if sum(valid_idx) < 2
        return (NaN, 0.0, box_sizes, box_counts)
    end

    valid_sizes = box_sizes[valid_idx]
    valid_counts = box_counts[valid_idx]

    log_inv_sizes = log.(1.0 ./ valid_sizes)
    log_counts = log.(Float64.(valid_counts))

    n = length(valid_sizes)
    sum_x = sum(log_inv_sizes)
    sum_y = sum(log_counts)
    sum_xy = sum(log_inv_sizes .* log_counts)
    sum_x2 = sum(log_inv_sizes .^ 2)

    denominator = n * sum_x2 - sum_x^2
    if abs(denominator) < 1e-10
        return (NaN, 0.0, box_sizes, box_counts)
    end

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    fractal_dim = slope

    y_pred = slope .* log_inv_sizes .+ intercept
    ss_res = sum((log_counts .- y_pred).^2)
    ss_tot = sum((log_counts .- mean(log_counts)).^2)
    r_squared = ss_tot > 0 ? 1.0 - (ss_res / ss_tot) : 0.0

    return (fractal_dim, r_squared, box_sizes, box_counts)
end
export load_sam3_masks, analyze_sam3_masks
export calculate_cell_fractal_dimensions, analyze_batch_sam3
export generate_fractal_report

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
SAM3MaskData - Data loaded from SAM-3 NPZ export
"""
struct SAM3MaskData
    masks::Array{UInt8, 3}        # (N, H, W) individual cell masks
    combined_mask::BitMatrix      # All cells combined
    edge_mask::BitMatrix          # Edge detection mask
    scores::Vector{Float64}       # Confidence scores
    boxes::Matrix{Float64}        # Bounding boxes (N, 4)
    n_cells::Int
    image_shape::Tuple{Int, Int}
    cell_type::String
    source_image::String
    prompt_used::String
    cell_properties::Vector{Dict{String, Any}}
end

"""
CellFractalMetrics - Fractal metrics for a single cell
"""
struct CellFractalMetrics
    cell_id::Int
    df_edge::Float64              # Fractal dimension of cell edge
    df_area::Float64              # Fractal dimension of cell area
    r_squared_edge::Float64
    r_squared_area::Float64
    area::Int                     # Cell area in pixels
    perimeter::Float64            # Estimated perimeter
    circularity::Float64          # 4π*area/perimeter²
    score::Float64                # SAM-3 confidence
end

"""
SAM3FractalResult - Complete fractal analysis of SAM-3 segmentation
"""
struct SAM3FractalResult
    # Overall metrics
    df_combined::Float64          # Fractal dim of combined mask
    df_edges::Float64             # Fractal dim of all edges
    df_distribution::Float64      # Fractal dim of cell distribution
    r_squared_combined::Float64
    r_squared_edges::Float64
    r_squared_distribution::Float64

    # Per-cell metrics
    cell_metrics::Vector{CellFractalMetrics}

    # Statistics
    mean_df_edge::Float64
    std_df_edge::Float64
    mean_df_area::Float64
    std_df_area::Float64
    mean_circularity::Float64

    # Metadata
    n_cells::Int
    cell_type::String
    source_image::String
end

# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

"""
load_sam3_masks(npz_path::String) -> SAM3MaskData

Load SAM-3 masks from NPZ file exported by Python.
Metadata is loaded from companion JSON file (NPZ.jl can't read Unicode strings).
"""
function load_sam3_masks(npz_path::String)::SAM3MaskData
    # Load NPZ file
    data = npzread(npz_path)

    # Extract arrays
    masks = data["masks"]
    scores = vec(data["scores"])
    boxes = data["boxes"]
    combined_mask = BitMatrix(data["combined_mask"] .> 0)
    edge_mask = BitMatrix(data["edge_mask"] .> 0)

    # Load metadata from companion JSON file
    json_path = replace(npz_path, r"\.npz$" => ".json")
    if !isfile(json_path)
        error("Metadata JSON file not found: $json_path")
    end
    metadata = JSON3.read(read(json_path, String))

    # Extract cell properties
    cell_properties = Dict{String, Any}[]
    if haskey(metadata, :cell_properties)
        for cp in metadata.cell_properties
            push!(cell_properties, Dict{String, Any}(
                "id" => cp.id,
                "area" => cp.area,
                "centroid_x" => cp.centroid_x,
                "centroid_y" => cp.centroid_y,
                "score" => cp.score
            ))
        end
    end

    n_cells = size(masks, 1)
    image_shape = (size(masks, 2), size(masks, 3))

    return SAM3MaskData(
        masks,
        combined_mask,
        edge_mask,
        scores,
        boxes,
        n_cells,
        image_shape,
        String(get(metadata, :cell_type, "unknown")),
        String(get(metadata, :source_image, "")),
        String(get(metadata, :prompt_used, "")),
        cell_properties
    )
end

# ============================================================================
# FRACTAL ANALYSIS FUNCTIONS
# ============================================================================

"""
calculate_perimeter(mask::BitMatrix) -> Float64

Estimate perimeter of a binary mask using edge counting.
"""
function calculate_perimeter(mask::BitMatrix)::Float64
    height, width = size(mask)
    perimeter = 0.0

    for i in 1:height
        for j in 1:width
            if mask[i, j]
                # Count boundary pixels (neighbors that are 0 or outside)
                neighbors = 0
                for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    ni, nj = i + di, j + dj
                    if ni < 1 || ni > height || nj < 1 || nj > width || !mask[ni, nj]
                        neighbors += 1
                    end
                end
                perimeter += neighbors
            end
        end
    end

    return perimeter
end

"""
extract_mask_edges(mask::BitMatrix) -> BitMatrix

Extract edges of a binary mask using morphological gradient.
"""
function extract_mask_edges(mask::BitMatrix)::BitMatrix
    height, width = size(mask)
    edges = falses(height, width)

    for i in 2:(height-1)
        for j in 2:(width-1)
            if mask[i, j]
                # Check if any neighbor is 0 (boundary pixel)
                is_edge = false
                for di in -1:1
                    for dj in -1:1
                        if !mask[i+di, j+dj]
                            is_edge = true
                            break
                        end
                    end
                    is_edge && break
                end
                edges[i, j] = is_edge
            end
        end
    end

    return edges
end

"""
analyze_single_cell(mask::Array{UInt8, 2}, score::Float64, cell_id::Int) -> CellFractalMetrics

Analyze fractal properties of a single cell mask.
"""
function analyze_single_cell(
    mask::Array{UInt8, 2},
    score::Float64,
    cell_id::Int
)::CellFractalMetrics

    # Convert to BitMatrix
    binary_mask = BitMatrix(mask .> 0)

    # Calculate area
    area = sum(binary_mask)

    if area < 10  # Too small for analysis
        return CellFractalMetrics(
            cell_id, NaN, NaN, 0.0, 0.0, area, 0.0, 0.0, score
        )
    end

    # Extract edges
    edges = extract_mask_edges(binary_mask)

    # Calculate perimeter
    perimeter = calculate_perimeter(binary_mask)

    # Circularity: 4π*area/perimeter² (1.0 = perfect circle)
    circularity = perimeter > 0 ? (4 * π * area) / (perimeter^2) : 0.0

    # Fractal dimension of edges
    df_edge, r2_edge, _, _ = box_counting_fractal_dimension(edges)

    # Fractal dimension of area
    df_area, r2_area, _, _ = box_counting_fractal_dimension(binary_mask)

    return CellFractalMetrics(
        cell_id,
        df_edge,
        df_area,
        r2_edge,
        r2_area,
        area,
        perimeter,
        circularity,
        score
    )
end

"""
calculate_distribution_fractal(cell_properties, image_shape) -> (df, r²)

Calculate fractal dimension of cell spatial distribution.
"""
function calculate_distribution_fractal(
    cell_properties::Vector{Dict{String, Any}},
    image_shape::Tuple{Int, Int}
)::Tuple{Float64, Float64}

    if isempty(cell_properties)
        return (NaN, 0.0)
    end

    height, width = image_shape

    # Create binary image of cell centroids (expanded to 5x5)
    centroid_map = falses(height, width)

    for cell in cell_properties
        cx = round(Int, cell["centroid_x"])
        cy = round(Int, cell["centroid_y"])

        # Expand to 5x5 region
        for di in -2:2
            for dj in -2:2
                ni, nj = cy + di, cx + dj
                if 1 <= ni <= height && 1 <= nj <= width
                    centroid_map[ni, nj] = true
                end
            end
        end
    end

    df, r2, _, _ = box_counting_fractal_dimension(BitMatrix(centroid_map))
    return (df, r2)
end

"""
analyze_sam3_masks(mask_data::SAM3MaskData) -> SAM3FractalResult

Complete fractal analysis of SAM-3 segmentation results.
"""
function analyze_sam3_masks(mask_data::SAM3MaskData)::SAM3FractalResult

    # Analyze combined mask
    df_combined, r2_combined, _, _ = box_counting_fractal_dimension(mask_data.combined_mask)

    # Analyze edge mask
    df_edges, r2_edges, _, _ = box_counting_fractal_dimension(mask_data.edge_mask)

    # Analyze spatial distribution
    df_distribution, r2_distribution = calculate_distribution_fractal(
        mask_data.cell_properties,
        mask_data.image_shape
    )

    # Analyze individual cells
    cell_metrics = CellFractalMetrics[]

    for i in 1:mask_data.n_cells
        cell_mask = mask_data.masks[i, :, :]
        score = i <= length(mask_data.scores) ? mask_data.scores[i] : 0.0

        metrics = analyze_single_cell(cell_mask, score, i)
        push!(cell_metrics, metrics)
    end

    # Calculate statistics (excluding NaN values)
    valid_df_edge = filter(!isnan, [m.df_edge for m in cell_metrics])
    valid_df_area = filter(!isnan, [m.df_area for m in cell_metrics])
    valid_circularity = filter(!isnan, [m.circularity for m in cell_metrics])

    mean_df_edge = isempty(valid_df_edge) ? NaN : mean(valid_df_edge)
    std_df_edge = length(valid_df_edge) > 1 ? std(valid_df_edge) : 0.0
    mean_df_area = isempty(valid_df_area) ? NaN : mean(valid_df_area)
    std_df_area = length(valid_df_area) > 1 ? std(valid_df_area) : 0.0
    mean_circularity = isempty(valid_circularity) ? NaN : mean(valid_circularity)

    return SAM3FractalResult(
        df_combined,
        df_edges,
        df_distribution,
        r2_combined,
        r2_edges,
        r2_distribution,
        cell_metrics,
        mean_df_edge,
        std_df_edge,
        mean_df_area,
        std_df_area,
        mean_circularity,
        mask_data.n_cells,
        mask_data.cell_type,
        mask_data.source_image
    )
end

# ============================================================================
# BATCH ANALYSIS
# ============================================================================

"""
analyze_batch_sam3(npz_dir::String) -> Vector{SAM3FractalResult}

Analyze all NPZ files in a directory.
"""
function analyze_batch_sam3(npz_dir::String)::Vector{SAM3FractalResult}
    results = SAM3FractalResult[]

    npz_files = filter(f -> endswith(f, ".npz"), readdir(npz_dir))

    for (i, filename) in enumerate(npz_files)
        filepath = joinpath(npz_dir, filename)
        println("[$i/$(length(npz_files))] Analyzing: $filename")

        try
            mask_data = load_sam3_masks(filepath)
            result = analyze_sam3_masks(mask_data)
            push!(results, result)
        catch e
            @warn "Error processing $filename: $e"
        end
    end

    return results
end

# ============================================================================
# REPORTING
# ============================================================================

"""
generate_fractal_report(results::Vector{SAM3FractalResult}) -> Dict

Generate summary report of fractal analysis.
"""
function generate_fractal_report(results::Vector{SAM3FractalResult})::Dict{String, Any}
    if isempty(results)
        return Dict{String, Any}("error" => "No results to analyze")
    end

    # Aggregate statistics
    all_df_combined = [r.df_combined for r in results if !isnan(r.df_combined)]
    all_df_edges = [r.df_edges for r in results if !isnan(r.df_edges)]
    all_df_distribution = [r.df_distribution for r in results if !isnan(r.df_distribution)]
    all_mean_df_edge = [r.mean_df_edge for r in results if !isnan(r.mean_df_edge)]
    all_circularity = [r.mean_circularity for r in results if !isnan(r.mean_circularity)]

    total_cells = sum(r.n_cells for r in results)

    report = Dict{String, Any}(
        "summary" => Dict(
            "n_images" => length(results),
            "total_cells" => total_cells,
            "cell_types" => unique([r.cell_type for r in results])
        ),
        "fractal_dimensions" => Dict(
            "combined_mask" => Dict(
                "mean" => isempty(all_df_combined) ? NaN : mean(all_df_combined),
                "std" => length(all_df_combined) > 1 ? std(all_df_combined) : 0.0,
                "min" => isempty(all_df_combined) ? NaN : minimum(all_df_combined),
                "max" => isempty(all_df_combined) ? NaN : maximum(all_df_combined)
            ),
            "edge_mask" => Dict(
                "mean" => isempty(all_df_edges) ? NaN : mean(all_df_edges),
                "std" => length(all_df_edges) > 1 ? std(all_df_edges) : 0.0
            ),
            "cell_distribution" => Dict(
                "mean" => isempty(all_df_distribution) ? NaN : mean(all_df_distribution),
                "std" => length(all_df_distribution) > 1 ? std(all_df_distribution) : 0.0
            ),
            "individual_cell_edges" => Dict(
                "mean" => isempty(all_mean_df_edge) ? NaN : mean(all_mean_df_edge),
                "std" => length(all_mean_df_edge) > 1 ? std(all_mean_df_edge) : 0.0
            )
        ),
        "morphology" => Dict(
            "mean_circularity" => isempty(all_circularity) ? NaN : mean(all_circularity),
            "std_circularity" => length(all_circularity) > 1 ? std(all_circularity) : 0.0
        ),
        "interpretation" => generate_interpretation(all_df_combined, all_df_edges, all_circularity)
    )

    return report
end

"""
Generate interpretation of fractal results.
"""
function generate_interpretation(
    df_combined::Vector{Float64},
    df_edges::Vector{Float64},
    circularity::Vector{Float64}
)::String

    interpretations = String[]

    if !isempty(df_combined)
        mean_df = mean(df_combined)
        if mean_df < 1.5
            push!(interpretations, "Low fractal dimension (<1.5): Cells show simple, regular morphology")
        elseif mean_df < 1.8
            push!(interpretations, "Moderate fractal dimension (1.5-1.8): Normal leukocyte complexity")
        else
            push!(interpretations, "High fractal dimension (>1.8): Cells show complex, irregular morphology (possible pathology)")
        end
    end

    if !isempty(df_edges)
        mean_edge = mean(df_edges)
        if mean_edge > 1.3
            push!(interpretations, "Complex cell edges (Df>1.3): May indicate activated or pathological cells")
        end
    end

    if !isempty(circularity)
        mean_circ = mean(circularity)
        if mean_circ > 0.8
            push!(interpretations, "High circularity (>0.8): Predominantly round cells (lymphocytes)")
        elseif mean_circ < 0.5
            push!(interpretations, "Low circularity (<0.5): Irregular cell shapes (granulocytes, monocytes)")
        end
    end

    return join(interpretations, "\n")
end

end  # module SAM3Integration
