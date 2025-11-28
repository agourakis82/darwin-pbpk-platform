"""
Fractal Molecular Descriptors for PBPK Modeling

Based on the insight that drug distribution is a fractal process occurring
on a fractal biological substrate. This module computes molecular fractal
properties that determine how drugs interact with the body's hierarchical
transport networks.

References:
- West, Brown, Enquist (1999) "The Fourth Dimension of Life"
- Molecular Complexity Calculated by Fractal Dimension (Sci Rep, 2019)
"""
module FractalDescriptors

export compute_fractal_features, molecular_fractal_dimension,
       topological_entropy, branching_complexity, wiener_index,
       balaban_j_index, fragment_complexity, compute_all_fractal_descriptors

using Statistics
using LinearAlgebra

# ============================================================================
# MOLECULAR GRAPH CONSTRUCTION FROM SMILES
# ============================================================================

"""
Simple molecular graph from SMILES (atoms and bonds)
Returns adjacency matrix and atom types
"""
function smiles_to_graph(smiles::String)
    # Parse SMILES to extract atoms and connectivity
    atoms = Char[]
    bonds = Tuple{Int,Int,Int}[]  # (from, to, order)

    # Simple SMILES parser (handles common cases)
    i = 1
    atom_idx = 0
    ring_opens = Dict{Char, Int}()
    branch_stack = Int[]
    current_atom = 0

    while i <= length(smiles)
        c = smiles[i]

        if c in "CNOPSFI"  # Common atoms
            atom_idx += 1
            push!(atoms, c)
            if current_atom > 0
                push!(bonds, (current_atom, atom_idx, 1))
            end
            current_atom = atom_idx

        elseif c == 'c' || c == 'n' || c == 'o' || c == 's'  # Aromatic
            atom_idx += 1
            push!(atoms, uppercase(c))
            if current_atom > 0
                push!(bonds, (current_atom, atom_idx, 1))
            end
            current_atom = atom_idx

        elseif c == '['  # Bracketed atom
            j = findnext(']', smiles, i)
            if j !== nothing
                bracket_content = smiles[i+1:j-1]
                # Extract element (first 1-2 uppercase letters)
                elem = 'C'
                for k in 1:min(2, length(bracket_content))
                    if isuppercase(bracket_content[k])
                        elem = bracket_content[k]
                        break
                    end
                end
                atom_idx += 1
                push!(atoms, elem)
                if current_atom > 0
                    push!(bonds, (current_atom, atom_idx, 1))
                end
                current_atom = atom_idx
                i = j
            end

        elseif c == '('  # Branch start
            push!(branch_stack, current_atom)

        elseif c == ')'  # Branch end
            if !isempty(branch_stack)
                current_atom = pop!(branch_stack)
            end

        elseif c == '='  # Double bond (next bond)
            # Mark next bond as double

        elseif c == '#'  # Triple bond
            # Mark next bond as triple

        elseif isdigit(c)  # Ring closure
            if haskey(ring_opens, c)
                ring_atom = ring_opens[c]
                push!(bonds, (ring_atom, current_atom, 1))
                delete!(ring_opens, c)
            else
                ring_opens[c] = current_atom
            end
        end

        i += 1
    end

    # Build adjacency matrix
    n = atom_idx
    if n == 0
        return zeros(1, 1), ['C']
    end

    adj = zeros(Int, n, n)
    for (a, b, order) in bonds
        if a >= 1 && a <= n && b >= 1 && b <= n
            adj[a, b] = order
            adj[b, a] = order
        end
    end

    return adj, atoms
end

# ============================================================================
# TOPOLOGICAL INDICES
# ============================================================================

"""
Wiener Index: Sum of all shortest path lengths in the molecular graph.
Measures molecular compactness/branching.
"""
function wiener_index(adj::Matrix{Int})
    n = size(adj, 1)
    if n <= 1
        return 0.0
    end

    # Floyd-Warshall for all-pairs shortest paths
    dist = fill(Inf, n, n)
    for i in 1:n
        dist[i, i] = 0.0
        for j in 1:n
            if adj[i, j] > 0
                dist[i, j] = 1.0
            end
        end
    end

    for k in 1:n
        for i in 1:n
            for j in 1:n
                if dist[i, k] + dist[k, j] < dist[i, j]
                    dist[i, j] = dist[i, k] + dist[k, j]
                end
            end
        end
    end

    # Sum of upper triangle
    w = 0.0
    for i in 1:n
        for j in i+1:n
            if isfinite(dist[i, j])
                w += dist[i, j]
            end
        end
    end

    return w
end

"""
Balaban J Index: Measures molecular cyclicity and branching.
J = (q / (μ + 1)) * Σ(d_i * d_j)^(-0.5) for all edges
"""
function balaban_j_index(adj::Matrix{Int})
    n = size(adj, 1)
    if n <= 1
        return 0.0
    end

    # Compute distance sums for each vertex
    dist = fill(Inf, n, n)
    for i in 1:n
        dist[i, i] = 0.0
        for j in 1:n
            if adj[i, j] > 0
                dist[i, j] = 1.0
            end
        end
    end

    for k in 1:n
        for i in 1:n
            for j in 1:n
                if dist[i, k] + dist[k, j] < dist[i, j]
                    dist[i, j] = dist[i, k] + dist[k, j]
                end
            end
        end
    end

    # Distance sum for each vertex
    d_sum = [sum(dist[i, j] for j in 1:n if isfinite(dist[i, j])) for i in 1:n]

    # Count edges
    q = sum(adj .> 0) ÷ 2
    if q == 0
        return 0.0
    end

    # Cyclomatic number
    μ = q - n + 1  # For connected graph

    # Compute J
    j_sum = 0.0
    for i in 1:n
        for j in i+1:n
            if adj[i, j] > 0 && d_sum[i] > 0 && d_sum[j] > 0
                j_sum += 1.0 / sqrt(d_sum[i] * d_sum[j])
            end
        end
    end

    return q / (μ + 1) * j_sum
end

"""
Randic Index: Molecular branching descriptor
χ = Σ (d_i * d_j)^(-0.5) for all edges
"""
function randic_index(adj::Matrix{Int})
    n = size(adj, 1)
    degrees = vec(sum(adj .> 0, dims=2))

    chi = 0.0
    for i in 1:n
        for j in i+1:n
            if adj[i, j] > 0 && degrees[i] > 0 && degrees[j] > 0
                chi += 1.0 / sqrt(degrees[i] * degrees[j])
            end
        end
    end

    return chi
end

"""
Zagreb Indices: M1 and M2
M1 = Σ d_i^2
M2 = Σ d_i * d_j for edges
"""
function zagreb_indices(adj::Matrix{Int})
    n = size(adj, 1)
    degrees = vec(sum(adj .> 0, dims=2))

    m1 = sum(degrees.^2)

    m2 = 0.0
    for i in 1:n
        for j in i+1:n
            if adj[i, j] > 0
                m2 += degrees[i] * degrees[j]
            end
        end
    end

    return m1, m2
end

# ============================================================================
# FRACTAL DIMENSION CALCULATIONS
# ============================================================================

"""
Topological Fractal Dimension via Branching Pattern Analysis

Based on the observation that molecular fragments show self-similarity.
Uses the relationship between substructure size and count.
"""
function molecular_fractal_dimension(adj::Matrix{Int})
    n = size(adj, 1)
    if n <= 2
        return 1.0  # Linear molecule
    end

    # Count subgraphs of different sizes using BFS from each node
    # This approximates fragment self-similarity

    sizes = Int[]
    counts = Int[]

    for k in 1:min(n-1, 8)  # Subgraph sizes 1 to 8
        count = count_connected_subgraphs(adj, k)
        if count > 0
            push!(sizes, k)
            push!(counts, count)
        end
    end

    if length(sizes) < 2
        return 1.0
    end

    # Fractal dimension from log-log regression
    # N(r) ∝ r^(-D) => log(N) = -D*log(r) + c
    log_sizes = log.(sizes)
    log_counts = log.(counts)

    # Linear regression
    n_pts = length(sizes)
    x_mean = mean(log_sizes)
    y_mean = mean(log_counts)

    num = sum((log_sizes .- x_mean) .* (log_counts .- y_mean))
    den = sum((log_sizes .- x_mean).^2)

    if abs(den) < 1e-10
        return 1.0
    end

    slope = num / den

    # Fractal dimension is negative of slope (typically 1-3)
    D = clamp(-slope, 1.0, 3.0)

    return D
end

"""
Count connected subgraphs of size k (approximate)
"""
function count_connected_subgraphs(adj::Matrix{Int}, k::Int)
    n = size(adj, 1)
    if k > n
        return 0
    end
    if k == 1
        return n
    end

    # Use BFS neighborhoods as approximation
    count = 0
    for start in 1:n
        # BFS to find k-neighborhoods
        visited = falses(n)
        queue = [start]
        visited[start] = true
        depth = 0
        nodes_at_depth = 1

        while !isempty(queue) && depth < k
            next_queue = Int[]
            for node in queue
                for neighbor in 1:n
                    if adj[node, neighbor] > 0 && !visited[neighbor]
                        visited[neighbor] = true
                        push!(next_queue, neighbor)
                    end
                end
            end
            queue = next_queue
            depth += 1
            nodes_at_depth = sum(visited)
            if nodes_at_depth >= k
                count += 1
                break
            end
        end
    end

    return count
end

"""
Topological Entropy: Information content of molecular graph

H = -Σ p_i * log(p_i) where p_i is probability of degree i
Measures structural complexity and heterogeneity.
"""
function topological_entropy(adj::Matrix{Int})
    n = size(adj, 1)
    if n <= 1
        return 0.0
    end

    degrees = vec(sum(adj .> 0, dims=2))

    # Degree distribution
    max_deg = maximum(degrees)
    if max_deg == 0
        return 0.0
    end

    deg_counts = zeros(max_deg + 1)
    for d in degrees
        deg_counts[d + 1] += 1
    end

    # Probabilities
    probs = deg_counts ./ n

    # Shannon entropy
    H = 0.0
    for p in probs
        if p > 0
            H -= p * log(p)
        end
    end

    return H
end

"""
Branching Complexity Index

Measures the hierarchical branching pattern of the molecule.
Higher values indicate more complex, tree-like structures.
"""
function branching_complexity(adj::Matrix{Int})
    n = size(adj, 1)
    if n <= 2
        return 0.0
    end

    degrees = vec(sum(adj .> 0, dims=2))

    # Count branching points (degree > 2)
    branch_points = sum(degrees .> 2)

    # Average branching
    avg_branch = mean(degrees)

    # Branching variance (heterogeneity)
    branch_var = var(degrees)

    # Composite index: normalized branching complexity
    bc = (branch_points / n) * (avg_branch / 4) * (1 + sqrt(branch_var))

    return bc
end

"""
Fragment Complexity: Self-similarity of molecular fragments

Analyzes how molecular fragments relate to the whole structure,
capturing the fractal nature of molecular topology.
"""
function fragment_complexity(adj::Matrix{Int}, atoms::Vector{Char})
    n = size(adj, 1)
    if n <= 3
        return 1.0
    end

    # Generate fragments by removing atoms one at a time
    # and measuring how structure degrades

    complexities = Float64[]

    for remove_idx in 1:n
        # Create subgraph without this atom
        remaining = setdiff(1:n, remove_idx)
        if length(remaining) < 2
            continue
        end

        sub_adj = adj[remaining, remaining]

        # Measure complexity of fragment
        sub_wiener = wiener_index(sub_adj)
        orig_wiener = wiener_index(adj)

        if orig_wiener > 0
            # Ratio of fragment complexity to whole
            ratio = sub_wiener / orig_wiener
            push!(complexities, ratio)
        end
    end

    if isempty(complexities)
        return 1.0
    end

    # Self-similarity: how consistent is complexity across fragments
    # High consistency = high self-similarity = higher fractal nature
    return 1.0 / (1.0 + std(complexities))
end

# ============================================================================
# PHYSIOLOGICAL FRACTAL COUPLING
# ============================================================================

"""
Tissue Fractal Dimensions (from literature)

These represent the fractal dimension of vascular networks in different tissues.
Source: Fractal parameters and vascular networks (Theor Biol Med Model, 2008)
"""
const TISSUE_FRACTAL_DIM = Dict{Symbol, Float64}(
    :plasma => 2.0,      # Flowing compartment
    :adipose => 2.4,     # Sparse vascularization
    :muscle => 2.7,      # Moderate vascularization
    :bone => 2.3,        # Limited vascularization
    :liver => 2.8,       # Dense vascularization (sinusoids)
    :brain => 2.9,       # Dense capillary network
    :lung => 2.97,       # Highly fractal for gas exchange
    :kidney => 2.85,     # Dense glomerular network
    :heart => 2.75,      # Coronary network
    :skin => 2.5,        # Dermal capillaries
    :gut => 2.7,         # Intestinal villi
    :spleen => 2.6       # Red pulp
)

"""
Effective Volume Contribution

Computes how molecular fractal dimension couples with tissue fractal dimension
to determine effective distribution volume.
"""
function fractal_coupling(mol_fd::Float64, tissue::Symbol)
    tissue_fd = get(TISSUE_FRACTAL_DIM, tissue, 2.5)

    # Coupling is strongest when fractal dimensions match
    # Based on WBE theory: transport efficiency depends on network matching
    mismatch = abs(mol_fd - tissue_fd)

    # Coupling factor (1 = perfect, <1 = reduced)
    coupling = exp(-mismatch^2 / 2)

    return coupling
end

"""
Compute effective distribution dimension across all tissues
"""
function effective_distribution_dimension(mol_fd::Float64)
    # Weighted average of coupling across tissues
    # Weights based on typical tissue volumes
    tissue_weights = Dict(
        :adipose => 0.20,
        :muscle => 0.40,
        :bone => 0.07,
        :liver => 0.03,
        :brain => 0.02,
        :lung => 0.02,
        :kidney => 0.005,
        :heart => 0.005,
        :skin => 0.10,
        :gut => 0.03,
        :spleen => 0.003
    )

    weighted_coupling = 0.0
    total_weight = 0.0

    for (tissue, weight) in tissue_weights
        coupling = fractal_coupling(mol_fd, tissue)
        weighted_coupling += weight * coupling
        total_weight += weight
    end

    return weighted_coupling / total_weight
end

# ============================================================================
# MAIN API
# ============================================================================

"""
Compute all fractal descriptors for a molecule given its SMILES string.

Returns a Dict with:
- fractal_dim: Molecular fractal dimension
- topological_entropy: Structural information content
- branching_complexity: Hierarchical branching measure
- wiener_index: Path length sum (normalized)
- balaban_j: Cyclicity and branching index
- randic_index: Branching descriptor
- zagreb_m1, zagreb_m2: Degree-based indices
- fragment_self_similarity: Self-similarity measure
- effective_dist_dim: Coupling with physiological fractals
"""
function compute_all_fractal_descriptors(smiles::String)
    adj, atoms = smiles_to_graph(smiles)
    n = size(adj, 1)

    # Topological indices
    wiener = wiener_index(adj)
    balaban = balaban_j_index(adj)
    randic = randic_index(adj)
    m1, m2 = zagreb_indices(adj)

    # Fractal measures
    fd = molecular_fractal_dimension(adj)
    entropy = topological_entropy(adj)
    branching = branching_complexity(adj)
    self_sim = fragment_complexity(adj, atoms)

    # Physiological coupling
    eff_dist = effective_distribution_dimension(fd)

    # Normalize some indices by molecule size for comparability
    wiener_norm = n > 1 ? wiener / (n * (n-1) / 2) : 0.0

    return Dict{String, Float64}(
        "n_atoms" => Float64(n),
        "fractal_dim" => fd,
        "topological_entropy" => entropy,
        "branching_complexity" => branching,
        "wiener_index" => wiener_norm,
        "balaban_j" => balaban,
        "randic_index" => randic,
        "zagreb_m1" => m1 / max(1, n),
        "zagreb_m2" => m2 / max(1, n),
        "fragment_self_similarity" => self_sim,
        "effective_dist_dim" => eff_dist
    )
end

"""
Compute fractal features as a vector for ML models
"""
function compute_fractal_features(smiles::String)::Vector{Float64}
    desc = compute_all_fractal_descriptors(smiles)

    return Float64[
        desc["fractal_dim"],
        desc["topological_entropy"],
        desc["branching_complexity"],
        desc["wiener_index"],
        desc["balaban_j"],
        desc["randic_index"],
        desc["zagreb_m1"],
        desc["zagreb_m2"],
        desc["fragment_self_similarity"],
        desc["effective_dist_dim"]
    ]
end

end # module
