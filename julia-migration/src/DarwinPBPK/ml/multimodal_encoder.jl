"""
Multimodal Encoder - SOTA Implementation for Molecular Property Prediction

SOTA Q1 2025 Implementation (December 2025 Update):
- ChemBERTa integration via PyCall (768d pre-trained embeddings)
- D-MPNN (Directed Message Passing) for molecular graphs
- Real SMILES tokenization with learned embeddings (fallback)
- GNN encoder with MolecularGraph.jl for molecular graph construction
- Cross-attention fusion with multi-head attention
- Quantum descriptor integration

This module provides the complete multimodal molecular encoder stack:
1. ChemBERTa (primary) / GRU (fallback) for SMILES sequences
2. D-MPNN / GAT for molecular graph structure
3. Quantum descriptors for electronic properties
4. Cross-attention fusion for unified representation

Autor: Dr. Sounio Agourakis + AI Assistant
Data: Novembro 2025
Atualizado: 2025-12-07 - ChemBERTa + D-MPNN integration
"""

module MultimodalEncoder

using Flux
using Functors: @functor
using GraphNeuralNetworks
using MolecularGraph
using Statistics
using Random

# Dimensions
const SMILES_VOCAB_SIZE = 128  # ASCII characters
const SMILES_MAX_LEN = 200
const SMILES_EMBED_DIM = 64
const SMILES_HIDDEN_DIM = 256
const SMILES_OUTPUT_DIM = 768  # Match ChemBERTa output dim

const GNN_NODE_DIM = 32  # Node feature dimension
const GNN_HIDDEN_DIM = 128
const GNN_OUTPUT_DIM = 256

const FUSION_DIM = 512

# Atom type encoding (one-hot)
const ATOM_TYPES = [:C, :N, :O, :S, :F, :Cl, :Br, :I, :P, :B, :Si, :Se, :other]
const N_ATOM_TYPES = length(ATOM_TYPES)

# Bond type encoding
const BOND_ORDERS = [1, 2, 3, 4]  # single, double, triple, aromatic
const N_BOND_TYPES = length(BOND_ORDERS)

"""
Get atom type index (1-indexed for one-hot encoding).
"""
function get_atom_type_idx(symbol::Symbol)::Int
    idx = findfirst(==(symbol), ATOM_TYPES)
    return idx === nothing ? N_ATOM_TYPES : idx  # "other" for unknown
end

"""
Create atom feature vector.

Features (32-dim total):
- Atom type one-hot (13)
- Formal charge (1, normalized)
- Is aromatic (1)
- Is in ring (1)
- Hybridization placeholder (4)
- Degree placeholder (6)
- Num H placeholder (5)
- Padding (1)
"""
function atom_features(mol, atom_idx::Int)::Vector{Float32}
    features = zeros(Float32, GNN_NODE_DIM)

    symbols = atom_symbol(mol)
    charges = atom_charge(mol)
    aromatic = is_aromatic(mol)

    symbol = symbols[atom_idx]

    # Atom type one-hot (1-13)
    type_idx = get_atom_type_idx(symbol)
    features[type_idx] = 1.0f0

    # Formal charge (14) - normalized
    features[14] = Float32(charges[atom_idx]) / 4.0f0

    # Is aromatic (15)
    features[15] = aromatic[atom_idx] ? 1.0f0 : 0.0f0

    # Is in ring (16) - approximate from aromatic
    features[16] = aromatic[atom_idx] ? 1.0f0 : 0.0f0

    # Degree (17-22) - count neighbors using adjacency list
    n_neighbors = length(mol.graph.fadjlist[atom_idx])
    if n_neighbors <= 6
        features[16+n_neighbors] = 1.0f0
    end

    return features
end

"""
Convert SMILES to GNNGraph with node features.
"""
function smiles_to_graph(smiles::String)::Union{GNNGraph,Nothing}
    try
        mol = smilestomol(smiles)
        n_atoms = length(mol.graph.fadjlist)  # Number of atoms = number of vertices

        if n_atoms == 0
            return nothing
        end

        # Build edge list (undirected) from adjacency list
        sources = Int[]
        targets = Int[]
        for src in 1:n_atoms
            for tgt in mol.graph.fadjlist[src]
                if src < tgt  # Only add each edge once
                    push!(sources, src)
                    push!(targets, tgt)
                    push!(sources, tgt)
                    push!(targets, src)
                end
            end
        end

        # Handle molecules with no bonds (single atoms)
        if isempty(sources)
            # Self-loop for single atom
            sources = [1]
            targets = [1]
        end

        # Build node features matrix [n_features, n_nodes]
        node_features = zeros(Float32, GNN_NODE_DIM, n_atoms)
        for i in 1:n_atoms
            node_features[:, i] = atom_features(mol, i)
        end

        # Create GNNGraph
        g = GNNGraph(sources, targets; ndata=(; x=node_features))

        return g
    catch e
        # Invalid SMILES
        @warn "Failed to parse SMILES: $smiles" exception = e
        return nothing
    end
end

"""
SMILES Encoder - Learned character-level embeddings with GRU.

This is a practical alternative to ChemBERTa that can be trained end-to-end.
For production, consider using HuggingFace Transformers via PyCall.
"""
struct SMILESEncoder
    embedding::Embedding
    gru  # Flux.Recur - type varies
    output_proj::Dense
end

@functor SMILESEncoder

function SMILESEncoder()
    embedding = Embedding(SMILES_VOCAB_SIZE => SMILES_EMBED_DIM)
    gru = GRU(SMILES_EMBED_DIM => SMILES_HIDDEN_DIM)
    output_proj = Dense(SMILES_HIDDEN_DIM => SMILES_OUTPUT_DIM)
    return SMILESEncoder(embedding, gru, output_proj)
end

"""
Encode SMILES string to fixed-size embedding.
"""
function (encoder::SMILESEncoder)(smiles::String)::Vector{Float32}
    # Tokenize: character-level with ASCII codes (1-indexed)
    tokens = [min(Int(c), SMILES_VOCAB_SIZE) for c in smiles]

    # Pad or truncate to max length
    if length(tokens) > SMILES_MAX_LEN
        tokens = tokens[1:SMILES_MAX_LEN]
    elseif length(tokens) < SMILES_MAX_LEN
        tokens = vcat(tokens, ones(Int, SMILES_MAX_LEN - length(tokens)))
    end

    # Embed: [embed_dim, seq_len]
    embedded = encoder.embedding(tokens)

    # GRU expects [features, seq_len, batch] or [features, seq_len]
    # Reshape to [features, seq_len, 1] for batch of 1
    embedded_3d = reshape(embedded, size(embedded, 1), size(embedded, 2), 1)

    # GRU: process sequence - returns [hidden_dim, seq_len, batch]
    Flux.reset!(encoder.gru)
    gru_out = encoder.gru(embedded_3d)

    # Take last hidden state: [hidden_dim]
    hidden = gru_out[:, end, 1]

    # Project to output dimension
    output = encoder.output_proj(hidden)

    return vec(output)
end

"""
Batch encode multiple SMILES strings.
"""
function (encoder::SMILESEncoder)(smiles_batch::Vector{String})::Matrix{Float32}
    outputs = [encoder(s) for s in smiles_batch]
    return hcat(outputs...)  # [output_dim, batch_size]
end

"""
GNN Encoder using GraphNeuralNetworks.jl with GATConv.

Real implementation using MolecularGraph.jl for SMILES -> Graph conversion.
"""
struct GNNEncoder
    conv1::GATConv
    conv2::GATConv
    conv3::GATConv
    pool::GlobalPool
    output_proj::Dense
end

@functor GNNEncoder

function GNNEncoder()
    # GAT layers with multi-head attention
    conv1 = GATConv(GNN_NODE_DIM => GNN_HIDDEN_DIM ÷ 4; heads=4, concat=true)
    conv2 = GATConv(GNN_HIDDEN_DIM => GNN_HIDDEN_DIM ÷ 4; heads=4, concat=true)
    conv3 = GATConv(GNN_HIDDEN_DIM => GNN_OUTPUT_DIM; heads=1, concat=false)

    # Global mean pooling
    pool = GlobalPool(mean)

    # Output projection
    output_proj = Dense(GNN_OUTPUT_DIM => GNN_OUTPUT_DIM)

    return GNNEncoder(conv1, conv2, conv3, pool, output_proj)
end

"""
Encode molecular graph to fixed-size embedding.
"""
function (encoder::GNNEncoder)(g::GNNGraph)::Vector{Float32}
    # Get node features
    x = g.ndata.x  # [node_dim, n_nodes]

    # Message passing
    x = relu.(encoder.conv1(g, x))
    x = relu.(encoder.conv2(g, x))
    x = encoder.conv3(g, x)

    # Global pooling
    x_pooled = encoder.pool(g, x)  # [output_dim, 1]

    # Output projection
    output = encoder.output_proj(x_pooled)

    return vec(output)
end

"""
Encode from SMILES string (convenience method).
"""
function (encoder::GNNEncoder)(smiles::String)::Union{Vector{Float32},Nothing}
    g = smiles_to_graph(smiles)
    if g === nothing
        return nothing
    end
    return encoder(g)
end

"""
Cross-Attention Fusion Layer.

Fuses multiple modality embeddings using multi-head attention.
"""
struct CrossAttentionFusion
    q_proj::Dense
    k_proj::Dense
    v_proj::Dense
    output_proj::Dense
    num_heads::Int
    head_dim::Int
end

@functor CrossAttentionFusion

function CrossAttentionFusion(
    input_dims::Vector{Int};
    output_dim::Int=FUSION_DIM,
    num_heads::Int=8
)
    total_input_dim = sum(input_dims)
    head_dim = output_dim ÷ num_heads

    q_proj = Dense(total_input_dim => output_dim)
    k_proj = Dense(total_input_dim => output_dim)
    v_proj = Dense(total_input_dim => output_dim)
    output_proj = Dense(output_dim => output_dim)

    return CrossAttentionFusion(q_proj, k_proj, v_proj, output_proj, num_heads, head_dim)
end

"""
Fuse multiple embeddings using self-attention.
"""
function (fusion::CrossAttentionFusion)(embeddings::Vector{Vector{Float32}})::Vector{Float32}
    # Concatenate all embeddings
    concat_emb = vcat(embeddings...)

    # Self-attention projections
    Q = fusion.q_proj(concat_emb)
    K = fusion.k_proj(concat_emb)
    V = fusion.v_proj(concat_emb)

    # Reshape for multi-head attention [num_heads, head_dim]
    Q_heads = reshape(Q, fusion.head_dim, fusion.num_heads)
    K_heads = reshape(K, fusion.head_dim, fusion.num_heads)
    V_heads = reshape(V, fusion.head_dim, fusion.num_heads)

    # Scaled dot-product attention per head
    scale = Float32(sqrt(fusion.head_dim))
    attn_scores = (Q_heads' * K_heads) ./ scale  # [num_heads, num_heads]
    attn_weights = softmax(attn_scores; dims=2)

    # Weighted combination
    attn_output = attn_weights * V_heads'  # [num_heads, head_dim]

    # Flatten and project
    output = fusion.output_proj(vec(attn_output'))

    return output
end

"""
Multimodal Molecular Encoder - Complete Implementation.

Combines:
- SMILESEncoder: Character-level GRU (768d)
- GNNEncoder: Graph attention network (256d)
- CrossAttentionFusion: Multi-head self-attention (512d)
"""
struct MultimodalMolecularEncoder
    smiles_encoder::SMILESEncoder
    gnn_encoder::GNNEncoder
    fusion::CrossAttentionFusion
    use_gnn::Bool
end

@functor MultimodalMolecularEncoder

function MultimodalMolecularEncoder(; use_gnn::Bool=true)
    smiles_encoder = SMILESEncoder()
    gnn_encoder = GNNEncoder()

    if use_gnn
        fusion = CrossAttentionFusion([SMILES_OUTPUT_DIM, GNN_OUTPUT_DIM])
    else
        fusion = CrossAttentionFusion([SMILES_OUTPUT_DIM])
    end

    return MultimodalMolecularEncoder(smiles_encoder, gnn_encoder, fusion, use_gnn)
end

"""
Encode molecule from SMILES to unified representation.
"""
function (encoder::MultimodalMolecularEncoder)(smiles::String)::Vector{Float32}
    embeddings = Vector{Vector{Float32}}()

    # SMILES embedding (always computed)
    smiles_emb = encoder.smiles_encoder(smiles)
    push!(embeddings, smiles_emb)

    # GNN embedding (if enabled and valid SMILES)
    if encoder.use_gnn
        gnn_emb = encoder.gnn_encoder(smiles)
        if gnn_emb !== nothing
            push!(embeddings, gnn_emb)
        else
            # Fallback: zero embedding if SMILES parsing fails
            push!(embeddings, zeros(Float32, GNN_OUTPUT_DIM))
        end
    end

    # Fuse embeddings
    unified = encoder.fusion(embeddings)

    return unified
end

"""
Batch encode molecules.
"""
function (encoder::MultimodalMolecularEncoder)(smiles_batch::Vector{String})::Matrix{Float32}
    outputs = [encoder(s) for s in smiles_batch]
    return hcat(outputs...)  # [fusion_dim, batch_size]
end

# ============================================================================
# QUANTUM DESCRIPTOR INTEGRATION
# ============================================================================

# Import quantum descriptors module
include("quantum_descriptors.jl")
using .QuantumDescriptors

const QUANTUM_DIM = QUANTUM_DESCRIPTOR_DIM  # 24 dimensions

"""
Quantum Descriptor Encoder - Projects quantum descriptors to embedding space.

Quantum descriptors provide electronic structure information that captures
drug-tissue interactions better than classical descriptors:
- HOMO/LUMO for CYP450 binding prediction
- Polarizability for membrane partitioning
- Abraham descriptors for solvation
"""
struct QuantumEncoder
    input_proj::Dense
    hidden::Dense
    output_proj::Dense
end

@functor QuantumEncoder

function QuantumEncoder(; output_dim::Int = 128)
    input_proj = Dense(QUANTUM_DIM => 64, relu)
    hidden = Dense(64 => 64, relu)
    output_proj = Dense(64 => output_dim)
    return QuantumEncoder(input_proj, hidden, output_proj)
end

"""
Encode quantum descriptors to embedding.
"""
function (encoder::QuantumEncoder)(qd::QuantumDescriptorSet)::Vector{Float32}
    # Normalize and convert to vector
    x = normalize_descriptors(qd)

    # Forward pass
    h = encoder.input_proj(x)
    h = encoder.hidden(h)
    out = encoder.output_proj(h)

    return vec(out)
end

function (encoder::QuantumEncoder)(smiles::String)::Union{Vector{Float32},Nothing}
    qd = calculate_quantum_descriptors(smiles)
    if qd === nothing
        return nothing
    end
    return encoder(qd)
end

const QUANTUM_OUTPUT_DIM = 128

"""
Enhanced Multimodal Encoder with Quantum Descriptors.

Combines FOUR modalities:
1. SMILESEncoder: Character-level GRU (768d) - sequential patterns
2. GNNEncoder: Graph attention (256d) - molecular topology
3. QuantumEncoder: Electronic structure (128d) - quantum properties
4. CrossAttentionFusion: Multi-head attention (512d) - unified embedding

The quantum descriptors add crucial information:
- HOMO energy predicts CYP450 oxidation susceptibility
- Polarizability improves membrane partition prediction
- Abraham descriptors capture H-bonding for protein binding
"""
struct EnhancedMultimodalEncoder
    smiles_encoder::SMILESEncoder
    gnn_encoder::GNNEncoder
    quantum_encoder::QuantumEncoder
    fusion::CrossAttentionFusion
    use_gnn::Bool
    use_quantum::Bool
end

@functor EnhancedMultimodalEncoder

function EnhancedMultimodalEncoder(; use_gnn::Bool=true, use_quantum::Bool=true)
    smiles_encoder = SMILESEncoder()
    gnn_encoder = GNNEncoder()
    quantum_encoder = QuantumEncoder()

    # Calculate fusion input dimensions
    input_dims = Int[SMILES_OUTPUT_DIM]
    if use_gnn
        push!(input_dims, GNN_OUTPUT_DIM)
    end
    if use_quantum
        push!(input_dims, QUANTUM_OUTPUT_DIM)
    end

    fusion = CrossAttentionFusion(input_dims)

    return EnhancedMultimodalEncoder(
        smiles_encoder, gnn_encoder, quantum_encoder,
        fusion, use_gnn, use_quantum
    )
end

"""
Encode molecule with all available modalities.
"""
function (encoder::EnhancedMultimodalEncoder)(smiles::String)::Vector{Float32}
    embeddings = Vector{Vector{Float32}}()

    # SMILES embedding (always computed)
    smiles_emb = encoder.smiles_encoder(smiles)
    push!(embeddings, smiles_emb)

    # GNN embedding
    if encoder.use_gnn
        gnn_emb = encoder.gnn_encoder(smiles)
        if gnn_emb !== nothing
            push!(embeddings, gnn_emb)
        else
            push!(embeddings, zeros(Float32, GNN_OUTPUT_DIM))
        end
    end

    # Quantum descriptor embedding
    if encoder.use_quantum
        quantum_emb = encoder.quantum_encoder(smiles)
        if quantum_emb !== nothing
            push!(embeddings, quantum_emb)
        else
            push!(embeddings, zeros(Float32, QUANTUM_OUTPUT_DIM))
        end
    end

    # Fuse all embeddings
    unified = encoder.fusion(embeddings)

    return unified
end

"""
Get raw quantum descriptors for a molecule (for interpretability).
"""
function get_quantum_descriptors(smiles::String)::Union{QuantumDescriptorSet,Nothing}
    return calculate_quantum_descriptors(smiles)
end

"""
Batch encode with enhanced encoder.
"""
function (encoder::EnhancedMultimodalEncoder)(smiles_batch::Vector{String})::Matrix{Float32}
    outputs = [encoder(s) for s in smiles_batch]
    return hcat(outputs...)
end

# ============================================================================
# SOTA MULTIMODAL ENCODER V2 (December 2025)
# ChemBERTa + D-MPNN + Quantum
# ============================================================================

# Try to import ChemBERTa and D-MPNN modules
# These are optional - fallback to basic encoders if unavailable

const CHEMBERTA_AVAILABLE = Ref{Bool}(false)
const DMPNN_AVAILABLE = Ref{Bool}(false)

"""
Check if ChemBERTa is available (requires PyCall + transformers).
"""
function check_chemberta_available()::Bool
    try
        # Check if ChemBERTaBridge module exists and is available
        # The module should be loaded at top level in DarwinPBPK.jl
        if isdefined(Main, :DarwinPBPK) && isdefined(Main.DarwinPBPK, :ChemBERTaBridge)
            CHEMBERTA_AVAILABLE[] = Main.DarwinPBPK.ChemBERTaBridge.is_available()
        else
            CHEMBERTA_AVAILABLE[] = false
        end
    catch
        CHEMBERTA_AVAILABLE[] = false
    end
    return CHEMBERTA_AVAILABLE[]
end

"""
Check if D-MPNN module is available.
"""
function check_dmpnn_available()::Bool
    try
        # Check if DMPNN module exists
        # The module should be loaded at top level in DarwinPBPK.jl
        if isdefined(Main, :DarwinPBPK) && isdefined(Main.DarwinPBPK, :DMPNN)
            DMPNN_AVAILABLE[] = true
        else
            DMPNN_AVAILABLE[] = false
        end
    catch
        DMPNN_AVAILABLE[] = false
    end
    return DMPNN_AVAILABLE[]
end

const CHEMBERTA_DIM = 768
const DMPNN_DIM = 256

"""
SOTA Multimodal Encoder V2 - Publication-Ready Architecture

Combines the best available encoders:
1. **ChemBERTa** (768d): Pre-trained transformer on 77M SMILES
   - Falls back to GRU encoder if PyCall unavailable
2. **D-MPNN** (256d): Directed message passing for molecular graphs
   - Falls back to GAT encoder if unavailable
3. **Quantum** (128d): Electronic structure descriptors
4. **Cross-Attention Fusion** (512d): Multi-head attention fusion

This is the recommended encoder for Q1 publication.

# Example
```julia
encoder = SOTAMultimodalEncoderV2()
emb = encoder("CCO")  # 512d unified embedding
batch_emb = encoder(["CCO", "CC(=O)O"])  # [512, 2]
```
"""
struct SOTAMultimodalEncoderV2
    # Primary encoders (SOTA when available)
    chemberta_available::Bool
    dmpnn_available::Bool

    # Fallback encoders (always available)
    smiles_encoder::SMILESEncoder
    gnn_encoder::GNNEncoder
    quantum_encoder::QuantumEncoder

    # Fusion layer
    fusion::CrossAttentionFusion

    # Configuration
    use_quantum::Bool
    output_dim::Int
end

@functor SOTAMultimodalEncoderV2

function SOTAMultimodalEncoderV2(;
    use_quantum::Bool = true,
    output_dim::Int = FUSION_DIM,
    force_fallback::Bool = false  # For testing without PyCall
)
    # Check SOTA encoder availability
    chemberta_ok = force_fallback ? false : check_chemberta_available()
    dmpnn_ok = force_fallback ? false : check_dmpnn_available()

    if chemberta_ok
        @info "SOTAMultimodalEncoderV2: Using ChemBERTa (768d)"
    else
        @info "SOTAMultimodalEncoderV2: Using GRU fallback for SMILES (768d)"
    end

    if dmpnn_ok
        @info "SOTAMultimodalEncoderV2: Using D-MPNN (256d)"
    else
        @info "SOTAMultimodalEncoderV2: Using GAT fallback for graphs (256d)"
    end

    # Create fallback encoders (always needed)
    smiles_encoder = SMILESEncoder()
    gnn_encoder = GNNEncoder()
    quantum_encoder = QuantumEncoder()

    # Calculate fusion dimensions based on what's available
    input_dims = Int[]

    # SMILES: ChemBERTa (768) or GRU (768)
    push!(input_dims, SMILES_OUTPUT_DIM)  # Both output 768d

    # Graph: D-MPNN (256) or GAT (256)
    push!(input_dims, GNN_OUTPUT_DIM)  # Both output 256d

    # Quantum descriptors (optional)
    if use_quantum
        push!(input_dims, QUANTUM_OUTPUT_DIM)  # 128d
    end

    fusion = CrossAttentionFusion(input_dims; output_dim = output_dim)

    return SOTAMultimodalEncoderV2(
        chemberta_ok,
        dmpnn_ok,
        smiles_encoder,
        gnn_encoder,
        quantum_encoder,
        fusion,
        use_quantum,
        output_dim
    )
end

"""
Encode a single molecule with SOTA multimodal encoder.
"""
function (encoder::SOTAMultimodalEncoderV2)(smiles::String)::Vector{Float32}
    embeddings = Vector{Vector{Float32}}()

    # ========================================
    # SMILES Encoding (ChemBERTa or GRU)
    # ========================================
    if encoder.chemberta_available
        try
            # Use ChemBERTa via bridge
            smiles_emb = Main.DarwinPBPK.ChemBERTaBridge.encode(smiles)
            push!(embeddings, smiles_emb)
        catch e
            @warn "ChemBERTa encoding failed, using fallback" exception=e
            smiles_emb = encoder.smiles_encoder(smiles)
            push!(embeddings, smiles_emb)
        end
    else
        # Use GRU fallback
        smiles_emb = encoder.smiles_encoder(smiles)
        push!(embeddings, smiles_emb)
    end

    # ========================================
    # Graph Encoding (D-MPNN or GAT)
    # ========================================
    if encoder.dmpnn_available
        try
            # Use D-MPNN via module
            graph_emb = Main.DarwinPBPK.DMPNN.encode_molecule(
                Main.DarwinPBPK.DMPNN.DMPNNEncoder(),
                smiles
            )
            if graph_emb !== nothing
                push!(embeddings, graph_emb)
            else
                push!(embeddings, zeros(Float32, GNN_OUTPUT_DIM))
            end
        catch e
            @warn "D-MPNN encoding failed, using fallback" exception=e
            graph_emb = encoder.gnn_encoder(smiles)
            push!(embeddings, graph_emb !== nothing ? graph_emb : zeros(Float32, GNN_OUTPUT_DIM))
        end
    else
        # Use GAT fallback
        graph_emb = encoder.gnn_encoder(smiles)
        if graph_emb !== nothing
            push!(embeddings, graph_emb)
        else
            push!(embeddings, zeros(Float32, GNN_OUTPUT_DIM))
        end
    end

    # ========================================
    # Quantum Descriptors (optional)
    # ========================================
    if encoder.use_quantum
        quantum_emb = encoder.quantum_encoder(smiles)
        if quantum_emb !== nothing
            push!(embeddings, quantum_emb)
        else
            push!(embeddings, zeros(Float32, QUANTUM_OUTPUT_DIM))
        end
    end

    # ========================================
    # Fusion
    # ========================================
    unified = encoder.fusion(embeddings)

    return unified
end

"""
Batch encode molecules with SOTA encoder.
"""
function (encoder::SOTAMultimodalEncoderV2)(smiles_batch::Vector{String})::Matrix{Float32}
    n = length(smiles_batch)
    embeddings = Matrix{Float32}(undef, encoder.output_dim, n)

    # Use batch encoding when ChemBERTa available for efficiency
    if encoder.chemberta_available && n > 1
        try
            # Batch SMILES encoding
            smiles_embs = Main.DarwinPBPK.ChemBERTaBridge.encode_batch(smiles_batch)

            # Individual encoding for graphs and quantum (can't easily batch)
            for (i, smiles) in enumerate(smiles_batch)
                emb_list = Vector{Vector{Float32}}()

                # SMILES from batch
                push!(emb_list, smiles_embs[:, i])

                # Graph (fallback to GAT since D-MPNN doesn't batch well)
                graph_emb = encoder.gnn_encoder(smiles)
                push!(emb_list, graph_emb !== nothing ? graph_emb : zeros(Float32, GNN_OUTPUT_DIM))

                # Quantum
                if encoder.use_quantum
                    quantum_emb = encoder.quantum_encoder(smiles)
                    push!(emb_list, quantum_emb !== nothing ? quantum_emb : zeros(Float32, QUANTUM_OUTPUT_DIM))
                end

                # Fuse
                embeddings[:, i] = encoder.fusion(emb_list)
            end

            return embeddings
        catch e
            @warn "Batch encoding failed, falling back to individual" exception=e
        end
    end

    # Individual encoding fallback
    for (i, smiles) in enumerate(smiles_batch)
        embeddings[:, i] = encoder(smiles)
    end

    return embeddings
end

"""
Get encoder configuration info.
"""
function encoder_info(encoder::SOTAMultimodalEncoderV2)::Dict{String, Any}
    return Dict(
        "name" => "SOTAMultimodalEncoderV2",
        "smiles_encoder" => encoder.chemberta_available ? "ChemBERTa" : "GRU",
        "graph_encoder" => encoder.dmpnn_available ? "D-MPNN" : "GAT",
        "quantum_enabled" => encoder.use_quantum,
        "output_dim" => encoder.output_dim,
        "total_input_dim" => SMILES_OUTPUT_DIM + GNN_OUTPUT_DIM + (encoder.use_quantum ? QUANTUM_OUTPUT_DIM : 0),
    )
end

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

"""
Create the recommended encoder for production use.

Returns SOTAMultimodalEncoderV2 which uses the best available components.
"""
function create_sota_encoder(; kwargs...)::SOTAMultimodalEncoderV2
    return SOTAMultimodalEncoderV2(; kwargs...)
end

"""
Quick molecular encoding for single SMILES.

Uses a cached encoder instance for efficiency.
"""
const _cached_encoder = Ref{Union{SOTAMultimodalEncoderV2, Nothing}}(nothing)

function quick_encode(smiles::String)::Vector{Float32}
    if _cached_encoder[] === nothing
        _cached_encoder[] = SOTAMultimodalEncoderV2(; use_quantum=false)
    end
    return _cached_encoder[](smiles)
end

function quick_encode(smiles_batch::Vector{String})::Matrix{Float32}
    if _cached_encoder[] === nothing
        _cached_encoder[] = SOTAMultimodalEncoderV2(; use_quantum=false)
    end
    return _cached_encoder[](smiles_batch)
end

export SOTAMultimodalEncoderV2, create_sota_encoder, quick_encode, encoder_info

# Export public API
export MultimodalMolecularEncoder, SMILESEncoder, GNNEncoder, CrossAttentionFusion
export EnhancedMultimodalEncoder, QuantumEncoder, get_quantum_descriptors
export smiles_to_graph, atom_features
export SMILES_OUTPUT_DIM, GNN_OUTPUT_DIM, QUANTUM_OUTPUT_DIM, FUSION_DIM
export CHEMBERTA_DIM, DMPNN_DIM

end # module
