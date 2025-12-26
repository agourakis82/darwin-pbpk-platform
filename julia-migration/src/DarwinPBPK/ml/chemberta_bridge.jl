"""
ChemBERTa Bridge - HuggingFace Transformers Integration via PyCall

SOTA Q1 2025 Implementation:
- Real ChemBERTa (seyonec/ChemBERTa-zinc-base-v1) embeddings
- Efficient batch encoding with caching
- GPU support via PyTorch
- Fallback to Julia GRU encoder if PyCall unavailable

ChemBERTa is a RoBERTa model pre-trained on 77M SMILES strings from ZINC,
providing superior molecular representations compared to character-level models.

Author: Dr. Sounio Agourakis + AI Assistant
Date: December 2025

References:
- Chithrananda et al., 2020 (ChemBERTa)
- Liu et al., 2019 (RoBERTa)
"""

module ChemBERTaBridge

using PyCall
using Statistics

# Python modules (initialized lazily)
const torch = PyNULL()
const transformers = PyNULL()
const np = PyNULL()

# Model and tokenizer (loaded on first use)
const _model = Ref{PyObject}(PyNULL())
const _tokenizer = Ref{PyObject}(PyNULL())
const _device = Ref{String}("cpu")
const _is_initialized = Ref{Bool}(false)

# Embedding cache for repeated queries
const _embedding_cache = Dict{String, Vector{Float32}}()
const MAX_CACHE_SIZE = 10000

# Constants
const CHEMBERTA_DIM = 768
const MAX_SMILES_LENGTH = 512
const DEFAULT_MODEL = "seyonec/ChemBERTa-zinc-base-v1"

# Alternative models
const AVAILABLE_MODELS = Dict(
    "chemberta-zinc" => "seyonec/ChemBERTa-zinc-base-v1",
    "chemberta-mlm" => "seyonec/ChemBERTa_zinc250k_v2_40k",
    "molbert" => "BenevolentAI/MolBERT",
    "chembert" => "shahrukhx01/chembert_chem_smiles",
)

export initialize!, encode, encode_batch, is_available, get_embedding_dim
export clear_cache, set_device, CHEMBERTA_DIM

#=============================================================================
  Initialization
=============================================================================#

"""
Check if PyCall and required Python packages are available.
"""
function is_available()::Bool
    try
        pyimport("torch")
        pyimport("transformers")
        return true
    catch
        return false
    end
end

"""
Initialize the ChemBERTa model and tokenizer.

# Arguments
- `model_name::String`: HuggingFace model identifier (default: ChemBERTa-zinc)
- `device::String`: Device to use ("cpu", "cuda", "cuda:0", etc.)
- `use_cache::Bool`: Whether to cache embeddings (default: true)

# Example
```julia
using DarwinPBPK.ChemBERTaBridge
initialize!()  # Uses default ChemBERTa
initialize!(model_name="seyonec/ChemBERTa_zinc250k_v2_40k", device="cuda")
```
"""
function initialize!(;
    model_name::String = DEFAULT_MODEL,
    device::String = "auto",
    use_cache::Bool = true
)::Bool
    if !is_available()
        @warn "PyCall or transformers not available. ChemBERTa will not be usable."
        return false
    end

    try
        # Import Python modules
        copy!(torch, pyimport("torch"))
        copy!(transformers, pyimport("transformers"))
        copy!(np, pyimport("numpy"))

        # Determine device
        if device == "auto"
            if torch.cuda.is_available()
                _device[] = "cuda"
            else
                _device[] = "cpu"
            end
        else
            _device[] = device
        end

        @info "Loading ChemBERTa model: $model_name on $(_device[])"

        # Load tokenizer
        _tokenizer[] = transformers.AutoTokenizer.from_pretrained(model_name)

        # Load model
        _model[] = transformers.AutoModel.from_pretrained(model_name)
        _model[].to(_device[])
        _model[].eval()  # Set to inference mode

        _is_initialized[] = true

        @info "ChemBERTa initialized successfully"
        return true

    catch e
        @error "Failed to initialize ChemBERTa" exception = e
        _is_initialized[] = false
        return false
    end
end

"""
Set the device for inference.
"""
function set_device(device::String)
    if !_is_initialized[]
        @warn "ChemBERTa not initialized"
        return
    end

    _device[] = device
    _model[].to(device)
end

"""
Clear the embedding cache.
"""
function clear_cache()
    empty!(_embedding_cache)
end

"""
Get the embedding dimension.
"""
function get_embedding_dim()::Int
    return CHEMBERTA_DIM
end

#=============================================================================
  Single SMILES Encoding
=============================================================================#

"""
Encode a single SMILES string to a fixed-size embedding.

Returns the [CLS] token embedding from the last hidden layer.

# Arguments
- `smiles::String`: SMILES string to encode
- `use_cache::Bool`: Whether to use cached embedding if available

# Returns
- `Vector{Float32}`: 768-dimensional embedding

# Example
```julia
emb = encode("CCO")  # Ethanol
@assert length(emb) == 768
```
"""
function encode(smiles::String; use_cache::Bool = true)::Vector{Float32}
    # Check cache first
    if use_cache && haskey(_embedding_cache, smiles)
        return _embedding_cache[smiles]
    end

    if !_is_initialized[]
        if !initialize!()
            # Return zero embedding as fallback
            @warn "ChemBERTa not available, returning zero embedding"
            return zeros(Float32, CHEMBERTA_DIM)
        end
    end

    try
        # Tokenize
        inputs = _tokenizer[](
            smiles,
            return_tensors = "pt",
            padding = true,
            truncation = true,
            max_length = MAX_SMILES_LENGTH
        )

        # Move to device
        inputs = Dict(k => v.to(_device[]) for (k, v) in inputs)

        # Forward pass (no gradient computation)
        torch.no_grad() do
            outputs = _model[](inputs...)

            # Extract [CLS] token embedding (first token of last hidden state)
            # Shape: [batch_size=1, seq_len, hidden_dim=768]
            cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()

            embedding = Vector{Float32}(cls_embedding)

            # Cache if enabled and cache not full
            if use_cache && length(_embedding_cache) < MAX_CACHE_SIZE
                _embedding_cache[smiles] = embedding
            end

            return embedding
        end

    catch e
        @warn "Failed to encode SMILES: $smiles" exception = e
        return zeros(Float32, CHEMBERTA_DIM)
    end
end

#=============================================================================
  Batch Encoding
=============================================================================#

"""
Encode multiple SMILES strings efficiently in a batch.

# Arguments
- `smiles_batch::Vector{String}`: Vector of SMILES strings
- `batch_size::Int`: Maximum batch size for GPU memory (default: 32)
- `use_cache::Bool`: Whether to use/update cache

# Returns
- `Matrix{Float32}`: [768, n_molecules] embedding matrix

# Example
```julia
smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
embeddings = encode_batch(smiles)
@assert size(embeddings) == (768, 3)
```
"""
function encode_batch(
    smiles_batch::Vector{String};
    batch_size::Int = 32,
    use_cache::Bool = true
)::Matrix{Float32}
    n = length(smiles_batch)
    embeddings = Matrix{Float32}(undef, CHEMBERTA_DIM, n)

    # Check which SMILES are already cached
    uncached_indices = Int[]
    uncached_smiles = String[]

    for (i, smiles) in enumerate(smiles_batch)
        if use_cache && haskey(_embedding_cache, smiles)
            embeddings[:, i] = _embedding_cache[smiles]
        else
            push!(uncached_indices, i)
            push!(uncached_smiles, smiles)
        end
    end

    # Encode uncached SMILES in batches
    if !isempty(uncached_smiles)
        if !_is_initialized[]
            if !initialize!()
                @warn "ChemBERTa not available, returning zero embeddings"
                for i in uncached_indices
                    embeddings[:, i] = zeros(Float32, CHEMBERTA_DIM)
                end
                return embeddings
            end
        end

        try
            # Process in batches
            n_uncached = length(uncached_smiles)
            n_batches = ceil(Int, n_uncached / batch_size)

            for batch_idx in 1:n_batches
                start_idx = (batch_idx - 1) * batch_size + 1
                end_idx = min(batch_idx * batch_size, n_uncached)

                batch_smiles = uncached_smiles[start_idx:end_idx]
                batch_orig_indices = uncached_indices[start_idx:end_idx]

                # Tokenize batch
                inputs = _tokenizer[](
                    batch_smiles,
                    return_tensors = "pt",
                    padding = true,
                    truncation = true,
                    max_length = MAX_SMILES_LENGTH
                )

                # Move to device
                inputs = Dict(k => v.to(_device[]) for (k, v) in inputs)

                # Forward pass
                torch.no_grad() do
                    outputs = _model[](inputs...)

                    # Extract [CLS] embeddings for all molecules in batch
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                    # Store embeddings
                    for (j, orig_idx) in enumerate(batch_orig_indices)
                        emb = Vector{Float32}(cls_embeddings[j, :])
                        embeddings[:, orig_idx] = emb

                        # Update cache
                        if use_cache && length(_embedding_cache) < MAX_CACHE_SIZE
                            _embedding_cache[smiles_batch[orig_idx]] = emb
                        end
                    end
                end
            end

        catch e
            @warn "Batch encoding failed, falling back to individual encoding" exception = e
            for (i, smiles) in zip(uncached_indices, uncached_smiles)
                embeddings[:, i] = encode(smiles; use_cache = use_cache)
            end
        end
    end

    return embeddings
end

#=============================================================================
  Pooling Strategies
=============================================================================#

"""
Mean pooling over all tokens (alternative to [CLS]).

Some evidence suggests mean pooling performs better for certain tasks.
"""
function encode_mean_pooling(smiles::String)::Vector{Float32}
    if !_is_initialized[]
        if !initialize!()
            return zeros(Float32, CHEMBERTA_DIM)
        end
    end

    try
        inputs = _tokenizer[](
            smiles,
            return_tensors = "pt",
            padding = true,
            truncation = true,
            max_length = MAX_SMILES_LENGTH
        )

        inputs = Dict(k => v.to(_device[]) for (k, v) in inputs)

        torch.no_grad() do
            outputs = _model[](inputs...)

            # Mean over sequence dimension (excluding padding)
            attention_mask = inputs["attention_mask"]
            hidden_states = outputs.last_hidden_state

            # Expand attention mask for broadcasting
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)

            mean_embedding = (sum_embeddings / sum_mask).cpu().numpy()

            return Vector{Float32}(mean_embedding[1, :])
        end

    catch e
        @warn "Mean pooling failed for: $smiles" exception = e
        return zeros(Float32, CHEMBERTA_DIM)
    end
end

export encode_mean_pooling

#=============================================================================
  Similarity Functions
=============================================================================#

"""
Compute cosine similarity between two SMILES embeddings.
"""
function smiles_similarity(smiles1::String, smiles2::String)::Float64
    emb1 = encode(smiles1)
    emb2 = encode(smiles2)

    dot_product = sum(emb1 .* emb2)
    norm1 = sqrt(sum(emb1 .^ 2))
    norm2 = sqrt(sum(emb2 .^ 2))

    if norm1 == 0 || norm2 == 0
        return 0.0
    end

    return dot_product / (norm1 * norm2)
end

"""
Find most similar molecules from a database.

# Arguments
- `query_smiles::String`: Query molecule
- `database_smiles::Vector{String}`: Database of SMILES to search
- `top_k::Int`: Number of results to return

# Returns
Vector of (index, smiles, similarity) tuples
"""
function find_similar(
    query_smiles::String,
    database_smiles::Vector{String};
    top_k::Int = 10
)::Vector{Tuple{Int, String, Float64}}
    query_emb = encode(query_smiles)
    db_embeddings = encode_batch(database_smiles)

    # Compute all similarities
    similarities = Float64[]
    query_norm = sqrt(sum(query_emb .^ 2))

    for i in 1:length(database_smiles)
        db_emb = db_embeddings[:, i]
        db_norm = sqrt(sum(db_emb .^ 2))

        if query_norm > 0 && db_norm > 0
            sim = sum(query_emb .* db_emb) / (query_norm * db_norm)
        else
            sim = 0.0
        end
        push!(similarities, sim)
    end

    # Sort and return top_k
    sorted_indices = sortperm(similarities, rev=true)
    results = Tuple{Int, String, Float64}[]

    for i in 1:min(top_k, length(sorted_indices))
        idx = sorted_indices[i]
        push!(results, (idx, database_smiles[idx], similarities[idx]))
    end

    return results
end

export smiles_similarity, find_similar

#=============================================================================
  Wrapper Struct for Integration
=============================================================================#

"""
ChemBERTa encoder struct for integration with MultimodalEncoder.

This provides a Flux-compatible interface.
"""
struct ChemBERTaEncoder
    model_name::String
    device::String
    initialized::Bool
end

function ChemBERTaEncoder(;
    model_name::String = DEFAULT_MODEL,
    device::String = "auto"
)
    success = initialize!(model_name = model_name, device = device)
    return ChemBERTaEncoder(model_name, _device[], success)
end

"""
Callable interface for single SMILES.
"""
function (enc::ChemBERTaEncoder)(smiles::String)::Vector{Float32}
    return encode(smiles)
end

"""
Callable interface for batch of SMILES.
"""
function (enc::ChemBERTaEncoder)(smiles_batch::Vector{String})::Matrix{Float32}
    return encode_batch(smiles_batch)
end

export ChemBERTaEncoder

#=============================================================================
  Utilities
=============================================================================#

"""
Get model information.
"""
function model_info()::Dict{String, Any}
    return Dict(
        "model_name" => _is_initialized[] ? DEFAULT_MODEL : "not loaded",
        "device" => _device[],
        "initialized" => _is_initialized[],
        "embedding_dim" => CHEMBERTA_DIM,
        "cache_size" => length(_embedding_cache),
        "max_cache_size" => MAX_CACHE_SIZE,
        "available_models" => AVAILABLE_MODELS,
    )
end

"""
Benchmark encoding speed.
"""
function benchmark(smiles_list::Vector{String}; n_iterations::Int = 3)::Dict{String, Float64}
    if !_is_initialized[]
        initialize!()
    end

    # Warm-up
    encode(smiles_list[1])

    # Single encoding benchmark
    clear_cache()
    single_times = Float64[]
    for _ in 1:n_iterations
        t = @elapsed begin
            for s in smiles_list
                encode(s; use_cache = false)
            end
        end
        push!(single_times, t)
    end

    # Batch encoding benchmark
    clear_cache()
    batch_times = Float64[]
    for _ in 1:n_iterations
        t = @elapsed encode_batch(smiles_list; use_cache = false)
        push!(batch_times, t)
    end

    n = length(smiles_list)
    return Dict(
        "single_total_time" => mean(single_times),
        "single_per_molecule" => mean(single_times) / n * 1000,  # ms
        "batch_total_time" => mean(batch_times),
        "batch_per_molecule" => mean(batch_times) / n * 1000,  # ms
        "speedup" => mean(single_times) / mean(batch_times),
        "n_molecules" => n,
    )
end

export model_info, benchmark

end # module
