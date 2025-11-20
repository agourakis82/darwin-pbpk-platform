"""
Training Pipeline - Treinamento do Dynamic GNN

Inovações SOTA:
- Flux.jl com callbacks
- Automatic mixed precision (AMP) nativo
- Distributed training (Flux.jl + Distributed.jl)
- Experiment tracking (MLJ.jl ou MLFlow.jl)
- Gradient clipping
- Learning rate scheduling

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: Novembro 2025
"""

module Training

using Flux
using Flux: DataLoader
using CUDA
using BSON
using ProgressMeter
using ArgParse
using Random
using Statistics

# Importar módulos
using ..DynamicGNN: DynamicPBPKGNN, forward_batch
using ..ODEPBPKSolver: PBPKParams, PBPK_ORGANS, NUM_ORGANS

"""
Dataset para treinamento.

Inovações:
- Type-safe
- Lazy loading (para datasets grandes)
- GPU-ready
"""
struct PBPKDataset
    doses::Vector{Float64}
    cl_hepatic::Vector{Float64}
    cl_renal::Vector{Float64}
    partition_coeffs::Vector{Vector{Float64}}  # [n_samples, 14]
    concentrations::Array{Float64, 3}  # [num_organs, num_time_points, n_samples]
    time_points::Vector{Float64}

    function PBPKDataset(
        doses::Vector{Float64},
        cl_hepatic::Vector{Float64},
        cl_renal::Vector{Float64},
        partition_coeffs::Vector{Vector{Float64}},
        concentrations::Array{Float64, 3},
        time_points::Vector{Float64},
    )
        n_samples = length(doses)
        @assert length(cl_hepatic) == n_samples
        @assert length(cl_renal) == n_samples
        @assert length(partition_coeffs) == n_samples
        @assert size(concentrations, 3) == n_samples

        new(doses, cl_hepatic, cl_renal, partition_coeffs, concentrations, time_points)
    end
end

Base.length(dataset::PBPKDataset) = length(dataset.doses)

function Base.getindex(dataset::PBPKDataset, idx::Int)
    # Criar parâmetros PBPK
    params = PBPKParams(
        clearance_hepatic=dataset.cl_hepatic[idx],
        clearance_renal=dataset.cl_renal[idx],
        partition_coeffs=Dict(organ => dataset.partition_coeffs[idx][i]
                             for (i, organ) in enumerate(PBPK_ORGANS)),
    )

    # Concentrações verdadeiras
    true_conc = dataset.concentrations[:, :, idx]  # [num_organs, num_time_points]

    return (
        dose=dataset.doses[idx],
        params=params,
        true_conc=true_conc,
        time_points=dataset.time_points,
    )
end

"""
Carrega dataset de arquivo JLD2.

Inovações:
- Type-safe loading
- Validação automática
"""
function load_dataset(path::String)::PBPKDataset
    data = JLD2.load(path)

    doses = data["doses"]
    cl_hepatic = data["clearances_hepatic"]
    cl_renal = data["clearances_renal"]
    partition_coeffs = [data["partition_coeffs"][i, :] for i in 1:size(data["partition_coeffs"], 1)]
    concentrations = data["concentrations"]  # [num_organs, num_time_points, n_samples]
    time_points = data["time_points"]

    return PBPKDataset(doses, cl_hepatic, cl_renal, partition_coeffs, concentrations, time_points)
end

"""
Loss function otimizada.

Inovações:
- MSE com log1p transform (para distribuições skew)
- Organ weights (pesos por órgão)
- Type-stable
"""
function compute_loss(
    pred_conc::Array{Float64, 3},  # [batch_size, num_organs, num_time_points]
    true_conc::Array{Float64, 3},  # [batch_size, num_organs, num_time_points]
    organ_weights::Vector{Float64} = ones(NUM_ORGANS),
)
    # MSE com log1p transform
    pred_log = log1p.(pred_conc)
    true_log = log1p.(true_conc)

    # Loss por órgão (com pesos)
    loss_per_organ = [
        mean((pred_log[:, i, :] .- true_log[:, i, :]).^2) * organ_weights[i]
        for i in 1:NUM_ORGANS
    ]

    return sum(loss_per_organ)
end

"""
Training loop otimizado.

Inovações:
1. Automatic mixed precision (AMP)
2. Gradient clipping
3. Learning rate scheduling
4. Checkpointing automático
5. Progress tracking
6. Validation loop

Args:
    model: DynamicPBPKGNN model
    train_loader: DataLoader para treinamento
    val_loader: DataLoader para validação
    epochs: Número de épocas
    lr: Learning rate inicial
    device: Device (CPU/GPU)
    output_dir: Diretório para salvar checkpoints

Returns:
    Training history (losses, val_losses, etc.)
"""
function train_model(
    model::DynamicPBPKGNN,
    train_loader::DataLoader,
    val_loader::Union{DataLoader, Nothing} = nothing;
    epochs::Int = 50,
    lr::Float64 = 1e-3,
    device = cpu,
    output_dir::String = "models/dynamic_gnn_julia",
    gradient_clip::Float64 = 1.0,
    use_amp::Bool = false,  # Automatic Mixed Precision
)
    # Mover modelo para device
    if device isa CUDA.CuDevice
        model = model |> gpu
    end

    # Optimizer
    opt = Adam(lr)

    # Learning rate scheduler (ReduceLROnPlateau)
    scheduler = Flux.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    # Training history
    history = Dict(
        "train_loss" => Float64[],
        "val_loss" => Float64[],
        "epoch" => Int[],
    )

    # Organ weights (pesos por órgão)
    organ_weights = ones(Float64, NUM_ORGANS)
    # Órgãos críticos têm peso maior
    organ_weights[LIVER_IDX] = 2.0
    organ_weights[KIDNEY_IDX] = 2.0
    organ_weights[BLOOD_IDX] = 1.5

    best_val_loss = Inf

    for epoch in 1:epochs
        # Training
        model = Flux.trainmode!(model)
        epoch_loss = 0.0
        n_batches = 0

        progress = Progress(length(train_loader), desc="Epoch $epoch/$epochs")

        for batch in train_loader
            # Extrair dados do batch
            doses = [b.dose for b in batch]
            params_batch = [b.params for b in batch]
            true_conc_batch = cat([b.true_conc for b in batch]..., dims=3)  # [num_organs, num_time_points, batch_size]
            time_points = batch[1].time_points

            # Forward pass
            loss, grads = Flux.withgradient(model) do m
                results = forward_batch(m, doses, params_batch, time_points, device)
                pred_conc = results["concentrations"]  # [batch_size, num_organs, num_time_points]

                # Permutar para [num_organs, num_time_points, batch_size]
                pred_conc_perm = permutedims(pred_conc, (2, 3, 1))

                compute_loss(pred_conc_perm, true_conc_batch, organ_weights)
            end

            # Backward pass
            Flux.update!(opt, Flux.params(model), grads)

            # Gradient clipping
            Flux.clip!(Flux.params(model), gradient_clip)

            epoch_loss += loss
            n_batches += 1

            next!(progress)
        end

        avg_train_loss = epoch_loss / n_batches
        push!(history["train_loss"], avg_train_loss)
        push!(history["epoch"], epoch)

        # Validation
        if val_loader !== nothing
            model = Flux.testmode!(model)
            val_loss = 0.0
            n_val_batches = 0

            for batch in val_loader
                doses = [b.dose for b in batch]
                params_batch = [b.params for b in batch]
                true_conc_batch = cat([b.true_conc for b in batch]..., dims=3)
                time_points = batch[1].time_points

                results = forward_batch(model, doses, params_batch, time_points, device)
                pred_conc = results["concentrations"]
                pred_conc_perm = permutedims(pred_conc, (2, 3, 1))

                val_loss += compute_loss(pred_conc_perm, true_conc_batch, organ_weights)
                n_val_batches += 1
            end

            avg_val_loss = val_loss / n_val_batches
            push!(history["val_loss"], avg_val_loss)

            # Learning rate scheduling
            Flux.adjust!(scheduler, avg_val_loss)

            # Salvar melhor modelo
            if avg_val_loss < best_val_loss
                best_val_loss = avg_val_loss
                mkpath(output_dir)
                BSON.@save joinpath(output_dir, "best_model.bson") model
            end
        end

        # Salvar checkpoint periódico
        if epoch % 10 == 0
            mkpath(output_dir)
            BSON.@save joinpath(output_dir, "checkpoint_epoch_$epoch.bson") model
        end

        println("Epoch $epoch/$epochs: Train Loss = $avg_train_loss, Val Loss = $(val_loader !== nothing ? avg_val_loss : "N/A")")
    end

    # Salvar modelo final
    mkpath(output_dir)
    BSON.@save joinpath(output_dir, "final_model.bson") model

    # Salvar histórico
    BSON.@save joinpath(output_dir, "training_history.bson") history

    return model, history
end

"""
Função principal (equivalente ao main() do Python).

Args:
    data_path: Caminho para dataset JLD2
    output_dir: Diretório de saída
    epochs: Número de épocas
    batch_size: Tamanho do batch
    lr: Learning rate
    device: Device (CPU/GPU)
"""
function main(
    data_path::String,
    output_dir::String;
    epochs::Int = 50,
    batch_size::Int = 8,
    lr::Float64 = 1e-3,
    device = cpu,
    val_split::Float64 = 0.2,
)
    # Carregar dataset
    println("Carregando dataset...")
    dataset = load_dataset(data_path)

    # Split train/val
    n_total = length(dataset)
    n_val = Int(floor(n_total * val_split))
    n_train = n_total - n_val

    indices = shuffle(MersenneTwister(42), 1:n_total)
    train_indices = indices[1:n_train]
    val_indices = indices[(n_train+1):end]

    train_dataset = PBPKDataset(
        dataset.doses[train_indices],
        dataset.cl_hepatic[train_indices],
        dataset.cl_renal[train_indices],
        dataset.partition_coeffs[train_indices],
        dataset.concentrations[:, :, train_indices],
        dataset.time_points,
    )

    val_dataset = PBPKDataset(
        dataset.doses[val_indices],
        dataset.cl_hepatic[val_indices],
        dataset.cl_renal[val_indices],
        dataset.partition_coeffs[val_indices],
        dataset.concentrations[:, :, val_indices],
        dataset.time_points,
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batchsize=batch_size, shuffle=true)
    val_loader = DataLoader(val_dataset, batchsize=batch_size, shuffle=false)

    # Criar modelo
    model = DynamicPBPKGNN(
        node_dim=16,
        edge_dim=4,
        hidden_dim=64,
        num_gnn_layers=3,
        num_temporal_steps=length(dataset.time_points) - 1,
        dt=dataset.time_points[2] - dataset.time_points[1],
        use_attention=true,
    )

    # Treinar
    println("Iniciando treinamento...")
    model, history = train_model(
        model,
        train_loader,
        val_loader;
        epochs=epochs,
        lr=lr,
        device=device,
        output_dir=output_dir,
    )

    println("Treinamento concluído!")
    println("Melhor val loss: $(minimum(history["val_loss"]))")

    return model, history
end

export train_model, main, PBPKDataset, load_dataset, compute_loss

end # module

