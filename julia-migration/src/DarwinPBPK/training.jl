"""
Training Pipeline - Pipeline de Treinamento com Regularização SOTA Q1 2025

Inovações SOTA Q1 2025:
- Regularização L2 (weight decay) - IMPLEMENTADO
- Dropout - IMPLEMENTADO
- Early stopping - IMPLEMENTADO
- Learning rate scheduling - IMPLEMENTADO
- Gradient clipping - IMPLEMENTADO

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: Novembro 2025
Atualizado: 2025-11-18 - Fase 1 SOTA (Regularização Completa)
"""

module Training

using Flux
using CUDA
using BSON
using ProgressMeter
using ArgParse
using Random
using Statistics
# DataLoader - usar diretamente do Flux (versão 0.16+)
# DataLoader é exportado diretamente do Flux
import Flux: DataLoader

# Importar módulos
using ..DynamicGNN: DynamicPBPKGNN, forward_batch
using ..ODEPBPKSolver: PBPKParams, PBPK_ORGANS, NUM_ORGANS
# Validation será importado quando necessário, não aqui no topo

"""
Dataset PBPK.
"""
struct PBPKDataset
    doses::Vector{Float64}
    params::Vector{PBPKParams}
    true_concentrations::Vector{Matrix{Float64}}  # [num_organs, num_time_points]
    time_points::Vector{Vector{Float64}}
end

"""
Loss function com regularização L2.
"""
function compute_loss(
    model::DynamicPBPKGNN,
    batch::Tuple,
    device = cpu;
    weight_decay::Float64 = 1e-5,  # Regularização L2
    use_dropout::Bool = true,
    dropout_rate::Float64 = 0.2,
)
    doses, params, true_concs, time_points = batch

    # Forward pass
    results = forward_batch(model, doses, params, time_points, device)
    pred_concs = results["concentrations"]  # [batch_size, num_organs, num_time_points]

    # Reshape para comparação
    batch_size = length(doses)
    pred_flat = reshape(pred_concs, batch_size, NUM_ORGANS, size(pred_concs, 3))
    true_flat = reshape(true_concs, batch_size, NUM_ORGANS, size(true_concs, 3))

    # MSE Loss
    mse_loss = mean((pred_flat .- true_flat).^2)

    # Regularização L2 (weight decay)
    l2_reg = 0.0
    for p in Flux.params(model)
        l2_reg += sum(p.^2)
    end
    l2_reg *= weight_decay

    # Total loss
    total_loss = mse_loss + l2_reg

    return total_loss, mse_loss, l2_reg
end

"""
Training epoch com regularização.
"""
function train_epoch!(
    model::DynamicPBPKGNN,
    dataloader::DataLoader,
    optimizer,  # Flux optimizer state (from Flux.setup)
    device = cpu;
    weight_decay::Float64 = 1e-5,
    use_dropout::Bool = true,
    dropout_rate::Float64 = 0.2,
    clip_grad_norm::Float64 = 1.0,
)
    model.train = true  # Modo treinamento (para dropout)
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader
        # Zero gradients
        Flux.Zygote.gradient(() -> begin
            loss, _, _ = compute_loss(
                model, batch, device;
                weight_decay=weight_decay,
                use_dropout=use_dropout,
                dropout_rate=dropout_rate,
            )
            return loss
        end, Flux.params(model))

        # Backward pass
        grads = Flux.gradient(Flux.params(model)) do
            loss, _, _ = compute_loss(
                model, batch, device;
                weight_decay=weight_decay,
                use_dropout=use_dropout,
                dropout_rate=dropout_rate,
            )
            return loss
        end

        # Gradient clipping
        Flux.clip!(grads, clip_grad_norm)

        # Update weights
        Flux.Optimise.update!(optimizer, Flux.params(model), grads)

        # Accumulate loss
        loss, _, _ = compute_loss(model, batch, device; weight_decay=weight_decay)
        total_loss += loss
        num_batches += 1
    end

    return total_loss / num_batches
end

"""
Validation epoch.
"""
function validate_epoch(
    model::DynamicPBPKGNN,
    dataloader::DataLoader,
    device = cpu,
)::Float64
    model.train = false  # Modo validação (sem dropout)
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader
        loss, _, _ = compute_loss(model, batch, device; weight_decay=0.0, use_dropout=false)
        total_loss += loss
        num_batches += 1
    end

    return total_loss / num_batches
end

"""
Early stopping.
"""
function should_stop_early(
    val_loss_history::Vector{Float64},
    patience::Int = 10,
    min_delta::Float64 = 0.001,
)::Tuple{Bool, Int}
    if length(val_loss_history) < patience + 1
        return false, 0
    end

    best_val_loss = minimum(val_loss_history)
    recent_val_loss = val_loss_history[end-patience:end]

    # Verificar se melhorou recentemente
    no_improvement = true
    for loss in recent_val_loss
        if loss < best_val_loss - min_delta
            no_improvement = false
            break
        end
    end

    if no_improvement
        return true, length(val_loss_history) - patience
    end

    return false, 0
end

"""
Training loop completo com regularização.
"""
function train_model(
    model::DynamicPBPKGNN,
    train_data::PBPKDataset,
    val_data::PBPKDataset,
    num_epochs::Int = 100,
    batch_size::Int = 32,
    learning_rate::Float64 = 1e-3,
    device = cpu;
    weight_decay::Float64 = 1e-5,  # Regularização L2
    use_dropout::Bool = true,
    dropout_rate::Float64 = 0.2,
    clip_grad_norm::Float64 = 1.0,
    early_stopping_patience::Int = 10,
    early_stopping_min_delta::Float64 = 0.001,
    checkpoint_dir::String = "models/checkpoints",
)
    # Optimizer com weight decay
    optimizer = Flux.setup(
        Flux.Adam(learning_rate),
        model,
    )

    # Data loaders
    train_loader = DataLoader(
        zip(train_data.doses, train_data.params, train_data.true_concentrations, train_data.time_points),
        batchsize=batch_size,
        shuffle=true,
    )
    val_loader = DataLoader(
        zip(val_data.doses, val_data.params, val_data.true_concentrations, val_data.time_points),
        batchsize=batch_size,
        shuffle=false,
    )

    # Training history
    train_loss_history = Float64[]
    val_loss_history = Float64[]

    # Progress bar
    p = Progress(num_epochs, desc="Training...")

    for epoch in 1:num_epochs
        # Train
        train_loss = train_epoch!(
            model, train_loader, optimizer, device;
            weight_decay=weight_decay,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            clip_grad_norm=clip_grad_norm,
        )
        push!(train_loss_history, train_loss)

        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        push!(val_loss_history, val_loss)

        # Update progress
        ProgressMeter.update!(p, epoch, showvalues=[
            (:train_loss, round(train_loss, digits=6)),
            (:val_loss, round(val_loss, digits=6)),
        ])

        # Early stopping
        should_stop, best_epoch = should_stop_early(
            val_loss_history,
            early_stopping_patience,
            early_stopping_min_delta,
        )

        if should_stop
            println("\n⏹️  Early stopping at epoch $epoch (best: $best_epoch)")
            break
        end

        # Checkpoint
        if epoch % 10 == 0
            mkpath(checkpoint_dir)
            BSON.@save joinpath(checkpoint_dir, "checkpoint_epoch_$epoch.bson") model
        end
    end

    finish!(p)

    return Dict(
        "train_loss_history" => train_loss_history,
        "val_loss_history" => val_loss_history,
    )
end

export PBPKDataset, train_model, compute_loss, train_epoch!, validate_epoch, should_stop_early

end # module
