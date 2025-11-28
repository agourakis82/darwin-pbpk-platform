"""
Training Pipeline - Pipeline de Treinamento com Regularização SOTA Q1 2025

Inovações SOTA Q1 2025:
- Regularização L2 (weight decay) - IMPLEMENTADO
- Dropout - IMPLEMENTADO
- Early stopping - IMPLEMENTADO
- Learning rate scheduling (Cosine Annealing with Warmup) - IMPLEMENTADO
- Gradient clipping - IMPLEMENTADO

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: Novembro 2025
Atualizado: 2025-11-28 - LR Scheduling com Cosine Annealing
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

# ============================================================================
# Learning Rate Schedulers - SOTA Q1 2025
# ============================================================================

"""
Abstract type for learning rate schedulers.
"""
abstract type LRScheduler end

"""
Cosine Annealing with Warm Restarts (SGDR).

Reference: Loshchilov & Hutter, 2017 - SGDR: Stochastic Gradient Descent with Warm Restarts

Parameters:
- η_max: Maximum learning rate
- η_min: Minimum learning rate
- T_0: Initial period (epochs before first restart)
- T_mult: Period multiplier after each restart
- warmup_epochs: Number of warmup epochs (linear warmup)
"""
struct CosineAnnealingWarmRestarts <: LRScheduler
    η_max::Float64
    η_min::Float64
    T_0::Int
    T_mult::Int
    warmup_epochs::Int
end

function CosineAnnealingWarmRestarts(;
    η_max::Float64=1e-3,
    η_min::Float64=1e-6,
    T_0::Int=10,
    T_mult::Int=2,
    warmup_epochs::Int=5,
)
    return CosineAnnealingWarmRestarts(η_max, η_min, T_0, T_mult, warmup_epochs)
end

"""
Get learning rate for given epoch using cosine annealing with warm restarts.
"""
function get_lr(scheduler::CosineAnnealingWarmRestarts, epoch::Int)::Float64
    # Warmup phase: linear increase from η_min to η_max
    if epoch <= scheduler.warmup_epochs
        α = epoch / scheduler.warmup_epochs
        return scheduler.η_min + α * (scheduler.η_max - scheduler.η_min)
    end

    # Adjust epoch for warmup
    epoch_adj = epoch - scheduler.warmup_epochs

    # Find current cycle
    T_cur = scheduler.T_0
    T_i = 0
    cycle = 0

    while T_i + T_cur <= epoch_adj
        T_i += T_cur
        T_cur *= scheduler.T_mult
        cycle += 1
    end

    # Position within current cycle [0, 1]
    t = (epoch_adj - T_i) / T_cur

    # Cosine annealing
    η = scheduler.η_min + 0.5 * (scheduler.η_max - scheduler.η_min) * (1 + cos(π * t))

    return η
end

"""
Step Learning Rate Scheduler.

Reduces learning rate by factor gamma every step_size epochs.
"""
struct StepLR <: LRScheduler
    η_init::Float64
    step_size::Int
    gamma::Float64
    warmup_epochs::Int
end

function StepLR(;
    η_init::Float64=1e-3,
    step_size::Int=30,
    gamma::Float64=0.1,
    warmup_epochs::Int=5,
)
    return StepLR(η_init, step_size, gamma, warmup_epochs)
end

function get_lr(scheduler::StepLR, epoch::Int)::Float64
    # Warmup phase
    if epoch <= scheduler.warmup_epochs
        α = epoch / scheduler.warmup_epochs
        η_min = scheduler.η_init * 0.01
        return η_min + α * (scheduler.η_init - η_min)
    end

    epoch_adj = epoch - scheduler.warmup_epochs
    num_steps = epoch_adj ÷ scheduler.step_size
    return scheduler.η_init * (scheduler.gamma^num_steps)
end

"""
One Cycle Learning Rate Scheduler (Smith, 2018).

Increases LR linearly for first half, then decreases with cosine annealing.
Best for training from scratch.
"""
struct OneCycleLR <: LRScheduler
    η_max::Float64
    η_min::Float64
    total_epochs::Int
    pct_start::Float64  # Percentage of cycle spent increasing LR
end

function OneCycleLR(;
    η_max::Float64=1e-3,
    η_min::Float64=1e-6,
    total_epochs::Int=100,
    pct_start::Float64=0.3,
)
    return OneCycleLR(η_max, η_min, total_epochs, pct_start)
end

function get_lr(scheduler::OneCycleLR, epoch::Int)::Float64
    pct = epoch / scheduler.total_epochs

    if pct <= scheduler.pct_start
        # Increasing phase: linear warmup
        α = pct / scheduler.pct_start
        return scheduler.η_min + α * (scheduler.η_max - scheduler.η_min)
    else
        # Decreasing phase: cosine annealing
        α = (pct - scheduler.pct_start) / (1.0 - scheduler.pct_start)
        return scheduler.η_min + 0.5 * (scheduler.η_max - scheduler.η_min) * (1 + cos(π * α))
    end
end

"""
Update optimizer learning rate.
"""
function set_lr!(optimizer, lr::Float64)
    # For Flux.Optimisers, we need to recreate with new LR
    # This is a workaround since Flux doesn't support direct LR modification
    return Flux.Adam(lr)
end

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
    device=cpu;
    weight_decay::Float64=1e-5,  # Regularização L2
    use_dropout::Bool=true,
    dropout_rate::Float64=0.2,
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
    mse_loss = mean((pred_flat .- true_flat) .^ 2)

    # Regularização L2 (weight decay)
    l2_reg = 0.0
    for p in Flux.params(model)
        l2_reg += sum(p .^ 2)
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
    device=cpu;
    weight_decay::Float64=1e-5,
    use_dropout::Bool=true,
    dropout_rate::Float64=0.2,
    clip_grad_norm::Float64=1.0,
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
    device=cpu,
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
    patience::Int=10,
    min_delta::Float64=0.001,
)::Tuple{Bool,Int}
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
Training loop completo com regularização e LR scheduling.
"""
function train_model(
    model::DynamicPBPKGNN,
    train_data::PBPKDataset,
    val_data::PBPKDataset,
    num_epochs::Int=100,
    batch_size::Int=32,
    learning_rate::Float64=1e-3,
    device=cpu;
    weight_decay::Float64=1e-5,  # Regularização L2
    use_dropout::Bool=true,
    dropout_rate::Float64=0.2,
    clip_grad_norm::Float64=1.0,
    early_stopping_patience::Int=10,
    early_stopping_min_delta::Float64=0.001,
    checkpoint_dir::String="models/checkpoints",
    lr_scheduler::Union{LRScheduler,Nothing}=nothing,
    lr_scheduler_type::Symbol=:cosine,  # :cosine, :step, :onecycle, :none
)
    # Create LR scheduler if not provided
    if lr_scheduler === nothing && lr_scheduler_type != :none
        lr_scheduler = if lr_scheduler_type == :cosine
            CosineAnnealingWarmRestarts(
                η_max=learning_rate,
                η_min=learning_rate * 0.001,
                T_0=max(10, num_epochs ÷ 10),
                T_mult=2,
                warmup_epochs=max(3, num_epochs ÷ 20),
            )
        elseif lr_scheduler_type == :step
            StepLR(
                η_init=learning_rate,
                step_size=max(10, num_epochs ÷ 5),
                gamma=0.1,
                warmup_epochs=max(3, num_epochs ÷ 20),
            )
        elseif lr_scheduler_type == :onecycle
            OneCycleLR(
                η_max=learning_rate,
                η_min=learning_rate * 0.001,
                total_epochs=num_epochs,
                pct_start=0.3,
            )
        else
            nothing
        end
    end

    # Initial optimizer with scheduled LR
    current_lr = lr_scheduler !== nothing ? get_lr(lr_scheduler, 1) : learning_rate
    optimizer = Flux.setup(
        Flux.Adam(current_lr),
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
    lr_history = Float64[]

    # Progress bar
    p = Progress(num_epochs, desc="Training...")

    for epoch in 1:num_epochs
        # Update learning rate from scheduler
        if lr_scheduler !== nothing
            current_lr = get_lr(lr_scheduler, epoch)
            # Recreate optimizer with new LR (Flux doesn't support in-place LR changes)
            optimizer = Flux.setup(Flux.Adam(current_lr), model)
            push!(lr_history, current_lr)
        end

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

        # Update progress with LR info
        lr_display = lr_scheduler !== nothing ? current_lr : learning_rate
        ProgressMeter.update!(p, epoch, showvalues=[
            (:train_loss, round(train_loss, digits=6)),
            (:val_loss, round(val_loss, digits=6)),
            (:lr, round(lr_display, sigdigits=3)),
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
        "lr_history" => lr_history,
    )
end

# Export LR schedulers and training functions
export LRScheduler, CosineAnnealingWarmRestarts, StepLR, OneCycleLR, get_lr
export PBPKDataset, train_model, compute_loss, train_epoch!, validate_epoch, should_stop_early

end # module
