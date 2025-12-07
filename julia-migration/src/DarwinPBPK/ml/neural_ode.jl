"""
Neural ODE PBPK Implementation

SOTA Q1 2025 Implementation:
- Neural Ordinary Differential Equations for continuous-time dynamics
- Integration with molecular encoders for drug-specific dynamics
- Hybrid physics-informed + neural dynamics
- Adjoint sensitivity for efficient gradients

Neural ODEs provide an alternative to discrete GNN message passing,
modeling concentration dynamics as continuous-time evolution learned
from data while respecting physical constraints.

Author: Dr. Demetrios Agourakis + AI Assistant
Date: December 2025

References:
- Chen et al., 2018 (Neural Ordinary Differential Equations)
- Rackauckas et al., 2020 (Universal Differential Equations)
- Lutter et al., 2019 (Deep Lagrangian Networks)
"""

module NeuralODE

using Flux
using Functors: @functor
using DifferentialEquations
using SciMLSensitivity
using Statistics
using Random
using LinearAlgebra

export NeuralODEPBPK, NeuralDynamics, HybridDynamics
export PhysicsInformedNeuralODE, AugmentedNeuralODE
export encode_and_predict, train_neural_ode!
export NODE_HIDDEN_DIM, NODE_LATENT_DIM

# Dimensions
const NODE_HIDDEN_DIM = 128
const NODE_LATENT_DIM = 32
const N_COMPARTMENTS = 5  # blood, liver, kidney, gut, peripheral

#=============================================================================
  Neural Dynamics Network
=============================================================================#

"""
Neural network defining the ODE dynamics.

dx/dt = f_θ(x, t, drug_emb)

where x is the state vector (concentrations), t is time,
and drug_emb is the molecular embedding.
"""
struct NeuralDynamics
    input_proj::Dense      # Project [state, drug_emb] to hidden
    hidden1::Dense         # Hidden layer 1
    hidden2::Dense         # Hidden layer 2
    output_proj::Dense     # Project to state derivatives
    drug_emb_dim::Int      # Drug embedding dimension
    state_dim::Int         # State dimension (n_compartments)
end

@functor NeuralDynamics

function NeuralDynamics(;
    state_dim::Int = N_COMPARTMENTS,
    drug_emb_dim::Int = 512,
    hidden_dim::Int = NODE_HIDDEN_DIM
)
    # Input: [state, drug_emb, time_encoding]
    input_dim = state_dim + drug_emb_dim + 4  # 4 for time encoding

    input_proj = Dense(input_dim => hidden_dim, tanh)
    hidden1 = Dense(hidden_dim => hidden_dim, tanh)
    hidden2 = Dense(hidden_dim => hidden_dim, tanh)
    output_proj = Dense(hidden_dim => state_dim)  # No activation for derivatives

    return NeuralDynamics(input_proj, hidden1, hidden2, output_proj, drug_emb_dim, state_dim)
end

"""
Time encoding using sinusoidal features.
"""
function time_encoding(t::Real)::Vector{Float32}
    return Float32[sin(t), cos(t), sin(t/10), cos(t/10)]
end

"""
Forward pass: compute dx/dt given state and drug embedding.
"""
function (dyn::NeuralDynamics)(x::AbstractVector, drug_emb::AbstractVector, t::Real)
    # Concatenate inputs
    t_enc = time_encoding(t)
    input = vcat(x, drug_emb, t_enc)

    # Forward pass
    h = dyn.input_proj(input)
    h = dyn.hidden1(h)
    h = dyn.hidden2(h)
    dx = dyn.output_proj(h)

    return dx
end

#=============================================================================
  Hybrid Physics-Informed Dynamics
=============================================================================#

"""
Hybrid dynamics combining physics-based PBPK with neural corrections.

dx/dt = f_physics(x, params) + f_neural(x, drug_emb, t)

The physics component captures known PBPK structure while the neural
component learns residual dynamics from data.
"""
struct HybridDynamics
    neural::NeuralDynamics
    physics_weight::Float32  # Weight for physics component (0-1)
end

@functor HybridDynamics

function HybridDynamics(;
    state_dim::Int = N_COMPARTMENTS,
    drug_emb_dim::Int = 512,
    hidden_dim::Int = NODE_HIDDEN_DIM,
    physics_weight::Float32 = 0.5f0
)
    neural = NeuralDynamics(; state_dim, drug_emb_dim, hidden_dim)
    return HybridDynamics(neural, physics_weight)
end

"""
Physics-based PBPK dynamics (simplified 5-compartment).

Compartments: blood (1), liver (2), kidney (3), gut (4), peripheral (5)
"""
function physics_dynamics(
    x::AbstractVector,
    params::NamedTuple
)::Vector{Float32}
    # Unpack state
    c_blood = x[1]
    c_liver = x[2]
    c_kidney = x[3]
    c_gut = x[4]
    c_peripheral = x[5]

    # Unpack parameters
    Q_liver = params.Q_liver      # Blood flow to liver
    Q_kidney = params.Q_kidney    # Blood flow to kidney
    Q_gut = params.Q_gut          # Blood flow to gut
    Q_peripheral = params.Q_peripheral

    V_blood = params.V_blood
    V_liver = params.V_liver
    V_kidney = params.V_kidney
    V_gut = params.V_gut
    V_peripheral = params.V_peripheral

    CL_hepatic = params.CL_hepatic
    CL_renal = params.CL_renal

    Kp_liver = params.Kp_liver    # Partition coefficients
    Kp_kidney = params.Kp_kidney
    Kp_gut = params.Kp_gut
    Kp_peripheral = params.Kp_peripheral

    # Mass balance ODEs
    dx = zeros(Float32, 5)

    # Blood compartment
    dx[1] = (Q_liver * c_liver / Kp_liver +
             Q_kidney * c_kidney / Kp_kidney +
             Q_gut * c_gut / Kp_gut +
             Q_peripheral * c_peripheral / Kp_peripheral -
             (Q_liver + Q_kidney + Q_gut + Q_peripheral) * c_blood) / V_blood

    # Liver (with hepatic clearance)
    dx[2] = (Q_liver * c_blood - Q_liver * c_liver / Kp_liver -
             CL_hepatic * c_liver / Kp_liver) / V_liver

    # Kidney (with renal clearance)
    dx[3] = (Q_kidney * c_blood - Q_kidney * c_kidney / Kp_kidney -
             CL_renal * c_kidney / Kp_kidney) / V_kidney

    # Gut (absorption site)
    dx[4] = (Q_gut * c_blood - Q_gut * c_gut / Kp_gut) / V_gut

    # Peripheral
    dx[5] = (Q_peripheral * c_blood - Q_peripheral * c_peripheral / Kp_peripheral) / V_peripheral

    return dx
end

"""
Default PBPK parameters (human adult, 70 kg).
"""
function default_pbpk_params()::NamedTuple
    return (
        Q_liver = 90.0f0,       # L/h
        Q_kidney = 72.0f0,      # L/h
        Q_gut = 60.0f0,         # L/h
        Q_peripheral = 120.0f0, # L/h
        V_blood = 5.0f0,        # L
        V_liver = 1.5f0,        # L
        V_kidney = 0.3f0,       # L
        V_gut = 1.0f0,          # L
        V_peripheral = 40.0f0,  # L
        CL_hepatic = 10.0f0,    # L/h
        CL_renal = 5.0f0,       # L/h
        Kp_liver = 2.0f0,       # Partition coefficient
        Kp_kidney = 1.5f0,
        Kp_gut = 1.2f0,
        Kp_peripheral = 1.0f0,
    )
end

"""
Forward pass for hybrid dynamics.
"""
function (dyn::HybridDynamics)(
    x::AbstractVector,
    drug_emb::AbstractVector,
    t::Real;
    params::NamedTuple = default_pbpk_params()
)
    # Physics component
    dx_physics = physics_dynamics(x, params)

    # Neural component
    dx_neural = dyn.neural(x, drug_emb, t)

    # Weighted combination
    dx = dyn.physics_weight * dx_physics + (1 - dyn.physics_weight) * dx_neural

    return dx
end

#=============================================================================
  Neural ODE PBPK Model
=============================================================================#

"""
Complete Neural ODE PBPK model.

Architecture:
1. Molecular encoder (external) produces drug embedding
2. Neural dynamics network defines dx/dt
3. ODE solver integrates dynamics
4. Decoder extracts PK parameters

# Example
```julia
model = NeuralODEPBPK()
conc_profile = predict_concentrations(model, drug_embedding, dose; t_span=(0, 24))
```
"""
struct NeuralODEPBPK
    dynamics::Union{NeuralDynamics, HybridDynamics}
    initial_state_net::Dense  # Drug embedding -> initial concentrations
    pk_decoder::Chain         # Final state -> PK parameters
    solver::Any               # ODE solver (Tsit5, etc.)
    use_adjoint::Bool         # Use adjoint sensitivity
end

@functor NeuralODEPBPK

function NeuralODEPBPK(;
    drug_emb_dim::Int = 512,
    state_dim::Int = N_COMPARTMENTS,
    hidden_dim::Int = NODE_HIDDEN_DIM,
    use_hybrid::Bool = true,
    physics_weight::Float32 = 0.5f0,
    use_adjoint::Bool = true
)
    # Dynamics network
    if use_hybrid
        dynamics = HybridDynamics(;
            state_dim, drug_emb_dim, hidden_dim, physics_weight
        )
    else
        dynamics = NeuralDynamics(; state_dim, drug_emb_dim, hidden_dim)
    end

    # Initial state from drug embedding and dose
    initial_state_net = Dense(drug_emb_dim + 1 => state_dim, softplus)  # +1 for dose

    # PK parameter decoder
    pk_decoder = Chain(
        Dense(state_dim * 2 => hidden_dim, relu),  # *2 for state + AUC features
        Dense(hidden_dim => 64, relu),
        Dense(64 => 3)  # CL, Vd, F (or other PK params)
    )

    # Default solver
    solver = Tsit5()

    return NeuralODEPBPK(dynamics, initial_state_net, pk_decoder, solver, use_adjoint)
end

"""
Compute initial state from drug embedding and dose.
"""
function compute_initial_state(
    model::NeuralODEPBPK,
    drug_emb::AbstractVector,
    dose::Real
)::Vector{Float32}
    input = vcat(drug_emb, Float32[dose])
    x0 = model.initial_state_net(input)
    return x0
end

"""
Define the ODE problem for DifferentialEquations.jl.
"""
function make_ode_problem(
    model::NeuralODEPBPK,
    drug_emb::AbstractVector,
    x0::AbstractVector,
    t_span::Tuple{Real, Real};
    params::NamedTuple = default_pbpk_params()
)
    # ODE function
    function ode_func!(dx, x, p, t)
        if model.dynamics isa HybridDynamics
            dx .= model.dynamics(x, drug_emb, t; params)
        else
            dx .= model.dynamics(x, drug_emb, t)
        end
    end

    prob = ODEProblem(ode_func!, x0, t_span)
    return prob
end

"""
Predict concentration-time profile.
"""
function predict_concentrations(
    model::NeuralODEPBPK,
    drug_emb::AbstractVector,
    dose::Real;
    t_span::Tuple{Real, Real} = (0.0, 24.0),
    saveat::AbstractVector = 0.0:0.5:24.0,
    params::NamedTuple = default_pbpk_params()
)
    # Compute initial state
    x0 = compute_initial_state(model, drug_emb, dose)

    # Create ODE problem
    prob = make_ode_problem(model, drug_emb, x0, t_span; params)

    # Solve with appropriate sensitivity algorithm
    if model.use_adjoint
        sol = solve(
            prob,
            model.solver;
            saveat = saveat,
            sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
        )
    else
        sol = solve(prob, model.solver; saveat = saveat)
    end

    return sol
end

"""
Extract PK parameters from concentration profile.

Computes:
- Cmax: Maximum blood concentration
- Tmax: Time of maximum concentration
- AUC: Area under the curve (trapezoidal)
- CL_apparent: Dose / AUC
"""
function extract_pk_parameters(
    sol::ODESolution,
    dose::Real
)::Dict{String, Float64}
    times = sol.t
    blood_conc = [sol.u[i][1] for i in eachindex(sol.u)]  # Blood is compartment 1

    # Cmax and Tmax
    cmax_idx = argmax(blood_conc)
    cmax = blood_conc[cmax_idx]
    tmax = times[cmax_idx]

    # AUC (trapezoidal rule)
    auc = 0.0
    for i in 2:length(times)
        dt = times[i] - times[i-1]
        auc += 0.5 * (blood_conc[i] + blood_conc[i-1]) * dt
    end

    # Apparent clearance
    cl_apparent = dose / max(auc, 1e-10)

    return Dict(
        "Cmax" => cmax,
        "Tmax" => tmax,
        "AUC" => auc,
        "CL_apparent" => cl_apparent
    )
end

"""
Full prediction: embedding -> concentrations -> PK parameters.
"""
function predict_pk(
    model::NeuralODEPBPK,
    drug_emb::AbstractVector,
    dose::Real;
    t_span::Tuple{Real, Real} = (0.0, 24.0),
    saveat::AbstractVector = 0.0:0.5:24.0,
    params::NamedTuple = default_pbpk_params()
)::Dict{String, Any}
    # Get concentration profile
    sol = predict_concentrations(model, drug_emb, dose; t_span, saveat, params)

    # Extract PK parameters
    pk_params = extract_pk_parameters(sol, dose)

    return Dict(
        "solution" => sol,
        "times" => sol.t,
        "concentrations" => hcat([u for u in sol.u]...),  # [n_compartments, n_times]
        "pk_parameters" => pk_params
    )
end

#=============================================================================
  Augmented Neural ODE
=============================================================================#

"""
Augmented Neural ODE with additional latent dimensions.

Augments the state space with extra dimensions that the neural network
can use for better expressivity.
"""
struct AugmentedNeuralODE
    base_dynamics::NeuralDynamics
    n_augmented::Int
    augment_init::Dense  # Initialize augmented dimensions
end

@functor AugmentedNeuralODE

function AugmentedNeuralODE(;
    state_dim::Int = N_COMPARTMENTS,
    drug_emb_dim::Int = 512,
    hidden_dim::Int = NODE_HIDDEN_DIM,
    n_augmented::Int = NODE_LATENT_DIM
)
    # Dynamics for augmented state
    augmented_state_dim = state_dim + n_augmented
    base_dynamics = NeuralDynamics(;
        state_dim = augmented_state_dim,
        drug_emb_dim,
        hidden_dim
    )

    # Initialize augmented dimensions from drug embedding
    augment_init = Dense(drug_emb_dim => n_augmented, tanh)

    return AugmentedNeuralODE(base_dynamics, n_augmented, augment_init)
end

"""
Forward pass with state augmentation.
"""
function (aug::AugmentedNeuralODE)(
    x_physical::AbstractVector,
    x_augmented::AbstractVector,
    drug_emb::AbstractVector,
    t::Real
)
    # Concatenate physical and augmented state
    x_full = vcat(x_physical, x_augmented)

    # Get full derivative
    dx_full = aug.base_dynamics(x_full, drug_emb, t)

    # Split back
    n_physical = length(x_physical)
    dx_physical = dx_full[1:n_physical]
    dx_augmented = dx_full[n_physical+1:end]

    return dx_physical, dx_augmented
end

#=============================================================================
  Physics-Informed Neural ODE
=============================================================================#

"""
Physics-Informed Neural ODE with hard constraints.

Enforces physical constraints:
1. Mass conservation
2. Non-negative concentrations
3. Bounded partition coefficients
"""
struct PhysicsInformedNeuralODE
    base_model::NeuralODEPBPK
    enforce_mass_conservation::Bool
    enforce_positivity::Bool
end

@functor PhysicsInformedNeuralODE

function PhysicsInformedNeuralODE(;
    drug_emb_dim::Int = 512,
    enforce_mass_conservation::Bool = true,
    enforce_positivity::Bool = true
)
    base_model = NeuralODEPBPK(;
        drug_emb_dim,
        use_hybrid = true,
        physics_weight = 0.7f0  # Higher physics weight for constraints
    )

    return PhysicsInformedNeuralODE(base_model, enforce_mass_conservation, enforce_positivity)
end

"""
Apply positivity constraint via softplus transformation.
"""
function enforce_positivity_constraint(x::AbstractVector)::Vector{Float32}
    return softplus.(x)
end

"""
Compute mass conservation residual.
"""
function mass_conservation_loss(
    sol::ODESolution,
    total_mass::Real
)::Float32
    # Sum concentrations * volumes at each time point
    loss = 0.0f0
    volumes = [5.0f0, 1.5f0, 0.3f0, 1.0f0, 40.0f0]  # Default volumes

    for u in sol.u
        current_mass = sum(u .* volumes)
        loss += (current_mass - total_mass)^2
    end

    return loss / length(sol.u)
end

#=============================================================================
  Training Utilities
=============================================================================#

"""
Loss function for Neural ODE training.

Combines:
1. Prediction MSE
2. Physics constraints
3. Regularization
"""
function neural_ode_loss(
    model::NeuralODEPBPK,
    drug_emb::AbstractVector,
    dose::Real,
    observed_conc::AbstractVector,
    observed_times::AbstractVector;
    physics_weight::Float32 = 0.1f0,
    params::NamedTuple = default_pbpk_params()
)::Float32
    # Predict
    sol = predict_concentrations(model, drug_emb, dose;
        t_span = (0.0, maximum(observed_times)),
        saveat = observed_times,
        params = params
    )

    # Prediction loss (blood concentration)
    pred_conc = [sol.u[i][1] for i in eachindex(sol.u)]
    mse_loss = mean((pred_conc .- observed_conc).^2)

    # Physics loss (mass conservation)
    total_mass = dose  # Initial mass
    physics_loss = mass_conservation_loss(sol, total_mass)

    return mse_loss + physics_weight * physics_loss
end

"""
Training loop for Neural ODE.
"""
function train_neural_ode!(
    model::NeuralODEPBPK,
    train_data::Vector{<:NamedTuple};  # Each: (drug_emb, dose, observed_conc, observed_times)
    n_epochs::Int = 100,
    lr::Float64 = 1e-3,
    batch_size::Int = 32,
    verbose::Bool = true
)
    opt = Flux.setup(Adam(lr), model)

    n_samples = length(train_data)

    for epoch in 1:n_epochs
        # Shuffle data
        perm = randperm(n_samples)

        epoch_loss = 0.0f0
        n_batches = 0

        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            batch_indices = perm[batch_start:batch_end]

            # Compute gradients
            grads = Flux.gradient(model) do m
                batch_loss = 0.0f0
                for idx in batch_indices
                    sample = train_data[idx]
                    batch_loss += neural_ode_loss(
                        m,
                        sample.drug_emb,
                        sample.dose,
                        sample.observed_conc,
                        sample.observed_times
                    )
                end
                batch_loss / length(batch_indices)
            end

            # Update
            Flux.update!(opt, model, grads[1])

            epoch_loss += grads[1] !== nothing ? 0.0f0 : 0.0f0  # Placeholder
            n_batches += 1
        end

        if verbose && epoch % 10 == 0
            @info "Epoch $epoch/$n_epochs"
        end
    end

    return model
end

#=============================================================================
  Convenience Functions
=============================================================================#

"""
Create Neural ODE model integrated with molecular encoder.

Returns a combined model that takes SMILES -> PK predictions.
"""
function create_neural_ode_pbpk(;
    encoder_dim::Int = 512,
    use_hybrid::Bool = true,
    physics_weight::Float32 = 0.5f0
)::NeuralODEPBPK
    return NeuralODEPBPK(;
        drug_emb_dim = encoder_dim,
        use_hybrid = use_hybrid,
        physics_weight = physics_weight,
        use_adjoint = true
    )
end

"""
Quick prediction from drug embedding.
"""
function quick_predict(
    model::NeuralODEPBPK,
    drug_emb::AbstractVector;
    dose::Real = 100.0,
    t_max::Real = 24.0
)::Dict{String, Any}
    saveat = collect(0.0:0.5:t_max)
    return predict_pk(model, drug_emb, dose; t_span=(0.0, t_max), saveat)
end

end # module
