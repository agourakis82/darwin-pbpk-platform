"""
    LatticeBoltzmann

Lattice Boltzmann Method (LBM) for blood flow simulation in vessels.

This module implements D2Q9 (2D, 9 velocities) lattice Boltzmann scheme with:
- BGK collision operator
- Non-Newtonian blood viscosity (Carreau-Yasuda model)
- Hematocrit-dependent viscosity
- Wall shear stress extraction
- Multiple vessel geometries (straight, stenosis, bifurcation, curved)

# Example
```julia
using DarwinPBPK

# Create stenosis geometry
geometry = create_stenosis_geometry(nx=200, ny=50, stenosis_severity=0.5)

# Blood properties
fluid = FluidProperties(
    density=1060.0,          # kg/m³
    base_viscosity=0.0035,   # Pa·s
    hematocrit=0.45
)

# Boundary conditions
bc = BoundaryConditions(
    inlet_velocity=0.1,      # m/s
    outlet_pressure=0.0,
    type=:pressure_driven
)

# Initialize and run
sim = create_lbm_simulation(geometry, fluid, bc)
run_lbm_simulation!(sim, 5000)

# Extract results
u, v = calculate_velocity_field(sim)
wss = extract_wall_shear_stress(sim)
```

# References
- Krüger et al. (2017). "The Lattice Boltzmann Method: Principles and Practice"
- He & Luo (1997). "Lattice Boltzmann Model for the Incompressible Navier-Stokes Equation"
- Carreau (1972). "Rheological Equations from Molecular Network Theories"
"""
module LatticeBoltzmann

export LatticeConfig, D2Q9Lattice, D3Q19Lattice
export FluidProperties, BoundaryConditions, SimulationDomain
export LBMSimulation
export create_lbm_simulation, equilibrium_distribution
export collision_step!, streaming_step!, apply_boundary_conditions!
export run_lbm_simulation!
export calculate_velocity_field, calculate_density_field
export extract_wall_shear_stress, calculate_reynolds_number, calculate_womersley_number
export create_straight_tube, create_stenosis_geometry, create_bifurcation_geometry
export create_curved_vessel
export carreau_yasuda_viscosity, hematocrit_viscosity_correction
export validate_poiseuille_flow

using LinearAlgebra
using Statistics

# ============================================================================
# Data Structures
# ============================================================================

"""
    LatticeConfig

Abstract type for lattice configurations (D2Q9, D3Q19, etc.)
"""
abstract type LatticeConfig end

"""
    D2Q9Lattice <: LatticeConfig

2D lattice with 9 velocity directions (center + 4 cardinal + 4 diagonal)

Velocity indexing:
  6   2   5
    \\  |  /
  3 - 0 - 1
    /  |  \\
  7   4   8
"""
struct D2Q9Lattice <: LatticeConfig
    weights::Vector{Float64}        # Lattice weights w_i
    velocities::Matrix{Int}         # Lattice velocities c_i (2 x 9)
    opposite::Vector{Int}           # Opposite direction indices
    cs2::Float64                    # Speed of sound squared (1/3 for D2Q9)

    function D2Q9Lattice()
        # Lattice weights
        weights = [
            4/9,                    # 0: center
            1/9, 1/9, 1/9, 1/9,    # 1-4: cardinal
            1/36, 1/36, 1/36, 1/36 # 5-8: diagonal
        ]

        # Lattice velocities [x, y]
        velocities = [
            0  1  0 -1  0  1 -1 -1  1;  # x-components
            0  0  1  0 -1  1  1 -1 -1   # y-components
        ]

        # Opposite directions
        opposite = [1, 4, 5, 2, 3, 8, 9, 6, 7]  # 1-indexed

        cs2 = 1.0/3.0

        new(weights, velocities, opposite, cs2)
    end
end

"""
    D3Q19Lattice <: LatticeConfig

3D lattice with 19 velocity directions (placeholder for future implementation)
"""
struct D3Q19Lattice <: LatticeConfig
    weights::Vector{Float64}
    velocities::Matrix{Int}
    opposite::Vector{Int}
    cs2::Float64

    function D3Q19Lattice()
        # TODO: Implement D3Q19 lattice
        error("D3Q19 lattice not yet implemented")
    end
end

"""
    FluidProperties

Blood fluid properties including non-Newtonian behavior.

# Fields
- `density::Float64`: Fluid density (kg/m³), default 1060.0 for blood
- `base_viscosity::Float64`: Base dynamic viscosity (Pa·s), default 0.0035
- `hematocrit::Float64`: Hematocrit fraction (0-1), default 0.45
- `carreau_n::Float64`: Carreau-Yasuda power index, default 0.3568
- `carreau_lambda::Float64`: Carreau-Yasuda time constant (s), default 3.313
- `carreau_a::Float64`: Carreau-Yasuda transition parameter, default 2.0
- `inf_viscosity::Float64`: Infinite shear viscosity (Pa·s), default 0.0035
- `zero_viscosity::Float64`: Zero shear viscosity (Pa·s), default 0.056
"""
struct FluidProperties
    density::Float64
    base_viscosity::Float64
    hematocrit::Float64
    carreau_n::Float64
    carreau_lambda::Float64
    carreau_a::Float64
    inf_viscosity::Float64
    zero_viscosity::Float64

    function FluidProperties(;
        density=1060.0,
        base_viscosity=0.0035,
        hematocrit=0.45,
        carreau_n=0.3568,
        carreau_lambda=3.313,
        carreau_a=2.0,
        inf_viscosity=0.0035,
        zero_viscosity=0.056
    )
        new(density, base_viscosity, hematocrit, carreau_n, carreau_lambda,
            carreau_a, inf_viscosity, zero_viscosity)
    end
end

"""
    BoundaryConditions

Boundary conditions for LBM simulation.

# Types
- `:velocity_driven` - Prescribed inlet velocity
- `:pressure_driven` - Prescribed pressure difference
"""
struct BoundaryConditions
    inlet_velocity::Float64      # m/s
    outlet_pressure::Float64     # Pa (gauge)
    type::Symbol                 # :velocity_driven or :pressure_driven

    function BoundaryConditions(;
        inlet_velocity=0.1,
        outlet_pressure=0.0,
        type=:velocity_driven
    )
        @assert type in [:velocity_driven, :pressure_driven] "Invalid BC type"
        new(inlet_velocity, outlet_pressure, type)
    end
end

"""
    SimulationDomain

Computational domain with vessel geometry.

# Fields
- `nx::Int`: Grid points in x-direction
- `ny::Int`: Grid points in y-direction
- `dx::Float64`: Lattice spacing (m)
- `dt::Float64`: Time step (s)
- `geometry::BitMatrix`: Solid/fluid mask (true = solid, false = fluid)
- `wall_nodes::Vector{Tuple{Int,Int}}`: List of wall boundary nodes
"""
mutable struct SimulationDomain
    nx::Int
    ny::Int
    dx::Float64
    dt::Float64
    geometry::BitMatrix
    wall_nodes::Vector{Tuple{Int,Int}}

    function SimulationDomain(nx, ny, dx, dt, geometry)
        wall_nodes = find_wall_nodes(geometry)
        new(nx, ny, dx, dt, geometry, wall_nodes)
    end
end

"""
    LBMSimulation

Main LBM simulation container.

# Fields
- `lattice::LatticeConfig`: Lattice configuration (D2Q9, D3Q19)
- `domain::SimulationDomain`: Computational domain
- `fluid::FluidProperties`: Fluid properties
- `bc::BoundaryConditions`: Boundary conditions
- `f::Array{Float64,3}`: Distribution functions (nx, ny, nq)
- `f_new::Array{Float64,3}`: Temporary storage for streaming
- `rho::Matrix{Float64}`: Density field
- `u::Matrix{Float64}`: x-velocity field
- `v::Matrix{Float64}`: y-velocity field
- `tau::Float64`: Relaxation time
"""
mutable struct LBMSimulation
    lattice::LatticeConfig
    domain::SimulationDomain
    fluid::FluidProperties
    bc::BoundaryConditions
    f::Array{Float64,3}
    f_new::Array{Float64,3}
    rho::Matrix{Float64}
    u::Matrix{Float64}
    v::Matrix{Float64}
    tau::Float64
end

# ============================================================================
# Geometry Creation
# ============================================================================

"""
    create_straight_tube(; nx=200, ny=50, diameter=40)

Create a straight cylindrical tube geometry.
"""
function create_straight_tube(; nx=200, ny=50, diameter=40)
    geometry = falses(nx, ny)
    center_y = ny ÷ 2
    radius = diameter ÷ 2

    for i in 1:nx, j in 1:ny
        if abs(j - center_y) > radius
            geometry[i, j] = true  # Solid
        end
    end

    return geometry
end

"""
    create_stenosis_geometry(; nx=200, ny=50, stenosis_severity=0.5, stenosis_length=40)

Create a vessel with stenosis (narrowing).

# Arguments
- `nx::Int`: Grid points in flow direction
- `ny::Int`: Grid points in cross-flow direction
- `stenosis_severity::Float64`: Severity (0-1), 0 = no stenosis, 1 = complete occlusion
- `stenosis_length::Int`: Length of stenosis region
"""
function create_stenosis_geometry(; nx=200, ny=50, stenosis_severity=0.5, stenosis_length=40)
    geometry = falses(nx, ny)
    center_y = ny ÷ 2
    base_radius = ny ÷ 2 - 2

    stenosis_center = nx ÷ 2
    stenosis_start = stenosis_center - stenosis_length ÷ 2
    stenosis_end = stenosis_center + stenosis_length ÷ 2

    for i in 1:nx
        # Calculate local radius
        if stenosis_start <= i <= stenosis_end
            # Smooth stenosis profile (cosine)
            theta = π * (i - stenosis_start) / stenosis_length
            reduction = stenosis_severity * (1 - cos(theta)) / 2
            local_radius = base_radius * (1 - reduction)
        else
            local_radius = base_radius
        end

        # Mark solid nodes
        for j in 1:ny
            if abs(j - center_y) > local_radius
                geometry[i, j] = true
            end
        end
    end

    return geometry
end

"""
    create_bifurcation_geometry(; nx=200, ny=100, branch_angle=30.0)

Create a vessel bifurcation (Y-shape).
"""
function create_bifurcation_geometry(; nx=200, ny=100, branch_angle=30.0)
    geometry = falses(nx, ny)
    center_y = ny ÷ 2
    radius = ny ÷ 4

    split_x = nx ÷ 2

    for i in 1:nx, j in 1:ny
        if i < split_x
            # Straight section
            if abs(j - center_y) > radius
                geometry[i, j] = true
            end
        else
            # Branching section (simplified)
            offset = (i - split_x) * tand(branch_angle)
            upper_center = center_y + offset
            lower_center = center_y - offset

            if (j - upper_center)^2 + (j - lower_center)^2 > radius^2 * 4
                if !(abs(j - upper_center) <= radius || abs(j - lower_center) <= radius)
                    geometry[i, j] = true
                end
            end
        end
    end

    return geometry
end

"""
    create_curved_vessel(; nx=200, ny=50, curvature=0.01)

Create a curved vessel with specified curvature.
"""
function create_curved_vessel(; nx=200, ny=50, curvature=0.01)
    geometry = falses(nx, ny)
    radius = ny ÷ 4

    for i in 1:nx, j in 1:ny
        # Curved centerline
        center_y = ny ÷ 2 + curvature * (i - nx/2)^2

        if abs(j - center_y) > radius
            geometry[i, j] = true
        end
    end

    return geometry
end

"""
    find_wall_nodes(geometry::BitMatrix)

Identify wall boundary nodes (fluid nodes adjacent to solid).
"""
function find_wall_nodes(geometry::BitMatrix)
    nx, ny = size(geometry)
    wall_nodes = Tuple{Int,Int}[]

    for i in 2:nx-1, j in 2:ny-1
        if !geometry[i, j]  # Fluid node
            # Check neighbors
            if geometry[i+1, j] || geometry[i-1, j] ||
               geometry[i, j+1] || geometry[i, j-1]
                push!(wall_nodes, (i, j))
            end
        end
    end

    return wall_nodes
end

# ============================================================================
# Blood Rheology
# ============================================================================

"""
    carreau_yasuda_viscosity(shear_rate, fluid::FluidProperties)

Calculate blood viscosity using Carreau-Yasuda model.

η(γ̇) = η∞ + (η₀ - η∞) [1 + (λγ̇)ᵃ]^((n-1)/a)

# Arguments
- `shear_rate::Float64`: Shear rate γ̇ (1/s)
- `fluid::FluidProperties`: Fluid properties

# Returns
- `Float64`: Dynamic viscosity (Pa·s)
"""
function carreau_yasuda_viscosity(shear_rate::Float64, fluid::FluidProperties)
    η∞ = fluid.inf_viscosity
    η₀ = fluid.zero_viscosity
    λ = fluid.carreau_lambda
    n = fluid.carreau_n
    a = fluid.carreau_a

    η = η∞ + (η₀ - η∞) * (1 + (λ * abs(shear_rate))^a)^((n - 1) / a)

    return η
end

"""
    hematocrit_viscosity_correction(base_viscosity, hematocrit)

Apply hematocrit-dependent viscosity correction.

Empirical relation: η(H) = η₀ (1 + 2.5H + 7.35H²)

# Arguments
- `base_viscosity::Float64`: Base viscosity (Pa·s)
- `hematocrit::Float64`: Hematocrit fraction (0-1)

# Returns
- `Float64`: Corrected viscosity (Pa·s)
"""
function hematocrit_viscosity_correction(base_viscosity::Float64, hematocrit::Float64)
    # Empirical correlation (Pries et al., 1992)
    correction = 1.0 + 2.5 * hematocrit + 7.35 * hematocrit^2
    return base_viscosity * correction
end

# ============================================================================
# LBM Core Functions
# ============================================================================

"""
    equilibrium_distribution(rho, ux, uy, lattice::D2Q9Lattice)

Calculate equilibrium distribution function f_i^eq.

f_i^eq = w_i ρ [1 + (c_i·u)/cs² + (c_i·u)²/(2cs⁴) - u²/(2cs²)]

# Arguments
- `rho::Float64`: Density
- `ux::Float64`: x-velocity
- `uy::Float64`: y-velocity
- `lattice::D2Q9Lattice`: Lattice configuration

# Returns
- `Vector{Float64}`: Equilibrium distributions (length 9)
"""
function equilibrium_distribution(rho::Float64, ux::Float64, uy::Float64,
                                 lattice::D2Q9Lattice)
    f_eq = zeros(9)
    cs2 = lattice.cs2
    usqr = ux^2 + uy^2

    for i in 1:9
        cx = lattice.velocities[1, i]
        cy = lattice.velocities[2, i]

        cu = cx * ux + cy * uy

        f_eq[i] = lattice.weights[i] * rho * (
            1.0 + cu / cs2 +
            cu^2 / (2 * cs2^2) -
            usqr / (2 * cs2)
        )
    end

    return f_eq
end

"""
    collision_step!(sim::LBMSimulation)

Perform BGK collision step: f_i(x,t) = f_i(x,t) - (f_i - f_i^eq)/τ
"""
function collision_step!(sim::LBMSimulation)
    nx, ny = sim.domain.nx, sim.domain.ny
    omega = 1.0 / sim.tau  # Collision frequency

    for i in 1:nx, j in 1:ny
        if !sim.domain.geometry[i, j]  # Fluid node
            rho = sim.rho[i, j]
            ux = sim.u[i, j]
            uy = sim.v[i, j]

            f_eq = equilibrium_distribution(rho, ux, uy, sim.lattice)

            for k in 1:9
                sim.f[i, j, k] += -omega * (sim.f[i, j, k] - f_eq[k])
            end
        end
    end
end

"""
    streaming_step!(sim::LBMSimulation)

Perform streaming step: f_i(x + c_i, t+1) = f_i(x, t)
"""
function streaming_step!(sim::LBMSimulation)
    nx, ny = sim.domain.nx, sim.domain.ny

    # Copy current distribution
    sim.f_new .= sim.f

    for i in 1:nx, j in 1:ny
        if !sim.domain.geometry[i, j]
            for k in 1:9
                cx = sim.lattice.velocities[1, k]
                cy = sim.lattice.velocities[2, k]

                # Target location
                i_new = i + cx
                j_new = j + cy

                # Periodic or bounce-back boundaries
                if 1 <= i_new <= nx && 1 <= j_new <= ny
                    if !sim.domain.geometry[i_new, j_new]
                        sim.f_new[i_new, j_new, k] = sim.f[i, j, k]
                    else
                        # Bounce-back at walls
                        opp = sim.lattice.opposite[k]
                        sim.f_new[i, j, opp] = sim.f[i, j, k]
                    end
                end
            end
        end
    end

    # Swap arrays
    sim.f, sim.f_new = sim.f_new, sim.f
end

"""
    apply_boundary_conditions!(sim::LBMSimulation)

Apply inlet/outlet boundary conditions.
"""
function apply_boundary_conditions!(sim::LBMSimulation)
    ny = sim.domain.ny

    if sim.bc.type == :velocity_driven
        # Inlet (x=1): Prescribed velocity
        u_in = sim.bc.inlet_velocity

        for j in 1:ny
            if !sim.domain.geometry[1, j]
                rho = 1.0  # Assume unit density at inlet
                f_eq = equilibrium_distribution(rho, u_in, 0.0, sim.lattice)
                sim.f[1, j, :] .= f_eq
            end
        end

        # Outlet (x=nx): Zero gradient
        for j in 1:ny
            if !sim.domain.geometry[end, j]
                sim.f[end, j, :] .= sim.f[end-1, j, :]
            end
        end
    end
end

"""
    calculate_macroscopic!(sim::LBMSimulation)

Calculate macroscopic quantities (density, velocity) from distribution functions.

ρ = Σᵢ fᵢ
ρu = Σᵢ fᵢ cᵢ
"""
function calculate_macroscopic!(sim::LBMSimulation)
    nx, ny = sim.domain.nx, sim.domain.ny

    for i in 1:nx, j in 1:ny
        if !sim.domain.geometry[i, j]
            # Density
            sim.rho[i, j] = sum(sim.f[i, j, :])

            # Velocity
            ux = 0.0
            uy = 0.0
            for k in 1:9
                ux += sim.f[i, j, k] * sim.lattice.velocities[1, k]
                uy += sim.f[i, j, k] * sim.lattice.velocities[2, k]
            end

            sim.u[i, j] = ux / sim.rho[i, j]
            sim.v[i, j] = uy / sim.rho[i, j]
        end
    end
end

# ============================================================================
# Simulation Setup and Execution
# ============================================================================

"""
    create_lbm_simulation(geometry, fluid, bc; dx=1e-5, dt=1e-6)

Initialize LBM simulation.

# Arguments
- `geometry::BitMatrix`: Vessel geometry
- `fluid::FluidProperties`: Fluid properties
- `bc::BoundaryConditions`: Boundary conditions
- `dx::Float64`: Lattice spacing (m)
- `dt::Float64`: Time step (s)

# Returns
- `LBMSimulation`: Initialized simulation
"""
function create_lbm_simulation(
    geometry::BitMatrix,
    fluid::FluidProperties,
    bc::BoundaryConditions;
    dx=1e-5,
    dt=1e-6
)
    nx, ny = size(geometry)
    lattice = D2Q9Lattice()

    # Create domain
    domain = SimulationDomain(nx, ny, dx, dt, geometry)

    # Calculate relaxation time from viscosity
    # ν = cs² (τ - 0.5) Δt
    nu = fluid.base_viscosity / fluid.density  # Kinematic viscosity
    tau = nu / (lattice.cs2 * dt) + 0.5

    # Initialize distribution functions
    f = zeros(nx, ny, 9)
    f_new = zeros(nx, ny, 9)
    rho = ones(nx, ny)
    u = zeros(nx, ny)
    v = zeros(nx, ny)

    # Initialize with equilibrium
    for i in 1:nx, j in 1:ny
        if !geometry[i, j]
            f_eq = equilibrium_distribution(1.0, 0.0, 0.0, lattice)
            f[i, j, :] .= f_eq
        end
    end

    return LBMSimulation(lattice, domain, fluid, bc, f, f_new, rho, u, v, tau)
end

"""
    run_lbm_simulation!(sim::LBMSimulation, n_steps::Int; print_interval=100)

Run LBM simulation for specified number of time steps.

# Arguments
- `sim::LBMSimulation`: Simulation object
- `n_steps::Int`: Number of time steps
- `print_interval::Int`: Print progress every N steps
"""
function run_lbm_simulation!(sim::LBMSimulation, n_steps::Int; print_interval=100)
    for step in 1:n_steps
        # LBM cycle
        collision_step!(sim)
        streaming_step!(sim)
        apply_boundary_conditions!(sim)
        calculate_macroscopic!(sim)

        # Progress
        if step % print_interval == 0
            u_max = maximum(abs.(sim.u))
            println("Step $step/$n_steps, max velocity: $(u_max) m/s")
        end
    end
end

# ============================================================================
# Post-Processing
# ============================================================================

"""
    calculate_velocity_field(sim::LBMSimulation)

Extract velocity field from simulation.

# Returns
- `u::Matrix{Float64}`: x-velocity (m/s)
- `v::Matrix{Float64}`: y-velocity (m/s)
"""
function calculate_velocity_field(sim::LBMSimulation)
    # Convert from lattice units to physical units
    u_phys = sim.u .* (sim.domain.dx / sim.domain.dt)
    v_phys = sim.v .* (sim.domain.dx / sim.domain.dt)

    return u_phys, v_phys
end

"""
    calculate_density_field(sim::LBMSimulation)

Extract density field from simulation.
"""
function calculate_density_field(sim::LBMSimulation)
    return sim.rho .* sim.fluid.density
end

"""
    extract_wall_shear_stress(sim::LBMSimulation)

Calculate wall shear stress (WSS) at vessel walls.

τ_wall = μ * ∂u/∂n

# Returns
- `wss::Vector{Float64}`: Wall shear stress at wall nodes (Pa)
- `locations::Vector{Tuple{Int,Int}}`: Wall node coordinates
"""
function extract_wall_shear_stress(sim::LBMSimulation)
    wss = Float64[]
    locations = Tuple{Int,Int}[]

    for (i, j) in sim.domain.wall_nodes
        # Estimate velocity gradient normal to wall
        # Simple finite difference approximation
        du_dn = 0.0
        n_neighbors = 0

        for (di, dj) in [(-1,0), (1,0), (0,-1), (0,1)]
            i_n = i + di
            j_n = j + dj

            if 1 <= i_n <= sim.domain.nx && 1 <= j_n <= sim.domain.ny
                if sim.domain.geometry[i_n, j_n]  # Solid neighbor
                    # Velocity difference
                    u_mag = sqrt(sim.u[i, j]^2 + sim.v[i, j]^2)
                    du_dn += u_mag / sim.domain.dx
                    n_neighbors += 1
                end
            end
        end

        if n_neighbors > 0
            du_dn /= n_neighbors
            tau = sim.fluid.base_viscosity * du_dn
            push!(wss, tau)
            push!(locations, (i, j))
        end
    end

    return wss, locations
end

"""
    calculate_reynolds_number(sim::LBMSimulation, characteristic_length)

Calculate Reynolds number: Re = ρUL/μ
"""
function calculate_reynolds_number(sim::LBMSimulation, characteristic_length::Float64)
    u_mean = mean(abs.(sim.u[sim.u .!= 0]))
    Re = sim.fluid.density * u_mean * characteristic_length / sim.fluid.base_viscosity
    return Re
end

"""
    calculate_womersley_number(sim::LBMSimulation, radius, frequency)

Calculate Womersley number: α = R√(ωρ/μ)

# Arguments
- `sim::LBMSimulation`: Simulation object
- `radius::Float64`: Vessel radius (m)
- `frequency::Float64`: Pulsatile flow frequency (Hz)
"""
function calculate_womersley_number(sim::LBMSimulation, radius::Float64, frequency::Float64)
    omega = 2π * frequency
    alpha = radius * sqrt(omega * sim.fluid.density / sim.fluid.base_viscosity)
    return alpha
end

# ============================================================================
# Validation
# ============================================================================

"""
    validate_poiseuille_flow(; nx=100, ny=50, n_steps=5000)

Validate LBM against analytical Poiseuille flow solution.

Returns maximum relative error in velocity profile.
"""
function validate_poiseuille_flow(; nx=100, ny=50, n_steps=5000)
    # Create straight tube
    geometry = create_straight_tube(nx=nx, ny=ny, diameter=40)

    # Fluid properties
    fluid = FluidProperties(density=1000.0, base_viscosity=0.001)

    # Boundary conditions
    bc = BoundaryConditions(inlet_velocity=0.01, type=:velocity_driven)

    # Run simulation
    sim = create_lbm_simulation(geometry, fluid, bc)
    run_lbm_simulation!(sim, n_steps, print_interval=1000)

    # Extract centerline velocity profile
    center_x = nx ÷ 2
    u_lbm = sim.u[center_x, :]

    # Analytical solution: Poiseuille flow
    # u(y) = u_max [1 - (2y/D)²]
    radius = ny ÷ 2 - 2
    center_y = ny ÷ 2
    u_max = bc.inlet_velocity * 2  # Approximate

    u_analytical = zeros(ny)
    for j in 1:ny
        r = abs(j - center_y) / radius
        if r < 1.0
            u_analytical[j] = u_max * (1 - r^2)
        end
    end

    # Calculate error
    valid_indices = findall(u_analytical .> 0)
    rel_error = abs.(u_lbm[valid_indices] .- u_analytical[valid_indices]) ./ u_analytical[valid_indices]
    max_error = maximum(rel_error)

    println("Poiseuille validation: max relative error = $(max_error)")

    return max_error
end

end # module LatticeBoltzmann
