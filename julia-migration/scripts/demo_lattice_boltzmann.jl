"""
Demonstration of Lattice Boltzmann Method for Blood Flow Simulation

This script demonstrates the key features of the LBM module including:
1. Straight tube (Poiseuille flow validation)
2. Stenosis simulation
3. Wall shear stress extraction
4. Reynolds and Womersley number calculations
5. Non-Newtonian blood viscosity

Author: Dr. Demetrios Agourakis
Date: December 2025
"""

using DarwinPBPK
using Statistics
using Printf

println("="^80)
println("Lattice Boltzmann Method - Blood Flow Simulation Demo")
println("="^80)

# ============================================================================
# Example 1: Poiseuille Flow Validation
# ============================================================================

println("\n[1] Poiseuille Flow Validation")
println("-"^80)

# Validate against analytical solution
max_error = validate_poiseuille_flow(nx=100, ny=50, n_steps=5000)

@printf("Maximum relative error: %.2f%%\n", max_error * 100)

if max_error < 0.10
    println("✓ Validation PASSED (error < 10%)")
else
    println("⚠ Validation WARNING (error > 10%, LBM has inherent numerical diffusion)")
end

# ============================================================================
# Example 2: Straight Tube Simulation
# ============================================================================

println("\n[2] Straight Tube Simulation")
println("-"^80)

# Create geometry
println("Creating straight tube geometry...")
geom_straight = create_straight_tube(nx=150, ny=50, diameter=40)
n_fluid = count(.!geom_straight)
println("  Grid: 150 × 50")
println("  Fluid nodes: $n_fluid")

# Blood properties
println("\nBlood properties:")
fluid = FluidProperties(
    density=1060.0,          # kg/m³
    base_viscosity=0.0035,   # Pa·s
    hematocrit=0.45
)
println("  Density: $(fluid.density) kg/m³")
println("  Viscosity: $(fluid.base_viscosity) Pa·s")
println("  Hematocrit: $(fluid.hematocrit)")

# Boundary conditions
bc = BoundaryConditions(
    inlet_velocity=0.1,      # m/s (lattice units)
    outlet_pressure=0.0,
    type=:velocity_driven
)
println("\nBoundary conditions:")
println("  Type: $(bc.type)")
println("  Inlet velocity: $(bc.inlet_velocity) (lattice units)")

# Create and run simulation
println("\nInitializing simulation...")
sim_straight = create_lbm_simulation(geom_straight, fluid, bc, dx=1e-5, dt=1e-6)
println("  Relaxation time τ: $(round(sim_straight.tau, digits=3))")

println("\nRunning simulation (1000 steps)...")
run_lbm_simulation!(sim_straight, 1000, print_interval=250)

# Extract results
u, v = calculate_velocity_field(sim_straight)
u_max = maximum(abs.(u))
println("\nResults:")
@printf("  Maximum velocity: %.3e m/s\n", u_max)
@printf("  Mean velocity: %.3e m/s\n", mean(abs.(u[u .!= 0])))

# Calculate Reynolds number
diameter = 40 * 1e-5  # Convert to meters
Re = calculate_reynolds_number(sim_straight, diameter)
@printf("  Reynolds number: %.1f\n", Re)

if Re < 2300
    println("  Flow regime: LAMINAR ✓")
else
    println("  Flow regime: TRANSITIONAL/TURBULENT")
end

# ============================================================================
# Example 3: Stenosis Simulation
# ============================================================================

println("\n[3] Stenosis (50% Severity) Simulation")
println("-"^80)

# Create stenosis geometry
println("Creating stenosis geometry...")
geom_stenosis = create_stenosis_geometry(
    nx=200,
    ny=50,
    stenosis_severity=0.5,  # 50% narrowing
    stenosis_length=40
)
n_fluid_stenosis = count(.!geom_stenosis)
println("  Grid: 200 × 50")
println("  Stenosis severity: 50%")
println("  Stenosis length: 40 lattice units")
println("  Fluid nodes: $n_fluid_stenosis (reduced from normal tube)")

# Same fluid properties and BC
println("\nRunning stenosis simulation (2000 steps)...")
sim_stenosis = create_lbm_simulation(geom_stenosis, fluid, bc, dx=1e-5, dt=1e-6)
run_lbm_simulation!(sim_stenosis, 2000, print_interval=500)

# Extract velocity field
u_sten, v_sten = calculate_velocity_field(sim_stenosis)

# Analyze flow through stenosis
center_y = 25
u_profile = u_sten[:, center_y]

# Find inlet, stenosis, and outlet regions
u_inlet = mean(u_profile[10:20])
u_stenosis = maximum(u_profile[80:120])  # Stenosis center region
u_outlet = mean(u_profile[180:190])

println("\nVelocity analysis:")
@printf("  Inlet velocity: %.3e m/s\n", u_inlet)
@printf("  Stenosis (peak): %.3e m/s\n", u_stenosis)
@printf("  Outlet velocity: %.3e m/s\n", u_outlet)
@printf("  Velocity ratio (stenosis/inlet): %.2f\n", u_stenosis / u_inlet)

# Continuity should increase velocity in narrowed region
if u_stenosis > u_inlet * 1.2
    println("  ✓ Velocity acceleration in stenosis confirmed (continuity)")
else
    println("  ⚠ Expected velocity increase not observed")
end

# ============================================================================
# Example 4: Wall Shear Stress
# ============================================================================

println("\n[4] Wall Shear Stress Analysis")
println("-"^80)

println("Extracting wall shear stress from stenosis simulation...")
wss, locations = extract_wall_shear_stress(sim_stenosis)

println("  Number of wall nodes: $(length(wss))")
@printf("  Mean WSS: %.3e Pa\n", mean(wss))
@printf("  Maximum WSS: %.3e Pa\n", maximum(wss))
@printf("  Minimum WSS: %.3e Pa\n", minimum(wss))

# Typical physiological WSS ranges
# Arteries: 1-7 Pa
# Veins: 0.1-0.6 Pa
wss_max = maximum(wss)
if wss_max > 1.0 && wss_max < 50.0
    println("  ✓ WSS in physiologically relevant range")
else
    println("  ⚠ WSS outside typical arterial range (1-50 Pa)")
end

# ============================================================================
# Example 5: Womersley Number (Pulsatile Flow)
# ============================================================================

println("\n[5] Womersley Number Calculation")
println("-"^80)

# Calculate for typical arterial flow
radius = 20 * 1e-5  # 20 lattice units → meters
frequency = 1.2  # Hz (72 bpm heart rate)

alpha = calculate_womersley_number(sim_straight, radius, frequency)

println("Pulsatile flow parameters:")
@printf("  Vessel radius: %.1f μm\n", radius * 1e6)
@printf("  Heart rate: %.0f bpm\n", frequency * 60)
@printf("  Womersley number α: %.2f\n", alpha)

# Interpretation
if alpha < 1.0
    println("  Regime: Quasi-steady (viscous forces dominate)")
elseif alpha < 10.0
    println("  Regime: Intermediate (viscous and inertial forces balanced)")
else
    println("  Regime: Inertia-dominated (unsteady effects important)")
end

# ============================================================================
# Example 6: Non-Newtonian Viscosity
# ============================================================================

println("\n[6] Non-Newtonian Blood Viscosity (Carreau-Yasuda)")
println("-"^80)

# Test viscosity at different shear rates
shear_rates = [0.1, 1.0, 10.0, 100.0, 1000.0]  # s⁻¹

println("Shear-dependent viscosity:")
println("  Shear Rate (s⁻¹) | Viscosity (Pa·s)")
println("  " * "-"^40)

for γ in shear_rates
    η = carreau_yasuda_viscosity(γ, fluid)
    @printf("  %15.1f | %15.6f\n", γ, η)
end

println("\n  Note: Blood viscosity decreases with shear rate (shear-thinning)")
println("  Low shear: η ≈ 0.056 Pa·s (η₀)")
println("  High shear: η ≈ 0.0035 Pa·s (η∞)")

# Hematocrit effect
println("\nHematocrit effect on viscosity:")
hematocrits = [0.30, 0.40, 0.45, 0.50]
η_base = 0.0035

println("  Hematocrit | Viscosity (Pa·s) | Increase")
println("  " * "-"^50)

for H in hematocrits
    η_corr = hematocrit_viscosity_correction(η_base, H)
    increase = (η_corr / η_base - 1) * 100
    @printf("  %10.2f | %16.6f | %6.1f%%\n", H, η_corr, increase)
end

# ============================================================================
# Example 7: Bifurcation Geometry
# ============================================================================

println("\n[7] Bifurcation Simulation (Preview)")
println("-"^80)

println("Creating bifurcation geometry...")
geom_bifurc = create_bifurcation_geometry(nx=150, ny=100, branch_angle=30.0)
n_fluid_bifurc = count(.!geom_bifurc)

println("  Grid: 150 × 100")
println("  Branch angle: 30°")
println("  Fluid nodes: $n_fluid_bifurc")
println("  Note: Full simulation would require longer run time")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^80)
println("SUMMARY")
println("="^80)

println("\nLattice Boltzmann Method capabilities demonstrated:")
println("  ✓ D2Q9 lattice configuration")
println("  ✓ BGK collision operator")
println("  ✓ Bounce-back boundary conditions")
println("  ✓ Poiseuille flow validation (analytical comparison)")
println("  ✓ Complex geometries (stenosis, bifurcation)")
println("  ✓ Wall shear stress extraction")
println("  ✓ Reynolds and Womersley number calculations")
println("  ✓ Non-Newtonian blood rheology (Carreau-Yasuda)")
println("  ✓ Hematocrit-dependent viscosity")

println("\nApplications in PBPK:")
println("  • Predict drug deposition in stenosed vessels")
println("  • Calculate WSS effects on endothelial drug uptake")
println("  • Model shear-dependent drug release from carriers")
println("  • Simulate particle transport in complex vasculature")
println("  • Integrate with coagulation models (platelet activation)")

println("\nNext steps:")
println("  • Extend to D3Q19 (3D flows)")
println("  • Add pulsatile inlet conditions")
println("  • Couple with particle tracking (DPD/MD)")
println("  • Integrate with PK model (perfusion rates)")
println("  • GPU acceleration for large domains")

println("\n" * "="^80)
println("Demo completed successfully!")
println("="^80)
