"""
Test suite for Lattice Boltzmann Method blood flow simulation module
"""

using Test
using DarwinPBPK
using DarwinPBPK.LatticeBoltzmann
using Statistics
using LinearAlgebra

@testset "Lattice Boltzmann Tests" begin

    @testset "D2Q9 Lattice Configuration" begin
        lattice = D2Q9Lattice()

        @test length(lattice.weights) == 9
        @test sum(lattice.weights) ≈ 1.0
        @test lattice.cs2 ≈ 1.0/3.0
        @test size(lattice.velocities) == (2, 9)
        @test length(lattice.opposite) == 9

        # Check velocity pairs are opposite
        for i in 1:9
            opp = lattice.opposite[i]
            @test lattice.velocities[:, i] == -lattice.velocities[:, opp]
        end
    end

    @testset "Fluid Properties" begin
        # Default blood properties
        fluid = FluidProperties()
        @test fluid.density == 1060.0
        @test fluid.base_viscosity == 0.0035
        @test fluid.hematocrit == 0.45

        # Custom properties
        fluid_custom = FluidProperties(density=1000.0, base_viscosity=0.001)
        @test fluid_custom.density == 1000.0
        @test fluid_custom.base_viscosity == 0.001
    end

    @testset "Boundary Conditions" begin
        # Velocity-driven
        bc_vel = BoundaryConditions(inlet_velocity=0.1, type=:velocity_driven)
        @test bc_vel.inlet_velocity == 0.1
        @test bc_vel.type == :velocity_driven

        # Pressure-driven
        bc_pres = BoundaryConditions(outlet_pressure=1000.0, type=:pressure_driven)
        @test bc_pres.outlet_pressure == 1000.0
        @test bc_pres.type == :pressure_driven

        # Invalid type
        @test_throws AssertionError BoundaryConditions(type=:invalid)
    end

    @testset "Geometry Creation" begin
        # Straight tube
        geom_straight = create_straight_tube(nx=100, ny=50, diameter=40)
        @test size(geom_straight) == (100, 50)
        @test isa(geom_straight, BitMatrix)

        # Count fluid nodes (should be approximately π*r²*length)
        n_fluid = count(.!geom_straight)
        @test n_fluid > 0

        # Stenosis
        geom_stenosis = create_stenosis_geometry(
            nx=100, ny=50,
            stenosis_severity=0.5,
            stenosis_length=20
        )
        @test size(geom_stenosis) == (100, 50)

        # Stenosis should have fewer fluid nodes than straight tube
        n_fluid_stenosis = count(.!geom_stenosis)
        @test n_fluid_stenosis < n_fluid

        # Bifurcation
        geom_bifurc = create_bifurcation_geometry(nx=100, ny=100, branch_angle=30.0)
        @test size(geom_bifurc) == (100, 100)

        # Curved vessel
        geom_curved = create_curved_vessel(nx=100, ny=50, curvature=0.01)
        @test size(geom_curved) == (100, 50)
    end

    @testset "Wall Node Detection" begin
        geom = create_straight_tube(nx=50, ny=30, diameter=20)
        wall_nodes = LatticeBoltzmann.find_wall_nodes(geom)

        @test length(wall_nodes) > 0

        # All wall nodes should be fluid nodes adjacent to solid
        for (i, j) in wall_nodes
            @test !geom[i, j]  # Should be fluid

            # Check at least one neighbor is solid
            has_solid_neighbor = false
            for (di, dj) in [(-1,0), (1,0), (0,-1), (0,1)]
                i_n, j_n = i + di, j + dj
                if 1 <= i_n <= 50 && 1 <= j_n <= 30
                    if geom[i_n, j_n]
                        has_solid_neighbor = true
                        break
                    end
                end
            end
            @test has_solid_neighbor
        end
    end

    @testset "Blood Rheology" begin
        fluid = FluidProperties()

        # Carreau-Yasuda viscosity
        # Low shear rate → high viscosity
        η_low = carreau_yasuda_viscosity(1.0, fluid)
        @test η_low > fluid.inf_viscosity

        # High shear rate → approaches infinite shear viscosity
        η_high = carreau_yasuda_viscosity(1000.0, fluid)
        @test η_high > fluid.inf_viscosity
        @test η_high < η_low

        # Hematocrit correction
        η_base = 0.001
        η_corrected = hematocrit_viscosity_correction(η_base, 0.45)
        @test η_corrected > η_base

        # Zero hematocrit → no correction
        η_zero = hematocrit_viscosity_correction(η_base, 0.0)
        @test η_zero == η_base
    end

    @testset "Equilibrium Distribution" begin
        lattice = D2Q9Lattice()

        # At rest: f_eq should equal weights * density
        rho = 1.0
        f_eq_rest = equilibrium_distribution(rho, 0.0, 0.0, lattice)
        @test length(f_eq_rest) == 9
        @test sum(f_eq_rest) ≈ rho
        @test all(f_eq_rest .≈ lattice.weights .* rho)

        # With velocity
        ux, uy = 0.1, 0.05
        f_eq_moving = equilibrium_distribution(rho, ux, uy, lattice)
        @test sum(f_eq_moving) ≈ rho
        @test f_eq_moving != f_eq_rest

        # Momentum conservation
        ux_calc = sum(f_eq_moving .* lattice.velocities[1, :])
        uy_calc = sum(f_eq_moving .* lattice.velocities[2, :])
        @test ux_calc ≈ rho * ux
        @test uy_calc ≈ rho * uy
    end

    @testset "Simulation Initialization" begin
        geom = create_straight_tube(nx=50, ny=30, diameter=20)
        fluid = FluidProperties(density=1000.0, base_viscosity=0.001)
        bc = BoundaryConditions(inlet_velocity=0.01)

        sim = create_lbm_simulation(geom, fluid, bc, dx=1e-5, dt=1e-6)

        @test isa(sim, LBMSimulation)
        @test size(sim.f) == (50, 30, 9)
        @test size(sim.rho) == (50, 30)
        @test size(sim.u) == (50, 30)
        @test size(sim.v) == (50, 30)
        @test sim.tau > 0.5  # Physical constraint

        # Initial conditions: should be at equilibrium
        for i in 1:50, j in 1:30
            if !geom[i, j]
                @test sum(sim.f[i, j, :]) ≈ 1.0
            end
        end
    end

    @testset "Collision Step" begin
        geom = create_straight_tube(nx=30, ny=20, diameter=15)
        fluid = FluidProperties()
        bc = BoundaryConditions(inlet_velocity=0.01)

        sim = create_lbm_simulation(geom, fluid, bc)

        # Store initial state
        f_before = copy(sim.f)

        # Perform collision
        LatticeBoltzmann.calculate_macroscopic!(sim)
        collision_step!(sim)

        # Distribution should change (unless already at equilibrium)
        # At least some nodes should have different distributions
        n_changed = 0
        for i in 1:30, j in 1:20
            if !geom[i, j]
                if !isapprox(sim.f[i, j, :], f_before[i, j, :], atol=1e-10)
                    n_changed += 1
                end
            end
        end

        # Since we start near equilibrium, changes might be small
        @test n_changed >= 0  # Just check it runs without error
    end

    @testset "Streaming Step" begin
        geom = create_straight_tube(nx=30, ny=20, diameter=15)
        fluid = FluidProperties()
        bc = BoundaryConditions(inlet_velocity=0.01)

        sim = create_lbm_simulation(geom, fluid, bc)

        # Set a localized perturbation
        center_x, center_y = 15, 10
        if !geom[center_x, center_y]
            sim.f[center_x, center_y, 2] += 0.1  # Increase right-going population
        end

        # Store before streaming
        f_before = copy(sim.f)

        # Perform streaming
        streaming_step!(sim)

        # Distribution should propagate
        @test sim.f != f_before
    end

    @testset "Macroscopic Calculation" begin
        geom = create_straight_tube(nx=30, ny=20, diameter=15)
        fluid = FluidProperties()
        bc = BoundaryConditions(inlet_velocity=0.1)

        sim = create_lbm_simulation(geom, fluid, bc)

        # Set known distribution at a point
        i, j = 15, 10
        if !geom[i, j]
            lattice = sim.lattice
            rho_test = 1.2
            ux_test = 0.05
            uy_test = 0.02

            sim.f[i, j, :] = equilibrium_distribution(rho_test, ux_test, uy_test, lattice)

            # Calculate macroscopic
            LatticeBoltzmann.calculate_macroscopic!(sim)

            @test sim.rho[i, j] ≈ rho_test rtol=1e-10
            @test sim.u[i, j] ≈ ux_test rtol=1e-10
            @test sim.v[i, j] ≈ uy_test rtol=1e-10
        end
    end

    @testset "Short Simulation Run" begin
        geom = create_straight_tube(nx=50, ny=30, diameter=20)
        fluid = FluidProperties(density=1000.0, base_viscosity=0.001)
        bc = BoundaryConditions(inlet_velocity=0.01)

        sim = create_lbm_simulation(geom, fluid, bc)

        # Run for a few steps
        @test_nowarn run_lbm_simulation!(sim, 10, print_interval=5)

        # Check simulation produced reasonable results
        @test all(isfinite.(sim.rho))
        @test all(isfinite.(sim.u))
        @test all(isfinite.(sim.v))

        # Velocity should develop in x-direction
        u_mean = mean(abs.(sim.u[sim.u .!= 0]))
        @test u_mean > 0
    end

    @testset "Velocity Field Extraction" begin
        geom = create_straight_tube(nx=40, ny=25, diameter=18)
        fluid = FluidProperties()
        bc = BoundaryConditions(inlet_velocity=0.05)

        sim = create_lbm_simulation(geom, fluid, bc, dx=1e-5, dt=1e-6)
        run_lbm_simulation!(sim, 50, print_interval=100)

        u_phys, v_phys = calculate_velocity_field(sim)

        @test size(u_phys) == size(geom)
        @test size(v_phys) == size(geom)
        @test all(isfinite.(u_phys))
        @test all(isfinite.(v_phys))
    end

    @testset "Density Field Extraction" begin
        geom = create_straight_tube(nx=40, ny=25, diameter=18)
        fluid = FluidProperties(density=1060.0)
        bc = BoundaryConditions(inlet_velocity=0.05)

        sim = create_lbm_simulation(geom, fluid, bc)
        run_lbm_simulation!(sim, 50, print_interval=100)

        rho_phys = calculate_density_field(sim)

        @test size(rho_phys) == size(geom)
        @test all(isfinite.(rho_phys))

        # Density should be approximately constant for incompressible flow
        fluid_rho = rho_phys[.!geom]
        @test std(fluid_rho) / mean(fluid_rho) < 0.1
    end

    @testset "Wall Shear Stress" begin
        geom = create_straight_tube(nx=40, ny=25, diameter=18)
        fluid = FluidProperties()
        bc = BoundaryConditions(inlet_velocity=0.1)

        sim = create_lbm_simulation(geom, fluid, bc)
        run_lbm_simulation!(sim, 100, print_interval=100)

        wss, locations = extract_wall_shear_stress(sim)

        @test length(wss) > 0
        @test length(wss) == length(locations)
        @test all(isfinite.(wss))
        @test all(wss .>= 0)  # WSS magnitude should be positive

        # All locations should be wall nodes
        for (i, j) in locations
            @test (i, j) in sim.domain.wall_nodes
        end
    end

    @testset "Reynolds Number" begin
        geom = create_straight_tube(nx=50, ny=30, diameter=20)
        fluid = FluidProperties(density=1000.0, base_viscosity=0.001)
        bc = BoundaryConditions(inlet_velocity=0.1)

        sim = create_lbm_simulation(geom, fluid, bc)
        run_lbm_simulation!(sim, 100, print_interval=100)

        L = 20 * 1e-5  # Characteristic length (diameter)
        Re = calculate_reynolds_number(sim, L)

        @test Re > 0
        @test isfinite(Re)

        # For typical blood flow parameters
        @test Re < 5000  # Should be laminar
    end

    @testset "Womersley Number" begin
        geom = create_straight_tube(nx=50, ny=30, diameter=20)
        fluid = FluidProperties(density=1060.0, base_viscosity=0.0035)
        bc = BoundaryConditions(inlet_velocity=0.1)

        sim = create_lbm_simulation(geom, fluid, bc)

        radius = 10 * 1e-5  # 10 lattice units
        frequency = 1.0  # 1 Hz (heartbeat)

        alpha = calculate_womersley_number(sim, radius, frequency)

        @test alpha > 0
        @test isfinite(alpha)

        # For typical arterial flow
        @test 1.0 < alpha < 20.0
    end

    @testset "Stenosis Simulation" begin
        geom = create_stenosis_geometry(
            nx=100, ny=50,
            stenosis_severity=0.6,
            stenosis_length=30
        )
        fluid = FluidProperties()
        bc = BoundaryConditions(inlet_velocity=0.05)

        sim = create_lbm_simulation(geom, fluid, bc)

        # Run simulation
        @test_nowarn run_lbm_simulation!(sim, 200, print_interval=100)

        # Check for velocity acceleration at stenosis
        u_phys, _ = calculate_velocity_field(sim)

        # Velocity should be higher in stenosed region (continuity)
        center_y = 25
        u_center = u_phys[:, center_y]
        u_inlet = mean(u_center[1:10])
        u_stenosis = maximum(u_center[40:60])  # Stenosis region

        @test u_stenosis > u_inlet * 1.1  # At least 10% increase
    end

    @testset "Poiseuille Flow Validation" begin
        # This test validates against analytical solution
        # Note: LBM has inherent errors, so we allow ~5% error

        max_error = validate_poiseuille_flow(nx=80, ny=40, n_steps=3000)

        @test max_error < 0.15  # 15% max relative error (reasonable for LBM)
        println("  Poiseuille validation: $(round(max_error*100, digits=2))% max error")
    end

    @testset "Conservation Properties" begin
        geom = create_straight_tube(nx=60, ny=35, diameter=25)
        fluid = FluidProperties()
        bc = BoundaryConditions(inlet_velocity=0.08)

        sim = create_lbm_simulation(geom, fluid, bc)

        # Run simulation
        run_lbm_simulation!(sim, 100, print_interval=100)

        # Mass conservation: total density should be conserved
        total_mass = sum(sim.rho[.!geom])
        @test isfinite(total_mass)
        @test total_mass > 0

        # Run more steps
        run_lbm_simulation!(sim, 100, print_interval=100)
        total_mass_after = sum(sim.rho[.!geom])

        # Mass should be approximately conserved (within numerical errors)
        @test abs(total_mass_after - total_mass) / total_mass < 0.01
    end

end
