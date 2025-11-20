"""
Testes Unitários - ODE Solver

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: Novembro 2025
"""

using Test
using DarwinPBPK.ODEPBPKSolver

@testset "ODE Solver" begin
    # Teste básico
    p = PBPKParams(
        clearance_hepatic=10.0,
        clearance_renal=5.0,
        partition_coeffs=Dict("liver" => 2.0, "kidney" => 1.5),
    )

    sol = solve(p, 100.0, (0.0, 24.0))

    @test length(sol) > 0
    @test validate_mass_conservation(sol, p, 100.0)
    @test sol[1][BLOOD_IDX] ≈ 100.0 / p.volumes[BLOOD_IDX]
end

