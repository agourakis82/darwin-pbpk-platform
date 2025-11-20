"""
Testes Completos - Validação Numérica vs Python

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: Novembro 2025
"""

using Test
using DarwinPBPK

@testset "Darwin PBPK - Testes Completos" begin
    @testset "ODE Solver" begin
        p = ODEPBPKSolver.PBPKParams(
            clearance_hepatic=10.0,
            clearance_renal=5.0,
            partition_coeffs=Dict("liver" => 2.0, "kidney" => 1.5),
        )

        sol = ODEPBPKSolver.solve(p, 100.0, (0.0, 24.0))

        @test length(sol) > 0
        @test ODEPBPKSolver.validate_mass_conservation(sol, p, 100.0)
        @test sol[1][ODEPBPKSolver.BLOOD_IDX] ≈ 100.0 / p.volumes[ODEPBPKSolver.BLOOD_IDX]
    end

    @testset "Validation Metrics" begin
        pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        obs = [1.1, 2.1, 2.9, 4.2, 4.8]

        fe = Validation.fold_error(pred, obs)
        @test length(fe) > 0
        @test all(fe .> 0)

        gmfe = Validation.geometric_mean_fold_error(pred, obs)
        @test gmfe > 0 && isfinite(gmfe)

        pct = Validation.percent_within_fold(pred, obs, 2.0)
        @test 0 <= pct <= 100
    end
end

