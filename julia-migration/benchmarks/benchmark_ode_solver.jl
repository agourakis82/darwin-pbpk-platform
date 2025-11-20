"""
Benchmark - ODE Solver vs Python

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: Novembro 2025
"""

using BenchmarkTools
using DarwinPBPK.ODEPBPKSolver

# Benchmark
p = PBPKParams(
    clearance_hepatic=10.0,
    clearance_renal=5.0,
    partition_coeffs=Dict("liver" => 2.0, "kidney" => 1.5),
)

# Benchmark single simulation
@benchmark solve($p, 100.0, (0.0, 24.0))

# Benchmark 100 simulations
@benchmark begin
    for _ in 1:100
        solve(p, 100.0, (0.0, 24.0))
    end
end

