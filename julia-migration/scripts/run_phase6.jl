#!/usr/bin/env julia
"""
Script para executar FASE 6 completa - Otimização Final

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: Novembro 2025
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Adicionar src ao LOAD_PATH
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

println("=" ^ 80)
println("FASE 6: Otimização Final - Darwin PBPK Platform")
println("=" ^ 80)
println()

# 1. Verificar ambiente
println("1️⃣  Verificando ambiente...")
try
    using DifferentialEquations
    using BenchmarkTools
    using Test
    println("✅ Dependências básicas disponíveis!")
    # CUDA é opcional
    try
        using CUDA
        println("✅ CUDA disponível!")
    catch
        println("⚠️  CUDA não disponível (opcional)")
    end
    # Flux é opcional para testes básicos
    try
        using Flux
        println("✅ Flux disponível!")
    catch
        println("⚠️  Flux não disponível (opcional)")
    end
catch e
    println("❌ Erro ao carregar dependências: ", e)
    exit(1)
end

# 2. Carregar módulo
println("\n2️⃣  Carregando módulo DarwinPBPK...")
try
    include(joinpath(@__DIR__, "..", "src", "DarwinPBPK.jl"))
    using .DarwinPBPK
    println("✅ Módulo carregado com sucesso!")
catch e
    println("❌ Erro ao carregar módulo: ", e)
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end

# 3. Executar testes básicos
println("\n3️⃣  Executando testes básicos...")
try
    using .DarwinPBPK
    using .DarwinPBPK.ODEPBPKSolver
    using .DarwinPBPK.DynamicGNN

    @testset "Testes Básicos" begin
        # Teste ODE Solver
        println("  - Testando ODE Solver...")
        params = ODEPBPKSolver.PBPKParams(
            clearance_hepatic=10.0,
            clearance_renal=5.0,
            partition_coeffs=Dict("liver" => 2.0, "kidney" => 1.5)
        )
        @test params.clearance_hepatic ≈ 10.0
        @test params.clearance_renal ≈ 5.0
        println("  ✅ ODE Solver: OK")

        # Teste Dynamic GNN
        println("  - Testando Dynamic GNN...")
        model = DynamicGNN.DynamicPBPKGNN(node_dim=16, hidden_dim=32, num_gnn_layers=2)
        @test model.node_dim == 16
        @test model.hidden_dim == 32
        println("  ✅ Dynamic GNN: OK")

        # Teste Validation
        println("  - Testando Validation...")
        using .DarwinPBPK.Validation
        pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        obs = [1.1, 2.1, 2.9, 4.2, 4.8]
        fe = Validation.fold_error(pred, obs)
        @test length(fe) == 5
        @test all(fe .> 0)
        println("  ✅ Validation: OK")
    end
    println("✅ Testes básicos passaram!")
catch e
    println("❌ Erro nos testes: ", e)
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

# 4. Benchmarks básicos
println("\n4️⃣  Executando benchmarks básicos...")
try
    using BenchmarkTools
    using .DarwinPBPK.ODEPBPKSolver

    params = ODEPBPKSolver.PBPKParams(
        clearance_hepatic=10.0,
        clearance_renal=5.0,
        partition_coeffs=Dict("liver" => 2.0, "kidney" => 1.5)
    )

    # Benchmark criação de parâmetros
    println("  - Benchmark: Criação de parâmetros...")
    result = @benchmark ODEPBPKSolver.PBPKParams(
        clearance_hepatic=10.0,
        clearance_renal=5.0,
        partition_coeffs=Dict("liver" => 2.0, "kidney" => 1.5)
    )
    println("    Tempo médio: $(mean(result.times) / 1e6) ms")
    println("    Alocações: $(result.allocs)")
    println("  ✅ Benchmarks básicos concluídos!")
catch e
    println("⚠️  Benchmarks básicos: ", e)
end

println("\n" * "=" ^ 80)
println("✅ FASE 6 - Execução básica concluída!")
println("=" ^ 80)

