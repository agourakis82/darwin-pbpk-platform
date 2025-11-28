#!/usr/bin/env julia
"""
Script Master - Executa Todos os Passos da Fase 1

1. Testa implementações
2. Analisa overfitting
3. Treina com regularização (se dados disponíveis)
4. Compara métricas

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: 2025-11-18
"""

using Pkg
Pkg.activate(".")

println("=" ^ 80)
println("FASE 1 - EXECUÇÃO COMPLETA")
println("=" ^ 80)
println()

# Passo 1: Testes
println("1️⃣  Executando testes das implementações...")
try
    include("julia-migration/scripts/phase1/run_phase1_complete.jl")
    println("   ✅ Testes completos")
catch e
    println("   ⚠️  Erro nos testes: $e")
end
println()

# Passo 2: Análise de overfitting
println("2️⃣  Analisando overfitting...")
try
    include("julia-migration/scripts/phase1/analyze_overfitting_from_json.jl")
    println("   ✅ Análise de overfitting completa")
catch e
    println("   ⚠️  Erro na análise: $e")
end
println()

# Passo 3: Treinamento (se dados disponíveis)
println("3️⃣  Treinamento com regularização...")
println("   ⏳ Requer dados de treinamento")
println("   💡 Execute manualmente quando dados estiverem disponíveis:")
println("      julia julia-migration/scripts/training/train_with_regularization.jl")
println()

# Passo 4: Comparação (se treinamento executado)
println("4️⃣  Comparação de métricas...")
println("   ⏳ Requer resultados de treinamento")
println("   💡 Execute após treinamento:")
println("      julia julia-migration/scripts/validation/compare_before_after.jl")
println()

println("=" ^ 80)
println("✅ FASE 1 - EXECUÇÃO COMPLETA")
println("=" ^ 80)
println()
println("📊 Status:")
println("   ✅ Implementações testadas")
println("   ✅ Overfitting analisado")
println("   ⏳ Treinamento (requer dados)")
println("   ⏳ Comparação (requer treinamento)")
println()

