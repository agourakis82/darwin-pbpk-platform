#!/usr/bin/env julia
"""
Test SAM-3 Basic Integration in Julia
=====================================

Quick test to validate SAM-3 can be called from Julia via PyCall.

Created: 2025-12-01
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

println("=" ^ 80)
println("🧪 TESTE SAM-3 JULIA INTEGRATION")
println("=" ^ 80)
println()

# Step 1: Check PyCall
println("📦 Passo 1: Verificando PyCall.jl...")
try
    using PyCall
    println("✅ PyCall.jl carregado")
catch e
    println("⚠️  PyCall.jl não encontrado - tentando instalar...")
    Pkg.add("PyCall")
    using PyCall
    println("✅ PyCall.jl instalado e carregado")
end

# Step 2: Test Python import
println()
println("📦 Passo 2: Testando importação Python...")
try
    sys = pyimport("sys")
    println("✅ Python sys importado")
    println("   Versão: $(sys.version)")
catch e
    println("❌ Erro ao importar Python: $e")
    exit(1)
end

# Step 3: Test PIL
println()
println("📦 Passo 3: Testando PIL...")
try
    PIL = pyimport("PIL.Image")
    println("✅ PIL importado")
catch e
    println("⚠️  PIL não disponível (opcional): $e")
end

# Step 4: Test PyTorch
println()
println("📦 Passo 4: Testando PyTorch...")
try
    torch = pyimport("torch")
    println("✅ PyTorch importado: $(torch.__version__)")
    if torch.cuda.is_available()
        println("   ✅ CUDA disponível: $(torch.cuda.get_device_name(0))")
    else
        println("   ⚠️  CUDA não disponível")
    end
catch e
    println("⚠️  PyTorch não disponível: $e")
end

# Step 5: Test SAM-3 directory
println()
println("📦 Passo 5: Verificando diretório SAM-3...")
sam3_dir = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", "sam3")
if isdir(sam3_dir)
    println("✅ Diretório SAM-3 encontrado: $sam3_dir")
    
    # Add to Python path
    sys_path = pyimport("sys").path
    pushfirst!(PyVector(sys_path), sam3_dir)
    println("✅ SAM-3 adicionado ao Python path")
    
    # Try importing SAM-3 modules
    println()
    println("📦 Passo 6: Testando importação SAM-3...")
    try
        model_builder = pyimport("sam3.model_builder")
        println("✅ sam3.model_builder importado")
    catch e
        println("⚠️  Erro ao importar sam3.model_builder:")
        println("   $e")
    end
    
    try
        processor_module = pyimport("sam3.model.sam3_image_processor")
        println("✅ sam3.model.sam3_image_processor importado")
    catch e
        println("⚠️  Erro ao importar sam3.model.sam3_image_processor:")
        println("   $e")
    end
else
    println("⚠️  Diretório SAM-3 não encontrado: $sam3_dir")
end

println()
println("=" ^ 80)
println("✅ TESTE DE INTEGRAÇÃO COMPLETO!")
println("=" ^ 80)
println()
println("Se todas as dependências estão OK, podemos prosseguir com teste completo.")








