#!/usr/bin/env julia
"""
Test SAM-3 Simple - Minimal Dependencies
========================================

Test SAM-3 integration with minimal dependencies.
Only tests PyCall connection to Python SAM-3.

Created: 2025-12-01
"""

println("=" ^ 80)
println("🧪 TESTE SAM-3 JULIA - VERSÃO SIMPLES")
println("=" ^ 80)
println()

# Try to use PyCall (might already be in environment)
println("📦 Tentando carregar PyCall...")
try
    using PyCall
    println("✅ PyCall carregado!")
catch e
    println("❌ PyCall não disponível: $e")
    println()
    println("💡 Para instalar PyCall:")
    println("   julia --project=julia-migration -e 'using Pkg; Pkg.add(\"PyCall\")'")
    exit(1)
end

# Test Python
println()
println("📦 Testando Python...")
try
    sys = pyimport("sys")
    println("✅ Python disponível: $(sys.version)")
catch e
    println("❌ Erro ao importar Python: $e")
    exit(1)
end

# Test PyTorch
println()
println("📦 Testando PyTorch...")
try
    torch = pyimport("torch")
    println("✅ PyTorch disponível: $(torch.__version__)")
    if torch.cuda.is_available()
        println("   ✅ CUDA: $(torch.cuda.get_device_name(0))")
    else
        println("   ⚠️  CUDA não disponível")
    end
catch e
    println("⚠️  PyTorch não disponível: $e")
end

# Test SAM-3 directory
println()
println("📦 Verificando SAM-3...")
sam3_dir = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", "sam3")
if isdir(sam3_dir)
    println("✅ Diretório encontrado: $sam3_dir")
    
    # Add to path
    sys_path = pyimport("sys").path
    pushfirst!(PyVector(sys_path), sam3_dir)
    println("✅ Adicionado ao Python path")
    
    # Try importing
    println()
    println("📦 Tentando importar SAM-3...")
    try
        model_builder = pyimport("sam3.model_builder")
        println("✅ sam3.model_builder importado!")
    catch e
        println("⚠️  Erro ao importar sam3.model_builder:")
        println("   $(sprint(showerror, e, catch_backtrace()))")
    end
    
    try
        processor = pyimport("sam3.model.sam3_image_processor")
        println("✅ sam3.model.sam3_image_processor importado!")
    catch e
        println("⚠️  Erro ao importar sam3.model.sam3_image_processor:")
        println("   $(sprint(showerror, e, catch_backtrace()))")
    end
else
    println("⚠️  Diretório não encontrado: $sam3_dir")
end

println()
println("=" ^ 80)
println("✅ TESTE COMPLETO!")
println("=" ^ 80)








