#!/usr/bin/env julia
"""
Test SAM-3 Standalone - Independent Test
=========================================

Test SAM-3 integration independently, using global PyCall.
Doesn't require project dependencies.

Created: 2025-12-01
"""

println("=" ^ 80)
println("🧪 TESTE SAM-3 JULIA - STANDALONE")
println("=" ^ 80)
println()

# Load PyCall (global environment)
println("📦 Carregando PyCall...")
try
    using PyCall
    println("✅ PyCall carregado!")
catch e
    println("❌ Erro ao carregar PyCall: $e")
    println()
    println("💡 Instalar: julia -e 'using Pkg; Pkg.add(\"PyCall\")'")
    exit(1)
end

# Test Python
println()
println("📦 Testando Python...")
try
    sys = pyimport("sys")
    py_version = sys.version
    println("✅ Python disponível!")
    println("   Versão: $py_version")
catch e
    println("❌ Erro ao acessar Python: $e")
    exit(1)
end

# Test PyTorch
println()
println("📦 Testando PyTorch...")
try
    torch = pyimport("torch")
    torch_version = torch.__version__
    println("✅ PyTorch disponível!")
    println("   Versão: $torch_version")
    
    if torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0)
        println("   ✅ CUDA disponível: $device_name")
    else
        println("   ⚠️  CUDA não disponível (usando CPU)")
    end
catch e
    println("⚠️  PyTorch não disponível: $e")
    println("   (Necessário para SAM-3)")
end

# Test PIL
println()
println("📦 Testando PIL...")
try
    PIL = pyimport("PIL.Image")
    println("✅ PIL disponível!")
catch e
    println("⚠️  PIL não disponível: $e")
    println("   (Necessário para processar imagens)")
end

# Test SAM-3 directory and imports
println()
println("📦 Verificando SAM-3...")
sam3_dir = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", "sam3")

if !isdir(sam3_dir)
    println("❌ Diretório SAM-3 não encontrado: $sam3_dir")
    println("   Verifique se o SAM-3 foi clonado corretamente.")
    exit(1)
end

println("✅ Diretório SAM-3 encontrado: $sam3_dir")

# Add to Python path
println()
println("📦 Adicionando SAM-3 ao Python path...")
try
    sys_path = pyimport("sys").path
    pushfirst!(PyVector(sys_path), sam3_dir)
    println("✅ SAM-3 adicionado ao Python path")
catch e
    println("❌ Erro ao adicionar ao path: $e")
    exit(1)
end

# Test SAM-3 imports
println()
println("📦 Testando importação SAM-3...")

# Test model_builder
println("   Tentando importar sam3.model_builder...")
try
    model_builder = pyimport("sam3.model_builder")
    println("   ✅ sam3.model_builder importado com sucesso!")
catch e
    println("   ❌ Erro ao importar sam3.model_builder:")
    println("      $e")
end

# Test image processor
println("   Tentando importar sam3.model.sam3_image_processor...")
try
    processor_module = pyimport("sam3.model.sam3_image_processor")
    println("   ✅ sam3.model.sam3_image_processor importado com sucesso!")
catch e
    println("   ❌ Erro ao importar sam3.model.sam3_image_processor:")
    println("      $e")
end

# Summary
println()
println("=" ^ 80)
println("📊 RESUMO DO TESTE")
println("=" ^ 80)
println()

# Check what worked
components = Dict(
    "PyCall" => true,  # We got here, so it works
    "Python" => true,  # We checked sys, so it works
    "PyTorch" => false,
    "PIL" => false,
    "SAM-3 Model Builder" => false,
    "SAM-3 Processor" => false
)

# Re-check silently
try
    pyimport("torch")
    components["PyTorch"] = true
catch
end

try
    pyimport("PIL.Image")
    components["PIL"] = true
catch
end

try
    pyimport("sam3.model_builder")
    components["SAM-3 Model Builder"] = true
catch
end

try
    pyimport("sam3.model.sam3_image_processor")
    components["SAM-3 Processor"] = true
catch
end

# Print summary
for (component, status) in components
    icon = status ? "✅" : "❌"
    println("$icon $component")
end

println()
println("=" ^ 80)

if all(values(components))
    println("✅ TODOS OS COMPONENTES DISPONÍVEIS!")
    println("   Pronto para testar SAM-3 em Julia!")
else
    println("⚠️  ALGUNS COMPONENTES FALTANDO")
    println("   Revise os erros acima antes de prosseguir.")
end

println("=" ^ 80)








