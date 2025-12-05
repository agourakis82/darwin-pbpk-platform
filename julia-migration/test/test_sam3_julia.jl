#!/usr/bin/env julia
"""
Test SAM-3 Julia Integration
============================

Quick test to validate SAM-3 integration via PyCall in Julia.

Created: 2025-12-01
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Check if PyCall is available
try
    using PyCall
    @info "✅ PyCall.jl carregado"
catch e
    @warn "⚠️  PyCall.jl não encontrado - instalando..."
    Pkg.add("PyCall")
    using PyCall
end

# Test basic Python import
@info "🧪 Testando importação Python básica..."

try
    sys = pyimport("sys")
    @info "✅ Python sys importado: $(sys.version)"
catch e
    @error "❌ Erro ao importar Python" exception=(e, catch_backtrace())
    exit(1)
end

# Test PIL import
@info "🧪 Testando importação PIL..."
try
    PIL = pyimport("PIL.Image")
    @info "✅ PIL importado"
catch e
    @warn "⚠️  PIL não disponível: $e"
end

# Test torch import
@info "🧪 Testando importação PyTorch..."
try
    torch = pyimport("torch")
    @info "✅ PyTorch importado: $(torch.__version__)"
    if torch.cuda.is_available()
        @info "   CUDA disponível: $(torch.cuda.get_device_name(0))"
    else
        @warn "   CUDA não disponível"
    end
catch e
    @warn "⚠️  PyTorch não disponível: $e"
end

# Test SAM-3 import
@info "🧪 Testando importação SAM-3..."
sam3_dir = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", "sam3")
if isdir(sam3_dir)
    @info "   Diretório SAM-3 encontrado: $sam3_dir"
    
    try
        pushfirst!(PyVector(pyimport("sys")."path"), sam3_dir)
        @info "   SAM-3 adicionado ao Python path"
        
        # Try importing SAM-3 modules
        try
            model_builder = pyimport("sam3.model_builder")
            @info "✅ sam3.model_builder importado"
        catch e
            @warn "⚠️  Erro ao importar sam3.model_builder: $e"
        end
        
        try
            processor_module = pyimport("sam3.model.sam3_image_processor")
            @info "✅ sam3.model.sam3_image_processor importado"
        catch e
            @warn "⚠️  Erro ao importar sam3.model.sam3_image_processor: $e"
        end
        
    catch e
        @error "❌ Erro ao configurar SAM-3" exception=(e, catch_backtrace())
    end
else
    @warn "⚠️  Diretório SAM-3 não encontrado: $sam3_dir"
end

@info ""
@info "=" ^ 80
@info "✅ TESTE DE INTEGRAÇÃO COMPLETO!"
@info "=" ^ 80








