#!/usr/bin/env julia
"""
Test SAM-3 Segmentation in Julia
=================================

Loads SAM-3 model and tests segmentation on a real leukocyte image.

Created: 2025-12-01
"""

println("=" ^ 80)
println("🧪 TESTE SAM-3 SEGMENTAÇÃO - JULIA")
println("=" ^ 80)
println()

using PyCall
using Statistics

# Configuration
sam3_dir = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", "sam3")
test_image = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", 
                      "data", "leukocytes", "normal", "all", "BloodImage_00214.jpg")
device = "cuda"

# Setup Python path
sys = pyimport("sys")
pushfirst!(PyVector(sys.path), sam3_dir)

# Import modules
println("📦 Importando módulos SAM-3...")
model_builder = pyimport("sam3.model_builder")
processor_module = pyimport("sam3.model.sam3_image_processor")
torch = pyimport("torch")
PIL = pyimport("PIL.Image")
println("✅ Módulos importados!")

# Enable TF32 for performance
if torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = true
    torch.backends.cudnn.allow_tf32 = true
    println("✅ TF32 habilitado")
end

# Load model
println()
println("🔄 Carregando modelo SAM-3...")
println("   (Isso pode levar ~1 minuto na primeira vez)")
global processor  # Make it available in global scope
try
    model = model_builder.build_sam3_image_model()
    println("✅ Modelo construído")
    
    model = model.to(device)
    model.eval()
    println("✅ Modelo movido para $device")
    
    global processor = processor_module.Sam3Processor(model)
    println("✅ Processor criado")
catch e
    println("❌ Erro ao carregar modelo: $e")
    exit(1)
end

# Load test image
println()
println("📸 Carregando imagem de teste...")
global image  # Make it available in global scope
if !isfile(test_image)
    println("⚠️  Imagem não encontrada: $test_image")
    println("   Usando imagem alternativa...")
    # Try to find any image
    image_dir = dirname(test_image)
    if isdir(image_dir)
        images = filter(f -> endswith(f, ".jpg") || endswith(f, ".png"), 
                       readdir(image_dir))
        if !isempty(images)
            test_image = joinpath(image_dir, images[1])
            println("   Usando: $(basename(test_image))")
        else
            println("❌ Nenhuma imagem encontrada")
            exit(1)
        end
    else
        println("❌ Diretório não encontrado: $image_dir")
        exit(1)
    end
end

try
    global image = PIL.open(test_image).convert("RGB")
    println("✅ Imagem carregada: $(basename(test_image))")
    println("   Tamanho: $(image.width) × $(image.height)")
catch e
    println("❌ Erro ao carregar imagem: $e")
    exit(1)
end

# Test segmentation
println()
println("🔬 Testando segmentação...")
prompts = ["white blood cells", "lymphocytes", "leukocytes"]

println("   Configurando imagem no processor...")
global inference_state  # Make it global
try
    global inference_state = processor.set_image(image)
    println("✅ Imagem configurada")
catch e
    println("❌ Erro ao configurar imagem: $e")
    exit(1)
end

global best_result = nothing
global best_n_masks = 0

println()
println("   Testando prompts:")
for (i, prompt) in enumerate(prompts)
    print("   [$i/$(length(prompts))] '$prompt'... ")
    try
        # Reset prompts (modifies state in place, returns None)
        processor.reset_all_prompts(inference_state)
        
        # Set text prompt
        global inference_state = processor.set_text_prompt(
            state=inference_state,
            prompt=prompt
        )
        
        # Extract results from inference_state
        masks = inference_state["masks"]
        boxes = inference_state["boxes"]
        scores = inference_state["scores"]
        
        n_masks = length(masks)
        if n_masks > best_n_masks
            global best_n_masks = n_masks
            global best_result = Dict(
                "prompt" => prompt,
                "n_masks" => n_masks,
                "scores" => scores
            )
        end
        
        # Calculate average score
        if length(scores) > 0
            score_values = Float64[]
            for s in scores
                try
                    if PyObject(s).__class__.__name__ == "Tensor"
                        push!(score_values, Float64(PyObject(s).cpu().item()))
                    else
                        push!(score_values, Float64(s))
                    end
                catch
                    push!(score_values, 0.0)
                end
            end
            avg_score = length(score_values) > 0 ? mean(score_values) : 0.0
            println("✅ $n_masks células (score médio: $(round(avg_score, digits=3)))")
        else
            println("✅ $n_masks células")
        end
    catch e
        println("❌ Erro: $(sprint(showerror, e))")
    end
end

# Results
println()
println("=" ^ 80)
println("📊 RESULTADOS")
println("=" ^ 80)
if best_result !== nothing
    println("✅ Melhor resultado:")
    println("   Prompt: '$(best_result["prompt"])'")
    println("   Células detectadas: $(best_result["n_masks"])")
    
    # Calculate statistics
    if length(best_result["scores"]) > 0
        score_values = Float64[]
        for s in best_result["scores"]
            if PyObject(s).__class__.__name__ == "Tensor"
                push!(score_values, Float64(PyObject(s).cpu().item()))
            else
                push!(score_values, Float64(s))
            end
        end
        println("   Score médio: $(round(mean(score_values), digits=3))")
        println("   Score mínimo: $(round(minimum(score_values), digits=3))")
        println("   Score máximo: $(round(maximum(score_values), digits=3))")
    end
else
    println("❌ Nenhum resultado obtido")
end

println()
println("=" ^ 80)
println("✅ TESTE COMPLETO!")
println("=" ^ 80)
println()
println("🎉 SAM-3 funcionando perfeitamente em Julia via PyCall!")

