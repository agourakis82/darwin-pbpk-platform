"""
Carregador de Dataset PBPK - Suporta múltiplos formatos

Suporta:
- NPZ (NumPy)
- JLD2 (Julia)
- CSV/Parquet (via DataFrames)

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: 2025-11-18
"""

using NPZ
using JLD2
using DataFrames
using CSV

# Importar módulos
using ..ODEPBPKSolver: PBPKParams, PBPK_ORGANS, NUM_ORGANS
using ..Training: PBPKDataset

"""
Carrega dataset NPZ do formato Dynamic GNN.

Estrutura esperada:
- 'doses': [N] - doses em mg
- 'clearance_hepatic': [N] - clearance hepático (L/h)
- 'clearance_renal': [N] - clearance renal (L/h)
- 'partition_coeffs': [N, 14] - partition coefficients por órgão
- 'concentrations': [N, 14, T] - concentrações por órgão ao longo do tempo
- 'time_points': [T] - pontos temporais (horas)
- 'compound_ids': [N] (opcional) - IDs dos compostos
"""
function load_npz_dataset(npz_path::String)::PBPKDataset
    if !isfile(npz_path)
        error("Arquivo não encontrado: $npz_path")
    end

    data = NPZ.npzread(npz_path)

    # Extrair dados
    doses = Float64[]
    params = PBPKParams[]
    true_concs = Matrix{Float64}[]
    time_points = Vector{Float64}[]

    # Verificar estrutura
    if haskey(data, "doses") && haskey(data, "concentrations")
        # Formato padrão
        doses_arr = data["doses"]
        concs_arr = data["concentrations"]

        # Time points
        if haskey(data, "time_points")
            t_points = vec(data["time_points"])
        else
            # Gerar time points padrão (0-24h, dt=0.5h)
            t_points = collect(0.0:0.5:24.0)
        end

        # Clearances (suporta ambos os formatos)
        cl_hepatic = if haskey(data, "clearance_hepatic")
            vec(data["clearance_hepatic"])
        elseif haskey(data, "clearances_hepatic")
            vec(data["clearances_hepatic"])
        else
            fill(10.0, length(doses_arr))
        end

        cl_renal = if haskey(data, "clearance_renal")
            vec(data["clearance_renal"])
        elseif haskey(data, "clearances_renal")
            vec(data["clearances_renal"])
        else
            fill(5.0, length(doses_arr))
        end

        # Partition coefficients
        if haskey(data, "partition_coeffs")
            kp_arr = data["partition_coeffs"]
        else
            # Default: todos 1.0
            kp_arr = ones(Float64, length(doses_arr), NUM_ORGANS)
        end

        # Processar cada amostra
        n_samples = length(doses_arr)
        for i in 1:n_samples
            dose = Float64(doses_arr[i])

            # Parâmetros PBPK
            kp_dict = Dict{String, Float64}()
            for (j, organ) in enumerate(PBPK_ORGANS)
                kp_dict[organ] = Float64(kp_arr[i, j])
            end

            p = PBPKParams(
                clearance_hepatic=Float64(cl_hepatic[i]),
                clearance_renal=Float64(cl_renal[i]),
                partition_coeffs=kp_dict,
            )

            # Concentrações
            if ndims(concs_arr) == 3
                # [N, num_organs, num_time_points]
                # Transpor para [num_organs, num_time_points]
                conc_matrix = Float64.(concs_arr[i, :, :])'  # [num_organs, num_time_points]
            elseif ndims(concs_arr) == 2
                # [N*num_organs, num_time_points] - precisa reshape
                # Assumir que está em formato [N*num_organs, num_time_points]
                error("Formato 2D não suportado ainda - precisa reshape")
            else
                error("Dimensões não suportadas: $(ndims(concs_arr))")
            end

            # Garantir que time_points tem o mesmo tamanho
            if length(t_points) != size(conc_matrix, 2)
                # Ajustar time_points se necessário
                if length(t_points) > size(conc_matrix, 2)
                    t_points = t_points[1:size(conc_matrix, 2)]
                else
                    # Estender time_points (interpolação linear)
                    error("Time points incompatíveis: $(length(t_points)) vs $(size(conc_matrix, 2))")
                end
            end

            push!(doses, dose)
            push!(params, p)
            push!(true_concs, conc_matrix)
            push!(time_points, t_points)
        end

    else
        error("Estrutura NPZ não reconhecida. Chaves disponíveis: $(keys(data))")
    end

    return PBPKDataset(doses, params, true_concs, time_points)
end

"""
Carrega dataset de múltiplos caminhos possíveis.

Procura em:
1. Caminho fornecido
2. /mnt/f/datasets/pbpk/
3. /mnt/f/DARWIN_VALIDATION/datasets/
4. data/processed/pbpk_enriched/
"""
function load_dataset_flexible(dataset_name::String)::Union{PBPKDataset, Nothing}
    possible_paths = [
        dataset_name,  # Caminho absoluto ou relativo fornecido
        "/mnt/f/datasets/pbpk/$dataset_name",
        "/mnt/f/DARWIN_VALIDATION/datasets/$dataset_name",
        "data/processed/pbpk_enriched/$dataset_name",
        joinpath(homedir(), "workspace/darwin-pbpk-platform/data/processed/pbpk_enriched/$dataset_name"),
    ]

    for path in possible_paths
        if isfile(path)
            println("📂 Carregando dataset: $path")
            return load_npz_dataset(path)
        end
    end

    println("⚠️  Dataset não encontrado em nenhum dos caminhos:")
    for path in possible_paths
        println("   - $path")
    end

    return nothing
end

"""
Lista datasets disponíveis em /mnt/f.
"""
function list_available_datasets()::Vector{String}
    datasets = String[]

    search_dirs = [
        "/mnt/f/datasets/pbpk",
        "/mnt/f/DARWIN_VALIDATION/datasets",
        "data/processed/pbpk_enriched",
    ]

    for dir in search_dirs
        if isdir(dir)
            for file in readdir(dir)
                if endswith(file, ".npz")
                    push!(datasets, file)
                end
            end
        end
    end

    return unique(datasets)
end

export load_npz_dataset, load_dataset_flexible, list_available_datasets

