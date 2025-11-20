#!/usr/bin/env julia
"""
Remove todos os arquivos Python após migração completa.

⚠️ ATENÇÃO: Este script remove permanentemente arquivos Python!
Use apenas após confirmar que toda funcionalidade foi migrada.

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: 2025-11-18
"""

using Printf

const ROOT = dirname(dirname(@__DIR__))

function remove_python_files(dry_run::Bool = true)
    """Remove arquivos Python (dry-run por padrão)"""
    removed = 0
    skipped = 0

    for (root, dirs, files) in walkdir(ROOT)
        # Pular julia-migration
        if occursin("julia-migration", root)
            continue
        end

        # Pular __pycache__ (será removido separadamente)
        if occursin("__pycache__", root)
            continue
        end

        for file in files
            if endswith(file, ".py")
                filepath = joinpath(root, file)
                relpath_file = relpath(filepath, ROOT)

                if dry_run
                    println("  [DRY-RUN] Removeria: ", relpath_file)
                else
                    try
                        rm(filepath)
                        println("  ✅ Removido: ", relpath_file)
                        removed += 1
                    catch e
                        println("  ❌ Erro ao remover ", relpath_file, ": ", e)
                        skipped += 1
                    end
                end
            end
        end
    end

    # Remover __pycache__
    for (root, dirs, files) in walkdir(ROOT)
        if occursin("__pycache__", root) && !occursin("julia-migration", root)
            if dry_run
                println("  [DRY-RUN] Removeria diretório: ", relpath(root, ROOT))
            else
                try
                    rm(root, recursive=true)
                    println("  ✅ Removido diretório: ", relpath(root, ROOT))
                    removed += 1
                catch e
                    println("  ❌ Erro ao remover ", root, ": ", e)
                    skipped += 1
                end
            end
        end
    end

    return removed, skipped
end

function remove_python_dependencies()
    """Remove arquivos relacionados a Python"""
    files_to_remove = [
        "requirements.txt",
        "setup.py",
        "pyproject.toml",  # Se for Python-only
        ".python-version",
    ]

    removed = 0
    for file in files_to_remove
        filepath = joinpath(ROOT, file)
        if isfile(filepath)
            try
                rm(filepath)
                println("  ✅ Removido: ", file)
                removed += 1
            catch e
                println("  ❌ Erro ao remover ", file, ": ", e)
            end
        end
    end

    return removed
end

function main()
    println("=" ^ 80)
    println("REMOÇÃO DE ARQUIVOS PYTHON")
    println("=" ^ 80)
    println()
    println("⚠️  ATENÇÃO: Este script remove permanentemente arquivos Python!")
    println()

    # Dry-run primeiro
    println("1️⃣  Executando DRY-RUN...")
    removed_dry, skipped_dry = remove_python_files(true)
    println()
    println(@sprintf("  Arquivos que seriam removidos: %d", removed_dry))
    println(@sprintf("  Arquivos que seriam pulados: %d", skipped_dry))
    println()

    println("2️⃣  Removendo dependências Python...")
    deps_removed = remove_python_dependencies()
    println(@sprintf("  Dependências removidas: %d", deps_removed))
    println()

    println("=" ^ 80)
    println("Para executar a remoção real, edite o script e mude dry_run=false")
    println("=" ^ 80)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

