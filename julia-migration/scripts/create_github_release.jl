#!/usr/bin/env julia
"""
Cria release no GitHub usando a API

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: 2025-11-18
"""

using HTTP
using JSON
using Printf

const GITHUB_API = "https://api.github.com"
const REPO = "agourakis82/darwin-pbpk-platform"
const TAG = "v2.0.0-julia"

function get_github_token()
    """Obtém token do GitHub de variável de ambiente ou arquivo"""
    token = get(ENV, "GITHUB_TOKEN", "")

    if isempty(token)
        # Tentar ler de arquivo
        token_file = joinpath(homedir(), ".github_token")
        if isfile(token_file)
            token = strip(read(token_file, String))
        end
    end

    if isempty(token)
        error("GITHUB_TOKEN não encontrado. Configure via ENV ou ~/.github_token")
    end

    return token
end

function read_release_notes()
    """Lê release notes do arquivo"""
    notes_file = joinpath(dirname(dirname(@__DIR__)), "RELEASE_v2.0.0-julia.md")

    if !isfile(notes_file)
        error("Arquivo de release notes não encontrado: ", notes_file)
    end

    return read(notes_file, String)
end

function create_github_release()
    """Cria release no GitHub"""
    token = get_github_token()
    release_notes = read_release_notes()

    url = "$GITHUB_API/repos/$REPO/releases"

    headers = [
        "Authorization" => "token $token",
        "Accept" => "application/vnd.github.v3+json",
        "Content-Type" => "application/json",
    ]

    body = Dict(
        "tag_name" => TAG,
        "name" => "v2.0.0-julia - Migração Completa para Julia",
        "body" => release_notes,
        "draft" => false,
        "prerelease" => false,
    )

    println("=" ^ 80)
    println("CRIANDO RELEASE NO GITHUB")
    println("=" ^ 80)
    println()
    println("Tag: ", TAG)
    println("Repo: ", REPO)
    println()

    try
        response = HTTP.post(
            url,
            headers,
            body = JSON.json(body),
        )

        if response.status == 201
            result = JSON.parse(String(response.body))
            println("✅ Release criada com sucesso!")
            println()
            println("URL: ", result["html_url"])
            println("ID: ", result["id"])
            return result
        else
            error_msg = String(response.body)
            if occursin("already exists", error_msg)
                println("⚠️  Release já existe. Atualizando...")
                # Tentar atualizar release existente
                return update_existing_release(token, release_notes)
            else
                error("Erro ao criar release: ", response.status, " - ", error_msg)
            end
        end
    catch e
        error("Erro ao criar release: ", e)
    end
end

function update_existing_release(token, release_notes)
    """Atualiza release existente"""
    # Buscar release existente
    url = "$GITHUB_API/repos/$REPO/releases/tags/$TAG"

    headers = [
        "Authorization" => "token $token",
        "Accept" => "application/vnd.github.v3+json",
    ]

    try
        response = HTTP.get(url, headers)
        if response.status == 200
            release = JSON.parse(String(response.body))
            release_id = release["id"]

            # Atualizar release
            update_url = "$GITHUB_API/repos/$REPO/releases/$release_id"
            body = Dict(
                "body" => release_notes,
                "draft" => false,
                "prerelease" => false,
            )

            response = HTTP.patch(
                update_url,
                vcat(headers, ["Content-Type" => "application/json"]),
                body = JSON.json(body),
            )

            if response.status == 200
                println("✅ Release atualizada com sucesso!")
                result = JSON.parse(String(response.body))
                println("URL: ", result["html_url"])
                return result
            else
                error("Erro ao atualizar release: ", response.status)
            end
        end
    catch e
        error("Erro ao atualizar release: ", e)
    end
end

function main()
    try
        result = create_github_release()
        println()
        println("=" ^ 80)
        println("✅ RELEASE CRIADA COM SUCESSO!")
        println("=" ^ 80)
    catch e
        println()
        println("=" ^ 80)
        println("❌ ERRO AO CRIAR RELEASE")
        println("=" ^ 80)
        println()
        println("Erro: ", e)
        println()
        println("Solução:")
        println("1. Configure GITHUB_TOKEN como variável de ambiente:")
        println("   export GITHUB_TOKEN=seu_token")
        println()
        println("2. Ou crie arquivo ~/.github_token com o token")
        println()
        println("3. Para criar token:")
        println("   https://github.com/settings/tokens")
        println("   (permissões: repo)")
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


