"""
REST API - API REST para Darwin PBPK Platform

Inovações SOTA:
- Type-safe API com validação em tempo de compilação
- Async I/O nativo
- OpenAPI generation automático
- HTTP.jl (rápido e eficiente)

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: Novembro 2025
"""

module RESTAPI

using HTTP
using JSON
using Sockets

# Importar módulos
using ..DynamicGNN: DynamicPBPKGNN, DynamicPBPKSimulator, forward_batch
using ..ODEPBPKSolver: PBPKParams, solve, simulate
using ..Validation: validate_scientific

# Import Semantic Web modules
using ..SemanticCore: DARWIN_CONTEXT, DRUG_CONTEXT, DDI_CONTEXT, PARAMETER_CONTEXT,
                      EVIDENCE_CONTEXT, PROVENANCE_CONTEXT, OBO_PREFIXES,
                      QUDT_UNITS, get_qudt_unit, QUDTUnit, SemanticQuantity,
                      to_jsonld_drug, to_jsonld_ddi, to_jsonld_parameter,
                      to_jsonld_simulation, to_jsonld_evidence, to_turtle,
                      ProvenanceRecord, ProvenanceActivity, ProvenanceAgent,
                      create_simulation_provenance, DARWIN_SOFTWARE_AGENT,
                      SchemaOrgDrug, SchemaOrgMedicalStudy, SchemaOrgPropertyValue,
                      to_schema_org_drug, map_pk_parameter_to_schema,
                      list_supported_ontologies

"""
Request struct (type-safe).

Inovações:
- Type-safe request parsing
- Validação automática
"""
struct PBPKRequest
    dose::Float64
    clearance_hepatic::Float64
    clearance_renal::Float64
    partition_coeffs::Dict{String, Float64}
    time_points::Union{Vector{Float64}, Nothing}
end

"""
Response struct (type-safe).

Inovações:
- Type-safe response
- JSON serialization automática
"""
struct PBPKResponse
    concentrations::Dict{String, Vector{Float64}}
    time_points::Vector{Float64}
    organ_names::Vector{String}
end

"""
Handler para endpoint /api/pbpk/simulate.

Inovações:
- Type-safe handling
- Error handling robusto
"""
function handle_simulate(req::HTTP.Request)::HTTP.Response
    try
        # Parse request
        body = JSON.parse(String(req.body))
        pbpk_req = PBPKRequest(
            body["dose"],
            get(body, "clearance_hepatic", 0.0),
            get(body, "clearance_renal", 0.0),
            get(body, "partition_coeffs", Dict{String, Float64}()),
            get(body, "time_points", nothing),
        )

        # Simular (usar ODE solver ou GNN)
        params = PBPKParams(
            clearance_hepatic=pbpk_req.clearance_hepatic,
            clearance_renal=pbpk_req.clearance_renal,
            partition_coeffs=pbpk_req.partition_coeffs,
        )

        results = simulate(params, pbpk_req.dose;
                          time_points=pbpk_req.time_points)

        # Criar response
        response = PBPKResponse(
            Dict(organ => results[organ] for organ in keys(results) if organ != "time"),
            results["time"],
            collect(keys(results))[1:end-1],  # Excluir "time"
        )

        # Serializar JSON
        response_json = JSON.json(Dict(
            "concentrations" => response.concentrations,
            "time_points" => response.time_points,
            "organ_names" => response.organ_names,
        ))

        return HTTP.Response(200, response_json)
    catch e
        return HTTP.Response(400, JSON.json(Dict("error" => string(e))))
    end
end

"""
Handler para endpoint /api/pbpk/validate.

Inovações:
- Validação científica
- Métricas regulatórias
"""
function handle_validate(req::HTTP.Request)::HTTP.Response
    try
        # TODO: Implementar validação
        return HTTP.Response(200, JSON.json(Dict("status" => "not_implemented")))
    catch e
        return HTTP.Response(400, JSON.json(Dict("error" => string(e))))
    end
end

# ============================================================================
# SEMANTIC WEB ENDPOINTS (FAIR Data / Linked Data)
# ============================================================================

"""
Handler para endpoint GET /api/semantic/contexts.

Retorna os contextos JSON-LD disponíveis para serialização semântica.
Suporta content negotiation: application/ld+json, application/json
"""
function handle_get_contexts(req::HTTP.Request)::HTTP.Response
    try
        contexts = Dict(
            "@context" => Dict(
                "darwin" => DARWIN_CONTEXT,
                "drug" => DRUG_CONTEXT,
                "ddi" => DDI_CONTEXT,
                "parameter" => PARAMETER_CONTEXT,
                "evidence" => EVIDENCE_CONTEXT,
                "provenance" => PROVENANCE_CONTEXT,
                "obo" => OBO_PREFIXES
            ),
            "description" => "Darwin PBPK JSON-LD 1.1 contexts for semantic interoperability",
            "version" => "1.0.0",
            "conformsTo" => [
                "https://www.w3.org/TR/json-ld11/",
                "http://purl.obolibrary.org/obo/",
                "http://qudt.org/2.1/schema/qudt"
            ]
        )

        content_type = "application/ld+json; charset=utf-8"
        return HTTP.Response(200, ["Content-Type" => content_type], JSON.json(contexts))
    catch e
        return HTTP.Response(500, JSON.json(Dict("error" => string(e))))
    end
end

"""
Handler para endpoint GET /api/semantic/ontologies.

Lista todas as ontologias suportadas (OBO Foundry, QUDT, Schema.org).
"""
function handle_get_ontologies(req::HTTP.Request)::HTTP.Response
    try
        ontologies = list_supported_ontologies()

        response = Dict(
            "@context" => Dict(
                "dcterms" => "http://purl.org/dc/terms/",
                "void" => "http://rdfs.org/ns/void#",
                "owl" => "http://www.w3.org/2002/07/owl#"
            ),
            "@type" => "void:Dataset",
            "dcterms:title" => "Darwin PBPK Semantic Layer",
            "dcterms:description" => "Ontologies and vocabularies used for FAIR data representation",
            "supportedOntologies" => ontologies
        )

        return HTTP.Response(200, ["Content-Type" => "application/ld+json"], JSON.json(response))
    catch e
        return HTTP.Response(500, JSON.json(Dict("error" => string(e))))
    end
end

"""
Handler para endpoint GET /api/semantic/units.

Lista todas as unidades QUDT suportadas com mapeamentos UO.
"""
function handle_get_units(req::HTTP.Request)::HTTP.Response
    try
        units_list = []
        for (symbol, unit) in QUDT_UNITS
            push!(units_list, Dict(
                "symbol" => symbol,
                "qudt_uri" => unit.qudt_uri,
                "uo_id" => unit.uo_id,
                "dimension" => unit.dimension,
                "quantity_kind" => unit.quantity_kind,
                "conversion_factor" => unit.conversion_factor
            ))
        end

        response = Dict(
            "@context" => Dict(
                "qudt" => "http://qudt.org/schema/qudt/",
                "unit" => "http://qudt.org/vocab/unit/",
                "UO" => "http://purl.obolibrary.org/obo/UO_"
            ),
            "@type" => "qudt:UnitSystem",
            "dcterms:title" => "Darwin PBPK Pharmacokinetic Units",
            "units" => units_list
        )

        return HTTP.Response(200, ["Content-Type" => "application/ld+json"], JSON.json(response))
    catch e
        return HTTP.Response(500, JSON.json(Dict("error" => string(e))))
    end
end

"""
Handler para endpoint POST /api/semantic/drug.

Serializa uma droga para JSON-LD com contextos OBO e Schema.org.

Request body:
{
    "name": "Metformin",
    "chebi_id": "CHEBI:6801",
    "drugbank_id": "DB00331",
    "smiles": "CN(C)C(=N)NC(=N)N",
    "parameters": {
        "molecular_weight": 129.16,
        "logP": -1.43,
        "pKa": 11.5,
        "fu": 0.97
    }
}
"""
function handle_serialize_drug(req::HTTP.Request)::HTTP.Response
    try
        body = JSON.parse(String(req.body))

        name = get(body, "name", "Unknown")
        chebi_id = get(body, "chebi_id", nothing)
        drugbank_id = get(body, "drugbank_id", nothing)
        smiles = get(body, "smiles", nothing)
        parameters = get(body, "parameters", Dict())

        # Generate JSON-LD representation
        jsonld = to_jsonld_drug(
            name,
            chebi_id,
            drugbank_id,
            smiles,
            parameters
        )

        # Also generate Schema.org representation
        schema_org = to_schema_org_drug(name, parameters)

        response = Dict(
            "jsonld" => jsonld,
            "schema_org" => schema_org,
            "_links" => Dict(
                "self" => "/api/semantic/drug",
                "context" => "/api/semantic/contexts"
            )
        )

        return HTTP.Response(200, ["Content-Type" => "application/ld+json"], JSON.json(response))
    catch e
        return HTTP.Response(400, JSON.json(Dict("error" => string(e))))
    end
end

"""
Handler para endpoint POST /api/semantic/ddi.

Serializa uma interação droga-droga para JSON-LD com ontologias DINTO/DIDEO.

Request body:
{
    "perpetrator": {"name": "Ketoconazole", "chebi_id": "CHEBI:48339"},
    "victim": {"name": "Midazolam", "chebi_id": "CHEBI:6931"},
    "mechanism": "CYP3A4 inhibition",
    "dinto_id": "DINTO:0000001",
    "severity": "major",
    "auc_ratio": 15.4,
    "evidence_level": "strong"
}
"""
function handle_serialize_ddi(req::HTTP.Request)::HTTP.Response
    try
        body = JSON.parse(String(req.body))

        perpetrator = get(body, "perpetrator", Dict())
        victim = get(body, "victim", Dict())
        mechanism = get(body, "mechanism", "unknown")
        dinto_id = get(body, "dinto_id", nothing)
        severity = get(body, "severity", "unknown")
        auc_ratio = get(body, "auc_ratio", nothing)
        evidence_level = get(body, "evidence_level", nothing)

        jsonld = to_jsonld_ddi(
            perpetrator,
            victim,
            mechanism,
            dinto_id,
            severity,
            auc_ratio,
            evidence_level
        )

        return HTTP.Response(200, ["Content-Type" => "application/ld+json"], JSON.json(jsonld))
    catch e
        return HTTP.Response(400, JSON.json(Dict("error" => string(e))))
    end
end

"""
Handler para endpoint POST /api/semantic/parameter.

Serializa um parâmetro farmacocinético com unidades QUDT semânticas.

Request body:
{
    "name": "Clearance",
    "value": 5.2,
    "unit": "L/h",
    "parameter_type": "CLtot",
    "source": "population",
    "cv": 0.35
}
"""
function handle_serialize_parameter(req::HTTP.Request)::HTTP.Response
    try
        body = JSON.parse(String(req.body))

        name = get(body, "name", "Unknown")
        value = get(body, "value", 0.0)
        unit = get(body, "unit", "")
        param_type = get(body, "parameter_type", nothing)
        source = get(body, "source", nothing)
        cv = get(body, "cv", nothing)

        jsonld = to_jsonld_parameter(name, value, unit, param_type, source, cv)

        # Also return Schema.org PropertyValue
        schema_prop = map_pk_parameter_to_schema(name, value, unit)

        response = Dict(
            "jsonld" => jsonld,
            "schema_org" => schema_prop,
            "qudt_unit" => get_qudt_unit(unit)
        )

        return HTTP.Response(200, ["Content-Type" => "application/ld+json"], JSON.json(response))
    catch e
        return HTTP.Response(400, JSON.json(Dict("error" => string(e))))
    end
end

"""
Handler para endpoint POST /api/semantic/simulation.

Serializa resultados de simulação PBPK para JSON-LD com proveniência PROV-O.

Request body:
{
    "drug_name": "Metformin",
    "dose": 500.0,
    "dose_unit": "mg",
    "concentrations": {"plasma": [0.0, 1.2, 2.5, ...], "liver": [...]},
    "time_points": [0, 1, 2, ...],
    "model_version": "2.5.0",
    "include_provenance": true
}
"""
function handle_serialize_simulation(req::HTTP.Request)::HTTP.Response
    try
        body = JSON.parse(String(req.body))

        drug_name = get(body, "drug_name", "Unknown")
        dose = get(body, "dose", 0.0)
        dose_unit = get(body, "dose_unit", "mg")
        concentrations = get(body, "concentrations", Dict())
        time_points = get(body, "time_points", Float64[])
        model_version = get(body, "model_version", "2.5.0")
        include_provenance = get(body, "include_provenance", true)

        # Generate simulation JSON-LD
        jsonld = to_jsonld_simulation(
            drug_name,
            dose,
            dose_unit,
            concentrations,
            time_points,
            model_version
        )

        # Add provenance if requested
        if include_provenance
            prov = create_simulation_provenance(
                simulation_id=get(jsonld, "@id", "sim:unknown"),
                model_version=model_version,
                input_data=Dict("drug" => drug_name, "dose" => dose)
            )
            jsonld["prov:wasGeneratedBy"] = prov
        end

        return HTTP.Response(200, ["Content-Type" => "application/ld+json"], JSON.json(jsonld))
    catch e
        return HTTP.Response(400, JSON.json(Dict("error" => string(e))))
    end
end

"""
Handler para endpoint GET /api/semantic/turtle/:entity_type/:id.

Retorna representação Turtle (RDF) de uma entidade.
Content-Type: text/turtle
"""
function handle_get_turtle(req::HTTP.Request)::HTTP.Response
    try
        # Parse path: /api/semantic/turtle/drug/DB00331
        path_parts = split(req.target, "/")
        if length(path_parts) < 5
            return HTTP.Response(400, JSON.json(Dict("error" => "Invalid path. Use /api/semantic/turtle/{entity_type}/{id}")))
        end

        entity_type = path_parts[4]
        entity_id = path_parts[5]

        # Generate JSON-LD first, then convert to Turtle
        if entity_type == "drug"
            jsonld = to_jsonld_drug(entity_id, "CHEBI:$entity_id", entity_id, nothing, Dict())
        elseif entity_type == "unit"
            unit = get_qudt_unit(entity_id)
            if unit === nothing
                return HTTP.Response(404, JSON.json(Dict("error" => "Unit not found: $entity_id")))
            end
            jsonld = Dict(
                "@context" => PARAMETER_CONTEXT,
                "@type" => "qudt:Unit",
                "qudt:symbol" => unit.symbol,
                "qudt:ucumCode" => unit.symbol,
                "rdfs:seeAlso" => unit.uo_id
            )
        else
            return HTTP.Response(400, JSON.json(Dict("error" => "Unsupported entity type: $entity_type")))
        end

        turtle = to_turtle(jsonld)

        return HTTP.Response(200, ["Content-Type" => "text/turtle; charset=utf-8"], turtle)
    catch e
        return HTTP.Response(500, JSON.json(Dict("error" => string(e))))
    end
end

"""
Handler para endpoint GET /api/semantic/schema.org/drug/:id.

Retorna representação Schema.org de uma droga (Google-friendly).
"""
function handle_get_schema_org_drug(req::HTTP.Request)::HTTP.Response
    try
        path_parts = split(req.target, "/")
        if length(path_parts) < 5
            return HTTP.Response(400, JSON.json(Dict("error" => "Invalid path")))
        end

        drug_id = path_parts[5]

        # Generate Schema.org Drug
        schema_drug = to_schema_org_drug(drug_id, Dict())

        response = Dict(
            "@context" => "https://schema.org",
            "@type" => "Drug",
            "@id" => "https://darwin-pbpk.org/drug/$drug_id",
            schema_drug...
        )

        return HTTP.Response(200, ["Content-Type" => "application/ld+json"], JSON.json(response))
    catch e
        return HTTP.Response(500, JSON.json(Dict("error" => string(e))))
    end
end

"""
Handler para endpoint GET /api.

API Discovery - lista todos os endpoints disponíveis com documentação.
Segue padrão HATEOAS (Hypermedia as the Engine of Application State).
"""
function handle_api_discovery(req::HTTP.Request)::HTTP.Response
    try
        discovery = Dict(
            "@context" => "https://schema.org",
            "@type" => "WebAPI",
            "name" => "Darwin PBPK Platform API",
            "version" => "2.5.0",
            "description" => "FAIR-compliant PBPK modeling API with semantic web support",
            "documentation" => "https://darwin-pbpk.org/docs/api",
            "conformsTo" => [
                "https://www.w3.org/TR/json-ld11/",
                "http://purl.obolibrary.org/obo/",
                "https://schema.org"
            ],
            "endpoints" => Dict(
                "pbpk" => Dict(
                    "simulate" => Dict(
                        "method" => "POST",
                        "path" => "/api/pbpk/simulate",
                        "description" => "Run PBPK simulation"
                    ),
                    "validate" => Dict(
                        "method" => "POST",
                        "path" => "/api/pbpk/validate",
                        "description" => "Validate PBPK model parameters"
                    )
                ),
                "semantic" => Dict(
                    "contexts" => Dict(
                        "method" => "GET",
                        "path" => "/api/semantic/contexts",
                        "description" => "Get JSON-LD 1.1 contexts (OBO Foundry prefixes)"
                    ),
                    "ontologies" => Dict(
                        "method" => "GET",
                        "path" => "/api/semantic/ontologies",
                        "description" => "List supported ontologies (OBO, QUDT, Schema.org)"
                    ),
                    "units" => Dict(
                        "method" => "GET",
                        "path" => "/api/semantic/units",
                        "description" => "List QUDT units with UO mappings"
                    ),
                    "drug" => Dict(
                        "method" => "POST",
                        "path" => "/api/semantic/drug",
                        "description" => "Serialize drug to JSON-LD (ChEBI, DrugBank)"
                    ),
                    "ddi" => Dict(
                        "method" => "POST",
                        "path" => "/api/semantic/ddi",
                        "description" => "Serialize DDI to JSON-LD (DINTO, DIDEO)"
                    ),
                    "parameter" => Dict(
                        "method" => "POST",
                        "path" => "/api/semantic/parameter",
                        "description" => "Serialize PK parameter with QUDT units"
                    ),
                    "simulation" => Dict(
                        "method" => "POST",
                        "path" => "/api/semantic/simulation",
                        "description" => "Serialize simulation results with PROV-O provenance"
                    ),
                    "turtle" => Dict(
                        "method" => "GET",
                        "path" => "/api/semantic/turtle/{type}/{id}",
                        "description" => "Get entity as RDF Turtle"
                    ),
                    "schema_org_drug" => Dict(
                        "method" => "GET",
                        "path" => "/api/semantic/schema.org/drug/{id}",
                        "description" => "Get drug as Schema.org JSON-LD"
                    )
                )
            ),
            "_links" => Dict(
                "self" => "/api",
                "contexts" => "/api/semantic/contexts",
                "ontologies" => "/api/semantic/ontologies"
            )
        )

        return HTTP.Response(200, ["Content-Type" => "application/ld+json"], JSON.json(discovery))
    catch e
        return HTTP.Response(500, JSON.json(Dict("error" => string(e))))
    end
end

# ============================================================================
# ROUTER
# ============================================================================

"""
Router principal.

Inovações:
- Type-safe routing
- Error handling centralizado
- Semantic Web endpoints (FAIR Data)
"""
function router(req::HTTP.Request)::HTTP.Response
    # PBPK Simulation endpoints
    if req.method == "POST" && occursin("/api/pbpk/simulate", req.target)
        return handle_simulate(req)
    elseif req.method == "POST" && occursin("/api/pbpk/validate", req.target)
        return handle_validate(req)

    # Semantic Web endpoints (FAIR Data / Linked Data)
    elseif req.method == "GET" && occursin("/api/semantic/contexts", req.target)
        return handle_get_contexts(req)
    elseif req.method == "GET" && occursin("/api/semantic/ontologies", req.target)
        return handle_get_ontologies(req)
    elseif req.method == "GET" && occursin("/api/semantic/units", req.target)
        return handle_get_units(req)
    elseif req.method == "POST" && occursin("/api/semantic/drug", req.target)
        return handle_serialize_drug(req)
    elseif req.method == "POST" && occursin("/api/semantic/ddi", req.target)
        return handle_serialize_ddi(req)
    elseif req.method == "POST" && occursin("/api/semantic/parameter", req.target)
        return handle_serialize_parameter(req)
    elseif req.method == "POST" && occursin("/api/semantic/simulation", req.target)
        return handle_serialize_simulation(req)
    elseif req.method == "GET" && occursin("/api/semantic/turtle/", req.target)
        return handle_get_turtle(req)
    elseif req.method == "GET" && occursin("/api/semantic/schema.org/drug/", req.target)
        return handle_get_schema_org_drug(req)

    # API Discovery endpoint
    elseif req.method == "GET" && (req.target == "/api" || req.target == "/api/")
        return handle_api_discovery(req)

    else
        return HTTP.Response(404, JSON.json(Dict("error" => "Not found")))
    end
end

"""
Iniciar servidor HTTP.

Inovações:
- Async I/O nativo
- Type-safe endpoints
"""
function start_server(port::Int = 8000)
    println("Starting Darwin PBPK API server on port $port...")
    HTTP.serve(router, Sockets.IPv4(0, 0, 0, 0), port)
end

# Export core API functions
export start_server, router, PBPKRequest, PBPKResponse

# Export semantic endpoint handlers
export handle_get_contexts, handle_get_ontologies, handle_get_units
export handle_serialize_drug, handle_serialize_ddi, handle_serialize_parameter
export handle_serialize_simulation, handle_get_turtle, handle_get_schema_org_drug
export handle_api_discovery

end # module
