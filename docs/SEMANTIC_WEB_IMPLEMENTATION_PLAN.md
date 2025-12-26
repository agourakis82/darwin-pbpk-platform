# JSON-LD and Ontologies Integration Plan
## Darwin PBPK Platform — SOTA Semantic Web Layer

**Version:** 1.0.0  
**Author:** Dr. Sounio Agourakis + Claude AI  
**Date:** December 2025  
**Status:** Planning Phase

---

## Executive Summary

This plan outlines the integration of JSON-LD serialization and OBO Foundry ontologies into the Darwin PBPK Platform, transforming it into a FAIR (Findable, Accessible, Interoperable, Reusable) pharmacometric knowledge platform.

**Current State:**
- Strong DDI ontology foundation (DINTO, ChEBI, DOID, GO mappings)
- Basic RDF export capability
- No JSON-LD contexts
- No QUDT unit semantics
- No semantic API endpoints

**Target State:**
- Full JSON-LD 1.1 serialization for all entities
- QUDT + UO dual unit system with dimensional analysis
- OBO Foundry alignment (BFO, ChEBI, DrOn, OBI, STATO)
- Semantic REST API with content negotiation
- SPARQL query interface
- PROV-O provenance tracking

---

## Phase 1: Semantic Foundation (Week 1-2)

### 1.1 Create Semantic Module Structure

**New Directory:** `julia-migration/src/DarwinPBPK/semantic/`

```
semantic/
├── SemanticCore.jl          # Main module, re-exports
├── contexts.jl              # JSON-LD context definitions
├── qudt_units.jl            # QUDT unit mappings + UO crosslinks
├── obo_terms.jl             # OBO Foundry term registry
├── jsonld_serializer.jl     # Entity → JSON-LD conversion
├── rdf_export.jl            # RDF/Turtle generation (enhanced)
├── provenance.jl            # PROV-O annotations
└── sparql_client.jl         # SPARQL query interface (Phase 3)
```

### 1.2 JSON-LD Context Definitions

**File:** `semantic/contexts.jl`

Define three core contexts:

1. **Drug Context** — ChEBI, DrOn, DrugBank, RxNorm mappings
2. **Parameter Context** — QUDT quantities, PATO qualities, IAO data items
3. **Evidence Context** — DIDEO, OBI, PROV-O

```julia
# Core context structure
const DARWIN_CONTEXT = Dict{String, Any}(
    "@context" => Dict{String, Any}(
        "@version" => 1.1,
        "@vocab" => "https://schema.org/",
        "@base" => "https://darwin-pbpk.org/",
        
        # Namespace prefixes
        "darwin" => "https://darwin-pbpk.org/schema/",
        "qudt" => "http://qudt.org/schema/qudt/",
        "unit" => "http://qudt.org/vocab/unit/",
        "obo" => "http://purl.obolibrary.org/obo/",
        
        # OBO prefixes with @prefix: true for underscore handling
        "CHEBI" => Dict("@id" => "http://purl.obolibrary.org/obo/CHEBI_", "@prefix" => true),
        "DINTO" => Dict("@id" => "http://purl.obolibrary.org/obo/DINTO_", "@prefix" => true),
        "OBI" => Dict("@id" => "http://purl.obolibrary.org/obo/OBI_", "@prefix" => true),
        "STATO" => Dict("@id" => "http://purl.obolibrary.org/obo/STATO_", "@prefix" => true),
        "UO" => Dict("@id" => "http://purl.obolibrary.org/obo/UO_", "@prefix" => true),
        "PATO" => Dict("@id" => "http://purl.obolibrary.org/obo/PATO_", "@prefix" => true),
        "IAO" => Dict("@id" => "http://purl.obolibrary.org/obo/IAO_", "@prefix" => true),
        "RO" => Dict("@id" => "http://purl.obolibrary.org/obo/RO_", "@prefix" => true),
        
        # Standard relations
        "has_role" => Dict("@id" => "obo:RO_0000087", "@type" => "@id"),
        "has_quality" => Dict("@id" => "obo:RO_0000086", "@type" => "@id"),
        "is_about" => Dict("@id" => "obo:IAO_0000136", "@type" => "@id"),
        
        # QUDT properties
        "numericValue" => "qudt:numericValue",
        "unit" => Dict("@id" => "qudt:unit", "@type" => "@id"),
    )
)
```

### 1.3 QUDT Unit Mappings

**File:** `semantic/qudt_units.jl`

Map MedLang/Julia units to QUDT URIs with UO crosslinks:

```julia
struct QUDTUnit
    qudt_uri::String      # e.g., "unit:MilliGM-PER-L"
    uo_id::String         # e.g., "UO:0000274"
    symbol::String        # e.g., "mg/L"
    dimension::String     # e.g., "[M][L]^-3"
    conversion_factor::Float64  # To SI base
end

const QUDT_UNITS = Dict{String, QUDTUnit}(
    # Mass
    "mg" => QUDTUnit("unit:MilliGM", "UO:0000022", "mg", "[M]", 1e-6),
    "g" => QUDTUnit("unit:GM", "UO:0000021", "g", "[M]", 1e-3),
    "kg" => QUDTUnit("unit:KiloGM", "UO:0000009", "kg", "[M]", 1.0),
    
    # Volume
    "L" => QUDTUnit("unit:L", "UO:0000099", "L", "[L]^3", 1e-3),
    "mL" => QUDTUnit("unit:MilliL", "UO:0000098", "mL", "[L]^3", 1e-6),
    
    # Time
    "h" => QUDTUnit("unit:HR", "UO:0000032", "h", "[T]", 3600.0),
    "min" => QUDTUnit("unit:MIN", "UO:0000031", "min", "[T]", 60.0),
    "s" => QUDTUnit("unit:SEC", "UO:0000010", "s", "[T]", 1.0),
    
    # Derived PK units
    "mg/L" => QUDTUnit("unit:MilliGM-PER-L", "UO:0000274", "mg/L", "[M][L]^-3", 1.0),
    "L/h" => QUDTUnit("unit:L-PER-HR", "UO:0000271", "L/h", "[L]^3[T]^-1", 2.78e-7),
    "L/h/kg" => QUDTUnit("unit:L-PER-HR-KiloGM", "", "L/h/kg", "[L]^3[T]^-1[M]^-1", 2.78e-7),
    
    # Dimensionless
    "%" => QUDTUnit("unit:PERCENT", "UO:0000187", "%", "1", 0.01),
    "fraction" => QUDTUnit("unit:UNITLESS", "UO:0000186", "", "1", 1.0),
)
```

### 1.4 Semantic Parameter Wrapper

**File:** `semantic/jsonld_serializer.jl`

Wrap existing `PBPKParams` with semantic annotations:

```julia
struct SemanticQuantity
    value::Float64
    unit::QUDTUnit
    iri::String                      # Parameter IRI
    pato_quality::Union{String,Nothing}  # e.g., "PATO:0001025" for rate
    source::Union{String,Nothing}    # Literature/provenance URI
    uncertainty::Union{NamedTuple,Nothing}  # (distribution, params)
end

function to_jsonld(sq::SemanticQuantity)::Dict{String,Any}
    result = Dict{String,Any}(
        "@type" => ["qudt:QuantityValue", "obo:IAO_0000032"],
        "numericValue" => sq.value,
        "unit" => sq.unit.qudt_uri,
    )
    
    # Add UO crosslink if available
    if !isempty(sq.unit.uo_id)
        result["darwin:unitOBO"] = Dict("@id" => sq.unit.uo_id)
    end
    
    # Add provenance
    if sq.source !== nothing
        result["prov:wasDerivedFrom"] = Dict("@id" => sq.source)
    end
    
    return result
end
```

---

## Phase 2: Enhanced DDI Ontology (Week 2-3)

### 2.1 Extend ddi_ontology.jl with JSON-LD Export

Add to existing `DDIOntology` module:

```julia
# New exports
export to_jsonld, generate_context, SemanticDDIResult

"""
Generate JSON-LD representation of a DDI result.
"""
function to_jsonld(
    perpetrator::Symbol,
    victim::Symbol,
    auc_ratio::Float64,
    mechanism::DDIMechanismType;
    include_context::Bool = true
)::Dict{String,Any}
    
    result = Dict{String,Any}()
    
    if include_context
        result["@context"] = DARWIN_DDI_CONTEXT
    end
    
    result["@type"] = ["darwin:DrugDrugInteraction", "obo:DINTO_0000001"]
    result["@id"] = "darwin:ddi/$(perpetrator)_$(victim)"
    
    # Perpetrator with full semantic annotations
    if haskey(CHEBI_DRUGS, perpetrator)
        perp = CHEBI_DRUGS[perpetrator]
        result["darwin:perpetrator"] = Dict{String,Any}(
            "@type" => ["Drug", "obo:CHEBI_24431"],
            "@id" => perp.chebi_id,
            "name" => perp.drug_name,
            "identifier" => [
                Dict("@type" => "PropertyValue", "propertyID" => "ChEBI", "value" => perp.chebi_id),
                Dict("@type" => "PropertyValue", "propertyID" => "DrugBank", "value" => perp.drugbank_id),
                Dict("@type" => "PropertyValue", "propertyID" => "RxNorm", "value" => perp.rxnorm_cui),
            ]
        )
    end
    
    # ... victim, mechanism, evidence similarly
    
    return result
end
```

### 2.2 Add DrOn (Drug Ontology) Mappings

Extend drug representation with DrOn product classifications:

```julia
struct DrugProductMapping
    dron_id::String           # e.g., "DRON:00018411"
    product_name::String      # e.g., "Metformin 500 MG Oral Tablet"
    active_ingredient::String # ChEBI reference
    dose_value::Float64
    dose_unit::String
    route::String             # oral, iv, etc.
end

const DRON_PRODUCTS = Dict{Symbol, Vector{DrugProductMapping}}(
    :metformin => [
        DrugProductMapping("DRON:00018411", "Metformin 500 MG Oral Tablet", "CHEBI:6801", 500.0, "mg", "oral"),
        DrugProductMapping("DRON:00019843", "Metformin 1000 MG Oral Tablet", "CHEBI:6801", 1000.0, "mg", "oral"),
    ],
    # ... extend for other drugs
)
```

### 2.3 Add OBI Study Design Annotations

For clinical evidence:

```julia
struct OBIStudyAnnotation
    obi_id::String            # e.g., "OBI:0000471" (study design execution)
    study_type::Symbol        # :crossover, :parallel, :retrospective
    population_type::String   # e.g., "OBI:0000181" (population)
    has_specified_input::Vector{String}   # Drug IRIs
    has_specified_output::Vector{String}  # Measurement IRIs
end
```

---

## Phase 3: Semantic REST API (Week 3-4)

### 3.1 Extend rest_api.jl with Semantic Endpoints

```julia
# New semantic endpoints
const SEMANTIC_ROUTES = [
    ("GET", "/api/v1/ontology/drug/:id", handle_drug_ontology),
    ("GET", "/api/v1/ontology/ddi/:perpetrator/:victim", handle_ddi_ontology),
    ("GET", "/api/v1/ontology/mechanism/:type", handle_mechanism_ontology),
    ("POST", "/api/v1/export/rdf", handle_rdf_export),
    ("POST", "/api/v1/export/jsonld", handle_jsonld_export),
]

"""
Handle drug ontology lookup with content negotiation.
"""
function handle_drug_ontology(req::HTTP.Request)::HTTP.Response
    # Parse Accept header for content negotiation
    accept = get_header(req, "Accept", "application/json")
    
    drug_id = parse_path_param(req, "id")
    drug_sym = Symbol(drug_id)
    
    if !haskey(CHEBI_DRUGS, drug_sym)
        return HTTP.Response(404, json_error("Drug not found: $drug_id"))
    end
    
    drug = CHEBI_DRUGS[drug_sym]
    
    if occursin("application/ld+json", accept)
        # Return JSON-LD
        jsonld = to_jsonld_drug(drug)
        return HTTP.Response(200, 
            ["Content-Type" => "application/ld+json"],
            JSON.json(jsonld))
    elseif occursin("text/turtle", accept)
        # Return RDF/Turtle
        turtle = to_turtle_drug(drug)
        return HTTP.Response(200,
            ["Content-Type" => "text/turtle"],
            turtle)
    else
        # Default JSON
        return HTTP.Response(200, JSON.json(drug_to_dict(drug)))
    end
end
```

### 3.2 Content Negotiation Middleware

```julia
"""
Middleware for semantic content negotiation.
Supports: application/json, application/ld+json, text/turtle, application/rdf+xml
"""
function content_negotiation_middleware(handler)
    return function(req::HTTP.Request)
        accept = get_header(req, "Accept", "application/json")
        
        # Store preferred format in request context
        req.context[:preferred_format] = parse_accept_header(accept)
        
        return handler(req)
    end
end

function parse_accept_header(accept::String)::Symbol
    if occursin("application/ld+json", accept)
        return :jsonld
    elseif occursin("text/turtle", accept)
        return :turtle
    elseif occursin("application/rdf+xml", accept)
        return :rdfxml
    else
        return :json
    end
end
```

---

## Phase 4: PROV-O Provenance (Week 4)

### 4.1 Provenance Annotations

**File:** `semantic/provenance.jl`

```julia
using Dates

struct ProvenanceRecord
    entity_iri::String
    generated_at::DateTime
    was_derived_from::Vector{String}      # Source data IRIs
    was_attributed_to::Vector{String}     # Agent IRIs (authors, algorithms)
    was_generated_by::String              # Activity IRI
    used_method::String                   # Algorithm/model IRI
end

function to_jsonld_provenance(prov::ProvenanceRecord)::Dict{String,Any}
    return Dict{String,Any}(
        "@type" => "prov:Entity",
        "@id" => prov.entity_iri,
        "prov:generatedAtTime" => Dict(
            "@type" => "xsd:dateTime",
            "@value" => Dates.format(prov.generated_at, "yyyy-mm-ddTHH:MM:SSZ")
        ),
        "prov:wasDerivedFrom" => [Dict("@id" => uri) for uri in prov.was_derived_from],
        "prov:wasAttributedTo" => [Dict("@id" => uri) for uri in prov.was_attributed_to],
        "prov:wasGeneratedBy" => Dict(
            "@type" => "prov:Activity",
            "@id" => prov.was_generated_by,
            "prov:used" => Dict("@id" => prov.used_method)
        )
    )
end
```

### 4.2 Annotate DDI Predictions with Provenance

```julia
function create_annotated_ddi_prediction(
    perpetrator::Symbol,
    victim::Symbol,
    prediction::DDIPredictionResult
)::Dict{String,Any}
    
    result = to_jsonld(perpetrator, victim, prediction.auc_ratio, prediction.mechanism)
    
    # Add provenance
    result["prov:wasGeneratedBy"] = Dict{String,Any}(
        "@type" => "prov:Activity",
        "@id" => "darwin:activity/ddi_prediction_$(now())",
        "prov:used" => [
            Dict("@id" => "darwin:model/bayesian_ddi_v1.0"),
            Dict("@id" => "darwin:database/ddi_knowledge_base_v1.0"),
        ],
        "prov:wasAssociatedWith" => Dict(
            "@type" => "prov:SoftwareAgent",
            "@id" => "darwin:agent/darwin_pbpk_v2.5.0",
            "rdfs:label" => "Darwin PBPK Platform v2.5.0"
        )
    )
    
    # Add confidence/uncertainty as STATO
    if haskey(prediction, :credible_interval)
        result["darwin:uncertainty"] = Dict{String,Any}(
            "@type" => "obo:STATO_0000039",  # variance
            "darwin:credibleInterval" => Dict(
                "darwin:lower" => prediction.credible_interval[1],
                "darwin:upper" => prediction.credible_interval[2],
                "darwin:level" => 0.95
            )
        )
    end
    
    return result
end
```

---

## Phase 5: SPARQL Interface (Week 5-6)

### 5.1 Triple Store Integration

**File:** `semantic/sparql_client.jl`

```julia
using HTTP

struct SPARQLEndpoint
    url::String
    update_url::Union{String,Nothing}
    auth::Union{Tuple{String,String},Nothing}  # (username, password)
end

# Default to Apache Jena Fuseki
const DEFAULT_ENDPOINT = SPARQLEndpoint(
    "http://localhost:3030/darwin/sparql",
    "http://localhost:3030/darwin/update",
    nothing
)

"""
Execute SPARQL SELECT query.
"""
function sparql_select(
    query::String;
    endpoint::SPARQLEndpoint = DEFAULT_ENDPOINT
)::Vector{Dict{String,Any}}
    
    headers = ["Accept" => "application/sparql-results+json"]
    
    if endpoint.auth !== nothing
        auth_str = Base64.base64encode("$(endpoint.auth[1]):$(endpoint.auth[2])")
        push!(headers, "Authorization" => "Basic $auth_str")
    end
    
    response = HTTP.request(
        "POST",
        endpoint.url,
        headers,
        "query=" * HTTP.escapeuri(query)
    )
    
    result = JSON.parse(String(response.body))
    return result["results"]["bindings"]
end

"""
Insert RDF triples into the store.
"""
function sparql_insert(
    turtle::String;
    graph::String = "darwin:default",
    endpoint::SPARQLEndpoint = DEFAULT_ENDPOINT
)::Bool
    
    update_query = """
    INSERT DATA {
        GRAPH <$graph> {
            $turtle
        }
    }
    """
    
    # Execute update
    # ...
end
```

### 5.2 Semantic Query Helpers

```julia
"""
Find all DDIs for a given drug.
"""
function query_ddis_for_drug(drug_chebi::String)::Vector{Dict}
    query = """
    PREFIX darwin: <https://darwin-pbpk.org/schema/>
    PREFIX chebi: <http://purl.obolibrary.org/obo/CHEBI_>
    PREFIX dinto: <http://purl.obolibrary.org/obo/DINTO_>
    
    SELECT ?ddi ?victim ?mechanism ?auc_ratio
    WHERE {
        ?ddi a darwin:DrugDrugInteraction ;
             darwin:perpetrator <$drug_chebi> ;
             darwin:victim ?victim ;
             darwin:mechanism ?mechanism ;
             darwin:aucRatio ?auc_ratio .
    }
    """
    
    return sparql_select(query)
end

"""
Find drugs by mechanism of action.
"""
function query_drugs_by_mechanism(mechanism_go::String)::Vector{Dict}
    query = """
    PREFIX darwin: <https://darwin-pbpk.org/schema/>
    PREFIX go: <http://purl.obolibrary.org/obo/GO_>
    
    SELECT ?drug ?name ?chebi
    WHERE {
        ?drug a darwin:Drug ;
              rdfs:label ?name ;
              darwin:chebiId ?chebi ;
              darwin:mechanismOfAction <$mechanism_go> .
    }
    """
    
    return sparql_select(query)
end
```

---

## Phase 6: Integration & Testing (Week 6-7)

### 6.1 Update Main Module

**File:** `julia-migration/src/DarwinPBPK.jl`

Add semantic module to main exports:

```julia
# Semantic Web Layer
include("DarwinPBPK/semantic/SemanticCore.jl")

using .SemanticCore

# Re-export semantic functions
export to_jsonld, to_turtle, sparql_select
export SemanticQuantity, QUDTUnit, ProvenanceRecord
export DARWIN_CONTEXT, QUDT_UNITS
```

### 6.2 Test Suite

**File:** `julia-migration/test/test_semantic.jl`

```julia
using Test
using DarwinPBPK.SemanticCore
using DarwinPBPK.DDIOntology
using JSON

@testset "JSON-LD Serialization" begin
    @testset "Drug to JSON-LD" begin
        jsonld = to_jsonld_drug(:ketoconazole)
        
        @test haskey(jsonld, "@context")
        @test haskey(jsonld, "@type")
        @test "obo:CHEBI_24431" in jsonld["@type"]
        @test jsonld["@id"] == "CHEBI:48339"
    end
    
    @testset "DDI to JSON-LD" begin
        jsonld = to_jsonld(:ketoconazole, :midazolam, 15.9, CYP_COMPETITIVE_INHIBITION)
        
        @test haskey(jsonld, "@context")
        @test jsonld["@type"] == ["darwin:DrugDrugInteraction", "obo:DINTO_0000001"]
        @test haskey(jsonld, "darwin:perpetrator")
        @test haskey(jsonld, "darwin:mechanism")
    end
    
    @testset "QUDT Unit Mapping" begin
        unit = QUDT_UNITS["mg/L"]
        
        @test unit.qudt_uri == "unit:MilliGM-PER-L"
        @test unit.uo_id == "UO:0000274"
        @test unit.dimension == "[M][L]^-3"
    end
end

@testset "RDF Export" begin
    @testset "Turtle Generation" begin
        turtle = export_to_rdf(create_ontology_annotated_ddi(
            :ketoconazole, :midazolam, 15.9, CYP_COMPETITIVE_INHIBITION
        ))
        
        @test occursin("@prefix dinto:", turtle)
        @test occursin("@prefix chebi:", turtle)
        @test occursin("darwin:hasPerpetrator", turtle)
    end
end
```

---

## File Modification Summary

### New Files

| File | Purpose | Priority |
|------|---------|----------|
| `semantic/SemanticCore.jl` | Main semantic module | P1 |
| `semantic/contexts.jl` | JSON-LD contexts | P1 |
| `semantic/qudt_units.jl` | QUDT/UO mappings | P1 |
| `semantic/obo_terms.jl` | OBO term registry | P2 |
| `semantic/jsonld_serializer.jl` | Entity serialization | P1 |
| `semantic/provenance.jl` | PROV-O annotations | P2 |
| `semantic/sparql_client.jl` | SPARQL interface | P3 |
| `test/test_semantic.jl` | Semantic tests | P1 |

### Modified Files

| File | Changes | Priority |
|------|---------|----------|
| `DarwinPBPK.jl` | Add semantic module | P1 |
| `medlang/ddi_ontology.jl` | Add `to_jsonld()`, enhance RDF | P1 |
| `api/rest_api.jl` | Add semantic endpoints | P2 |
| `medlang/parser.jl` | Integrate QUDT validation | P3 |

---

## Dependencies

### Julia Packages (add to Project.toml)

```toml
[deps]
# Existing...

# New for semantic layer
URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
# JSON already present
# HTTP already present
```

### External Services (Optional)

- **Apache Jena Fuseki** — Local triple store for SPARQL (Docker: `stain/jena-fuseki`)
- **GraphDB** — Enterprise triple store (alternative)

---

## Validation Criteria

### Phase 1 Complete When:
- [ ] JSON-LD context can be generated for drugs
- [ ] QUDT unit mappings cover all PK units
- [ ] `SemanticQuantity` wraps `PBPKParams` fields

### Phase 2 Complete When:
- [ ] `to_jsonld()` works for all DDI results
- [ ] DrOn product mappings exist for 10+ drugs
- [ ] OBI annotations added to clinical evidence

### Phase 3 Complete When:
- [ ] `/api/v1/ontology/*` endpoints functional
- [ ] Content negotiation returns JSON-LD/Turtle
- [ ] API tests pass with semantic responses

### Phase 4 Complete When:
- [ ] All predictions include PROV-O annotations
- [ ] Provenance chain is complete (data → model → result)
- [ ] STATO uncertainty annotations present

### Phase 5 Complete When:
- [ ] SPARQL SELECT queries work
- [ ] Triple store integration tested
- [ ] Semantic query helpers functional

### Phase 6 Complete When:
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Example notebooks demonstrate semantic features

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| OBO ontology version drift | Pin specific OBO releases, validate with ROBOT |
| QUDT coverage gaps | Fallback to custom darwin: units with skos:closeMatch |
| Triple store complexity | Start with file-based RDF, add SPARQL later |
| Performance overhead | Lazy serialization, cache JSON-LD contexts |

---

## References

1. JSON-LD 1.1 Specification: https://www.w3.org/TR/json-ld11/
2. QUDT Ontologies: http://qudt.org/
3. OBO Foundry: http://obofoundry.org/
4. PROV-O: https://www.w3.org/TR/prov-o/
5. Schema.org: https://schema.org/
6. BioSchemas: https://bioschemas.org/

---

*Plan generated for Darwin PBPK Platform v2.5.0 semantic web integration*
