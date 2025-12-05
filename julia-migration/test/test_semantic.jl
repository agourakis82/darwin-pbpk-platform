# =============================================================================
# Semantic Layer Test Suite
# =============================================================================
# Tests for JSON-LD serialization and OBO Foundry ontology integration.
#
# Author: Dr. Demetrios Agourakis
# Date: December 2025
# =============================================================================

using Test
using JSON
using Dates

# Include semantic modules directly for testing
include("../src/DarwinPBPK/semantic/contexts.jl")
include("../src/DarwinPBPK/semantic/qudt_units.jl")
include("../src/DarwinPBPK/semantic/jsonld_serializer.jl")
include("../src/DarwinPBPK/semantic/provenance.jl")

using .Contexts
using .QUDTUnits
using .JSONLDSerializer
using .Provenance

@testset "Darwin PBPK Semantic Layer" begin

    # =========================================================================
    @testset "JSON-LD Contexts" begin

        @testset "DARWIN_CONTEXT structure" begin
            @test haskey(DARWIN_CONTEXT, "@context")
            ctx = DARWIN_CONTEXT["@context"]

            # Check JSON-LD version
            @test ctx["@version"] == 1.1

            # Check base URIs
            @test ctx["@vocab"] == "https://schema.org/"
            @test ctx["@base"] == "https://darwin-pbpk.org/"

            # Check namespace prefixes
            @test ctx["darwin"] == "https://darwin-pbpk.org/schema/"
            @test ctx["qudt"] == "http://qudt.org/schema/qudt/"
            @test ctx["obo"] == "http://purl.obolibrary.org/obo/"
            @test ctx["prov"] == "http://www.w3.org/ns/prov#"
        end

        @testset "OBO Foundry prefixes" begin
            ctx = DARWIN_CONTEXT["@context"]

            # Check OBO prefixes have @prefix: true for underscore handling
            @test ctx["CHEBI"]["@prefix"] == true
            @test ctx["DINTO"]["@prefix"] == true
            @test ctx["OBI"]["@prefix"] == true
            @test ctx["STATO"]["@prefix"] == true
            @test ctx["PATO"]["@prefix"] == true
            @test ctx["UO"]["@prefix"] == true

            # Check URI patterns
            @test ctx["CHEBI"]["@id"] == "http://purl.obolibrary.org/obo/CHEBI_"
            @test ctx["DINTO"]["@id"] == "http://purl.obolibrary.org/obo/DINTO_"
        end

        @testset "get_context function" begin
            @test get_context(:darwin) == DARWIN_CONTEXT
            @test get_context(:drug) == DRUG_CONTEXT
            @test get_context(:ddi) == DDI_CONTEXT
            @test get_context(:parameter) == PARAMETER_CONTEXT
            @test get_context(:evidence) == EVIDENCE_CONTEXT
            @test get_context(:provenance) == PROVENANCE_CONTEXT

            @test_throws ErrorException get_context(:invalid)
        end

        @testset "merge_contexts function" begin
            merged = merge_contexts([:drug, :ddi])
            @test haskey(merged, "@context")

            # Should contain elements from both contexts
            ctx = merged["@context"]
            @test haskey(ctx, "Drug")  # from DRUG_CONTEXT
            @test haskey(ctx, "DrugDrugInteraction")  # from DDI_CONTEXT
        end

        @testset "context_to_json function" begin
            json_str = context_to_json(DARWIN_CONTEXT)
            @test !isempty(json_str)

            # Should be valid JSON
            parsed = JSON.parse(json_str)
            @test haskey(parsed, "@context")
        end
    end

    # =========================================================================
    @testset "QUDT Units" begin

        @testset "QUDT_UNITS registry" begin
            # Check common units exist
            @test haskey(QUDT_UNITS, "mg")
            @test haskey(QUDT_UNITS, "L")
            @test haskey(QUDT_UNITS, "h")
            @test haskey(QUDT_UNITS, "mg/L")
            @test haskey(QUDT_UNITS, "L/h")
            @test haskey(QUDT_UNITS, "%")
        end

        @testset "QUDTUnit structure" begin
            mg_l = QUDT_UNITS["mg/L"]

            @test mg_l.qudt_uri == "unit:MilliGM-PER-L"
            @test mg_l.uo_id == "UO:0000274"
            @test mg_l.symbol == "mg/L"
            @test mg_l.dimension == "[M][L]^-3"
            @test mg_l.quantity_kind == "quantitykind:MassConcentration"
        end

        @testset "get_qudt_unit function" begin
            unit = get_qudt_unit("L/h")
            @test unit.qudt_uri == "unit:L-PER-HR"
            @test unit.uo_id == "UO:0000271"

            # Test error for unknown unit
            @test_throws ErrorException get_qudt_unit("invalid_unit")
        end

        @testset "dimension_compatible function" begin
            mg_l = QUDT_UNITS["mg/L"]
            ug_l = QUDT_UNITS["ug/L"]
            l_h = QUDT_UNITS["L/h"]

            @test dimension_compatible(mg_l, ug_l) == true  # Same dimension
            @test dimension_compatible(mg_l, l_h) == false  # Different dimensions
        end

        @testset "convert_units function" begin
            # Convert mg to g
            mg = QUDT_UNITS["mg"]
            g = QUDT_UNITS["g"]

            result = convert_units(1000.0, mg, g)
            @test isapprox(result, 1.0, atol=1e-10)
        end

        @testset "SemanticQuantity" begin
            unit = get_qudt_unit("mg/L")
            sq = SemanticQuantity(25.5, unit)

            @test sq.value == 25.5
            @test sq.unit.symbol == "mg/L"
            @test sq.iri === nothing
            @test sq.pato_quality === nothing
        end

        @testset "to_jsonld for SemanticQuantity" begin
            unit = get_qudt_unit("L/h")
            sq = SemanticQuantity(
                10.5,
                unit,
                "darwin:param/clearance_001",
                "PATO:0001025",  # rate
                "https://pubmed.ncbi.nlm.nih.gov/12345/",
                (distribution="LogNormal", cv=0.30)
            )

            jsonld = QUDTUnits.to_jsonld(sq)

            @test jsonld["numericValue"] == 10.5
            @test jsonld["unitRef"] == "unit:L-PER-HR"
            @test haskey(jsonld, "darwin:unitOBO")
            @test jsonld["darwin:unitOBO"]["@id"] == "UO:0000271"
            @test jsonld["is_about"]["@id"] == "PATO:0001025"
            @test jsonld["prov:wasDerivedFrom"]["@id"] == "https://pubmed.ncbi.nlm.nih.gov/12345/"
        end

        @testset "to_quantity_value function" begin
            qv = to_quantity_value(100.0, "mg"; pato="PATO:0000125")

            @test qv["numericValue"] == 100.0
            @test qv["unitRef"] == "unit:MilliGM"
            @test qv["is_about"]["@id"] == "PATO:0000125"
        end
    end

    # =========================================================================
    @testset "JSON-LD Serialization" begin

        @testset "to_jsonld_drug" begin
            drug_data = Dict(
                "name" => "Ketoconazole",
                "chebi_id" => "CHEBI:48339",
                "drugbank_id" => "DB01026",
                "rxnorm_cui" => "6135",
                "smiles" => "CC(C)(C)C1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",
                "molecular_weight" => 531.43,
                "atc_codes" => ["J02AB02", "D01AC08"],
                "mechanism_of_action" => "CYP3A4 inhibitor"
            )

            jsonld = to_jsonld_drug(drug_data)

            @test haskey(jsonld, "@context")
            @test "Drug" in jsonld["@type"]
            @test "obo:CHEBI_24431" in jsonld["@type"]
            @test jsonld["@id"] == "CHEBI:48339"
            @test jsonld["name"] == "Ketoconazole"
            @test haskey(jsonld, "identifier")
            @test haskey(jsonld, "darwin:molecularWeight")
            @test jsonld["darwin:molecularWeight"]["numericValue"] == 531.43
        end

        @testset "to_jsonld_ddi" begin
            ddi_data = Dict(
                "perpetrator" => Dict("name" => "Ketoconazole"),
                "victim" => Dict("name" => "Midazolam"),
                "mechanism" => :CYP_COMPETITIVE_INHIBITION,
                "auc_ratio" => 15.9,
                "cmax_ratio" => 3.5
            )

            jsonld = to_jsonld_ddi(ddi_data)

            @test haskey(jsonld, "@context")
            @test "darwin:DrugDrugInteraction" in jsonld["@type"]
            @test "obo:DINTO_0000001" in jsonld["@type"]
            @test jsonld["darwin:aucRatio"] == 15.9
            @test jsonld["darwin:cmaxRatio"] == 3.5
            @test jsonld["darwin:fdaClassification"] == "strong"
            @test haskey(jsonld, "darwin:mechanism")
        end

        @testset "to_jsonld_parameter" begin
            param_data = Dict(
                "name" => "Clearance",
                "symbol" => "CL",
                "value" => 25.5,
                "unit" => "L/h",
                "omega" => 0.30
            )

            jsonld = to_jsonld_parameter(param_data)

            @test haskey(jsonld, "@context")
            @test "darwin:PKParameter" in jsonld["@type"]
            @test "obo:IAO_0000027" in jsonld["@type"]
            @test jsonld["name"] == "Clearance"
            @test jsonld["darwin:symbol"] == "CL"
            @test haskey(jsonld, "darwin:value")
            @test jsonld["darwin:value"]["numericValue"] == 25.5
            @test haskey(jsonld, "darwin:interIndividualVariability")
        end

        @testset "to_jsonld_evidence" begin
            evidence_data = Dict(
                "type" => :clinical_pk_healthy,
                "level" => :strong,
                "study_design" => :crossover,
                "n_subjects" => 12,
                "pubmed_ids" => [7573094, 9169157]
            )

            jsonld = to_jsonld_evidence(evidence_data)

            @test haskey(jsonld, "@context")
            @test "darwin:Evidence" in jsonld["@type"]
            @test jsonld["darwin:evidenceLevel"] == "strong"
            @test jsonld["darwin:nSubjects"] == 12
            @test length(jsonld["darwin:pubmedId"]) == 2
            @test haskey(jsonld, "dcterms:references")
        end

        @testset "to_jsonld_simulation" begin
            sim_result = Dict(
                "time" => [0.0, 1.0, 2.0, 4.0, 8.0],
                "blood" => [0.0, 50.0, 35.0, 20.0, 10.0],
                "liver" => [0.0, 45.0, 40.0, 25.0, 12.0],
                "cmax" => 50.0,
                "auc" => 200.0
            )

            jsonld = to_jsonld_simulation(sim_result)

            @test "darwin:SimulationResult" in jsonld["@type"]
            @test "obo:OBI_0000070" in jsonld["@type"]
            @test haskey(jsonld, "prov:generatedAtTime")
            @test haskey(jsonld, "darwin:timePoints")
            @test haskey(jsonld, "darwin:concentrations")
            @test haskey(jsonld, "darwin:cmax")
            @test haskey(jsonld, "darwin:auc")
        end

        @testset "serialize_entity function" begin
            drug_data = Dict("name" => "TestDrug", "chebi_id" => "CHEBI:12345")

            jsonld_str = serialize_entity(:drug, drug_data; format=:jsonld)
            @test !isempty(jsonld_str)
            @test occursin("TestDrug", jsonld_str)

            turtle_str = serialize_entity(:drug, drug_data; format=:turtle)
            @test !isempty(turtle_str)
            @test occursin("@prefix", turtle_str)
        end

        @testset "FDA DDI Classification" begin
            # Test different severity levels
            @test JSONLDSerializer.classify_ddi_severity(1.0) == "no_effect"
            @test JSONLDSerializer.classify_ddi_severity(1.5) == "weak"
            @test JSONLDSerializer.classify_ddi_severity(3.0) == "moderate"
            @test JSONLDSerializer.classify_ddi_severity(10.0) == "strong"
            @test JSONLDSerializer.classify_ddi_severity(0.3) == "strong_induction"
            @test JSONLDSerializer.classify_ddi_severity(0.6) == "moderate_induction"
        end

        @testset "JSONLDDocument builder" begin
            doc = JSONLDDocument()

            add_entity(doc, Dict("@id" => "drug:1", "@type" => "Drug", "name" => "Drug1"))
            add_entity(doc, Dict("@id" => "drug:2", "@type" => "Drug", "name" => "Drug2"))

            result = finalize_document(doc)

            @test haskey(result, "@context")
            @test haskey(result, "@graph")
            @test length(result["@graph"]) == 2
        end
    end

    # =========================================================================
    @testset "Provenance (PROV-O)" begin

        @testset "ProvenanceAgent" begin
            agent = ProvenanceAgent(
                "darwin:agent/test",
                :software,
                "Test Agent",
                "1.0.0",
                "Test description"
            )

            @test agent.iri == "darwin:agent/test"
            @test agent.type == :software
            @test agent.name == "Test Agent"
            @test agent.version == "1.0.0"
        end

        @testset "Pre-defined agents" begin
            @test DARWIN_SOFTWARE_AGENT.name == "Darwin PBPK Platform"
            @test DARWIN_SOFTWARE_AGENT.type == :software
            @test DARWIN_PBPK_MODEL.name == "Darwin PBPK ODE Model"
        end

        @testset "create_prediction_provenance" begin
            prov = create_prediction_provenance(
                "darwin:prediction/test_001",
                :pbpk,
                Dict("dose" => 100.0, "clearance" => 10.0);
                derived_from = ["darwin:drug/ketoconazole"]
            )

            @test prov.entity_iri == "darwin:prediction/test_001"
            @test prov.entity_type == :prediction
            @test !isempty(prov.was_derived_from)
            @test !isempty(prov.was_attributed_to)
            @test prov.was_generated_by !== nothing
        end

        @testset "create_simulation_provenance" begin
            prov = create_simulation_provenance(
                "darwin:simulation/test_001",
                Dict("time_span" => (0.0, 24.0), "dose" => 500.0);
                drug_iri = "darwin:drug/midazolam"
            )

            @test prov.entity_type == :simulation
            @test "darwin:drug/midazolam" in prov.was_derived_from
        end

        @testset "create_ddi_provenance" begin
            prov = create_ddi_provenance(
                "darwin:ddi/ketoconazole_midazolam",
                "CHEBI:48339",
                "CHEBI:6931",
                :CYP_COMPETITIVE_INHIBITION;
                evidence_sources = ["https://pubmed.ncbi.nlm.nih.gov/7573094/"]
            )

            @test prov.entity_type == :prediction
            @test "CHEBI:48339" in prov.was_derived_from
            @test "CHEBI:6931" in prov.was_derived_from
        end

        @testset "add_derivation and add_attribution" begin
            prov = ProvenanceRecord("darwin:test/001", :prediction)

            add_derivation(prov, "darwin:source/001")
            @test "darwin:source/001" in prov.was_derived_from

            # Adding same derivation again should not duplicate
            add_derivation(prov, "darwin:source/001")
            @test count(x -> x == "darwin:source/001", prov.was_derived_from) == 1

            add_attribution(prov, DARWIN_SOFTWARE_AGENT)
            @test length(prov.was_attributed_to) == 1
        end

        @testset "to_jsonld for ProvenanceAgent" begin
            jsonld = Provenance.to_jsonld(DARWIN_SOFTWARE_AGENT)

            @test jsonld["@id"] == "darwin:agent/darwin_pbpk"
            @test "prov:SoftwareAgent" in jsonld["@type"]
            @test jsonld["rdfs:label"] == "Darwin PBPK Platform"
            @test jsonld["darwin:version"] == "2.5.0"
        end

        @testset "to_jsonld for ProvenanceRecord" begin
            prov = create_prediction_provenance(
                "darwin:prediction/test_002",
                :ddi,
                Dict("mechanism" => "CYP_INHIBITION")
            )

            jsonld = Provenance.to_jsonld(prov)

            @test "prov:Entity" in jsonld["@type"]
            @test jsonld["@id"] == "darwin:prediction/test_002"
            @test haskey(jsonld, "prov:generatedAtTime")
            @test haskey(jsonld, "prov:wasGeneratedBy")
            @test haskey(jsonld, "prov:wasAttributedTo")
        end

        @testset "to_jsonld_with_context" begin
            prov = ProvenanceRecord("darwin:test/001", :prediction)
            jsonld = Provenance.to_jsonld_with_context(prov)

            @test haskey(jsonld, "@context")
            @test jsonld["@context"]["prov"] == "http://www.w3.org/ns/prov#"
        end
    end

    # =========================================================================
    @testset "Integration Tests" begin

        @testset "Complete DDI workflow with provenance" begin
            # Create drug entities
            ketoconazole = to_jsonld_drug(Dict(
                "name" => "Ketoconazole",
                "chebi_id" => "CHEBI:48339",
                "drugbank_id" => "DB01026"
            ); include_context=false)

            midazolam = to_jsonld_drug(Dict(
                "name" => "Midazolam",
                "chebi_id" => "CHEBI:6931",
                "drugbank_id" => "DB00683"
            ); include_context=false)

            # Create DDI
            ddi = to_jsonld_ddi(Dict(
                "perpetrator" => ketoconazole,
                "victim" => midazolam,
                "mechanism" => :CYP_COMPETITIVE_INHIBITION,
                "auc_ratio" => 15.9,
                "evidence" => Dict(
                    "type" => :clinical_pk_healthy,
                    "level" => :strong,
                    "n_subjects" => 12,
                    "pubmed_ids" => [7573094]
                )
            ))

            # Add provenance
            prov = create_ddi_provenance(
                ddi["@id"],
                ketoconazole["@id"],
                midazolam["@id"],
                :CYP_COMPETITIVE_INHIBITION
            )

            ddi["prov:wasGeneratedBy"] = Provenance.to_jsonld(prov.was_generated_by)

            # Verify complete structure
            @test haskey(ddi, "@context")
            @test haskey(ddi, "darwin:perpetrator")
            @test haskey(ddi, "darwin:victim")
            @test haskey(ddi, "darwin:mechanism")
            @test haskey(ddi, "darwin:evidence")
            @test haskey(ddi, "prov:wasGeneratedBy")
        end

        @testset "Parameter serialization with uncertainty" begin
            clearance = to_jsonld_parameter(Dict(
                "name" => "Hepatic Clearance",
                "symbol" => "CLh",
                "value" => 25.5,
                "unit" => "L/h",
                "omega" => 0.35,
                "source" => "https://pubmed.ncbi.nlm.nih.gov/12345/"
            ))

            @test clearance["darwin:value"]["numericValue"] == 25.5
            @test clearance["darwin:value"]["unitRef"] == "unit:L-PER-HR"
            @test clearance["darwin:interIndividualVariability"]["darwin:omega"] == 0.35
        end

        @testset "RDF/Turtle export" begin
            drug = to_jsonld_drug(Dict(
                "name" => "TestDrug",
                "chebi_id" => "CHEBI:12345"
            ))

            turtle = to_turtle(drug)

            @test occursin("@prefix darwin:", turtle)
            @test occursin("@prefix obo:", turtle)
            @test occursin("TestDrug", turtle)
        end
    end

end  # main testset

println("\n" * "="^60)
println("Semantic Layer Tests Complete")
println("="^60)
