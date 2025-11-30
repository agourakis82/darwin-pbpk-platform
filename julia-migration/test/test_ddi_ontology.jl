# =============================================================================
# DDI ONTOLOGY INTEGRATION TESTS
# =============================================================================
# Darwin PBPK Platform - MedLang DSL
# Comprehensive testing of DDI Ontology mappings and clinical database
#
# Test Coverage:
# - DINTO mechanism mappings
# - ChEBI drug identifiers
# - DOID disease mappings
# - CYP enzyme ontology
# - Transporter ontology
# - Clinical DDI database
# - RDF export functionality
#
# Author: Dr. Demetrios Agourakis
# Date: November 2025
# =============================================================================

using Test
using DarwinPBPK
using DarwinPBPK.MedLang
using DarwinPBPK.MedLang.DDIOntology
using DarwinPBPK.MedLang.BayesianDDIModel

@testset "DDI Ontology Integration Tests" begin

    # =========================================================================
    # DINTO MECHANISM MAPPINGS
    # =========================================================================
    @testset "DINTO Mechanism Mappings" begin
        # Test CYP competitive inhibition
        mech = get_dinto_term(CYP_COMPETITIVE_INHIBITION)
        @test mech isa DDIMechanismOntology
        @test startswith(mech.dinto_id, "DINTO:")
        @test !isempty(mech.dinto_label)
        @test !isempty(mech.description)

        # Test CYP induction
        mech_ind = get_dinto_term(CYP_INDUCTION)
        @test mech_ind.mechanism_type == CYP_INDUCTION
        @test startswith(mech_ind.dinto_id, "DINTO:")

        # Test MBI
        mech_mbi = get_dinto_term(CYP_MECHANISM_BASED_INHIBITION)
        @test !isempty(mech_mbi.dideo_evidence_types)

        # Test transporter inhibition
        mech_trans = get_dinto_term(TRANSPORTER_INHIBITION)
        @test startswith(mech_trans.ndfrt_interaction_type, "NDF-RT:")

        # All mechanism types should have mappings
        for mech_type in [CYP_COMPETITIVE_INHIBITION, CYP_NONCOMPETITIVE_INHIBITION,
                         CYP_MECHANISM_BASED_INHIBITION, CYP_INDUCTION,
                         TRANSPORTER_INHIBITION, TRANSPORTER_INDUCTION,
                         PROTEIN_BINDING_DISPLACEMENT, RENAL_COMPETITION]
            @test haskey(DINTO_MECHANISMS, mech_type)
        end
    end

    # =========================================================================
    # ChEBI DRUG MAPPINGS
    # =========================================================================
    @testset "ChEBI Drug Mappings" begin
        # Test ketoconazole
        chebi_keto = get_chebi_id(:ketoconazole)
        @test chebi_keto == "CHEBI:48339"

        # Test DrugBank ID
        db_keto = get_drugbank_id(:ketoconazole)
        @test startswith(db_keto, "DB")

        # Test RxNorm CUI
        rxn_keto = get_rxnorm_cui(:ketoconazole)
        @test !isempty(rxn_keto)

        # Test all drugs have valid mappings
        for drug in [:ketoconazole, :itraconazole, :rifampin, :fluconazole,
                    :cyclosporine, :midazolam, :warfarin, :rosuvastatin,
                    :tacrolimus, :simvastatin]
            @test haskey(CHEBI_DRUGS, drug)
            mapping = CHEBI_DRUGS[drug]
            @test !isempty(mapping.chebi_id)
            @test !isempty(mapping.drugbank_id)
            @test !isempty(mapping.rxnorm_cui)
        end
    end

    @testset "Drug Ontology Mapping Structure" begin
        mapping = CHEBI_DRUGS[:ketoconazole]

        @test mapping isa DrugOntologyMapping
        @test mapping.drug_name == "Ketoconazole"
        @test !isempty(mapping.atc_codes)
        @test !isempty(mapping.chemical_class)
        @test !isempty(mapping.moa_terms)

        # MoA terms should have valid ontology terms
        moa = mapping.moa_terms[1]
        @test moa isa OntologyTerm
        @test moa.ontology == :ChEBI
        @test startswith(moa.uri, "http://")
    end

    # =========================================================================
    # CYP ENZYME ONTOLOGY
    # =========================================================================
    @testset "CYP Enzyme Ontology Mappings" begin
        # Test CYP3A4
        cyp3a4 = map_cyp_to_ontology(CYP3A4)
        @test cyp3a4 isa CYPOntologyMapping
        @test cyp3a4.gene_symbol == "CYP3A4"
        @test startswith(cyp3a4.uniprot_id, "P") || startswith(cyp3a4.uniprot_id, "Q")
        @test !isempty(cyp3a4.ncbi_gene_id)
        @test startswith(cyp3a4.hgnc_id, "HGNC:")
        @test !isempty(cyp3a4.go_terms)

        # All CYP enzymes should have mappings
        for enzyme in [CYP3A4, CYP3A5, CYP2D6, CYP2C9, CYP2C19, CYP1A2, CYP2C8, CYP2B6]
            @test haskey(CYP_ONTOLOGY, enzyme)
            ont = CYP_ONTOLOGY[enzyme]
            @test !isempty(ont.gene_symbol)
            @test !isempty(ont.uniprot_id)
        end
    end

    # =========================================================================
    # TRANSPORTER ONTOLOGY
    # =========================================================================
    @testset "Transporter Ontology Mappings" begin
        # Test P-gp
        pgp = map_transporter_to_ontology(PGP)
        @test pgp isa TransporterOntologyMapping
        @test pgp.gene_symbol == "ABCB1"
        @test pgp.slc_family == "ABC"
        @test !isempty(pgp.tissue_expression)

        # Test OATP1B1
        oatp = map_transporter_to_ontology(OATP1B1)
        @test oatp.gene_symbol == "SLCO1B1"
        @test oatp.slc_family == "SLC"
        @test "liver" in oatp.tissue_expression

        # All transporters should have mappings
        for trans in [PGP, BCRP, OATP1B1, OATP1B3, OCT1, OCT2, OAT1, OAT3, MATE1]
            @test haskey(TRANSPORTER_ONTOLOGY, trans)
        end
    end

    # =========================================================================
    # DISEASE ONTOLOGY (DOID)
    # =========================================================================
    @testset "Disease Ontology Mappings" begin
        # Test hepatic impairment
        hepatic = get_doid_term(:hepatic_impairment)
        @test hepatic isa DiseaseOntologyMapping
        @test startswith(hepatic.doid, "DOID:")
        @test !isempty(hepatic.icd10_codes)
        @test !isempty(hepatic.relevant_ddi_modifiers)

        # Test all disease contexts
        for disease in [:hepatic_impairment, :renal_impairment, :heart_failure,
                       :diabetes, :obesity, :inflammatory_disease, :cancer]
            @test haskey(DISEASE_DDI_CONTEXT, disease)
            mapping = DISEASE_DDI_CONTEXT[disease]
            @test !isempty(mapping.doid)
            @test !isempty(mapping.relevant_ddi_modifiers)
        end
    end

    # =========================================================================
    # CLINICAL DDI DATABASE
    # =========================================================================
    @testset "Clinical DDI Database" begin
        # Test ketoconazole-midazolam interaction
        evidence = get_literature_ddi_evidence(:ketoconazole, :midazolam)
        @test evidence !== nothing
        @test evidence isa ClinicalDDIEvidence
        @test evidence.auc_ratio_mean > 10.0  # Strong inhibition
        @test evidence.n_subjects > 0
        @test evidence.study_design == :crossover
        @test evidence.evidence_level == :strong
        @test !isempty(evidence.pubmed_ids)
        @test evidence.fda_label_info == true

        # Test rifampin-midazolam (induction)
        rif_mid = get_literature_ddi_evidence(:rifampin, :midazolam)
        @test rif_mid !== nothing
        @test rif_mid.auc_ratio_mean < 0.1  # Strong induction
        @test rif_mid.primary_mechanism == CYP_INDUCTION

        # Test cyclosporine-rosuvastatin (transporter)
        cyclo_rosu = get_literature_ddi_evidence(:cyclosporine, :rosuvastatin)
        @test cyclo_rosu !== nothing
        @test cyclo_rosu.primary_mechanism == TRANSPORTER_INHIBITION

        # Check database coverage
        @test length(DDI_CLINICAL_DATABASE) >= 10
    end

    @testset "DDI Database Queries" begin
        # List available drugs
        drugs = list_available_drugs()
        @test length(drugs) >= 10
        @test :ketoconazole in drugs
        @test :midazolam in drugs

        # List DDI pairs
        pairs = list_clinical_ddi_pairs()
        @test length(pairs) >= 10
        @test (:ketoconazole, :midazolam) in pairs

        # Get DDIs by mechanism
        cyp_inh_ddis = get_ddi_by_mechanism(CYP_COMPETITIVE_INHIBITION)
        @test length(cyp_inh_ddis) >= 5

        induction_ddis = get_ddi_by_mechanism(CYP_INDUCTION)
        @test length(induction_ddis) >= 1

        transporter_ddis = get_ddi_by_mechanism(TRANSPORTER_INHIBITION)
        @test length(transporter_ddis) >= 1
    end

    # =========================================================================
    # EVIDENCE TYPES (DIDEO)
    # =========================================================================
    @testset "Evidence Type Ontology" begin
        @test haskey(EVIDENCE_TYPES, :in_vitro_enzyme)
        @test haskey(EVIDENCE_TYPES, :clinical_pk_healthy)
        @test haskey(EVIDENCE_TYPES, :pbpk_modeling)

        # Check evidence type structure
        clinical_evidence = EVIDENCE_TYPES[:clinical_pk_healthy]
        @test clinical_evidence isa EvidenceTypeOntology
        @test startswith(clinical_evidence.dideo_id, "DIDEO:")
        @test clinical_evidence.strength == :strong
    end

    # =========================================================================
    # ONTOLOGY VALIDATION
    # =========================================================================
    @testset "Ontology Mapping Validation" begin
        # Validate complete mapping
        validation = validate_ontology_mapping(:ketoconazole)
        @test validation.has_chebi == true
        @test validation.has_drugbank == true
        @test validation.has_rxnorm == true
        @test validation.complete == true

        # Unknown drug should fail
        validation_unknown = validate_ontology_mapping(:unknown_drug)
        @test validation_unknown.has_chebi == false
        @test validation_unknown.complete == false
    end

    # =========================================================================
    # ONTOLOGY-ANNOTATED DDI CREATION
    # =========================================================================
    @testset "Ontology-Annotated DDI Creation" begin
        result = create_ontology_annotated_ddi(
            :ketoconazole, :midazolam, 15.0, CYP_COMPETITIVE_INHIBITION
        )

        @test haskey(result, "perpetrator")
        @test haskey(result, "victim")
        @test haskey(result, "mechanism")
        @test haskey(result, "interaction")
        @test haskey(result, "evidence")

        # Check perpetrator annotation
        perp = result["perpetrator"]
        @test perp["name"] == "Ketoconazole"
        @test startswith(perp["chebi_id"], "CHEBI:")
        @test startswith(perp["drugbank_id"], "DB")
        @test startswith(perp["uri"], "http://")

        # Check mechanism annotation
        mech = result["mechanism"]
        @test startswith(mech["dinto_id"], "DINTO:")
        @test !isempty(mech["label"])

        # Check interaction
        @test result["interaction"]["auc_ratio"] == 15.0
        @test result["interaction"]["fda_classification"] == "strong"

        # Check evidence
        @test result["evidence"]["observed_auc_ratio"] > 10.0
        @test !isempty(result["evidence"]["pubmed_ids"])
    end

    # =========================================================================
    # RDF EXPORT
    # =========================================================================
    @testset "RDF/Turtle Export" begin
        result = create_ontology_annotated_ddi(
            :ketoconazole, :midazolam, 15.0, CYP_COMPETITIVE_INHIBITION
        )

        rdf = export_to_rdf(result)

        @test !isempty(rdf)
        @test contains(rdf, "@prefix dinto:")
        @test contains(rdf, "@prefix chebi:")
        @test contains(rdf, "darwin:DDI_")
        @test contains(rdf, "darwin:hasPerpetrator")
        @test contains(rdf, "darwin:hasVictim")
        @test contains(rdf, "darwin:aucRatio")
    end

    # =========================================================================
    # CROSS-REFERENCE CONSISTENCY
    # =========================================================================
    @testset "Cross-Reference Consistency" begin
        # All clinical DDIs should have ontology mappings for both drugs
        for (perp, vic) in list_clinical_ddi_pairs()
            @test haskey(CHEBI_DRUGS, perp)
            @test haskey(CHEBI_DRUGS, vic)
        end

        # All mechanisms in clinical database should have DINTO mappings
        for evidence in values(DDI_CLINICAL_DATABASE)
            @test haskey(DINTO_MECHANISMS, evidence.primary_mechanism)
        end
    end

    # =========================================================================
    # ONTOLOGY URI FORMAT
    # =========================================================================
    @testset "Ontology URI Formats" begin
        # ChEBI URIs
        for (drug, mapping) in CHEBI_DRUGS
            chebi_num = split(mapping.chebi_id, ":")[2]
            expected_uri = "http://purl.obolibrary.org/obo/CHEBI_$chebi_num"
            @test mapping.moa_terms[1].uri == expected_uri ||
                  startswith(mapping.moa_terms[1].uri, "http://")
        end

        # DINTO URIs should be PURL-compliant
        for (mech_type, mech) in DINTO_MECHANISMS
            @test startswith(mech.dinto_id, "DINTO:")
        end
    end

    # =========================================================================
    # LITERATURE EVIDENCE QUALITY
    # =========================================================================
    @testset "Literature Evidence Quality" begin
        for evidence in values(DDI_CLINICAL_DATABASE)
            # All evidence should have valid AUC ratios
            @test evidence.auc_ratio_mean > 0
            @test evidence.auc_ratio_range[1] <= evidence.auc_ratio_mean
            @test evidence.auc_ratio_range[2] >= evidence.auc_ratio_mean

            # All should have study design
            @test evidence.study_design in [:crossover, :parallel, :retrospective]

            # All should have evidence level
            @test evidence.evidence_level in [:strong, :moderate, :weak]

            # All should have PubMed references
            @test !isempty(evidence.pubmed_ids) || evidence.fda_label_info
        end
    end

    # =========================================================================
    # SUMMARY
    # =========================================================================
    @testset "Test Summary" begin
        println("\n" * "="^60)
        println("DDI ONTOLOGY INTEGRATION TEST SUMMARY")
        println("="^60)
        println("Ontologies integrated:")
        println("  - DINTO: $(length(DINTO_MECHANISMS)) mechanism mappings")
        println("  - ChEBI: $(length(CHEBI_DRUGS)) drug mappings")
        println("  - DOID: $(length(DISEASE_DDI_CONTEXT)) disease contexts")
        println("  - CYP: $(length(CYP_ONTOLOGY)) enzyme mappings")
        println("  - Transporters: $(length(TRANSPORTER_ONTOLOGY)) transporter mappings")
        println("  - Clinical DDIs: $(length(DDI_CLINICAL_DATABASE)) evidence entries")
        println("="^60)
    end

end  # Main testset
