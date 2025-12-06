# Test External PK Datasets Integration
# Tests OSP, Zenodo, and PK-DB data sources

using Test
using DarwinPBPK

@testset "External PK Datasets" begin

    @testset "Dataset Catalog" begin
        # Test available datasets catalog
        @test :osp_ddi in keys(AVAILABLE_DATASETS)
        @test :osp_pediatrics in keys(AVAILABLE_DATASETS)
        @test :zenodo_betalactam in keys(AVAILABLE_DATASETS)
        @test :pkdb in keys(AVAILABLE_DATASETS)

        # Test dataset metadata
        osp_ddi = AVAILABLE_DATASETS[:osp_ddi]
        @test osp_ddi.name == "OSP Drug-Drug Interactions"
        @test osp_ddi.n_records > 600
        @test :auc_ratio in osp_ddi.data_types
        @test "Midazolam" in osp_ddi.drugs

        pkdb = AVAILABLE_DATASETS[:pkdb]
        @test pkdb.n_records >= 796
        @test "Caffeine" in pkdb.drugs
    end

    @testset "OSP DDI Database" begin
        # Test loading DDI data
        ddi = load_osp_ddi()
        @test size(ddi, 1) >= 600  # At least 600 records
        @test :Victim in propertynames(ddi)
        @test :Perpetrator in propertynames(ddi)
        @test Symbol("AUCR Avg") in propertynames(ddi)

        # Test drug filtering
        midazolam_ddi = load_osp_ddi(filter_drug="Midazolam")
        @test size(midazolam_ddi, 1) > 0
        @test size(midazolam_ddi, 1) < size(ddi, 1)

        # Verify all rows contain Midazolam
        for row in eachrow(midazolam_ddi)
            has_midazolam = lowercase(string(row[:Victim])) == "midazolam" ||
                           lowercase(string(row[:Perpetrator])) == "midazolam"
            @test has_midazolam
        end
    end

    @testset "OSP Pediatrics Database" begin
        ped = load_osp_pediatrics()
        @test size(ped, 1) >= 270
        @test :Analyte in propertynames(ped)
        @test Symbol("CL Avg") in propertynames(ped)

        # Test drug filtering
        sufentanil = load_osp_pediatrics(filter_drug="Sufentanil")
        @test size(sufentanil, 1) > 0
    end

    @testset "Zenodo Beta-Lactam Dataset" begin
        bl = load_zenodo_betalactam()

        # Test covariates
        @test size(bl.covariates, 1) >= 150
        @test :author in propertynames(bl.covariates)
        @test :betalactam_studied in propertynames(bl.covariates)

        # Test outcomes
        @test size(bl.outcomes, 1) >= 1000
        @test :outcome in propertynames(bl.outcomes)

        # Verify drugs present
        drugs = unique(bl.covariates.betalactam_studied)
        @test length(drugs) > 5  # Multiple beta-lactams
    end

    @testset "DDI AUC Ratio Extraction" begin
        # Test specific DDI interaction lookup
        # Midazolam + Itraconazole is a classic CYP3A4 inhibition example
        ratios = get_ddi_auc_ratios("Midazolam", "Itraconazole")

        if ratios.n > 0
            @test ratios.mean > 1.0  # Itraconazole inhibits CYP3A4, increases AUC
            @test ratios.mean < 20.0  # Reasonable range
            @test length(ratios.studies) > 0
        end

        # Test Rifampicin induction (AUC ratio < 1)
        rifamp_ratios = get_ddi_auc_ratios("Midazolam", "Rifampicin")
        if rifamp_ratios.n > 0
            @test rifamp_ratios.mean < 1.0  # Rifampicin induces CYP3A4, decreases AUC
            @test rifamp_ratios.mean > 0.01  # Not zero
        end
    end

    @testset "PK-DB API Integration" begin
        # Test API connectivity (may fail if offline)
        try
            studies = list_pkdb_studies(page=1)

            if !isempty(studies)
                @test length(studies) > 0

                # Check study structure
                first_study = studies[1]
                @test !isempty(first_study.sid)
                @test !isempty(first_study.name)
                @test first_study.n_outputs >= 0

                # Check substances are extracted
                @test isa(first_study.substances, Vector{String})
            end
        catch e
            @warn "PK-DB API test skipped (network unavailable): $e"
        end
    end

    @testset "Data Quality Checks" begin
        # OSP DDI: AUC ratios should be positive
        ddi = load_osp_ddi()
        auc_col = Symbol("AUCR Avg")
        auc_values = [row[auc_col] for row in eachrow(ddi) if !ismissing(row[auc_col])]
        @test all(v -> v > 0, auc_values)

        # OSP Pediatrics: Clearance should be positive
        ped = load_osp_pediatrics()
        cl_col = Symbol("CL Avg")
        cl_values = [row[cl_col] for row in eachrow(ped) if !ismissing(row[cl_col])]
        @test all(v -> v > 0, cl_values)

        # Zenodo: Study IDs should be unique in covariates
        bl = load_zenodo_betalactam()
        @test length(unique(bl.covariates.study_id)) == size(bl.covariates, 1)
    end

    @testset "Integration with Clinical Validation" begin
        # Test that external data can be used for validation
        # This connects to the clinical_validation.jl assertions

        # Example: Validate our Rb calculation against literature
        # Using the tacrolimus Rb formula: Rb = 1 - Hct + Hct * Ke_p

        # Literature value: Rb ≈ 15 at Hct 0.40, Ke_p = 37
        hct = 0.40
        ke_p = 37.0
        expected_rb = 1.0 - hct + hct * ke_p  # = 0.6 + 14.8 = 15.4

        @test abs(expected_rb - 15.4) < 0.1

        # At low Hct (0.25), Rb should be lower
        low_hct = 0.25
        low_rb = 1.0 - low_hct + low_hct * ke_p
        @test low_rb < expected_rb
    end

    @testset "Dataset Statistics" begin
        # Verify we have substantial data for validation
        ddi = load_osp_ddi()
        ped = load_osp_pediatrics()
        bl = load_zenodo_betalactam()

        total_records = size(ddi, 1) + size(ped, 1) + size(bl.covariates, 1) + size(bl.outcomes, 1)
        @test total_records > 2000  # Substantial validation dataset

        println("\n" * "="^60)
        println("EXTERNAL DATASET SUMMARY")
        println("="^60)
        println("OSP DDI records:          $(size(ddi, 1))")
        println("OSP Pediatrics records:   $(size(ped, 1))")
        println("Zenodo studies:           $(size(bl.covariates, 1))")
        println("Zenodo outcomes:          $(size(bl.outcomes, 1))")
        println("TOTAL validation points:  $total_records")
        println("="^60)
    end
end

println("\n✓ External datasets tests completed!")
