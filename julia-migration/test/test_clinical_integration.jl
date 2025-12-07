# ===========================================================================
# Tests for Clinical Data Integration Module
# ===========================================================================

using Test

include(joinpath(@__DIR__, "..", "src", "DarwinPBPK", "clinical", "ClinicalDataIntegration.jl"))
using .ClinicalDataIntegration

@testset "Clinical Data Integration" begin

    @testset "Subject struct" begin
        subj = Subject(id="SUBJ001", age=35.0, weight=70.0, height=175.0, sex=:male)
        @test subj.id == "SUBJ001"
        @test subj.age == 35.0
        @test subj.weight == 70.0
        @test subj.height == 175.0
        @test subj.sex == :male
        @test subj.bmi ≈ 70.0 / (1.75)^2 atol=0.1
    end

    @testset "PKObservation struct" begin
        obs = PKObservation("SUBJ001", 1.0, 150.0, "ng/mL", :plasma, "DrugX", false, 1.0)
        @test obs.subject_id == "SUBJ001"
        @test obs.time == 1.0
        @test obs.concentration == 150.0
        @test obs.sample_type == :plasma
        @test !obs.blq
    end

    @testset "DosingRecord struct" begin
        dose = DosingRecord(0.0, 100.0, :oral, "tablet", :fasted)
        @test dose.time == 0.0
        @test dose.amount == 100.0
        @test dose.route == :oral
        @test dose.formulation == "tablet"
    end

    @testset "NCA computation - basic" begin
        obs_vec = [
            PKObservation("SUBJ001", 0.0, 0.0, "ng/mL", :plasma, "DrugX", false, 1.0),
            PKObservation("SUBJ001", 0.5, 80.0, "ng/mL", :plasma, "DrugX", false, 1.0),
            PKObservation("SUBJ001", 1.0, 150.0, "ng/mL", :plasma, "DrugX", false, 1.0),
            PKObservation("SUBJ001", 2.0, 120.0, "ng/mL", :plasma, "DrugX", false, 1.0),
            PKObservation("SUBJ001", 4.0, 80.0, "ng/mL", :plasma, "DrugX", false, 1.0),
            PKObservation("SUBJ001", 8.0, 40.0, "ng/mL", :plasma, "DrugX", false, 1.0),
            PKObservation("SUBJ001", 12.0, 20.0, "ng/mL", :plasma, "DrugX", false, 1.0),
            PKObservation("SUBJ001", 24.0, 5.0, "ng/mL", :plasma, "DrugX", false, 1.0),
        ]

        nca = compute_nca(obs_vec, 100.0; route=:oral)

        @test nca.cmax == 150.0
        @test nca.tmax == 1.0
        @test nca.auc_0_t > 0
        @test nca.half_life > 0
        @test nca.cl_f > 0
        @test nca.dose == 100.0
        @test nca.route == :oral
    end

    @testset "NCA computation - IV" begin
        obs_vec = [
            PKObservation("SUBJ002", 0.083, 500.0, "ng/mL", :plasma, "DrugY", false, 1.0),
            PKObservation("SUBJ002", 0.25, 400.0, "ng/mL", :plasma, "DrugY", false, 1.0),
            PKObservation("SUBJ002", 0.5, 300.0, "ng/mL", :plasma, "DrugY", false, 1.0),
            PKObservation("SUBJ002", 1.0, 200.0, "ng/mL", :plasma, "DrugY", false, 1.0),
            PKObservation("SUBJ002", 2.0, 100.0, "ng/mL", :plasma, "DrugY", false, 1.0),
            PKObservation("SUBJ002", 4.0, 50.0, "ng/mL", :plasma, "DrugY", false, 1.0),
            PKObservation("SUBJ002", 8.0, 12.5, "ng/mL", :plasma, "DrugY", false, 1.0),
        ]

        nca = compute_nca(obs_vec, 50.0; route=:iv)

        @test nca.cmax == 500.0
        @test nca.route == :iv
        @test !isnan(nca.c0)  # C0 should be calculated for IV
    end

    @testset "AUC calculation" begin
        times = [0.0, 1.0, 2.0, 4.0]
        concs = [0.0, 100.0, 50.0, 25.0]

        auc = ClinicalDataIntegration.compute_auc_linear_log(times, concs)

        @test auc > 0
        # First interval: linear (0+100)/2 * 1 = 50
        # Approximate check
        @test auc > 100  # At least some area
    end

    @testset "Bioequivalence assessment" begin
        # Create test and reference NCA results
        test_nca = [
            NCAResult("S1", 100.0, 1.0, 500.0, 550.0, 9.0, 4.0, 0.17, 10.0, 60.0, 5.0, NaN, 100.0, :oral),
            NCAResult("S2", 110.0, 1.5, 520.0, 570.0, 8.5, 4.5, 0.15, 9.5, 65.0, 5.5, NaN, 100.0, :oral),
            NCAResult("S3", 95.0, 1.0, 480.0, 530.0, 9.5, 3.8, 0.18, 10.5, 58.0, 4.8, NaN, 100.0, :oral),
        ]

        ref_nca = [
            NCAResult("S1", 105.0, 1.2, 510.0, 560.0, 8.9, 4.1, 0.17, 10.2, 62.0, 5.1, NaN, 100.0, :oral),
            NCAResult("S2", 108.0, 1.3, 515.0, 565.0, 8.8, 4.3, 0.16, 9.8, 64.0, 5.3, NaN, 100.0, :oral),
            NCAResult("S3", 98.0, 1.1, 490.0, 540.0, 9.3, 3.9, 0.18, 10.3, 59.0, 4.9, NaN, 100.0, :oral),
        ]

        be_results = bioequivalence_assessment(test_nca, ref_nca)

        @test length(be_results) >= 1
        @test all(r -> r.geometric_mean_ratio > 0, be_results)
        @test all(r -> r.ci_90_lower > 0, be_results)
        @test all(r -> r.ci_90_upper > 0, be_results)
    end

    @testset "Validation metrics" begin
        # Mock validation
        predicted = Dict(
            "cmax" => [100.0, 110.0, 95.0],
            "auc" => [500.0, 520.0, 480.0]
        )

        observed = [
            NCAResult("S1", 105.0, 1.0, 510.0, 510.0, 0.0, 4.0, 0.17, 10.0, 60.0, 5.0, NaN, 100.0, :oral),
            NCAResult("S2", 108.0, 1.5, 515.0, 515.0, 0.0, 4.5, 0.15, 9.5, 65.0, 5.5, NaN, 100.0, :oral),
            NCAResult("S3", 98.0, 1.0, 490.0, 490.0, 0.0, 3.8, 0.18, 10.5, 58.0, 4.8, NaN, 100.0, :oral),
        ]

        metrics = validate_model_against_clinical(predicted, observed; drug_name="TestDrug")

        @test metrics.drug_name == "TestDrug"
        @test metrics.n_subjects == 3
        @test metrics.gmfe > 0
        @test metrics.gmfe < 2.0  # Should be close to 1 for similar values
        @test 0 <= metrics.within_2fold <= 100
    end

    @testset "Literature data entry" begin
        entry = literature_pk_entry(
            drug_name = "Metformin",
            parameter = "Cmax",
            value = 1500.0,
            unit = "ng/mL",
            pmid = "12345678",
            cv_percent = 25.0,
            n_subjects = 24,
            dose = 500.0
        )

        @test entry.drug_name == "Metformin"
        @test entry.parameter == "Cmax"
        @test entry.value == 1500.0
        @test entry.pmid == "12345678"
    end

    @testset "Literature data aggregation" begin
        entries = [
            literature_pk_entry(drug_name="DrugA", parameter="Cmax", value=100.0, unit="ng/mL"),
            literature_pk_entry(drug_name="DrugA", parameter="Cmax", value=110.0, unit="ng/mL"),
            literature_pk_entry(drug_name="DrugA", parameter="AUC", value=500.0, unit="ng*h/mL"),
            literature_pk_entry(drug_name="DrugB", parameter="Cmax", value=200.0, unit="ng/mL"),
        ]

        agg = aggregate_literature_data(entries)

        @test haskey(agg, "DrugA")
        @test haskey(agg["DrugA"], "Cmax")
        @test agg["DrugA"]["Cmax"].mean == 105.0
        @test agg["DrugA"]["Cmax"].n_studies == 2
    end

    @testset "Clinical PK Database" begin
        db = ClinicalPKDatabase()

        # Create a minimal study
        subj = Subject(id="S1", age=30.0, weight=70.0, height=175.0, sex=:male)
        obs = [PKObservation("S1", t, 100.0 * exp(-0.1*t), "ng/mL", :plasma, "DrugX", false, 1.0)
               for t in [0.5, 1.0, 2.0, 4.0, 8.0]]

        study = ClinicalStudy(
            "STUDY001", "DrugX", "Phase I", "single-dose",
            [subj], Dict{String, Vector{DosingRecord}}(), obs,
            Dict{String, Any}()
        )

        add_study!(db, study)

        @test haskey(db.studies, "STUDY001")
        @test length(query_studies(db, "DrugX")) == 1
    end

    @testset "ClinicalStudy struct" begin
        subj = Subject(id="S1", age=30.0, weight=70.0, height=175.0, sex=:male)
        dose = DosingRecord(0.0, 100.0, :oral, "tablet", :fasted)
        obs = PKObservation("S1", 1.0, 150.0, "ng/mL", :plasma, "DrugX", false, 1.0)

        study = ClinicalStudy(
            "STUDY001", "DrugX", "Phase I", "crossover",
            [subj], Dict("S1" => [dose]), [obs],
            Dict{String, Any}("sponsor" => "Test")
        )

        @test study.study_id == "STUDY001"
        @test study.drug_name == "DrugX"
        @test study.phase == "Phase I"
        @test length(study.subjects) == 1
        @test length(study.observations) == 1
    end

end

println("Clinical Data Integration tests completed!")
