# ===========================================================================
# Tests for Regulatory Reports Module
# ===========================================================================

using Test
using Dates

include(joinpath(@__DIR__, "..", "src", "DarwinPBPK", "regulatory", "RegulatoryReports.jl"))
using .RegulatoryReports

@testset "Regulatory Reports" begin

    @testset "ModelDocumentation struct" begin
        doc = ModelDocumentation(
            "TestModel", "1.0.0", "Dr. Test", today(),
            "Darwin PBPK", "2.11.0", "Metformin", "Type 2 Diabetes",
            "TestPharma", :fda
        )

        @test doc.model_name == "TestModel"
        @test doc.drug_name == "Metformin"
        @test doc.regulatory_agency == :fda
        @test doc.version == "1.0.0"
    end

    @testset "Validation summary generation" begin
        pred = Dict(
            "cmax" => [100.0, 110.0, 95.0],
            "auc" => [500.0, 520.0, 480.0]
        )
        obs = Dict(
            "cmax" => [105.0, 108.0, 98.0],
            "auc" => [510.0, 515.0, 490.0]
        )

        validation = generate_validation_summary(pred, obs; n_studies=3)

        @test validation.n_studies == 3
        @test validation.gmfe > 0
        @test validation.gmfe < 2.0  # Should be close to 1
        @test validation.aafe > 0
        @test 0 <= validation.within_2fold <= 1.0
        @test validation.meets_criteria  # Should pass for similar values
    end

    @testset "Validation with FDA criteria" begin
        # Good predictions
        pred_good = Dict("cmax" => [100.0], "auc" => [500.0])
        obs_good = Dict("cmax" => [105.0], "auc" => [510.0])
        val_good = generate_validation_summary(pred_good, obs_good; criteria=:fda)
        @test val_good.meets_criteria

        # Poor predictions
        pred_bad = Dict("cmax" => [100.0], "auc" => [500.0])
        obs_bad = Dict("cmax" => [300.0], "auc" => [1500.0])
        val_bad = generate_validation_summary(pred_bad, obs_bad; criteria=:fda)
        @test !val_bad.meets_criteria
    end

    @testset "DDI assessment" begin
        ddi = generate_ddi_assessment(
            "Itraconazole", "Midazolam",
            10.5, 11.0;
            mechanism = "CYP3A4 inhibition",
            predicted_cmax_ratio = 5.0,
            observed_cmax_ratio = 5.5
        )

        @test ddi.perpetrator == "Itraconazole"
        @test ddi.victim == "Midazolam"
        @test ddi.predicted_class == "strong inhibitor"
        @test ddi.observed_class == "strong inhibitor"
        @test ddi.classification_match
        @test ddi.auc_fe < 1.5  # Good prediction
    end

    @testset "DDI classification" begin
        # Test various DDI classifications
        @test RegulatoryReports.classify_ddi(10.0) == "strong inhibitor"
        @test RegulatoryReports.classify_ddi(3.0) == "moderate inhibitor"
        @test RegulatoryReports.classify_ddi(1.5) == "weak inhibitor"
        @test RegulatoryReports.classify_ddi(1.1) == "no clinically significant interaction"
        @test RegulatoryReports.classify_ddi(0.4) == "strong inducer"
        @test RegulatoryReports.classify_ddi(0.7) == "moderate inducer"
    end

    @testset "Dose recommendations" begin
        rec_strong = RegulatoryReports.generate_dose_recommendation("strong inhibitor", "CYP3A4")
        @test occursin("75%", rec_strong) || occursin("Avoid", rec_strong)

        rec_moderate = RegulatoryReports.generate_dose_recommendation("moderate inhibitor", "CYP3A4")
        @test occursin("50%", rec_moderate)

        rec_weak = RegulatoryReports.generate_dose_recommendation("weak inhibitor", "CYP3A4")
        @test occursin("No dose adjustment", rec_weak) || occursin("monitor", lowercase(rec_weak))
    end

    @testset "FDA report generation" begin
        doc = ModelDocumentation(
            "TestModel", "1.0.0", "Dr. Test", today(),
            "Darwin PBPK", "2.11.0", "TestDrug", "Test Indication",
            "TestPharma", :fda
        )

        pred = Dict("cmax" => [100.0, 110.0], "auc" => [500.0, 520.0])
        obs = Dict("cmax" => [105.0, 108.0], "auc" => [510.0, 515.0])
        validation = generate_validation_summary(pred, obs)

        ddi = generate_ddi_assessment("DrugA", "DrugB", 2.5, 2.3)

        report = generate_fda_report(doc, validation, [ddi])

        @test report.documentation.drug_name == "TestDrug"
        @test !isempty(report.executive_summary)
        @test !isempty(report.model_description)
        @test !isempty(report.conclusions)
        @test length(report.ddi_assessments) == 1
    end

    @testset "EMA report generation" begin
        doc = ModelDocumentation(
            "TestModel", "1.0.0", "Dr. Test", today(),
            "Darwin PBPK", "2.11.0", "TestDrug", "Test Indication",
            "TestPharma", :ema
        )

        pred = Dict("cmax" => [100.0], "auc" => [500.0])
        obs = Dict("cmax" => [105.0], "auc" => [510.0])
        validation = generate_validation_summary(pred, obs; criteria=:ema)

        report = generate_ema_report(doc, validation)

        @test report.documentation.regulatory_agency == :ema
        @test !isempty(report.executive_summary)
    end

    @testset "ICH M12 report generation" begin
        ddis = [
            generate_ddi_assessment("DrugA", "DrugB", 2.5, 2.3; mechanism="CYP3A4"),
            generate_ddi_assessment("DrugC", "DrugB", 1.2, 1.15; mechanism="CYP2D6")
        ]

        report = generate_ich_m12_report(ddis; drug_name="TestDrug")

        @test occursin("ICH M12", report)
        @test occursin("TestDrug", report)
        @test occursin("DrugA", report)
        @test occursin("CYP3A4", report)
    end

    @testset "Markdown export" begin
        doc = ModelDocumentation(
            "TestModel", "1.0.0", "Dr. Test", today(),
            "Darwin PBPK", "2.11.0", "Metformin", "T2DM",
            "Pharma Inc", :fda
        )

        pred = Dict("cmax" => [100.0], "auc" => [500.0])
        obs = Dict("cmax" => [105.0], "auc" => [510.0])
        validation = generate_validation_summary(pred, obs)

        report = generate_fda_report(doc, validation)
        md = export_report_markdown(report)

        @test occursin("# PBPK Model Report", md)
        @test occursin("Metformin", md)
        @test occursin("GMFE", md)
        @test occursin("Validation", md)
        @test occursin("Conclusions", md)
    end

    @testset "JSON export" begin
        doc = ModelDocumentation(
            "TestModel", "1.0.0", "Dr. Test", today(),
            "Darwin PBPK", "2.11.0", "Metformin", "T2DM",
            "Pharma Inc", :fda
        )

        pred = Dict("cmax" => [100.0], "auc" => [500.0])
        obs = Dict("cmax" => [105.0], "auc" => [510.0])
        validation = generate_validation_summary(pred, obs)

        ddi = generate_ddi_assessment("Rifampin", "Metformin", 0.6, 0.55)
        report = generate_fda_report(doc, validation, [ddi])

        json = export_report_json(report)

        @test occursin("\"drug_name\"", json)
        @test occursin("Metformin", json)
        @test occursin("gmfe", json)
        @test occursin("ddi_assessments", json)
    end

    @testset "Acceptance criteria constants" begin
        @test FDA_ACCEPTANCE_CRITERIA["gmfe_threshold"] == 2.0
        @test FDA_ACCEPTANCE_CRITERIA["within_2fold_threshold"] == 0.80
        @test haskey(FDA_ACCEPTANCE_CRITERIA, "ddi_classification")

        @test EMA_ACCEPTANCE_CRITERIA["gmfe_threshold"] == 2.0
        @test haskey(EMA_ACCEPTANCE_CRITERIA, "prediction_bias")
    end

    @testset "ValidationSummary struct" begin
        val = ValidationSummary(
            5, 50, 200,  # n_studies, n_subjects, n_obs
            1.3, 1.4, 1.1,  # gmfe, aafe, afe
            0.85, 0.70,  # within_2fold, within_1.5fold
            1.25, 1.35, 1.5, 1.2,  # cmax_gmfe, auc_gmfe, tmax_gmfe, cl_gmfe
            true, "fda"  # meets_criteria, criteria_used
        )

        @test val.n_studies == 5
        @test val.n_subjects == 50
        @test val.gmfe == 1.3
        @test val.meets_criteria
    end

    @testset "SensitivitySummary struct" begin
        sens = SensitivitySummary(
            "CL", 10.0, (5.0, 20.0),
            25.0, 15.0, true, 1
        )

        @test sens.parameter == "CL"
        @test sens.is_sensitive
        @test sens.tornado_rank == 1
    end

    @testset "PopulationCovariates struct" begin
        cov = PopulationCovariates(
            "weight", -0.5, 0.8,
            "Yes - dose adjustment",
            ["Obese patients: reduce dose by 25%"]
        )

        @test cov.covariate == "weight"
        @test cov.clinical_relevance == "Yes - dose adjustment"
        @test length(cov.subgroup_recommendations) == 1
    end

end

println("Regulatory Reports tests completed!")
