# ===========================================================================
# Tests for Allometric Scaling Module
# ===========================================================================

using Test

include(joinpath(@__DIR__, "..", "src", "DarwinPBPK", "scaling", "AllometricScaling.jl"))
using .AllometricScaling

@testset "Allometric Scaling" begin

    @testset "Species database" begin
        # Test all species exist
        for species in [MOUSE, RAT, RABBIT, DOG, CYNOMOLGUS_MONKEY, RHESUS_MONKEY, MINIPIG, HUMAN]
            data = get_species_data(species)
            @test data.body_weight > 0
            @test data.brain_weight > 0
            @test data.liver_weight > 0
            @test data.cardiac_output > 0
        end

        # Test specific values
        human = get_species_data(HUMAN)
        @test human.body_weight == 70.0
        @test human.brain_weight == 1400.0

        rat = get_species_data(RAT)
        @test rat.body_weight == 0.25
    end

    @testset "Simple allometry fit" begin
        # Test data that should give ~0.75 exponent
        bw = [0.025, 0.25, 10.0, 70.0]  # Mouse, Rat, Dog, Human
        cl = [0.1, 0.5, 10.0, 50.0]  # Hypothetical clearances

        model = simple_allometry(cl, bw; parameter_name="clearance")

        @test model.parameter == "clearance"
        @test model.coefficient > 0
        @test 0.5 < model.exponent < 1.0  # Should be around 0.75
        @test model.r_squared > 0.9
    end

    @testset "Clearance scaling" begin
        # Scale from rat to human
        cl_rat = 2.0  # L/h
        cl_human = scale_clearance(cl_rat, RAT, HUMAN)

        # Human should have higher clearance
        @test cl_human > cl_rat

        # Check scaling factor is reasonable (70/0.25)^0.75 ≈ 68
        expected_factor = (70.0 / 0.25)^0.75
        @test cl_human / cl_rat ≈ expected_factor atol=0.1
    end

    @testset "Volume scaling" begin
        # Scale from rat to human
        vd_rat = 0.5  # L
        vd_human = scale_volume(vd_rat, RAT, HUMAN)

        # Check linear scaling (70/0.25)^1.0 = 280
        expected_factor = 70.0 / 0.25
        @test vd_human / vd_rat ≈ expected_factor atol=0.1
    end

    @testset "Half-life scaling" begin
        t12_rat = 1.0  # h
        t12_human = scale_half_life(t12_rat, RAT, HUMAN)

        # Should scale as BW^0.25
        expected_factor = (70.0 / 0.25)^0.25
        @test t12_human / t12_rat ≈ expected_factor atol=0.1
    end

    @testset "Rule of exponents" begin
        # Create model with different exponents
        bw = [0.025, 0.25, 10.0]

        # Test rule of exponents applies appropriate correction
        cl_low = [0.05, 0.15, 2.0]
        model_low = simple_allometry(cl_low, bw)
        result_low = rule_of_exponents(model_low, 70.0)
        @test !isempty(result_low.method)  # Method should be assigned

        # Test that prediction is made
        @test result_low.predicted_value > 0
        @test result_low.lower_95ci < result_low.predicted_value
        @test result_low.upper_95ci > result_low.predicted_value
    end

    @testset "Brain weight correction" begin
        bw = [0.25, 10.0, 70.0]
        brw = [1.8, 85.0, 1400.0]
        cl = [1.0, 8.0, 30.0]

        result = brain_weight_correction(cl, bw, brw)

        @test result.parameter == "clearance"
        @test result.predicted_value > 0
        @test result.method == "brain_weight_correction"
    end

    @testset "MLP correction" begin
        bw = [0.25, 10.0, 70.0]
        mlp = [5.0, 20.0, 122.0]
        cl = [1.0, 8.0, 30.0]

        result = mlp_correction(cl, bw, mlp)

        @test result.parameter == "clearance"
        @test result.predicted_value > 0
        @test result.method == "MLP_correction"
    end

    @testset "Human PK prediction" begin
        animal_data = Dict(
            MOUSE => Dict("cl" => 0.1, "vd" => 0.02, "t12" => 0.5),
            RAT => Dict("cl" => 0.5, "vd" => 0.2, "t12" => 1.0),
            DOG => Dict("cl" => 8.0, "vd" => 5.0, "t12" => 3.0),
        )

        results = predict_human_pk(animal_data)

        @test haskey(results, "clearance")
        @test haskey(results, "volume")
        @test haskey(results, "half_life")

        @test results["clearance"].predicted_value > 0
        @test results["volume"].predicted_value > 0
        @test results["half_life"].predicted_value > 0
    end

    @testset "First-in-human dose prediction" begin
        # Rat NOAEL of 10 mg/kg
        fih = predict_first_in_human_dose(10.0, RAT; method=:hep)

        @test haskey(fih, "hed_mg_kg")
        @test haskey(fih, "fih_mg_kg")
        @test haskey(fih, "fih_mg_70kg")
        @test haskey(fih, "safety_factor")

        @test fih["hed_mg_kg"] > 0
        @test fih["fih_mg_kg"] < fih["hed_mg_kg"]  # FIH should be lower due to safety factor
        @test fih["fih_mg_70kg"] == fih["fih_mg_kg"] * 70.0

        # Test different methods
        fih_allom = predict_first_in_human_dose(10.0, RAT; method=:allometric)
        @test fih_allom["hed_mg_kg"] > 0

        fih_mabel = predict_first_in_human_dose(10.0, RAT; method=:mabel)
        @test fih_mabel["fih_mg_kg"] < fih["fih_mg_kg"]  # MABEL is more conservative
    end

    @testset "IVIVE - Hepatocyte scaling" begin
        # 50 μL/min/10^6 cells intrinsic clearance
        result = hepatocyte_scaling(50.0, HUMAN)

        @test haskey(result, "cl_int_liver_ml_min")
        @test haskey(result, "cl_hepatic_ml_min")
        @test haskey(result, "cl_hepatic_L_h")
        @test haskey(result, "extraction_ratio")

        @test result["cl_int_liver_ml_min"] > 0
        @test result["cl_hepatic_L_h"] > 0
        @test 0 < result["extraction_ratio"] <= 1.0
    end

    @testset "IVIVE - Microsomal scaling" begin
        # 100 μL/min/mg protein
        result = microsomal_scaling(100.0, HUMAN)

        @test result["cl_hepatic_L_h"] > 0
        @test result["extraction_ratio"] <= 1.0
    end

    @testset "IVIVE complete pipeline" begin
        # In vitro clearance in rat hepatocytes
        result = ivive_clearance(30.0, RAT, HUMAN; source=:hepatocyte)

        @test result["source_species"] == RAT
        @test result["target_species"] == HUMAN
        @test result["cl_invivo_source_L_h"] > 0
        @test result["cl_invivo_target_L_h"] > 0
    end

    @testset "Dedrick plot" begin
        species = [RAT, DOG, CYNOMOLGUS_MONKEY]
        cl = [1.0, 10.0, 6.0]
        vd = [0.5, 8.0, 4.0]
        t12 = [0.5, 2.0, 1.5]

        dedrick = dedrick_plot(species, cl, vd, t12)

        @test haskey(dedrick, "kallynochrons")
        @test haskey(dedrick, "apolysichrons")
        @test haskey(dedrick, "dienetichrons")
        @test length(dedrick["kallynochrons"]) == 3
    end

    @testset "Allometric exponent calculation" begin
        species = [MOUSE, RAT, DOG, HUMAN]
        cl = [0.1, 0.5, 10.0, 50.0]

        exponent = calculate_allometric_exponent(species, cl)

        @test 0.5 < exponent < 1.0  # Should be around 0.75
    end

    @testset "Fit allometric model from dict" begin
        data = Dict(
            MOUSE => 0.1,
            RAT => 0.5,
            DOG => 10.0
        )

        model = fit_allometric_model(data; parameter="clearance")

        @test model.parameter == "clearance"
        @test length(model.species_used) == 3
        @test model.r_squared > 0
    end

    @testset "Standard exponents" begin
        @test STANDARD_EXPONENTS["clearance"] == 0.75
        @test STANDARD_EXPONENTS["volume"] == 1.0
        @test STANDARD_EXPONENTS["half_life"] == 0.25
    end

end

println("Allometric Scaling tests completed!")
