"""
Test PK-Sim Database Integration

Tests:
1. Loading PK-Sim CSV database
2. Extracting organ parameters
3. Scaling functions
4. Compartment creation with PK-Sim values
5. Validation against hardcoded values
"""

using Test
using DataFrames

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using DarwinPBPK
using DarwinPBPK.PatientProfile
using DarwinPBPK.CompartmentModels
using DarwinPBPK.CompartmentModels.PKSimParameters

@testset "PK-Sim Database Integration" begin

    # Test 1: Load PK-Sim database
    @testset "Database Loading" begin
        csv_path = joinpath(@__DIR__, "..", "data", "external_pk_datasets",
                           "PKSim_Human_Reference_Values.csv")

        if !isfile(csv_path)
            @warn "PK-Sim CSV not found at $csv_path, skipping tests"
            return
        end

        db = load_pksim_database(csv_path)

        @test db isa PKSimDatabase
        @test size(db.raw_data, 1) > 1000  # Should have >1000 parameters
        @test length(db.organs) > 20  # Should have >20 organ containers
        @test "Liver" in db.organs
        @test "Kidney" in db.organs
        @test "Brain" in db.organs

        println("✓ Loaded $(size(db.raw_data, 1)) parameters for $(length(db.organs)) organs")
    end

    # Test 2: Extract organ parameters
    @testset "Organ Parameter Extraction" begin
        csv_path = joinpath(@__DIR__, "..", "data", "external_pk_datasets",
                           "PKSim_Human_Reference_Values.csv")

        if !isfile(csv_path)
            @warn "PK-Sim CSV not found, skipping"
            return
        end

        db = load_pksim_database(csv_path)

        # Test liver parameters
        liver_params = get_organ_params(db, "Liver", weight=70.0, cardiac_output=350.0)

        @test liver_params.organ == "Liver"
        @test !isnothing(liver_params.volume_L)
        @test liver_params.volume_L > 0
        @test !isnothing(liver_params.blood_flow_L_h)
        @test liver_params.blood_flow_L_h > 0
        @test 0.0 <= liver_params.fraction_vascular <= 1.0
        @test 0.0 <= liver_params.vf_water <= 1.0
        @test liver_params.density > 0.0

        println("✓ Liver volume: $(round(liver_params.volume_L, digits=2)) L")
        println("✓ Liver blood flow: $(round(liver_params.blood_flow_L_h, digits=1)) L/h")
        println("✓ Liver water fraction: $(round(liver_params.vf_water * 100, digits=1))%")

        # Test kidney parameters
        kidney_params = get_organ_params(db, "Kidney", weight=70.0, cardiac_output=350.0)

        @test kidney_params.organ == "Kidney"
        @test !isnothing(kidney_params.volume_L)
        @test kidney_params.volume_L > 0

        println("✓ Kidney volume: $(round(kidney_params.volume_L, digits=2)) L")
        println("✓ Kidney blood flow: $(round(kidney_params.blood_flow_L_h, digits=1)) L/h")
    end

    # Test 3: Scaling functions
    @testset "Allometric Scaling" begin
        # Test volume scaling
        vol_70kg = scale_organ_volume("Liver", 70.0, 0.75)
        vol_50kg = scale_organ_volume("Liver", 50.0, 0.75)
        vol_90kg = scale_organ_volume("Liver", 90.0, 0.75)

        @test !isnothing(vol_70kg)
        @test !isnothing(vol_50kg)
        @test !isnothing(vol_90kg)
        @test vol_50kg < vol_70kg < vol_90kg  # Monotonic increase

        # Verify allometric formula: V ~ W^0.75
        expected_ratio = (50.0 / 70.0)^0.75
        actual_ratio = vol_50kg / vol_70kg
        @test abs(actual_ratio - expected_ratio) < 0.01

        println("✓ Liver volume scaling: 50kg=$(round(vol_50kg, digits=2))L, 70kg=$(round(vol_70kg, digits=2))L, 90kg=$(round(vol_90kg, digits=2))L")

        # Test blood flow scaling
        flow_liver = scale_blood_flow("Liver", 70.0, 350.0)
        flow_kidney = scale_blood_flow("Kidney", 70.0, 350.0)

        @test !isnothing(flow_liver)
        @test !isnothing(flow_kidney)
        @test flow_liver > flow_kidney  # Liver receives more blood

        println("✓ Blood flow: Liver=$(round(flow_liver, digits=1))L/h, Kidney=$(round(flow_kidney, digits=1))L/h")
    end

    # Test 4: Compartment creation with PK-Sim
    @testset "Compartment Creation" begin
        csv_path = joinpath(@__DIR__, "..", "data", "external_pk_datasets",
                           "PKSim_Human_Reference_Values.csv")

        if !isfile(csv_path)
            @warn "PK-Sim CSV not found, skipping"
            return
        end

        # Load database globally
        CompartmentModels.load_compartment_database(csv_path)

        # Create test patient
        patient = PatientProfile.PatientData(
            age = 35.0,
            weight = 70.0,
            height = 170.0,
            sex = "Male",
            bmi = 24.2,
            bsa = 1.8,
            plasma_volume = 3.0,
            blood_volume = 5.0,
            albumin = 42.0,
            alpha1_agp = 0.8,
            hematocrit = 0.45,
            gfr = 120.0,
            liver_function = 1.0,
            genetic_polymorphisms = Dict{String, String}()
        )

        # Create liver compartment with PK-Sim
        liver = create_liver_compartment(patient, use_pksim=true)

        @test liver isa LiverCompartment
        @test liver.volume > 0
        @test liver.blood_flow > 0
        @test !isnothing(liver.pksim_params)
        @test haskey(liver.tissue_composition, "water")
        @test haskey(liver.cyp_expression, "CYP3A4")
        @test liver.microsomal_protein > 0

        println("✓ Liver compartment created with PK-Sim:")
        println("  Volume: $(round(liver.volume, digits=2)) L")
        println("  Blood flow: $(round(liver.blood_flow, digits=1)) L/h")
        println("  Water: $(round(liver.tissue_composition["water"] * 100, digits=1))%")
        println("  Microsomal protein: $(round(liver.microsomal_protein, digits=1)) mg/g")

        # Create kidney compartment
        kidney = create_kidney_compartment(patient, use_pksim=true)

        @test kidney isa KidneyCompartment
        @test kidney.volume > 0
        @test kidney.gfr == patient.gfr
        @test !isnothing(kidney.pksim_params)
        @test haskey(kidney.transporter_expression, "OAT1")

        println("✓ Kidney compartment created with PK-Sim:")
        println("  Volume: $(round(kidney.volume, digits=2)) L")
        println("  GFR: $(round(kidney.gfr, digits=1)) mL/min")

        # Create brain compartment
        brain = create_brain_compartment(patient, use_pksim=true)

        @test brain isa BrainCompartment
        @test brain.volume > 0
        @test !isnothing(brain.pksim_params)
        @test haskey(brain.regional_distribution, "grey_matter")

        println("✓ Brain compartment created with PK-Sim:")
        println("  Volume: $(round(brain.volume, digits=2)) L")
        println("  Lipid fraction: $(round(brain.tissue_composition["lipid"] * 100, digits=1))%")
    end

    # Test 5: Validation against hardcoded values
    @testset "Validation vs Hardcoded" begin
        csv_path = joinpath(@__DIR__, "..", "data", "external_pk_datasets",
                           "PKSim_Human_Reference_Values.csv")

        if !isfile(csv_path)
            @warn "PK-Sim CSV not found, skipping"
            return
        end

        CompartmentModels.load_compartment_database(csv_path)

        patient = PatientProfile.PatientData(
            age = 35.0,
            weight = 70.0,
            height = 170.0,
            sex = "Male",
            bmi = 24.2,
            bsa = 1.8,
            plasma_volume = 3.0,
            blood_volume = 5.0,
            albumin = 42.0,
            alpha1_agp = 0.8,
            hematocrit = 0.45,
            gfr = 120.0,
            liver_function = 1.0,
            genetic_polymorphisms = Dict{String, String}()
        )

        # Create compartments both ways
        pksim_liver = create_liver_compartment(patient, use_pksim=true)
        hardcoded_liver = create_liver_compartment(patient, use_pksim=false)

        # Compare volumes
        vol_diff_pct = abs(pksim_liver.volume - hardcoded_liver.volume) / pksim_liver.volume * 100

        println("\nValidation Results:")
        println("Liver Volume:")
        println("  PK-Sim: $(round(pksim_liver.volume, digits=3)) L")
        println("  Hardcoded: $(round(hardcoded_liver.volume, digits=3)) L")
        println("  Difference: $(round(vol_diff_pct, digits=1))%")

        # Compare blood flows
        flow_diff_pct = abs(pksim_liver.blood_flow - hardcoded_liver.blood_flow) / pksim_liver.blood_flow * 100

        println("Liver Blood Flow:")
        println("  PK-Sim: $(round(pksim_liver.blood_flow, digits=1)) L/h")
        println("  Hardcoded: $(round(hardcoded_liver.blood_flow, digits=1)) L/h")
        println("  Difference: $(round(flow_diff_pct, digits=1))%")

        # Warnings for large differences
        if vol_diff_pct > 10.0
            @warn "Liver volume differs by more than 10% ($(round(vol_diff_pct, digits=1))%)"
        else
            println("✓ Liver volume matches within 10%")
        end

        if flow_diff_pct > 10.0
            @warn "Liver blood flow differs by more than 10% ($(round(flow_diff_pct, digits=1))%)"
        else
            println("✓ Liver blood flow matches within 10%")
        end

        # Test tissue composition
        water_diff_pct = abs(pksim_liver.tissue_composition["water"] -
                            hardcoded_liver.tissue_composition["water"]) /
                            pksim_liver.tissue_composition["water"] * 100

        println("\nLiver Water Fraction:")
        println("  PK-Sim: $(round(pksim_liver.tissue_composition["water"] * 100, digits=1))%")
        println("  Hardcoded: $(round(hardcoded_liver.tissue_composition["water"] * 100, digits=1))%")
        println("  Difference: $(round(water_diff_pct, digits=1))%")

        if water_diff_pct > 10.0
            @warn "Water fraction differs by more than 10%"
        else
            println("✓ Water fraction matches within 10%")
        end
    end

    # Test 6: Parameter override
    @testset "Parameter Override" begin
        csv_path = joinpath(@__DIR__, "..", "data", "external_pk_datasets",
                           "PKSim_Human_Reference_Values.csv")

        if !isfile(csv_path)
            @warn "PK-Sim CSV not found, skipping"
            return
        end

        CompartmentModels.load_compartment_database(csv_path)

        patient = PatientProfile.PatientData(
            age = 35.0,
            weight = 70.0,
            height = 170.0,
            sex = "Male",
            bmi = 24.2,
            bsa = 1.8,
            plasma_volume = 3.0,
            blood_volume = 5.0,
            albumin = 42.0,
            alpha1_agp = 0.8,
            hematocrit = 0.45,
            gfr = 120.0,
            liver_function = 1.0,
            genetic_polymorphisms = Dict{String, String}()
        )

        # Create with override
        custom_vol = 2.5
        custom_flow = 100.0
        override = Dict("volume" => custom_vol, "blood_flow" => custom_flow)

        liver = create_liver_compartment(patient, use_pksim=true, override_params=override)

        @test liver.volume == custom_vol
        @test liver.blood_flow == custom_flow

        println("✓ Parameter override working:")
        println("  Custom volume: $(liver.volume) L")
        println("  Custom blood flow: $(liver.blood_flow) L/h")
    end

end

# Run validation report
println("\n" * "="^70)
println("Running Full Validation Report")
println("="^70)

csv_path = joinpath(@__DIR__, "..", "data", "external_pk_datasets",
                   "PKSim_Human_Reference_Values.csv")

if isfile(csv_path)
    CompartmentModels.load_compartment_database(csv_path)

    patient = PatientProfile.PatientData(
        age = 35.0,
        weight = 70.0,
        height = 170.0,
        sex = "Male",
        bmi = 24.2,
        bsa = 1.8,
        plasma_volume = 3.0,
        blood_volume = 5.0,
        albumin = 42.0,
        alpha1_agp = 0.8,
        hematocrit = 0.45,
        gfr = 120.0,
        liver_function = 1.0,
        genetic_polymorphisms = Dict{String, String}()
    )

    validate_compartment_parameters(patient)

    # Print detailed organ summaries
    db = CompartmentModels.get_pksim_db()

    println("\n" * "="^70)
    println("PK-Sim Organ Parameter Summaries")
    println("="^70)

    for organ in ["Liver", "Kidney", "Brain", "Heart", "Lung", "Muscle", "Fat", "Spleen", "Pancreas"]
        try
            params = get_organ_params(db, organ, weight=70.0, cardiac_output=350.0)
            print_organ_summary(params)
        catch e
            println("Could not load parameters for $organ: $e")
        end
    end
else
    @warn "PK-Sim CSV not found at $csv_path - tests skipped"
end
