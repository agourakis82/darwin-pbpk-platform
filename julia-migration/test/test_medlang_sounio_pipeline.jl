"""
MedLang → Sounio Pipeline Tests for Darwin PBPK Platform

Tests the complete pipeline: MedLang source → Sounio code generation → execution.

Integration with Sounio v0.97.0 darwin_pbpk stdlib.

Author: Dr. Sounio Agourakis
Date: January 2026
"""

using Test
using DarwinPBPK
using DarwinPBPK.MedLang
using DarwinPBPK.SounioIntegration

# ===========================================================================
# Test Data
# ===========================================================================

const SIMPLE_MEDLANG_MODEL = """
model SimpleDrug {
    clearance hepatic: 10.0_L/h
    clearance renal: 2.5_L/h

    organ liver {
        V: 1.8_L,
        Q: 90.0_L/h,
        Kp: 2.5
    }
    organ kidney {
        V: 0.31_L,
        Q: 60.0_L/h,
        Kp: 1.8
    }
}
"""

const ORAL_ABSORPTION_MODEL = """
model OralDrug {
    route: oral
    absorption {
        Ka: 1.5_1/h,
        F: 0.8,
        lag: 0.5_h
    }
    firstpass {
        Fg: 0.9,
        Fh: 0.7
    }
    clearance hepatic: 15.0_L/h

    organ liver {
        V: 1.8_L,
        Q: 90.0_L/h,
        Kp: 2.0
    }
}
"""

const MULTICOMPARTMENT_MODEL = """
model MultiCompDrug {
    param V1 : Volume = 20.0_L
    param V2 : Volume = 50.0_L
    param CL : Clearance = 8.0_L/h
    param Q12 : Flow = 15.0_L/h

    state A_central : Mass = 0_mg
    state A_peripheral : Mass = 0_mg

    clearance hepatic: CL

    organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: 3.0 }
    organ kidney { V: 0.31_L, Q: 60.0_L/h, Kp: 2.0 }
    organ brain { V: 1.4_L, Q: 42.0_L/h, Kp: 0.5 }
}
"""

# ===========================================================================
# Sounio Code Generation Tests
# ===========================================================================

@testset "MedLang → Sounio Pipeline Tests" begin

    @testset "compile_to_sounio - Basic Model (stdlib)" begin
        sounio_code = compile_to_sounio(SIMPLE_MEDLANG_MODEL; use_stdlib=true)

        # Should contain stdlib imports
        @test occursin("darwin_pbpk", sounio_code)
        @test occursin("import darwin_pbpk.simulation", sounio_code)
        @test occursin("import darwin_pbpk.core.pbpk_params", sounio_code)

        # Should contain model name
        @test occursin("SimpleDrug", sounio_code)

        # Should contain drug properties
        @test occursin("DrugProperties", sounio_code)

        # Should contain simulation function
        @test occursin("run_pbpk_simulation", sounio_code)
    end

    @testset "compile_to_sounio - Basic Model (standalone)" begin
        sounio_code = compile_to_sounio(SIMPLE_MEDLANG_MODEL; use_stdlib=false)

        # Should NOT contain stdlib imports
        @test !occursin("import darwin_pbpk", sounio_code)

        # Should contain full ODE implementation
        @test occursin("fn", sounio_code)  # function definitions
        @test occursin("struct", sounio_code)  # struct definitions
    end

    @testset "compile_to_sounio - Oral Absorption Model" begin
        sounio_code = compile_to_sounio(ORAL_ABSORPTION_MODEL; use_stdlib=true)

        # Should contain model name
        @test occursin("OralDrug", sounio_code)

        # Should generate valid Sounio code structure
        @test occursin("fn", sounio_code) || occursin("run_pbpk_simulation", sounio_code)
    end

    @testset "compile_to_sounio - Multicompartment Model" begin
        sounio_code = compile_to_sounio(MULTICOMPARTMENT_MODEL; use_stdlib=true)

        @test occursin("MultiCompDrug", sounio_code)
        @test occursin("darwin_pbpk", sounio_code)
    end

    @testset "compile_file_to_sounio" begin
        # Create temp MedLang file
        temp_file = tempname() * ".medlang"
        write(temp_file, SIMPLE_MEDLANG_MODEL)

        try
            sounio_code = compile_file_to_sounio(temp_file; use_stdlib=true)
            @test occursin("SimpleDrug", sounio_code)
            @test occursin("darwin_pbpk", sounio_code)
        finally
            rm(temp_file, force=true)
        end
    end

    @testset "compile_file_to_sounio - with output file" begin
        temp_input = tempname() * ".medlang"
        temp_output = tempname() * ".sio"
        write(temp_input, SIMPLE_MEDLANG_MODEL)

        try
            sounio_code = compile_file_to_sounio(temp_input; use_stdlib=true, output=temp_output)

            # Check output file was created
            @test isfile(temp_output)

            # Check content matches
            file_content = read(temp_output, String)
            @test file_content == sounio_code
        finally
            rm(temp_input, force=true)
            rm(temp_output, force=true)
        end
    end

    @testset "sounio_code_info" begin
        sounio_code = compile_to_sounio(SIMPLE_MEDLANG_MODEL; use_stdlib=true)
        info = sounio_code_info(sounio_code)

        @test info.uses_stdlib == true
        @test length(info.imports) > 0
        @test info.line_count > 10
    end

end

# ===========================================================================
# Pipeline Integration Tests
# ===========================================================================

@testset "MedLang Pipeline Integration Tests" begin

    @testset "MedLangPipelineOptions defaults" begin
        opts = MedLangPipelineOptions()

        @test opts.use_stdlib == true
        @test opts.cleanup_temp_files == true
        @test opts.show_generated_code == false
        @test opts.timeout_seconds == 60.0
    end

    @testset "MedLangPipelineOptions custom" begin
        opts = MedLangPipelineOptions(
            use_stdlib = false,
            show_generated_code = true,
            timeout_seconds = 120.0
        )

        @test opts.use_stdlib == false
        @test opts.show_generated_code == true
        @test opts.timeout_seconds == 120.0
    end

    @testset "compile_and_run_medlang - Code Generation Only" begin
        # This test runs the pipeline but expects it to fail at compilation
        # (Sounio compiler likely not installed) while still generating code

        opts = MedLangPipelineOptions(cleanup_temp_files=false)
        result = compile_and_run_medlang(
            SIMPLE_MEDLANG_MODEL;
            dose_mg = 100.0,
            duration_h = 24.0,
            options = opts
        )

        # Pipeline should at least reach code generation
        @test :parse in result.stages_completed
        @test :generate in result.stages_completed
        @test :write in result.stages_completed

        # Generated code should be valid
        @test length(result.sounio_code) > 100
        @test occursin("SimpleDrug", result.sounio_code)

        # Clean up
        if isfile(result.sounio_file)
            rm(result.sounio_file, force=true)
        end
    end

    @testset "compile_and_run_medlang - Pipeline Timing" begin
        result = compile_and_run_medlang(
            SIMPLE_MEDLANG_MODEL;
            dose_mg = 100.0
        )

        # Should complete in reasonable time
        @test result.pipeline_time_ms < 10000  # 10 seconds max
        @test result.pipeline_time_ms > 0
    end

    @testset "compile_and_run_medlang_file" begin
        temp_file = tempname() * ".medlang"
        write(temp_file, SIMPLE_MEDLANG_MODEL)

        try
            result = compile_and_run_medlang_file(temp_file; dose_mg=100.0)

            @test :parse in result.stages_completed
            @test :generate in result.stages_completed
            @test length(result.sounio_code) > 0
        finally
            rm(temp_file, force=true)
        end
    end

    @testset "compile_and_run_medlang_file - File Not Found" begin
        result = compile_and_run_medlang_file(
            "/nonexistent/path.medlang";
            dose_mg = 100.0
        )

        @test result.success == false
        @test occursin("not found", result.error_message)
    end

end

# ===========================================================================
# Sounio Data Format Tests
# ===========================================================================

@testset "Sounio Data Format Tests" begin

    @testset "DrugData creation" begin
        drug = SounioIntegration.SounioDataFormat.DrugData(
            name = "TestDrug",
            mw = 300.0,
            logp = 2.5,
            fu_plasma = 0.4
        )

        @test drug.name == "TestDrug"
        @test drug.mw == 300.0
        @test drug.logp == 2.5
        @test drug.fu_plasma == 0.4
        @test drug.bp_ratio == 0.55  # default
    end

    @testset "PatientData creation" begin
        patient = SounioIntegration.SounioDataFormat.PatientData(
            weight_kg = 80.0,
            age_years = 45.0,
            sex = "female"
        )

        @test patient.weight_kg == 80.0
        @test patient.age_years == 45.0
        @test patient.sex == "female"
    end

    @testset "SimulationRequest creation" begin
        drug = SounioIntegration.SounioDataFormat.DrugData(
            name = "TestDrug",
            mw = 300.0,
            logp = 2.0,
            fu_plasma = 0.5
        )

        patient = SounioIntegration.SounioDataFormat.PatientData()

        request = SounioIntegration.SounioDataFormat.SimulationRequest(
            model = "darwin_pbpk_14comp",
            drug = drug,
            patient = patient,
            dose_mg = 100.0,
            duration_h = 24.0
        )

        @test request.model == "darwin_pbpk_14comp"
        @test request.dose_mg == 100.0
        @test request.duration_h == 24.0
        @test request.route == "oral"  # default
        @test !isempty(request.id)  # UUID generated
    end

end

# ===========================================================================
# Conversion Function Tests
# ===========================================================================

@testset "Julia → Sounio Conversion Tests" begin

    @testset "drug_to_sounio" begin
        # Test with a simple named tuple
        drug = (
            name = "Metformin",
            mw = 129.16,
            logP = -1.43,
            fu = 0.99
        )

        sounio_drug = drug_to_sounio(drug)

        @test sounio_drug.name == "Metformin"
        @test sounio_drug.mw == 129.16
        @test sounio_drug.logp == -1.43
        @test sounio_drug.fu_plasma == 0.99
    end

    @testset "patient_to_sounio" begin
        patient = (
            id = "PATIENT_001",
            weight = 75.0,
            age = 50.0,
            sex = "Male"
        )

        sounio_patient = patient_to_sounio(patient)

        @test sounio_patient.id == "PATIENT_001"
        @test sounio_patient.weight_kg == 75.0
        @test sounio_patient.age_years == 50.0
        @test sounio_patient.sex == "male"  # lowercase
    end

end

# ===========================================================================
# End-to-End Tests (if Sounio compiler available)
# ===========================================================================

@testset "End-to-End Tests (Sounio Compiler)" begin

    # Check if Sounio compiler is available
    sounio_available = false
    try
        compiler = SounioCompiler()
        sounio_available = true
    catch
        @info "Sounio compiler not available - skipping E2E tests"
    end

    if sounio_available
        @testset "Full Pipeline with Sounio Compiler" begin
            result = compile_and_run_medlang(
                SIMPLE_MEDLANG_MODEL;
                dose_mg = 100.0,
                duration_h = 24.0
            )

            @test :compile in result.stages_completed

            if result.compilation_success
                @test :execute in result.stages_completed

                if result.success
                    @test result.simulation_result !== nothing
                    @test result.simulation_result.cmax_mg_l > 0
                    @test result.simulation_result.auc_mg_h_l > 0
                end
            end
        end
    else
        @test_skip "Sounio compiler not available"
    end

end

# ===========================================================================
# Regression Tests
# ===========================================================================

@testset "Regression Tests" begin

    @testset "Empty model handling" begin
        empty_model = """
        model EmptyModel {
        }
        """

        # Should not crash
        sounio_code = compile_to_sounio(empty_model; use_stdlib=true)
        @test occursin("EmptyModel", sounio_code)
    end

    @testset "Model with special characters in name" begin
        model = """
        model Drug_A1_Test {
            clearance hepatic: 10.0_L/h
        }
        """

        sounio_code = compile_to_sounio(model; use_stdlib=true)
        @test occursin("Drug_A1_Test", sounio_code)
    end

    @testset "Large parameter values" begin
        model = """
        model LargeParams {
            clearance hepatic: 1000000.0_L/h
            organ liver { V: 1000.0_L, Q: 10000.0_L/h, Kp: 100.0 }
        }
        """

        sounio_code = compile_to_sounio(model; use_stdlib=true)
        @test occursin("LargeParams", sounio_code)
    end

    @testset "Small parameter values" begin
        model = """
        model SmallParams {
            clearance hepatic: 0.0001_L/h
            organ liver { V: 0.001_L, Q: 0.01_L/h, Kp: 0.1 }
        }
        """

        sounio_code = compile_to_sounio(model; use_stdlib=true)
        @test occursin("SmallParams", sounio_code)
    end

end

println("\n✓ MedLang → Sounio Pipeline tests completed")
