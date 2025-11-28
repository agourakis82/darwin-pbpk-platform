"""
MedLang DSL Tests for Darwin PBPK Platform

Tests the first real implementation of MedLang DSL.
Reference: github.com/agourakis82/medlang

Author: Dr. Demetrios Agourakis
Date: November 2025
"""

using Test
using DarwinPBPK
using DarwinPBPK.MedLang
using DarwinPBPK.ODEPBPKSolver

@testset "MedLang DSL Tests" begin

    @testset "Parser - Basic Tokenization" begin
        source = """
        model Test {
            param CL : Clearance = 10.0_L/h
        }
        """
        ast = parse_medlang(source)

        @test length(ast.models) == 1
        @test ast.models[1].name == "Test"
    end

    @testset "Parser - State Declarations" begin
        source = """
        model StateTest {
            state A_central : DoseMass = 0_mg
            state A_peripheral : DoseMass = 0_mg
        }
        """
        ast = parse_medlang(source)

        @test length(ast.models[1].states) == 2
        @test ast.models[1].states[1].name == "A_central"
        @test ast.models[1].states[2].name == "A_peripheral"
    end

    @testset "Parser - Parameter Declarations" begin
        source = """
        model ParamTest {
            param Ka : RateConst = 1.0_1/h
            param CL : Clearance = 10.0_L/h
            param V : Volume = 50.0_L
        }
        """
        ast = parse_medlang(source)
        model = ast.models[1]

        @test length(model.params) == 3
        @test model.params[1].name == "Ka"
        @test model.params[2].name == "CL"
        @test model.params[3].name == "V"
    end

    @testset "Parser - Organ Definitions" begin
        source = """
        model OrganTest {
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
        ast = parse_medlang(source)
        model = ast.models[1]

        @test length(model.organs) == 2
        @test model.organs[1].name == "liver"
        @test model.organs[2].name == "kidney"
    end

    @testset "Parser - Clearance Definitions" begin
        source = """
        model ClearanceTest {
            clearance hepatic: 10.0_L/h
            clearance renal: 2.5_L/h
        }
        """
        ast = parse_medlang(source)
        model = ast.models[1]

        @test length(model.clearances) == 2
        @test model.clearances[1].mechanism == :hepatic
        @test model.clearances[2].mechanism == :renal
    end

    @testset "Parser - ODE Equations" begin
        source = """
        model ODETest {
            state A_gut : DoseMass
            state A_central : DoseMass
            param Ka : RateConst

            d/dt A_gut = -Ka * A_gut
            d/dt A_central = Ka * A_gut
        }
        """
        ast = parse_medlang(source)
        model = ast.models[1]

        @test length(model.odes) == 2
        @test model.odes[1].state == "A_gut"
        @test model.odes[2].state == "A_central"
    end

    @testset "Parser - Observables" begin
        source = """
        model ObsTest {
            state A_central : DoseMass
            param V : Volume = 50.0_L

            obs C_plasma : ConcMass = A_central / V
        }
        """
        ast = parse_medlang(source)
        model = ast.models[1]

        @test length(model.observables) == 1
        @test model.observables[1].name == "C_plasma"
    end

    @testset "Parser - Timeline Events" begin
        source = """
        timeline TestTimeline {
            at 0_h: dose { amount = 100_mg, to = A_blood }
            at 1_h: observe C_plasma
            at 2_h: observe C_plasma
        }
        """
        ast = parse_medlang(source)

        @test length(ast.timelines) == 1
        @test ast.timelines[1].name == "TestTimeline"
        @test length(ast.timelines[1].events) == 3
    end

    @testset "Parser - Population Definitions" begin
        source = """
        population TestPop {
            model TestModel
            param theta_CL : Clearance = 10.0_L/h
            rand eta_CL : f64 ~ Normal(0, 0.3)
            input WT : Mass
        }
        """
        ast = parse_medlang(source)

        @test length(ast.populations) == 1
        @test ast.populations[1].name == "TestPop"
        @test ast.populations[1].model == "TestModel"
        @test length(ast.populations[1].random_effects) == 1
    end

    @testset "Parser - Expression Types" begin
        source = """
        model ExprTest {
            param a : f64 = 1.0
            param b : f64 = 2.0
            param c : f64 = a + b * 3.0
            param d : f64 = exp(-a)
            param e : f64 = (a + b) / c
        }
        """
        ast = parse_medlang(source)
        # Should parse without errors
        @test length(ast.models[1].params) == 5
    end

    @testset "Transpiler - Basic Model to PBPKParams" begin
        source = """
        model BasicModel {
            organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: 2.0 }
            organ kidney { V: 0.31_L, Q: 60.0_L/h, Kp: 1.5 }
            clearance hepatic: 10.0_L/h
            clearance renal: 2.5_L/h
        }
        """

        params = compile_model(source)

        @test params isa PBPKParams
        @test params.clearance_hepatic ≈ 10.0
        @test params.clearance_renal ≈ 2.5
    end

    @testset "Transpiler - Full PBPK Model" begin
        source = """
        model FullPBPK {
            organ blood { V: 5.0_L, Q: 0.0_L/h, Kp: 1.0 }
            organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: 1.5 }
            organ kidney { V: 0.31_L, Q: 60.0_L/h, Kp: 1.2 }
            organ brain { V: 1.4_L, Q: 50.0_L/h, Kp: 0.5 }
            organ heart { V: 0.33_L, Q: 20.0_L/h, Kp: 1.0 }
            organ lung { V: 0.5_L, Q: 300.0_L/h, Kp: 1.0 }
            organ muscle { V: 30.0_L, Q: 75.0_L/h, Kp: 0.8 }
            organ adipose { V: 15.0_L, Q: 12.0_L/h, Kp: 2.0 }
            organ gut { V: 1.1_L, Q: 45.0_L/h, Kp: 1.0 }
            organ skin { V: 3.3_L, Q: 10.0_L/h, Kp: 0.9 }
            organ bone { V: 10.0_L, Q: 5.0_L/h, Kp: 0.3 }
            organ spleen { V: 0.18_L, Q: 15.0_L/h, Kp: 1.0 }
            organ pancreas { V: 0.1_L, Q: 5.0_L/h, Kp: 1.0 }
            organ other { V: 5.0_L, Q: 20.0_L/h, Kp: 1.0 }

            clearance hepatic: 15.0_L/h
            clearance renal: 3.0_L/h
        }
        """

        params = compile_model(source)

        @test params.clearance_hepatic ≈ 15.0
        @test params.clearance_renal ≈ 3.0
        # Test that volumes are populated
        @test params.volumes[1] > 0  # blood
        @test params.volumes[2] > 0  # liver
    end

    @testset "Transpiler - Julia Code Generation" begin
        source = """
        model CodeGenTest {
            state A_central : DoseMass
            param CL : Clearance = 10.0_L/h
            param V : Volume = 50.0_L

            d/dt A_central = -(CL / V) * A_central
            obs C_plasma : ConcMass = A_central / V
        }
        """

        julia_code = generate_julia_module(source)

        @test contains(julia_code, "module GeneratedPBPK")
        @test contains(julia_code, "CodeGenTestParams")
        @test contains(julia_code, "CodeGenTest_ode!")
    end

    @testset "Validation - Model Validation" begin
        # Valid model
        valid_source = """
        model ValidModel {
            clearance hepatic: 10.0_L/h
        }
        """
        issues = validate_model(valid_source)
        @test !any(startswith(i, "ERROR") for i in issues)

        # Model without clearance
        no_cl_source = """
        model NoClearance {
            organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: 1.0 }
        }
        """
        issues = validate_model(no_cl_source)
        @test any(contains(i, "no clearance") for i in issues)
    end

    @testset "Describe Model" begin
        source = """
        model DescribeTest {
            state A : DoseMass
            param CL : Clearance = 10.0_L/h
            organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: 1.5 }
            clearance hepatic: 10.0_L/h
            obs C : ConcMass = A / 5.0
        }
        """

        description = describe_model(source)

        @test contains(description, "DescribeTest")
        @test contains(description, "States (1)")
        @test contains(description, "Parameters (1)")
        @test contains(description, "Organs (1)")
    end

    @testset "Integration - Simulate from MedLang" begin
        source = """
        model SimTest {
            organ blood { V: 5.0_L, Q: 0.0_L/h, Kp: 1.0 }
            organ liver { V: 1.8_L, Q: 90.0_L/h, Kp: 1.5 }
            organ kidney { V: 0.31_L, Q: 60.0_L/h, Kp: 1.2 }
            clearance hepatic: 10.0_L/h
            clearance renal: 2.5_L/h
        }
        """

        results = simulate_medlang(source, 100.0; t_max=24.0, num_points=50)

        @test haskey(results, "time")
        @test haskey(results, "blood")
        @test haskey(results, "liver")
        @test length(results["time"]) == 50
        @test results["blood"][1] > 0  # Initial dose in blood
    end

    @testset "Multiple Models in Single File" begin
        source = """
        model ModelA {
            clearance hepatic: 5.0_L/h
        }

        model ModelB {
            clearance hepatic: 15.0_L/h
        }
        """

        ast = parse_medlang(source)
        @test length(ast.models) == 2

        params_a = compile_model(source; model_name="ModelA")
        params_b = compile_model(source; model_name="ModelB")

        @test params_a.clearance_hepatic ≈ 5.0
        @test params_b.clearance_hepatic ≈ 15.0
    end

    @testset "Unit Conversions" begin
        # Test mL/min to L/h conversion
        source = """
        model UnitTest {
            clearance hepatic: 500_mL/min
        }
        """

        params = compile_model(source)
        # 500 mL/min = 500 * 0.06 L/h = 30 L/h
        @test params.clearance_hepatic ≈ 30.0
    end

    @testset "Comments and Whitespace" begin
        source = """
        // This is a line comment
        model CommentTest {
            /* This is a
               block comment */
            clearance hepatic: 10.0_L/h  // inline comment
        }
        """

        ast = parse_medlang(source)
        @test ast.models[1].name == "CommentTest"
    end

    @testset "Version and Info" begin
        @test MedLang.version() isa VersionNumber
        @test contains(MedLang.info(), "MedLang DSL")
        @test contains(MedLang.info(), "github.com/agourakis82/medlang")
    end

end

println("\n✅ All MedLang DSL tests passed!")
