#!/usr/bin/env julia
"""
Test Suite for White Blood Cell Compartment Implementation

Tests:
1. WBC compartment creation (normal state)
2. Pathology models (leukemia, sepsis)
3. Fractal parameter correction
4. Integration with FractalBlood
5. Parameter calculations
"""

using Pkg

# Activate the project environment
project_dir = joinpath(@__DIR__, "..")
Pkg.activate(project_dir)

# Add src to load path
push!(LOAD_PATH, joinpath(project_dir, "src"))

using Test

# Import modules
include(joinpath(project_dir, "src", "DarwinPBPK.jl"))
using .DarwinPBPK
using .DarwinPBPK.PatientProfile
using .DarwinPBPK.WhiteBloodCells
using .DarwinPBPK.FractalBlood

println("=" ^ 80)
println("🧪 TESTING WBC IMPLEMENTATION")
println("=" ^ 80)
println()

# ============================================================================
# TEST 1: Create Normal WBC Compartment
# ============================================================================

println("📋 TEST 1: Creating Normal WBC Compartment...")

try
    # Create a test patient
    patient = PatientProfile.create_patient(
        age=45.0,
        sex="M",
        weight=70.0,
        height=175.0
    )
    
    # Create normal WBC compartment
    wbc_normal = WhiteBloodCells.create_WBC_compartment(
        patient,
        pathology="normal",
        pathology_severity=0.0
    )
    
    @test wbc_normal.pathology == "normal"
    @test wbc_normal.pathology_severity == 0.0
    @test wbc_normal.neutrophils.cell_count > 0
    @test wbc_normal.lymphocytes_T.cell_count > 0
    @test wbc_normal.total_WBC_count > 0
    @test wbc_normal.total_volume_fraction > 0.0
    @test wbc_normal.total_volume_fraction < 0.1  # Should be ~1.3%
    
    println("✅ TEST 1 PASSED: Normal WBC compartment created successfully")
    println("   - Total WBC count: $(round(wbc_normal.total_WBC_count/1e6, digits=1)) × 10⁶ cells/L")
    println("   - Volume fraction: $(round(wbc_normal.total_volume_fraction * 100, digits=2))%")
    println()
    
catch e
    println("❌ TEST 1 FAILED: $e")
    rethrow(e)
end

# ============================================================================
# TEST 2: Leukemia Pathology
# ============================================================================

println("📋 TEST 2: Testing Leukemia Pathology...")

try
    patient = PatientProfile.create_patient(age=45.0, sex="M", weight=70.0, height=175.0)
    
    # Normal state
    wbc_normal = WhiteBloodCells.create_WBC_compartment(patient, pathology="normal")
    
    # Leukemia (severity = 1.0)
    wbc_leukemia = WhiteBloodCells.create_WBC_compartment(
        patient,
        pathology="leukemia",
        pathology_severity=1.0
    )
    
    # Check that lymphocytes increased dramatically
    @test wbc_leukemia.lymphocytes_T.cell_count > wbc_normal.lymphocytes_T.cell_count * 50
    @test wbc_leukemia.total_WBC_count > wbc_normal.total_WBC_count * 10
    @test wbc_leukemia.total_volume_fraction > wbc_normal.total_volume_fraction * 5
    
    println("✅ TEST 2 PASSED: Leukemia pathology model working")
    println("   - Normal lymphocytes T: $(round(wbc_normal.lymphocytes_T.cell_count/1e6, digits=1)) × 10⁶/L")
    println("   - Leukemia lymphocytes T: $(round(wbc_leukemia.lymphocytes_T.cell_count/1e6, digits=1)) × 10⁶/L")
    println("   - Increase: $(round(wbc_leukemia.lymphocytes_T.cell_count / wbc_normal.lymphocytes_T.cell_count, digits=1))×")
    println()
    
catch e
    println("❌ TEST 2 FAILED: $e")
    rethrow(e)
end

# ============================================================================
# TEST 3: Sepsis Pathology
# ============================================================================

println("📋 TEST 3: Testing Sepsis Pathology...")

try
    patient = PatientProfile.create_patient(age=45.0, sex="M", weight=70.0, height=175.0)
    
    wbc_normal = WhiteBloodCells.create_WBC_compartment(patient, pathology="normal")
    wbc_sepsis = WhiteBloodCells.create_WBC_compartment(
        patient,
        pathology="sepsis",
        pathology_severity=1.0
    )
    
    # Check neutrophil increase
    @test wbc_sepsis.neutrophils.cell_count > wbc_normal.neutrophils.cell_count * 5
    # Check lymphopenia
    @test wbc_sepsis.lymphocytes_T.cell_count < wbc_normal.lymphocytes_T.cell_count * 0.5
    
    println("✅ TEST 3 PASSED: Sepsis pathology model working")
    println("   - Normal neutrophils: $(round(wbc_normal.neutrophils.cell_count/1e6, digits=1)) × 10⁶/L")
    println("   - Sepsis neutrophils: $(round(wbc_sepsis.neutrophils.cell_count/1e6, digits=1)) × 10⁶/L")
    println("   - Increase: $(round(wbc_sepsis.neutrophils.cell_count / wbc_normal.neutrophils.cell_count, digits=1))×")
    println()
    
catch e
    println("❌ TEST 3 FAILED: $e")
    rethrow(e)
end

# ============================================================================
# TEST 4: Fractal Parameter Correction
# ============================================================================

println("📋 TEST 4: Testing Fractal Parameter Correction...")

try
    patient = PatientProfile.create_patient(age=45.0, sex="M", weight=70.0, height=175.0)
    
    # Create WBC with custom fractal parameters
    fractal_params = Dict(
        "neutrophil" => Dict(
            "df_edge" => 1.5,  # Lower df = simpler membrane
            "df_distribution" => 1.3
        ),
        "lymphocyte_T" => Dict(
            "df_edge" => 1.8,  # Higher df = more complex
            "df_distribution" => 1.7
        )
    )
    
    wbc_fractal = WhiteBloodCells.create_WBC_compartment(
        patient,
        pathology="normal",
        fractal_params=fractal_params
    )
    
    # Check that fractal parameters were set
    @test wbc_fractal.neutrophils.fractal_dimension_edge == 1.5
    @test wbc_fractal.lymphocytes_T.fractal_dimension_edge == 1.8
    
    # Test fractal-corrected parameters for a drug (azithromycin)
    pk_params = WhiteBloodCells.get_fractal_corrected_parameters(
        wbc_fractal,
        "azithromycin",
        drug_pKa=8.7,  # Basic drug
        drug_logP=4.02  # Lipophilic
    )
    
    @test haskey(pk_params, "neutrophil")
    @test haskey(pk_params["neutrophil"], "partition_coefficient")
    @test haskey(pk_params["neutrophil"], "internalization_rate")
    
    println("✅ TEST 4 PASSED: Fractal parameter correction working")
    println("   - Neutrophil partition: $(round(pk_params["neutrophil"]["partition_coefficient"], digits=2))")
    println("   - Neutrophil internalization: $(round(pk_params["neutrophil"]["internalization_rate"], digits=2)) 1/h")
    println()
    
catch e
    println("❌ TEST 4 FAILED: $e")
    rethrow(e)
end

# ============================================================================
# TEST 5: Integration with FractalBlood
# ============================================================================

println("📋 TEST 5: Testing Integration with FractalBlood...")

try
    patient = PatientProfile.create_patient(age=45.0, sex="M", weight=70.0, height=175.0)
    
    wbc_compartment = WhiteBloodCells.create_WBC_compartment(patient, pathology="normal")
    
    # Create WBC phases for FractalBlood
    wbc_phases = WhiteBloodCells.create_WBC_phases_for_fractal_blood(wbc_compartment)
    
    @test length(wbc_phases) == 7  # 7 subpopulations
    @test all(phase -> phase isa FractalBlood.BloodPhase, wbc_phases)
    
    # Check that each phase has correct structure
    for phase in wbc_phases
        @test phase.volume_fraction > 0.0
        @test phase.volume_fraction < 1.0
        @test phase.velocity_factor > 0.0
        @test phase.partition_coeff > 0.0
    end
    
    println("✅ TEST 5 PASSED: FractalBlood integration working")
    println("   - Created $(length(wbc_phases)) WBC phases")
    println("   - Total WBC volume fraction: $(round(sum(p.volume_fraction for p in wbc_phases) * 100, digits=2))%")
    println()
    
catch e
    println("❌ TEST 5 FAILED: $e")
    rethrow(e)
end

# ============================================================================
# TEST 6: Partition Coefficient Calculation
# ============================================================================

println("📋 TEST 6: Testing Partition Coefficient Calculation...")

try
    patient = PatientProfile.create_patient(age=45.0, sex="M", weight=70.0, height=175.0)
    wbc = WhiteBloodCells.create_WBC_compartment(patient, pathology="normal")
    
    # Test with different drugs
    drugs = [
        ("azithromycin", 8.7, 4.02),  # Basic, lipophilic
        ("amoxicillin", 2.7, 0.87),   # Acid, hydrophilic
        ("chloroquine", 10.2, 4.56)   # Very basic, very lipophilic
    ]
    
    for (drug_name, pKa, logP) in drugs
        partition = WhiteBloodCells.calculate_partition_coefficient(
            wbc.neutrophils,
            drug_name,
            pKa,
            logP,
            use_fractal_correction=true
        )
        
        @test partition > 0.0
        @test isfinite(partition)
        
        # Basic drugs should have higher partition (ion trapping)
        if pKa > 7.0
            @test partition > 1.0
        end
    end
    
    println("✅ TEST 6 PASSED: Partition coefficient calculation working")
    println("   - Tested $(length(drugs)) different drugs")
    println()
    
catch e
    println("❌ TEST 6 FAILED: $e")
    rethrow(e)
end

# ============================================================================
# TEST 7: Pathology Multipliers
# ============================================================================

println("📋 TEST 7: Testing Pathology Multipliers...")

try
    multipliers = WhiteBloodCells.get_pathology_multipliers("leukemia", 1.0)
    
    @test multipliers["neutrophil"] < 1.0  # Suppressed
    @test multipliers["lymphocyte_T"] > 50.0  # Massive increase
    @test multipliers["lymphocyte_B"] > 50.0  # Massive increase
    
    multipliers_sepsis = WhiteBloodCells.get_pathology_multipliers("sepsis", 1.0)
    @test multipliers_sepsis["neutrophil"] > 5.0  # Increased
    @test multipliers_sepsis["lymphocyte_T"] < 0.5  # Lymphopenia
    
    println("✅ TEST 7 PASSED: Pathology multipliers working correctly")
    println()
    
catch e
    println("❌ TEST 7 FAILED: $e")
    rethrow(e)
end

# ============================================================================
# SUMMARY
# ============================================================================

println("=" ^ 80)
println("✅ ALL TESTS PASSED!")
println("=" ^ 80)
println()
println("📊 SUMMARY:")
println("   - WBC compartment creation: ✅")
println("   - Leukemia pathology: ✅")
println("   - Sepsis pathology: ✅")
println("   - Fractal parameter correction: ✅")
println("   - FractalBlood integration: ✅")
println("   - Partition coefficient calculation: ✅")
println("   - Pathology multipliers: ✅")
println()
println("🎯 Implementation is working correctly!")

