#!/usr/bin/env julia
"""
Basic WBC Test - Tests WBC modeling without image analysis dependencies
"""

using Pkg
project_dir = joinpath(@__DIR__, "..")
Pkg.activate(project_dir)
push!(LOAD_PATH, joinpath(project_dir, "src"))

using Test

# Load dependencies in order
include(joinpath(project_dir, "src", "DarwinPBPK", "patient_profile.jl"))
include(joinpath(project_dir, "src", "DarwinPBPK", "fractal_blood.jl"))
include(joinpath(project_dir, "src", "DarwinPBPK", "compartments", "white_blood_cells.jl"))

using .DarwinPBPK.PatientProfile
using .DarwinPBPK.WhiteBloodCells

println("=" ^ 80)
println("🧪 TESTING WBC IMPLEMENTATION (Basic)")
println("=" ^ 80)
println()

# ============================================================================
# TEST 1: Create Normal WBC Compartment
# ============================================================================

println("📋 TEST 1: Creating Normal WBC Compartment...")

try
    patient = PatientProfile.create_patient(
        age=45.0,
        sex="M",
        weight=70.0,
        height=175.0
    )
    
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
    @test wbc_normal.total_volume_fraction < 0.1
    
    println("✅ TEST 1 PASSED")
    println("   - Total WBC: $(round(wbc_normal.total_WBC_count/1e6, digits=1)) × 10⁶/L")
    println("   - Volume: $(round(wbc_normal.total_volume_fraction * 100, digits=2))%")
    println()
    
catch e
    println("❌ TEST 1 FAILED: $e")
    rethrow(e)
end

# ============================================================================
# TEST 2: Leukemia
# ============================================================================

println("📋 TEST 2: Testing Leukemia...")

try
    patient = PatientProfile.create_patient(age=45.0, sex="M", weight=70.0, height=175.0)
    wbc_normal = WhiteBloodCells.create_WBC_compartment(patient, pathology="normal")
    wbc_leukemia = WhiteBloodCells.create_WBC_compartment(
        patient,
        pathology="leukemia",
        pathology_severity=1.0
    )
    
    @test wbc_leukemia.lymphocytes_T.cell_count > wbc_normal.lymphocytes_T.cell_count * 50
    @test wbc_leukemia.total_WBC_count > wbc_normal.total_WBC_count * 10
    
    println("✅ TEST 2 PASSED")
    println("   - Lymphocytes T increase: $(round(wbc_leukemia.lymphocytes_T.cell_count / wbc_normal.lymphocytes_T.cell_count, digits=1))×")
    println()
    
catch e
    println("❌ TEST 2 FAILED: $e")
    rethrow(e)
end

# ============================================================================
# TEST 3: Sepsis
# ============================================================================

println("📋 TEST 3: Testing Sepsis...")

try
    patient = PatientProfile.create_patient(age=45.0, sex="M", weight=70.0, height=175.0)
    wbc_normal = WhiteBloodCells.create_WBC_compartment(patient, pathology="normal")
    wbc_sepsis = WhiteBloodCells.create_WBC_compartment(
        patient,
        pathology="sepsis",
        pathology_severity=1.0
    )
    
    @test wbc_sepsis.neutrophils.cell_count > wbc_normal.neutrophils.cell_count * 5
    @test wbc_sepsis.lymphocytes_T.cell_count < wbc_normal.lymphocytes_T.cell_count * 0.5
    
    println("✅ TEST 3 PASSED")
    println("   - Neutrophils increase: $(round(wbc_sepsis.neutrophils.cell_count / wbc_normal.neutrophils.cell_count, digits=1))×")
    println()
    
catch e
    println("❌ TEST 3 FAILED: $e")
    rethrow(e)
end

# ============================================================================
# TEST 4: Fractal Parameters
# ============================================================================

println("📋 TEST 4: Testing Fractal Parameters...")

try
    patient = PatientProfile.create_patient(age=45.0, sex="M", weight=70.0, height=175.0)
    
    fractal_params = Dict(
        "neutrophil" => Dict("df_edge" => 1.5, "df_distribution" => 1.3),
        "lymphocyte_T" => Dict("df_edge" => 1.8, "df_distribution" => 1.7)
    )
    
    wbc = WhiteBloodCells.create_WBC_compartment(
        patient,
        pathology="normal",
        fractal_params=fractal_params
    )
    
    @test wbc.neutrophils.fractal_dimension_edge == 1.5
    @test wbc.lymphocytes_T.fractal_dimension_edge == 1.8
    
    # Test partition coefficient
    partition = WhiteBloodCells.calculate_partition_coefficient(
        wbc.neutrophils,
        "azithromycin",
        8.7,
        4.02,
        use_fractal_correction=true
    )
    
    @test partition > 0.0
    @test isfinite(partition)
    
    println("✅ TEST 4 PASSED")
    println("   - Neutrophil partition: $(round(partition, digits=2))")
    println()
    
catch e
    println("❌ TEST 4 FAILED: $e")
    rethrow(e)
end

println("=" ^ 80)
println("✅ ALL BASIC TESTS PASSED!")
println("=" ^ 80)

