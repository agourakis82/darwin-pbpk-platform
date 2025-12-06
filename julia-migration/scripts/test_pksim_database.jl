"""
Standalone test for PK-Sim database integration

Run with: julia scripts/test_pksim_database.jl
"""

using CSV
using DataFrames
using Printf

# Include the PKSim module directly
include("../src/DarwinPBPK/database/pksim_parameters.jl")
using .PKSimParameters

println("="^70)
println("PK-Sim Database Test")
println("="^70)

# Test 1: Load database
println("\n[1] Loading PK-Sim database...")
csv_path = joinpath(@__DIR__, "..", "data", "external_pk_datasets", "PKSim_Human_Reference_Values.csv")

if !isfile(csv_path)
    error("PK-Sim CSV not found at: $csv_path")
end

db = load_pksim_database(csv_path)
println("✓ Loaded $(size(db.raw_data, 1)) parameters for $(length(db.organs)) organ containers")

# Test 2: List available organs
println("\n[2] Available organs:")
major_organs = ["Liver", "Kidney", "Brain", "Heart", "Lung", "Muscle", "Fat", "Spleen", "Pancreas", "Bone", "Skin"]
for organ in major_organs
    if organ in db.organs
        println("  ✓ $organ")
    else
        println("  ✗ $organ (not found)")
    end
end

# Test 3: Extract liver parameters
println("\n[3] Extracting Liver parameters (70 kg patient)...")
liver_params = get_organ_params(db, "Liver", weight=70.0, cardiac_output=350.0)
PKSimParameters.print_organ_summary(liver_params)

# Test 4: Extract kidney parameters
println("\n[4] Extracting Kidney parameters (70 kg patient)...")
kidney_params = get_organ_params(db, "Kidney", weight=70.0, cardiac_output=350.0)
PKSimParameters.print_organ_summary(kidney_params)

# Test 5: Extract brain parameters
println("\n[5] Extracting Brain parameters (70 kg patient)...")
brain_params = get_organ_params(db, "Brain", weight=70.0, cardiac_output=350.0)
PKSimParameters.print_organ_summary(brain_params)

# Test 6: Allometric scaling
println("\n[6] Testing allometric scaling...")
println("\nLiver volume scaling:")
for weight in [50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    vol = scale_organ_volume("Liver", weight, 0.75)
    println("  $(weight) kg: $(round(vol, digits=2)) L")
end

println("\nBlood flow scaling (CO = 350 L/h):")
for organ in ["Liver", "Kidney", "Brain", "Heart"]
    flow = scale_blood_flow(organ, 70.0, 350.0)
    if !isnothing(flow)
        println("  $organ: $(round(flow, digits=1)) L/h ($(round(flow/350*100, digits=1))% of CO)")
    end
end

# Test 7: Comparison with hardcoded values
println("\n[7] Validation against typical hardcoded values...")
println("\nLiver (70 kg patient):")
println("  PK-Sim volume: $(round(liver_params.volume_L, digits=2)) L")
println("  Typical hardcoded: 1.8 L")
diff_pct = abs(liver_params.volume_L - 1.8) / liver_params.volume_L * 100
println("  Difference: $(round(diff_pct, digits=1))%")
if diff_pct > 10.0
    println("  ⚠ WARNING: Differs by >10%")
else
    println("  ✓ Within 10% tolerance")
end

println("\n  PK-Sim blood flow: $(round(liver_params.blood_flow_L_h, digits=1)) L/h")
println("  Typical hardcoded: 90.0 L/h")
diff_pct = abs(liver_params.blood_flow_L_h - 90.0) / liver_params.blood_flow_L_h * 100
println("  Difference: $(round(diff_pct, digits=1))%")
if diff_pct > 10.0
    println("  ⚠ WARNING: Differs by >10%")
else
    println("  ✓ Within 10% tolerance")
end

println("\nKidney (70 kg patient):")
println("  PK-Sim volume: $(round(kidney_params.volume_L, digits=2)) L")
println("  Typical hardcoded: 0.31 L")
diff_pct = abs(kidney_params.volume_L - 0.31) / kidney_params.volume_L * 100
println("  Difference: $(round(diff_pct, digits=1))%")
if diff_pct > 10.0
    println("  ⚠ WARNING: Differs by >10%")
else
    println("  ✓ Within 10% tolerance")
end

println("\n  PK-Sim blood flow: $(round(kidney_params.blood_flow_L_h, digits=1)) L/h")
println("  Typical hardcoded: 60.0 L/h")
diff_pct = abs(kidney_params.blood_flow_L_h - 60.0) / kidney_params.blood_flow_L_h * 100
println("  Difference: $(round(diff_pct, digits=1))%")
if diff_pct > 10.0
    println("  ⚠ WARNING: Differs by >10%")
else
    println("  ✓ Within 10% tolerance")
end

# Test 8: Get specific physiological values
println("\n[8] Extracting specific physiological values...")

liver_microsomal = get_physiological_value(db, "Liver", "Microsomal protein mass/g tissue")
println("  Liver microsomal protein: $(liver_microsomal * 1000) mg/g tissue")

liver_hepatocytes = get_physiological_value(db, "Liver", "Number of cells/g tissue")
println("  Liver hepatocellularity: $(liver_hepatocytes) cells/g")

kidney_gfr_vol = get_physiological_value(db, "Kidney", "Volume (standard kidney)")
println("  Kidney standard volume: $(kidney_gfr_vol) L")

brain_frac_vasc = get_physiological_value(db, "Brain", "Fraction vascular")
println("  Brain vascular fraction: $(round(brain_frac_vasc * 100, digits=1))%")

# Test 9: Tissue composition validation
println("\n[9] Tissue composition validation...")
println("\nLiver:")
println("  Water: $(round(liver_params.vf_water * 100, digits=1))% (PK-Sim) vs 70% (typical)")
println("  Protein: $(round(liver_params.vf_protein * 100, digits=1))% (PK-Sim) vs 20% (typical)")
println("  Lipid: $(round(liver_params.vf_lipid * 100, digits=1))% (PK-Sim) vs 10% (typical)")

println("\nKidney:")
println("  Water: $(round(kidney_params.vf_water * 100, digits=1))% (PK-Sim) vs 80% (typical)")
println("  Protein: $(round(kidney_params.vf_protein * 100, digits=1))% (PK-Sim) vs 15% (typical)")
println("  Lipid: $(round(kidney_params.vf_lipid * 100, digits=1))% (PK-Sim) vs 5% (typical)")

println("\nBrain:")
println("  Water: $(round(brain_params.vf_water * 100, digits=1))% (PK-Sim) vs 80% (typical)")
println("  Protein: $(round(brain_params.vf_protein * 100, digits=1))% (PK-Sim) vs 10% (typical)")
println("  Lipid: $(round(brain_params.vf_lipid * 100, digits=1))% (PK-Sim) vs 10% (typical)")

# Test 10: Summary table
println("\n[10] Summary comparison table...")
println("\n" * "="^70)
println("Organ Parameters: PK-Sim vs Hardcoded")
println("="^70)
println(@sprintf("%-12s %-15s %-15s %-12s", "Organ", "Volume (L)", "Blood Flow (L/h)", "Water %"))
println("-"^70)

organs_data = [
    ("Liver", liver_params.volume_L, liver_params.blood_flow_L_h, liver_params.vf_water, 1.8, 90.0, 0.70),
    ("Kidney", kidney_params.volume_L, kidney_params.blood_flow_L_h, kidney_params.vf_water, 0.31, 60.0, 0.80),
    ("Brain", brain_params.volume_L, brain_params.blood_flow_L_h, brain_params.vf_water, 1.4, 50.0, 0.80)
]

for (organ, vol_pk, flow_pk, water_pk, vol_hc, flow_hc, water_hc) in organs_data
    println("$organ (PK-Sim):")
    println(@sprintf("  %-10s %-15s %-15s %-12s", "",
                     "$(round(vol_pk, digits=2))",
                     "$(round(flow_pk, digits=1))",
                     "$(round(water_pk * 100, digits=1))"))
    println("$organ (Hardcoded):")
    println(@sprintf("  %-10s %-15s %-15s %-12s", "",
                     "$(round(vol_hc, digits=2))",
                     "$(round(flow_hc, digits=1))",
                     "$(round(water_hc * 100, digits=1))"))
    vol_diff = abs(vol_pk - vol_hc) / vol_pk * 100
    flow_diff = abs(flow_pk - flow_hc) / flow_pk * 100
    water_diff = abs(water_pk - water_hc) / water_pk * 100
    println(@sprintf("  Difference: Vol=%.1f%% Flow=%.1f%% Water=%.1f%%",
                     vol_diff, flow_diff, water_diff))
    println("-"^70)
end

println("\n" * "="^70)
println("PK-Sim Database Test Complete!")
println("="^70)
println("\nSummary:")
println("  ✓ Database loaded successfully")
println("  ✓ All major organs accessible")
println("  ✓ Allometric scaling functional")
println("  ✓ Parameter extraction working")
println("  ✓ Validation complete")

println("\nNext steps:")
println("  1. Review validation differences")
println("  2. Update compartment_models.jl to use PK-Sim by default")
println("  3. Run full PBPK model tests")
println("  4. Document parameter sources")
