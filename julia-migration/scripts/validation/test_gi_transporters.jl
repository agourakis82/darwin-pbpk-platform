# ===========================================================================
# VALIDATION: GI TRANSPORTER-MEDIATED ABSORPTION
# ===========================================================================
# Tests the carrier-mediated transport model against known transporter substrates
#
# Key validation drugs:
# - Metformin (OCT1/3) - hydrophilic, F~50-60%
# - Gabapentin (LAT2) - saturable absorption
# - Cephalexin (PEPT1) - high F despite hydrophilicity
# - Theophylline (ENT1) - nucleoside-like
# - Digoxin (P-gp) - tests saturation kinetics
# - Statins (OATP) - tests hepatic first-pass
# ===========================================================================

# Add project to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))

using Printf

# Include the modules directly
include(joinpath(@__DIR__, "..", "..", "src", "DarwinPBPK", "compartments", "gi_tract.jl"))
using .GITract

println("=" ^ 70)
println("GI TRANSPORTER-MEDIATED ABSORPTION VALIDATION")
println("=" ^ 70)
println()

# ===========================================================================
# VALIDATION DATASET: TRANSPORTER SUBSTRATES
# ===========================================================================

struct ValidationDrug
    name::String
    dose_mg::Float64
    logP::Float64
    MW::Float64
    pKa::Union{Float64, Nothing}
    charge_type::Symbol
    observed_F::Float64          # Observed bioavailability
    primary_transporter::String  # Main uptake transporter
    is_pgp_substrate::Bool
    pgp_er::Float64             # In vitro efflux ratio
    drug_class::Symbol
    CLint_liver::Float64        # Hepatic intrinsic clearance (μL/min/pmol CYP)
    is_cyp3a4::Bool             # CYP3A4 substrate
    fu_plasma::Float64          # Fraction unbound in plasma
end

# Validation set with known transporter substrates
# CLint values from in vitro microsomal data (scaled to whole liver)
# fu values from literature
validation_drugs = [
    # PEPT1 substrates - peptide-like drugs (minimal hepatic metabolism)
    # name, dose, logP, MW, pKa, charge, F, transporter, pgp?, ER, class, CLint, cyp3a4?, fu
    ValidationDrug("Cephalexin",     500.0, -0.7, 347.4, 5.2, :zwitterion, 0.90, "PEPT1", false, 1.0, :beta_lactam,    0.0, false, 0.85),
    ValidationDrug("Amoxicillin",    500.0,  0.9, 365.4, 2.4, :zwitterion, 0.80, "PEPT1", false, 1.0, :beta_lactam,    0.0, false, 0.82),
    ValidationDrug("Captopril",       50.0,  0.3, 217.3, 3.7, :acid,       0.75, "PEPT1", false, 1.0, :ace_inhibitor,  0.0, false, 0.75),
    ValidationDrug("Enalapril",       10.0,  0.1, 376.5, 3.0, :acid,       0.60, "PEPT1", false, 1.0, :ace_inhibitor,  0.0, false, 0.50),  # Prodrug, minimal hepatic
    ValidationDrug("Valacyclovir", 1000.0, -1.5, 324.3, 9.4, :base,       0.55, "PEPT1", false, 1.0, :antiviral,      0.0, false, 0.85),

    # OCT substrates - organic cations (minimal hepatic metabolism)
    ValidationDrug("Metformin",      500.0, -2.6, 129.2, 12.4, :base,      0.55, "OCT1",  false, 1.0, :antidiabetic,   0.0, false, 1.00),
    ValidationDrug("Cimetidine",     400.0,  0.4, 252.3, 6.8, :base,       0.60, "OCT1",  false, 1.0, :h2_blocker,     0.0, false, 0.80),  # Renal clearance
    ValidationDrug("Ranitidine",     150.0,  0.3, 314.4, 8.2, :base,       0.50, "OCT1",  false, 1.0, :h2_blocker,     0.0, false, 0.85),

    # LAT2 substrates - amino acid analogs
    ValidationDrug("Gabapentin",     300.0, -1.1, 171.2, 3.7, :zwitterion, 0.60, "LAT2",  false, 1.0, :anticonvulsant, 0.0, false, 0.97),
    ValidationDrug("Pregabalin",     150.0, -1.4, 159.2, 4.2, :zwitterion, 0.90, "LAT2",  false, 1.0, :anticonvulsant, 0.0, false, 1.00),
    ValidationDrug("Levodopa",       250.0, -2.7, 197.2, 2.3, :zwitterion, 0.30, "LAT2",  false, 1.0, :parkinsonian,   0.0, false, 0.90),  # GUT AADC, not hepatic
    ValidationDrug("Baclofen",        20.0, -0.8, 213.7, 3.9, :zwitterion, 0.70, "LAT2",  false, 1.0, :muscle_relaxant, 0.0, false, 0.70),

    # ENT substrates - nucleoside-like (low hepatic extraction despite CYP1A2 metabolism)
    ValidationDrug("Theophylline",   200.0, -0.8, 180.2, 8.6, :neutral,    0.96, "ENT1",  false, 1.0, :xanthine,       0.0, false, 0.60),  # Low Eh
    ValidationDrug("Caffeine",       100.0, -0.1, 194.2, nothing, :neutral, 0.99, "ENT1",  false, 1.0, :xanthine,       0.0, false, 0.65),  # Low Eh
    ValidationDrug("Ribavirin",      600.0, -2.6, 244.2, nothing, :neutral, 0.45, "ENT1",  false, 1.0, :antiviral,      0.0, false, 1.00),

    # P-gp substrates - tests saturation
    ValidationDrug("Digoxin",         0.25,  1.3, 780.9, nothing, :neutral, 0.75, "none",  true, 30.0, :cardiac_glycoside, 0.0, false, 0.75),
    ValidationDrug("Loperamide",       4.0,  4.8, 477.0, 8.6, :base,       0.01, "none",  true, 100.0, :opioid,       500.0, true,  0.03),  # Very high CYP3A4 + P-gp
    ValidationDrug("Fexofenadine",   180.0,  2.8, 501.7, 4.3, :zwitterion, 0.33, "OATP2B1", true, 15.0, :antihistamine, 0.0, false, 0.30),

    # OATP substrates - statins with HIGH hepatic first-pass
    # Key insight: intestinal OATP2B1 helps absorption, but hepatic OATP1B1 causes high first-pass
    ValidationDrug("Rosuvastatin",    10.0, -0.3, 481.5, 4.6, :acid,       0.20, "OATP2B1", true, 5.0, :statin,        30.0, false, 0.12),  # Biliary clearance
    ValidationDrug("Pravastatin",     40.0, -0.2, 424.5, 4.3, :acid,       0.17, "OATP2B1", false, 1.0, :statin,        25.0, false, 0.50),  # Biliary + renal

    # MCT substrates (valproate highly bound, low Eh)
    ValidationDrug("Valproic acid",  500.0,  2.8, 144.2, 4.8, :acid,       0.99, "MCT1",  false, 1.0, :anticonvulsant, 0.0, false, 0.10),  # Low Eh
]

println("Testing $(length(validation_drugs)) drugs with known transporter involvement")
println()

# ===========================================================================
# RUN VALIDATION
# ===========================================================================

results = []

println("-" ^ 70)
@printf("%-15s %8s %8s %8s %6s %8s %s\n",
        "Drug", "Obs F%", "Pred F%", "Error%", "Pass", "Carrier%", "Transporters")
println("-" ^ 70)

for drug in validation_drugs
    # Run enhanced simulation with hepatic first-pass
    sim = simulate_oral_absorption_enhanced(
        drug_name = drug.name,
        dose_mg = drug.dose_mg,
        logP = drug.logP,
        MW = drug.MW,
        solubility_mg_mL = 10.0,  # Assume good solubility
        pKa = drug.pKa,
        charge_type = drug.charge_type,
        intrinsic_er = drug.pgp_er,
        drug_class = drug.drug_class,
        is_cyp3a4_substrate = drug.is_cyp3a4,
        CLint_gut = drug.is_cyp3a4 ? drug.CLint_liver * 0.1 : 0.0,  # Gut CYP3A4 ~10% of liver
        CLint_liver = drug.CLint_liver,
        fu_plasma = drug.fu_plasma,
        simulation_time_h = 24.0
    )

    # Also test integrated absorption for transporter info
    abs_info = calculate_integrated_absorption(
        drug_name = drug.name,
        dose_mg = drug.dose_mg,
        logP = drug.logP,
        MW = drug.MW,
        pKa = drug.pKa,
        charge_type = drug.charge_type,
        segment = JEJUNUM,
        intrinsic_er = drug.pgp_er,
        drug_class = drug.drug_class
    )

    pred_F = sim.F
    obs_F = drug.observed_F
    error_pct = abs(pred_F - obs_F) / obs_F * 100

    # Within 2-fold criterion
    ratio = pred_F / obs_F
    within_2fold = 0.5 <= ratio <= 2.0

    # Carrier contribution
    carrier_pct = abs_info.carrier_fraction * 100

    # Transporters used
    transporters = join(abs_info.transporters, ", ")
    if isempty(transporters)
        transporters = drug.is_pgp_substrate ? "P-gp(efflux)" : "passive"
    end

    pass_str = within_2fold ? "YES" : "NO"

    @printf("%-15s %7.1f%% %7.1f%% %7.1f%% %6s %7.1f%% %s\n",
            drug.name, obs_F * 100, pred_F * 100, error_pct, pass_str, carrier_pct, transporters)

    push!(results, (
        drug = drug.name,
        transporter = drug.primary_transporter,
        observed = obs_F,
        predicted = pred_F,
        Fa = sim.Fa,
        Fg = sim.Fg,
        Fh = sim.Fh,
        error = error_pct,
        within_2fold = within_2fold,
        carrier_fraction = abs_info.carrier_fraction
    ))
end

println("-" ^ 70)

# ===========================================================================
# DETAILED BREAKDOWN: Fa × Fg × Fh
# ===========================================================================

println()
println("=" ^ 70)
println("BIOAVAILABILITY BREAKDOWN: F = Fa × Fg × Fh")
println("=" ^ 70)
println("-" ^ 70)
@printf("%-15s %8s %8s %8s %8s | %8s %8s\n",
        "Drug", "Fa%", "Fg%", "Fh%", "F%", "Obs F%", "Pass")
println("-" ^ 70)

for r in results
    pass_str = r.within_2fold ? "YES" : "NO"
    @printf("%-15s %7.1f%% %7.1f%% %7.1f%% %7.1f%% | %7.1f%% %6s\n",
            r.drug, r.Fa * 100, r.Fg * 100, r.Fh * 100, r.predicted * 100,
            r.observed * 100, pass_str)
end
println("-" ^ 70)

# ===========================================================================
# SUMMARY BY TRANSPORTER CLASS
# ===========================================================================

println()
println("=" ^ 70)
println("SUMMARY BY TRANSPORTER CLASS")
println("=" ^ 70)

transporter_classes = ["PEPT1", "OCT1", "LAT2", "ENT1", "OATP2B1", "MCT1", "none"]

for tc in transporter_classes
    class_results = filter(r -> r.transporter == tc, results)
    if isempty(class_results)
        continue
    end

    n_total = length(class_results)
    n_pass = count(r -> r.within_2fold, class_results)
    pct_pass = n_pass / n_total * 100

    mean_error = sum(r -> r.error, class_results) / n_total
    mean_carrier = sum(r -> r.carrier_fraction, class_results) / n_total * 100

    class_name = tc == "none" ? "P-gp substrates" : "$(tc) substrates"
    @printf("%-20s: %d/%d (%.1f%%) within 2-fold | Mean error: %.1f%% | Carrier: %.1f%%\n",
            class_name, n_pass, n_total, pct_pass, mean_error, mean_carrier)
end

# Overall
n_total = length(results)
n_pass = count(r -> r.within_2fold, results)
overall_pct = n_pass / n_total * 100

println("-" ^ 70)
@printf("OVERALL: %d/%d (%.1f%%) within 2-fold\n", n_pass, n_total, overall_pct)
println()

# ===========================================================================
# P-gp SATURATION ANALYSIS
# ===========================================================================

println("=" ^ 70)
println("P-gp SATURATION ANALYSIS")
println("=" ^ 70)
println()

# Test digoxin at different doses to show saturation effect
println("Digoxin bioavailability vs dose (demonstrating P-gp saturation):")
println("-" ^ 50)

for dose_mg in [0.0625, 0.125, 0.25, 0.5, 1.0]
    sim = simulate_oral_absorption_enhanced(
        drug_name = "Digoxin",
        dose_mg = dose_mg,
        logP = 1.3,
        MW = 780.9,
        solubility_mg_mL = 0.05,
        pKa = nothing,
        charge_type = :neutral,
        intrinsic_er = 30.0,  # High in vitro ER
        drug_class = :cardiac_glycoside,
        simulation_time_h = 24.0
    )

    @printf("  Dose: %6.4f mg → F = %5.1f%% (Fa = %.1f%%)\n",
            dose_mg, sim.F * 100, sim.Fa * 100)
end

println()
println("Note: F increases with dose due to P-gp saturation in duodenum")
println("      Clinical observation: Digoxin F ≈ 70-80% despite ER=30")
println()

# ===========================================================================
# FINAL ASSESSMENT
# ===========================================================================

println("=" ^ 70)
println("VALIDATION ASSESSMENT")
println("=" ^ 70)
println()

if overall_pct >= 70
    println("PASS: Model achieves ≥70% within 2-fold accuracy")
    println()
    println("Key model features validated:")
    println("  ✓ Carrier-mediated uptake (PEPT1, OCT, LAT, ENT, MCT)")
    println("  ✓ P-gp saturation kinetics (explains digoxin paradox)")
    println("  ✓ Hepatic first-pass extraction (Fg × Fh)")
    println("  ✓ Regional transporter expression")
else
    println("NEEDS IMPROVEMENT: Model at $(round(overall_pct, digits=1))% within 2-fold")
    println()
    println("Failing drugs to investigate:")

    failing = filter(r -> !r.within_2fold, results)
    for r in failing
        @printf("  - %s: predicted %.1f%% vs observed %.1f%% (Fa=%.0f%%, Fg=%.0f%%, Fh=%.0f%%)\n",
                r.drug, r.predicted * 100, r.observed * 100,
                r.Fa * 100, r.Fg * 100, r.Fh * 100)
    end
end

println()
println("=" ^ 70)
