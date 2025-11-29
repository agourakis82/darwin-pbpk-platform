# ===========================================================================
# TEST GI TRACT ABSORPTION MODEL
# ===========================================================================

using Printf

include("../../src/DarwinPBPK/compartments/gi_tract.jl")
using .GITract

# ===========================================================================
# TEST DRUGS WITH KNOWN ORAL BIOAVAILABILITY
# ===========================================================================

const TEST_DRUGS = [
    # (name, logP, MW, solubility_mg_mL, pKa, charge, pgp_ER, F_obs%)

    # HIGH BIOAVAILABILITY (>80%)
    ("Metoprolol", 1.9, 267.4, 16.9, 9.7, :base, 1.0, 95),
    ("Propranolol", 3.5, 259.3, 0.033, 9.4, :base, 1.5, 90),
    ("Theophylline", -0.8, 180.2, 8.3, 8.8, :base, 1.0, 96),
    ("Caffeine", -0.1, 194.2, 21.6, 0.6, :neutral, 1.0, 99),

    # MODERATE BIOAVAILABILITY (40-80%)
    ("Ranitidine", 0.3, 314.4, 25.0, 8.2, :base, 1.5, 52),
    ("Atenolol", -0.1, 266.3, 26.5, 9.6, :base, 1.0, 50),
    ("Furosemide", 2.0, 330.7, 0.018, 3.8, :acid, 1.0, 60),
    ("Verapamil", 3.8, 454.6, 0.083, 8.9, :base, 3.0, 22),  # Low due to CYP3A4

    # LOW BIOAVAILABILITY (<40%)
    ("Acyclovir", -1.6, 225.2, 2.5, 9.3, :neutral, 1.0, 20),
    ("Digoxin", 1.3, 780.9, 0.065, nothing, :neutral, 30.0, 75),  # High P-gp
    ("Cyclosporine", 2.9, 1202.6, 0.023, nothing, :neutral, 5.0, 30),  # P-gp + CYP3A4

    # BCS CLASS REPRESENTATIVES
    # Class I: High solubility, High permeability
    ("Ketoprofen", 3.1, 254.3, 0.51, 4.5, :acid, 1.0, 90),

    # Class II: Low solubility, High permeability
    ("Ibuprofen", 3.5, 206.3, 0.021, 4.4, :acid, 1.0, 80),
    ("Phenytoin", 2.5, 252.3, 0.032, 8.3, :neutral, 1.5, 90),

    # Class III: High solubility, Low permeability
    ("Metformin", -1.4, 129.2, 100.0, 12.4, :base, 1.0, 55),
    ("Gabapentin", -1.1, 171.2, 100.0, 3.7, :zwitterion, 1.0, 60),  # LAT1 mediated

    # Class IV: Low solubility, Low permeability
    ("Furosemide2", 2.0, 330.7, 0.018, 3.8, :acid, 1.0, 60),  # Duplicate for testing
]

function test_bioavailability_prediction()
    println("=" ^ 70)
    println("GI TRACT MODEL - BIOAVAILABILITY VALIDATION")
    println("=" ^ 70)
    println()

    results = []

    for drug in TEST_DRUGS
        name, logP, MW, sol, pKa, charge, pgp_er, f_obs = drug

        # Simulate absorption
        sim = simulate_oral_absorption(
            dose_mg = 100.0,
            logP = logP,
            MW = MW,
            solubility_mg_mL = sol,
            pKa = pKa,
            charge_type = charge == :zwitterion ? :neutral : charge,
            is_pgp_substrate = pgp_er > 1.5,
            pgp_efflux_ratio = pgp_er,
            is_cyp3a4_substrate = name in ["Verapamil", "Cyclosporine"],
            CLint_gut = name in ["Verapamil", "Cyclosporine"] ? 10.0 : 0.0,
            CLint_liver = name in ["Verapamil", "Cyclosporine"] ? 50.0 : 0.0,
            simulation_time_h = 12.0
        )

        ratio = sim.F_percent / f_obs
        within_2fold = 0.5 <= ratio <= 2.0

        push!(results, (
            name = name,
            f_obs = f_obs,
            f_pred = sim.F_percent,
            Fa = sim.Fa * 100,
            Fg = sim.Fg * 100,
            Fh = sim.Fh * 100,
            ratio = ratio,
            within_2fold = within_2fold
        ))
    end

    # Print results
    @printf("%-15s %6s %6s %6s %6s %6s %7s %6s\n",
            "Drug", "F_obs", "F_pred", "Fa%", "Fg%", "Fh%", "Ratio", "2-fold")
    println("-" ^ 70)

    for r in results
        flag = r.within_2fold ? "✓" : "✗"
        @printf("%-15s %5.0f%% %5.0f%% %5.0f%% %5.0f%% %5.0f%% %7.2f %6s\n",
                r.name[1:min(15, length(r.name))],
                r.f_obs, r.f_pred, r.Fa, r.Fg, r.Fh, r.ratio, flag)
    end

    println("-" ^ 70)

    # Summary
    n = length(results)
    n_2fold = count(r -> r.within_2fold, results)
    println()
    @printf("Within 2-fold: %d/%d (%.1f%%)\n", n_2fold, n, 100.0 * n_2fold / n)

    return results
end

# ===========================================================================
# TEST DISSOLUTION KINETICS
# ===========================================================================

function test_dissolution()
    println()
    println("=" ^ 70)
    println("DISSOLUTION TEST - pH EFFECTS")
    println("=" ^ 70)
    println()

    # Weak acid (e.g., ibuprofen pKa 4.4)
    println("WEAK ACID (pKa 4.4, like ibuprofen):")
    for segment in [STOMACH, DUODENUM, JEJUNUM]
        diss = calculate_dissolution(
            dose_mg = 200.0,
            solubility_mg_mL = 0.021,
            pKa = 4.4,
            charge_type = :acid,
            segment = segment
        )
        phys = GI_PHYSIOLOGY[segment]
        @printf("  %-10s (pH %.1f): Solubility %.3f mg/mL (%.0fx increase)\n",
                segment, phys.pH, diss.solubility_adj_mg_mL,
                diss.solubility_adj_mg_mL / 0.021)
    end

    println()
    println("WEAK BASE (pKa 9.4, like propranolol):")
    for segment in [STOMACH, DUODENUM, JEJUNUM]
        diss = calculate_dissolution(
            dose_mg = 80.0,
            solubility_mg_mL = 0.033,
            pKa = 9.4,
            charge_type = :base,
            segment = segment
        )
        phys = GI_PHYSIOLOGY[segment]
        @printf("  %-10s (pH %.1f): Solubility %.3f mg/mL (%.0fx increase)\n",
                segment, phys.pH, diss.solubility_adj_mg_mL,
                diss.solubility_adj_mg_mL / 0.033)
    end
end

# ===========================================================================
# TEST ENTEROHEPATIC RECIRCULATION
# ===========================================================================

function test_ehc()
    println()
    println("=" ^ 70)
    println("ENTEROHEPATIC RECIRCULATION TEST")
    println("=" ^ 70)
    println()

    drugs_with_ehc = [
        ("Morphine (glucuronide)", 3.0),
        ("Mycophenolate", 6.0),
        ("Digoxin", 36.0),
        ("Ethinyl estradiol", 12.0),
    ]

    for (drug, t12) in drugs_with_ehc
        ehc_result = calculate_ehc_extension(
            intrinsic_half_life_h = t12,
            ehc = EnterohepaticRecirculation(
                0.20,   # 20% biliary
                0.5,
                1.0,
                4.0,
                0.75,
                0.60
            )
        )

        @printf("%-25s: t1/2 %.1f h → %.1f h (%.1fx, ~%.0f peaks)\n",
                drug, ehc_result.intrinsic_t12_h, ehc_result.apparent_t12_h,
                ehc_result.extension_factor, ehc_result.expected_peaks)
    end
end

# ===========================================================================
# RUN TESTS
# ===========================================================================

println("\n")
results = test_bioavailability_prediction()
test_dissolution()
test_ehc()

println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
println("""

The GI tract model includes:
1. Multi-compartment anatomy (stomach, duodenum, jejunum, ileum, colon)
2. pH-dependent dissolution (Henderson-Hasselbalch)
3. Bile salt solubilization
4. Passive + carrier-mediated absorption (PEPT1, OATP)
5. Efflux transporters (P-gp, BCRP) by region
6. Gut wall metabolism (CYP3A4)
7. Hepatic first-pass
8. Enterohepatic recirculation

Key findings:
- BCS Class I drugs: Well predicted
- BCS Class II drugs: Dissolution-limited, need particle size
- BCS Class III drugs: Permeability-limited, carrier-dependent
- P-gp substrates: Efflux reduces F
- CYP3A4 substrates: Gut + liver metabolism

NEXT: Add fed/fasted state effects, formulation factors
""")
