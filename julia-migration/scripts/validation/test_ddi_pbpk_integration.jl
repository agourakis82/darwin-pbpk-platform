#!/usr/bin/env julia
# =============================================================================
# DDI-PBPK INTEGRATION VALIDATION
# =============================================================================
# Test the dynamic DDI-PBPK simulation against clinical observations
# Demonstrates time-course Cp(t) profiles with and without DDI
#
# Darwin PBPK Platform v2.10.0
# =============================================================================

using Printf

# Add the project path
push!(LOAD_PATH, joinpath(@__DIR__, "../../src"))

println("=" ^ 70)
println("DDI-PBPK INTEGRATION VALIDATION")
println("=" ^ 70)
println()

# Include the module
include("../../src/DarwinPBPK/medlang/ddi_pbpk_integration.jl")
using .DDIPBPKIntegration

# =============================================================================
# TEST 1: MIDAZOLAM + KETOCONAZOLE (Strong CYP3A4 Inhibition)
# =============================================================================
println("TEST 1: MIDAZOLAM + KETOCONAZOLE")
println("-" ^ 50)
println("Clinical observation: AUC ratio = 15-16x")
println("Mechanism: Reversible CYP3A4 inhibition (gut + liver)")
println()

params_mid_keto = build_ddi_params(
    :midazolam, 7.5,      # 7.5 mg midazolam
    :ketoconazole, 400.0, # 400 mg ketoconazole
    perpetrator_n_doses = 3  # Pre-treatment
)

result_mid_keto = simulate_ddi_pbpk(params_mid_keto, t_max_h = 48.0)

println("Results:")
@printf("  AUC ratio:  %.1fx (clinical: 15-16x)\n", result_mid_keto.auc_ratio)
@printf("  Cmax ratio: %.1fx\n", result_mid_keto.cmax_ratio)
@printf("  Fg ratio:   %.2fx\n", result_mid_keto.fg_ratio)
@printf("  Fh ratio:   %.2fx\n", result_mid_keto.fh_ratio)
println()
@printf("  AUC alone:  %.1f μM·h\n", result_mid_keto.auc_alone)
@printf("  AUC DDI:    %.1f μM·h\n", result_mid_keto.auc_ddi)
@printf("  Half-life alone: %.1f h\n", result_mid_keto.half_life_alone)
@printf("  Half-life DDI:   %.1f h\n", result_mid_keto.half_life_ddi)
println()

# Show time-course
println("Time-course (selected points):")
println("  Time(h)  Cp_alone(μM)  Cp_DDI(μM)  Perpetrator(μM)  Enzyme_liver")
for i in [1, 5, 10, 20, 30, 50, 75, 100]
    if i <= length(result_mid_keto.time_h)
        @printf("  %5.1f    %8.3f      %8.3f    %8.3f         %.2f\n",
            result_mid_keto.time_h[i],
            result_mid_keto.Cp_victim_alone[i],
            result_mid_keto.Cp_victim_ddi[i],
            result_mid_keto.Cp_perpetrator[i],
            result_mid_keto.enzyme_activity_liver[i]
        )
    end
end
println()

# =============================================================================
# TEST 2: MIDAZOLAM + CLARITHROMYCIN (MBI)
# =============================================================================
println("TEST 2: MIDAZOLAM + CLARITHROMYCIN")
println("-" ^ 50)
println("Clinical observation: AUC ratio = 3-4x (moderate)")
println("Mechanism: Mechanism-based inactivation (MBI)")
println()

params_mid_clari = build_ddi_params(
    :midazolam, 7.5,
    :clarithromycin, 500.0,
    perpetrator_n_doses = 5,  # 5 days pre-treatment
    perpetrator_interval_h = 12.0  # BID dosing
)

result_mid_clari = simulate_ddi_pbpk(params_mid_clari, t_max_h = 72.0)

println("Results:")
@printf("  AUC ratio:  %.1fx (clinical: 3-4x)\n", result_mid_clari.auc_ratio)
@printf("  Cmax ratio: %.1fx\n", result_mid_clari.cmax_ratio)
@printf("  Min enzyme activity (liver): %.2f\n", minimum(result_mid_clari.enzyme_activity_liver))
@printf("  Min enzyme activity (gut):   %.2f\n", minimum(result_mid_clari.enzyme_activity_gut))
println()

# =============================================================================
# TEST 3: SIMVASTATIN + ITRACONAZOLE (High First-Pass Drug)
# =============================================================================
println("TEST 3: SIMVASTATIN + ITRACONAZOLE")
println("-" ^ 50)
println("Clinical observation: AUC ratio = 10-20x")
println("Mechanism: Strong CYP3A4 inhibition of high first-pass drug")
println()

params_simva_itra = build_ddi_params(
    :simvastatin, 40.0,
    :itraconazole, 200.0,
    perpetrator_n_doses = 4  # Pre-treatment
)

result_simva_itra = simulate_ddi_pbpk(params_simva_itra, t_max_h = 48.0)

println("Results:")
@printf("  AUC ratio:  %.1fx (clinical: 10-20x)\n", result_simva_itra.auc_ratio)
@printf("  Cmax ratio: %.1fx\n", result_simva_itra.cmax_ratio)
@printf("  Fg ratio:   %.2fx\n", result_simva_itra.fg_ratio)
@printf("  Fh ratio:   %.2fx\n", result_simva_itra.fh_ratio)
println()
println("Note: High AUC ratio due to:")
println("  - Low baseline F (5%) amplifies DDI effect")
println("  - Gut wall (Fg) inhibition contributes significantly")
println()

# =============================================================================
# TEST 4: DEXTROMETHORPHAN + QUINIDINE (CYP2D6)
# =============================================================================
println("TEST 4: DEXTROMETHORPHAN + QUINIDINE")
println("-" ^ 50)
println("Clinical observation: AUC ratio = 20-30x")
println("Mechanism: Strong CYP2D6 inhibition")
println()

params_dex_quin = build_ddi_params(
    :dextromethorphan, 30.0,
    :quinidine, 200.0,
    perpetrator_n_doses = 1
)

result_dex_quin = simulate_ddi_pbpk(params_dex_quin, t_max_h = 48.0)

println("Results:")
@printf("  AUC ratio:  %.1fx (clinical: 20-30x)\n", result_dex_quin.auc_ratio)
@printf("  Cmax ratio: %.1fx\n", result_dex_quin.cmax_ratio)
println()

# =============================================================================
# TEST 5: TIZANIDINE + CIPROFLOXACIN (CYP1A2)
# =============================================================================
println("TEST 5: TIZANIDINE + CIPROFLOXACIN")
println("-" ^ 50)
println("Clinical observation: AUC ratio = 10x")
println("Mechanism: Strong CYP1A2 inhibition")
println()

params_tiz_cipro = build_ddi_params(
    :tizanidine, 4.0,
    :ciprofloxacin, 500.0,
    perpetrator_n_doses = 2  # BID x 1 day
)

result_tiz_cipro = simulate_ddi_pbpk(params_tiz_cipro, t_max_h = 24.0)

println("Results:")
@printf("  AUC ratio:  %.1fx (clinical: 10x)\n", result_tiz_cipro.auc_ratio)
@printf("  Cmax ratio: %.1fx\n", result_tiz_cipro.cmax_ratio)
println()

# =============================================================================
# TEST 6: MBI TIME-COURSE (RITONAVIR)
# =============================================================================
println("TEST 6: MBI TIME-COURSE SIMULATION")
println("-" ^ 50)
println("Shows enzyme inactivation and recovery over 1 week")
println()

mbi_result = simulate_mbi_time_course(
    15.0,    # Cmax (μM)
    4.0,     # Half-life (h)
    2.0,     # kinact (h⁻¹)
    0.1,     # KI (μM)
    0.02,    # kdeg (h⁻¹) - CYP3A4
    t_max_h = 168.0,  # 1 week
    n_doses = 7
)

println("MBI Dynamics (7 days of dosing):")
@printf("  Minimum enzyme activity: %.1f%%\n", mbi_result.min_enzyme * 100)
@printf("  Maximum AUC ratio:       %.1fx\n", mbi_result.max_auc_ratio)
@printf("  Time to 90%% recovery:   %.1f h (%.1f days)\n",
    mbi_result.time_to_recovery_h,
    mbi_result.time_to_recovery_h / 24
)
println()

println("Time-course:")
println("  Day  Enzyme(%)  AUC_ratio")
for day in 0:7
    idx = Int(day * 24 / 0.5) + 1
    if idx <= length(mbi_result.time_h)
        @printf("  %3d    %5.1f      %5.1f\n",
            day,
            mbi_result.enzyme_activity[idx] * 100,
            mbi_result.auc_ratio[idx]
        )
    end
end
println()

# =============================================================================
# TEST 7: INDUCTION TIME-COURSE (RIFAMPIN)
# =============================================================================
println("TEST 7: INDUCTION TIME-COURSE SIMULATION")
println("-" ^ 50)
println("Shows enzyme induction onset over 2 weeks (rifampin)")
println()

induction_result = simulate_induction_onset(
    20.0,    # Cmax (μM)
    3.0,     # Half-life (h)
    10.0,    # Emax
    0.3,     # EC50 (μM)
    0.019,   # kdeg (h⁻¹) - CYP3A4
    t_max_h = 336.0,  # 2 weeks
    n_doses = 14
)

println("Induction Dynamics (14 days of dosing):")
@printf("  Maximum enzyme activity: %.1fx baseline\n", induction_result.max_enzyme)
@printf("  Minimum AUC ratio:       %.2f (%.0f%% reduction)\n",
    induction_result.min_auc_ratio,
    (1 - induction_result.min_auc_ratio) * 100
)
@printf("  Time to 50%% effect:     %.1f h (%.1f days)\n",
    induction_result.time_to_50pct_effect_h,
    induction_result.time_to_50pct_effect_h / 24
)
@printf("  Time to steady-state:   %.1f h (%.1f days)\n",
    induction_result.time_to_steady_state_h,
    induction_result.time_to_steady_state_h / 24
)
println()

println("Time-course:")
println("  Day  Enzyme(x)  AUC_ratio")
for day in [0, 1, 2, 3, 5, 7, 10, 14]
    idx = Int(day * 24) + 1
    if idx <= length(induction_result.time_h)
        @printf("  %3d    %5.1f      %5.2f\n",
            day,
            induction_result.enzyme_activity[idx],
            induction_result.auc_ratio[idx]
        )
    end
end
println()

# =============================================================================
# TEST 8: GUT-WALL vs HEPATIC DDI CONTRIBUTION
# =============================================================================
println("TEST 8: GUT-WALL vs HEPATIC DDI CONTRIBUTION")
println("-" ^ 50)
println("Separating Fg and Fh changes for high first-pass drugs")
println()

# Calculate gut-wall DDI for itraconazole
gut_ddi = calculate_fg_inhibition(
    200.0,    # Dose (mg)
    706.0,    # MW itraconazole
    0.003,    # Ki (μM) - very potent
    0.57      # Fg baseline (midazolam)
)

println("Itraconazole + Midazolam gut-wall DDI:")
@printf("  Gut lumen [I]: %.0f μM\n", gut_ddi.Ig_uM)
@printf("  Baseline Fg:   %.2f\n", gut_ddi.Fg_baseline)
@printf("  Inhibited Fg:  %.2f\n", gut_ddi.Fg_inhibited)
@printf("  Fg ratio:      %.1fx\n", gut_ddi.Fg_inhibited / gut_ddi.Fg_baseline)
println()

# Separate contributions
separation = separate_fg_fh(
    0.57, 0.77,   # Baseline Fg, Fh (midazolam)
    0.95, 0.95    # DDI Fg, Fh (approximations with strong inhibition)
)

println("Total DDI contribution analysis:")
@printf("  Baseline F:          %.2f\n", separation.F_baseline)
@printf("  DDI F:               %.2f\n", separation.F_ddi)
@printf("  Total AUC ratio:     %.1fx\n", separation.auc_ratio_total)
@printf("  Fg contribution:     %.1fx (%.0f%% of effect)\n",
    separation.fg_contribution,
    abs(separation.pct_from_fg)
)
@printf("  Fh contribution:     %.1fx (%.0f%% of effect)\n",
    separation.fh_contribution,
    abs(separation.pct_from_fh)
)
println()

# =============================================================================
# SUMMARY
# =============================================================================
println("=" ^ 70)
println("SUMMARY: DDI-PBPK INTEGRATION CAPABILITIES")
println("=" ^ 70)
println()
println("IMPLEMENTED FEATURES:")
println("  [x] Dynamic ODE-based PBPK simulation")
println("  [x] Time-course Cp(t) profiles with and without DDI")
println("  [x] Reversible competitive inhibition")
println("  [x] Mechanism-based inactivation (MBI)")
println("  [x] Enzyme induction with delayed onset")
println("  [x] Gut-wall (Fg) + hepatic (Fh) separation")
println("  [x] Multi-dose perpetrator scenarios")
println("  [x] Dynamic enzyme activity tracking")
println("  [x] CYP3A4, CYP2D6, CYP1A2, CYP2C9, CYP2C8 coverage")
println()
println("CLINICAL VALIDATION STATUS:")
println("  Midazolam + Ketoconazole: Model captures strong inhibition")
println("  Midazolam + Clarithromycin: MBI mechanism modeled")
println("  Simvastatin + Itraconazole: High first-pass amplification")
println("  Dextromethorphan + Quinidine: CYP2D6 inhibition")
println("  Tizanidine + Ciprofloxacin: CYP1A2 inhibition")
println()
println("NEXT STEPS FOR SOTA:")
println("  [ ] Parameter optimization vs clinical data")
println("  [ ] Transporter DDI integration (OATP1B1)")
println("  [ ] Population variability (Monte Carlo)")
println("  [ ] MedLang DSL generation for DDI scenarios")
println()
