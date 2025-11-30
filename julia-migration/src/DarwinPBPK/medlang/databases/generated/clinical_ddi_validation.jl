# =============================================================================
# CLINICAL DDI VALIDATION DATASET
# =============================================================================
# Gold-standard clinical DDI study results for PBPK model validation
# Sources: FDA drug labels, published clinical trials, DDI databases
# Generated: 2025-11-29
# =============================================================================

"""
Clinical DDI study results - gold standard for validation.
All AUC ratios are geometric mean ratios (GMR) from clinical studies.

Structure:
- perpetrator: inhibitor/inducer drug
- victim: substrate drug (object drug)
- auc_ratio: observed AUC ratio (treatment/control)
- cmax_ratio: observed Cmax ratio
- mechanism: :inhibition, :induction, :mbi
- enzyme: primary enzyme involved
- n_subjects: number of subjects in study
- source: citation
"""
const CLINICAL_DDI_STUDIES = [
    # =========================================================================
    # CYP3A4 INHIBITION - Strong Inhibitors
    # =========================================================================
    (
        perpetrator = :itraconazole,
        victim = :midazolam,
        auc_ratio = 10.1,
        cmax_ratio = 3.4,
        mechanism = :inhibition,
        enzyme = :CYP3A4,
        n_subjects = 12,
        source = "Olkkola 1994"
    ),
    (
        perpetrator = :ketoconazole,
        victim = :midazolam,
        auc_ratio = 15.9,
        cmax_ratio = 3.7,
        mechanism = :inhibition,
        enzyme = :CYP3A4,
        n_subjects = 6,
        source = "Tsunoda 1999"
    ),
    (
        perpetrator = :ritonavir,
        victim = :midazolam,
        auc_ratio = 28.3,
        cmax_ratio = 4.4,
        mechanism = :mbi,
        enzyme = :CYP3A4,
        n_subjects = 10,
        source = "Hsu 1998"
    ),
    (
        perpetrator = :clarithromycin,
        victim = :midazolam,
        auc_ratio = 8.4,
        cmax_ratio = 3.2,
        mechanism = :mbi,
        enzyme = :CYP3A4,
        n_subjects = 16,
        source = "Gorski 1998"
    ),
    (
        perpetrator = :itraconazole,
        victim = :triazolam,
        auc_ratio = 27.0,
        cmax_ratio = 3.0,
        mechanism = :inhibition,
        enzyme = :CYP3A4,
        n_subjects = 9,
        source = "Varhe 1994"
    ),
    (
        perpetrator = :ketoconazole,
        victim = :simvastatin,
        auc_ratio = 12.6,
        cmax_ratio = 8.5,
        mechanism = :inhibition,
        enzyme = :CYP3A4,
        n_subjects = 10,
        source = "Kivisto 1998"
    ),
    (
        perpetrator = :itraconazole,
        victim = :buspirone,
        auc_ratio = 19.0,
        cmax_ratio = 13.0,
        mechanism = :inhibition,
        enzyme = :CYP3A4,
        n_subjects = 8,
        source = "Kivisto 1997"
    ),

    # CYP3A4 INHIBITION - Moderate Inhibitors
    (
        perpetrator = :erythromycin,
        victim = :midazolam,
        auc_ratio = 4.4,
        cmax_ratio = 2.8,
        mechanism = :mbi,
        enzyme = :CYP3A4,
        n_subjects = 12,
        source = "Olkkola 1993"
    ),
    (
        perpetrator = :fluconazole,
        victim = :midazolam,
        auc_ratio = 3.5,
        cmax_ratio = 1.8,
        mechanism = :inhibition,
        enzyme = :CYP3A4,
        n_subjects = 12,
        source = "Olkkola 1996"
    ),
    (
        perpetrator = :diltiazem,
        victim = :midazolam,
        auc_ratio = 3.8,
        cmax_ratio = 1.7,
        mechanism = :mbi,
        enzyme = :CYP3A4,
        n_subjects = 9,
        source = "Backman 1994"
    ),
    (
        perpetrator = :verapamil,
        victim = :midazolam,
        auc_ratio = 2.9,
        cmax_ratio = 1.5,
        mechanism = :inhibition,
        enzyme = :CYP3A4,
        n_subjects = 6,
        source = "Backman 1994"
    ),
    (
        perpetrator = :grapefruit_juice,
        victim = :midazolam,
        auc_ratio = 1.5,
        cmax_ratio = 1.5,
        mechanism = :mbi,
        enzyme = :CYP3A4,
        n_subjects = 10,
        source = "Kupferschmidt 1995"
    ),
    (
        perpetrator = :grapefruit_juice,
        victim = :felodipine,
        auc_ratio = 2.8,
        cmax_ratio = 2.5,
        mechanism = :mbi,
        enzyme = :CYP3A4,
        n_subjects = 12,
        source = "Bailey 1998"
    ),

    # =========================================================================
    # CYP3A4 INDUCTION
    # =========================================================================
    (
        perpetrator = :rifampin,
        victim = :midazolam,
        auc_ratio = 0.04,
        cmax_ratio = 0.12,
        mechanism = :induction,
        enzyme = :CYP3A4,
        n_subjects = 10,
        source = "Backman 1996"
    ),
    (
        perpetrator = :rifampin,
        victim = :triazolam,
        auc_ratio = 0.05,
        cmax_ratio = 0.12,
        mechanism = :induction,
        enzyme = :CYP3A4,
        n_subjects = 10,
        source = "Villikka 1997"
    ),
    (
        perpetrator = :rifampin,
        victim = :simvastatin,
        auc_ratio = 0.13,
        cmax_ratio = 0.10,
        mechanism = :induction,
        enzyme = :CYP3A4,
        n_subjects = 10,
        source = "Kyrklund 2000"
    ),
    (
        perpetrator = :carbamazepine,
        victim = :midazolam,
        auc_ratio = 0.09,
        cmax_ratio = 0.25,
        mechanism = :induction,
        enzyme = :CYP3A4,
        n_subjects = 7,
        source = "Backman 1996"
    ),
    (
        perpetrator = :phenytoin,
        victim = :midazolam,
        auc_ratio = 0.06,
        cmax_ratio = 0.20,
        mechanism = :induction,
        enzyme = :CYP3A4,
        n_subjects = 8,
        source = "Backman 1996"
    ),
    (
        perpetrator = :st_johns_wort,
        victim = :midazolam,
        auc_ratio = 0.20,
        cmax_ratio = 0.40,
        mechanism = :induction,
        enzyme = :CYP3A4,
        n_subjects = 12,
        source = "Wang 2001"
    ),
    (
        perpetrator = :efavirenz,
        victim = :midazolam,
        auc_ratio = 0.36,
        cmax_ratio = 0.54,
        mechanism = :induction,
        enzyme = :CYP3A4,
        n_subjects = 14,
        source = "Marzolini 2001"
    ),

    # =========================================================================
    # CYP2D6 INHIBITION
    # =========================================================================
    (
        perpetrator = :quinidine,
        victim = :desipramine,
        auc_ratio = 7.5,
        cmax_ratio = 3.2,
        mechanism = :inhibition,
        enzyme = :CYP2D6,
        n_subjects = 6,
        source = "Brosen 1993"
    ),
    (
        perpetrator = :paroxetine,
        victim = :desipramine,
        auc_ratio = 5.2,
        cmax_ratio = 2.5,
        mechanism = :mbi,
        enzyme = :CYP2D6,
        n_subjects = 9,
        source = "Alderman 1997"
    ),
    (
        perpetrator = :fluoxetine,
        victim = :desipramine,
        auc_ratio = 4.8,
        cmax_ratio = 2.3,
        mechanism = :mbi,
        enzyme = :CYP2D6,
        n_subjects = 10,
        source = "Preskorn 1994"
    ),
    (
        perpetrator = :quinidine,
        victim = :dextromethorphan,
        auc_ratio = 30.0,
        cmax_ratio = 8.0,
        mechanism = :inhibition,
        enzyme = :CYP2D6,
        n_subjects = 8,
        source = "Zhang 2007"
    ),
    (
        perpetrator = :paroxetine,
        victim = :atomoxetine,
        auc_ratio = 6.5,
        cmax_ratio = 3.5,
        mechanism = :mbi,
        enzyme = :CYP2D6,
        n_subjects = 22,
        source = "Belle 2002"
    ),

    # =========================================================================
    # CYP2C19 INHIBITION
    # =========================================================================
    (
        perpetrator = :fluvoxamine,
        victim = :omeprazole,
        auc_ratio = 5.8,
        cmax_ratio = 2.4,
        mechanism = :inhibition,
        enzyme = :CYP2C19,
        n_subjects = 18,
        source = "Christensen 2002"
    ),
    (
        perpetrator = :fluconazole,
        victim = :omeprazole,
        auc_ratio = 2.6,
        cmax_ratio = 1.6,
        mechanism = :inhibition,
        enzyme = :CYP2C19,
        n_subjects = 18,
        source = "Kang 2002"
    ),

    # =========================================================================
    # CYP2C9 INHIBITION
    # =========================================================================
    (
        perpetrator = :fluconazole,
        victim = :warfarin_s,
        auc_ratio = 2.3,
        cmax_ratio = 1.0,
        mechanism = :inhibition,
        enzyme = :CYP2C9,
        n_subjects = 7,
        source = "Black 1996"
    ),
    (
        perpetrator = :amiodarone,
        victim = :warfarin_s,
        auc_ratio = 1.5,
        cmax_ratio = 1.0,
        mechanism = :inhibition,
        enzyme = :CYP2C9,
        n_subjects = 8,
        source = "Heimark 1992"
    ),

    # =========================================================================
    # CYP2C8 INHIBITION
    # =========================================================================
    (
        perpetrator = :gemfibrozil,
        victim = :repaglinide,
        auc_ratio = 8.1,
        cmax_ratio = 2.4,
        mechanism = :mbi,
        enzyme = :CYP2C8,
        n_subjects = 12,
        source = "Niemi 2003"
    ),
    (
        perpetrator = :gemfibrozil,
        victim = :rosiglitazone,
        auc_ratio = 2.3,
        cmax_ratio = 1.2,
        mechanism = :mbi,
        enzyme = :CYP2C8,
        n_subjects = 10,
        source = "Niemi 2003"
    ),
    (
        perpetrator = :clopidogrel,
        victim = :repaglinide,
        auc_ratio = 5.1,
        cmax_ratio = 2.5,
        mechanism = :mbi,
        enzyme = :CYP2C8,
        n_subjects = 12,
        source = "Tornio 2014"
    ),

    # =========================================================================
    # CYP1A2 INHIBITION
    # =========================================================================
    (
        perpetrator = :fluvoxamine,
        victim = :caffeine,
        auc_ratio = 5.0,
        cmax_ratio = 1.3,
        mechanism = :inhibition,
        enzyme = :CYP1A2,
        n_subjects = 8,
        source = "Jeppesen 1996"
    ),
    (
        perpetrator = :fluvoxamine,
        victim = :tizanidine,
        auc_ratio = 33.0,
        cmax_ratio = 12.0,
        mechanism = :inhibition,
        enzyme = :CYP1A2,
        n_subjects = 10,
        source = "Granfors 2004"
    ),
    (
        perpetrator = :ciprofloxacin,
        victim = :tizanidine,
        auc_ratio = 10.0,
        cmax_ratio = 7.0,
        mechanism = :inhibition,
        enzyme = :CYP1A2,
        n_subjects = 10,
        source = "Granfors 2004"
    ),

    # =========================================================================
    # TRANSPORTER-MEDIATED DDIs
    # =========================================================================
    (
        perpetrator = :cyclosporine,
        victim = :rosuvastatin,
        auc_ratio = 7.1,
        cmax_ratio = 10.6,
        mechanism = :inhibition,
        enzyme = :OATP1B1,
        n_subjects = 10,
        source = "Simonson 2004"
    ),
    (
        perpetrator = :rifampin_single_dose,
        victim = :atorvastatin,
        auc_ratio = 6.8,
        cmax_ratio = 10.5,
        mechanism = :inhibition,
        enzyme = :OATP1B1,
        n_subjects = 9,
        source = "Lau 2007"
    ),
    (
        perpetrator = :quinidine,
        victim = :digoxin,
        auc_ratio = 1.8,
        cmax_ratio = 1.5,
        mechanism = :inhibition,
        enzyme = :P_gp,
        n_subjects = 6,
        source = "Pedersen 1983"
    ),
    (
        perpetrator = :verapamil,
        victim = :digoxin,
        auc_ratio = 1.5,
        cmax_ratio = 1.4,
        mechanism = :inhibition,
        enzyme = :P_gp,
        n_subjects = 10,
        source = "Pedersen 1981"
    ),
    (
        perpetrator = :itraconazole,
        victim = :digoxin,
        auc_ratio = 1.5,
        cmax_ratio = 1.3,
        mechanism = :inhibition,
        enzyme = :P_gp,
        n_subjects = 10,
        source = "Jalava 1997"
    ),
]

# =============================================================================
# SUMMARY AND STATISTICS
# =============================================================================

const VALIDATION_SUMMARY = (
    total_studies = length(CLINICAL_DDI_STUDIES),
    inhibition_studies = count(s -> s.mechanism == :inhibition, CLINICAL_DDI_STUDIES),
    mbi_studies = count(s -> s.mechanism == :mbi, CLINICAL_DDI_STUDIES),
    induction_studies = count(s -> s.mechanism == :induction, CLINICAL_DDI_STUDIES),
    cyp3a4_studies = count(s -> s.enzyme == :CYP3A4, CLINICAL_DDI_STUDIES),
    cyp2d6_studies = count(s -> s.enzyme == :CYP2D6, CLINICAL_DDI_STUDIES),
    transporter_studies = count(s -> s.enzyme in [:OATP1B1, :P_gp, :BCRP], CLINICAL_DDI_STUDIES),
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
Get all clinical DDI studies for a specific perpetrator drug.
"""
function get_ddi_studies_by_perpetrator(drug::Symbol)
    return filter(s -> s.perpetrator == drug, CLINICAL_DDI_STUDIES)
end

"""
Get all clinical DDI studies for a specific victim drug.
"""
function get_ddi_studies_by_victim(drug::Symbol)
    return filter(s -> s.victim == drug, CLINICAL_DDI_STUDIES)
end

"""
Get all clinical DDI studies for a specific enzyme.
"""
function get_ddi_studies_by_enzyme(enzyme::Symbol)
    return filter(s -> s.enzyme == enzyme, CLINICAL_DDI_STUDIES)
end

"""
Calculate prediction accuracy metrics.
Returns (n, AFE, AAFE, within_2fold, within_3fold)
"""
function calculate_prediction_accuracy(observed::Vector{Float64}, predicted::Vector{Float64})
    n = length(observed)
    @assert n == length(predicted)

    # Average Fold Error (AFE) - bias measure
    log_ratios = log10.(predicted ./ observed)
    afe = 10^mean(log_ratios)

    # Absolute Average Fold Error (AAFE) - precision measure
    aafe = 10^mean(abs.(log_ratios))

    # Within X-fold accuracy
    fold_errors = max.(predicted ./ observed, observed ./ predicted)
    within_2fold = count(fe -> fe <= 2.0, fold_errors) / n * 100
    within_3fold = count(fe -> fe <= 3.0, fold_errors) / n * 100

    return (n=n, AFE=afe, AAFE=aafe, within_2fold=within_2fold, within_3fold=within_3fold)
end

# Print summary
println("Clinical DDI Validation Dataset loaded:")
println("  • $(VALIDATION_SUMMARY.total_studies) clinical DDI studies")
println("  • $(VALIDATION_SUMMARY.inhibition_studies) inhibition studies")
println("  • $(VALIDATION_SUMMARY.mbi_studies) MBI studies")
println("  • $(VALIDATION_SUMMARY.induction_studies) induction studies")
println("  • $(VALIDATION_SUMMARY.cyp3a4_studies) CYP3A4 studies")
println("  • $(VALIDATION_SUMMARY.cyp2d6_studies) CYP2D6 studies")
println("  • $(VALIDATION_SUMMARY.transporter_studies) transporter studies")
