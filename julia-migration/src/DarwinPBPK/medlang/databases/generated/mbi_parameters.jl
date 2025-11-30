# =============================================================================
# MECHANISM-BASED INHIBITION (MBI) PARAMETERS DATABASE
# =============================================================================
# Time-dependent/irreversible inhibition parameters for PBPK DDI prediction
# Sources: Published literature, FDA reviews, DIDB
# Generated: 2025-11-29
# =============================================================================

"""
Mechanism-based (time-dependent) inhibition parameters.

For MBI, the DDI magnitude depends on:
- kinact: Maximum inactivation rate constant (min⁻¹)
- KI: Concentration for half-maximal inactivation (μM)
- kdeg: Enzyme degradation rate constant (min⁻¹) - typically ~0.0005 for CYP3A4

The steady-state inhibition ratio is:
R = 1 + (kinact/kdeg) * [I]/(KI + [I])

Key reference: Grimm et al., Clin Pharmacol Ther 2009
"""
const MBI_PARAMETERS = Dict{Symbol, Dict{Symbol, NamedTuple}}(
    :CYP3A4 => Dict(
        # Macrolide antibiotics
        :clarithromycin => (
            kinact_per_min = 0.05,
            ki_um = 5.0,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 6.3,  # Updated: Greenblatt 2015 meta-analysis (6.5±10.9)
            metabolite = :N_desmethylclarithromycin,
            source = "Obach 2007; Polasek 2006; Greenblatt 2015"
        ),
        :erythromycin => (
            kinact_per_min = 0.03,
            ki_um = 10.0,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 4.0,
            metabolite = :nitrosoalkane_intermediate,
            source = "Wang 2004"
        ),
        :troleandomycin => (
            kinact_per_min = 0.08,
            ki_um = 2.0,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 10.0,
            metabolite = :nitrosoalkane_intermediate,
            source = "Pessayre 1981"
        ),
        :azithromycin => (
            kinact_per_min = 0.005,
            ki_um = 50.0,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 1.5,
            metabolite = :none,
            source = "Amsden 2002"
        ),

        # HIV protease inhibitors
        :ritonavir => (
            kinact_per_min = 0.15,
            ki_um = 0.07,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 30.0,
            metabolite = :isopropylthiazole,
            source = "Koudriakova 1998; Ernest 2005"
        ),
        :nelfinavir => (
            kinact_per_min = 0.06,
            ki_um = 1.0,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 6.0,
            metabolite = :unknown,
            source = "Ernest 2005"
        ),
        :lopinavir => (
            kinact_per_min = 0.04,
            ki_um = 0.5,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 8.0,
            metabolite = :unknown,
            source = "Lim 2004"
        ),
        :cobicistat => (
            kinact_per_min = 0.10,
            ki_um = 0.1,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 20.0,
            metabolite = :unknown,
            source = "Xu 2010"
        ),

        # Calcium channel blockers
        :diltiazem => (
            kinact_per_min = 0.03,
            ki_um = 3.0,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 4.0,
            metabolite = :ma_diltiazem,
            source = "Sutton 1997; Jones 1999"
        ),
        :verapamil => (
            kinact_per_min = 0.02,
            ki_um = 5.0,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 3.0,
            metabolite = :norverapamil,
            source = "Wang 2005"
        ),
        :nicardipine => (
            kinact_per_min = 0.02,
            ki_um = 3.0,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 2.0,
            metabolite = :unknown,
            source = "Ishiguro 2020"
        ),

        # Other MBI inhibitors
        :mibefradil => (
            kinact_per_min = 0.05,
            ki_um = 0.5,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 10.0,
            metabolite = :unknown,
            source = "Prueksaritanont 1999; withdrawn"
        ),
        :bergamottin => (
            kinact_per_min = 0.08,
            ki_um = 5.0,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 3.0,
            metabolite = :epoxide,
            source = "Paine 2006; grapefruit"
        ),
        :dihydroxybergamottin => (
            kinact_per_min = 0.10,
            ki_um = 2.0,
            kdeg_per_min = 0.0005,
            clinical_auc_ratio = 4.0,
            metabolite = :epoxide,
            source = "Paine 2006; grapefruit"
        ),
    ),

    :CYP2D6 => Dict(
        :paroxetine => (
            kinact_per_min = 0.15,
            ki_um = 0.2,
            kdeg_per_min = 0.0003,  # CYP2D6 has slower turnover
            clinical_auc_ratio = 8.0,
            metabolite = :carbene_intermediate,
            source = "Bertelsen 2003"
        ),
        :fluoxetine => (
            kinact_per_min = 0.08,
            ki_um = 0.3,
            kdeg_per_min = 0.0003,
            clinical_auc_ratio = 8.0,
            metabolite = :unknown,
            source = "Hemeryck 2000"
        ),
        :sertraline => (
            kinact_per_min = 0.03,
            ki_um = 1.0,
            kdeg_per_min = 0.0003,
            clinical_auc_ratio = 2.0,
            metabolite = :unknown,
            source = "Sproule 1997"
        ),
        :terbinafine => (
            kinact_per_min = 0.10,
            ki_um = 0.05,
            kdeg_per_min = 0.0003,
            clinical_auc_ratio = 5.0,
            metabolite = :unknown,
            source = "Abdel-Rahman 1999"
        ),
    ),

    :CYP2C19 => Dict(
        :ticlopidine => (
            kinact_per_min = 0.04,
            ki_um = 3.0,
            kdeg_per_min = 0.0003,
            clinical_auc_ratio = 2.5,
            metabolite = :thiol_reactive,
            source = "Ha-Duong 2001"
        ),
        :clopidogrel => (
            kinact_per_min = 0.03,
            ki_um = 5.0,
            kdeg_per_min = 0.0003,
            clinical_auc_ratio = 2.0,
            metabolite = :thiol_reactive,
            source = "Richter 2004"
        ),
    ),

    :CYP2B6 => Dict(
        :ticlopidine => (
            kinact_per_min = 0.06,
            ki_um = 0.5,
            kdeg_per_min = 0.0003,
            clinical_auc_ratio = 2.5,
            metabolite = :thiol_reactive,
            source = "Richter 2004"
        ),
        :clopidogrel => (
            kinact_per_min = 0.04,
            ki_um = 3.0,
            kdeg_per_min = 0.0003,
            clinical_auc_ratio = 2.0,
            metabolite = :thiol_reactive,
            source = "Richter 2004"
        ),
    ),

    :CYP2C8 => Dict(
        # Gemfibrozil parent + glucuronide both contribute to CYP2C8 MBI
        :gemfibrozil => (
            kinact_per_min = 0.21,
            ki_um = 10.0,  # Lower Ki to account for combined parent + metabolite effect
            kdeg_per_min = 0.0003,
            clinical_auc_ratio = 8.0,
            metabolite = :none,
            source = "Ogilvie 2006; Niemi 2003 - repaglinide 8.1x"
        ),
        :gemfibrozil_glucuronide => (
            kinact_per_min = 0.21,
            ki_um = 20.0,
            kdeg_per_min = 0.0003,
            clinical_auc_ratio = 8.0,
            metabolite = :glucuronide_adduct,
            source = "Ogilvie 2006; classic example"
        ),
        :clopidogrel_glucuronide => (
            kinact_per_min = 0.15,
            ki_um = 10.0,
            kdeg_per_min = 0.0003,
            clinical_auc_ratio = 5.0,
            metabolite = :glucuronide_adduct,
            source = "Tornio 2014"
        ),
    ),
)

# =============================================================================
# ENZYME TURNOVER RATES (kdeg)
# =============================================================================
# Critical for MBI predictions - determines time to reach new steady state

"""
Enzyme degradation rate constants (kdeg) in min⁻¹.
Half-life = ln(2) / kdeg
"""
const ENZYME_KDEG = Dict{Symbol, NamedTuple}(
    :CYP3A4 => (kdeg_per_min = 0.0005, half_life_hr = 23.0, tissue = :liver, source = "Yang 2008"),
    :CYP3A4_intestine => (kdeg_per_min = 0.0012, half_life_hr = 9.6, tissue = :intestine, source = "Galetin 2006"),
    :CYP2D6 => (kdeg_per_min = 0.00032, half_life_hr = 36.0, tissue = :liver, source = "Venkatakrishnan 1998"),
    :CYP2C9 => (kdeg_per_min = 0.0004, half_life_hr = 29.0, tissue = :liver, source = "Zanger 2013"),
    :CYP2C19 => (kdeg_per_min = 0.00038, half_life_hr = 30.0, tissue = :liver, source = "Zanger 2013"),
    :CYP2C8 => (kdeg_per_min = 0.0003, half_life_hr = 38.0, tissue = :liver, source = "Tornio 2012"),
    :CYP2B6 => (kdeg_per_min = 0.0003, half_life_hr = 38.0, tissue = :liver, source = "Zanger 2013"),
    :CYP1A2 => (kdeg_per_min = 0.0004, half_life_hr = 29.0, tissue = :liver, source = "Zanger 2013"),
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
Calculate the steady-state inhibition ratio for MBI.

R = 1 + (kinact/kdeg) * [I]/(KI + [I])

Where:
- kinact: inactivation rate constant (min⁻¹)
- kdeg: enzyme degradation rate (min⁻¹)
- I: inhibitor concentration (μM)
- KI: concentration for half-max inactivation (μM)
"""
function calculate_mbi_ratio(kinact::Float64, kdeg::Float64, inhibitor_conc::Float64, ki::Float64)
    return 1.0 + (kinact / kdeg) * inhibitor_conc / (ki + inhibitor_conc)
end

"""
Predict AUC ratio for mechanism-based inhibition.
Uses the standard static equation for MBI DDI prediction.
"""
function predict_mbi_auc_ratio(enzyme::Symbol, inhibitor::Symbol, inhibitor_conc_um::Float64, fm::Float64)
    if !haskey(MBI_PARAMETERS, enzyme) || !haskey(MBI_PARAMETERS[enzyme], inhibitor)
        return nothing
    end

    params = MBI_PARAMETERS[enzyme][inhibitor]
    kdeg = ENZYME_KDEG[enzyme].kdeg_per_min

    # Calculate inhibition ratio
    R = calculate_mbi_ratio(params.kinact_per_min, kdeg, inhibitor_conc_um, params.ki_um)

    # AUC ratio = 1 / (fm/R + (1-fm))
    auc_ratio = 1.0 / (fm / R + (1.0 - fm))

    return auc_ratio
end

# =============================================================================
# SUMMARY
# =============================================================================

const MBI_SUMMARY = (
    total_inhibitors = sum(length(v) for v in values(MBI_PARAMETERS)),
    enzymes_covered = length(MBI_PARAMETERS),
    cyp3a4_inhibitors = length(get(MBI_PARAMETERS, :CYP3A4, Dict())),
    cyp2d6_inhibitors = length(get(MBI_PARAMETERS, :CYP2D6, Dict())),
)

println("MBI Parameters Database loaded:")
println("  • $(MBI_SUMMARY.total_inhibitors) mechanism-based inhibitors")
println("  • $(MBI_SUMMARY.enzymes_covered) CYP enzymes covered")
println("  • $(MBI_SUMMARY.cyp3a4_inhibitors) CYP3A4 MBI inhibitors")
println("  • $(MBI_SUMMARY.cyp2d6_inhibitors) CYP2D6 MBI inhibitors")
