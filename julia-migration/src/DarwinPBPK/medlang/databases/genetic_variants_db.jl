# =============================================================================
# PHARMACOGENOMIC VARIANTS DATABASE - GENETIC MODIFIERS FOR DDI
# =============================================================================
# Darwin PBPK Platform - Publication-Ready
#
# Sources: PharmGKB, PharmVar, CPIC Guidelines, FDA Table of PGx Associations
# Coverage: Major pharmacogenomic variants affecting drug metabolism and transport
#
# Includes:
# - CYP2D6, CYP2C19, CYP2C9, CYP3A5 polymorphisms
# - Transporter variants (OATP1B1, P-gp)
# - UGT variants
# - Population frequencies
# - Clinical actionability
# =============================================================================

# =============================================================================
# CYP2D6 PHARMACOGENOMICS
# =============================================================================

"""
CYP2D6 allele activity scores and phenotype assignments.
Based on CPIC guidelines and PharmVar.
"""
const CYP2D6_ALLELES = Dict{Symbol, NamedTuple}(
    # === Normal Function Alleles (Activity Score = 1.0) ===
    :CYP2D6_1 => (activity_score=1.0, func_status=:normal, frequency_eur=0.35, frequency_afr=0.50, frequency_eas=0.25),
    :CYP2D6_2 => (activity_score=1.0, func_status=:normal, frequency_eur=0.25, frequency_afr=0.20, frequency_eas=0.15),
    :CYP2D6_35 => (activity_score=1.0, func_status=:normal, frequency_eur=0.02, frequency_afr=0.01, frequency_eas=0.01),

    # === Decreased Function Alleles (Activity Score = 0.5) ===
    :CYP2D6_9 => (activity_score=0.5, func_status=:decreased, frequency_eur=0.02, frequency_afr=0.01, frequency_eas=0.01),
    :CYP2D6_10 => (activity_score=0.5, func_status=:decreased, frequency_eur=0.02, frequency_afr=0.05, frequency_eas=0.50),
    :CYP2D6_17 => (activity_score=0.5, func_status=:decreased, frequency_eur=0.01, frequency_afr=0.20, frequency_eas=0.01),
    :CYP2D6_29 => (activity_score=0.5, func_status=:decreased, frequency_eur=0.01, frequency_afr=0.10, frequency_eas=0.01),
    :CYP2D6_41 => (activity_score=0.5, func_status=:decreased, frequency_eur=0.08, frequency_afr=0.03, frequency_eas=0.02),

    # === No Function Alleles (Activity Score = 0) ===
    :CYP2D6_3 => (activity_score=0.0, func_status=:no_function, frequency_eur=0.02, frequency_afr=0.01, frequency_eas=0.01),
    :CYP2D6_4 => (activity_score=0.0, func_status=:no_function, frequency_eur=0.20, frequency_afr=0.06, frequency_eas=0.01),
    :CYP2D6_5 => (activity_score=0.0, func_status=:no_function, frequency_eur=0.03, frequency_afr=0.06, frequency_eas=0.06),
    :CYP2D6_6 => (activity_score=0.0, func_status=:no_function, frequency_eur=0.01, frequency_afr=0.01, frequency_eas=0.01),

    # === Increased Function (Gene Duplications) ===
    :CYP2D6_1xN => (activity_score=2.0, func_status=:increased, frequency_eur=0.01, frequency_afr=0.03, frequency_eas=0.01),
    :CYP2D6_2xN => (activity_score=2.0, func_status=:increased, frequency_eur=0.01, frequency_afr=0.05, frequency_eas=0.01),
)

"""
CYP2D6 phenotype definitions based on activity score.
"""
const CYP2D6_PHENOTYPES = Dict{Symbol, NamedTuple}(
    :ultrarapid_metabolizer => (
        activity_score_min=2.0, activity_score_max=Inf,
        abbreviation="UM", enzyme_activity=3.0,
        frequency_eur=0.02, frequency_afr=0.05, frequency_eas=0.02,
        clinical_impact=:increased_metabolism,
        recommendation="May need higher doses of CYP2D6 substrates"
    ),
    :normal_metabolizer => (
        activity_score_min=1.5, activity_score_max=2.0,
        abbreviation="NM", enzyme_activity=1.0,
        frequency_eur=0.75, frequency_afr=0.65, frequency_eas=0.40,
        clinical_impact=:normal,
        recommendation="Standard dosing"
    ),
    :intermediate_metabolizer => (
        activity_score_min=0.5, activity_score_max=1.5,
        abbreviation="IM", enzyme_activity=0.5,
        frequency_eur=0.10, frequency_afr=0.20, frequency_eas=0.45,
        clinical_impact=:decreased_metabolism,
        recommendation="May need dose reduction of CYP2D6 substrates"
    ),
    :poor_metabolizer => (
        activity_score_min=0.0, activity_score_max=0.5,
        abbreviation="PM", enzyme_activity=0.0,
        frequency_eur=0.08, frequency_afr=0.05, frequency_eas=0.01,
        clinical_impact=:no_metabolism,
        recommendation="Avoid CYP2D6 substrates or use 25-50% dose"
    ),
)

"""
CYP2D6 substrate dose adjustments by phenotype.
"""
const CYP2D6_DOSE_ADJUSTMENTS = Dict{Tuple{Symbol, Symbol}, NamedTuple}(
    # === Opioids (Prodrugs) ===
    (:codeine, :poor_metabolizer) => (adjustment=0.0, recommendation="Avoid - no analgesic effect", contraindicated=true),
    (:codeine, :intermediate_metabolizer) => (adjustment=0.0, recommendation="Avoid or use alternative", contraindicated=true),
    (:codeine, :ultrarapid_metabolizer) => (adjustment=0.0, recommendation="Avoid - toxicity risk", contraindicated=true),
    (:tramadol, :poor_metabolizer) => (adjustment=0.0, recommendation="Avoid - reduced efficacy", contraindicated=true),
    (:tramadol, :ultrarapid_metabolizer) => (adjustment=0.0, recommendation="Avoid - toxicity risk", contraindicated=true),

    # === TCAs ===
    (:amitriptyline, :poor_metabolizer) => (adjustment=0.50, recommendation="Reduce dose 50%, monitor levels", contraindicated=false),
    (:amitriptyline, :ultrarapid_metabolizer) => (adjustment=1.50, recommendation="May need higher dose, consider alternative", contraindicated=false),
    (:nortriptyline, :poor_metabolizer) => (adjustment=0.50, recommendation="Reduce dose 50%, monitor levels", contraindicated=false),
    (:desipramine, :poor_metabolizer) => (adjustment=0.25, recommendation="Reduce dose 75%", contraindicated=false),

    # === Antipsychotics ===
    (:risperidone, :poor_metabolizer) => (adjustment=0.50, recommendation="Reduce dose 50%", contraindicated=false),
    (:aripiprazole, :poor_metabolizer) => (adjustment=0.50, recommendation="Reduce dose 50%", contraindicated=false),
    (:thioridazine, :poor_metabolizer) => (adjustment=0.0, recommendation="Avoid", contraindicated=true),
    (:pimozide, :poor_metabolizer) => (adjustment=0.0, recommendation="Avoid - QT prolongation risk", contraindicated=true),

    # === Beta Blockers ===
    (:metoprolol, :poor_metabolizer) => (adjustment=0.50, recommendation="Consider dose reduction", contraindicated=false),
    (:carvedilol, :poor_metabolizer) => (adjustment=0.75, recommendation="Monitor for excessive beta-blockade", contraindicated=false),

    # === Antiarrhythmics ===
    (:flecainide, :poor_metabolizer) => (adjustment=0.50, recommendation="Reduce dose 50%, monitor levels", contraindicated=false),
    (:propafenone, :poor_metabolizer) => (adjustment=0.50, recommendation="Reduce dose 50%", contraindicated=false),

    # === Tamoxifen ===
    (:tamoxifen, :poor_metabolizer) => (adjustment=0.0, recommendation="Consider alternative (aromatase inhibitor)", contraindicated=false),
    (:tamoxifen, :intermediate_metabolizer) => (adjustment=1.0, recommendation="Standard dose, monitor response", contraindicated=false),
)

# =============================================================================
# CYP2C19 PHARMACOGENOMICS
# =============================================================================

const CYP2C19_ALLELES = Dict{Symbol, NamedTuple}(
    # === Normal Function ===
    :CYP2C19_1 => (activity_score=1.0, func_status=:normal, frequency_eur=0.65, frequency_afr=0.70, frequency_eas=0.35),

    # === No Function ===
    :CYP2C19_2 => (activity_score=0.0, func_status=:no_function, frequency_eur=0.15, frequency_afr=0.18, frequency_eas=0.30),
    :CYP2C19_3 => (activity_score=0.0, func_status=:no_function, frequency_eur=0.01, frequency_afr=0.01, frequency_eas=0.08),
    :CYP2C19_4 => (activity_score=0.0, func_status=:no_function, frequency_eur=0.01, frequency_afr=0.01, frequency_eas=0.01),

    # === Increased Function ===
    :CYP2C19_17 => (activity_score=1.5, func_status=:increased, frequency_eur=0.20, frequency_afr=0.20, frequency_eas=0.05),
)

const CYP2C19_PHENOTYPES = Dict{Symbol, NamedTuple}(
    :ultrarapid_metabolizer => (
        activity_score_min=2.5, activity_score_max=Inf,
        abbreviation="UM", enzyme_activity=2.0,
        frequency_eur=0.04, frequency_afr=0.04, frequency_eas=0.01,
        clinical_impact=:increased_metabolism
    ),
    :rapid_metabolizer => (
        activity_score_min=1.5, activity_score_max=2.5,
        abbreviation="RM", enzyme_activity=1.5,
        frequency_eur=0.20, frequency_afr=0.18, frequency_eas=0.05,
        clinical_impact=:increased_metabolism
    ),
    :normal_metabolizer => (
        activity_score_min=1.0, activity_score_max=1.5,
        abbreviation="NM", enzyme_activity=1.0,
        frequency_eur=0.40, frequency_afr=0.50, frequency_eas=0.35,
        clinical_impact=:normal
    ),
    :intermediate_metabolizer => (
        activity_score_min=0.5, activity_score_max=1.0,
        abbreviation="IM", enzyme_activity=0.5,
        frequency_eur=0.25, frequency_afr=0.23, frequency_eas=0.35,
        clinical_impact=:decreased_metabolism
    ),
    :poor_metabolizer => (
        activity_score_min=0.0, activity_score_max=0.5,
        abbreviation="PM", enzyme_activity=0.0,
        frequency_eur=0.03, frequency_afr=0.04, frequency_eas=0.15,
        clinical_impact=:no_metabolism
    ),
)

const CYP2C19_DOSE_ADJUSTMENTS = Dict{Tuple{Symbol, Symbol}, NamedTuple}(
    # === Clopidogrel ===
    (:clopidogrel, :poor_metabolizer) => (adjustment=0.0, recommendation="Use alternative (prasugrel, ticagrelor)", contraindicated=true),
    (:clopidogrel, :intermediate_metabolizer) => (adjustment=0.0, recommendation="Consider alternative antiplatelet", contraindicated=false),

    # === PPIs ===
    (:omeprazole, :poor_metabolizer) => (adjustment=0.50, recommendation="May need lower dose", contraindicated=false),
    (:omeprazole, :ultrarapid_metabolizer) => (adjustment=2.0, recommendation="May need higher dose for H. pylori", contraindicated=false),
    (:pantoprazole, :poor_metabolizer) => (adjustment=0.75, recommendation="Standard dose usually adequate", contraindicated=false),

    # === SSRIs ===
    (:citalopram, :poor_metabolizer) => (adjustment=0.50, recommendation="Max dose 20 mg/day", contraindicated=false),
    (:escitalopram, :poor_metabolizer) => (adjustment=0.50, recommendation="Max dose 10 mg/day", contraindicated=false),
    (:sertraline, :poor_metabolizer) => (adjustment=0.50, recommendation="Reduce starting dose", contraindicated=false),

    # === Voriconazole ===
    (:voriconazole, :poor_metabolizer) => (adjustment=0.50, recommendation="Reduce dose, monitor levels", contraindicated=false),
    (:voriconazole, :ultrarapid_metabolizer) => (adjustment=2.0, recommendation="May need higher dose", contraindicated=false),
)

# =============================================================================
# CYP2C9 PHARMACOGENOMICS
# =============================================================================

const CYP2C9_ALLELES = Dict{Symbol, NamedTuple}(
    :CYP2C9_1 => (activity_score=1.0, func_status=:normal, frequency_eur=0.80, frequency_afr=0.90, frequency_eas=0.95),
    :CYP2C9_2 => (activity_score=0.5, func_status=:decreased, frequency_eur=0.12, frequency_afr=0.03, frequency_eas=0.01),
    :CYP2C9_3 => (activity_score=0.0, func_status=:no_function, frequency_eur=0.08, frequency_afr=0.02, frequency_eas=0.03),
    :CYP2C9_5 => (activity_score=0.5, func_status=:decreased, frequency_eur=0.01, frequency_afr=0.03, frequency_eas=0.01),
    :CYP2C9_6 => (activity_score=0.0, func_status=:no_function, frequency_eur=0.01, frequency_afr=0.01, frequency_eas=0.01),
    :CYP2C9_8 => (activity_score=0.5, func_status=:decreased, frequency_eur=0.01, frequency_afr=0.05, frequency_eas=0.01),
    :CYP2C9_11 => (activity_score=0.5, func_status=:decreased, frequency_eur=0.01, frequency_afr=0.03, frequency_eas=0.01),
)

const CYP2C9_PHENOTYPES = Dict{Symbol, NamedTuple}(
    :normal_metabolizer => (
        genotype_pattern=["*1/*1"],
        abbreviation="NM", enzyme_activity=1.0,
        clinical_impact=:normal
    ),
    :intermediate_metabolizer => (
        genotype_pattern=["*1/*2", "*1/*3", "*2/*2"],
        abbreviation="IM", enzyme_activity=0.5,
        clinical_impact=:decreased_metabolism
    ),
    :poor_metabolizer => (
        genotype_pattern=["*2/*3", "*3/*3"],
        abbreviation="PM", enzyme_activity=0.15,
        clinical_impact=:severely_decreased_metabolism
    ),
)

const CYP2C9_WARFARIN_DOSING = Dict{Symbol, NamedTuple}(
    # Warfarin dosing adjustments based on CYP2C9/VKORC1
    :NM_AA => (daily_dose_mg=5.0, adjustment=1.0, description="CYP2C9 NM, VKORC1 A/A"),
    :NM_AG => (daily_dose_mg=4.5, adjustment=0.9, description="CYP2C9 NM, VKORC1 A/G"),
    :NM_GG => (daily_dose_mg=6.0, adjustment=1.2, description="CYP2C9 NM, VKORC1 G/G"),
    :IM_AA => (daily_dose_mg=3.5, adjustment=0.7, description="CYP2C9 IM, VKORC1 A/A"),
    :IM_AG => (daily_dose_mg=3.5, adjustment=0.7, description="CYP2C9 IM, VKORC1 A/G"),
    :IM_GG => (daily_dose_mg=4.5, adjustment=0.9, description="CYP2C9 IM, VKORC1 G/G"),
    :PM_AA => (daily_dose_mg=2.0, adjustment=0.4, description="CYP2C9 PM, VKORC1 A/A"),
    :PM_AG => (daily_dose_mg=2.0, adjustment=0.4, description="CYP2C9 PM, VKORC1 A/G"),
    :PM_GG => (daily_dose_mg=3.0, adjustment=0.6, description="CYP2C9 PM, VKORC1 G/G"),
)

# =============================================================================
# CYP3A5 PHARMACOGENOMICS
# =============================================================================

const CYP3A5_ALLELES = Dict{Symbol, NamedTuple}(
    :CYP3A5_1 => (activity_score=1.0, func_status=:normal, frequency_eur=0.05, frequency_afr=0.70, frequency_eas=0.25),
    :CYP3A5_3 => (activity_score=0.0, func_status=:no_function, frequency_eur=0.90, frequency_afr=0.25, frequency_eas=0.70),
    :CYP3A5_6 => (activity_score=0.0, func_status=:no_function, frequency_eur=0.01, frequency_afr=0.10, frequency_eas=0.01),
    :CYP3A5_7 => (activity_score=0.0, func_status=:no_function, frequency_eur=0.01, frequency_afr=0.05, frequency_eas=0.01),
)

const CYP3A5_PHENOTYPES = Dict{Symbol, NamedTuple}(
    :expressor => (
        genotype_pattern=["*1/*1", "*1/*3", "*1/*6", "*1/*7"],
        abbreviation="EXP", enzyme_activity=1.0,
        frequency_eur=0.10, frequency_afr=0.80, frequency_eas=0.40
    ),
    :non_expressor => (
        genotype_pattern=["*3/*3", "*6/*6", "*7/*7", "*3/*6", "*3/*7", "*6/*7"],
        abbreviation="NEXP", enzyme_activity=0.0,
        frequency_eur=0.85, frequency_afr=0.10, frequency_eas=0.55
    ),
)

const CYP3A5_TACROLIMUS_DOSING = Dict{Symbol, NamedTuple}(
    :expressor => (starting_dose_multiplier=1.5, recommendation="Start at 1.5-2x standard dose"),
    :non_expressor => (starting_dose_multiplier=1.0, recommendation="Standard starting dose"),
)

# =============================================================================
# TRANSPORTER PHARMACOGENOMICS
# =============================================================================

const SLCO1B1_ALLELES = Dict{Symbol, NamedTuple}(
    # OATP1B1 (SLCO1B1) variants
    :SLCO1B1_1a => (activity=1.0, func_status=:normal, frequency_eur=0.60, frequency_afr=0.75, frequency_eas=0.85),
    :SLCO1B1_1b => (activity=0.8, func_status=:decreased, frequency_eur=0.25, frequency_afr=0.20, frequency_eas=0.10),
    :SLCO1B1_5 => (activity=0.3, func_status=:poor, frequency_eur=0.15, frequency_afr=0.05, frequency_eas=0.10),
    :SLCO1B1_15 => (activity=0.5, func_status=:decreased, frequency_eur=0.10, frequency_afr=0.03, frequency_eas=0.05),
    :SLCO1B1_17 => (activity=0.5, func_status=:decreased, frequency_eur=0.05, frequency_afr=0.02, frequency_eas=0.03),
)

const SLCO1B1_STATIN_RISK = Dict{Symbol, NamedTuple}(
    # Myopathy risk with simvastatin based on SLCO1B1 genotype
    :normal_function => (myopathy_risk=:low, simvastatin_max_mg=80, recommendation="Standard dosing"),
    :decreased_function => (myopathy_risk=:intermediate, simvastatin_max_mg=40, recommendation="Avoid high doses"),
    :poor_function => (myopathy_risk=:high, simvastatin_max_mg=20, recommendation="Consider alternative statin"),
)

const ABCB1_ALLELES = Dict{Symbol, NamedTuple}(
    # P-gp (ABCB1) variants
    :ABCB1_wt => (activity=1.0, frequency_eur=0.45, frequency_afr=0.80, frequency_eas=0.60),
    :ABCB1_3435T => (activity=0.7, frequency_eur=0.55, frequency_afr=0.20, frequency_eas=0.40),
)

# =============================================================================
# UGT PHARMACOGENOMICS
# =============================================================================

const UGT1A1_ALLELES = Dict{Symbol, NamedTuple}(
    :UGT1A1_1 => (activity=1.0, func_status=:normal, frequency_eur=0.65, frequency_afr=0.55, frequency_eas=0.70),
    :UGT1A1_6 => (activity=0.5, func_status=:decreased, frequency_eur=0.01, frequency_afr=0.01, frequency_eas=0.15),
    :UGT1A1_28 => (activity=0.3, func_status=:poor, frequency_eur=0.35, frequency_afr=0.45, frequency_eas=0.15),
    :UGT1A1_37 => (activity=0.5, func_status=:decreased, frequency_eur=0.05, frequency_afr=0.05, frequency_eas=0.01),
)

const UGT1A1_IRINOTECAN_DOSING = Dict{Symbol, NamedTuple}(
    :normal_metabolizer => (starting_dose_percent=100, recommendation="Standard irinotecan dosing"),
    :intermediate_metabolizer => (starting_dose_percent=75, recommendation="Consider dose reduction to 75%"),
    :poor_metabolizer => (starting_dose_percent=50, recommendation="Reduce starting dose by 50%, monitor toxicity"),
)

# =============================================================================
# DPYD PHARMACOGENOMICS (Fluoropyrimidines)
# =============================================================================

const DPYD_ALLELES = Dict{Symbol, NamedTuple}(
    :DPYD_1 => (activity=1.0, func_status=:normal, frequency_all=0.95),
    :DPYD_2A => (activity=0.0, func_status=:no_function, frequency_all=0.01),
    :DPYD_13 => (activity=0.0, func_status=:no_function, frequency_all=0.005),
    :DPYD_c2846A => (activity=0.5, func_status=:decreased, frequency_all=0.01),
    :DPYD_HapB3 => (activity=0.5, func_status=:decreased, frequency_all=0.02),
)

const DPYD_5FU_DOSING = Dict{Symbol, NamedTuple}(
    :normal_metabolizer => (starting_dose_percent=100, recommendation="Standard 5-FU/capecitabine dosing"),
    :intermediate_metabolizer => (starting_dose_percent=50, recommendation="Reduce starting dose by 50%"),
    :poor_metabolizer => (starting_dose_percent=0, recommendation="Avoid fluoropyrimidines - life-threatening toxicity risk", contraindicated=true),
)

# =============================================================================
# TPMT PHARMACOGENOMICS (Thiopurines)
# =============================================================================

const TPMT_ALLELES = Dict{Symbol, NamedTuple}(
    :TPMT_1 => (activity=1.0, func_status=:normal, frequency_all=0.90),
    :TPMT_2 => (activity=0.0, func_status=:no_function, frequency_all=0.002),
    :TPMT_3A => (activity=0.0, func_status=:no_function, frequency_all=0.05),
    :TPMT_3B => (activity=0.0, func_status=:no_function, frequency_all=0.01),
    :TPMT_3C => (activity=0.0, func_status=:no_function, frequency_all=0.02),
)

const TPMT_THIOPURINE_DOSING = Dict{Symbol, NamedTuple}(
    :normal_metabolizer => (starting_dose_percent=100, recommendation="Standard thiopurine dosing"),
    :intermediate_metabolizer => (starting_dose_percent=50, recommendation="Reduce dose by 30-50%"),
    :poor_metabolizer => (starting_dose_percent=10, recommendation="Reduce dose by 90% or use alternative, if thiopurine warranted", contraindicated=false),
)

# =============================================================================
# DISEASE STATE DDI MODIFIERS (GENETIC)
# =============================================================================

"""
Combined genetic modifiers for DDI predictions.
"""
const GENETIC_DDI_MODIFIERS = Dict{Symbol, NamedTuple}(
    :cyp2d6_pm => (
        enzyme_activity=Dict(:CYP2D6 => 0.0, :CYP3A4 => 1.0, :CYP2C9 => 1.0, :CYP2C19 => 1.0, :CYP1A2 => 1.0),
        ddi_sensitivity=0.5,  # 2D6 DDIs have less effect on already deficient metabolism
        description="CYP2D6 poor metabolizer - 2D6 inhibitor DDIs reduced"
    ),
    :cyp2d6_um => (
        enzyme_activity=Dict(:CYP2D6 => 3.0, :CYP3A4 => 1.0, :CYP2C9 => 1.0, :CYP2C19 => 1.0, :CYP1A2 => 1.0),
        ddi_sensitivity=1.5,  # 2D6 DDIs more pronounced
        description="CYP2D6 ultrarapid metabolizer - 2D6 inhibitor DDIs enhanced"
    ),
    :cyp2c19_pm => (
        enzyme_activity=Dict(:CYP2D6 => 1.0, :CYP3A4 => 1.0, :CYP2C9 => 1.0, :CYP2C19 => 0.0, :CYP1A2 => 1.0),
        ddi_sensitivity=0.5,
        description="CYP2C19 poor metabolizer"
    ),
    :cyp2c19_um => (
        enzyme_activity=Dict(:CYP2D6 => 1.0, :CYP3A4 => 1.0, :CYP2C9 => 1.0, :CYP2C19 => 2.0, :CYP1A2 => 1.0),
        ddi_sensitivity=1.3,
        description="CYP2C19 ultrarapid metabolizer"
    ),
    :cyp2c9_pm => (
        enzyme_activity=Dict(:CYP2D6 => 1.0, :CYP3A4 => 1.0, :CYP2C9 => 0.15, :CYP2C19 => 1.0, :CYP1A2 => 1.0),
        ddi_sensitivity=0.6,
        description="CYP2C9 poor metabolizer (*3/*3)"
    ),
    :cyp3a5_expressor => (
        enzyme_activity=Dict(:CYP2D6 => 1.0, :CYP3A4 => 1.0, :CYP3A5 => 1.0, :CYP2C9 => 1.0, :CYP2C19 => 1.0, :CYP1A2 => 1.0),
        ddi_sensitivity=0.8,  # Additional CYP3A5 provides partial backup
        description="CYP3A5 expressor - reduced CYP3A4 DDI impact"
    ),
    :slco1b1_poor_function => (
        transporter_activity=Dict(:OATP1B1 => 0.3, :OATP1B3 => 1.0, :PGP => 1.0),
        ddi_sensitivity=1.5,
        description="SLCO1B1*5 - increased statin exposure"
    ),
)

# Export all pharmacogenomic databases
export CYP2D6_ALLELES, CYP2D6_PHENOTYPES, CYP2D6_DOSE_ADJUSTMENTS
export CYP2C19_ALLELES, CYP2C19_PHENOTYPES, CYP2C19_DOSE_ADJUSTMENTS
export CYP2C9_ALLELES, CYP2C9_PHENOTYPES, CYP2C9_WARFARIN_DOSING
export CYP3A5_ALLELES, CYP3A5_PHENOTYPES, CYP3A5_TACROLIMUS_DOSING
export SLCO1B1_ALLELES, SLCO1B1_STATIN_RISK
export ABCB1_ALLELES
export UGT1A1_ALLELES, UGT1A1_IRINOTECAN_DOSING
export DPYD_ALLELES, DPYD_5FU_DOSING
export TPMT_ALLELES, TPMT_THIOPURINE_DOSING
export GENETIC_DDI_MODIFIERS
