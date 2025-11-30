# =============================================================================
# FDA DRUG INTERACTION CLASSIFICATION DATABASE
# =============================================================================
# Source: FDA Drug Development and Drug Interactions Table
# https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers
# Generated: 2025-11-29
# =============================================================================

# =============================================================================
# CYP SUBSTRATES - FDA Index Substrates
# =============================================================================
# Sensitive: AUC increases ≥5-fold with strong inhibitor
# Moderate: AUC increases ≥2-fold but <5-fold with strong inhibitor

"""
FDA-classified CYP substrates with sensitivity classification.
Used for clinical DDI studies as index substrates.
"""
const FDA_CYP_SUBSTRATES = Dict{Symbol, Dict{Symbol, NamedTuple}}(
    :CYP1A2 => Dict(
        :caffeine => (sensitivity = :sensitive, auc_fold_with_inhibitor = 5.0, notes = "Primary index substrate"),
        :tizanidine => (sensitivity = :sensitive, auc_fold_with_inhibitor = 33.0, notes = "Avoid with fluvoxamine"),
        :theophylline => (sensitivity = :moderate, auc_fold_with_inhibitor = 2.5, notes = "Also CYP3A4 substrate"),
        :melatonin => (sensitivity = :sensitive, auc_fold_with_inhibitor = 12.0, notes = "High first-pass"),
    ),
    :CYP2B6 => Dict(
        :efavirenz => (sensitivity = :moderate, auc_fold_with_inhibitor = 2.0, notes = "Also inducer"),
        :bupropion => (sensitivity = :moderate, auc_fold_with_inhibitor = 2.5, notes = "Hydroxybupropion formation"),
    ),
    :CYP2C8 => Dict(
        :repaglinide => (sensitivity = :sensitive, auc_fold_with_inhibitor = 8.0, notes = "Also OATP1B1 substrate"),
        :rosiglitazone => (sensitivity = :moderate, auc_fold_with_inhibitor = 2.3, notes = ""),
        :paclitaxel => (sensitivity = :moderate, auc_fold_with_inhibitor = 2.0, notes = "6α-hydroxypaclitaxel formation"),
    ),
    :CYP2C9 => Dict(
        :warfarin_s => (sensitivity = :moderate, auc_fold_with_inhibitor = 2.5, notes = "S-warfarin, active enantiomer"),
        :tolbutamide => (sensitivity = :moderate, auc_fold_with_inhibitor = 2.0, notes = "Historical index substrate"),
        :celecoxib => (sensitivity = :moderate, auc_fold_with_inhibitor = 2.0, notes = ""),
        :flurbiprofen => (sensitivity = :moderate, auc_fold_with_inhibitor = 2.0, notes = "4'-hydroxyflurbiprofen"),
    ),
    :CYP2C19 => Dict(
        :omeprazole => (sensitivity = :sensitive, auc_fold_with_inhibitor = 6.0, notes = "5-hydroxyomeprazole formation"),
        :lansoprazole => (sensitivity = :moderate, auc_fold_with_inhibitor = 3.0, notes = ""),
        :clopidogrel => (sensitivity = :sensitive, auc_fold_with_inhibitor = 0.0, notes = "Prodrug, reduced activation"),
        :escitalopram => (sensitivity = :moderate, auc_fold_with_inhibitor = 2.0, notes = ""),
    ),
    :CYP2D6 => Dict(
        :desipramine => (sensitivity = :sensitive, auc_fold_with_inhibitor = 8.0, notes = "Classic index substrate"),
        :dextromethorphan => (sensitivity = :sensitive, auc_fold_with_inhibitor = 30.0, notes = "Dextrorphan formation"),
        :atomoxetine => (sensitivity = :sensitive, auc_fold_with_inhibitor = 7.0, notes = ""),
        :nebivolol => (sensitivity = :sensitive, auc_fold_with_inhibitor = 8.0, notes = ""),
        :metoprolol => (sensitivity = :moderate, auc_fold_with_inhibitor = 3.0, notes = ""),
        :codeine => (sensitivity = :sensitive, auc_fold_with_inhibitor = 0.0, notes = "Prodrug, reduced morphine formation"),
        :tamoxifen => (sensitivity = :sensitive, auc_fold_with_inhibitor = 0.0, notes = "Prodrug, reduced endoxifen"),
    ),
    :CYP3A4 => Dict(
        :midazolam => (sensitivity = :sensitive, auc_fold_with_inhibitor = 10.0, notes = "Gold standard index substrate"),
        :triazolam => (sensitivity = :sensitive, auc_fold_with_inhibitor = 20.0, notes = ""),
        :alfentanil => (sensitivity = :sensitive, auc_fold_with_inhibitor = 15.0, notes = "IV substrate"),
        :buspirone => (sensitivity = :sensitive, auc_fold_with_inhibitor = 19.0, notes = ""),
        :felodipine => (sensitivity = :sensitive, auc_fold_with_inhibitor = 15.0, notes = ""),
        :lovastatin => (sensitivity = :sensitive, auc_fold_with_inhibitor = 20.0, notes = ""),
        :simvastatin => (sensitivity = :sensitive, auc_fold_with_inhibitor = 10.0, notes = ""),
        :atorvastatin => (sensitivity = :moderate, auc_fold_with_inhibitor = 4.0, notes = ""),
        :nifedipine => (sensitivity = :moderate, auc_fold_with_inhibitor = 4.0, notes = ""),
        :sildenafil => (sensitivity = :moderate, auc_fold_with_inhibitor = 3.0, notes = ""),
        :cyclosporine => (sensitivity = :moderate, auc_fold_with_inhibitor = 3.0, notes = "Also P-gp substrate"),
        :tacrolimus => (sensitivity = :sensitive, auc_fold_with_inhibitor = 8.0, notes = "Also P-gp substrate"),
    ),
)

# =============================================================================
# CYP INHIBITORS - With Ki values where known
# =============================================================================
# Strong: ≥5-fold AUC increase OR ≥80% decrease in clearance
# Moderate: ≥2-fold but <5-fold AUC increase
# Weak: ≥1.25-fold but <2-fold AUC increase

"""
FDA-classified CYP inhibitors with strength and Ki values.
Ki values in μM (micromolar) from in vitro studies.
"""
const FDA_CYP_INHIBITORS = Dict{Symbol, Dict{Symbol, NamedTuple}}(
    :CYP1A2 => Dict(
        :fluvoxamine => (strength = :strong, ki_um = 0.02, mechanism = :reversible, auc_ratio = 33.0),
        :ciprofloxacin => (strength = :moderate, ki_um = 15.0, mechanism = :reversible, auc_ratio = 2.5),
        :enoxacin => (strength = :moderate, ki_um = 20.0, mechanism = :reversible, auc_ratio = 2.0),
        :mexiletine => (strength = :weak, ki_um = 30.0, mechanism = :reversible, auc_ratio = 1.5),
        :zileuton => (strength = :weak, ki_um = 5.0, mechanism = :reversible, auc_ratio = 1.8),
    ),
    :CYP2B6 => Dict(
        :ticlopidine => (strength = :moderate, ki_um = 0.5, mechanism = :mbi, auc_ratio = 2.5),
        :clopidogrel => (strength = :moderate, ki_um = 3.0, mechanism = :mbi, auc_ratio = 2.0),
    ),
    :CYP2C8 => Dict(
        :gemfibrozil => (strength = :strong, ki_um = 4.0, mechanism = :mbi, auc_ratio = 8.0),
        :clopidogrel_glucuronide => (strength = :strong, ki_um = 1.0, mechanism = :mbi, auc_ratio = 5.0),
        :trimethoprim => (strength = :weak, ki_um = 32.0, mechanism = :reversible, auc_ratio = 1.5),
    ),
    :CYP2C9 => Dict(
        :fluconazole => (strength = :moderate, ki_um = 7.0, mechanism = :reversible, auc_ratio = 2.5),
        :amiodarone => (strength = :moderate, ki_um = 2.0, mechanism = :reversible, auc_ratio = 2.2),
        :miconazole => (strength = :strong, ki_um = 0.5, mechanism = :reversible, auc_ratio = 5.0),
        :voriconazole => (strength = :moderate, ki_um = 3.0, mechanism = :reversible, auc_ratio = 2.5),
        :sulfaphenazole => (strength = :strong, ki_um = 0.3, mechanism = :reversible, auc_ratio = 8.0),
    ),
    :CYP2C19 => Dict(
        :fluvoxamine => (strength = :strong, ki_um = 0.1, mechanism = :reversible, auc_ratio = 6.0),
        :fluconazole => (strength = :moderate, ki_um = 5.0, mechanism = :reversible, auc_ratio = 3.0),
        :omeprazole => (strength = :moderate, ki_um = 2.0, mechanism = :reversible, auc_ratio = 2.5),
        :esomeprazole => (strength = :moderate, ki_um = 2.0, mechanism = :reversible, auc_ratio = 2.5),
        :ticlopidine => (strength = :moderate, ki_um = 3.0, mechanism = :mbi, auc_ratio = 2.5),
    ),
    :CYP2D6 => Dict(
        :paroxetine => (strength = :strong, ki_um = 0.01, mechanism = :mbi, auc_ratio = 8.0),
        :fluoxetine => (strength = :strong, ki_um = 0.02, mechanism = :mbi, auc_ratio = 8.0),
        :quinidine => (strength = :strong, ki_um = 0.05, mechanism = :reversible, auc_ratio = 10.0),
        :bupropion => (strength = :moderate, ki_um = 5.0, mechanism = :reversible, auc_ratio = 2.5),
        :duloxetine => (strength = :moderate, ki_um = 0.3, mechanism = :reversible, auc_ratio = 3.0),
        :terbinafine => (strength = :strong, ki_um = 0.03, mechanism = :mbi, auc_ratio = 5.0),
        :mirabegron => (strength = :moderate, ki_um = 3.0, mechanism = :reversible, auc_ratio = 2.0),
        :cinacalcet => (strength = :strong, ki_um = 0.1, mechanism = :reversible, auc_ratio = 7.0),
    ),
    :CYP3A4 => Dict(
        # Strong inhibitors
        :itraconazole => (strength = :strong, ki_um = 0.002, mechanism = :reversible, auc_ratio = 10.0),
        :ketoconazole => (strength = :strong, ki_um = 0.015, mechanism = :reversible, auc_ratio = 15.0),
        :posaconazole => (strength = :strong, ki_um = 0.01, mechanism = :reversible, auc_ratio = 8.0),
        :voriconazole => (strength = :strong, ki_um = 0.05, mechanism = :reversible, auc_ratio = 8.0),
        :clarithromycin => (strength = :strong, ki_um = 3.0, mechanism = :mbi, auc_ratio = 8.0),
        :ritonavir => (strength = :strong, ki_um = 0.02, mechanism = :mbi, auc_ratio = 30.0),
        :cobicistat => (strength = :strong, ki_um = 0.01, mechanism = :mbi, auc_ratio = 20.0),
        :indinavir => (strength = :strong, ki_um = 0.5, mechanism = :reversible, auc_ratio = 8.0),
        :nelfinavir => (strength = :strong, ki_um = 1.0, mechanism = :mbi, auc_ratio = 6.0),
        # Moderate inhibitors
        :erythromycin => (strength = :moderate, ki_um = 30.0, mechanism = :mbi, auc_ratio = 4.0),
        :fluconazole => (strength = :moderate, ki_um = 10.0, mechanism = :reversible, auc_ratio = 3.5),
        :diltiazem => (strength = :moderate, ki_um = 3.0, mechanism = :mbi, auc_ratio = 4.0),
        :verapamil => (strength = :moderate, ki_um = 5.0, mechanism = :reversible, auc_ratio = 3.0),
        :aprepitant => (strength = :moderate, ki_um = 2.0, mechanism = :reversible, auc_ratio = 3.0),
        :cimetidine => (strength = :weak, ki_um = 200.0, mechanism = :reversible, auc_ratio = 1.5),
        :grapefruit_juice => (strength = :moderate, ki_um = 0.0, mechanism = :mbi, auc_ratio = 3.0),
        # Weak inhibitors
        :fluvoxamine => (strength = :weak, ki_um = 15.0, mechanism = :reversible, auc_ratio = 1.8),
        :cilostazol => (strength = :weak, ki_um = 10.0, mechanism = :reversible, auc_ratio = 1.5),
        :ranitidine => (strength = :weak, ki_um = 300.0, mechanism = :reversible, auc_ratio = 1.3),
    ),
)

# =============================================================================
# CYP INDUCERS - With induction parameters
# =============================================================================
# Strong: ≥80% decrease in AUC of sensitive substrate
# Moderate: ≥50% but <80% decrease in AUC
# Weak: ≥20% but <50% decrease in AUC

"""
FDA-classified CYP inducers with induction parameters.
Emax = maximum fold induction, EC50 in μM.
"""
const FDA_CYP_INDUCERS = Dict{Symbol, Dict{Symbol, NamedTuple}}(
    :CYP1A2 => Dict(
        :smoking => (strength = :strong, emax = 3.0, ec50_um = 0.0, auc_decrease_pct = 60.0),
        :omeprazole => (strength = :weak, emax = 1.5, ec50_um = 5.0, auc_decrease_pct = 30.0),
        :chargrilled_meat => (strength = :weak, emax = 1.5, ec50_um = 0.0, auc_decrease_pct = 25.0),
    ),
    :CYP2B6 => Dict(
        :rifampin => (strength = :strong, emax = 10.0, ec50_um = 0.5, auc_decrease_pct = 80.0),
        :efavirenz => (strength = :moderate, emax = 3.0, ec50_um = 2.0, auc_decrease_pct = 50.0),
        :ritonavir => (strength = :weak, emax = 2.0, ec50_um = 1.0, auc_decrease_pct = 30.0),
        :carbamazepine => (strength = :moderate, emax = 4.0, ec50_um = 10.0, auc_decrease_pct = 60.0),
    ),
    :CYP2C8 => Dict(
        :rifampin => (strength = :strong, emax = 6.0, ec50_um = 0.5, auc_decrease_pct = 80.0),
    ),
    :CYP2C9 => Dict(
        :rifampin => (strength = :moderate, emax = 3.0, ec50_um = 0.5, auc_decrease_pct = 60.0),
        :carbamazepine => (strength = :weak, emax = 1.5, ec50_um = 10.0, auc_decrease_pct = 30.0),
        :st_johns_wort => (strength = :weak, emax = 1.5, ec50_um = 0.0, auc_decrease_pct = 25.0),
    ),
    :CYP2C19 => Dict(
        :rifampin => (strength = :strong, emax = 5.0, ec50_um = 0.5, auc_decrease_pct = 85.0),
        :st_johns_wort => (strength = :moderate, emax = 2.0, ec50_um = 0.0, auc_decrease_pct = 50.0),
    ),
    :CYP3A4 => Dict(
        :rifampin => (strength = :strong, emax = 15.0, ec50_um = 0.5, auc_decrease_pct = 90.0),
        :carbamazepine => (strength = :strong, emax = 6.0, ec50_um = 10.0, auc_decrease_pct = 85.0),
        :phenytoin => (strength = :strong, emax = 6.0, ec50_um = 20.0, auc_decrease_pct = 85.0),
        :phenobarbital => (strength = :strong, emax = 5.0, ec50_um = 50.0, auc_decrease_pct = 80.0),
        :efavirenz => (strength = :moderate, emax = 3.0, ec50_um = 2.0, auc_decrease_pct = 60.0),
        :modafinil => (strength = :weak, emax = 1.5, ec50_um = 20.0, auc_decrease_pct = 35.0),
        :st_johns_wort => (strength = :strong, emax = 4.0, ec50_um = 0.0, auc_decrease_pct = 80.0),
        :bosentan => (strength = :moderate, emax = 2.5, ec50_um = 5.0, auc_decrease_pct = 50.0),
        :aprepitant => (strength = :weak, emax = 1.3, ec50_um = 2.0, auc_decrease_pct = 25.0),
        :pioglitazone => (strength = :weak, emax = 1.3, ec50_um = 10.0, auc_decrease_pct = 20.0),
    ),
)

# =============================================================================
# TRANSPORTER DATA
# =============================================================================

"""
Transporter substrates and inhibitors from FDA guidance.
"""
const FDA_TRANSPORTER_SUBSTRATES = Dict{Symbol, Vector{Symbol}}(
    :P_gp => [:digoxin, :fexofenadine, :loperamide, :quinidine, :talinolol, :vinblastine, :dabigatran, :colchicine],
    :BCRP => [:rosuvastatin, :sulfasalazine, :topotecan, :methotrexate],
    :OATP1B1 => [:pitavastatin, :pravastatin, :rosuvastatin, :atorvastatin, :repaglinide, :bosentan],
    :OATP1B3 => [:pitavastatin, :pravastatin, :rosuvastatin, :telmisartan, :olmesartan],
    :OAT1 => [:adefovir, :cidofovir, :tenofovir, :ciprofloxacin, :methotrexate],
    :OAT3 => [:pravastatin, :methotrexate, :ciprofloxacin, :furosemide],
    :OCT2 => [:metformin, :creatinine, :cimetidine],
    :MATE1 => [:metformin, :creatinine],
    :MATE2K => [:metformin, :creatinine],
)

"""
Transporter inhibitors with Ki values in μM.
"""
const FDA_TRANSPORTER_INHIBITORS = Dict{Symbol, Dict{Symbol, NamedTuple}}(
    :P_gp => Dict(
        :cyclosporine => (ki_um = 0.5, auc_ratio = 2.5),
        :ketoconazole => (ki_um = 1.0, auc_ratio = 1.8),
        :quinidine => (ki_um = 2.0, auc_ratio = 2.0),
        :verapamil => (ki_um = 3.0, auc_ratio = 1.8),
        :ritonavir => (ki_um = 1.0, auc_ratio = 2.5),
        :dronedarone => (ki_um = 0.3, auc_ratio = 2.5),
        :amiodarone => (ki_um = 5.0, auc_ratio = 1.8),
        :itraconazole => (ki_um = 0.5, auc_ratio = 2.0),
        :clarithromycin => (ki_um = 10.0, auc_ratio = 1.7),
    ),
    :BCRP => Dict(
        :elacridar => (ki_um = 0.03, auc_ratio = 3.0),
        :curcumin => (ki_um = 0.5, auc_ratio = 2.0),
        :cyclosporine => (ki_um = 0.2, auc_ratio = 2.5),
        :pantoprazole => (ki_um = 30.0, auc_ratio = 1.3),
    ),
    :OATP1B1 => Dict(
        :cyclosporine => (ki_um = 0.02, auc_ratio = 10.0),
        :rifampin => (ki_um = 0.5, auc_ratio = 8.0),
        :gemfibrozil => (ki_um = 10.0, auc_ratio = 2.0),
        :ritonavir => (ki_um = 0.5, auc_ratio = 3.0),
        :lopinavir => (ki_um = 0.3, auc_ratio = 4.0),
        :eltrombopag => (ki_um = 0.8, auc_ratio = 2.0),
    ),
    :OATP1B3 => Dict(
        :cyclosporine => (ki_um = 0.01, auc_ratio = 10.0),
        :rifampin => (ki_um = 0.3, auc_ratio = 8.0),
        :ritonavir => (ki_um = 0.2, auc_ratio = 4.0),
    ),
    :OAT1 => Dict(
        :probenecid => (ki_um = 2.0, auc_ratio = 3.0),
    ),
    :OAT3 => Dict(
        :probenecid => (ki_um = 1.5, auc_ratio = 4.0),
        :teriflunomide => (ki_um = 0.3, auc_ratio = 2.0),
    ),
    :OCT2 => Dict(
        :cimetidine => (ki_um = 100.0, auc_ratio = 1.5),
        :dolutegravir => (ki_um = 1.5, auc_ratio = 1.8),
        :vandetanib => (ki_um = 1.0, auc_ratio = 2.0),
        :trimethoprim => (ki_um = 30.0, auc_ratio = 1.4),
    ),
    :MATE1 => Dict(
        :cimetidine => (ki_um = 1.0, auc_ratio = 1.5),
        :pyrimethamine => (ki_um = 0.05, auc_ratio = 1.5),
        :trimethoprim => (ki_um = 5.0, auc_ratio = 1.3),
    ),
)

# =============================================================================
# QUANTITATIVE fm VALUES - From clinical studies
# =============================================================================

"""
Fraction metabolized (fm) values from clinical DDI studies and in vitro data.
These are the gold-standard values for PBPK modeling.
"""
const QUANTITATIVE_FM_VALUES = Dict{Symbol, NamedTuple}(
    # CYP3A4 sensitive substrates
    :midazolam => (fm_3a4 = 0.96, fm_3a5 = 0.02, source = "Gorski 1994, Thummel 1996"),
    :triazolam => (fm_3a4 = 0.91, source = "Greenblatt 2000"),
    :alfentanil => (fm_3a4 = 0.90, source = "Kharasch 1997"),
    :buspirone => (fm_3a4 = 0.95, source = "Kivisto 1997"),
    :felodipine => (fm_3a4 = 0.90, source = "Edgar 1992"),
    :simvastatin => (fm_3a4 = 0.85, source = "Prueksaritanont 1997"),
    :lovastatin => (fm_3a4 = 0.85, source = "Kantola 1998"),
    :atorvastatin => (fm_3a4 = 0.50, source = "Jacobsen 2000"),
    :tacrolimus => (fm_3a4 = 0.85, fm_3a5 = 0.10, source = "Dai 2006"),
    :cyclosporine => (fm_3a4 = 0.70, source = "Frassetto 2007"),
    :nifedipine => (fm_3a4 = 0.75, source = "Holtbecker 1996"),

    # CYP2D6 sensitive substrates
    :desipramine => (fm_2d6 = 0.90, source = "Brosen 1986"),
    :dextromethorphan => (fm_2d6 = 0.85, source = "Capon 1996"),
    :atomoxetine => (fm_2d6 = 0.80, source = "Ring 2002"),
    :metoprolol => (fm_2d6 = 0.70, source = "Lennard 1982"),
    :codeine => (fm_2d6 = 0.10, source = "Desmeules 1991"),  # For O-demethylation to morphine
    :tramadol => (fm_2d6 = 0.30, fm_3a4 = 0.30, source = "Subrahmanyam 2001"),
    :tamoxifen => (fm_2d6 = 0.15, fm_3a4 = 0.65, source = "Desta 2004"),  # For endoxifen

    # CYP2C19 substrates
    :omeprazole => (fm_2c19 = 0.80, fm_3a4 = 0.15, source = "Furuta 2001"),
    :lansoprazole => (fm_2c19 = 0.75, fm_3a4 = 0.20, source = "Furuta 2001"),
    :clopidogrel => (fm_2c19 = 0.45, fm_3a4 = 0.20, fm_2b6 = 0.20, source = "Kazui 2010"),
    :voriconazole => (fm_2c19 = 0.70, fm_3a4 = 0.20, fm_2c9 = 0.05, source = "Hyland 2003"),

    # CYP2C9 substrates
    :warfarin_s => (fm_2c9 = 0.90, source = "Rettie 1992"),
    :tolbutamide => (fm_2c9 = 0.85, source = "Miners 1988"),
    :celecoxib => (fm_2c9 = 0.70, fm_3a4 = 0.20, source = "Tang 2000"),
    :losartan => (fm_2c9 = 0.50, fm_3a4 = 0.35, source = "Stearns 2003"),
    :phenytoin => (fm_2c9 = 0.80, fm_2c19 = 0.15, source = "Mamiya 1998"),

    # CYP2C8 substrates
    :repaglinide => (fm_2c8 = 0.60, fm_3a4 = 0.30, fm_oatp1b1 = 0.80, source = "Kajosaari 2005"),
    :rosiglitazone => (fm_2c8 = 0.65, fm_2c9 = 0.25, source = "Baldwin 1999"),
    :paclitaxel => (fm_2c8 = 0.60, fm_3a4 = 0.30, source = "Rahman 1994"),

    # CYP1A2 substrates
    :caffeine => (fm_1a2 = 0.95, source = "Kalow 1991"),
    :tizanidine => (fm_1a2 = 0.90, source = "Granfors 2004"),
    :theophylline => (fm_1a2 = 0.70, fm_3a4 = 0.20, source = "Fuhr 1992"),
    :melatonin => (fm_1a2 = 0.90, source = "Hartter 2001"),

    # CYP2B6 substrates
    :efavirenz => (fm_2b6 = 0.75, fm_2a6 = 0.15, source = "Ward 2003"),
    :bupropion => (fm_2b6 = 0.85, source = "Faucette 2000"),
)

# =============================================================================
# SUMMARY COUNTS
# =============================================================================

const FDA_DATABASE_SUMMARY = (
    cyp_substrates = sum(length(v) for v in values(FDA_CYP_SUBSTRATES)),
    cyp_inhibitors = sum(length(v) for v in values(FDA_CYP_INHIBITORS)),
    cyp_inducers = sum(length(v) for v in values(FDA_CYP_INDUCERS)),
    transporter_inhibitors = sum(length(v) for v in values(FDA_TRANSPORTER_INHIBITORS)),
    fm_values = length(QUANTITATIVE_FM_VALUES),
)

# Export summary
println("FDA DDI Database loaded:")
println("  • $(FDA_DATABASE_SUMMARY.cyp_substrates) CYP substrates")
println("  • $(FDA_DATABASE_SUMMARY.cyp_inhibitors) CYP inhibitors with Ki values")
println("  • $(FDA_DATABASE_SUMMARY.cyp_inducers) CYP inducers with Emax/EC50")
println("  • $(FDA_DATABASE_SUMMARY.transporter_inhibitors) transporter inhibitors")
println("  • $(FDA_DATABASE_SUMMARY.fm_values) quantitative fm values")
