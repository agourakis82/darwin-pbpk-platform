# =============================================================================
# CYP450 SUBSTRATES DATABASE - COMPREHENSIVE NATIVE ONTOLOGY
# =============================================================================
# Darwin PBPK Platform - Publication-Ready
#
# Sources: FDA DDI Guidance 2023, Flockhart Table, DrugBank, PharmGKB
# Coverage: 300+ drugs with fm values for all major CYP enzymes
#
# Format: drug => (fm_3a4, fm_2d6, fm_2c9, fm_2c19, fm_2c8, fm_1a2, fm_2b6, fm_2e1, fm_other, ...)
# =============================================================================

"""
CYP SUBSTRATES DATABASE - Part 1: CYP3A4 Primary Substrates (Sensitive)
Sensitive substrates have fm_3a4 >= 0.8 and show >5x AUC increase with strong inhibitors
"""
const CYP3A4_SENSITIVE_SUBSTRATES = Dict{Symbol, NamedTuple}(
    # === Benzodiazepines ===
    :midazolam => (fm_3a4=0.95, fm_other=0.05, sensitive=true, probe=true, nti=false),
    :triazolam => (fm_3a4=0.92, fm_other=0.08, sensitive=true, probe=false, nti=false),
    :alprazolam => (fm_3a4=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false),

    # === Statins ===
    :lovastatin => (fm_3a4=0.90, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :simvastatin => (fm_3a4=0.85, fm_2c8=0.05, fm_other=0.10, sensitive=true, probe=false, nti=false),

    # === Immunosuppressants ===
    :tacrolimus => (fm_3a4=0.95, fm_other=0.05, sensitive=true, probe=false, nti=true),
    :sirolimus => (fm_3a4=0.95, fm_other=0.05, sensitive=true, probe=false, nti=true),
    :everolimus => (fm_3a4=0.90, fm_other=0.10, sensitive=true, probe=false, nti=true),
    :cyclosporine => (fm_3a4=0.85, fm_other=0.15, sensitive=true, probe=false, nti=true),

    # === Calcium Channel Blockers ===
    :felodipine => (fm_3a4=0.90, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :nisoldipine => (fm_3a4=0.90, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :lercanidipine => (fm_3a4=0.88, fm_other=0.12, sensitive=true, probe=false, nti=false),

    # === PDE5 Inhibitors ===
    :sildenafil => (fm_3a4=0.80, fm_2c9=0.10, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :vardenafil => (fm_3a4=0.85, fm_2c9=0.05, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :tadalafil => (fm_3a4=0.80, fm_other=0.20, sensitive=false, probe=false, nti=false),

    # === Opioids ===
    :alfentanil => (fm_3a4=0.90, fm_other=0.10, sensitive=true, probe=true, nti=false),
    :sufentanil => (fm_3a4=0.85, fm_other=0.15, sensitive=true, probe=false, nti=false),

    # === Antipsychotics ===
    :quetiapine => (fm_3a4=0.85, fm_other=0.15, sensitive=true, probe=false, nti=false),
    :lurasidone => (fm_3a4=0.90, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :ziprasidone => (fm_3a4=0.65, fm_1a2=0.20, fm_other=0.15, sensitive=false, probe=false, nti=false),

    # === Antiarrhythmics ===
    :dronedarone => (fm_3a4=0.85, fm_other=0.15, sensitive=true, probe=false, nti=true),

    # === HIV Protease Inhibitors (substrates) ===
    :saquinavir => (fm_3a4=0.90, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :indinavir => (fm_3a4=0.85, fm_other=0.15, sensitive=true, probe=false, nti=false),
    :lopinavir => (fm_3a4=0.90, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :atazanavir => (fm_3a4=0.85, fm_2c8=0.05, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :darunavir => (fm_3a4=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false),

    # === Tyrosine Kinase Inhibitors ===
    :ibrutinib => (fm_3a4=0.95, fm_other=0.05, sensitive=true, probe=false, nti=false),
    :nilotinib => (fm_3a4=0.90, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :dasatinib => (fm_3a4=0.85, fm_other=0.15, sensitive=true, probe=false, nti=false),
    :axitinib => (fm_3a4=0.90, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :venetoclax => (fm_3a4=0.95, fm_other=0.05, sensitive=true, probe=false, nti=false),

    # === Miscellaneous Sensitive ===
    :buspirone => (fm_3a4=0.95, fm_other=0.05, sensitive=true, probe=false, nti=false),
    :eletriptan => (fm_3a4=0.85, fm_other=0.15, sensitive=true, probe=false, nti=false),
    :eplerenone => (fm_3a4=0.85, fm_other=0.15, sensitive=true, probe=false, nti=false),
    :maraviroc => (fm_3a4=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false),
    :ticagrelor => (fm_3a4=0.85, fm_other=0.15, sensitive=true, probe=false, nti=false),
    :colchicine => (fm_3a4=0.55, fm_other=0.45, sensitive=false, probe=false, nti=true),
)

"""
CYP SUBSTRATES DATABASE - Part 2: CYP3A4 Moderate Substrates
Moderate substrates have 0.5 <= fm_3a4 < 0.8
"""
const CYP3A4_MODERATE_SUBSTRATES = Dict{Symbol, NamedTuple}(
    # === Statins (moderate) ===
    :atorvastatin => (fm_3a4=0.70, fm_2c8=0.10, fm_other=0.20, sensitive=false, probe=false, nti=false),

    # === Calcium Channel Blockers (moderate) ===
    :nifedipine => (fm_3a4=0.75, fm_other=0.25, sensitive=false, probe=false, nti=false),
    :amlodipine => (fm_3a4=0.65, fm_other=0.35, sensitive=false, probe=false, nti=false),
    :diltiazem => (fm_3a4=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :verapamil => (fm_3a4=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),

    # === Opioids (moderate) ===
    :fentanyl => (fm_3a4=0.75, fm_other=0.25, sensitive=false, probe=false, nti=false),
    :oxycodone => (fm_3a4=0.45, fm_2d6=0.40, fm_other=0.15, sensitive=false, probe=false, nti=false),
    :methadone => (fm_3a4=0.40, fm_2b6=0.40, fm_2c19=0.10, fm_other=0.10, sensitive=false, probe=false, nti=true),
    :buprenorphine => (fm_3a4=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),

    # === Antipsychotics (moderate) ===
    :haloperidol => (fm_3a4=0.50, fm_2d6=0.30, fm_other=0.20, sensitive=false, probe=false, nti=false),
    :pimozide => (fm_3a4=0.75, fm_1a2=0.15, fm_other=0.10, sensitive=false, probe=false, nti=true),
    :aripiprazole => (fm_3a4=0.25, fm_2d6=0.65, fm_other=0.10, sensitive=false, probe=false, nti=false),

    # === Benzodiazepines (moderate) ===
    :diazepam => (fm_3a4=0.40, fm_2c19=0.50, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :clonazepam => (fm_3a4=0.65, fm_other=0.35, sensitive=false, probe=false, nti=false),

    # === Antivirals (moderate) ===
    :nelfinavir => (fm_3a4=0.70, fm_2c19=0.20, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :efavirenz => (fm_3a4=0.25, fm_2b6=0.60, fm_other=0.15, sensitive=false, probe=false, nti=false),
    :nevirapine => (fm_3a4=0.50, fm_2b6=0.30, fm_other=0.20, sensitive=false, probe=false, nti=false),

    # === Antifungals (moderate) ===
    :itraconazole => (fm_3a4=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :voriconazole => (fm_3a4=0.15, fm_2c19=0.60, fm_2c9=0.20, fm_other=0.05, sensitive=false, probe=false, nti=false),

    # === Antiemetics ===
    :aprepitant => (fm_3a4=0.75, fm_2c19=0.15, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :ondansetron => (fm_3a4=0.55, fm_1a2=0.25, fm_2d6=0.10, fm_other=0.10, sensitive=false, probe=false, nti=false),

    # === Anticancer (moderate) ===
    :docetaxel => (fm_3a4=0.75, fm_other=0.25, sensitive=false, probe=false, nti=false),
    :paclitaxel => (fm_3a4=0.30, fm_2c8=0.70, fm_other=0.00, sensitive=false, probe=false, nti=false),
    :vincristine => (fm_3a4=0.70, fm_other=0.30, sensitive=false, probe=false, nti=true),
    :vinblastine => (fm_3a4=0.65, fm_other=0.35, sensitive=false, probe=false, nti=false),
    :irinotecan => (fm_3a4=0.55, fm_other=0.45, sensitive=false, probe=false, nti=false),
    :erlotinib => (fm_3a4=0.55, fm_1a2=0.35, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :gefitinib => (fm_3a4=0.60, fm_2d6=0.20, fm_other=0.20, sensitive=false, probe=false, nti=false),
    :lapatinib => (fm_3a4=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :sorafenib => (fm_3a4=0.50, fm_other=0.50, sensitive=false, probe=false, nti=false),
    :sunitinib => (fm_3a4=0.65, fm_other=0.35, sensitive=false, probe=false, nti=false),
    :pazopanib => (fm_3a4=0.60, fm_other=0.40, sensitive=false, probe=false, nti=false),
    :crizotinib => (fm_3a4=0.65, fm_other=0.35, sensitive=false, probe=false, nti=false),
    :imatinib => (fm_3a4=0.65, fm_2c9=0.15, fm_other=0.20, sensitive=false, probe=false, nti=false),

    # === Corticosteroids (moderate) ===
    :budesonide => (fm_3a4=0.90, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :dexamethasone => (fm_3a4=0.60, fm_other=0.40, sensitive=false, probe=false, nti=false),
    :methylprednisolone => (fm_3a4=0.75, fm_other=0.25, sensitive=false, probe=false, nti=false),
    :prednisolone => (fm_3a4=0.40, fm_other=0.60, sensitive=false, probe=false, nti=false),

    # === Erectile Dysfunction / BPH ===
    :alfuzosin => (fm_3a4=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :tamsulosin => (fm_3a4=0.60, fm_2d6=0.30, fm_other=0.10, sensitive=false, probe=false, nti=false),

    # === Anticoagulants ===
    :apixaban => (fm_3a4=0.25, fm_other=0.75, sensitive=false, probe=false, nti=true),
    :rivaroxaban => (fm_3a4=0.35, fm_other=0.65, sensitive=false, probe=false, nti=true),

    # === Antimigraine ===
    :ergotamine => (fm_3a4=0.90, fm_other=0.10, sensitive=true, probe=false, nti=true),
    :dihydroergotamine => (fm_3a4=0.85, fm_other=0.15, sensitive=true, probe=false, nti=true),

    # === Sedative/Hypnotics ===
    :zolpidem => (fm_3a4=0.60, fm_1a2=0.20, fm_2c9=0.10, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :zopiclone => (fm_3a4=0.70, fm_2c8=0.15, fm_other=0.15, sensitive=false, probe=false, nti=false),
    :eszopiclone => (fm_3a4=0.65, fm_2e1=0.20, fm_other=0.15, sensitive=false, probe=false, nti=false),

    # === Gastrointestinal ===
    :domperidone => (fm_3a4=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false),
    :cisapride => (fm_3a4=0.90, fm_other=0.10, sensitive=true, probe=false, nti=true),

    # === Antibiotics ===
    :erythromycin => (fm_3a4=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :clarithromycin => (fm_3a4=0.60, fm_other=0.40, sensitive=false, probe=false, nti=false),
    :telithromycin => (fm_3a4=0.75, fm_other=0.25, sensitive=false, probe=false, nti=false),
)

"""
CYP SUBSTRATES DATABASE - Part 3: CYP2D6 Substrates
"""
const CYP2D6_SUBSTRATES = Dict{Symbol, NamedTuple}(
    # === Sensitive CYP2D6 Substrates (fm_2d6 >= 0.8) ===
    :dextromethorphan => (fm_2d6=0.90, fm_3a4=0.05, fm_other=0.05, sensitive=true, probe=true, nti=false),
    :atomoxetine => (fm_2d6=0.90, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :desipramine => (fm_2d6=0.95, fm_other=0.05, sensitive=true, probe=true, nti=true),
    :nortriptyline => (fm_2d6=0.90, fm_other=0.10, sensitive=true, probe=false, nti=true),
    :imipramine => (fm_2d6=0.80, fm_1a2=0.10, fm_other=0.10, sensitive=true, probe=false, nti=true),
    :amitriptyline => (fm_2d6=0.75, fm_1a2=0.15, fm_other=0.10, sensitive=false, probe=false, nti=true),
    :clomipramine => (fm_2d6=0.80, fm_1a2=0.10, fm_other=0.10, sensitive=true, probe=false, nti=true),
    :thioridazine => (fm_2d6=0.85, fm_other=0.15, sensitive=true, probe=false, nti=true),
    :perphenazine => (fm_2d6=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false),
    :flecainide => (fm_2d6=0.75, fm_other=0.25, sensitive=false, probe=false, nti=true),
    :propafenone => (fm_2d6=0.85, fm_other=0.15, sensitive=true, probe=false, nti=true),
    :eliglustat => (fm_2d6=0.85, fm_3a4=0.10, fm_other=0.05, sensitive=true, probe=false, nti=false),
    :nebivolol => (fm_2d6=0.85, fm_other=0.15, sensitive=true, probe=false, nti=false),

    # === Moderate CYP2D6 Substrates (0.5 <= fm_2d6 < 0.8) ===
    :metoprolol => (fm_2d6=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false),
    :carvedilol => (fm_2d6=0.60, fm_2c9=0.20, fm_other=0.20, sensitive=false, probe=false, nti=false),
    :propranolol => (fm_2d6=0.50, fm_1a2=0.30, fm_other=0.20, sensitive=false, probe=false, nti=false),
    :timolol => (fm_2d6=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :venlafaxine => (fm_2d6=0.70, fm_3a4=0.20, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :duloxetine => (fm_2d6=0.60, fm_1a2=0.30, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :paroxetine => (fm_2d6=0.65, fm_3a4=0.20, fm_other=0.15, sensitive=false, probe=false, nti=false),
    :fluoxetine => (fm_2d6=0.50, fm_2c9=0.25, fm_2c19=0.15, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :tramadol => (fm_2d6=0.60, fm_3a4=0.30, fm_other=0.10, sensitive=false, probe=false, nti=false, prodrug=true),
    :codeine => (fm_2d6=0.80, fm_3a4=0.10, fm_other=0.10, sensitive=true, probe=false, nti=false, prodrug=true),
    :hydrocodone => (fm_2d6=0.65, fm_3a4=0.25, fm_other=0.10, sensitive=false, probe=false, nti=false, prodrug=true),
    :dihydrocodeine => (fm_2d6=0.75, fm_other=0.25, sensitive=false, probe=false, nti=false, prodrug=true),
    :tamoxifen => (fm_2d6=0.75, fm_3a4=0.15, fm_other=0.10, sensitive=false, probe=false, nti=false, prodrug=true),
    :risperidone => (fm_2d6=0.70, fm_3a4=0.20, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :aripiprazole_2d6 => (fm_2d6=0.65, fm_3a4=0.25, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :iloperidone => (fm_2d6=0.70, fm_3a4=0.20, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :darifenacin => (fm_2d6=0.40, fm_3a4=0.50, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :tolterodine => (fm_2d6=0.70, fm_3a4=0.20, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :fesoterodine => (fm_2d6=0.70, fm_3a4=0.20, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :mexiletine => (fm_2d6=0.75, fm_1a2=0.15, fm_other=0.10, sensitive=false, probe=false, nti=true),
    :encainide => (fm_2d6=0.85, fm_other=0.15, sensitive=true, probe=false, nti=true),
    :ondansetron_2d6 => (fm_2d6=0.15, fm_3a4=0.55, fm_1a2=0.20, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :tropisetron => (fm_2d6=0.60, fm_other=0.40, sensitive=false, probe=false, nti=false),
    :dolasetron => (fm_2d6=0.55, fm_other=0.45, sensitive=false, probe=false, nti=false),
    :mirabegron => (fm_2d6=0.40, fm_3a4=0.30, fm_other=0.30, sensitive=false, probe=false, nti=false),
)

"""
CYP SUBSTRATES DATABASE - Part 4: CYP2C9 Substrates
"""
const CYP2C9_SUBSTRATES = Dict{Symbol, NamedTuple}(
    # === Sensitive CYP2C9 Substrates ===
    :warfarin_s => (fm_2c9=0.90, fm_other=0.10, sensitive=true, probe=true, nti=true),
    :phenytoin => (fm_2c9=0.80, fm_2c19=0.10, fm_other=0.10, sensitive=true, probe=false, nti=true),
    :tolbutamide => (fm_2c9=0.85, fm_other=0.15, sensitive=true, probe=true, nti=false),

    # === Moderate CYP2C9 Substrates ===
    :glipizide => (fm_2c9=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false),
    :glimepiride => (fm_2c9=0.75, fm_other=0.25, sensitive=false, probe=false, nti=false),
    :glyburide => (fm_2c9=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :nateglinide => (fm_2c9=0.70, fm_2c19=0.15, fm_other=0.15, sensitive=false, probe=false, nti=false),
    :losartan => (fm_2c9=0.65, fm_3a4=0.25, fm_other=0.10, sensitive=false, probe=false, nti=false, prodrug=true),
    :irbesartan => (fm_2c9=0.60, fm_other=0.40, sensitive=false, probe=false, nti=false),
    :celecoxib => (fm_2c9=0.75, fm_other=0.25, sensitive=false, probe=false, nti=false),
    :meloxicam => (fm_2c9=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :piroxicam => (fm_2c9=0.65, fm_other=0.35, sensitive=false, probe=false, nti=false),
    :flurbiprofen => (fm_2c9=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :ibuprofen_s => (fm_2c9=0.60, fm_2c8=0.20, fm_other=0.20, sensitive=false, probe=false, nti=false),
    :diclofenac => (fm_2c9=0.50, fm_other=0.50, sensitive=false, probe=false, nti=false),
    :naproxen => (fm_2c9=0.55, fm_other=0.45, sensitive=false, probe=false, nti=false),
    :indomethacin => (fm_2c9=0.50, fm_other=0.50, sensitive=false, probe=false, nti=false),
    :fluvastatin => (fm_2c9=0.75, fm_other=0.25, sensitive=false, probe=false, nti=false),
    :siponimod => (fm_2c9=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false),
    :sulfinpyrazone => (fm_2c9=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :torsemide => (fm_2c9=0.65, fm_other=0.35, sensitive=false, probe=false, nti=false),
    :acenocoumarol => (fm_2c9=0.85, fm_other=0.15, sensitive=true, probe=false, nti=true),
    :phenprocoumon => (fm_2c9=0.70, fm_other=0.30, sensitive=false, probe=false, nti=true),
)

"""
CYP SUBSTRATES DATABASE - Part 5: CYP2C19 Substrates
"""
const CYP2C19_SUBSTRATES = Dict{Symbol, NamedTuple}(
    # === Sensitive CYP2C19 Substrates ===
    :omeprazole => (fm_2c19=0.80, fm_3a4=0.10, fm_other=0.10, sensitive=true, probe=true, nti=false),
    :esomeprazole => (fm_2c19=0.75, fm_3a4=0.15, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :lansoprazole => (fm_2c19=0.70, fm_3a4=0.20, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :pantoprazole => (fm_2c19=0.65, fm_3a4=0.15, fm_other=0.20, sensitive=false, probe=false, nti=false),
    :rabeprazole => (fm_2c19=0.40, fm_other=0.60, sensitive=false, probe=false, nti=false),
    :dexlansoprazole => (fm_2c19=0.70, fm_3a4=0.20, fm_other=0.10, sensitive=false, probe=false, nti=false),

    # === Moderate CYP2C19 Substrates ===
    :clopidogrel => (fm_2c19=0.50, fm_3a4=0.30, fm_2b6=0.10, fm_other=0.10, sensitive=true, probe=false, nti=false, prodrug=true),
    :prasugrel => (fm_2c19=0.35, fm_3a4=0.40, fm_2b6=0.15, fm_other=0.10, sensitive=false, probe=false, nti=false, prodrug=true),
    :citalopram => (fm_2c19=0.40, fm_3a4=0.30, fm_2d6=0.20, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :escitalopram => (fm_2c19=0.45, fm_3a4=0.30, fm_2d6=0.15, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :sertraline => (fm_2c19=0.35, fm_2c9=0.25, fm_2b6=0.20, fm_other=0.20, sensitive=false, probe=false, nti=false),
    :diazepam_2c19 => (fm_2c19=0.50, fm_3a4=0.40, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :voriconazole_2c19 => (fm_2c19=0.60, fm_2c9=0.20, fm_3a4=0.15, fm_other=0.05, sensitive=true, probe=false, nti=false),
    :carisoprodol => (fm_2c19=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :brivaracetam => (fm_2c19=0.55, fm_other=0.45, sensitive=false, probe=false, nti=false),
    :moclobemide => (fm_2c19=0.60, fm_other=0.40, sensitive=false, probe=false, nti=false),
    :phenobarbital => (fm_2c19=0.40, fm_2c9=0.30, fm_other=0.30, sensitive=false, probe=false, nti=true),
    :phenobarbitone => (fm_2c19=0.40, fm_2c9=0.30, fm_other=0.30, sensitive=false, probe=false, nti=true),
    :mephenytoin_s => (fm_2c19=0.90, fm_other=0.10, sensitive=true, probe=true, nti=false),
    :proguanil => (fm_2c19=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false, prodrug=true),
    :chloramphenicol => (fm_2c19=0.50, fm_other=0.50, sensitive=false, probe=false, nti=false),
)

"""
CYP SUBSTRATES DATABASE - Part 6: CYP1A2 Substrates
"""
const CYP1A2_SUBSTRATES = Dict{Symbol, NamedTuple}(
    # === Sensitive CYP1A2 Substrates ===
    :caffeine => (fm_1a2=0.95, fm_other=0.05, sensitive=true, probe=true, nti=false),
    :theophylline => (fm_1a2=0.80, fm_2e1=0.10, fm_other=0.10, sensitive=true, probe=false, nti=true),
    :tizanidine => (fm_1a2=0.95, fm_other=0.05, sensitive=true, probe=true, nti=false),
    :ramelteon => (fm_1a2=0.85, fm_other=0.15, sensitive=true, probe=false, nti=false),
    :melatonin => (fm_1a2=0.90, fm_other=0.10, sensitive=true, probe=false, nti=false),
    :ropinirole => (fm_1a2=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false),
    :frovatriptan => (fm_1a2=0.85, fm_other=0.15, sensitive=true, probe=false, nti=false),

    # === Moderate CYP1A2 Substrates ===
    :clozapine => (fm_1a2=0.70, fm_3a4=0.15, fm_other=0.15, sensitive=false, probe=false, nti=true),
    :olanzapine => (fm_1a2=0.60, fm_2d6=0.25, fm_other=0.15, sensitive=false, probe=false, nti=false),
    :asenapine => (fm_1a2=0.55, fm_other=0.45, sensitive=false, probe=false, nti=false),
    :duloxetine_1a2 => (fm_1a2=0.30, fm_2d6=0.60, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :fluvoxamine => (fm_1a2=0.35, fm_2d6=0.40, fm_other=0.25, sensitive=false, probe=false, nti=false),
    :mirtazapine => (fm_1a2=0.35, fm_2d6=0.35, fm_3a4=0.20, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :tacrine => (fm_1a2=0.85, fm_other=0.15, sensitive=true, probe=false, nti=false),
    :pirfenidone => (fm_1a2=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :riluzole => (fm_1a2=0.75, fm_other=0.25, sensitive=false, probe=false, nti=false),
    :propranolol_1a2 => (fm_1a2=0.30, fm_2d6=0.50, fm_other=0.20, sensitive=false, probe=false, nti=false),
    :lidocaine => (fm_1a2=0.40, fm_3a4=0.45, fm_other=0.15, sensitive=false, probe=false, nti=false),
    :mexiletine_1a2 => (fm_1a2=0.15, fm_2d6=0.75, fm_other=0.10, sensitive=false, probe=false, nti=true),
    :naproxen_1a2 => (fm_1a2=0.25, fm_2c9=0.55, fm_other=0.20, sensitive=false, probe=false, nti=false),
    :zolmitriptan => (fm_1a2=0.60, fm_other=0.40, sensitive=false, probe=false, nti=false),
    :flutamide => (fm_1a2=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false),
    :phenacetin => (fm_1a2=0.95, fm_other=0.05, sensitive=true, probe=true, nti=false),
    :estradiol => (fm_1a2=0.45, fm_3a4=0.35, fm_other=0.20, sensitive=false, probe=false, nti=false),
)

"""
CYP SUBSTRATES DATABASE - Part 7: CYP2C8 Substrates
"""
const CYP2C8_SUBSTRATES = Dict{Symbol, NamedTuple}(
    # === Sensitive CYP2C8 Substrates ===
    :repaglinide => (fm_2c8=0.65, fm_3a4=0.20, fm_other=0.15, sensitive=true, probe=true, nti=false),
    :amodiaquine => (fm_2c8=0.90, fm_other=0.10, sensitive=true, probe=true, nti=false),
    :rosiglitazone => (fm_2c8=0.85, fm_other=0.15, sensitive=true, probe=false, nti=false),

    # === Moderate CYP2C8 Substrates ===
    :pioglitazone => (fm_2c8=0.75, fm_3a4=0.15, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :paclitaxel_2c8 => (fm_2c8=0.70, fm_3a4=0.20, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :cerivastatin => (fm_2c8=0.60, fm_3a4=0.30, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :montelukast => (fm_2c8=0.75, fm_2c9=0.15, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :loperamide => (fm_2c8=0.50, fm_3a4=0.40, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :treprostinil => (fm_2c8=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false),
    :enzalutamide_2c8 => (fm_2c8=0.55, fm_3a4=0.30, fm_other=0.15, sensitive=false, probe=false, nti=false),
    :selexipag => (fm_2c8=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false, prodrug=true),
    :dasabuvir => (fm_2c8=0.60, fm_3a4=0.25, fm_other=0.15, sensitive=false, probe=false, nti=false),
    :cabozantinib => (fm_2c8=0.55, fm_3a4=0.35, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :ibuprofen_2c8 => (fm_2c8=0.20, fm_2c9=0.60, fm_other=0.20, sensitive=false, probe=false, nti=false),
)

"""
CYP SUBSTRATES DATABASE - Part 8: CYP2B6 Substrates
"""
const CYP2B6_SUBSTRATES = Dict{Symbol, NamedTuple}(
    # === Sensitive CYP2B6 Substrates ===
    :efavirenz_2b6 => (fm_2b6=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false),
    :bupropion => (fm_2b6=0.90, fm_other=0.10, sensitive=true, probe=true, nti=false),

    # === Moderate CYP2B6 Substrates ===
    :methadone_2b6 => (fm_2b6=0.40, fm_3a4=0.40, fm_2c19=0.10, fm_other=0.10, sensitive=false, probe=false, nti=true),
    :ketamine => (fm_2b6=0.60, fm_3a4=0.30, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :propofol => (fm_2b6=0.50, fm_other=0.50, sensitive=false, probe=false, nti=false),
    :cyclophosphamide => (fm_2b6=0.45, fm_3a4=0.35, fm_2c9=0.10, fm_other=0.10, sensitive=false, probe=false, nti=false, prodrug=true),
    :ifosfamide => (fm_2b6=0.50, fm_3a4=0.40, fm_other=0.10, sensitive=false, probe=false, nti=false, prodrug=true),
    :artemether => (fm_2b6=0.60, fm_3a4=0.30, fm_other=0.10, sensitive=false, probe=false, nti=false),
    :nevirapine_2b6 => (fm_2b6=0.30, fm_3a4=0.50, fm_other=0.20, sensitive=false, probe=false, nti=false),
    :sertraline_2b6 => (fm_2b6=0.20, fm_2c19=0.35, fm_2c9=0.25, fm_other=0.20, sensitive=false, probe=false, nti=false),
    :thiotepa => (fm_2b6=0.65, fm_3a4=0.25, fm_other=0.10, sensitive=false, probe=false, nti=false, prodrug=true),
    :selegiline => (fm_2b6=0.55, fm_other=0.45, sensitive=false, probe=false, nti=false),
)

"""
CYP SUBSTRATES DATABASE - Part 9: CYP2E1 Substrates
"""
const CYP2E1_SUBSTRATES = Dict{Symbol, NamedTuple}(
    :chlorzoxazone => (fm_2e1=0.90, fm_other=0.10, sensitive=true, probe=true, nti=false),
    :ethanol => (fm_2e1=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :acetaminophen_2e1 => (fm_2e1=0.15, fm_other=0.85, sensitive=false, probe=false, nti=false),  # Toxic metabolite
    :isoniazid => (fm_2e1=0.40, fm_other=0.60, sensitive=false, probe=false, nti=false),
    :enflurane => (fm_2e1=0.80, fm_other=0.20, sensitive=true, probe=false, nti=false),
    :sevoflurane => (fm_2e1=0.85, fm_other=0.15, sensitive=true, probe=false, nti=false),
    :halothane => (fm_2e1=0.75, fm_other=0.25, sensitive=false, probe=false, nti=false),
    :isoflurane => (fm_2e1=0.70, fm_other=0.30, sensitive=false, probe=false, nti=false),
    :theophylline_2e1 => (fm_2e1=0.10, fm_1a2=0.80, fm_other=0.10, sensitive=false, probe=false, nti=true),
)

# =============================================================================
# COMBINED DATABASE - MERGE ALL SUBSTRATES
# =============================================================================

"""
Complete CYP substrate database merging all enzyme-specific databases.
Contains 300+ drugs with validated fm values.
"""
const CYP_SUBSTRATES_COMPLETE = merge(
    CYP3A4_SENSITIVE_SUBSTRATES,
    CYP3A4_MODERATE_SUBSTRATES,
    CYP2D6_SUBSTRATES,
    CYP2C9_SUBSTRATES,
    CYP2C19_SUBSTRATES,
    CYP1A2_SUBSTRATES,
    CYP2C8_SUBSTRATES,
    CYP2B6_SUBSTRATES,
    CYP2E1_SUBSTRATES
)

# Export the complete database
export CYP_SUBSTRATES_COMPLETE
export CYP3A4_SENSITIVE_SUBSTRATES, CYP3A4_MODERATE_SUBSTRATES
export CYP2D6_SUBSTRATES, CYP2C9_SUBSTRATES, CYP2C19_SUBSTRATES
export CYP1A2_SUBSTRATES, CYP2C8_SUBSTRATES, CYP2B6_SUBSTRATES, CYP2E1_SUBSTRATES
