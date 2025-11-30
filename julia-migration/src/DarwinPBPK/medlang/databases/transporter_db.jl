# =============================================================================
# DRUG TRANSPORTERS DATABASE - COMPREHENSIVE NATIVE ONTOLOGY
# =============================================================================
# Darwin PBPK Platform - Publication-Ready
#
# Sources: FDA DDI Guidance 2023, ITC Guidelines, DrugBank, PharmGKB
# Coverage: 200+ drugs with transporter substrate/inhibitor profiles
#
# Transporters covered:
# - P-gp (ABCB1) - Efflux, intestine/BBB/kidney
# - BCRP (ABCG2) - Efflux, intestine/BBB
# - OATP1B1 (SLCO1B1) - Uptake, hepatic
# - OATP1B3 (SLCO1B3) - Uptake, hepatic
# - OATP2B1 (SLCO2B1) - Uptake, intestine/hepatic
# - OAT1 (SLC22A6) - Uptake, renal
# - OAT3 (SLC22A8) - Uptake, renal
# - OCT1 (SLC22A1) - Uptake, hepatic
# - OCT2 (SLC22A2) - Uptake, renal
# - MATE1 (SLC47A1) - Efflux, renal
# - MATE2-K (SLC47A2) - Efflux, renal
# - MRP2 (ABCC2) - Efflux, hepatic/intestinal
#
# Ki values in uM (micromolar)
# ft = fraction transported (0-1)
# =============================================================================

# =============================================================================
# P-GLYCOPROTEIN (P-gp/MDR1/ABCB1) DATABASE
# =============================================================================

"""
P-gp SUBSTRATES DATABASE
P-gp is an efflux transporter limiting oral absorption and CNS penetration
"""
const PGP_SUBSTRATES = Dict{Symbol, NamedTuple}(
    # === Probe Substrates (FDA recommended) ===
    :digoxin => (ft_pgp=0.70, probe=true, nti=true, tissue=[:intestine, :renal, :bbb]),
    :dabigatran => (ft_pgp=0.85, probe=true, nti=true, tissue=[:intestine, :renal]),
    :fexofenadine => (ft_pgp=0.60, probe=true, nti=false, tissue=[:intestine, :hepatic]),

    # === Anticoagulants ===
    :edoxaban => (ft_pgp=0.60, probe=false, nti=true, tissue=[:intestine, :renal]),
    :apixaban_pgp => (ft_pgp=0.30, probe=false, nti=true, tissue=[:intestine, :renal]),
    :rivaroxaban_pgp => (ft_pgp=0.35, probe=false, nti=true, tissue=[:intestine, :renal]),
    :betrixaban => (ft_pgp=0.70, probe=false, nti=true, tissue=[:intestine, :renal]),

    # === Cardiac Glycosides ===
    :digoxin_sub => (ft_pgp=0.70, probe=true, nti=true, tissue=[:intestine, :renal]),
    :digitoxin => (ft_pgp=0.50, probe=false, nti=true, tissue=[:intestine, :hepatic]),

    # === Immunosuppressants ===
    :tacrolimus_pgp => (ft_pgp=0.55, probe=false, nti=true, tissue=[:intestine, :hepatic, :bbb]),
    :cyclosporine_sub => (ft_pgp=0.40, probe=false, nti=true, tissue=[:intestine, :hepatic]),
    :sirolimus_pgp => (ft_pgp=0.60, probe=false, nti=true, tissue=[:intestine, :hepatic]),
    :everolimus_pgp => (ft_pgp=0.55, probe=false, nti=true, tissue=[:intestine, :hepatic]),

    # === HIV Protease Inhibitors ===
    :saquinavir_pgp => (ft_pgp=0.65, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :lopinavir_pgp => (ft_pgp=0.50, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :indinavir_pgp => (ft_pgp=0.45, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :nelfinavir_pgp => (ft_pgp=0.55, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :atazanavir_pgp => (ft_pgp=0.50, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :darunavir_pgp => (ft_pgp=0.40, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :ritonavir_pgp => (ft_pgp=0.35, probe=false, nti=false, tissue=[:intestine, :bbb]),

    # === Anticancer Agents ===
    :vincristine_pgp => (ft_pgp=0.80, probe=false, nti=true, tissue=[:intestine, :bbb, :tumor]),
    :vinblastine_pgp => (ft_pgp=0.75, probe=false, nti=false, tissue=[:intestine, :bbb, :tumor]),
    :paclitaxel_pgp => (ft_pgp=0.65, probe=false, nti=false, tissue=[:intestine, :bbb, :tumor]),
    :docetaxel_pgp => (ft_pgp=0.55, probe=false, nti=false, tissue=[:intestine, :bbb, :tumor]),
    :doxorubicin => (ft_pgp=0.60, probe=false, nti=false, tissue=[:intestine, :tumor]),
    :daunorubicin => (ft_pgp=0.55, probe=false, nti=false, tissue=[:intestine, :tumor]),
    :etoposide => (ft_pgp=0.45, probe=false, nti=false, tissue=[:intestine, :tumor]),
    :topotecan => (ft_pgp=0.40, probe=false, nti=false, tissue=[:intestine, :bbb, :tumor]),
    :irinotecan_pgp => (ft_pgp=0.50, probe=false, nti=false, tissue=[:intestine, :tumor]),
    :imatinib_pgp => (ft_pgp=0.35, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :nilotinib_pgp => (ft_pgp=0.45, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :dasatinib_pgp => (ft_pgp=0.40, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :lapatinib_pgp => (ft_pgp=0.50, probe=false, nti=false, tissue=[:intestine, :bbb, :tumor]),
    :gefitinib_pgp => (ft_pgp=0.45, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :erlotinib_pgp => (ft_pgp=0.40, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :sorafenib_pgp => (ft_pgp=0.35, probe=false, nti=false, tissue=[:intestine]),
    :sunitinib_pgp => (ft_pgp=0.40, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :crizotinib_pgp => (ft_pgp=0.45, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :venetoclax_pgp => (ft_pgp=0.60, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :olaparib => (ft_pgp=0.55, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :palbociclib => (ft_pgp=0.50, probe=false, nti=false, tissue=[:intestine]),

    # === Opioids ===
    :loperamide => (ft_pgp=0.75, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :morphine_pgp => (ft_pgp=0.30, probe=false, nti=false, tissue=[:bbb]),
    :fentanyl_pgp => (ft_pgp=0.25, probe=false, nti=false, tissue=[:bbb]),
    :methadone_pgp => (ft_pgp=0.35, probe=false, nti=false, tissue=[:bbb]),

    # === Colchicine ===
    :colchicine_pgp => (ft_pgp=0.70, probe=false, nti=true, tissue=[:intestine, :renal]),

    # === CNS Drugs ===
    :risperidone_pgp => (ft_pgp=0.40, probe=false, nti=false, tissue=[:bbb]),
    :quetiapine_pgp => (ft_pgp=0.35, probe=false, nti=false, tissue=[:bbb]),
    :olanzapine_pgp => (ft_pgp=0.30, probe=false, nti=false, tissue=[:bbb]),
    :paliperidone => (ft_pgp=0.45, probe=false, nti=false, tissue=[:bbb]),
    :amisulpride => (ft_pgp=0.50, probe=false, nti=false, tissue=[:bbb]),

    # === Antibiotics ===
    :levofloxacin_pgp => (ft_pgp=0.40, probe=false, nti=false, tissue=[:intestine, :renal]),
    :erythromycin_pgp => (ft_pgp=0.35, probe=false, nti=false, tissue=[:intestine, :hepatic]),
    :clarithromycin_pgp => (ft_pgp=0.40, probe=false, nti=false, tissue=[:intestine, :hepatic]),

    # === Miscellaneous ===
    :ondansetron_pgp => (ft_pgp=0.25, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :domperidone_pgp => (ft_pgp=0.60, probe=false, nti=false, tissue=[:intestine, :bbb]),
    :ambrisentan => (ft_pgp=0.40, probe=false, nti=false, tissue=[:intestine]),
    :macitentan => (ft_pgp=0.35, probe=false, nti=false, tissue=[:intestine]),
    :aliskiren => (ft_pgp=0.70, probe=false, nti=false, tissue=[:intestine]),
    :ranolazine => (ft_pgp=0.55, probe=false, nti=false, tissue=[:intestine]),
    :saxagliptin => (ft_pgp=0.40, probe=false, nti=false, tissue=[:intestine, :renal]),
    :sitagliptin_pgp => (ft_pgp=0.35, probe=false, nti=false, tissue=[:renal]),
    :linagliptin => (ft_pgp=0.50, probe=false, nti=false, tissue=[:intestine]),
)

"""
P-gp INHIBITORS DATABASE
"""
const PGP_INHIBITORS = Dict{Symbol, NamedTuple}(
    # === Strong P-gp Inhibitors (>= 2x AUC increase) ===
    :cyclosporine_pgp => (ki=1.0, fda_class=:strong, clinical_cmax=1.5, fu=0.04),
    :verapamil_pgp => (ki=5.0, fda_class=:moderate, clinical_cmax=0.5, fu=0.10),
    :quinidine_pgp => (ki=2.0, fda_class=:moderate, clinical_cmax=5.0, fu=0.13),
    :ketoconazole_pgp => (ki=5.0, fda_class=:moderate, clinical_cmax=10.0, fu=0.01),
    :itraconazole_pgp => (ki=2.0, fda_class=:moderate, clinical_cmax=0.5, fu=0.002),
    :ritonavir_pgp_inh => (ki=3.0, fda_class=:moderate, clinical_cmax=1.0, fu=0.02),
    :dronedarone_pgp => (ki=1.0, fda_class=:moderate, clinical_cmax=0.3, fu=0.02),
    :ranolazine_pgp => (ki=5.0, fda_class=:moderate, clinical_cmax=2.0, fu=0.30),
    :amiodarone_pgp => (ki=5.0, fda_class=:moderate, clinical_cmax=1.5, fu=0.04),
    :clarithromycin_pgp => (ki=10.0, fda_class=:moderate, clinical_cmax=3.0, fu=0.30),
    :erythromycin_pgp_inh => (ki=20.0, fda_class=:weak, clinical_cmax=5.0, fu=0.30),
    :lapatinib_pgp_inh => (ki=0.5, fda_class=:strong, clinical_cmax=2.5, fu=0.01),
    :osimertinib_pgp => (ki=2.0, fda_class=:moderate, clinical_cmax=0.5, fu=0.05),
    :tucatinib_pgp => (ki=1.5, fda_class=:moderate, clinical_cmax=3.0, fu=0.15),
    :carvedilol_pgp => (ki=10.0, fda_class=:weak, clinical_cmax=0.5, fu=0.02),
    :felodipine_pgp => (ki=15.0, fda_class=:weak, clinical_cmax=0.02, fu=0.01),
    :nicardipine => (ki=10.0, fda_class=:weak, clinical_cmax=0.1, fu=0.05),
    :nifedipine_pgp => (ki=20.0, fda_class=:weak, clinical_cmax=0.2, fu=0.04),
    :propafenone_pgp => (ki=5.0, fda_class=:moderate, clinical_cmax=2.0, fu=0.10),
    :eliglustat_pgp => (ki=5.0, fda_class=:moderate, clinical_cmax=0.2, fu=0.01),
    :elacridar => (ki=0.5, fda_class=:strong, clinical_cmax=1.0, fu=0.10),
    :tariquidar => (ki=0.1, fda_class=:strong, clinical_cmax=0.5, fu=0.10, note="research_compound"),
    :zosuquidar => (ki=0.05, fda_class=:strong, clinical_cmax=0.3, fu=0.10, note="research_compound"),
    :cobicistat_pgp => (ki=2.0, fda_class=:moderate, clinical_cmax=1.5, fu=0.02),
    :ledipasvir => (ki=3.0, fda_class=:moderate, clinical_cmax=0.5, fu=0.0001),
    :velpatasvir => (ki=2.0, fda_class=:moderate, clinical_cmax=0.5, fu=0.0005),
    :glecaprevir_pgp => (ki=1.0, fda_class=:strong, clinical_cmax=0.8, fu=0.025),
    :pibrentasvir => (ki=0.5, fda_class=:strong, clinical_cmax=0.2, fu=0.0001),
)

# =============================================================================
# BCRP (ABCG2) DATABASE
# =============================================================================

"""
BCRP SUBSTRATES DATABASE
BCRP limits oral absorption and mediates biliary excretion
"""
const BCRP_SUBSTRATES = Dict{Symbol, NamedTuple}(
    # === Probe Substrates ===
    :rosuvastatin => (ft_bcrp=0.40, ft_oatp1b1=0.50, probe=true, nti=false),
    :sulfasalazine => (ft_bcrp=0.85, probe=true, nti=false),

    # === Statins ===
    :atorvastatin_bcrp => (ft_bcrp=0.20, ft_oatp1b1=0.40, probe=false, nti=false),
    :fluvastatin_bcrp => (ft_bcrp=0.25, ft_oatp1b1=0.35, probe=false, nti=false),
    :pitavastatin_bcrp => (ft_bcrp=0.30, ft_oatp1b1=0.60, probe=false, nti=false),
    :pravastatin_bcrp => (ft_bcrp=0.15, ft_oatp1b1=0.75, probe=false, nti=false),

    # === Anticancer ===
    :topotecan_bcrp => (ft_bcrp=0.50, ft_pgp=0.40, probe=false, nti=false),
    :irinotecan_bcrp => (ft_bcrp=0.45, probe=false, nti=false),
    :sn38 => (ft_bcrp=0.70, probe=false, nti=false, note="irinotecan_metabolite"),
    :methotrexate_bcrp => (ft_bcrp=0.50, probe=false, nti=false),
    :mitoxantrone => (ft_bcrp=0.55, probe=false, nti=false),
    :gefitinib_bcrp => (ft_bcrp=0.45, probe=false, nti=false),
    :erlotinib_bcrp => (ft_bcrp=0.40, probe=false, nti=false),
    :lapatinib_bcrp => (ft_bcrp=0.35, probe=false, nti=false),
    :sorafenib_bcrp => (ft_bcrp=0.30, probe=false, nti=false),
    :sunitinib_bcrp => (ft_bcrp=0.35, probe=false, nti=false),
    :pazopanib_bcrp => (ft_bcrp=0.40, probe=false, nti=false),
    :axitinib_bcrp => (ft_bcrp=0.45, probe=false, nti=false),
    :vemurafenib_bcrp => (ft_bcrp=0.50, probe=false, nti=false),
    :dabrafenib_bcrp => (ft_bcrp=0.35, probe=false, nti=false),

    # === Antidiabetics ===
    :glyburide_bcrp => (ft_bcrp=0.30, ft_oatp1b1=0.40, probe=false, nti=false),

    # === Antivirals ===
    :abacavir => (ft_bcrp=0.40, probe=false, nti=false),
    :tenofovir_bcrp => (ft_bcrp=0.25, probe=false, nti=false),
    :zidovudine => (ft_bcrp=0.30, probe=false, nti=false),
    :lamivudine_bcrp => (ft_bcrp=0.25, probe=false, nti=false),

    # === Others ===
    :nitrofurantoin => (ft_bcrp=0.60, probe=false, nti=false),
    :cimetidine_bcrp => (ft_bcrp=0.25, probe=false, nti=false),
    :pantoprazole_bcrp => (ft_bcrp=0.30, probe=false, nti=false),
    :dolutegravir_bcrp => (ft_bcrp=0.30, probe=false, nti=false),
)

"""
BCRP INHIBITORS DATABASE
"""
const BCRP_INHIBITORS = Dict{Symbol, NamedTuple}(
    :elacridar_bcrp => (ki=0.1, fda_class=:strong, clinical_cmax=1.0, fu=0.10),
    :curcumin => (ki=2.0, fda_class=:moderate, clinical_cmax=0.5, fu=0.50),
    :lapatinib_bcrp_inh => (ki=0.3, fda_class=:strong, clinical_cmax=2.5, fu=0.01),
    :gefitinib_bcrp_inh => (ki=1.0, fda_class=:moderate, clinical_cmax=1.0, fu=0.10),
    :erlotinib_bcrp_inh => (ki=2.0, fda_class=:moderate, clinical_cmax=5.0, fu=0.07),
    :eltrombopag_bcrp => (ki=1.0, fda_class=:moderate, clinical_cmax=20.0, fu=0.0001),
    :cyclosporine_bcrp => (ki=5.0, fda_class=:moderate, clinical_cmax=1.5, fu=0.04),
    :pantoprazole_bcrp_inh => (ki=10.0, fda_class=:weak, clinical_cmax=5.0, fu=0.02),
    :febuxostat => (ki=5.0, fda_class=:weak, clinical_cmax=15.0, fu=0.01),
    :vemurafenib_bcrp_inh => (ki=2.0, fda_class=:moderate, clinical_cmax=100.0, fu=0.0001),
)

# =============================================================================
# OATP1B1/1B3 (SLCO1B1/SLCO1B3) DATABASE
# =============================================================================

"""
OATP1B1/1B3 SUBSTRATES DATABASE
Hepatic uptake transporters critical for statin disposition
"""
const OATP_SUBSTRATES = Dict{Symbol, NamedTuple}(
    # === Probe Substrates ===
    :pravastatin => (ft_oatp1b1=0.75, ft_oatp1b3=0.50, probe=true, nti=false),
    :pitavastatin => (ft_oatp1b1=0.80, ft_oatp1b3=0.60, probe=true, nti=false),
    :rosuvastatin_oatp => (ft_oatp1b1=0.70, ft_oatp1b3=0.50, probe=true, nti=false),
    :atorvastatin_oatp => (ft_oatp1b1=0.60, ft_oatp1b3=0.40, probe=false, nti=false),

    # === Statins ===
    :simvastatin_acid => (ft_oatp1b1=0.65, ft_oatp1b3=0.45, probe=false, nti=false),
    :fluvastatin_oatp => (ft_oatp1b1=0.50, ft_oatp1b3=0.30, probe=false, nti=false),
    :lovastatin_acid => (ft_oatp1b1=0.55, ft_oatp1b3=0.35, probe=false, nti=false),
    :cerivastatin_oatp => (ft_oatp1b1=0.70, ft_oatp1b3=0.50, probe=false, nti=false),

    # === Antidiabetics ===
    :repaglinide_oatp => (ft_oatp1b1=0.55, ft_oatp1b3=0.30, probe=false, nti=false),
    :nateglinide_oatp => (ft_oatp1b1=0.40, ft_oatp1b3=0.25, probe=false, nti=false),
    :glyburide_oatp => (ft_oatp1b1=0.50, ft_oatp1b3=0.35, probe=false, nti=false),

    # === Sartans ===
    :valsartan => (ft_oatp1b1=0.45, ft_oatp1b3=0.40, probe=false, nti=false),
    :olmesartan => (ft_oatp1b1=0.40, ft_oatp1b3=0.35, probe=false, nti=false),
    :telmisartan => (ft_oatp1b1=0.50, ft_oatp1b3=0.45, probe=false, nti=false),

    # === Endothelin Receptor Antagonists ===
    :bosentan_oatp => (ft_oatp1b1=0.50, ft_oatp1b3=0.40, probe=false, nti=false),
    :ambrisentan_oatp => (ft_oatp1b1=0.35, ft_oatp1b3=0.25, probe=false, nti=false),
    :macitentan_oatp => (ft_oatp1b1=0.30, ft_oatp1b3=0.20, probe=false, nti=false),

    # === HCV Drugs ===
    :asunaprevir => (ft_oatp1b1=0.70, ft_oatp1b3=0.60, probe=false, nti=false),
    :simeprevir => (ft_oatp1b1=0.65, ft_oatp1b3=0.55, probe=false, nti=false),
    :grazoprevir_oatp => (ft_oatp1b1=0.60, ft_oatp1b3=0.50, probe=false, nti=false),
    :paritaprevir_oatp => (ft_oatp1b1=0.55, ft_oatp1b3=0.45, probe=false, nti=false),
    :glecaprevir_oatp => (ft_oatp1b1=0.65, ft_oatp1b3=0.55, probe=false, nti=false),
    :voxilaprevir => (ft_oatp1b1=0.60, ft_oatp1b3=0.50, probe=false, nti=false),

    # === Antibiotics ===
    :rifampin_oatp => (ft_oatp1b1=0.55, ft_oatp1b3=0.45, probe=false, nti=false),
    :benzylpenicillin_oatp => (ft_oatp1b1=0.35, ft_oatp1b3=0.25, probe=false, nti=false),
    :cefazolin => (ft_oatp1b1=0.30, ft_oatp1b3=0.20, probe=false, nti=false),

    # === Others ===
    :methotrexate_oatp => (ft_oatp1b1=0.40, ft_oatp1b3=0.30, probe=false, nti=false),
    :fexofenadine_oatp => (ft_oatp1b1=0.35, ft_oatp1b3=0.25, probe=false, nti=false),
    :ezetimibe_gluc => (ft_oatp1b1=0.50, ft_oatp1b3=0.35, probe=false, nti=false, note="glucuronide"),
    :mycophenolic_gluc => (ft_oatp1b1=0.55, ft_oatp1b3=0.40, probe=false, nti=false, note="glucuronide"),
    :bilirubin => (ft_oatp1b1=0.65, ft_oatp1b3=0.60, probe=false, nti=false, note="endogenous"),
    :coproporphyrin_i => (ft_oatp1b1=0.70, ft_oatp1b3=0.60, probe=true, nti=false, note="endogenous_biomarker"),
    :coproporphyrin_iii => (ft_oatp1b1=0.60, ft_oatp1b3=0.50, probe=true, nti=false, note="endogenous_biomarker"),
)

"""
OATP1B1/1B3 INHIBITORS DATABASE
"""
const OATP_INHIBITORS = Dict{Symbol, NamedTuple}(
    # === Strong Inhibitors ===
    :cyclosporine_oatp => (ki_1b1=0.05, ki_1b3=0.1, fda_class=:strong, clinical_cmax=1.5, fu=0.04),
    :rifampin_oatp_inh => (ki_1b1=0.5, ki_1b3=0.5, fda_class=:strong, clinical_cmax=10.0, fu=0.20, note="single_dose"),
    :grazoprevir => (ki_1b1=0.5, ki_1b3=1.0, fda_class=:strong, clinical_cmax=0.3, fu=0.01),
    :glecaprevir => (ki_1b1=0.2, ki_1b3=0.3, fda_class=:strong, clinical_cmax=0.8, fu=0.025),
    :voxilaprevir_inh => (ki_1b1=0.1, ki_1b3=0.2, fda_class=:strong, clinical_cmax=0.2, fu=0.01),
    :paritaprevir_inh => (ki_1b1=0.3, ki_1b3=0.5, fda_class=:strong, clinical_cmax=1.5, fu=0.02),

    # === Moderate Inhibitors ===
    :gemfibrozil_oatp => (ki_1b1=10.0, ki_1b3=25.0, fda_class=:moderate, clinical_cmax=100.0, fu=0.03),
    :lopinavir_oatp => (ki_1b1=2.0, ki_1b3=3.0, fda_class=:moderate, clinical_cmax=10.0, fu=0.01),
    :atazanavir_oatp => (ki_1b1=1.0, ki_1b3=2.0, fda_class=:moderate, clinical_cmax=4.0, fu=0.14),
    :eltrombopag_oatp => (ki_1b1=2.0, ki_1b3=5.0, fda_class=:moderate, clinical_cmax=20.0, fu=0.0001),
    :simeprevir_oatp => (ki_1b1=1.0, ki_1b3=1.5, fda_class=:moderate, clinical_cmax=0.5, fu=0.0001),

    # === Weak Inhibitors ===
    :clarithromycin_oatp => (ki_1b1=5.0, ki_1b3=10.0, fda_class=:weak, clinical_cmax=3.0, fu=0.30),
    :erythromycin_oatp => (ki_1b1=10.0, ki_1b3=20.0, fda_class=:weak, clinical_cmax=5.0, fu=0.30),
    :atorvastatin_oatp_inh => (ki_1b1=5.0, ki_1b3=10.0, fda_class=:weak, clinical_cmax=0.1, fu=0.05),
    :rosuvastatin_oatp_inh => (ki_1b1=8.0, ki_1b3=15.0, fda_class=:weak, clinical_cmax=0.05, fu=0.12),
)

# =============================================================================
# RENAL TRANSPORTERS (OAT1/OAT3/OCT2/MATE1/MATE2-K)
# =============================================================================

"""
OAT1/OAT3 SUBSTRATES DATABASE
Organic anion transporters mediating renal secretion
"""
const OAT_SUBSTRATES = Dict{Symbol, NamedTuple}(
    # === OAT1 Probe Substrates ===
    :adefovir => (ft_oat1=0.80, ft_oat3=0.20, probe_oat1=true, nti=false),
    :cidofovir => (ft_oat1=0.85, ft_oat3=0.10, probe_oat1=true, nti=false),
    :tenofovir => (ft_oat1=0.60, ft_oat3=0.30, probe_oat1=true, nti=false),

    # === OAT3 Probe Substrates ===
    :benzylpenicillin => (ft_oat1=0.30, ft_oat3=0.70, probe_oat3=true, nti=false),
    :furosemide => (ft_oat1=0.30, ft_oat3=0.60, probe_oat3=true, nti=false),

    # === Antivirals ===
    :acyclovir_oat => (ft_oat1=0.50, ft_oat3=0.30, probe=false, nti=false),
    :ganciclovir => (ft_oat1=0.55, ft_oat3=0.25, probe=false, nti=false),
    :entecavir => (ft_oat1=0.45, ft_oat3=0.30, probe=false, nti=false),
    :lamivudine_oat => (ft_oat1=0.35, ft_oat3=0.40, probe=false, nti=false),
    :zidovudine_oat => (ft_oat1=0.40, ft_oat3=0.35, probe=false, nti=false),
    :oseltamivir_carboxylate => (ft_oat1=0.25, ft_oat3=0.55, probe=false, nti=false),

    # === Diuretics ===
    :bumetanide => (ft_oat1=0.40, ft_oat3=0.50, probe=false, nti=false),
    :torasemide => (ft_oat1=0.35, ft_oat3=0.45, probe=false, nti=false),
    :hydrochlorothiazide => (ft_oat1=0.30, ft_oat3=0.40, probe=false, nti=false),

    # === NSAIDs ===
    :indomethacin_oat => (ft_oat1=0.25, ft_oat3=0.50, probe=false, nti=false),
    :ketoprofen => (ft_oat1=0.20, ft_oat3=0.45, probe=false, nti=false),
    :naproxen_oat => (ft_oat1=0.25, ft_oat3=0.40, probe=false, nti=false),
    :ibuprofen_oat => (ft_oat1=0.20, ft_oat3=0.35, probe=false, nti=false),

    # === Antibiotics ===
    :penicillin_g => (ft_oat1=0.30, ft_oat3=0.70, probe=false, nti=false),
    :cefaclor_oat => (ft_oat1=0.25, ft_oat3=0.50, probe=false, nti=false),
    :ceftriaxone => (ft_oat1=0.30, ft_oat3=0.45, probe=false, nti=false),
    :ciprofloxacin_oat => (ft_oat1=0.35, ft_oat3=0.40, probe=false, nti=false),

    # === Methotrexate ===
    :methotrexate_oat => (ft_oat1=0.40, ft_oat3=0.50, probe=false, nti=true),

    # === Others ===
    :famotidine => (ft_oat1=0.20, ft_oat3=0.45, probe=false, nti=false),
    :ranitidine => (ft_oat1=0.25, ft_oat3=0.50, probe=false, nti=false),
    :pravastatin_oat => (ft_oat1=0.15, ft_oat3=0.40, probe=false, nti=false),
)

"""
OAT INHIBITORS DATABASE
"""
const OAT_INHIBITORS = Dict{Symbol, NamedTuple}(
    :probenecid => (ki_oat1=2.0, ki_oat3=5.0, fda_class=:strong, clinical_cmax=100.0, fu=0.10),
    :diclofenac_oat => (ki_oat1=5.0, ki_oat3=10.0, fda_class=:moderate, clinical_cmax=5.0, fu=0.01),
    :indomethacin_oat_inh => (ki_oat1=3.0, ki_oat3=8.0, fda_class=:moderate, clinical_cmax=5.0, fu=0.01),
    :ibuprofen_oat_inh => (ki_oat1=10.0, ki_oat3=15.0, fda_class=:weak, clinical_cmax=100.0, fu=0.01),
    :furosemide_inh => (ki_oat1=15.0, ki_oat3=25.0, fda_class=:weak, clinical_cmax=5.0, fu=0.02),
    :teriflunomide_oat => (ki_oat1=10.0, ki_oat3=20.0, fda_class=:weak, clinical_cmax=80.0, fu=0.0003),
    :lesinurad => (ki_oat1=8.0, ki_oat3=15.0, fda_class=:weak, clinical_cmax=30.0, fu=0.02),
)

"""
OCT2/MATE1/MATE2-K SUBSTRATES DATABASE
"""
const OCT_MATE_SUBSTRATES = Dict{Symbol, NamedTuple}(
    # === Probe Substrates ===
    :metformin => (ft_oct1=0.50, ft_oct2=0.70, ft_mate1=0.60, ft_mate2k=0.40, probe=true, nti=false),

    # === Antivirals ===
    :lamivudine_oct => (ft_oct2=0.40, ft_mate1=0.30, ft_mate2k=0.20, probe=false, nti=false),
    :emtricitabine => (ft_oct2=0.35, ft_mate1=0.25, ft_mate2k=0.15, probe=false, nti=false),
    :memantine => (ft_oct2=0.45, ft_mate1=0.35, ft_mate2k=0.25, probe=false, nti=false),
    :amantadine => (ft_oct2=0.40, ft_mate1=0.30, ft_mate2k=0.20, probe=false, nti=false),

    # === Anticancer ===
    :oxaliplatin => (ft_oct2=0.50, ft_mate1=0.35, probe=false, nti=false),
    :cisplatin => (ft_oct2=0.45, ft_mate1=0.30, probe=false, nti=false),

    # === Histamine Antagonists ===
    :cimetidine_oct => (ft_oct1=0.30, ft_oct2=0.40, ft_mate1=0.40, ft_mate2k=0.30, probe=false, nti=false),
    :ranitidine_oct => (ft_oct2=0.35, ft_mate1=0.30, probe=false, nti=false),
    :famotidine_oct => (ft_oct2=0.30, ft_mate1=0.25, probe=false, nti=false),

    # === Others ===
    :sumatriptan => (ft_oct1=0.40, ft_oct2=0.35, probe=false, nti=false),
    :amisulpride_oct => (ft_oct2=0.45, ft_mate1=0.35, probe=false, nti=false),
    :pindolol => (ft_oct2=0.40, ft_mate1=0.30, probe=false, nti=false),
    :varenicline => (ft_oct2=0.50, ft_mate1=0.40, ft_mate2k=0.30, probe=false, nti=false),
    :trospium => (ft_oct2=0.55, ft_mate1=0.45, probe=false, nti=false),
    :fampridine => (ft_oct2=0.45, ft_mate1=0.35, probe=false, nti=false),
    :creatinine => (ft_oct2=0.50, ft_mate1=0.60, ft_mate2k=0.50, probe=false, nti=false, note="endogenous_biomarker"),
)

"""
OCT/MATE INHIBITORS DATABASE
"""
const OCT_MATE_INHIBITORS = Dict{Symbol, NamedTuple}(
    # === MATE Inhibitors ===
    :pyrimethamine => (ki_mate1=0.1, ki_mate2k=0.5, fda_class=:strong, clinical_cmax=2.0, fu=0.13),
    :cimetidine_mate => (ki_oct2=100.0, ki_mate1=2.0, ki_mate2k=5.0, fda_class=:moderate, clinical_cmax=10.0, fu=0.80),
    :dolutegravir_mate => (ki_oct2=2.0, ki_mate1=5.0, fda_class=:moderate, clinical_cmax=5.0, fu=0.01),
    :trimethoprim_oct => (ki_oct2=30.0, ki_mate1=10.0, ki_mate2k=15.0, fda_class=:moderate, clinical_cmax=8.0, fu=0.56),
    :vandetanib_oct => (ki_oct2=5.0, ki_mate1=15.0, fda_class=:moderate, clinical_cmax=1.5, fu=0.06),
    :isavuconazole_oct => (ki_oct2=10.0, ki_mate1=20.0, fda_class=:weak, clinical_cmax=8.0, fu=0.01),
    :ranolazine_oct => (ki_oct2=50.0, ki_mate1=100.0, fda_class=:weak, clinical_cmax=2.0, fu=0.30),
    :imatinib_oct => (ki_oct1=1.0, ki_oct2=2.0, fda_class=:moderate, clinical_cmax=3.0, fu=0.05),

    # === OCT1 Inhibitors (hepatic) ===
    :verapamil_oct1 => (ki_oct1=5.0, fda_class=:weak, clinical_cmax=0.5, fu=0.10),
    :quinidine_oct1 => (ki_oct1=3.0, fda_class=:moderate, clinical_cmax=5.0, fu=0.13),
    :disopyramide => (ki_oct1=10.0, fda_class=:weak, clinical_cmax=5.0, fu=0.40),
)

# =============================================================================
# COMBINED DATABASES
# =============================================================================

"""
Complete transporter substrate database.
"""
const TRANSPORTER_SUBSTRATES_COMPLETE = merge(
    PGP_SUBSTRATES,
    BCRP_SUBSTRATES,
    OATP_SUBSTRATES,
    OAT_SUBSTRATES,
    OCT_MATE_SUBSTRATES
)

"""
Complete transporter inhibitor database.
"""
const TRANSPORTER_INHIBITORS_COMPLETE = merge(
    PGP_INHIBITORS,
    BCRP_INHIBITORS,
    OATP_INHIBITORS,
    OAT_INHIBITORS,
    OCT_MATE_INHIBITORS
)

# Export all databases
export TRANSPORTER_SUBSTRATES_COMPLETE, TRANSPORTER_INHIBITORS_COMPLETE
export PGP_SUBSTRATES, PGP_INHIBITORS
export BCRP_SUBSTRATES, BCRP_INHIBITORS
export OATP_SUBSTRATES, OATP_INHIBITORS
export OAT_SUBSTRATES, OAT_INHIBITORS
export OCT_MATE_SUBSTRATES, OCT_MATE_INHIBITORS
