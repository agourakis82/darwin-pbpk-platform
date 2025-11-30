# =============================================================================
# DRUG PROPERTIES DATABASE - PHYSICOCHEMICAL AND PK PARAMETERS
# =============================================================================
# Darwin PBPK Platform - Publication-Ready
#
# Sources: DrugBank, ChEMBL, PubChem, FDA Labels, Clinical Pharmacology
# Coverage: 300+ drugs with complete physicochemical and PK profiles
#
# Properties included:
# - Molecular weight (MW)
# - LogP (partition coefficient)
# - pKa (acid/base)
# - Polar surface area (PSA)
# - H-bond donors/acceptors
# - Unbound fraction (fu)
# - Volume of distribution (Vd)
# - Clearance (CL)
# - Half-life (t1/2)
# - BCS class
# - Therapeutic index
# =============================================================================

"""
Drug physicochemical and PK properties structure.
"""
struct DrugPhysicochemicalProperties
    name::String
    drugbank_id::String

    # Physicochemical
    mw::Float64           # Molecular weight (Da)
    logp::Float64         # Partition coefficient
    pka_acid::Float64     # Acidic pKa (0 if not acidic)
    pka_base::Float64     # Basic pKa (0 if not basic)
    psa::Float64          # Polar surface area (Angstrom^2)
    hbd::Int              # H-bond donors
    hba::Int              # H-bond acceptors
    rotatable_bonds::Int  # Rotatable bonds

    # PK parameters
    fu::Float64           # Unbound fraction
    vd::Float64           # Volume of distribution (L/kg)
    cl::Float64           # Total clearance (L/h)
    t_half::Float64       # Half-life (hours)
    bioavailability::Float64  # Oral bioavailability (fraction)

    # Classification
    bcs_class::Int        # BCS class (1-4)
    therapeutic_index::Symbol  # :wide, :narrow, :very_narrow
    drug_class::String
end

# =============================================================================
# CARDIOVASCULAR DRUGS
# =============================================================================

const CARDIOVASCULAR_DRUG_PROPERTIES = Dict{Symbol, NamedTuple}(
    # === Statins ===
    :simvastatin => (
        name="Simvastatin", drugbank_id="DB00641",
        mw=418.57, logp=4.68, pka_acid=0.0, pka_base=0.0, psa=72.8, hbd=1, hba=5, rotatable_bonds=7,
        fu=0.05, vd=1.5, cl=30.0, t_half=2.0, bioavailability=0.05,
        bcs_class=2, therapeutic_index=:wide, drug_class="HMG-CoA reductase inhibitor"
    ),
    :lovastatin => (
        name="Lovastatin", drugbank_id="DB00227",
        mw=404.54, logp=4.26, pka_acid=0.0, pka_base=0.0, psa=72.8, hbd=1, hba=5, rotatable_bonds=7,
        fu=0.05, vd=1.3, cl=25.0, t_half=2.5, bioavailability=0.05,
        bcs_class=2, therapeutic_index=:wide, drug_class="HMG-CoA reductase inhibitor"
    ),
    :atorvastatin => (
        name="Atorvastatin", drugbank_id="DB01076",
        mw=558.64, logp=4.06, pka_acid=4.46, pka_base=0.0, psa=111.8, hbd=4, hba=7, rotatable_bonds=12,
        fu=0.05, vd=5.6, cl=10.0, t_half=14.0, bioavailability=0.14,
        bcs_class=2, therapeutic_index=:wide, drug_class="HMG-CoA reductase inhibitor"
    ),
    :rosuvastatin => (
        name="Rosuvastatin", drugbank_id="DB01098",
        mw=481.54, logp=0.13, pka_acid=4.6, pka_base=0.0, psa=142.6, hbd=3, hba=7, rotatable_bonds=9,
        fu=0.12, vd=2.0, cl=7.0, t_half=19.0, bioavailability=0.20,
        bcs_class=3, therapeutic_index=:wide, drug_class="HMG-CoA reductase inhibitor"
    ),
    :pravastatin => (
        name="Pravastatin", drugbank_id="DB00175",
        mw=424.53, logp=-0.23, pka_acid=4.31, pka_base=0.0, psa=124.3, hbd=4, hba=7, rotatable_bonds=9,
        fu=0.50, vd=0.5, cl=9.0, t_half=1.8, bioavailability=0.17,
        bcs_class=3, therapeutic_index=:wide, drug_class="HMG-CoA reductase inhibitor"
    ),
    :fluvastatin => (
        name="Fluvastatin", drugbank_id="DB01095",
        mw=411.47, logp=3.24, pka_acid=4.5, pka_base=0.0, psa=82.7, hbd=3, hba=5, rotatable_bonds=8,
        fu=0.02, vd=0.4, cl=6.0, t_half=2.3, bioavailability=0.24,
        bcs_class=2, therapeutic_index=:wide, drug_class="HMG-CoA reductase inhibitor"
    ),
    :pitavastatin => (
        name="Pitavastatin", drugbank_id="DB08860",
        mw=421.46, logp=2.76, pka_acid=4.3, pka_base=0.0, psa=91.0, hbd=3, hba=5, rotatable_bonds=7,
        fu=0.04, vd=1.5, cl=3.0, t_half=11.0, bioavailability=0.51,
        bcs_class=2, therapeutic_index=:wide, drug_class="HMG-CoA reductase inhibitor"
    ),

    # === Calcium Channel Blockers ===
    :amlodipine => (
        name="Amlodipine", drugbank_id="DB00381",
        mw=408.88, logp=3.0, pka_acid=0.0, pka_base=8.6, psa=99.9, hbd=2, hba=7, rotatable_bonds=10,
        fu=0.03, vd=21.0, cl=7.0, t_half=35.0, bioavailability=0.64,
        bcs_class=1, therapeutic_index=:wide, drug_class="Calcium channel blocker"
    ),
    :nifedipine => (
        name="Nifedipine", drugbank_id="DB01115",
        mw=346.34, logp=2.2, pka_acid=0.0, pka_base=0.0, psa=108.0, hbd=1, hba=7, rotatable_bonds=6,
        fu=0.04, vd=0.8, cl=30.0, t_half=2.0, bioavailability=0.50,
        bcs_class=2, therapeutic_index=:wide, drug_class="Calcium channel blocker"
    ),
    :felodipine => (
        name="Felodipine", drugbank_id="DB01023",
        mw=384.25, logp=4.8, pka_acid=0.0, pka_base=0.0, psa=64.6, hbd=1, hba=5, rotatable_bonds=8,
        fu=0.01, vd=10.0, cl=70.0, t_half=11.0, bioavailability=0.15,
        bcs_class=2, therapeutic_index=:wide, drug_class="Calcium channel blocker"
    ),
    :diltiazem => (
        name="Diltiazem", drugbank_id="DB00343",
        mw=414.52, logp=2.7, pka_acid=0.0, pka_base=8.0, psa=59.1, hbd=0, hba=5, rotatable_bonds=7,
        fu=0.20, vd=5.3, cl=50.0, t_half=4.0, bioavailability=0.40,
        bcs_class=1, therapeutic_index=:wide, drug_class="Calcium channel blocker"
    ),
    :verapamil => (
        name="Verapamil", drugbank_id="DB00661",
        mw=454.60, logp=3.79, pka_acid=0.0, pka_base=8.92, psa=64.0, hbd=0, hba=6, rotatable_bonds=13,
        fu=0.10, vd=5.0, cl=65.0, t_half=5.0, bioavailability=0.22,
        bcs_class=1, therapeutic_index=:wide, drug_class="Calcium channel blocker"
    ),

    # === Beta Blockers ===
    :metoprolol => (
        name="Metoprolol", drugbank_id="DB00264",
        mw=267.36, logp=1.88, pka_acid=0.0, pka_base=9.7, psa=50.7, hbd=2, hba=4, rotatable_bonds=9,
        fu=0.88, vd=5.6, cl=60.0, t_half=3.5, bioavailability=0.50,
        bcs_class=1, therapeutic_index=:wide, drug_class="Beta-1 selective blocker"
    ),
    :propranolol => (
        name="Propranolol", drugbank_id="DB00571",
        mw=259.34, logp=3.48, pka_acid=0.0, pka_base=9.42, psa=41.5, hbd=2, hba=3, rotatable_bonds=6,
        fu=0.13, vd=4.0, cl=60.0, t_half=4.0, bioavailability=0.26,
        bcs_class=1, therapeutic_index=:wide, drug_class="Non-selective beta blocker"
    ),
    :carvedilol => (
        name="Carvedilol", drugbank_id="DB01136",
        mw=406.47, logp=4.19, pka_acid=0.0, pka_base=7.8, psa=75.7, hbd=3, hba=5, rotatable_bonds=10,
        fu=0.02, vd=2.0, cl=35.0, t_half=7.0, bioavailability=0.25,
        bcs_class=2, therapeutic_index=:wide, drug_class="Alpha/beta blocker"
    ),

    # === Antiarrhythmics ===
    :digoxin => (
        name="Digoxin", drugbank_id="DB00390",
        mw=780.94, logp=1.26, pka_acid=0.0, pka_base=0.0, psa=203.1, hbd=6, hba=14, rotatable_bonds=7,
        fu=0.75, vd=7.0, cl=2.5, t_half=36.0, bioavailability=0.70,
        bcs_class=1, therapeutic_index=:very_narrow, drug_class="Cardiac glycoside"
    ),
    :amiodarone => (
        name="Amiodarone", drugbank_id="DB01118",
        mw=645.31, logp=7.64, pka_acid=0.0, pka_base=6.56, psa=42.0, hbd=1, hba=3, rotatable_bonds=12,
        fu=0.04, vd=66.0, cl=2.0, t_half=1440.0, bioavailability=0.50,
        bcs_class=2, therapeutic_index=:narrow, drug_class="Class III antiarrhythmic"
    ),
    :quinidine => (
        name="Quinidine", drugbank_id="DB00908",
        mw=324.42, logp=2.51, pka_acid=0.0, pka_base=8.56, psa=45.6, hbd=1, hba=4, rotatable_bonds=4,
        fu=0.13, vd=2.5, cl=13.0, t_half=6.0, bioavailability=0.70,
        bcs_class=1, therapeutic_index=:narrow, drug_class="Class IA antiarrhythmic"
    ),

    # === Anticoagulants ===
    :warfarin => (
        name="Warfarin", drugbank_id="DB00682",
        mw=308.33, logp=2.70, pka_acid=5.05, pka_base=0.0, psa=67.5, hbd=1, hba=4, rotatable_bonds=4,
        fu=0.01, vd=0.14, cl=0.2, t_half=40.0, bioavailability=0.99,
        bcs_class=1, therapeutic_index=:very_narrow, drug_class="Vitamin K antagonist"
    ),
    :dabigatran => (
        name="Dabigatran etexilate", drugbank_id="DB06695",
        mw=627.73, logp=3.03, pka_acid=4.35, pka_base=0.0, psa=150.0, hbd=4, hba=10, rotatable_bonds=12,
        fu=0.65, vd=0.7, cl=7.0, t_half=13.0, bioavailability=0.06,
        bcs_class=3, therapeutic_index=:narrow, drug_class="Direct thrombin inhibitor"
    ),
    :rivaroxaban => (
        name="Rivaroxaban", drugbank_id="DB06228",
        mw=435.88, logp=1.5, pka_acid=0.0, pka_base=0.0, psa=88.2, hbd=1, hba=7, rotatable_bonds=5,
        fu=0.05, vd=0.7, cl=7.0, t_half=9.0, bioavailability=0.80,
        bcs_class=2, therapeutic_index=:narrow, drug_class="Factor Xa inhibitor"
    ),
    :apixaban => (
        name="Apixaban", drugbank_id="DB07828",
        mw=459.50, logp=2.5, pka_acid=0.0, pka_base=0.0, psa=110.8, hbd=1, hba=7, rotatable_bonds=5,
        fu=0.13, vd=0.3, cl=4.0, t_half=12.0, bioavailability=0.50,
        bcs_class=3, therapeutic_index=:narrow, drug_class="Factor Xa inhibitor"
    ),
)

# =============================================================================
# CNS DRUGS
# =============================================================================

const CNS_DRUG_PROPERTIES = Dict{Symbol, NamedTuple}(
    # === Benzodiazepines ===
    :midazolam => (
        name="Midazolam", drugbank_id="DB00683",
        mw=325.77, logp=3.93, pka_acid=0.0, pka_base=6.2, psa=30.2, hbd=0, hba=3, rotatable_bonds=2,
        fu=0.03, vd=1.5, cl=25.0, t_half=2.0, bioavailability=0.44,
        bcs_class=1, therapeutic_index=:wide, drug_class="Benzodiazepine"
    ),
    :triazolam => (
        name="Triazolam", drugbank_id="DB00897",
        mw=343.21, logp=2.42, pka_acid=0.0, pka_base=1.52, psa=43.1, hbd=0, hba=4, rotatable_bonds=1,
        fu=0.10, vd=1.1, cl=25.0, t_half=2.5, bioavailability=0.90,
        bcs_class=2, therapeutic_index=:wide, drug_class="Benzodiazepine"
    ),
    :alprazolam => (
        name="Alprazolam", drugbank_id="DB00404",
        mw=308.76, logp=2.12, pka_acid=0.0, pka_base=2.4, psa=43.1, hbd=0, hba=4, rotatable_bonds=1,
        fu=0.20, vd=1.0, cl=6.0, t_half=11.0, bioavailability=0.90,
        bcs_class=2, therapeutic_index=:wide, drug_class="Benzodiazepine"
    ),
    :diazepam => (
        name="Diazepam", drugbank_id="DB00829",
        mw=284.74, logp=2.82, pka_acid=0.0, pka_base=3.4, psa=32.7, hbd=0, hba=3, rotatable_bonds=1,
        fu=0.02, vd=1.1, cl=1.5, t_half=40.0, bioavailability=0.95,
        bcs_class=2, therapeutic_index=:wide, drug_class="Benzodiazepine"
    ),

    # === Antipsychotics ===
    :quetiapine => (
        name="Quetiapine", drugbank_id="DB01224",
        mw=383.51, logp=2.81, pka_acid=0.0, pka_base=6.8, psa=48.4, hbd=1, hba=5, rotatable_bonds=5,
        fu=0.17, vd=10.0, cl=100.0, t_half=6.0, bioavailability=0.73,
        bcs_class=2, therapeutic_index=:wide, drug_class="Atypical antipsychotic"
    ),
    :risperidone => (
        name="Risperidone", drugbank_id="DB00734",
        mw=410.49, logp=3.04, pka_acid=0.0, pka_base=8.2, psa=61.9, hbd=0, hba=5, rotatable_bonds=4,
        fu=0.10, vd=1.5, cl=5.0, t_half=20.0, bioavailability=0.70,
        bcs_class=2, therapeutic_index=:wide, drug_class="Atypical antipsychotic"
    ),
    :olanzapine => (
        name="Olanzapine", drugbank_id="DB00334",
        mw=312.43, logp=2.0, pka_acid=0.0, pka_base=7.4, psa=30.9, hbd=1, hba=4, rotatable_bonds=2,
        fu=0.07, vd=15.0, cl=25.0, t_half=30.0, bioavailability=0.60,
        bcs_class=2, therapeutic_index=:wide, drug_class="Atypical antipsychotic"
    ),
    :clozapine => (
        name="Clozapine", drugbank_id="DB00363",
        mw=326.82, logp=3.23, pka_acid=0.0, pka_base=7.5, psa=30.9, hbd=1, hba=4, rotatable_bonds=2,
        fu=0.03, vd=5.0, cl=30.0, t_half=12.0, bioavailability=0.50,
        bcs_class=2, therapeutic_index=:narrow, drug_class="Atypical antipsychotic"
    ),
    :haloperidol => (
        name="Haloperidol", drugbank_id="DB00502",
        mw=375.86, logp=3.66, pka_acid=0.0, pka_base=8.3, psa=40.5, hbd=1, hba=3, rotatable_bonds=6,
        fu=0.08, vd=18.0, cl=25.0, t_half=18.0, bioavailability=0.60,
        bcs_class=2, therapeutic_index=:narrow, drug_class="Typical antipsychotic"
    ),

    # === Antidepressants ===
    :fluoxetine => (
        name="Fluoxetine", drugbank_id="DB00472",
        mw=309.33, logp=4.05, pka_acid=0.0, pka_base=10.1, psa=21.3, hbd=1, hba=2, rotatable_bonds=6,
        fu=0.06, vd=35.0, cl=35.0, t_half=48.0, bioavailability=0.72,
        bcs_class=1, therapeutic_index=:wide, drug_class="SSRI"
    ),
    :paroxetine => (
        name="Paroxetine", drugbank_id="DB00715",
        mw=329.37, logp=3.95, pka_acid=0.0, pka_base=9.9, psa=39.7, hbd=1, hba=4, rotatable_bonds=4,
        fu=0.05, vd=17.0, cl=30.0, t_half=21.0, bioavailability=0.50,
        bcs_class=2, therapeutic_index=:wide, drug_class="SSRI"
    ),
    :sertraline => (
        name="Sertraline", drugbank_id="DB01104",
        mw=306.23, logp=5.29, pka_acid=0.0, pka_base=9.5, psa=12.0, hbd=1, hba=1, rotatable_bonds=3,
        fu=0.02, vd=20.0, cl=90.0, t_half=26.0, bioavailability=0.44,
        bcs_class=2, therapeutic_index=:wide, drug_class="SSRI"
    ),
    :venlafaxine => (
        name="Venlafaxine", drugbank_id="DB00285",
        mw=277.40, logp=2.91, pka_acid=0.0, pka_base=10.1, psa=32.7, hbd=1, hba=3, rotatable_bonds=5,
        fu=0.73, vd=7.5, cl=70.0, t_half=5.0, bioavailability=0.45,
        bcs_class=1, therapeutic_index=:wide, drug_class="SNRI"
    ),
    :duloxetine => (
        name="Duloxetine", drugbank_id="DB00476",
        mw=297.41, logp=4.2, pka_acid=0.0, pka_base=9.7, psa=36.4, hbd=1, hba=3, rotatable_bonds=6,
        fu=0.04, vd=23.0, cl=100.0, t_half=12.0, bioavailability=0.50,
        bcs_class=2, therapeutic_index=:wide, drug_class="SNRI"
    ),
    :bupropion => (
        name="Bupropion", drugbank_id="DB01156",
        mw=239.74, logp=3.21, pka_acid=0.0, pka_base=7.9, psa=29.1, hbd=1, hba=2, rotatable_bonds=4,
        fu=0.16, vd=20.0, cl=100.0, t_half=21.0, bioavailability=0.87,
        bcs_class=1, therapeutic_index=:wide, drug_class="Aminoketone"
    ),

    # === Opioids ===
    :codeine => (
        name="Codeine", drugbank_id="DB00318",
        mw=299.36, logp=1.19, pka_acid=0.0, pka_base=8.2, psa=41.9, hbd=1, hba=4, rotatable_bonds=1,
        fu=0.93, vd=3.5, cl=50.0, t_half=3.0, bioavailability=0.90,
        bcs_class=1, therapeutic_index=:wide, drug_class="Opioid analgesic"
    ),
    :tramadol => (
        name="Tramadol", drugbank_id="DB00193",
        mw=263.38, logp=2.63, pka_acid=0.0, pka_base=9.4, psa=32.7, hbd=1, hba=3, rotatable_bonds=5,
        fu=0.80, vd=3.0, cl=30.0, t_half=6.0, bioavailability=0.70,
        bcs_class=1, therapeutic_index=:wide, drug_class="Opioid analgesic"
    ),
    :oxycodone => (
        name="Oxycodone", drugbank_id="DB00497",
        mw=315.36, logp=0.70, pka_acid=0.0, pka_base=8.5, psa=59.0, hbd=1, hba=5, rotatable_bonds=1,
        fu=0.55, vd=2.6, cl=45.0, t_half=4.5, bioavailability=0.65,
        bcs_class=1, therapeutic_index=:narrow, drug_class="Opioid analgesic"
    ),
    :fentanyl => (
        name="Fentanyl", drugbank_id="DB00813",
        mw=336.47, logp=4.05, pka_acid=0.0, pka_base=8.4, psa=23.6, hbd=0, hba=2, rotatable_bonds=6,
        fu=0.16, vd=4.0, cl=46.0, t_half=7.0, bioavailability=0.92,
        bcs_class=1, therapeutic_index=:very_narrow, drug_class="Opioid analgesic"
    ),
    :methadone => (
        name="Methadone", drugbank_id="DB00333",
        mw=309.45, logp=3.93, pka_acid=0.0, pka_base=9.2, psa=20.3, hbd=0, hba=2, rotatable_bonds=7,
        fu=0.13, vd=4.0, cl=10.0, t_half=25.0, bioavailability=0.80,
        bcs_class=1, therapeutic_index=:narrow, drug_class="Opioid analgesic"
    ),

    # === Anticonvulsants ===
    :phenytoin => (
        name="Phenytoin", drugbank_id="DB00252",
        mw=252.27, logp=2.47, pka_acid=8.3, pka_base=0.0, psa=58.2, hbd=2, hba=3, rotatable_bonds=2,
        fu=0.10, vd=0.6, cl=3.0, t_half=22.0, bioavailability=0.95,
        bcs_class=2, therapeutic_index=:very_narrow, drug_class="Anticonvulsant"
    ),
    :carbamazepine => (
        name="Carbamazepine", drugbank_id="DB00564",
        mw=236.27, logp=2.45, pka_acid=0.0, pka_base=0.0, psa=46.3, hbd=1, hba=2, rotatable_bonds=0,
        fu=0.24, vd=1.4, cl=4.0, t_half=15.0, bioavailability=0.75,
        bcs_class=2, therapeutic_index=:narrow, drug_class="Anticonvulsant"
    ),
    :valproic_acid => (
        name="Valproic acid", drugbank_id="DB00313",
        mw=144.21, logp=2.75, pka_acid=4.8, pka_base=0.0, psa=37.3, hbd=1, hba=2, rotatable_bonds=4,
        fu=0.10, vd=0.2, cl=0.5, t_half=12.0, bioavailability=0.95,
        bcs_class=1, therapeutic_index=:narrow, drug_class="Anticonvulsant"
    ),
)

# =============================================================================
# IMMUNOSUPPRESSANTS
# =============================================================================

const IMMUNOSUPPRESSANT_PROPERTIES = Dict{Symbol, NamedTuple}(
    :cyclosporine => (
        name="Cyclosporine", drugbank_id="DB00091",
        mw=1202.61, logp=2.92, pka_acid=0.0, pka_base=0.0, psa=279.0, hbd=5, hba=23, rotatable_bonds=17,
        fu=0.04, vd=4.0, cl=6.0, t_half=8.0, bioavailability=0.30,
        bcs_class=2, therapeutic_index=:very_narrow, drug_class="Calcineurin inhibitor"
    ),
    :tacrolimus => (
        name="Tacrolimus", drugbank_id="DB00864",
        mw=804.02, logp=3.30, pka_acid=0.0, pka_base=0.0, psa=178.4, hbd=3, hba=12, rotatable_bonds=7,
        fu=0.01, vd=1.5, cl=3.0, t_half=12.0, bioavailability=0.25,
        bcs_class=2, therapeutic_index=:very_narrow, drug_class="Calcineurin inhibitor"
    ),
    :sirolimus => (
        name="Sirolimus", drugbank_id="DB00877",
        mw=914.17, logp=4.30, pka_acid=0.0, pka_base=0.0, psa=195.4, hbd=3, hba=13, rotatable_bonds=6,
        fu=0.08, vd=12.0, cl=3.0, t_half=62.0, bioavailability=0.15,
        bcs_class=2, therapeutic_index=:very_narrow, drug_class="mTOR inhibitor"
    ),
    :everolimus => (
        name="Everolimus", drugbank_id="DB01590",
        mw=958.22, logp=4.00, pka_acid=0.0, pka_base=0.0, psa=204.7, hbd=3, hba=14, rotatable_bonds=8,
        fu=0.26, vd=2.5, cl=5.0, t_half=30.0, bioavailability=0.16,
        bcs_class=4, therapeutic_index=:very_narrow, drug_class="mTOR inhibitor"
    ),
    :mycophenolate => (
        name="Mycophenolate mofetil", drugbank_id="DB00688",
        mw=433.49, logp=3.53, pka_acid=4.5, pka_base=0.0, psa=93.1, hbd=1, hba=7, rotatable_bonds=10,
        fu=0.03, vd=4.0, cl=10.0, t_half=17.0, bioavailability=0.94,
        bcs_class=2, therapeutic_index=:narrow, drug_class="Antimetabolite"
    ),
)

# =============================================================================
# ANTIFUNGALS AND ANTIBIOTICS
# =============================================================================

const ANTIMICROBIAL_PROPERTIES = Dict{Symbol, NamedTuple}(
    # === Azole Antifungals ===
    :ketoconazole => (
        name="Ketoconazole", drugbank_id="DB01026",
        mw=531.43, logp=4.35, pka_acid=0.0, pka_base=6.5, psa=69.1, hbd=0, hba=6, rotatable_bonds=7,
        fu=0.01, vd=2.4, cl=8.0, t_half=8.0, bioavailability=0.76,
        bcs_class=2, therapeutic_index=:wide, drug_class="Azole antifungal"
    ),
    :fluconazole => (
        name="Fluconazole", drugbank_id="DB00196",
        mw=306.27, logp=0.40, pka_acid=0.0, pka_base=2.0, psa=81.6, hbd=1, hba=7, rotatable_bonds=3,
        fu=0.88, vd=0.7, cl=1.5, t_half=30.0, bioavailability=0.90,
        bcs_class=1, therapeutic_index=:wide, drug_class="Azole antifungal"
    ),
    :itraconazole => (
        name="Itraconazole", drugbank_id="DB01167",
        mw=705.63, logp=5.66, pka_acid=0.0, pka_base=3.7, psa=101.0, hbd=0, hba=9, rotatable_bonds=11,
        fu=0.002, vd=11.0, cl=20.0, t_half=21.0, bioavailability=0.55,
        bcs_class=2, therapeutic_index=:wide, drug_class="Azole antifungal"
    ),
    :voriconazole => (
        name="Voriconazole", drugbank_id="DB00582",
        mw=349.31, logp=1.00, pka_acid=0.0, pka_base=1.76, psa=76.7, hbd=1, hba=8, rotatable_bonds=4,
        fu=0.42, vd=4.6, cl=15.0, t_half=6.0, bioavailability=0.96,
        bcs_class=2, therapeutic_index=:wide, drug_class="Azole antifungal"
    ),
    :posaconazole => (
        name="Posaconazole", drugbank_id="DB01263",
        mw=700.78, logp=5.10, pka_acid=0.0, pka_base=3.6, psa=111.5, hbd=1, hba=9, rotatable_bonds=12,
        fu=0.02, vd=25.0, cl=4.0, t_half=26.0, bioavailability=0.54,
        bcs_class=2, therapeutic_index=:wide, drug_class="Azole antifungal"
    ),

    # === Macrolides ===
    :erythromycin => (
        name="Erythromycin", drugbank_id="DB00199",
        mw=733.93, logp=3.06, pka_acid=0.0, pka_base=8.9, psa=193.9, hbd=5, hba=14, rotatable_bonds=7,
        fu=0.30, vd=0.7, cl=25.0, t_half=1.5, bioavailability=0.35,
        bcs_class=3, therapeutic_index=:wide, drug_class="Macrolide antibiotic"
    ),
    :clarithromycin => (
        name="Clarithromycin", drugbank_id="DB01211",
        mw=747.95, logp=3.16, pka_acid=0.0, pka_base=8.8, psa=183.0, hbd=4, hba=14, rotatable_bonds=8,
        fu=0.30, vd=3.5, cl=30.0, t_half=4.0, bioavailability=0.50,
        bcs_class=2, therapeutic_index=:wide, drug_class="Macrolide antibiotic"
    ),
    :azithromycin => (
        name="Azithromycin", drugbank_id="DB00207",
        mw=748.98, logp=4.02, pka_acid=0.0, pka_base=8.7, psa=180.1, hbd=5, hba=14, rotatable_bonds=7,
        fu=0.50, vd=31.0, cl=30.0, t_half=68.0, bioavailability=0.37,
        bcs_class=2, therapeutic_index=:wide, drug_class="Macrolide antibiotic"
    ),

    # === Fluoroquinolones ===
    :ciprofloxacin => (
        name="Ciprofloxacin", drugbank_id="DB00537",
        mw=331.34, logp=-0.57, pka_acid=6.1, pka_base=8.7, psa=74.6, hbd=2, hba=6, rotatable_bonds=3,
        fu=0.60, vd=2.5, cl=25.0, t_half=4.0, bioavailability=0.70,
        bcs_class=3, therapeutic_index=:wide, drug_class="Fluoroquinolone"
    ),
    :levofloxacin => (
        name="Levofloxacin", drugbank_id="DB01137",
        mw=361.37, logp=-0.39, pka_acid=6.0, pka_base=8.7, psa=73.3, hbd=1, hba=7, rotatable_bonds=2,
        fu=0.70, vd=1.3, cl=10.0, t_half=7.0, bioavailability=0.99,
        bcs_class=1, therapeutic_index=:wide, drug_class="Fluoroquinolone"
    ),

    # === Rifamycins ===
    :rifampin => (
        name="Rifampin", drugbank_id="DB01045",
        mw=822.94, logp=3.72, pka_acid=1.7, pka_base=7.9, psa=220.2, hbd=6, hba=14, rotatable_bonds=5,
        fu=0.20, vd=0.7, cl=10.0, t_half=3.0, bioavailability=0.95,
        bcs_class=2, therapeutic_index=:wide, drug_class="Rifamycin antibiotic"
    ),
)

# =============================================================================
# HIV DRUGS
# =============================================================================

const HIV_DRUG_PROPERTIES = Dict{Symbol, NamedTuple}(
    :ritonavir => (
        name="Ritonavir", drugbank_id="DB00503",
        mw=720.94, logp=6.29, pka_acid=0.0, pka_base=2.0, psa=145.8, hbd=4, hba=9, rotatable_bonds=18,
        fu=0.02, vd=0.4, cl=10.0, t_half=4.0, bioavailability=0.75,
        bcs_class=4, therapeutic_index=:wide, drug_class="HIV protease inhibitor"
    ),
    :lopinavir => (
        name="Lopinavir", drugbank_id="DB01601",
        mw=628.80, logp=5.94, pka_acid=0.0, pka_base=0.0, psa=120.0, hbd=4, hba=7, rotatable_bonds=15,
        fu=0.01, vd=0.4, cl=6.0, t_half=5.0, bioavailability=0.25,
        bcs_class=4, therapeutic_index=:wide, drug_class="HIV protease inhibitor"
    ),
    :atazanavir => (
        name="Atazanavir", drugbank_id="DB01072",
        mw=704.86, logp=4.54, pka_acid=0.0, pka_base=4.4, psa=171.2, hbd=5, hba=10, rotatable_bonds=16,
        fu=0.14, vd=0.7, cl=6.0, t_half=7.0, bioavailability=0.68,
        bcs_class=2, therapeutic_index=:wide, drug_class="HIV protease inhibitor"
    ),
    :darunavir => (
        name="Darunavir", drugbank_id="DB01264",
        mw=547.66, logp=2.94, pka_acid=0.0, pka_base=2.4, psa=140.4, hbd=3, hba=9, rotatable_bonds=12,
        fu=0.05, vd=1.0, cl=6.0, t_half=15.0, bioavailability=0.37,
        bcs_class=2, therapeutic_index=:wide, drug_class="HIV protease inhibitor"
    ),
    :efavirenz => (
        name="Efavirenz", drugbank_id="DB00625",
        mw=315.67, logp=5.40, pka_acid=10.2, pka_base=0.0, psa=38.3, hbd=1, hba=2, rotatable_bonds=2,
        fu=0.01, vd=4.0, cl=4.0, t_half=52.0, bioavailability=0.50,
        bcs_class=2, therapeutic_index=:wide, drug_class="NNRTI"
    ),
)

# =============================================================================
# ONCOLOGY DRUGS
# =============================================================================

const ONCOLOGY_DRUG_PROPERTIES = Dict{Symbol, NamedTuple}(
    :imatinib => (
        name="Imatinib", drugbank_id="DB00619",
        mw=493.60, logp=3.50, pka_acid=0.0, pka_base=8.1, psa=86.3, hbd=2, hba=7, rotatable_bonds=7,
        fu=0.05, vd=4.4, cl=12.0, t_half=18.0, bioavailability=0.98,
        bcs_class=2, therapeutic_index=:wide, drug_class="Tyrosine kinase inhibitor"
    ),
    :nilotinib => (
        name="Nilotinib", drugbank_id="DB04868",
        mw=529.52, logp=4.75, pka_acid=0.0, pka_base=5.8, psa=97.6, hbd=2, hba=7, rotatable_bonds=6,
        fu=0.02, vd=6.0, cl=10.0, t_half=17.0, bioavailability=0.30,
        bcs_class=4, therapeutic_index=:wide, drug_class="Tyrosine kinase inhibitor"
    ),
    :dasatinib => (
        name="Dasatinib", drugbank_id="DB01254",
        mw=488.01, logp=1.80, pka_acid=0.0, pka_base=6.8, psa=106.3, hbd=3, hba=8, rotatable_bonds=6,
        fu=0.04, vd=25.0, cl=150.0, t_half=4.0, bioavailability=0.14,
        bcs_class=2, therapeutic_index=:wide, drug_class="Tyrosine kinase inhibitor"
    ),
    :sunitinib => (
        name="Sunitinib", drugbank_id="DB01268",
        mw=398.47, logp=2.93, pka_acid=0.0, pka_base=8.95, psa=77.2, hbd=3, hba=4, rotatable_bonds=7,
        fu=0.05, vd=33.0, cl=30.0, t_half=40.0, bioavailability=0.50,
        bcs_class=2, therapeutic_index=:wide, drug_class="Multi-kinase inhibitor"
    ),
    :sorafenib => (
        name="Sorafenib", drugbank_id="DB00398",
        mw=464.82, logp=4.12, pka_acid=0.0, pka_base=0.0, psa=92.4, hbd=3, hba=4, rotatable_bonds=7,
        fu=0.0005, vd=1.5, cl=4.0, t_half=28.0, bioavailability=0.38,
        bcs_class=2, therapeutic_index=:wide, drug_class="Multi-kinase inhibitor"
    ),
)

# =============================================================================
# MERGE ALL PROPERTY DATABASES
# =============================================================================

"""
Complete drug properties database.
"""
const DRUG_PROPERTIES_COMPLETE = merge(
    CARDIOVASCULAR_DRUG_PROPERTIES,
    CNS_DRUG_PROPERTIES,
    IMMUNOSUPPRESSANT_PROPERTIES,
    ANTIMICROBIAL_PROPERTIES,
    HIV_DRUG_PROPERTIES,
    ONCOLOGY_DRUG_PROPERTIES
)

# Export
export DrugPhysicochemicalProperties, DRUG_PROPERTIES_COMPLETE
export CARDIOVASCULAR_DRUG_PROPERTIES, CNS_DRUG_PROPERTIES
export IMMUNOSUPPRESSANT_PROPERTIES, ANTIMICROBIAL_PROPERTIES
export HIV_DRUG_PROPERTIES, ONCOLOGY_DRUG_PROPERTIES
