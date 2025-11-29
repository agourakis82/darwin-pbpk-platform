# ===========================================================================
# INTESTINAL TRANSPORTER-MEDIATED ABSORPTION
# ===========================================================================
# Comprehensive model of carrier-mediated drug uptake in the GI tract
#
# This module goes beyond P-gp efflux to include all major uptake transporters
# that enable absorption of hydrophilic drugs that would otherwise have
# negligible passive permeability.
#
# Key transporters modeled:
# 1. PEPT1 (SLC15A1) - Peptides, β-lactams, ACE inhibitors
# 2. OCT1/3 (SLC22A1/3) - Metformin, cationic drugs
# 3. OATP2B1 (SLCO2B1) - Statins, anionic drugs
# 4. ENT1/2 (SLC29A1/2) - Nucleosides, theophylline
# 5. MCT1 (SLC16A1) - Short-chain fatty acids, valproate
# 6. LAT2 (SLC7A8) - Amino acids, gabapentin, levodopa
# 7. ASBT (SLC10A2) - Bile acids, bile acid prodrugs
# 8. THTR1/2 (SLC19A2/3) - Thiamine-like drugs
#
# P-gp saturation kinetics also included
#
# References:
# - Estudante et al. 2013 (Adv Drug Deliv Rev) - Intestinal drug transporters
# - Giacomini et al. 2010 (Nat Rev Drug Discov) - Membrane transporters
# - Müller et al. 2017 (Clin Pharmacokinet) - Transporter PBPK
# ===========================================================================

module GITransportersModule

export TransporterSubstrate, calculate_carrier_mediated_uptake
export calculate_pgp_saturation, IntestinalTransporterParams
export predict_transporter_substrates, TransporterPrediction
export PEPT1_SUBSTRATES, OCT_SUBSTRATES, OATP_SUBSTRATES

# ===========================================================================
# TRANSPORTER KINETIC PARAMETERS
# ===========================================================================

"""
Transporter kinetic parameters (Km, Vmax) from literature.

Units:
- Km: μM (micromolar)
- Vmax: pmol/min/cm² (per unit intestinal surface)
- Jmax: effective permeability contribution at saturation

References compiled from:
- Brandsch 2013 (Pharm Res) - PEPT1 kinetics
- Graham et al. 2011 (Clin Pharmacokinet) - OCT/MATE
- Tamai & Nakanishi 2012 (Curr Opin Pharmacol) - OATP
"""
struct TransporterKinetics
    name::String
    gene::String
    km_uM::Float64          # Michaelis constant
    vmax_pmol_min_cm2::Float64  # Maximum velocity
    regional_expression::Vector{Float64}  # [stomach, duodenum, jejunum, ileum, colon]
end

# Intestinal uptake transporters
const PEPT1 = TransporterKinetics(
    "PEPT1", "SLC15A1",
    200.0,      # Km ~0.2 mM for most substrates
    5000.0,     # High capacity
    [0.0, 1.2, 1.0, 0.6, 0.1]  # Peak in proximal small intestine
)

const OCT1 = TransporterKinetics(
    "OCT1", "SLC22A1",
    500.0,      # Km ~0.5 mM
    1000.0,     # Moderate capacity
    [0.1, 0.8, 1.0, 0.5, 0.2]
)

const OCT3 = TransporterKinetics(
    "OCT3", "SLC22A3",
    1500.0,     # Higher Km (lower affinity)
    2000.0,     # Higher capacity
    [0.2, 0.6, 1.0, 0.8, 0.4]  # More uniform distribution
)

const OATP2B1 = TransporterKinetics(
    "OATP2B1", "SLCO2B1",
    50.0,       # Low Km (high affinity)
    500.0,      # Lower capacity
    [0.1, 1.0, 1.0, 0.8, 0.3]
)

const ENT1 = TransporterKinetics(
    "ENT1", "SLC29A1",
    100.0,      # Moderate affinity
    3000.0,     # Good capacity
    [0.3, 1.0, 1.0, 0.8, 0.5]
)

const ENT2 = TransporterKinetics(
    "ENT2", "SLC29A2",
    500.0,      # Lower affinity
    2000.0,
    [0.2, 0.8, 1.0, 0.9, 0.6]
)

const MCT1 = TransporterKinetics(
    "MCT1", "SLC16A1",
    1000.0,     # Km ~1 mM
    8000.0,     # High capacity
    [0.5, 1.0, 1.0, 0.7, 0.8]  # Also in colon
)

const LAT2 = TransporterKinetics(
    "LAT2", "SLC7A8",
    100.0,      # Moderate affinity
    2000.0,
    [0.1, 1.0, 1.0, 0.8, 0.2]
)

const ASBT = TransporterKinetics(
    "ASBT", "SLC10A2",
    10.0,       # High affinity
    500.0,      # Lower capacity
    [0.0, 0.0, 0.1, 1.0, 0.0]  # ONLY in terminal ileum
)

# Efflux transporters
const PGP = TransporterKinetics(
    "P-gp", "ABCB1",
    50.0,       # Typical Km range 10-100 μM
    10000.0,    # High capacity
    [0.1, 1.5, 1.0, 0.8, 0.3]  # Peak in duodenum
)

const BCRP = TransporterKinetics(
    "BCRP", "ABCG2",
    30.0,       # Often lower Km than P-gp
    5000.0,
    [0.1, 1.2, 1.0, 0.7, 0.2]
)

const MRP2 = TransporterKinetics(
    "MRP2", "ABCC2",
    100.0,
    3000.0,
    [0.1, 1.3, 1.0, 0.5, 0.2]
)

# ===========================================================================
# SUBSTRATE CLASSIFICATION
# ===========================================================================

"""
Known transporter substrates with affinity data.

This enables prediction of carrier-mediated absorption for known drugs.
"""
struct TransporterSubstrate
    drug_name::String
    transporter::String
    km_uM::Float64          # Drug-specific Km if known
    relative_affinity::Float64  # 0-1 scale
    clinical_evidence::Bool
end

# PEPT1 substrates (peptide-like drugs)
const PEPT1_SUBSTRATES = [
    TransporterSubstrate("Cephalexin", "PEPT1", 1200.0, 0.7, true),
    TransporterSubstrate("Amoxicillin", "PEPT1", 2800.0, 0.5, true),
    TransporterSubstrate("Ampicillin", "PEPT1", 3500.0, 0.4, true),
    TransporterSubstrate("Captopril", "PEPT1", 400.0, 0.8, true),
    TransporterSubstrate("Enalapril", "PEPT1", 300.0, 0.85, true),
    TransporterSubstrate("Lisinopril", "PEPT1", 150.0, 0.9, true),
    TransporterSubstrate("Valacyclovir", "PEPT1", 900.0, 0.75, true),
    TransporterSubstrate("Valganciclovir", "PEPT1", 600.0, 0.8, true),
    TransporterSubstrate("Oseltamivir", "PEPT1", 500.0, 0.8, true),
    TransporterSubstrate("Bestatin", "PEPT1", 50.0, 0.95, true),
]

# OCT substrates (organic cations)
const OCT_SUBSTRATES = [
    TransporterSubstrate("Metformin", "OCT1", 1500.0, 0.6, true),
    TransporterSubstrate("Cimetidine", "OCT1", 100.0, 0.9, true),
    TransporterSubstrate("Ranitidine", "OCT1", 200.0, 0.85, true),
    TransporterSubstrate("Famotidine", "OCT1", 300.0, 0.8, true),
    TransporterSubstrate("Metoclopramide", "OCT1", 150.0, 0.85, true),
    TransporterSubstrate("Procainamide", "OCT1", 100.0, 0.9, true),
    TransporterSubstrate("Quinidine", "OCT3", 50.0, 0.95, false),
    TransporterSubstrate("Propranolol", "OCT3", 200.0, 0.8, false),
]

# OATP2B1 substrates (organic anions, statins)
const OATP_SUBSTRATES = [
    TransporterSubstrate("Rosuvastatin", "OATP2B1", 2.0, 0.99, true),
    TransporterSubstrate("Pravastatin", "OATP2B1", 10.0, 0.95, true),
    TransporterSubstrate("Atorvastatin", "OATP2B1", 5.0, 0.97, true),
    TransporterSubstrate("Fexofenadine", "OATP2B1", 20.0, 0.9, true),
    TransporterSubstrate("Glibenclamide", "OATP2B1", 1.0, 0.99, true),
    TransporterSubstrate("Montelukast", "OATP2B1", 0.5, 0.99, true),
]

# ENT substrates (nucleosides, xanthines)
const ENT_SUBSTRATES = [
    TransporterSubstrate("Adenosine", "ENT1", 20.0, 0.95, true),
    TransporterSubstrate("Theophylline", "ENT1", 150.0, 0.85, true),
    TransporterSubstrate("Caffeine", "ENT1", 200.0, 0.8, true),
    TransporterSubstrate("Ribavirin", "ENT1", 50.0, 0.9, true),
    TransporterSubstrate("Gemcitabine", "ENT1", 30.0, 0.95, true),
    TransporterSubstrate("Cytarabine", "ENT1", 100.0, 0.85, true),
]

# MCT substrates (monocarboxylates)
const MCT_SUBSTRATES = [
    TransporterSubstrate("Valproic acid", "MCT1", 500.0, 0.85, true),
    TransporterSubstrate("Salicylic acid", "MCT1", 2000.0, 0.6, true),
    TransporterSubstrate("Probenecid", "MCT1", 200.0, 0.9, true),
    TransporterSubstrate("Nateglinide", "MCT1", 100.0, 0.9, true),
    TransporterSubstrate("GHB", "MCT1", 1000.0, 0.7, true),
]

# LAT substrates (amino acids, amino acid-like)
const LAT_SUBSTRATES = [
    TransporterSubstrate("Gabapentin", "LAT2", 250.0, 0.85, true),
    TransporterSubstrate("Pregabalin", "LAT2", 100.0, 0.9, true),
    TransporterSubstrate("Levodopa", "LAT2", 50.0, 0.95, true),
    TransporterSubstrate("Melphalan", "LAT2", 30.0, 0.95, true),
    TransporterSubstrate("Baclofen", "LAT2", 200.0, 0.85, true),
    TransporterSubstrate("Vigabatrin", "LAT2", 300.0, 0.8, true),
]

# P-gp substrates with Km data
const PGP_SUBSTRATES = [
    TransporterSubstrate("Digoxin", "P-gp", 70.0, 0.9, true),
    TransporterSubstrate("Loperamide", "P-gp", 5.0, 0.99, true),
    TransporterSubstrate("Cyclosporine", "P-gp", 3.0, 0.99, true),
    TransporterSubstrate("Tacrolimus", "P-gp", 5.0, 0.99, true),
    TransporterSubstrate("Ritonavir", "P-gp", 10.0, 0.95, true),
    TransporterSubstrate("Verapamil", "P-gp", 20.0, 0.9, true),
    TransporterSubstrate("Quinidine", "P-gp", 30.0, 0.9, true),
    TransporterSubstrate("Fexofenadine", "P-gp", 50.0, 0.85, true),
    TransporterSubstrate("Dabigatran", "P-gp", 15.0, 0.95, true),
]

# ===========================================================================
# CARRIER-MEDIATED UPTAKE CALCULATION
# ===========================================================================

"""
Calculate carrier-mediated uptake permeability.

Uses Michaelis-Menten kinetics:
    J = Vmax × C / (Km + C)

Effective permeability contribution:
    Peff_carrier = J / C = Vmax / (Km + C)

At low concentrations (C << Km):
    Peff_carrier ≈ Vmax / Km (first-order)

At high concentrations (C >> Km):
    Peff_carrier ≈ Vmax / C (approaches zero, saturated)
"""
function calculate_carrier_mediated_uptake(;
    drug_name::String,
    lumen_conc_uM::Float64,
    segment_index::Int = 3,  # 1=stomach, 2=duo, 3=jej, 4=ile, 5=col
    passive_peff::Float64 = 1.0e-4
)
    peff_total = passive_peff
    transporters_used = String[]

    # Check each transporter class
    for substrate_list in [PEPT1_SUBSTRATES, OCT_SUBSTRATES, OATP_SUBSTRATES,
                          ENT_SUBSTRATES, MCT_SUBSTRATES, LAT_SUBSTRATES]
        for substrate in substrate_list
            if lowercase(substrate.drug_name) == lowercase(drug_name)
                # Get transporter kinetics
                transporter = get_transporter(substrate.transporter)
                if transporter !== nothing
                    # Regional expression
                    expression = transporter.regional_expression[segment_index]

                    # Michaelis-Menten: Peff = Vmax / (Km + C)
                    km = substrate.km_uM
                    vmax = transporter.vmax_pmol_min_cm2 * expression * substrate.relative_affinity

                    # Convert Vmax to Peff units (cm/s)
                    # Vmax in pmol/min/cm² → cm/s: divide by concentration (pmol/mL = μM)
                    peff_carrier = (vmax * 1e-12) / (60.0 * (km + lumen_conc_uM) * 1e-6)
                    # Simplifies to: vmax / (60 * (km + C)) with units cm/s
                    peff_carrier = vmax / (60.0 * (km + lumen_conc_uM)) * 1e-4

                    peff_total += peff_carrier
                    push!(transporters_used, substrate.transporter)
                end
            end
        end
    end

    return (
        peff_total_cm_s = peff_total,
        peff_carrier_cm_s = peff_total - passive_peff,
        transporters = transporters_used,
        carrier_fraction = (peff_total - passive_peff) / peff_total
    )
end

function get_transporter(name::String)
    transporters = Dict(
        "PEPT1" => PEPT1,
        "OCT1" => OCT1,
        "OCT3" => OCT3,
        "OATP2B1" => OATP2B1,
        "ENT1" => ENT1,
        "ENT2" => ENT2,
        "MCT1" => MCT1,
        "LAT2" => LAT2,
        "ASBT" => ASBT,
        "P-gp" => PGP,
        "BCRP" => BCRP,
        "MRP2" => MRP2
    )
    return get(transporters, name, nothing)
end

# ===========================================================================
# P-gp SATURATION MODEL
# ===========================================================================

"""
Calculate P-gp-mediated efflux with saturation kinetics.

The key insight: clinical efflux ratio (ER) is measured at low concentrations.
At therapeutic doses, lumen concentration often >> Km, causing saturation.

ER_apparent = 1 + (ER_intrinsic - 1) × Km / (Km + C)

At low C: ER_apparent ≈ ER_intrinsic (what's measured in vitro)
At high C: ER_apparent → 1 (P-gp saturated, no net efflux)

This explains digoxin paradox:
- ER = 30 in vitro
- 250 mg dose in 250 mL → 1000 μM lumen concentration
- Digoxin Km ≈ 70 μM
- ER_apparent = 1 + 29 × 70 / (70 + 1000) = 1 + 29 × 0.065 = 2.9
- Actual efflux only 3x, not 30x!
"""
function calculate_pgp_saturation(;
    drug_name::String = "",
    intrinsic_er::Float64 = 1.0,
    lumen_conc_uM::Float64 = 100.0,
    km_uM::Float64 = 50.0,  # Default P-gp Km
    segment_index::Int = 3
)
    # Check if we have drug-specific Km
    for substrate in PGP_SUBSTRATES
        if lowercase(substrate.drug_name) == lowercase(drug_name)
            km_uM = substrate.km_uM
            break
        end
    end

    # Regional P-gp expression
    expression = PGP.regional_expression[segment_index]

    # Saturation effect
    saturation_factor = km_uM / (km_uM + lumen_conc_uM)

    # Apparent ER
    er_apparent = 1.0 + (intrinsic_er - 1.0) * saturation_factor * expression

    # Net effect on permeability
    # Peff_net = Peff_passive / ER_apparent
    efflux_reduction = 1.0 / er_apparent

    return (
        er_intrinsic = intrinsic_er,
        er_apparent = er_apparent,
        saturation_factor = saturation_factor,
        efflux_reduction = efflux_reduction,
        km_uM = km_uM,
        lumen_conc_uM = lumen_conc_uM
    )
end

# ===========================================================================
# TRANSPORTER SUBSTRATE PREDICTION
# ===========================================================================

"""
Predict which transporters a drug might use based on structure.

Uses simple rules:
- PEPT1: Peptide bonds, α-amino acids, carboxylic acid + amine
- OCT: Cationic at pH 6-7, quaternary nitrogen, hydrophilic bases
- OATP: Anionic, organic acids, high MW
- ENT: Purine/pyrimidine-like, nucleoside mimics
- MCT: Short-chain carboxylic acids
- LAT: α-amino acid structure, aromatic amino acids
"""
struct TransporterPrediction
    transporter::String
    probability::Float64  # 0-1
    rationale::String
end

function predict_transporter_substrates(;
    logP::Float64,
    MW::Float64,
    pKa::Union{Float64, Nothing},
    charge_type::Symbol,
    has_peptide_bond::Bool = false,
    has_amino_acid::Bool = false,
    has_nucleoside::Bool = false,
    has_carboxylic_acid::Bool = false,
    drug_class::Symbol = :unknown
)
    predictions = TransporterPrediction[]

    # PEPT1: Peptide-like
    if has_peptide_bond || (has_amino_acid && has_carboxylic_acid)
        prob = 0.8
        push!(predictions, TransporterPrediction("PEPT1", prob, "Peptide/amino acid structure"))
    elseif drug_class in [:beta_lactam, :ace_inhibitor, :antiviral]
        push!(predictions, TransporterPrediction("PEPT1", 0.7, "Drug class association"))
    end

    # OCT1/3: Cationic drugs
    if charge_type == :base && pKa !== nothing && pKa > 7.0
        if logP < 0
            push!(predictions, TransporterPrediction("OCT1", 0.8, "Hydrophilic cation"))
        elseif logP < 2
            push!(predictions, TransporterPrediction("OCT3", 0.6, "Moderate lipophilicity cation"))
        end
    end

    # OATP2B1: Anionic drugs
    if charge_type == :acid || has_carboxylic_acid
        if MW > 400
            push!(predictions, TransporterPrediction("OATP2B1", 0.7, "Organic anion, high MW"))
        else
            push!(predictions, TransporterPrediction("OATP2B1", 0.4, "Organic anion"))
        end
    end

    # ENT: Nucleoside-like
    if has_nucleoside
        push!(predictions, TransporterPrediction("ENT1", 0.85, "Nucleoside structure"))
    elseif drug_class == :xanthine
        push!(predictions, TransporterPrediction("ENT1", 0.7, "Xanthine class"))
    end

    # MCT: Short-chain acids
    if has_carboxylic_acid && MW < 250
        push!(predictions, TransporterPrediction("MCT1", 0.6, "Small carboxylic acid"))
    end

    # LAT2: Amino acid analogs
    if has_amino_acid && !has_peptide_bond
        if drug_class in [:anticonvulsant, :parkinsonian]
            push!(predictions, TransporterPrediction("LAT2", 0.8, "Amino acid analog"))
        else
            push!(predictions, TransporterPrediction("LAT2", 0.5, "Amino acid-like structure"))
        end
    end

    return predictions
end

# ===========================================================================
# INTEGRATED PERMEABILITY CALCULATION
# ===========================================================================

"""
Calculate total effective permeability including all mechanisms.

Peff_total = (Peff_passive + Peff_paracellular + Peff_carrier) × efflux_factor
"""
function calculate_integrated_permeability(;
    drug_name::String,
    logP::Float64,
    MW::Float64,
    pKa::Union{Float64, Nothing},
    charge_type::Symbol,
    dose_mg::Float64,
    volume_mL::Float64 = 250.0,  # GI fluid volume
    segment_index::Int = 3,
    intrinsic_er::Float64 = 1.0,
    drug_class::Symbol = :unknown
)
    # Estimate lumen concentration
    lumen_conc_uM = (dose_mg / MW) * 1e6 / volume_mL  # mg/mL → μM

    # Passive permeability (simplified)
    if logP < -2
        peff_passive = 1.0e-4
    elseif logP < 0
        peff_passive = 1.0e-4 + (logP + 2) * 3.0e-4
    elseif logP < 2
        peff_passive = 7.0e-4 + logP * 6.0e-4
    elseif logP < 4
        peff_passive = 19.0e-4 + (logP - 2) * 5.0e-4
    else
        peff_passive = 29.0e-4 / (1.0 + 0.3 * (logP - 4))
    end

    # MW penalty
    if MW > 500
        peff_passive *= exp(-0.002 * (MW - 500))
    end

    # Zwitterion penalty (poor membrane permeability due to permanent charge)
    # Key for fexofenadine, statins - charged at all pH
    if charge_type == :zwitterion && MW > 400
        zwitterion_factor = 0.3  # 70% reduction in passive permeability
        peff_passive *= zwitterion_factor
    end

    # Paracellular
    peff_paracellular = 0.0
    if MW < 350 && logP < 1
        size_factor = (350 - MW) / 350
        hydrophilicity_factor = max(0.2, (1 - logP) / 2)
        peff_paracellular = 4.0e-4 * size_factor * hydrophilicity_factor
    end

    # Carrier-mediated
    carrier = calculate_carrier_mediated_uptake(
        drug_name = drug_name,
        lumen_conc_uM = lumen_conc_uM,
        segment_index = segment_index,
        passive_peff = 0.0
    )
    peff_carrier = carrier.peff_carrier_cm_s

    # P-gp efflux with saturation
    pgp = calculate_pgp_saturation(
        drug_name = drug_name,
        intrinsic_er = intrinsic_er,
        lumen_conc_uM = lumen_conc_uM,
        segment_index = segment_index
    )

    # Total
    peff_total = (peff_passive + peff_paracellular + peff_carrier) * pgp.efflux_reduction

    return (
        peff_total = peff_total,
        peff_passive = peff_passive,
        peff_paracellular = peff_paracellular,
        peff_carrier = peff_carrier,
        pgp_er_apparent = pgp.er_apparent,
        lumen_conc_uM = lumen_conc_uM,
        transporters = carrier.transporters,
        carrier_fraction = peff_carrier / (peff_passive + peff_paracellular + peff_carrier)
    )
end

end # module
