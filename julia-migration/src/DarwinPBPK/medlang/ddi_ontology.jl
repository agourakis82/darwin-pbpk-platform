# =============================================================================
# DDI ONTOLOGY INTEGRATION MODULE - MedLang v1.0
# =============================================================================
# Darwin PBPK Platform - Publication-Ready Mechanistic Model
#
# ONTOLOGY INTEGRATION FOR DRUG-DRUG INTERACTIONS:
# This module provides mappings to standard biomedical ontologies for DDI,
# enabling interoperability, semantic validation, and literature-based inference.
#
# Integrated Ontologies:
# 1. DINTO (Drug-Drug Interactions Ontology) - OBO Foundry
#    - DDI mechanisms, PK/PD classifications
#    - SWRL rules for inference
#    - http://purl.obolibrary.org/obo/dinto.owl
#
# 2. DIDEO (DDI Evidence Ontology)
#    - Evidence types and clinical study classifications
#    - http://purl.obolibrary.org/obo/dideo.owl
#
# 3. ChEBI (Chemical Entities of Biological Interest)
#    - Drug chemical structures and classifications
#    - https://www.ebi.ac.uk/chebi/
#
# 4. Human Disease Ontology (DOID)
#    - Disease classifications for DDI context
#    - http://purl.obolibrary.org/obo/doid.owl
#
# 5. NDF-RT/MED-RT (National Drug File Reference Terminology)
#    - VA drug classifications, mechanisms of action
#    - CYP enzyme interactions
#
# 6. DrugBank
#    - Comprehensive CYP450 metabolism data
#    - Clinical DDI evidence
#
# 7. RxNorm
#    - Drug nomenclature standardization
#    - Cross-reference identifiers
#
# Literature Basis:
# - OBO Foundry Principles (Smith et al., 2007)
# - DINTO: Herrero-Zazo et al. (2015) J Chem Inf Model
# - DIDEO: Brochhausen et al. (2014) ICBO
# - ChEBI: Hastings et al. (2016) Nucleic Acids Res
# - Disease Ontology: Schriml et al. (2019) Nucleic Acids Res
#
# Author: Dr. Demetrios Agourakis
# Date: November 2025
# =============================================================================

module DDIOntology

using ..BayesianDDIModel

export OntologyTerm, DrugOntologyMapping, DDIMechanismOntology
export DiseaseOntologyMapping, EvidenceTypeOntology
export CYPOntologyMapping, TransporterOntologyMapping
export get_dinto_term, get_chebi_id, get_doid_term
export get_ndfrt_mechanism, get_drugbank_id, get_rxnorm_cui
export map_cyp_to_ontology, map_transporter_to_ontology
export map_ddi_mechanism_to_dinto, map_disease_context
export validate_ontology_mapping, get_literature_ddi_evidence
export DDI_CLINICAL_DATABASE, DINTO_MECHANISMS, CHEBI_DRUGS
export create_ontology_annotated_ddi, export_to_rdf
export ClinicalDDIEvidence
export list_available_drugs, list_clinical_ddi_pairs, get_ddi_by_mechanism
export DISEASE_DDI_CONTEXT, CYP_ONTOLOGY, TRANSPORTER_ONTOLOGY, EVIDENCE_TYPES

# =============================================================================
# ONTOLOGY TERM STRUCTURES
# =============================================================================

"""
    OntologyTerm

Represents a term from a biomedical ontology.
"""
struct OntologyTerm
    id::String              # e.g., "DINTO:00001", "CHEBI:6801"
    label::String           # Human-readable label
    ontology::Symbol        # :DINTO, :DIDEO, :ChEBI, :DOID, :NDFRT, :RxNorm
    uri::String             # Full URI (PURL)
    definition::String      # Textual definition
    synonyms::Vector{String}
    parent_terms::Vector{String}  # is_a relationships
end

"""
    DrugOntologyMapping

Maps a drug to multiple ontology identifiers.
"""
struct DrugOntologyMapping
    drug_name::String

    # Primary identifiers
    chebi_id::String        # ChEBI ID (e.g., "CHEBI:6801")
    drugbank_id::String     # DrugBank ID (e.g., "DB00001")
    rxnorm_cui::String      # RxNorm CUI

    # Classification terms
    atc_codes::Vector{String}       # ATC classification
    ndfrt_classes::Vector{String}   # NDF-RT therapeutic classes

    # Chemical classification
    chemical_class::String          # ChEBI chemical class
    molecular_framework::String     # ChemOnt classification

    # Mechanism of action
    moa_terms::Vector{OntologyTerm}
end

"""
    DDIMechanismOntology

Maps DDI mechanisms to DINTO ontology terms.
"""
struct DDIMechanismOntology
    mechanism_type::BayesianDDIModel.DDIMechanismType
    dinto_id::String
    dinto_label::String
    dideo_evidence_types::Vector{String}
    ndfrt_interaction_type::String
    description::String
end

"""
    DiseaseOntologyMapping

Maps disease contexts to DOID terms.
"""
struct DiseaseOntologyMapping
    disease_name::String
    doid::String            # e.g., "DOID:9352" for diabetes
    icd10_codes::Vector{String}
    mesh_terms::Vector{String}
    snomed_ct_id::String
    relevant_ddi_modifiers::Vector{String}  # How disease affects DDI
end

"""
    EvidenceTypeOntology

Maps evidence types to DIDEO ontology.
"""
struct EvidenceTypeOntology
    evidence_type::Symbol
    dideo_id::String
    dideo_label::String
    strength::Symbol        # :strong, :moderate, :weak
    study_design::String
end

"""
    CYPOntologyMapping

Maps CYP enzymes to ontology terms.
"""
struct CYPOntologyMapping
    enzyme::BayesianDDIModel.CYPEnzyme
    gene_symbol::String     # e.g., "CYP3A4"
    uniprot_id::String      # UniProt accession
    ncbi_gene_id::String    # NCBI Gene ID
    hgnc_id::String         # HGNC ID
    go_terms::Vector{String}  # Gene Ontology terms
    chebi_cofactors::Vector{String}  # Required cofactors
end

"""
    TransporterOntologyMapping

Maps transporters to ontology terms.
"""
struct TransporterOntologyMapping
    transporter::BayesianDDIModel.TransporterType
    gene_symbol::String
    protein_name::String
    uniprot_id::String
    slc_family::String      # SLC or ABC family
    go_cellular_component::String
    tissue_expression::Vector{String}
end

# =============================================================================
# DINTO MECHANISM MAPPINGS
# =============================================================================

"""
DINTO mechanism type mappings.
Based on DINTO ontology (http://purl.obolibrary.org/obo/dinto.owl)
"""
const DINTO_MECHANISMS = Dict{BayesianDDIModel.DDIMechanismType, DDIMechanismOntology}(
    BayesianDDIModel.CYP_COMPETITIVE_INHIBITION => DDIMechanismOntology(
        BayesianDDIModel.CYP_COMPETITIVE_INHIBITION,
        "DINTO:0000102",
        "competitive enzyme inhibition-based drug-drug interaction",
        ["DIDEO:0000001", "DIDEO:0000015"],  # in vitro, clinical study
        "NDF-RT:C1512698",
        "DDI where perpetrator competes with victim for CYP active site"
    ),

    BayesianDDIModel.CYP_NONCOMPETITIVE_INHIBITION => DDIMechanismOntology(
        BayesianDDIModel.CYP_NONCOMPETITIVE_INHIBITION,
        "DINTO:0000103",
        "non-competitive enzyme inhibition-based drug-drug interaction",
        ["DIDEO:0000001", "DIDEO:0000015"],
        "NDF-RT:C1512699",
        "DDI where perpetrator binds allosteric site reducing enzyme activity"
    ),

    BayesianDDIModel.CYP_MECHANISM_BASED_INHIBITION => DDIMechanismOntology(
        BayesianDDIModel.CYP_MECHANISM_BASED_INHIBITION,
        "DINTO:0000104",
        "mechanism-based enzyme inhibition drug-drug interaction",
        ["DIDEO:0000001", "DIDEO:0000015", "DIDEO:0000020"],
        "NDF-RT:C2825146",
        "Time-dependent irreversible CYP inactivation (suicide inhibition)"
    ),

    BayesianDDIModel.CYP_INDUCTION => DDIMechanismOntology(
        BayesianDDIModel.CYP_INDUCTION,
        "DINTO:0000105",
        "enzyme induction-based drug-drug interaction",
        ["DIDEO:0000001", "DIDEO:0000015"],
        "NDF-RT:C1512697",
        "DDI where perpetrator increases CYP expression via nuclear receptors"
    ),

    BayesianDDIModel.TRANSPORTER_INHIBITION => DDIMechanismOntology(
        BayesianDDIModel.TRANSPORTER_INHIBITION,
        "DINTO:0000106",
        "transporter inhibition-based drug-drug interaction",
        ["DIDEO:0000001", "DIDEO:0000015"],
        "NDF-RT:C3853614",
        "DDI mediated by inhibition of drug transporters (P-gp, OATP, etc.)"
    ),

    BayesianDDIModel.TRANSPORTER_INDUCTION => DDIMechanismOntology(
        BayesianDDIModel.TRANSPORTER_INDUCTION,
        "DINTO:0000107",
        "transporter induction-based drug-drug interaction",
        ["DIDEO:0000001"],
        "NDF-RT:C3853615",
        "DDI mediated by induction of drug transporters"
    ),

    BayesianDDIModel.PROTEIN_BINDING_DISPLACEMENT => DDIMechanismOntology(
        BayesianDDIModel.PROTEIN_BINDING_DISPLACEMENT,
        "DINTO:0000108",
        "plasma protein binding displacement drug-drug interaction",
        ["DIDEO:0000001", "DIDEO:0000015"],
        "NDF-RT:C0678780",
        "DDI where perpetrator displaces victim from plasma proteins"
    ),

    BayesianDDIModel.RENAL_COMPETITION => DDIMechanismOntology(
        BayesianDDIModel.RENAL_COMPETITION,
        "DINTO:0000109",
        "renal elimination-based drug-drug interaction",
        ["DIDEO:0000001", "DIDEO:0000015"],
        "NDF-RT:C0877182",
        "DDI where drugs compete for renal tubular secretion"
    )
)

# =============================================================================
# ChEBI DRUG MAPPINGS
# =============================================================================

"""
ChEBI identifiers for common DDI drugs.
Source: https://www.ebi.ac.uk/chebi/
"""
const CHEBI_DRUGS = Dict{Symbol, DrugOntologyMapping}(
    # Strong CYP3A4 inhibitors
    :ketoconazole => DrugOntologyMapping(
        "Ketoconazole",
        "CHEBI:48339",
        "DB01026",
        "6135",
        ["J02AB02", "D01AC08"],
        ["NDF-RT:N0000175577"],
        "imidazole antifungal agent",
        "Organic heterocyclic compounds",
        [OntologyTerm(
            "CHEBI:48339", "ketoconazole", :ChEBI,
            "http://purl.obolibrary.org/obo/CHEBI_48339",
            "An imidazole antifungal agent", String[],
            ["CHEBI:35718"]
        )]
    ),

    :itraconazole => DrugOntologyMapping(
        "Itraconazole",
        "CHEBI:6076",
        "DB01167",
        "28031",
        ["J02AC02"],
        ["NDF-RT:N0000175579"],
        "triazole antifungal agent",
        "Organic heterocyclic compounds",
        [OntologyTerm(
            "CHEBI:6076", "itraconazole", :ChEBI,
            "http://purl.obolibrary.org/obo/CHEBI_6076",
            "A triazole antifungal agent", String[],
            ["CHEBI:38996"]
        )]
    ),

    # Strong CYP3A4 inducer
    :rifampin => DrugOntologyMapping(
        "Rifampin",
        "CHEBI:28077",
        "DB01045",
        "55672",
        ["J04AB02"],
        ["NDF-RT:N0000006016"],
        "rifamycin antibiotic",
        "Organic heteropolycyclic compounds",
        [OntologyTerm(
            "CHEBI:28077", "rifampicin", :ChEBI,
            "http://purl.obolibrary.org/obo/CHEBI_28077",
            "A rifamycin antibiotic", String[],
            ["CHEBI:71823"]
        )]
    ),

    # CYP2C9 inhibitor
    :fluconazole => DrugOntologyMapping(
        "Fluconazole",
        "CHEBI:46081",
        "DB00196",
        "4450",
        ["J02AC01"],
        ["NDF-RT:N0000175575"],
        "triazole antifungal agent",
        "Organic heterocyclic compounds",
        [OntologyTerm(
            "CHEBI:46081", "fluconazole", :ChEBI,
            "http://purl.obolibrary.org/obo/CHEBI_46081",
            "A triazole antifungal agent", String[],
            ["CHEBI:38996"]
        )]
    ),

    # OATP inhibitor
    :cyclosporine => DrugOntologyMapping(
        "Cyclosporine",
        "CHEBI:4031",
        "DB00091",
        "36437",
        ["L04AD01"],
        ["NDF-RT:N0000008004"],
        "cyclic peptide immunosuppressant",
        "Organic cyclic compounds",
        [OntologyTerm(
            "CHEBI:4031", "ciclosporin", :ChEBI,
            "http://purl.obolibrary.org/obo/CHEBI_4031",
            "A cyclic undecapeptide", String[],
            ["CHEBI:23449"]
        )]
    ),

    # CYP3A4 probe substrate
    :midazolam => DrugOntologyMapping(
        "Midazolam",
        "CHEBI:6931",
        "DB00683",
        "6129",
        ["N05CD08"],
        ["NDF-RT:N0000007632"],
        "benzodiazepine",
        "Organic heterocyclic compounds",
        [OntologyTerm(
            "CHEBI:6931", "midazolam", :ChEBI,
            "http://purl.obolibrary.org/obo/CHEBI_6931",
            "A benzodiazepine", String[],
            ["CHEBI:22720"]
        )]
    ),

    # CYP2C9 substrate (NTI)
    :warfarin => DrugOntologyMapping(
        "Warfarin",
        "CHEBI:10033",
        "DB00682",
        "11289",
        ["B01AA03"],
        ["NDF-RT:N0000146165"],
        "coumarin anticoagulant",
        "Organic oxygen compounds",
        [OntologyTerm(
            "CHEBI:10033", "warfarin", :ChEBI,
            "http://purl.obolibrary.org/obo/CHEBI_10033",
            "A coumarin that is 4-hydroxycoumarin", String[],
            ["CHEBI:28794"]
        )]
    ),

    # OATP substrate
    :rosuvastatin => DrugOntologyMapping(
        "Rosuvastatin",
        "CHEBI:38545",
        "DB01098",
        "83600",
        ["C10AA07"],
        ["NDF-RT:N0000175879"],
        "statin",
        "Organic heterocyclic compounds",
        [OntologyTerm(
            "CHEBI:38545", "rosuvastatin", :ChEBI,
            "http://purl.obolibrary.org/obo/CHEBI_38545",
            "A statin", String[],
            ["CHEBI:87631"]
        )]
    ),

    # CYP3A4 substrate (NTI)
    :tacrolimus => DrugOntologyMapping(
        "Tacrolimus",
        "CHEBI:61049",
        "DB00864",
        "42316",
        ["L04AD02"],
        ["NDF-RT:N0000175874"],
        "macrolide immunosuppressant",
        "Organic macrocyclic compounds",
        [OntologyTerm(
            "CHEBI:61049", "tacrolimus", :ChEBI,
            "http://purl.obolibrary.org/obo/CHEBI_61049",
            "A macrolide immunosuppressant", String[],
            ["CHEBI:25106"]
        )]
    ),

    :simvastatin => DrugOntologyMapping(
        "Simvastatin",
        "CHEBI:9150",
        "DB00641",
        "36567",
        ["C10AA01"],
        ["NDF-RT:N0000005997"],
        "statin",
        "Organic oxygen compounds",
        [OntologyTerm(
            "CHEBI:9150", "simvastatin", :ChEBI,
            "http://purl.obolibrary.org/obo/CHEBI_9150",
            "A statin", String[],
            ["CHEBI:87631"]
        )]
    )
)

# =============================================================================
# CYP ENZYME ONTOLOGY MAPPINGS
# =============================================================================

"""
CYP enzyme ontology mappings.
Sources: UniProt, NCBI Gene, HGNC, GO
"""
const CYP_ONTOLOGY = Dict{BayesianDDIModel.CYPEnzyme, CYPOntologyMapping}(
    BayesianDDIModel.CYP3A4 => CYPOntologyMapping(
        BayesianDDIModel.CYP3A4,
        "CYP3A4",
        "P08684",
        "1576",
        "HGNC:2637",
        ["GO:0004497", "GO:0020037", "GO:0005783"],
        ["CHEBI:60240", "CHEBI:18070"]  # NADPH, heme
    ),

    BayesianDDIModel.CYP3A5 => CYPOntologyMapping(
        BayesianDDIModel.CYP3A5,
        "CYP3A5",
        "P20815",
        "1577",
        "HGNC:2638",
        ["GO:0004497", "GO:0020037", "GO:0005783"],
        ["CHEBI:60240", "CHEBI:18070"]
    ),

    BayesianDDIModel.CYP2D6 => CYPOntologyMapping(
        BayesianDDIModel.CYP2D6,
        "CYP2D6",
        "P10635",
        "1565",
        "HGNC:2625",
        ["GO:0004497", "GO:0020037", "GO:0005783"],
        ["CHEBI:60240", "CHEBI:18070"]
    ),

    BayesianDDIModel.CYP2C9 => CYPOntologyMapping(
        BayesianDDIModel.CYP2C9,
        "CYP2C9",
        "P11712",
        "1559",
        "HGNC:2623",
        ["GO:0004497", "GO:0020037", "GO:0005783"],
        ["CHEBI:60240", "CHEBI:18070"]
    ),

    BayesianDDIModel.CYP2C19 => CYPOntologyMapping(
        BayesianDDIModel.CYP2C19,
        "CYP2C19",
        "P33261",
        "1557",
        "HGNC:2621",
        ["GO:0004497", "GO:0020037", "GO:0005783"],
        ["CHEBI:60240", "CHEBI:18070"]
    ),

    BayesianDDIModel.CYP1A2 => CYPOntologyMapping(
        BayesianDDIModel.CYP1A2,
        "CYP1A2",
        "P05177",
        "1544",
        "HGNC:2596",
        ["GO:0004497", "GO:0020037", "GO:0005783"],
        ["CHEBI:60240", "CHEBI:18070"]
    ),

    BayesianDDIModel.CYP2C8 => CYPOntologyMapping(
        BayesianDDIModel.CYP2C8,
        "CYP2C8",
        "P10632",
        "1558",
        "HGNC:2622",
        ["GO:0004497", "GO:0020037", "GO:0005783"],
        ["CHEBI:60240", "CHEBI:18070"]
    ),

    BayesianDDIModel.CYP2B6 => CYPOntologyMapping(
        BayesianDDIModel.CYP2B6,
        "CYP2B6",
        "P20813",
        "1555",
        "HGNC:2615",
        ["GO:0004497", "GO:0020037", "GO:0005783"],
        ["CHEBI:60240", "CHEBI:18070"]
    )
)

# =============================================================================
# TRANSPORTER ONTOLOGY MAPPINGS
# =============================================================================

"""
Transporter ontology mappings.
Sources: UniProt, SLC/ABC nomenclature, GO
"""
const TRANSPORTER_ONTOLOGY = Dict{BayesianDDIModel.TransporterType, TransporterOntologyMapping}(
    BayesianDDIModel.PGP => TransporterOntologyMapping(
        BayesianDDIModel.PGP,
        "ABCB1",
        "P-glycoprotein 1",
        "P08183",
        "ABC",
        "GO:0005886",  # plasma membrane
        ["liver", "intestine", "kidney", "brain", "placenta"]
    ),

    BayesianDDIModel.BCRP => TransporterOntologyMapping(
        BayesianDDIModel.BCRP,
        "ABCG2",
        "Breast cancer resistance protein",
        "Q9UNQ0",
        "ABC",
        "GO:0005886",
        ["liver", "intestine", "placenta", "brain"]
    ),

    BayesianDDIModel.OATP1B1 => TransporterOntologyMapping(
        BayesianDDIModel.OATP1B1,
        "SLCO1B1",
        "Organic anion transporting polypeptide 1B1",
        "Q9Y6L6",
        "SLC",
        "GO:0016021",  # integral membrane
        ["liver"]
    ),

    BayesianDDIModel.OATP1B3 => TransporterOntologyMapping(
        BayesianDDIModel.OATP1B3,
        "SLCO1B3",
        "Organic anion transporting polypeptide 1B3",
        "Q9NPD5",
        "SLC",
        "GO:0016021",
        ["liver"]
    ),

    BayesianDDIModel.OCT1 => TransporterOntologyMapping(
        BayesianDDIModel.OCT1,
        "SLC22A1",
        "Organic cation transporter 1",
        "O15245",
        "SLC",
        "GO:0005886",
        ["liver"]
    ),

    BayesianDDIModel.OCT2 => TransporterOntologyMapping(
        BayesianDDIModel.OCT2,
        "SLC22A2",
        "Organic cation transporter 2",
        "O15244",
        "SLC",
        "GO:0005886",
        ["kidney"]
    ),

    BayesianDDIModel.OAT1 => TransporterOntologyMapping(
        BayesianDDIModel.OAT1,
        "SLC22A6",
        "Organic anion transporter 1",
        "Q4U2R8",
        "SLC",
        "GO:0005886",
        ["kidney"]
    ),

    BayesianDDIModel.OAT3 => TransporterOntologyMapping(
        BayesianDDIModel.OAT3,
        "SLC22A8",
        "Organic anion transporter 3",
        "Q8TCC7",
        "SLC",
        "GO:0005886",
        ["kidney", "brain"]
    ),

    BayesianDDIModel.MATE1 => TransporterOntologyMapping(
        BayesianDDIModel.MATE1,
        "SLC47A1",
        "Multidrug and toxin extrusion protein 1",
        "Q96FL8",
        "SLC",
        "GO:0005886",
        ["liver", "kidney"]
    )
)

# =============================================================================
# DISEASE ONTOLOGY MAPPINGS (DOID)
# =============================================================================

"""
Disease conditions affecting DDI.
Source: Human Disease Ontology (https://disease-ontology.org/)
"""
const DISEASE_DDI_CONTEXT = Dict{Symbol, DiseaseOntologyMapping}(
    :hepatic_impairment => DiseaseOntologyMapping(
        "Hepatic impairment",
        "DOID:409",  # liver disease
        ["K70-K77"],
        ["D008107"],
        "SCTID:235856003",
        ["reduced_CYP_activity", "reduced_protein_binding", "increased_bioavailability"]
    ),

    :renal_impairment => DiseaseOntologyMapping(
        "Renal impairment",
        "DOID:557",  # kidney disease
        ["N17-N19"],
        ["D007674"],
        "SCTID:90688005",
        ["reduced_renal_clearance", "altered_protein_binding", "reduced_metabolism"]
    ),

    :heart_failure => DiseaseOntologyMapping(
        "Heart failure",
        "DOID:6000",
        ["I50"],
        ["D006333"],
        "SCTID:84114007",
        ["reduced_hepatic_blood_flow", "reduced_renal_perfusion", "edema"]
    ),

    :diabetes => DiseaseOntologyMapping(
        "Diabetes mellitus",
        "DOID:9352",
        ["E10-E14"],
        ["D003920"],
        "SCTID:73211009",
        ["altered_CYP2E1", "protein_glycation", "nephropathy_risk"]
    ),

    :obesity => DiseaseOntologyMapping(
        "Obesity",
        "DOID:9970",
        ["E66"],
        ["D009765"],
        "SCTID:414916001",
        ["increased_Vd_lipophilic", "altered_CYP3A4", "increased_clearance"]
    ),

    :inflammatory_disease => DiseaseOntologyMapping(
        "Inflammatory disease",
        "DOID:7",
        ["R00-R99"],
        ["D007249"],
        "SCTID:128139000",
        ["reduced_CYP_activity", "altered_transporter_expression", "acute_phase_response"]
    ),

    :cancer => DiseaseOntologyMapping(
        "Neoplasm",
        "DOID:162",
        ["C00-D49"],
        ["D009369"],
        "SCTID:363346000",
        ["altered_organ_function", "cachexia", "polypharmacy_risk"]
    )
)

# =============================================================================
# CLINICAL DDI DATABASE
# =============================================================================

"""
Clinical DDI database with ontology annotations.
Data from: FDA DDI guidance, University of Washington DDI database, DrugBank
"""
struct ClinicalDDIEvidence
    perpetrator::Symbol
    victim::Symbol

    # Observed interaction
    auc_ratio_mean::Float64
    auc_ratio_range::Tuple{Float64, Float64}
    cmax_ratio::Float64

    # Study characteristics
    n_subjects::Int
    study_design::Symbol    # :crossover, :parallel, :retrospective
    population::Symbol      # :healthy, :patient, :special

    # Mechanism
    primary_mechanism::BayesianDDIModel.DDIMechanismType
    secondary_mechanisms::Vector{BayesianDDIModel.DDIMechanismType}

    # Ontology annotations
    dinto_interaction_id::String
    evidence_level::Symbol  # :strong, :moderate, :weak

    # References
    pubmed_ids::Vector{Int}
    fda_label_info::Bool
end

"""
Curated clinical DDI evidence database.
"""
const DDI_CLINICAL_DATABASE = Dict{Tuple{Symbol, Symbol}, ClinicalDDIEvidence}(
    # Ketoconazole + Midazolam (classic strong inhibition)
    (:ketoconazole, :midazolam) => ClinicalDDIEvidence(
        :ketoconazole, :midazolam,
        15.9, (10.0, 24.0), 3.5,
        12, :crossover, :healthy,
        BayesianDDIModel.CYP_COMPETITIVE_INHIBITION,
        BayesianDDIModel.DDIMechanismType[],
        "DINTO:DDI_0001",
        :strong,
        [7573094, 9169157],
        true
    ),

    # Itraconazole + Midazolam
    (:itraconazole, :midazolam) => ClinicalDDIEvidence(
        :itraconazole, :midazolam,
        10.8, (6.0, 18.0), 3.4,
        10, :crossover, :healthy,
        BayesianDDIModel.CYP_COMPETITIVE_INHIBITION,
        BayesianDDIModel.DDIMechanismType[],
        "DINTO:DDI_0002",
        :strong,
        [8841154],
        true
    ),

    # Rifampin + Midazolam (strong induction)
    (:rifampin, :midazolam) => ClinicalDDIEvidence(
        :rifampin, :midazolam,
        0.04, (0.02, 0.08), 0.1,
        14, :crossover, :healthy,
        BayesianDDIModel.CYP_INDUCTION,
        BayesianDDIModel.DDIMechanismType[],
        "DINTO:DDI_0003",
        :strong,
        [9169157, 10223772],
        true
    ),

    # Fluconazole + Warfarin
    (:fluconazole, :warfarin) => ClinicalDDIEvidence(
        :fluconazole, :warfarin,
        2.0, (1.5, 2.5), 1.3,
        8, :crossover, :healthy,
        BayesianDDIModel.CYP_COMPETITIVE_INHIBITION,
        BayesianDDIModel.DDIMechanismType[],
        "DINTO:DDI_0004",
        :strong,
        [2191589],
        true
    ),

    # Cyclosporine + Rosuvastatin (OATP inhibition)
    (:cyclosporine, :rosuvastatin) => ClinicalDDIEvidence(
        :cyclosporine, :rosuvastatin,
        7.1, (4.0, 11.0), 10.6,
        10, :crossover, :healthy,
        BayesianDDIModel.TRANSPORTER_INHIBITION,
        BayesianDDIModel.DDIMechanismType[],
        "DINTO:DDI_0005",
        :strong,
        [15100173],
        true
    ),

    # Ketoconazole + Tacrolimus
    (:ketoconazole, :tacrolimus) => ClinicalDDIEvidence(
        :ketoconazole, :tacrolimus,
        5.0, (3.0, 8.0), 2.5,
        8, :crossover, :patient,
        BayesianDDIModel.CYP_COMPETITIVE_INHIBITION,
        [BayesianDDIModel.TRANSPORTER_INHIBITION],
        "DINTO:DDI_0006",
        :strong,
        [8033517],
        true
    ),

    # Itraconazole + Simvastatin
    (:itraconazole, :simvastatin) => ClinicalDDIEvidence(
        :itraconazole, :simvastatin,
        19.0, (10.0, 30.0), 17.0,
        10, :crossover, :healthy,
        BayesianDDIModel.CYP_COMPETITIVE_INHIBITION,
        BayesianDDIModel.DDIMechanismType[],
        "DINTO:DDI_0007",
        :strong,
        [10223772],
        true
    ),

    # Rifampin + Tacrolimus
    (:rifampin, :tacrolimus) => ClinicalDDIEvidence(
        :rifampin, :tacrolimus,
        0.1, (0.05, 0.2), 0.15,
        6, :parallel, :patient,
        BayesianDDIModel.CYP_INDUCTION,
        [BayesianDDIModel.TRANSPORTER_INDUCTION],
        "DINTO:DDI_0008",
        :strong,
        [8841154],
        true
    ),

    # Fluconazole + Midazolam
    (:fluconazole, :midazolam) => ClinicalDDIEvidence(
        :fluconazole, :midazolam,
        3.6, (2.5, 5.0), 2.0,
        12, :crossover, :healthy,
        BayesianDDIModel.CYP_COMPETITIVE_INHIBITION,
        BayesianDDIModel.DDIMechanismType[],
        "DINTO:DDI_0009",
        :strong,
        [7752770],
        true
    ),

    # Cyclosporine + Simvastatin
    (:cyclosporine, :simvastatin) => ClinicalDDIEvidence(
        :cyclosporine, :simvastatin,
        8.0, (5.0, 12.0), 7.0,
        12, :crossover, :patient,
        BayesianDDIModel.TRANSPORTER_INHIBITION,
        [BayesianDDIModel.CYP_COMPETITIVE_INHIBITION],
        "DINTO:DDI_0010",
        :strong,
        [10223772],
        true
    )
)

# =============================================================================
# EVIDENCE TYPE MAPPINGS (DIDEO)
# =============================================================================

"""
Evidence type ontology mappings based on DIDEO.
"""
const EVIDENCE_TYPES = Dict{Symbol, EvidenceTypeOntology}(
    :in_vitro_enzyme => EvidenceTypeOntology(
        :in_vitro_enzyme,
        "DIDEO:0000001",
        "in vitro enzyme study",
        :weak,
        "microsomal incubation"
    ),

    :in_vitro_hepatocyte => EvidenceTypeOntology(
        :in_vitro_hepatocyte,
        "DIDEO:0000002",
        "in vitro hepatocyte study",
        :moderate,
        "cultured hepatocyte"
    ),

    :clinical_pk_healthy => EvidenceTypeOntology(
        :clinical_pk_healthy,
        "DIDEO:0000015",
        "clinical pharmacokinetic study in healthy volunteers",
        :strong,
        "prospective crossover"
    ),

    :clinical_pk_patient => EvidenceTypeOntology(
        :clinical_pk_patient,
        "DIDEO:0000016",
        "clinical pharmacokinetic study in patients",
        :strong,
        "prospective clinical"
    ),

    :population_pk => EvidenceTypeOntology(
        :population_pk,
        "DIDEO:0000020",
        "population pharmacokinetic analysis",
        :moderate,
        "retrospective modeling"
    ),

    :case_report => EvidenceTypeOntology(
        :case_report,
        "DIDEO:0000025",
        "clinical case report",
        :weak,
        "retrospective observation"
    ),

    :pbpk_modeling => EvidenceTypeOntology(
        :pbpk_modeling,
        "DIDEO:0000030",
        "physiologically-based pharmacokinetic modeling",
        :moderate,
        "in silico prediction"
    )
)

# =============================================================================
# API FUNCTIONS
# =============================================================================

"""
    get_dinto_term(mechanism::DDIMechanismType) -> DDIMechanismOntology

Get DINTO ontology term for a DDI mechanism.
"""
function get_dinto_term(mechanism::BayesianDDIModel.DDIMechanismType)::DDIMechanismOntology
    if haskey(DINTO_MECHANISMS, mechanism)
        return DINTO_MECHANISMS[mechanism]
    else
        error("No DINTO mapping for mechanism: $mechanism")
    end
end

"""
    get_chebi_id(drug::Symbol) -> String

Get ChEBI identifier for a drug.
"""
function get_chebi_id(drug::Symbol)::String
    if haskey(CHEBI_DRUGS, drug)
        return CHEBI_DRUGS[drug].chebi_id
    else
        error("No ChEBI mapping for drug: $drug")
    end
end

"""
    get_drugbank_id(drug::Symbol) -> String

Get DrugBank identifier for a drug.
"""
function get_drugbank_id(drug::Symbol)::String
    if haskey(CHEBI_DRUGS, drug)
        return CHEBI_DRUGS[drug].drugbank_id
    else
        error("No DrugBank mapping for drug: $drug")
    end
end

"""
    get_rxnorm_cui(drug::Symbol) -> String

Get RxNorm CUI for a drug.
"""
function get_rxnorm_cui(drug::Symbol)::String
    if haskey(CHEBI_DRUGS, drug)
        return CHEBI_DRUGS[drug].rxnorm_cui
    else
        error("No RxNorm mapping for drug: $drug")
    end
end

"""
    get_doid_term(disease::Symbol) -> DiseaseOntologyMapping

Get Disease Ontology mapping for a disease context.
"""
function get_doid_term(disease::Symbol)::DiseaseOntologyMapping
    if haskey(DISEASE_DDI_CONTEXT, disease)
        return DISEASE_DDI_CONTEXT[disease]
    else
        error("No DOID mapping for disease: $disease")
    end
end

"""
    map_cyp_to_ontology(enzyme::CYPEnzyme) -> CYPOntologyMapping

Map CYP enzyme to ontology identifiers.
"""
function map_cyp_to_ontology(enzyme::BayesianDDIModel.CYPEnzyme)::CYPOntologyMapping
    if haskey(CYP_ONTOLOGY, enzyme)
        return CYP_ONTOLOGY[enzyme]
    else
        error("No ontology mapping for CYP enzyme: $enzyme")
    end
end

"""
    map_transporter_to_ontology(transporter::TransporterType) -> TransporterOntologyMapping

Map transporter to ontology identifiers.
"""
function map_transporter_to_ontology(transporter::BayesianDDIModel.TransporterType)::TransporterOntologyMapping
    if haskey(TRANSPORTER_ONTOLOGY, transporter)
        return TRANSPORTER_ONTOLOGY[transporter]
    else
        error("No ontology mapping for transporter: $transporter")
    end
end

"""
    get_literature_ddi_evidence(perpetrator::Symbol, victim::Symbol) -> Union{ClinicalDDIEvidence, Nothing}

Get clinical DDI evidence from the curated database.
"""
function get_literature_ddi_evidence(
    perpetrator::Symbol,
    victim::Symbol
)::Union{ClinicalDDIEvidence, Nothing}
    key = (perpetrator, victim)
    if haskey(DDI_CLINICAL_DATABASE, key)
        return DDI_CLINICAL_DATABASE[key]
    else
        return nothing
    end
end

"""
    validate_ontology_mapping(drug::Symbol) -> NamedTuple

Validate that a drug has complete ontology mappings.
"""
function validate_ontology_mapping(drug::Symbol)::NamedTuple
    has_chebi = haskey(CHEBI_DRUGS, drug)
    mapping = has_chebi ? CHEBI_DRUGS[drug] : nothing

    return (
        drug = drug,
        has_chebi = has_chebi,
        has_drugbank = has_chebi && !isempty(mapping.drugbank_id),
        has_rxnorm = has_chebi && !isempty(mapping.rxnorm_cui),
        has_atc = has_chebi && !isempty(mapping.atc_codes),
        complete = has_chebi && !isempty(mapping.drugbank_id) && !isempty(mapping.rxnorm_cui)
    )
end

"""
    create_ontology_annotated_ddi(perpetrator, victim, prediction) -> Dict

Create ontology-annotated DDI result.
"""
function create_ontology_annotated_ddi(
    perpetrator::Symbol,
    victim::Symbol,
    auc_ratio::Float64,
    mechanism::BayesianDDIModel.DDIMechanismType
)::Dict{String, Any}
    result = Dict{String, Any}()

    # Drug annotations
    if haskey(CHEBI_DRUGS, perpetrator)
        perp_mapping = CHEBI_DRUGS[perpetrator]
        result["perpetrator"] = Dict(
            "name" => perp_mapping.drug_name,
            "chebi_id" => perp_mapping.chebi_id,
            "drugbank_id" => perp_mapping.drugbank_id,
            "rxnorm_cui" => perp_mapping.rxnorm_cui,
            "uri" => "http://purl.obolibrary.org/obo/$(replace(perp_mapping.chebi_id, ":" => "_"))"
        )
    end

    if haskey(CHEBI_DRUGS, victim)
        vic_mapping = CHEBI_DRUGS[victim]
        result["victim"] = Dict(
            "name" => vic_mapping.drug_name,
            "chebi_id" => vic_mapping.chebi_id,
            "drugbank_id" => vic_mapping.drugbank_id,
            "rxnorm_cui" => vic_mapping.rxnorm_cui,
            "uri" => "http://purl.obolibrary.org/obo/$(replace(vic_mapping.chebi_id, ":" => "_"))"
        )
    end

    # Mechanism annotation
    if haskey(DINTO_MECHANISMS, mechanism)
        mech_mapping = DINTO_MECHANISMS[mechanism]
        result["mechanism"] = Dict(
            "dinto_id" => mech_mapping.dinto_id,
            "label" => mech_mapping.dinto_label,
            "description" => mech_mapping.description,
            "ndfrt_id" => mech_mapping.ndfrt_interaction_type
        )
    end

    # Interaction data
    result["interaction"] = Dict(
        "auc_ratio" => auc_ratio,
        "fda_classification" => auc_ratio < 1.25 ? "no_effect" :
                               auc_ratio < 2.0 ? "weak" :
                               auc_ratio < 5.0 ? "moderate" : "strong"
    )

    # Literature evidence
    evidence = get_literature_ddi_evidence(perpetrator, victim)
    if evidence !== nothing
        result["evidence"] = Dict(
            "observed_auc_ratio" => evidence.auc_ratio_mean,
            "observed_range" => evidence.auc_ratio_range,
            "n_subjects" => evidence.n_subjects,
            "study_design" => String(evidence.study_design),
            "evidence_level" => String(evidence.evidence_level),
            "pubmed_ids" => evidence.pubmed_ids
        )
    end

    return result
end

"""
    export_to_rdf(ddi_result::Dict) -> String

Export DDI result to RDF/Turtle format for semantic web integration.
"""
function export_to_rdf(ddi_result::Dict{String, Any})::String
    buf = IOBuffer()

    # Prefixes
    println(buf, "@prefix dinto: <http://purl.obolibrary.org/obo/DINTO_> .")
    println(buf, "@prefix chebi: <http://purl.obolibrary.org/obo/CHEBI_> .")
    println(buf, "@prefix obo: <http://purl.obolibrary.org/obo/> .")
    println(buf, "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
    println(buf, "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .")
    println(buf, "@prefix darwin: <http://darwin-pbpk.org/ontology#> .")
    println(buf)

    # DDI instance
    if haskey(ddi_result, "perpetrator") && haskey(ddi_result, "victim")
        perp = ddi_result["perpetrator"]
        vic = ddi_result["victim"]

        println(buf, "darwin:DDI_$(perp["drugbank_id"])_$(vic["drugbank_id"]) a dinto:drug_drug_interaction ;")
        println(buf, "    rdfs:label \"$(perp["name"]) - $(vic["name"]) interaction\" ;")
        println(buf, "    darwin:hasPerpetrator chebi:$(split(perp["chebi_id"], ":")[2]) ;")
        println(buf, "    darwin:hasVictim chebi:$(split(vic["chebi_id"], ":")[2]) ;")

        if haskey(ddi_result, "mechanism")
            mech = ddi_result["mechanism"]
            println(buf, "    darwin:hasMechanism obo:$(replace(mech["dinto_id"], ":" => "_")) ;")
        end

        if haskey(ddi_result, "interaction")
            inter = ddi_result["interaction"]
            println(buf, "    darwin:aucRatio \"$(inter["auc_ratio"])\"^^xsd:float ;")
            println(buf, "    darwin:fdaClassification \"$(inter["fda_classification"])\" ;")
        end

        println(buf, "    .")
    end

    return String(take!(buf))
end

"""
    list_available_drugs() -> Vector{Symbol}

List all drugs with ontology mappings.
"""
function list_available_drugs()::Vector{Symbol}
    return collect(keys(CHEBI_DRUGS))
end

"""
    list_clinical_ddi_pairs() -> Vector{Tuple{Symbol, Symbol}}

List all DDI pairs with clinical evidence.
"""
function list_clinical_ddi_pairs()::Vector{Tuple{Symbol, Symbol}}
    return collect(keys(DDI_CLINICAL_DATABASE))
end

"""
    get_ddi_by_mechanism(mechanism::DDIMechanismType) -> Vector{ClinicalDDIEvidence}

Get all clinical DDIs with a specific mechanism.
"""
function get_ddi_by_mechanism(mechanism::BayesianDDIModel.DDIMechanismType)::Vector{ClinicalDDIEvidence}
    results = ClinicalDDIEvidence[]
    for evidence in values(DDI_CLINICAL_DATABASE)
        if evidence.primary_mechanism == mechanism || mechanism in evidence.secondary_mechanisms
            push!(results, evidence)
        end
    end
    return results
end

end # module DDIOntology
