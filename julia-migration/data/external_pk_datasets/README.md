# External PK Datasets for Validation

This directory contains publicly available pharmacokinetic datasets used for external validation of the Darwin PBPK Platform.

## Available Datasets

### 1. OSP DDI Database (`OSP_DDI.csv`)
**Source**: [Open-Systems-Pharmacology/Database-for-observed-data](https://github.com/Open-Systems-Pharmacology/Database-for-observed-data)

- **Records**: 634 drug-drug interactions
- **Data**: AUC ratios, Cmax ratios, dose information
- **Key Drugs**: Midazolam, Alfentanil, Triazolam, Rifampicin, Itraconazole, Clarithromycin
- **Use Case**: DDI model qualification, CYP3A4 inhibition/induction validation
- **License**: Open Source (OSP Foundation)

### 2. OSP Pediatrics Database (`OSP_Pediatrics.csv`)
**Source**: [Open-Systems-Pharmacology/Database-for-observed-data](https://github.com/Open-Systems-Pharmacology/Database-for-observed-data)

- **Records**: 277 pediatric PK observations
- **Data**: Clearance (CL), AUC, Cmax values
- **Key Drugs**: Sufentanil, Fentanyl, Alfentanil, Morphine
- **Use Case**: Pediatric scaling validation, ontogeny models
- **License**: Open Source (OSP Foundation)

### 3. Zenodo Beta-Lactam ICU Dataset
**Source**: [Zenodo Record 8241522](https://zenodo.org/records/8241522)
**DOI**: 10.5281/zenodo.8241522

**Files**:
- `Zenodo_BetaLactam_CriticallyIll_covariates.csv` - 151 studies
- `Zenodo_BetaLactam_CriticallyIll_outcomes.csv` - 1083 outcome records

**Data**: Population PK parameters, covariates, PTA targets
**Key Drugs**: Piperacillin, Meropenem, Cefepime, Doripenem, Ceftazidime
**Use Case**: Critically ill patient PK, renal function effects
**License**: CC-BY 4.0

### 4. PK-DB (Pharmacokinetics Database) - API Access
**Source**: [pk-db.com](https://pk-db.com)
**API**: `https://pk-db.com/api/v1/`

- **Studies**: 796+ curated clinical studies
- **Individuals**: 6,308+
- **Outputs**: 73,017+ PK parameters
- **Time-courses**: 3,148 concentration-time profiles
- **Key Drugs**: Caffeine, Morphine, Codeine, Midazolam, Acetaminophen, Simvastatin
- **Use Case**: Meta-analysis, concentration-time validation, stratified modeling
- **License**: MIT (code) / CC-BY-SA 4.0 (data)
- **Citation**: König M et al. Nucleic Acids Res. 2021;49(D1):D1358-D1363. PMID:33151297

## Download Scripts

```bash
# OSP Database
curl -sL "https://raw.githubusercontent.com/Open-Systems-Pharmacology/Database-for-observed-data/master/DDI.csv" -o OSP_DDI.csv
curl -sL "https://raw.githubusercontent.com/Open-Systems-Pharmacology/Database-for-observed-data/master/Pediatrics.csv" -o OSP_Pediatrics.csv

# Zenodo Beta-Lactam
curl -sL "https://zenodo.org/records/8241522/files/cis.csv" -o Zenodo_BetaLactam_CriticallyIll_covariates.csv
curl -sL "https://zenodo.org/records/8241522/files/outcomes.csv" -o Zenodo_BetaLactam_CriticallyIll_outcomes.csv
```

## Julia Usage

```julia
using DarwinPBPK

# List available datasets
list_available_datasets()

# Load OSP DDI data
ddi = load_osp_ddi()
midazolam_ddi = load_osp_ddi(filter_drug="Midazolam")

# Get DDI AUC ratios for a specific interaction
ratios = get_ddi_auc_ratios("Midazolam", "Itraconazole")
println("Mean AUC ratio: $(ratios.mean) (n=$(ratios.n))")

# Load pediatric data
ped = load_osp_pediatrics()
morphine_cl = get_pediatric_clearance("Morphine")

# Load beta-lactam ICU data
bl = load_zenodo_betalactam()
println("Studies: $(nrow(bl.covariates))")

# Query PK-DB API
studies = list_pkdb_studies(page=1)
for s in studies[1:5]
    println("$(s.name): $(s.n_individuals) individuals")
end

# Get detailed study
detail = get_pkdb_study_detail("PKDB00111")
```

## Data Quality Notes

1. **OSP Data**: Manually curated from peer-reviewed publications by pharmaceutical scientists
2. **Zenodo Data**: Systematic review data, may include heterogeneous study designs
3. **PK-DB**: Digitized from figures + extracted from tables, includes measurement errors

## Citation Requirements

When using these datasets, please cite:

1. **OSP**: "Open Systems Pharmacology Suite. Database for observed data. GitHub, 2024."
2. **Zenodo Beta-Lactam**: Abdul-Aziz MH et al. DOI:10.5281/zenodo.8241522
3. **PK-DB**: König M et al. Nucleic Acids Res. 2021;49(D1):D1358-D1363

## Related Resources

- [DDMoRe Model Repository](http://repository.ddmore.eu) - NONMEM/PBPK models
- [PKdata R Package](https://github.com/billdenney/PKdata) - Real and simulated PK data
- [PK-Sim](https://github.com/Open-Systems-Pharmacology/PK-Sim) - PBPK modeling software
- [eICU Database](https://eicu-crd.mit.edu/) - ICU patient data (requires application)
