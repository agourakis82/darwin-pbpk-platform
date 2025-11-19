# Changelog

All notable changes to Darwin PBPK Platform will be documented in this file.

## [1.0.1] - 2025-11-08

### Added
- Validated PBPK dataset publication with complete metadata
- Experimental validation protocols documentation
- Quality assurance documentation for dataset
- Dataset README with usage instructions
- Enhanced CITATION.cff with dataset publication info
- Zenodo release update (concept DOI: 10.5281/zenodo.17536674, version DOI: 10.5281/zenodo.17561017)

### Changed
- Updated version to reflect dataset publication milestone

### Fixed
- Dataset validation improvements
- Metadata completeness checks

## [1.0.0] - 2025-11-05

### Added
- Initial production release
- Multi-modal molecular representations (ChemBERTa + GNN + RDKit)
- PBPK parameter prediction (Fu, Vd, CL)
- Training on 44,779 compounds from ChEMBL and TDC
- GNN architectures (GAT + TransformerConv)
- Multi-task learning with physics-informed constraints
- Production deployment on Kubernetes
