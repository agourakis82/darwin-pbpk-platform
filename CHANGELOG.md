# Changelog

All notable changes to Darwin PBPK Platform will be documented in this file.

## [2.0.0-julia] - 2025-11-18

### 🚨 BREAKING CHANGES
- **Migração completa para Julia**: Repositório agora é 100% Julia, sem código Python
- **Requisitos**: Julia 1.9+ necessário (Python não mais suportado)
- **96 arquivos Python removidos**: Todo código Python foi migrado ou removido

### Added
- ODE Solver em Julia (4× mais rápido que Python)
- Dataset Generation em Julia
- Dynamic GNN em Julia
- Training Pipeline em Julia
- Validation em Julia (GMFE 1.036, 100% within folds)
- REST API em Julia (HTTP.jl)
- Scripts Julia para treinamento e validação
- Documentação completa da migração
- Scripts de migração e remoção de Python

### Changed
- Performance: ODE Solver 4.5ms (4× vs Python)
- Validação científica: GMFE 1.036, 100% within 1.25x, 1.5x, 2.0x
- Testes: 6/6 passando

### Removed
- 96 arquivos Python (apps/, scripts/*.py, tests/*.py)
- requirements.txt
- Dependências Python (PyTorch, NumPy, SciPy, etc.)

### Documentation
- `docs/MIGRATION_TO_JULIA_COMPLETE.md` - Guia completo de migração
- `docs/PYTHON_REMOVAL_PLAN.md` - Plano de remoção
- `README_JULIA_ONLY.md` - README para versão Julia
- `RELEASE_v2.0.0-julia.md` - Release notes detalhadas

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
