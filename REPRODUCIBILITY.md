# Darwin PBPK Platform - Reproducibility Guide

**Version**: 2.11.0  
**Date**: December 2025  
**DOI**: [To be assigned upon publication]

This document provides instructions for reproducing all results in the Darwin PBPK Platform, following FAIR principles and Q1 journal standards.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Model Checkpoints](#model-checkpoints)
4. [Dataset Access](#dataset-access)
5. [Reproducing Results](#reproducing-results)
6. [Docker Reproducibility](#docker-reproducibility)
7. [Verification](#verification)

---

## System Requirements

### Hardware (Minimum)
- CPU: 4 cores, x86_64 or ARM64
- RAM: 8 GB
- Storage: 20 GB free

### Hardware (Recommended for Training)
- GPU: NVIDIA RTX 3080+ (12+ GB VRAM)
- RAM: 32 GB
- Storage: 100 GB SSD

### Software
| Component | Version | Notes |
|-----------|---------|-------|
| Julia | 1.10.4+ | Primary runtime |
| Python | 3.10+ | API and utilities |
| CUDA | 12.2+ | Optional, for GPU |
| Docker | 24.0+ | For containerized runs |

---

## Installation

### Option 1: Native Installation

```bash
# Clone repository
git clone --recursive https://github.com/agourakis82/darwin-pbpk-platform.git
cd darwin-pbpk-platform

# Install Julia dependencies
cd julia-migration
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Verify installation
julia --project=. -e 'using DarwinPBPK; println("v", DarwinPBPK.version())'
```

### Option 2: Docker (Recommended)

```bash
# Build image
docker build -t darwin-pbpk:2.11.0 -f docker/Dockerfile .

# Verify
docker run darwin-pbpk:2.11.0 julia --project=/app/julia-migration -e 'using DarwinPBPK'
```

---

## Model Checkpoints

Pre-trained model checkpoints are available in `models/` directory:

| Checkpoint | Description | Size | SHA256 |
|------------|-------------|------|--------|
| `dynamic_gnn_v4_compound/best_model.pt` | Best GNN model (compound split) | ~50 MB | `[hash]` |
| `dynamic_gnn_sweep_c/best_model.pt` | Hyperparameter sweep best | ~50 MB | `[hash]` |
| `dynamic_gnn_sota_rtx4000/best_model.pt` | SOTA model (RTX 4000) | ~50 MB | `[hash]` |

### Downloading Checkpoints

If checkpoints are not included in the repository:

```bash
# Download from release
curl -L https://github.com/agourakis82/darwin-pbpk-platform/releases/download/v2.11.0/models.tar.gz | tar xz
```

### Verifying Checkpoints

```bash
# Verify SHA256
sha256sum models/dynamic_gnn_v4_compound/best_model.pt
```

---

## Dataset Access

### Training Data

The PBPK training dataset is generated from mechanistic simulations:

```bash
# Generate dataset (requires ~2 hours)
cd julia-migration
julia --project=. scripts/generate_dataset.jl --n-compounds 10000 --output ../data/pbpk_dataset.csv
```

### Validation Data

Clinical validation data from literature is in `data/validation/`:

| File | Description | Source |
|------|-------------|--------|
| `clinical_pk_literature.csv` | Published PK parameters | Various Q1 journals |
| `theophylline_pk.csv` | Theophylline validation set | [PMID: xxxxxx] |
| `midazolam_pk.csv` | Midazolam validation set | [PMID: xxxxxx] |

---

## Reproducing Results

### 1. ODE Solver Benchmarks

```julia
using DarwinPBPK
using BenchmarkTools

# Benchmark ODE solver
params = default_human_params()
@benchmark simulate($params, 100.0; t_max=24.0)
# Expected: ~0.04-0.36 ms
```

### 2. GNN Model Training

```bash
cd analysis/training
python train_dynamic_gnn.py \
    --config configs/sweep_c.yaml \
    --data ../../data/pbpk_dataset.csv \
    --output ../../models/reproduced_model \
    --seed 42
```

### 3. Validation Metrics

```julia
using DarwinPBPK

# Load model and run validation
results = external_validation_pipeline(
    "models/dynamic_gnn_v4_compound/best_model.pt",
    "data/validation/clinical_pk_literature.csv"
)

# Expected metrics:
# GMFE: 1.3-1.5
# AAFE: 1.4-1.6
# Within 2-fold: >70%
```

### 4. MedLang Compilation

```julia
using DarwinPBPK.MedLang

# Compile example model
source = read("examples/one_comp_oral.medlang", String)
params = compile_model(source)
result = simulate_medlang(source, 100.0)
```

---

## Docker Reproducibility

For exact reproducibility, use the Docker image:

```bash
# Pull exact version
docker pull ghcr.io/agourakis82/darwin-pbpk:2.11.0

# Run validation suite
docker run -v $(pwd)/results:/results darwin-pbpk:2.11.0 \
    julia --project=/app/julia-migration /app/scripts/run_validation.jl \
    --output /results/validation_report.json

# Compare results
diff results/validation_report.json expected/validation_report.json
```

### Singularity (HPC)

```bash
# Convert Docker to Singularity
singularity pull darwin-pbpk.sif docker://ghcr.io/agourakis82/darwin-pbpk:2.11.0

# Run on HPC
singularity exec darwin-pbpk.sif julia --project=/app/julia-migration -e 'using DarwinPBPK'
```

---

## Verification

### Test Suite

```bash
# Run all tests
cd julia-migration
julia --project=. -e 'using Pkg; Pkg.test()'

# Expected: All tests pass
```

### Numerical Verification

Key numerical results for verification:

| Metric | Expected Value | Tolerance |
|--------|---------------|-----------|
| Theophylline Cmax prediction | 8.2 mg/L | ±10% |
| Midazolam AUC prediction | 45 mg·h/L | ±15% |
| ODE solver RK45 vs Tsit5 | <0.1% diff | - |
| GMFE on test set | 1.35 | ±0.1 |

### Continuous Integration

All results are verified on each commit via GitHub Actions:

```yaml
# .github/workflows/reproducibility.yml
- name: Verify results
  run: julia --project=julia-migration scripts/verify_reproducibility.jl
```

---

## Citation

If you use this platform, please cite:

```bibtex
@software{darwin_pbpk_2025,
  author = {Agourakis, Demetrios},
  title = {Darwin PBPK Platform: AI-Powered Pharmacokinetic Prediction},
  version = {2.11.0},
  year = {2025},
  url = {https://github.com/agourakis82/darwin-pbpk-platform},
  doi = {10.5281/zenodo.xxxxxxx}
}
```

---

## Support

- Issues: https://github.com/agourakis82/darwin-pbpk-platform/issues
- Documentation: https://darwin-pbpk.readthedocs.io
- Email: agourakis82@darwin-pbpk.org
