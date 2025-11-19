#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Script de apoio para instalar extensões CUDA do PyG (torch-scatter, torch-sparse)
# compatíveis com a versão do PyTorch utilizada nos treinos PBPK.
#
# Uso:
#     ./scripts/install_pyg_cuda_extensions.sh
#
# Notas:
# - Requer ambiente com NVCC/toolchain compatível com a versão CUDA relatada por `torch.version.cuda`.
# - Ajuste as URLs de wheels caso utilize versões diferentes de PyTorch/CUDA.
# -----------------------------------------------------------------------------

set -euo pipefail


