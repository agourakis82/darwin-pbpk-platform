# ===========================================================================
# Darwin PBPK Platform - Production Docker Image
# ===========================================================================
# Multi-stage build for Julia + Python PBPK simulation platform
# Supports CPU and GPU (CUDA) inference
#
# Version: 2.11.0
# Author: Dr. Demetrios Agourakis
# ===========================================================================

# Stage 1: Julia base with dependencies
FROM julia:1.10.4-bookworm AS julia-base

LABEL maintainer="Dr. Demetrios Agourakis <agourakis82@darwin-pbpk.org>"
LABEL version="2.11.0"
LABEL description="Darwin PBPK Platform - AI-powered PBPK prediction"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Julia depot path
ENV JULIA_DEPOT_PATH=/opt/julia/depot
ENV JULIA_NUM_THREADS=auto

# Create project directory
WORKDIR /app

# Copy Julia project files first (for caching)
COPY julia-migration/Project.toml julia-migration/Manifest.toml* ./julia-migration/

# Instantiate Julia dependencies (cached layer)
RUN cd julia-migration && \
    julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Stage 2: Python environment
FROM julia-base AS python-env

# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy Python requirements
COPY requirements.txt* ./
COPY apps/api/requirements.txt* ./apps/api/

# Install Python dependencies (if requirements exist)
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi && \
    if [ -f apps/api/requirements.txt ]; then pip install --no-cache-dir -r apps/api/requirements.txt; fi

# Stage 3: Final production image
FROM python-env AS production

# Copy entire project
COPY . /app

# Set up Julia project
WORKDIR /app/julia-migration

# Precompile main module
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); using DarwinPBPK'

# Create non-root user
RUN useradd -m -s /bin/bash darwin && \
    chown -R darwin:darwin /app /opt/julia

USER darwin

# Environment variables
ENV DARWIN_PBPK_VERSION=2.11.0
ENV JULIA_PROJECT=/app/julia-migration

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD julia --project=/app/julia-migration -e 'using DarwinPBPK; println("OK")' || exit 1

# Default command: Julia REPL with project
CMD ["julia", "--project=/app/julia-migration"]

# ===========================================================================
# CUDA-enabled variant (use --target=cuda)
# ===========================================================================
FROM production AS cuda

USER root

# Install CUDA toolkit (runtime only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-12-2 \
    libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

# Add CUDA to Julia
RUN julia --project=/app/julia-migration -e 'using Pkg; Pkg.add("CUDA"); Pkg.add("cuDNN")'

USER darwin

ENV JULIA_CUDA_MEMORY_POOL=none
