# Darwin PBPK Platform - Docker Deployment

This directory contains Docker configuration for reproducible deployment of the Darwin PBPK Platform.

## Quick Start

```bash
# Build the image
docker build -t darwin-pbpk:2.11.0 -f docker/Dockerfile .

# Run Julia REPL
docker run -it darwin-pbpk:2.11.0

# Run with mounted data
docker run -it -v $(pwd)/data:/app/data darwin-pbpk:2.11.0
```

## Docker Compose

```bash
cd docker

# Start all services
docker-compose up -d

# Julia REPL only
docker-compose up julia

# API server
docker-compose up api
# Access: http://localhost:8000/api/v1/docs

# GPU-enabled (requires NVIDIA Docker)
docker-compose --profile gpu up julia-gpu
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| `julia` | - | Interactive Julia REPL with DarwinPBPK |
| `api` | 8000 | FastAPI REST server |
| `worker` | - | Batch simulation workers (2 replicas) |
| `julia-gpu` | - | GPU-accelerated Julia (profile: gpu) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JULIA_NUM_THREADS` | auto | Number of Julia threads |
| `DARWIN_PBPK_VERSION` | 2.11.0 | Platform version |
| `JULIA_CUDA_MEMORY_POOL` | none | CUDA memory management |

## Volumes

- `data/` - Input datasets (read-only)
- `models/` - Trained model checkpoints (read-only)
- `simulation-results` - Output directory (persistent volume)

## Resource Requirements

### CPU Mode
- Memory: 4-8 GB
- CPUs: 2-4 cores
- Disk: 10 GB (image + deps)

### GPU Mode
- NVIDIA GPU with CUDA 12.2+
- NVIDIA Container Toolkit
- Memory: 8+ GB GPU RAM

## Building Custom Images

```bash
# CPU-only (default)
docker build -t darwin-pbpk:2.11.0 --target production -f docker/Dockerfile .

# With CUDA support
docker build -t darwin-pbpk:2.11.0-cuda --target cuda -f docker/Dockerfile .
```

## Example Usage

### Run PBPK Simulation

```bash
docker run -it darwin-pbpk:2.11.0 julia --project=/app/julia-migration -e '
using DarwinPBPK
params = default_human_params()
result = simulate(params, 100.0; t_max=24.0)
println("Cmax: ", maximum(result["plasma"]))
'
```

### Batch Processing

```bash
docker run -v $(pwd)/input:/input -v $(pwd)/output:/output darwin-pbpk:2.11.0 \
  julia --project=/app/julia-migration /app/scripts/batch_simulate.jl \
  --input /input/compounds.csv \
  --output /output/results.csv
```

## Troubleshooting

### Out of Memory
Increase Docker memory limit or reduce `JULIA_NUM_THREADS`.

### GPU Not Detected
Ensure NVIDIA Container Toolkit is installed:
```bash
nvidia-smi  # Should show GPU
docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Slow First Run
Julia precompilation occurs on first use. Subsequent runs are faster.
