#!/bin/bash
# =============================================================================
# PBPK Benchmark Comparison: Demetrios vs Julia
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================================================="
echo "PBPK Benchmark Comparison: Demetrios vs Julia"
echo "=============================================================================="
echo ""
echo "Project Root: $PROJECT_ROOT"
echo "Date: $(date)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Check prerequisites
# -----------------------------------------------------------------------------

echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Demetrios compiler
DEMETRIOS_COMPILER="$PROJECT_ROOT/Darwin-demetrios/compiler/target/release/dc"
if [ ! -f "$DEMETRIOS_COMPILER" ]; then
    echo -e "${YELLOW}Building Demetrios compiler...${NC}"
    cd "$PROJECT_ROOT/Darwin-demetrios/compiler"
    cargo build --release --features jit
    cd "$PROJECT_ROOT"
fi

# Check Julia
if ! command -v julia &> /dev/null; then
    echo -e "${RED}Julia not found. Please install Julia.${NC}"
    exit 1
fi

echo -e "${GREEN}Prerequisites OK${NC}"
echo ""

# -----------------------------------------------------------------------------
# Run Demetrios Benchmarks
# -----------------------------------------------------------------------------

echo "=============================================================================="
echo -e "${BLUE}Running Demetrios Benchmarks${NC}"
echo "=============================================================================="
echo ""

DEMETRIOS_BENCH="$PROJECT_ROOT/Darwin-demetrios/examples/pbpk/pbpk_benchmarks.d"

# Type check first
echo "Type checking Demetrios module..."
$DEMETRIOS_COMPILER check "$DEMETRIOS_BENCH" 2>&1 | grep -v "^Warning:" || true
echo ""

# Run with JIT if available
if $DEMETRIOS_COMPILER --help 2>&1 | grep -q "run"; then
    echo "Running Demetrios benchmarks with JIT..."

    # Time the execution
    DEMETRIOS_START=$(date +%s%N)
    $DEMETRIOS_COMPILER run "$DEMETRIOS_BENCH" --features jit 2>&1 || echo "JIT execution not available yet"
    DEMETRIOS_END=$(date +%s%N)

    DEMETRIOS_TIME=$(( (DEMETRIOS_END - DEMETRIOS_START) / 1000000 ))
    echo ""
    echo "Demetrios total execution time: ${DEMETRIOS_TIME}ms"
else
    echo "Demetrios JIT not available - showing type-check only metrics"
    echo "(Full runtime benchmarks require JIT feature)"
fi

echo ""

# -----------------------------------------------------------------------------
# Run Julia Benchmarks
# -----------------------------------------------------------------------------

echo "=============================================================================="
echo -e "${BLUE}Running Julia Benchmarks${NC}"
echo "=============================================================================="
echo ""

JULIA_BENCH="$PROJECT_ROOT/julia-migration/benchmarks/pbpk_benchmarks.jl"

cd "$PROJECT_ROOT/julia-migration"

# Install BenchmarkTools if needed
julia --project=. -e 'using Pkg; "BenchmarkTools" in keys(Pkg.project().dependencies) || Pkg.add("BenchmarkTools")' 2>/dev/null

# Run benchmarks
julia --project=. "$JULIA_BENCH"

echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo "=============================================================================="
echo -e "${GREEN}Benchmark Comparison Complete${NC}"
echo "=============================================================================="
echo ""
echo "Files:"
echo "  Demetrios: $DEMETRIOS_BENCH"
echo "  Julia:     $JULIA_BENCH"
echo ""
echo "Notes:"
echo "  - Demetrios benchmarks require JIT feature for runtime execution"
echo "  - Julia uses BenchmarkTools for accurate measurements"
echo "  - Both implementations use equivalent algorithms (RK4, same metrics)"
echo ""
