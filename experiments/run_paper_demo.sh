#!/bin/bash
# run_paper_demo.sh — End-to-end stress-resilient ecosystem design demo
#
# Builds the paper_demo binary, runs evolution, and generates dashboard figures.
#
# Usage:
#   ./experiments/run_paper_demo.sh [--full] [--population N] [--generations N]

set -euo pipefail
cd "$(dirname "$0")/.."

# Defaults
POP=${POP:-16}
GEN=${GEN:-10}
FRAMES=${FRAMES:-200}
MODE="--lite"
EXTRA_ARGS=""

# Parse args
for arg in "$@"; do
    case "$arg" in
        --full) MODE="--full" ;;
        --population) shift; POP="$1" ;;
        --generations) shift; GEN="$1" ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

TELEMETRY="experiments/paper_telemetry.json"
PARETO="experiments/paper_pareto.json"
FIGURES="experiments/figures"

mkdir -p experiments/figures

echo "=== Building paper_demo binary ==="
cargo build --profile fast --no-default-features --bin paper_demo 2>&1 | tail -5

echo ""
echo "=== Running Paper Demo ==="
./target/fast/paper_demo \
    --population "$POP" \
    --generations "$GEN" \
    --frames "$FRAMES" \
    $MODE \
    --telemetry "$TELEMETRY" \
    --pareto "$PARETO" \
    $EXTRA_ARGS

echo ""
echo "=== Generating Dashboard Figures ==="
if command -v python3 &>/dev/null; then
    python3 experiments/evolve_dashboard.py \
        --telemetry "$TELEMETRY" \
        --pareto "$PARETO" \
        --output "$FIGURES"
    echo "Figures saved to: $FIGURES/"
else
    echo "python3 not found — skipping dashboard generation"
    echo "Run manually: python3 experiments/evolve_dashboard.py --telemetry $TELEMETRY --pareto $PARETO --output $FIGURES"
fi

echo ""
echo "=== Done ==="
echo "Telemetry: $TELEMETRY"
echo "Pareto:    $PARETO"
echo "Figures:   $FIGURES/"
