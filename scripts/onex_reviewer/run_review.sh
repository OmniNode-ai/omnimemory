#!/usr/bin/env bash
# Convenience script to run ONEX reviews

set -euo pipefail

MODE=${1:-help}

function show_help() {
    echo "ONEX Reviewer - Baseline & Nightly Review System"
    echo ""
    echo "Usage: $0 <mode>"
    echo ""
    echo "Modes:"
    echo "  baseline   - Run baseline review of entire codebase"
    echo "  nightly    - Run incremental nightly review"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 baseline    # Review entire codebase against ONEX standards"
    echo "  $0 nightly     # Review changes since last successful run"
}

function run_baseline() {
    echo "Running baseline producer..."
    ./scripts/onex_reviewer/baseline_producer.sh

    # Get latest output directory
    LATEST_DIR=$(find .onex_baseline -type d -name "20*" | sort | tail -1)

    if [[ -z "$LATEST_DIR" ]]; then
        echo "Error: No baseline output found"
        exit 1
    fi

    echo "Running baseline review on $LATEST_DIR..."
    python3 -m src.onex_reviewer.run_baseline "$LATEST_DIR"
}

function run_nightly() {
    echo "Running nightly producer..."
    ./scripts/onex_reviewer/nightly_producer.sh

    # Get latest output directory
    LATEST_DIR=$(find .onex_nightly -type d -name "20*" | sort | tail -1)

    if [[ -z "$LATEST_DIR" ]]; then
        echo "Error: No nightly output found"
        exit 1
    fi

    echo "Running nightly review on $LATEST_DIR..."
    python3 -m src.onex_reviewer.run_nightly "$LATEST_DIR"
}

case "$MODE" in
    baseline)
        run_baseline
        ;;
    nightly)
        run_nightly
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        echo ""
        show_help
        exit 1
        ;;
esac