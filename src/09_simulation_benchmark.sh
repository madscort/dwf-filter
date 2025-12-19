#!/bin/bash

# mads - 2025-11-01

set -euo pipefail

echo "Starting dwf-sim benchmarking batch at: $(date)"

# -------- SETTINGS --------
REFERENCE="data/gatk_bundle_hg38/Homo_sapiens_assembly38.fasta"
TARGET_BED="data/targets/TruSight_Hereditary_Cancer_TargetedRegions_v2_hg38.bed"
SDF_DIR="data/gatk_bundle_hg38/Homo_sapiens_assembly38.sdf"

MATRIX_SIZES=("2" "5" "10" "24")
COVERAGES=("5" "10" "30" "100")
CALLER_NAME="GATK"

# -------- MAIN LOOP --------
for MATRIX in "${MATRIX_SIZES[@]}"; do
    for COV in "${COVERAGES[@]}"; do
        
        # Paths
        SIM_BASE="./1kg_runs/${MATRIX}x${MATRIX}_${COV}x"
        RESULTS_BASE="./dobseq_sim_results/${MATRIX}x${MATRIX}_${COV}x"
        BENCH_BASE="./1kg_benchmark_results/${MATRIX}x${MATRIX}_${COV}x"
        TRUTH_VCFS="./1kg_samples/${MATRIX}x${MATRIX}/vcfs/ngsngs"

        echo ""
        echo "--- POOLS ---"
        
        pixi run dwf-sim benchmark \
            --matrix-context "${SIM_BASE}/matrix_context.json" \
            --gold-standard-dir "${SIM_BASE}/pools" \
            --caller-name "${CALLER_NAME}" \
            --caller-dir "${RESULTS_BASE}/unfiltered/variants" \
            --output-dir "${BENCH_BASE}/unfiltered/pools" \
            --filtration-status unfiltered \
            --coverage "${COV}" \
            --matrix-size "${MATRIX}" \
            --reference "${REFERENCE}" \
            --bed "${TARGET_BED}" \
            --sdf-dir "${SDF_DIR}" \
            --caller-suffix "vcf.gz" \
            --verbose
        
        echo "Finished pool benchmarking"
        
        echo ""
        echo "--- PINs ---"
        
        pixi run dwf-sim benchmark-pins \
            --matrix-context "${SIM_BASE}/matrix_context.json" \
            --truth-dir "${TRUTH_VCFS}" \
            --caller-dir "${RESULTS_BASE}/unfiltered/pinpoint_variants/unique_pins" \
            --output-dir "${BENCH_BASE}/unfiltered/pins" \
            --filtration-status unfiltered \
            --coverage "${COV}" \
            --matrix-size "${MATRIX}" \
            --reference "${REFERENCE}" \
            --bed "${TARGET_BED}" \
            --sdf-dir "${SDF_DIR}" \
            --caller-suffix "_unique_pins.vcf.gz" \
            --verbose \
            --threads 32

        echo ""
        echo "Completed benchmarking for ${MATRIX}x${MATRIX} @ ${COV}x"
        echo ""
    done
done

echo "============================================================"
echo "All benchmarking complete at: $(date)"
echo "============================================================"
