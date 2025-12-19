#!/bin/bash
set -euo pipefail

# mads - 2025-11-01

echo "Starting DWF-SIM simulation batch at: $(date)"
echo "Running on host: $(hostname)"

# -------- SETTINGS --------
REFERENCE="data/gatk_bundle_hg38/Homo_sapiens_assembly38.fasta"
TARGET_BED="data/targets/TruSight_Hereditary_Cancer_TargetedRegions_v2_hg38_ext151.bed"
READ_LENGTH=150

MATRIX_SIZES=("2" "5" "10" "24")
COVERAGES=("5" "10" "30" "100")

# -------- MAIN LOOP --------
for MATRIX in "${MATRIX_SIZES[@]}"; do
    for COV in "${COVERAGES[@]}"; do
        
        OUTDIR="./1kg_runs/${MATRIX}x${MATRIX}_${COV}x"
        VCF_DIR="1kg_samples/${MATRIX}x${MATRIX}/vcfs/ngsngs"

        echo "------------------------------------------------------------"
        echo "Running DWF-SIM:"
        echo "  Matrix size: ${MATRIX}x${MATRIX}"
        echo "  Coverage: ${COV}x"
        echo "  Total reads: $(( COV * 10000 ))"
        echo "  Output: ${OUTDIR}"
        echo "  Time: $(date)"
        echo "------------------------------------------------------------"

        mkdir -p "${OUTDIR}"

        dwf-sim simulate \
            --matrix-size "$MATRIX" \
            --vcf-dir "$VCF_DIR" \
            --read-length "$READ_LENGTH" \
            --reference "$REFERENCE" \
            --targeted-bed "$TARGET_BED" \
            --output-dir "$OUTDIR" \
            --coverage "$(( COV * 10000 ))" \
            --threads 10

        echo "Finished simulation for ${MATRIX}x${MATRIX} at ${COV}x"
        echo
    done
done

echo "All simulations complete at: $(date)"
