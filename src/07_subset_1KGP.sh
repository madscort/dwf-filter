#!/bin/bash
set -euo pipefail

# mads - 2025-10-10

# Script for subsetting to target regions, concatenating and sorting 1000KGP VCF files.

# CONFIG
INDIR="1kg"
TARGET_REGIONS="data/TruSight_Hereditary_Cancer_TargetedRegions_v2_hg38_ext151.bed"
OUTDIR="data/subset_1kGP"
REFERENCE="data/gatk-resource-bundle/hg38/v0/Homo_sapiens_assembly38.fasta"
THREADS=1

CHRS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X)

mkdir -p "${OUTDIR}"

# STEP 1: Subset each chromosome
for chr in "${CHRS[@]}"; do
  if [ "$chr" = "X" ]; then
    VCF="${INDIR}/1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.v2.vcf.gz"
  else
    VCF="${INDIR}/1kGP_high_coverage_Illumina.chr${chr}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
  fi

  if [ ! -f "${VCF}" ]; then
    echo "ERROR: file ${VCF} not found, skipping chr${chr}"
    exit 1
  fi

  echo "Subsetting chr${chr} …"
  bcftools view -R "${TARGET_REGIONS}" "${VCF}" -Oz -o "${OUTDIR}/subset.chr${chr}.vcf.gz" --threads ${THREADS}
  bcftools index "${OUTDIR}/subset.chr${chr}.vcf.gz"
done

# STEP 2: Concatenate all chromosomes into one
echo "Concatenating all chromosomes …"
ls ${OUTDIR}/subset.chr{1..22}.vcf.gz ${OUTDIR}/subset.chrX.vcf.gz > ${OUTDIR}/vcf_list.txt

bcftools concat --threads ${THREADS} --file-list ${OUTDIR}/vcf_list.txt -Oz -o ${OUTDIR}/all.subset.vcf.gz
bcftools index ${OUTDIR}/all.subset.vcf.gz

# STEP 3: Remove SVs (structural variants)
bcftools view \
  -Oz \
  -o ${OUTDIR}/all.subset.noSV.vcf.gz \
  ${OUTDIR}/all.subset.vcf.gz \
  --threads ${THREADS} \
  -i 'INFO/SVTYPE=""'

# Coordinate sort
bcftools sort \
  -Oz \
  -o ${OUTDIR}/all.subset.noSV.sorted.vcf.gz \
  ${OUTDIR}/all.subset.noSV.vcf.gz

# Index final file
bcftools index ${OUTDIR}/all.subset.noSV.sorted.vcf.gz
