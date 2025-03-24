#!/usr/bin/env python3
"""
VCF ML Annotation Pipeline
--------------------------
This script reads multiple VCF files, extracts necessary features, applies normalization,
runs predictions through separate ML models for SNVs and indels, and annotates the VCFs with
prediction probabilities and filter status.
"""

import argparse
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import pysam
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

INCLUDE_ANNOTATIONS = [
    "ASSEMBLED_HAPS","AS_BaseQRankSum","AS_FS","AS_MQ","AS_MQRankSum",
    "AS_QD","AS_ReadPosRankSum","AS_SOR","ClippingRankSum","HAPCOMP","HAPDOM",
    "HEC","LikelihoodRankSum","MLEAC","MLEAF","X_GCC","X_HIL"
]

IMPUTE_ANNOT = ["AS_MQ", "ClippingRankSum", "HAPCOMP", "HAPDOM", 
                "LikelihoodRankSum", "AS_MQRankSum", "AS_BaseQRankSum",
                "AS_ReadPosRankSum", "X_HIL"]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Annotate VCF with ML-based variant classification")
    parser.add_argument("--input", "-i", required=False, nargs='+', help="Input VCF file(s)", default=['input/vcf/E2_2_8_S18.GATKGVCF.vcf','input/vcf/E2_2_4_S14.GATKGVCF.vcf'])
    parser.add_argument("--output-dir", "-o", required=False, help="Output directory for annotated VCFs",default='output/pred_vcfs')
    parser.add_argument("--snv-model", "-sm", required=False, help="Path to trained ML model for SNVs", default='input/model/logistic_regression_snv_model.joblib')
    parser.add_argument("--indel-model", "-im", required=False, help="Path to trained ML model for indels", default='input/model/random_forest_indel_model.joblib')
    parser.add_argument("--snv-threshold", "-st", type=float, default=None, 
                        help="Custom probability threshold for SNVs (default: use model's threshold)")
    parser.add_argument("--indel-threshold", "-it", type=float, default=None, 
                        help="Custom probability threshold for indels (default: use model's threshold)")
    parser.add_argument("--threshold-type", "-tt", choices=['f1', 'sensitivity', 'custom'], default='sensitivity',
                        help="Type of threshold to use (f1, sensitivity, or custom)")
    parser.add_argument("--tmp-dir", "-td", type=str, default=None,
                        help="Temporary directory for intermediate files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()

def extract_variant_info(vcf_files: List[str]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    First pass through all VCF files to extract variant information needed for feature calculation.
    Assumes single-sample VCF with pre-split multi-allelic sites.
    
    Returns:
        variants_data: List of dictionaries with variant data
        variant_ids: List of variant IDs in the format "CHROM:POS:REF:ALT"
        vcf_sources: List of source VCF files for each variant
    """
    logging.info(f"First pass: Extracting variant information from {len(vcf_files)} VCF file(s)")
    variants_data = []
    variant_ids = []
    vcf_sources = []
    
    for vcf_file in vcf_files:
        logging.info(f"Processing file: {vcf_file}")
        
        try:
            vcf_reader = pysam.VariantFile(vcf_file)
            sample_name = list(vcf_reader.header.samples)[0]  # Get the name of the sample
            
            for record in vcf_reader:
                # Skip spanning deletions
                if '*' in record.alts or record.ref == '*':
                    continue
                
                chrom = record.chrom
                pos = record.pos
                ref = record.ref
                alt = record.alts[0]
                
                # Create a unique variant ID
                variant_id = f"{chrom}:{pos}:{ref}:{alt}"
                variant_ids.append(variant_id)
                vcf_sources.append(vcf_file)
                
                is_snv = 1 if len(alt) == 1 and len(ref) == 1 else 0
                
                # Extract required annotations
                annotations = {}
                for annot in INCLUDE_ANNOTATIONS:
                    if annot in record.info:
                        annotations[annot] = record.info.get(annot)
                    else:
                        annotations[annot] = 'NA'
                
                sample = record.samples[sample_name]

                ad = sample.get('AD', None)
                if ad and len(ad) > 1:
                    ad_ref = ad[0]
                    ad_alt = ad[1]
                else:
                    ad_ref = 0
                    ad_alt = 0

                dp_sample = sample.get('DP', 0)
                gq = sample.get('GQ', 0)
                dp_site = record.info.get('DP', 0)
                
                # Combine all information
                variant_data = {
                    'CHROM': chrom,
                    'POS': pos,
                    'REF': ref,
                    'ALT': alt,
                    'AD_REF': ad_ref,
                    'AD_ALT': ad_alt,
                    'DP_SITE': dp_site,
                    'DP_SAMPLE': dp_sample,
                    'GQ': gq,
                    'IS_SNV': is_snv,
                    **annotations
                }
                
                variants_data.append(variant_data)
                
            vcf_reader.close()
            
        except Exception as e:
            logging.error(f"Error processing file {vcf_file}: {str(e)}")
            raise
    
    logging.info(f"Extracted information for {len(variants_data)} variants across all files")
    return variants_data, variant_ids, vcf_sources

def impute_missing_values(variants_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate means for specified annotations across all variants
    
    Args:
        variants_data: List of dictionaries with variant data
        
    Returns:
        Dictionary of mean values
    """
    logging.info("Calculating mean values across all input files")
    sums = {a: 0.0 for a in IMPUTE_ANNOT}
    counts = {a: 0 for a in IMPUTE_ANNOT}
    
    for variant in variants_data:
        for annot in IMPUTE_ANNOT:
            val = variant.get(annot, 'NA')
            if val not in ('NA', '.', '', 'nan', None):
                try:
                    num_val = float(val)
                    sums[annot] += num_val
                    counts[annot] += 1
                except (ValueError, TypeError):
                    pass
    
    means = {}
    for annot in IMPUTE_ANNOT:
        means[annot] = sums[annot] / counts[annot] if counts[annot] > 0 else 0.0
        logging.debug(f"Mean value for {annot}: {means[annot]}")
    
    return means

def prepare_features(variants_data: List[Dict[str, Any]], means: Dict[str, float]) -> pd.DataFrame:
    """
    Prepare features for ML models with correct formatting
    
    Args:
        variants_data: List of dictionaries with variant data
        means: Dictionary of mean values
        
    Returns:
        DataFrame with prepared features
    """
    logging.info("Preparing features for ML models")
    
    feature_rows = []
    
    for variant in variants_data:
        try:
            ad_alt = float(variant['AD_ALT'])
            dp_site = float(variant['DP_SITE'])
            allele_ratio = round(ad_alt / dp_site, 6) if dp_site > 0 else 0
        except (ValueError, ZeroDivisionError):
            allele_ratio = 0
        try:
            dp_site = float(variant['DP_SITE'])
            dp_sample = float(variant['DP_SAMPLE'])
            unused_bases = int(dp_site) - int(dp_sample)
        except (ValueError, TypeError):
            unused_bases = 0

        feature_row = {
            'AD_sample': variant['AD_ALT'],
            'DP_site': variant['DP_SITE'],
            'VAF': allele_ratio,
            'dDP': unused_bases,
            'GQ_sample': variant['GQ'],
            'IS_SNV': variant['IS_SNV']
        }

        for annot in INCLUDE_ANNOTATIONS:
            value = variant.get(annot, 'NA')
            if annot in IMPUTE_ANNOT and value in ('NA', '.', '', 'nan', None):
                feature_row[annot] = means[annot]
            else:
                try:
                    feature_row[annot] = float(value) if value not in ('NA', '.', '', 'nan', None) else 0.0
                except (ValueError, TypeError):
                    feature_row[annot] = 0.0
        
        feature_rows.append(feature_row)
    
    features_df = pd.DataFrame(feature_rows)

    for i, variant in enumerate(variants_data):
        features_df.loc[i, 'CHROM'] = variant['CHROM']
        features_df.loc[i, 'POS'] = variant['POS']
        features_df.loc[i, 'REF'] = variant['REF']
        features_df.loc[i, 'ALT'] = variant['ALT']
    
    return features_df

def load_model(model_path: str):
    """Load the saved ML model"""
    logging.info(f"Loading ML model from {model_path}")
    try:
        model = joblib.load(model_path)
        logging.info(f"Loaded model type: {getattr(model, 'model_type', 'Unknown')}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

def get_model_threshold(model, threshold_type: str, custom_threshold: Optional[float]) -> float:
    """Determine which threshold to use based on arguments"""
    if custom_threshold is not None:
        logging.info(f"Using custom threshold: {custom_threshold}")
        return custom_threshold
    
    if threshold_type == 'sensitivity' and hasattr(model, 'sensitivity_threshold') and model.sensitivity_threshold is not None:
        threshold = model.sensitivity_threshold
        logging.info(f"Using sensitivity threshold: {threshold} (target: {getattr(model, 'target_sensitivity', 'Unknown')})")
        return threshold
    
    # Default to F1-optimal threshold
    threshold = getattr(model, 'optimal_threshold', 0.5)
    logging.info(f"Using F1-optimal threshold: {threshold}")
    return threshold

def make_predictions(model, features_df: pd.DataFrame, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using the ML model
    
    Returns:
        probabilities: Numpy array of probabilities
        predictions: Numpy array of binary predictions
    """
    logging.info("Making predictions")
    
    # Remove metadata columns
    meta_columns = ['CHROM', 'POS', 'REF', 'ALT', 'IS_SNV']
    X = features_df.drop(columns=meta_columns, errors='ignore')

    feature_subset = None
    if hasattr(model, 'best_params') and isinstance(model.best_params, dict):
        feature_subset = model.best_params.get('feature_subset', None)
    
    # Generate probabilities
    try:
        probabilities = model.predict_proba(X, feature_subset=feature_subset)
    except Exception as e:
        logging.error(f"Error during probability prediction: {e}")
        sys.exit(1)

    predictions = (probabilities >= threshold).astype(int)
    
    return probabilities, predictions

def annotate_vcf(input_vcf: str, output_vcf: str, variant_lookup: Dict[str, Dict[str, Any]], threshold: float):
    """Annotate a single VCF with predictions and write to output VCF"""
    logging.info(f"Annotating VCF: {input_vcf} -> {output_vcf}")
    
    # Use original VCF as draft
    vcf_in = pysam.VariantFile(input_vcf)
    header = vcf_in.header
    
    # Filtering header info
    header.add_meta('INFO', items=[
        ('ID', 'ML_PROB'),
        ('Number', 1),
        ('Type', 'Float'),
        ('Description', 'Machine learning model probability of true positive')
    ])
    header.add_meta('INFO', items=[
        ('ID', 'ML_PREDICTION'),
        ('Number', 1),
        ('Type', 'Integer'),
        ('Description', 'Machine learning prediction (1=true positive, 0=false positive)')
    ])
    header.add_meta('FILTER', items=[
        ('ID', 'ML_FILTERED'),
        ('Description', f'Filtered by ML model (probability < {threshold})')
    ])
    
    # Create output VCF
    vcf_out = pysam.VariantFile(output_vcf, 'w', header=header)
    
    # Process variants
    variant_count = 0
    annotated_count = 0
    filtered_count = 0
    
    for record in vcf_in:
        variant_count += 1
        
        # Get variant ID
        variant_id = f"{record.chrom}:{record.pos}:{record.ref}:{record.alts[0]}"
        
        # Check if we have a prediction for this variant
        if variant_id in variant_lookup:
            annotated_count += 1
            pred_info = variant_lookup[variant_id]
            prob = pred_info['probability']
            pred = pred_info['prediction']
            
            # Add INFO fields
            record.info['ML_PROB'] = prob
            record.info['ML_PREDICTION'] = int(pred)
            
            # Update FILTER field if prediction is false positive
            if pred == 0:
                filtered_count += 1
                # Clear existing filters
                if 'PASS' in record.filter:
                    record.filter.clear()
                # Add ML_FILTERED
                record.filter.add('ML_FILTERED')
            elif len(record.filter) == 0 or 'PASS' in record.filter:
                # Set to PASS by clearing filters
                record.filter.clear()
        vcf_out.write(record)
    vcf_in.close()
    vcf_out.close()
    
    logging.info(f"File {input_vcf}: Processed {variant_count} variant records")
    logging.info(f"File {input_vcf}: Annotated {annotated_count} variants with predictions")
    logging.info(f"File {input_vcf}: Applied ML filter to {filtered_count} variants")

def main():
    
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tmp_dir = None
    if args.tmp_dir:
        tmp_dir = Path(args.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract variant information from all VCF files
    variants_data, variant_ids, vcf_sources = extract_variant_info(args.input)
    means = impute_missing_values(variants_data)
    features_df = prepare_features(variants_data, means)

    # Split variants to perform split predictions
    snv_mask = features_df['IS_SNV'] == 1
    indel_mask = ~snv_mask
    
    snv_features = features_df[snv_mask].reset_index(drop=True)
    indel_features = features_df[indel_mask].reset_index(drop=True)
    
    snv_variant_ids = [variant_ids[i] for i, is_snv in enumerate(snv_mask) if is_snv]
    indel_variant_ids = [variant_ids[i] for i, is_snv in enumerate(snv_mask) if not is_snv]
    
    logging.info(f"Split variants into {len(snv_variant_ids)} SNVs and {len(indel_variant_ids)} indels")

    snv_model = load_model(args.snv_model)
    indel_model = load_model(args.indel_model)

    snv_threshold = get_model_threshold(snv_model, args.threshold_type, args.snv_threshold)
    indel_threshold = get_model_threshold(indel_model, args.threshold_type, args.indel_threshold)

    predictions_lookup = {}
    
    if len(snv_features) > 0:
        snv_probabilities, snv_predictions = make_predictions(snv_model, snv_features, snv_threshold)
        for i, variant_id in enumerate(snv_variant_ids):
            predictions_lookup[variant_id] = {
                'probability': snv_probabilities[i],
                'prediction': snv_predictions[i],
                'threshold': snv_threshold
            }
    
    if len(indel_features) > 0:
        indel_probabilities, indel_predictions = make_predictions(indel_model, indel_features, indel_threshold)
        for i, variant_id in enumerate(indel_variant_ids):
            predictions_lookup[variant_id] = {
                'probability': indel_probabilities[i],
                'prediction': indel_predictions[i],
                'threshold': indel_threshold
            }

    # Create mapping of VCF files to keep track of variants
    vcf_variants = {}
    for i, vcf_file in enumerate(vcf_sources):
        if vcf_file not in vcf_variants:
            vcf_variants[vcf_file] = []
        vcf_variants[vcf_file].append(variant_ids[i])

    # Annotate annotate each VCF file and create output
    for vcf_file in args.input:

        vcf_name = Path(vcf_file).name
        output_vcf = output_dir / vcf_name
        
        # Lookup variants
        file_variant_ids = vcf_variants.get(vcf_file, [])
        file_predictions = {vid: predictions_lookup[vid] for vid in file_variant_ids if vid in predictions_lookup}
        threshold = f'{snv_threshold:.2f}/{indel_threshold:.2f}'

        annotate_vcf(vcf_file, str(output_vcf), file_predictions, threshold)
    
    logging.info("VCF annotation complete")

if __name__ == "__main__":
    main()