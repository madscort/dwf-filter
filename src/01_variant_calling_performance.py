from matrix_context import get_context
from utils import (
    merge_vcfs,
    subset_multisample_vcf,
    normalize_vcf,
    intersect_vcfs_vcfeval,
    call_state_by_type,
    calculate_metrics
)
from pathlib import Path
import tempfile
import pandas as pd

# Setup paths
output = Path('./benchmark_pool_E1')
output.mkdir(exist_ok=True)
sampletable = Path('data/E1/sampletable_E1.tsv')
decodetable = Path('data/E1/decodetable.tsv')
wgs_vcfs = Path('data/E1/wgs')
sdf = Path('data/reference/Homo_sapiens_assembly38_noAlt.sdf')
ref = Path('data/reference/Homo_sapiens_assembly38_noAlt.fna')
bedfile = Path('data/TruSight_Hereditary_Cancer_TargetedRegions_v2_hg38.bed')
rtg_img = Path('data/dwf/bin/container_cache/quay.io-singularity-rtg-tools-3.12.1--hdfd78af_0.img')

caller_config = {
    'gatk_joint': {
        'suffix': 'GATK',
        'is_joint': True,
        'format_filter': None
    },
    'gatk_lenient': {
        'suffix': 'GATKGVCF',
        'is_joint': False,
        'format_filter': None
    },
    'gatk': {
        'suffix': 'GATKGVCF',
        'is_joint': False,
        'format_filter': None
    },
    'lofreq_default': {
        'suffix': 'lofreq',
        'is_joint': False,
        'format_filter': None
    },
    'lofreq_lenient': {
        'suffix': 'lofreq',
        'is_joint': False,
        'format_filter': None
    },
    'octopus': {
        'suffix': 'octopus',
        'is_joint': False,
        'format_filter': None
    },
    'crisp': {
        'suffix': 'crisp',
        'is_joint': True,
        'format_filter': 'AC'
    },
    'ngsep': {
        'suffix': 'ngsep',
        'is_joint': False,
        'format_filter': None
    }
}

benchmarking_results = []

mc = get_context(sampletable=sampletable, output_dir=output, decodetable=decodetable)

for pool_id in mc.all_pools:
    samples = mc.samples_in_pool(pool_id)
    print(f"\n{'='*60}")
    print(f"Processing Pool {pool_id}: {len(samples)} samples")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory(prefix=f"pool_{pool_id}_") as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Merge DeepVariant VCFs
        print("Merging DeepVariant VCFs...")
        dv = merge_vcfs(
            sample_ids=samples,
            input_dir=wgs_vcfs / 'DV',
            output_vcf=tmpdir / 'dv.vcf.gz',
            caller='DV',
            reference=ref,
            bed=bedfile
        )
        
        # Subset GATK joint VCF
        print("Subsetting GATK VCF...")
        gatk = subset_multisample_vcf(
            sample_ids=samples,
            input_vcf=wgs_vcfs / 'GATK_joint' / 'joint.GATK.vcf.gz',
            output_vcf=tmpdir / 'gatk.vcf.gz',
            reference=ref,
            bed=bedfile
        )
        
        # Create gold standard using vcfeval
        print("Creating gold standard with rtg vcfeval...")
        gold_standard = intersect_vcfs_vcfeval(
            a_vcf=gatk,
            b_vcf=dv,
            sdf_dir=sdf,
            singularity_img=rtg_img,
            out_prefix=tmpdir / 'gold_vcfeval'
        )
        
        # Benchmark each caller against gold standard
        for caller, config in caller_config.items():
            suffix = config['suffix']
            format_filter = config['format_filter']
            print(f"\n--- Benchmarking {caller} ({suffix}) ---")
            
            # Get the pooled VCF
            if config['is_joint']:
                pooled_raw = Path(f"data/E1/pool/{caller}/joint.{suffix}.vcf.gz")
                pooled_norm = subset_multisample_vcf(
                    sample_ids=[pool_id],
                    input_vcf=pooled_raw,
                    output_vcf=tmpdir / f"pooled_{suffix}.vcf.gz",
                    reference=ref,
                    bed=bedfile,
                    format_filter=config['format_filter']
                )
            else:
                pooled_raw = Path(f"data/E1/pool/{caller}/{pool_id}.{suffix}.vcf.gz")
                if not pooled_raw.exists():
                    print(f"  [WARN] Missing pooled VCF for {caller} in {pool_id}: {pooled_raw}")
                    continue
                pooled_norm = normalize_vcf(
                    pooled_raw,
                    tmpdir / f"pooled_{suffix}.vcf.gz",
                    reference=ref,
                    bed=bedfile
                )
            
            # Compare against gold standard
            print(f"  Comparing against vcfeval gold standard...")
            comparison = intersect_vcfs_vcfeval(
                a_vcf=pooled_norm,
                b_vcf=gold_standard['both'],
                sdf_dir=sdf,
                singularity_img=rtg_img,
                out_prefix=tmpdir / f'benchmark_{suffix}'
            )
            
            # Get counts by variant type
            counts = call_state_by_type(
                tp=comparison['both'],
                fp=comparison['a_only'],
                fn=comparison['b_only']
            )
            
            # Calculate metrics and store results
            for vartype, stats in counts.items():
                tp, fp, fn = stats['TP'], stats['FP'], stats['FN']
                metrics = calculate_metrics(tp, fp, fn)
                
                print(f"    {vartype.upper():6} -> TP: {tp:6d}, FP: {fp:6d}, FN: {fn:6d}, "
                      f"Sens: {metrics['sensitivity']:.4f}, F1: {metrics['F1_score']:.4f}")
                
                benchmarking_results.append({
                    'pool_id': pool_id,
                    'caller': caller,
                    'call_suffix': suffix,
                    'variant_type': vartype.upper(),
                    'total': metrics['total_variant_calls'],
                    'total_in_truthset': metrics['total_variants_in_truthset'],
                    'TP': tp,
                    'FP': fp,
                    'FN': fn,
                    'sensitivity': metrics['sensitivity'],
                    'precision': metrics['precision'],
                    'F1_score': metrics['F1_score'],
                    'FDR': metrics['FDR']
                })

# Write per-pool benchmarking results
benchmark_df = pd.DataFrame(benchmarking_results)
benchmark_output = output / 'benchmarking_results.tsv'
benchmark_df.to_csv(benchmark_output, sep='\t', index=False, float_format='%.6f')
print(f"\nPer-pool results written to: {benchmark_output}")
print(f"  Total rows: {len(benchmark_df)}")

# Aggregate results by caller and variant_type
aggregated_df = (
    benchmark_df
    .groupby(['caller', 'call_suffix', 'variant_type'])
    .agg({
        'total': 'sum',
        'total_in_truthset': 'sum',
        'TP': 'sum',
        'FP': 'sum',
        'FN': 'sum'
    })
    .reset_index()
)

# Recalculate metrics from aggregated counts
aggregated_df['sensitivity'] = aggregated_df['TP'] / (aggregated_df['TP'] + aggregated_df['FN'])
aggregated_df['precision'] = aggregated_df['TP'] / (aggregated_df['TP'] + aggregated_df['FP'])
aggregated_df['F1_score'] = (2 * aggregated_df['TP']) / (2 * aggregated_df['TP'] + aggregated_df['FP'] + aggregated_df['FN'])
aggregated_df['FDR'] = aggregated_df['FP'] / (aggregated_df['FP'] + aggregated_df['TP'])
aggregated_df = aggregated_df.fillna(0.0)

# Write aggregated results
aggregated_output = output / 'benchmarking_results_aggregated.tsv'
aggregated_df.to_csv(aggregated_output, sep='\t', index=False, float_format='%.6f')
print(f"\nAggregated results written to: {aggregated_output}")
print(f"  Total rows: {len(aggregated_df)}")

print("\nDone!")
