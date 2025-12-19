from matrix_context import get_context
from utils import normalize_all_vcfs, find_private_variants, calculate_metrics, intersect_vcfs_vcfeval, call_state_by_type
from pathlib import Path
import logging
import subprocess
import pandas as pd
from typing import Dict, List
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

def evaluate_caller(
    config_name: str,
    caller_dir: Path,
    caller_suffix: str,
    filter_method: str,
    caller_type: str,
    private_vcfs: Dict,
    output_dir: Path,
    reference: Path,
    bed: Path,
    sdf_dir: Path,
    rtg_img: Path
) -> List[Dict]:
    """Evaluate a single caller configuration against truth set."""
    
    logging.info(f"Evaluating {config_name}: {filter_method} ({caller_type})")
    
    # Collect caller VCF paths
    caller_vcfs = {}
    for individual_id in private_vcfs.keys():
        caller_vcf = caller_dir / f"{individual_id}{caller_suffix}"
        if caller_vcf.exists():
            caller_vcfs[individual_id] = caller_vcf
        else:
            logging.warning(f"Caller VCF not found for {individual_id}: {caller_vcf}")
    
    if not caller_vcfs:
        logging.error(f"No caller VCFs found for {config_name}")
        return []
    
    logging.info(f"Found {len(caller_vcfs)} caller VCFs for {config_name}")
    
    # Normalize caller VCFs
    norm_caller = normalize_all_vcfs(
        caller_vcfs,
        output_dir,
        reference,
        bed=bed,
        label=f"caller_{config_name}"
    )
    
    # Perform comparisons
    results = []
    comparisons_dir = output_dir / "comparisons" / config_name
    comparisons_dir.mkdir(exist_ok=True, parents=True)
    
    for individual_id, (private_vcf, n_private) in private_vcfs.items():
        if individual_id not in norm_caller:
            continue
            
        logging.debug(f"Comparing {individual_id} for {config_name}")
        
        work_dir = comparisons_dir / individual_id
        work_dir.mkdir(exist_ok=True)
        
        # Compare with vcfeval
        comparison = intersect_vcfs_vcfeval(
            a_vcf=norm_caller[individual_id],
            b_vcf=private_vcf,
            sdf_dir=sdf_dir,
            singularity_img=rtg_img,
            out_prefix=work_dir / "vcfeval"
        )
        
        # Get counts by variant type
        counts = call_state_by_type(
            tp=comparison['both'],
            fp=comparison['a_only'],
            fn=comparison['b_only']
        )
        
        # Calculate metrics
        for vartype, stats in counts.items():
            tp, fp, fn = stats['TP'], stats['FP'], stats['FN']
            metrics = calculate_metrics(tp, fp, fn)
            
            logging.debug(
                f"  {vartype.upper():6} -> "
                f"TP: {tp:6d}, FP: {fp:6d}, FN: {fn:6d}, "
                f"Sens: {metrics['sensitivity']:.4f}, F1: {metrics['F1_score']:.4f}"
            )
            
            results.append({
                'config': config_name,
                'filter_method': filter_method,
                'caller_type': caller_type,
                'individual_id': individual_id,
                'variant_type': vartype.upper(),
                'total_variant_calls': metrics['total_variant_calls'],
                'total_variants_in_truthset': metrics['total_variants_in_truthset'],
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'sensitivity': metrics['sensitivity'],
                'precision': metrics['precision'],
                'F1_score': metrics['F1_score'],
                'FDR': metrics['FDR']
            })
    
    return results

# Setup paths
output_dir = Path('./output_pins_E2')
output_dir.mkdir(exist_ok=True)
sampletable = Path('data/E2/pooltable.tsv')
decodetable = Path('data/E2/decodetable.tsv')

# Two truth directories for intersection
truth_dir_dv = Path('data/E2/wgs/DV')
truth_dir_gatk = Path('data/E2/wgs/GATK_joint')

sdf_dir = Path('data/reference/Homo_sapiens_assembly38_noAlt.sdf')
reference = Path('data/reference/Homo_sapiens_assembly38_noAlt.fna')
bed = Path('data/TruSight_Hereditary_Cancer_TargetedRegions_v2_hg38.bed')
rtg_img = Path('data/dwf/bin/container_cache/quay.io-singularity-rtg-tools-3.12.1--hdfd78af_0.img')

caller_suffix = '_unique_pins.vcf.gz'

caller_config = {
    'single': {
        'caller_dir': Path('data/E2/pinpoint/single'),
        'filter_method': 'baseline',
        'caller_type': 'single'
    },
    'joint': {
        'caller_dir': Path('data/E2/pinpoint/joint'),
        'filter_method': 'baseline',
        'caller_type': 'joint'
    },
    'mlf1': {
        'caller_dir': Path('data/E2/pinpoint/mlf1_pinfilter'),
        'filter_method': 'ML-F1',
        'caller_type': 'single',
        'baseline': 'single'
    },
    'mlsens': {
        'caller_dir': Path('data/E2/pinpoint/mlsens_pinfilter'),
        'filter_method': 'ML-S',
        'caller_type': 'single',
        'baseline': 'single'
    },
    'vqsr': {
        'caller_dir': Path('data/E2/pinpoint/vqsr'),
        'filter_method': 'VQSR',
        'caller_type': 'joint',
        'baseline': 'joint'
    },
    'ht': {
        'caller_dir': Path('data/E2/pinpoint/ht'),
        'filter_method': 'HT',
        'caller_type': 'single',
        'baseline': 'single'
    }
}

mc = get_context(sampletable=sampletable, output_dir=output_dir, decodetable=decodetable)

# ============================================================
# STEP 1: Collect and normalize truth VCFs from both callers
# ============================================================
logging.info("=" * 60)
logging.info("STEP 1: Collecting and normalizing truth VCFs from DV and GATK")
logging.info("=" * 60)

truth_vcfs_dv = {}
truth_vcfs_gatk = {}

for cell in mc._ctx["matrix"]["cells"]:
    individual_id = cell["sample_id"]
    
    # DeepVariant
    dv_vcf = truth_dir_dv / f"{individual_id}.DV.vcf.gz"
    if not dv_vcf.exists():
        logging.error(f"DV truth VCF not found: {dv_vcf}")
        raise FileNotFoundError
    truth_vcfs_dv[individual_id] = dv_vcf
    
    # GATK
    gatk_vcf = truth_dir_gatk / f"{individual_id}.GATK.vcf.gz"
    if not gatk_vcf.exists():
        logging.error(f"GATK truth VCF not found: {gatk_vcf}")
        raise FileNotFoundError
    truth_vcfs_gatk[individual_id] = gatk_vcf

logging.info(f"Found {len(truth_vcfs_dv)} DV truth VCFs")
logging.info(f"Found {len(truth_vcfs_gatk)} GATK truth VCFs")

# Normalize both sets
norm_truth_dv = normalize_all_vcfs(
    truth_vcfs_dv,
    output_dir,
    reference,
    bed=bed,
    label="truth_dv"
)

norm_truth_gatk = normalize_all_vcfs(
    truth_vcfs_gatk,
    output_dir,
    reference,
    bed=bed,
    label="truth_gatk"
)

# ============================================================
# STEP 2: Compute intersection of DV and GATK for each individual
# ============================================================
logging.info("=" * 60)
logging.info("STEP 2: Computing DV ∩ GATK intersection for each individual")
logging.info("=" * 60)

intersection_dir = output_dir / "truth_intersections"
intersection_dir.mkdir(exist_ok=True)

truth_intersections = {}

for cell in mc._ctx["matrix"]["cells"]:
    individual_id = cell["sample_id"]
    
    if individual_id not in norm_truth_dv or individual_id not in norm_truth_gatk:
        logging.warning(f"Missing normalized truth for {individual_id}, skipping")
        continue
    
    logging.info(f"Computing intersection for {individual_id}")
    
    intersection_prefix = intersection_dir / f"{individual_id}_intersection"
    
    # Run vcfeval to get intersection (TP = both callers agree)
    comparison = intersect_vcfs_vcfeval(
        a_vcf=norm_truth_dv[individual_id],
        b_vcf=norm_truth_gatk[individual_id],
        sdf_dir=sdf_dir,
        singularity_img=rtg_img,
        out_prefix=intersection_prefix
    )
    
    # Use the TP (both) as the intersection truth
    intersection_vcf = comparison['both']
    
    # Count variants in intersection
    count_cmd = ["bcftools", "view", "-H", str(intersection_vcf)]
    count_result = subprocess.run(count_cmd, capture_output=True, text=True)
    n_intersection = len(count_result.stdout.strip().split('\n')) if count_result.stdout.strip() else 0
    
    logging.info(f"  {individual_id}: {n_intersection} variants in DV ∩ GATK")
    
    truth_intersections[individual_id] = intersection_vcf

# ============================================================
# STEP 3: Find private variants using intersection truth set
# ============================================================
logging.info("=" * 60)
logging.info("STEP 3: Finding private variants from intersection truth set")
logging.info("=" * 60)

private_vcfs = {}
private_dir = output_dir / "private_variants"
private_dir.mkdir(exist_ok=True)

for cell in mc._ctx["matrix"]["cells"]:
    individual_id = cell["sample_id"]
    
    if individual_id not in truth_intersections:
        logging.warning(f"No intersection truth for {individual_id}, skipping")
        continue

    other_individuals = [
        c["sample_id"] for c in mc._ctx["matrix"]["cells"]
        if c["sample_id"] != individual_id
        and c["sample_id"] in truth_intersections
    ]

    if not other_individuals:
        logging.warning(f"No other individuals found for {individual_id}")
        continue
    
    logging.debug(f"Finding private variants for {individual_id} (comparing to {len(other_individuals)} others)")
    
    other_vcfs = [truth_intersections[oid] for oid in other_individuals]
    
    private_vcf = find_private_variants(
        individual_id=individual_id,
        individual_vcf=truth_intersections[individual_id],
        other_vcfs=other_vcfs,
        output_dir=private_dir,
    )
    
    # Count private variants
    count_cmd = ["bcftools", "view", "-H", str(private_vcf)]
    count_result = subprocess.run(count_cmd, capture_output=True, text=True)
    n_private = len(count_result.stdout.strip().split('\n')) if count_result.stdout.strip() else 0
    
    logging.info(f"  {individual_id}: {n_private} private variants")
    private_vcfs[individual_id] = (private_vcf, n_private)

# ============================================================
# STEP 4: Evaluate caller configurations
# ============================================================
logging.info("=" * 60)
logging.info("STEP 4: Evaluating caller configurations")
logging.info("=" * 60)

all_results = []

for config_name, config in caller_config.items():
    results = evaluate_caller(
        config_name=config_name,
        caller_dir=config['caller_dir'],
        caller_suffix=caller_suffix,
        filter_method=config['filter_method'],
        caller_type=config['caller_type'],
        private_vcfs=private_vcfs,
        output_dir=output_dir,
        reference=reference,
        bed=bed,
        sdf_dir=sdf_dir,
        rtg_img=rtg_img
    )
    
    all_results.extend(results)

# Create DataFrame and save
df = pd.DataFrame(all_results)
output_file = output_dir / "variant_filter_benchmark_pins.tsv"
df.to_csv(output_file, sep='\t', index=False, float_format='%.6f')

logging.info("=" * 60)
logging.info(f"Results written to: {output_file}")
logging.info(f"Total rows: {len(df)}")
logging.info(f"Truth intersections saved in: {intersection_dir}")
logging.info("=" * 60)