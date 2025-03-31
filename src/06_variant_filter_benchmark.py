from pathlib import Path
from utils import get_truth_variants, get_pool_variants, pin_truth_variants, pin_pool_variants

def split_variants(variants):
    # Remove spanning deletions
    variants = {variant for variant in variants 
                            if not variant.split(':')[4] == '*'}
    return {
        'all': variants,
        'SNV': {variant for variant in variants if len(variant.split(':')[3]) == len(variant.split(':')[4]) == 1},
        'INDEL': {variant for variant in variants if len(variant.split(':')[3]) != len(variant.split(':')[4])}
    }

def get_vardict(pooltable,variant_tables,is_joint,variant_type,tool):
    variants = get_pool_variants(
                    pooltable=pooltable,
                    variant_tables=variant_tables,
                    tool=tool,
                    joint=is_joint,
                    variant_type=variant_type)
	
	# Create a variant dictionary for pinpointing.
    vardict = {}
    vardict['row'] = {}
    vardict['column'] = {}
    for n in range(10):
        vardict['row'][n] = set()
        vardict['column'][n] = set()
    rc = ['row','column']
    for variant in variants:
        varid = ":".join(variant.split(':')[1:])
        d,idx = variant.split(':')[0].split('_')
        vardict[rc[int(d)-1]][int(idx)-1].add(varid)

    return vardict

def calculate_performance_metrics(truth_set, predicted_set):
    
    TP = len(predicted_set & truth_set)
    FP = len(predicted_set - truth_set)
    FN = len(truth_set - predicted_set)

    sensitivity = TP / (TP + FN) if (TP + FN) else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    f1_score = 2 * (precision * sensitivity) / (sensitivity + precision + 1e-10)
    fdr = FP / (TP + FP) if (TP + FP) else 0
    
    return {'TP':TP,'FP':FP,'FN':FN}, {"sensitivity": sensitivity, "precision": precision, "F1": f1_score, "FDR": fdr}

def process_variant_data(pooltable, methods, truth_set, baseline=None, pinpoint=False):
    performance = {}
    for method_name, (tool, filtering, is_joint, variant_table_path) in methods.items():
        performance[method_name] = {}
        for variant_type, variants in truth_set.items():
            print(f"RUNNING: {method_name} {variant_type}")
            
            if pinpoint:
                vardict = get_vardict(
                    pooltable=pooltable,
                    variant_tables=variant_table_path,
                    is_joint=is_joint,
                    variant_type=variant_type,
                    tool=tool)
                _, pinpointed_variants = pin_pool_variants(
                    decodetable=decodetable, 
                    pooltable=pooltable,
                    variant_tables=variant_table_path,
                    matrix_size=10,
                    tool=tool,
                    vardict=vardict)
                split_pinpoints = split_variants(pinpointed_variants)
                predicted_variants = split_pinpoints[variant_type]
            else:
                predicted_variants = get_pool_variants(
                    pooltable=pooltable,
                    variant_tables=variant_table_path,
                    tool=tool,
                    joint=is_joint,
                    variant_type=variant_type
                )
            if pinpoint:
                # Only keep one instance of each pin
                variants = {':'.join(variant.split(':')[1:]) for variant in variants}
                predicted_variants = {':'.join(variant.split(':')[1:]) for variant in predicted_variants}

            counts, metrics = calculate_performance_metrics(variants, predicted_variants)

            performance[method_name][variant_type] = {
                "data": {
                    "variant_type": variant_type,
                    "caller": tool,
                    "filtering": filtering,
                    "caller_type": "joint" if is_joint else "single",
                    "TP": counts['TP'],
                    "FP": counts['FP'],
                    "FN": counts['FN']
                },
                "metrics": metrics
            }
    return performance


if __name__ == '__main__':
    decodetable = Path("")
    wgs_vartable_folder = Path("")
    pooltable = Path("")
    output_table = Path("")
    pinpointable_performance = False

    # Get "true" variants
    if pinpointable_performance:
        theoretical_pins = set()
        _, theoretical_pins_gatkj = pin_truth_variants(decodetable=decodetable, pooltable=pooltable, variant_tables=wgs_vartable_folder, matrix_size=10, tool='GATK', joint=True)
        _, theoretical_pins_dv = pin_truth_variants(decodetable=decodetable, pooltable=pooltable, variant_tables=wgs_vartable_folder, matrix_size=10, tool='DV')
        combined_variants = theoretical_pins_gatkj & theoretical_pins_dv
    else:
        dv_variants = get_truth_variants(decodetable, wgs_vartable_folder, tool='DV', joint=False)
        gatkj_variants = get_truth_variants(decodetable, wgs_vartable_folder, tool='GATK', joint=True)
        combined_variants = dv_variants & gatkj_variants

    truth_set = split_variants(combined_variants)

    methods = {
        'single': ('GATKGVCF', 'baseline', False, Path("")),
        'joint': ('GATK', 'baseline', True, Path(""))
    }
    
    # Compute baseline performance
    baseline_performance = process_variant_data(pooltable, methods, truth_set, pinpoint=pinpointable_performance)
    
    # Compare methods to baseline
    filter_methods = {
        'GATK_HT': ('GATKGVCF', 'HT', True, Path("/path/to/method/pool_variant_tables/tsv")),
        'GATK_MLF1': ('GATKGVCF', 'ML-F1', True, Path("/path/to/method/pool_variant_tables/tsv")),
        'GATK_MLS': ('GATKGVCF', 'ML-S', True, Path("/path/to/method/pool_variant_tables/tsv")),
        'GATK_VQSR': ('GATK', 'VQSR', True, Path("/path/to/method/pool_variant_tables/tsv"))
    }
    performance_ml = process_variant_data(pooltable, filter_methods, truth_set, baseline=baseline_performance, pinpoint=pinpointable_performance)
    
    with open(output_table, 'w') as fout:
        print(
            "variant_type",
            "filtering",
            "caller_type",
            "TP",
            "FP",
            "FN",
            "metric",
            "value",
            "baseline",
            sep='\t',
            file=fout
        )
        for method, results in {**baseline_performance, **performance_ml}.items():
            for variant_type, data in results.items():
                performance_data = data["data"]
                metrics = data["metrics"]
                for metric_name, metric_value in metrics.items():
                    # Get the baseline value for this metric and variant type
                    baseline_method = performance_data["caller_type"]
                    baseline_value = baseline_performance[baseline_method][variant_type]["metrics"][metric_name] if baseline_method in baseline_performance else None
                    print(
                        performance_data['variant_type'],
                        performance_data['filtering'],
                        performance_data['caller_type'],
                        performance_data['TP'],
                        performance_data['FP'],
                        performance_data['FN'],
                        metric_name,
                        metric_value,
                        baseline_value,
                        sep='\t',
                        file=fout
                    )
