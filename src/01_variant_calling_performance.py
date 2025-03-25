from pathlib import Path
from utils import get_truth_variants, get_pool_variants

def calculate_performance(pool_variants, combined_variants):
    TP = len(pool_variants.intersection(combined_variants))
    FP = len(pool_variants.difference(combined_variants))
    FN = len(combined_variants.difference(pool_variants))
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * ((prec * sens) / (sens + prec)) if (sens + prec) > 0 else 0
    return len(combined_variants), TP, FP, FN, sens, prec, f1

## mads - 2025-03-23
# Script used for comparing variant calling results in pooled sequencing data.

if __name__ == '__main__':
    experiment = 'E1'
    base_path = Path("")
    wgs_vartable_folder = base_path / f"data/{experiment}_wgs/results/variant_tables"
    decodetable = base_path / "data/decodetable.tsv"
    pooltable = base_path / f"data/{experiment}_pool/pooltable.tsv"

    dv_variants, _ = get_truth_variants(decodetable, wgs_vartable_folder, tool='DV', joint=False, get_info=True)
    gatkj_variants, _ = get_truth_variants(decodetable, wgs_vartable_folder, tool='GATK', joint=True, get_info=True)
    
    combined_variants = dv_variants.intersection(gatkj_variants)

    tools = {
        "GATK joint": base_path / f"data/{experiment}_pool/gatk_jointgenotypin/pinpoint/results_lenient/variant_tables",
        "CRISP": base_path / f"data/{experiment}_pool/crisp/results/variant_tables",
        "GATK": base_path / f"data/{experiment}_pool/gatk/pinpoint/results/variant_tables",
        "GATK lenient": base_path / f"data/{experiment}_pool/gatk/pinpoint/results_lenient/variant_tables",
        "LoFreq lenient": base_path / f"data/{experiment}_pool/lofreq_lenient/results/variant_tables",
        "LoFreq": base_path / f"data/{experiment}_pool/lofreq/results/variant_tables",
    }

    performance = {}
    for tool_name, vartable_path in tools.items():
        pool_variants = get_pool_variants(pooltable=pooltable, variant_tables=vartable_path, tool=tool_name.split()[0], joint="joint" in tool_name)
        performance[tool_name] = calculate_performance(pool_variants, combined_variants)

        print(f"{tool_name}: TP={performance[tool_name][1]}, FP={performance[tool_name][2]}, FN={performance[tool_name][3]}, Sensitivity={performance[tool_name][4]:.4f}, Precision={performance[tool_name][5]:.4f}, F1={performance[tool_name][6]:.4f}")

    # Handle failed octopus run independently:
    octopustable = base_path / f"data/{experiment}_pool/octopus/pooltable.tsv"
    octopus_vartables = base_path / f"data/{experiment}_pool/octopus/results/variant_tables/"
    octopus_pool_variants = get_pool_variants(pooltable=octopustable, variant_tables=octopus_vartables, tool='octopus', joint=False)

    octodecode = base_path / f"data/{experiment}_pool/octopus/decodetable.tsv"
    dv_variants_octo, _ = get_truth_variants(octodecode, wgs_vartable_folder, tool='DV', joint=False, get_info=True, acmg=acmg_file)
    gatkj_variants_octo, _ = get_truth_variants(octodecode, wgs_vartable_folder, tool='GATK', joint=True, get_info=True, acmg=acmg_file)

    combined_variants_octo = dv_variants_octo.intersection(gatkj_variants_octo)
    performance["Octopus"] = calculate_performance(octopus_pool_variants, combined_variants_octo)

    print(f"Octopus: TP={performance['Octopus'][1]}, FP={performance['Octopus'][2]}, FN={performance['Octopus'][3]}, Sensitivity={performance['Octopus'][4]:.4f}, Precision={performance['Octopus'][5]:.4f}, F1={performance['Octopus'][6]:.4f}")

    # Save performance
    output_file = base_path / "output/performance.tsv"
    with open(output_file, 'w') as fout:
        print('tool', 'TP / total', 'truth', 'TP', 'FP', 'FN', 'sensitivity', 'precision', 'F1', sep='\t', file=fout)
        for tool, metrics in performance.items():
            print(tool, f"{metrics[1]} / {metrics[0]}", *metrics, sep='\t', file=fout)
