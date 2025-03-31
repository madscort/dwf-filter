from pathlib import Path
from utils import get_truth_variants, get_pool_variants, calculate_performance_metrics

## mads - 2025-03-23
# Script used for comparing variant calling results in pooled sequencing data.
# Takes variant tables (GATK VariantToTable conversion from VCFs)
# Outputs a table with: method,total_calls,TP,FP,FN,Sensitivity,F1,FDR

if __name__ == '__main__':
    experiment = 'E1'
    base_path = Path("")
    wgs_vartable_folder = base_path / f"data/{experiment}_wgs/results/variant_tables"
    decodetable = base_path / "data/decodetable.tsv"
    pooltable = base_path / f"data/{experiment}_pool/pooltable.tsv"

    dv_variants = get_truth_variants(decodetable, wgs_vartable_folder, tool='DV', joint=False)
    gatkj_variants = get_truth_variants(decodetable, wgs_vartable_folder, tool='GATK', joint=True)
    
    combined_variants = dv_variants.intersection(gatkj_variants)

    # Paths for variant calling tools
    tools_config = {
        "GATK": {
            "path": base_path / f"data/{experiment}_pool/gatk/results/variant_tables",
            "tool": 'GATKGVCF',
            "joint": False
        },
        "GATK lenient": {
            "path": base_path / f"data/{experiment}_pool/gatk/results_lenient/variant_tables",
            "tool": 'GATKGVCF',
            "joint": False
        },
        "GATK joint": {
            "path": base_path / f"data/{experiment}_pool/gatk_jointgenotyping/variant_tables",
            "tool": 'GATK',
            "joint": True
        },
        "CRISP": {
            "path": base_path / f"data/{experiment}_pool/crisp/results/variant_tables",
            "tool": 'CRISP',
            "joint": True
        },
        "LoFreq": {
            "path": base_path / f"data/{experiment}_pool/lofreq/results/variant_tables",
            "tool": 'lofreq',
            "joint": False
        },
        "LoFreq lenient": {
            "path": base_path / f"data/{experiment}_pool/lofreq/results_lenient/variant_tables",
            "tool": 'lofreq',
            "joint": False
        }
    }

    performance = {}
    for tool_name, config in tools_config.items():
        pool_variants = get_pool_variants(pooltable=pooltable, variant_tables=config["path"], 
                                          tool=config["tool"], joint=config["joint"])
        
        counts, metrics = calculate_performance_metrics(combined_variants, pool_variants)
        
        performance[tool_name] = {
            "Total": len(combined_variants),
            "TP": counts["TP"],
            "FP": counts["FP"],
            "FN": counts["FN"],
            "Sensitivity": metrics["sensitivity"],
            "Precision": metrics["precision"],
            "F1": metrics["F1"],
            "FDR": metrics["FDR"]
        }

        print(f"{tool_name}: TP={counts['TP']}, FP={counts['FP']}, FN={counts['FN']}, "
              f"Sensitivity={metrics['sensitivity']:.4f}, Precision={metrics['precision']:.4f}, "
              f"F1={metrics['F1']:.4f}, FDR={metrics['FDR']:.4f}")

    # Handle failed Octopus run independently
    octopustable = base_path / f"data/{experiment}_pool/octopus/pooltable.tsv"
    octopus_vartables = base_path / f"data/{experiment}_pool/octopus/results/variant_tables/"
    octopus_pool_variants = get_pool_variants(pooltable=octopustable, variant_tables=octopus_vartables, 
                                             tool='octopus', joint=False)

    octodecode = base_path / f"data/{experiment}_pool/octopus/decodetable.tsv"
    dv_variants_octo = get_truth_variants(octodecode, wgs_vartable_folder, tool='DV', joint=False)
    gatkj_variants_octo = get_truth_variants(octodecode, wgs_vartable_folder, tool='GATK', joint=True)

    combined_variants_octo = dv_variants_octo.intersection(gatkj_variants_octo)
    
    counts_octo, metrics_octo = calculate_performance_metrics(combined_variants_octo, octopus_pool_variants)
    
    performance["Octopus"] = {
        "Total": len(combined_variants_octo),
        "TP": counts_octo["TP"],
        "FP": counts_octo["FP"],
        "FN": counts_octo["FN"],
        "Sensitivity": metrics_octo["sensitivity"],
        "Precision": metrics_octo["precision"],
        "F1": metrics_octo["F1"],
        "FDR": metrics_octo["FDR"]
    }

    print(f"Octopus: TP={counts_octo['TP']}, FP={counts_octo['FP']}, FN={counts_octo['FN']}, "
          f"Sensitivity={metrics_octo['sensitivity']:.4f}, Precision={metrics_octo['precision']:.4f}, "
          f"F1={metrics_octo['F1']:.4f}, FDR={metrics_octo['FDR']:.4f}")

    output_file = base_path / "01_variant_calling_performance.tsv"
    with open(output_file, 'w') as fout:
        print('method', 'Total', 'TP', 'FP', 'FN', 'Sensitivity', 'F1', 'FDR', sep='\t', file=fout)

        for tool, metrics in performance.items():
            print(tool, 
                  metrics["Total"],
                  metrics["TP"], 
                  metrics["FP"], 
                  metrics["FN"],
                  f"{metrics['Sensitivity']:.4f}",
                  f"{metrics['F1']:.4f}",
                  f"{metrics['FDR']:.4f}",
                  sep='\t', file=fout)