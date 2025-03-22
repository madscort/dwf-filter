from pathlib import Path
from utils import get_truth_variants, pin_truth_variants, pin_pool_variants
import sys

include_annotations = [
	"ASSEMBLED_HAPS","AS_BaseQRankSum","AS_FS","AS_MQ","AS_MQRankSum",
	"AS_QD","AS_ReadPosRankSum","AS_SOR","ClippingRankSum","HAPCOMP","HAPDOM",
	"HEC","LikelihoodRankSum","MLEAC","MLEAF","X_GCC","X_HIL"]

def impute_wmean(vartable_folder, pools, annotations, tool='GATKGVCF'):
	"""Calculate means for specified annotations across all variants."""
	sums = {a: 0.0 for a in annotations}
	counts = {a: 0 for a in annotations}

	# Process each pool
	for pool in pools:
		with open(vartable_folder / (pool + f".{tool}.tsv"), 'r') as fin:
			header = fin.readline().strip().split('\t')
			
			# Get column indices
			idx = {a: header.index(a) if a in header else -1 for a in annotations}

			for line in fin:
				var_info = line.strip().split('\t')
				
				# Skip spanning deletions
				if var_info[4] == '*' or var_info[3] == '*':
					continue

				for a in annotations:
					if idx[a] != -1:
						val = var_info[idx[a]]
						if val not in ('NA', '.', '', 'nan'):
							try:
								num_val = float(val)
								sums[a] += num_val
								counts[a] += 1
							except ValueError:
								pass
	means = {}
	for a in annotations:
		means[a] = sums[a] / counts[a] if counts[a] > 0 else 0.0
	
	return means

def make_dataset(true_variants, pooltable, theoretical_pins, pool_pins, acmg, vartable_folder, output_table, tool='GATKGVCF'):

	impute_annot = ["AS_MQ", "ClippingRankSum", "HAPCOMP", "HAPDOM", 
				   "LikelihoodRankSum", "AS_MQRankSum", "AS_BaseQRankSum",
				   "AS_ReadPosRankSum", "X_HIL"]
	pools = []
	with open(pooltable, "r") as fin:
		for line in fin:
			pools.append(line.strip().split("\t")[0])
	
	# Impute NA values with mean
	means = impute_wmean(vartable_folder, pools, impute_annot, tool)
	output_table_snv = output_table.parent / (output_table.stem + "_snv.tsv")
	output_table_indel = output_table.parent / (output_table.stem + "_indel.tsv")

	with open(output_table, 'w') as fout, open(output_table_snv, 'w') as fouts, open(output_table_indel, 'w') as fouti:
		for file_handle in (fout, fouts, fouti):
			print(
				"gene_target", "variant_id", "unique_variant_id", "label",
				"is_pool_pin", "is_wgs_theoretical_pin", "is_lof", "is_p",
				"is_lofp", "is_acmg", "is_snv", "CHROM", "POS", "REF", "ALT",
				"AD_sample", "DP_site", "VAF", "dDP", "GQ_sample",
				"\t".join(include_annotations),
				sep="\t", file=file_handle)
			
		for pool in pools:
			with open(vartable_folder / (pool + f".{tool}.tsv"), 'r') as fin:
				header = fin.readline().strip().split('\t')

				# Get column indices
				ANN_idx = header.index('ANN')
				GENE_idx = header.index('GENE')
				LOF_idx = header.index('LOF')
				CLNSIG_idx = header.index('CLNSIG')
				CLN_GENE_idx = header.index('GENEINFO')
				DP_site_idx = header.index('DP')
				
				for idx, element in enumerate(header):
					if '.AD' in element:
						AD_idx = idx
					if '.DP' in element:
						DP_idx = idx
					if '.GQ' in element:
						GQ_idx = idx

				annotation_idx = {a: header.index(a) for a in include_annotations}

				for line in fin:
					var_info = line.strip().split('\t')

					CHROM, POS, _, REF, ALT = var_info[0:5]
					# Skip spanning deletions
					if ALT == '*' or REF == '*':
						continue
						
					variant_id = f"{CHROM}:{POS}:{REF}:{ALT}" 
					is_snv = 1 if len(ALT) == 1 and len(REF) == 1 else 0

					# Process variant data
					gene_target = var_info[GENE_idx]
					
					# Functional annotation
					effects = var_info[ANN_idx].split(',')
					gene_name = ",".join([effect.strip().split('|')[3] for effect in effects]) if effects != ['NA'] else 'NA'

					is_lof = 0
					lof_gene = 'NA'
					lof = var_info[LOF_idx]
					if lof != 'NA':
						is_lof = 1
						lof_gene = var_info[LOF_idx].split('|')[0].replace('(','')

					# ClinVar annotation
					is_p = 0
					clnsig = var_info[CLNSIG_idx].strip()
					cln_gene = var_info[CLN_GENE_idx].split(':')[0].strip()
		
					if "Pathogenic" in clnsig or "Likely_pathogenic" in clnsig:
						is_p = 1
					
					is_lofp = 1 if (is_p or is_lof) else 0

					# ACMG state
					is_acmg = 0
					for g in gene_name.split(','):
						if g in acmg:
							is_acmg = 1
					
					if lof_gene in acmg or gene_target in acmg or cln_gene in acmg:
						is_acmg = 1
					
					# Sample metrics
					DP_site = var_info[DP_site_idx].strip()
					AD = var_info[AD_idx].strip()
					DP = var_info[DP_idx].strip()
					GQ = var_info[GQ_idx].strip()
					
					if AD.count(',') > 1:
						print('Multi-allelic sites are not split:', var_info)
						sys.exit(1)
					AD = AD.split(',')[1].strip()

					pool_short_id = "_".join(pool.split("_")[1:3])
					unique_variant_id = f"{pool_short_id}:{CHROM}:{POS}:{REF}:{ALT}"

					# Get feature values with imputation
					feature_values = []
					for a in include_annotations:
						value = var_info[annotation_idx[a]]
						if a in impute_annot and value in ('NA', '.', '', 'nan'):
							feature_values.append(str(means[a]))
						else:
							feature_values.append(value)
					
					# Determine variant status
					is_theo_pin = 1 if unique_variant_id in theoretical_pins else 0
					is_pool_pin = 1 if unique_variant_id in pool_pins else 0
					label = 1 if unique_variant_id in true_variants else 0

					type_specific_file = fouts if is_snv == 1 else fouti
					for file_handle in (fout, type_specific_file):
						print(
							gene_target, variant_id, unique_variant_id, label,
							is_pool_pin, is_theo_pin, is_lof, is_p,
							is_lofp, is_acmg, is_snv, CHROM, POS, REF, ALT,
							AD, DP_site, round(int(AD)/int(DP_site),6), int(DP_site) - int(DP), GQ,
							"\t".join(feature_values),
							sep="\t", file=file_handle)

if __name__ == '__main__':
	pool_vartables = Path("/ngc/projects2/dp_00005/data/dwf/projects/variant_filter/data/E1_pool/gatk_jointgenotyping/pinpoint/results_lenient/variant_tables")
	pooltable = Path("/ngc/projects2/dp_00005/data/dwf/projects/variant_filter/data/E1_pool/gatk_hc/call/data/cramtable_pool_E1.tsv")
	decodetable = Path("/ngc/projects2/dp_00005/data/dwf/projects/variant_filter/_old/doba/data/raw/pool/decodetable.tsv")
	wgs_vartable_folder = Path("/ngc/projects2/dp_00005/data/dwf/projects/variant_filter/data/E1_wgs/combined_pipeline/results/variant_tables")
	output_table = Path(f"/ngc/projects2/dp_00005/data/dwf/projects/variant_filter/manuscript_repo/output/dataset.tsv")
	acmg_file = Path("/ngc/projects2/dp_00005/data/dwf/databases/acmg/v3.2/acmg.tsv")
	acmg = {}

	with open(acmg_file, 'r') as fin:
		fin.readline()
		for line in fin:
			try:
				info = line.split('\t')
				gene = info[0]
				disease = info[2]
				category = info[4]
			except IndexError:
				if len(line.strip()) == 0:
					continue
				else:
					print('error')
					print(line)
					sys.exit(1)
			acmg[gene] = (category,disease)

	true_variants = set()
	dv_variants, dv_variants_info = get_truth_variants(decodetable=decodetable, variant_tables=wgs_vartable_folder, tool='DV', joint=False, get_info=True, acmg=acmg)
	gatkj_variants, gatkj_variants_info = get_truth_variants(decodetable=decodetable, variant_tables=wgs_vartable_folder, tool='GATK', joint=True, get_info=True, acmg=acmg)

	theoretical_pins = set()
	_, theoretical_pins_gatkj = pin_truth_variants(decodetable=decodetable, pooltable=pooltable, variant_tables=wgs_vartable_folder, matrix_size=10, tool='GATK')
	_, theoretical_pins_dv = pin_truth_variants(decodetable=decodetable, pooltable=pooltable, variant_tables=wgs_vartable_folder, matrix_size=10, tool='DV')
	
	# Define which variants are "true".
	# Union == low + high confidence
	# Intersection == high confidence only
	true_variants.update(dv_variants.union(gatkj_variants))
	theoretical_pins.update(theoretical_pins_dv.union(theoretical_pins_gatkj))

	_, pool_pins = pin_pool_variants(decodetable=decodetable, pooltable=pooltable, variant_tables=pool_vartables, matrix_size=10, tool='GATKGVCF')
	make_dataset(true_variants=true_variants, theoretical_pins=theoretical_pins, pool_pins=pool_pins, acmg=acmg, vartable_folder=pool_vartables, pooltable=pooltable, output_table=output_table)
