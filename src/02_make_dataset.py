from pathlib import Path
from utils import get_truth_variants, pin_truth_variants, pin_pool_variants
import sys

annotations = [
	"ASSEMBLED_HAPS","AS_BaseQRankSum","AS_FS","AS_MQ","AS_MQRankSum",
	"AS_QD","AS_ReadPosRankSum","AS_SOR","ClippingRankSum","HAPCOMP","HAPDOM",
	"HEC","LikelihoodRankSum","MLEAC","MLEAF","X_GCC","X_HIL"]

def make_dataset(true_variants:set, pooltable:Path, theoretical_pins:set, pool_pins:set, acmg:set, vartable_folder:Path, output_table:Path, tool:str='GATKGVCF'):
	pools = []
	with open(pooltable, "r") as fin:
		for line in fin:
			pools.append(line.strip().split("\t")[0])
	
	with open(output_table, 'w') as fout:
		print(
			"gene_target",
			"variant_id",
			"unique_variant_id",
			"label",
			"is_pool_pin",
			"is_wgs_theoretical_pin",
			"is_lof",
			"is_p",
			"is_lofp",
			"is_acmg",
			"is_snv",
			"CHROM",
			"POS",
			"REF",
			"ALT",
			"AD_sample",
			"DP_site",
			"VAF",
			"dDP",
			"GQ_sample",
			"\t".join(annotations),
			sep="\t",
			file=fout)
		for pool in pools:
			with open(vartable_folder / (pool + f".{tool}.tsv"), 'r' ) as fin:
				header = fin.readline().strip().split('\t')

				# Confirm that the header is as expected
				if header[0:5] != ['CHROM', 'POS', 'ID', 'REF', 'ALT']:
					raise ValueError("Variant table is in unexpected format.")

				# Specific columns to include
				ANN_idx = header.index('ANN')
				GENE_idx = header.index('GENE')
				LOF_idx = header.index('LOF')
				CLNSIG_idx = header.index('CLNSIG')
				CLN_GENE_idx = header.index('GENEINFO')
				DP_site_idx = header.index('DP')
				
				for idx,element in enumerate(header):
					if '.AD' in element:
						AD_idx = idx
					if '.DP' in element:
						DP_idx = idx
					if '.GQ' in element:
						GQ_idx = idx

				# Include all columns except the ones in not_include, get idx for rest
				annotation_idx = {}
				for annotation in annotations:
					annotation_idx[annotation] = header.index(annotation)

				for line in fin:
					var_info = line.strip().split('\t')

					# Basic info
					CHROM,POS,_,REF,ALT = var_info[0:5]
					variant_id = f"{CHROM}:{POS}:{REF}:{ALT}" 
					is_snv = 1
					if ALT == '*' or REF == '*':
						continue
					if len(ALT) > 1 or len(REF) > 1:
						is_snv = 0

					# Target region ID
					gene_target = var_info[GENE_idx]
					
					# Functional annotation:
					effects = var_info[ANN_idx].split(',')
					if effects == ['NA']:
						gene_name = 'NA'
					else:
						gene_name = ",".join([effect.strip().split('|')[3] for effect in effects])

					is_lof = 0
					lof_gene = 'NA'
					lof = var_info[LOF_idx]
					if lof != 'NA':
						is_lof = 1
						lof_gene = var_info[LOF_idx].split('|')[0].replace('(','')

					# ClinVar annotation:
					is_p = 0
					clnsig = var_info[CLNSIG_idx].strip()
					cln_gene = var_info[CLN_GENE_idx].split(':')[0].strip()
		
					if "Pathogenic" in clnsig or "Likely_pathogenic" in clnsig:
						is_p = 1
					
					is_lofp = 0
					if is_p or is_lof:
						is_lofp = 1

					# ACMG state
					is_acmg = 0
					for g in gene_name.split(','):
						if g in acmg:
							is_acmg = 1
					
					if lof_gene in acmg:
						is_acmg = 1

					if gene_target in acmg:
						is_acmg = 1
					
					if cln_gene in acmg:
						is_acmg = 1
					
					DP_site = var_info[DP_site_idx].strip()
					
					AD = var_info[AD_idx].strip()
					DP = var_info[DP_idx].strip()
					GQ = var_info[GQ_idx].strip()
					if AD.count(',') > 1:
						print(var_info)
						print('Multi-allelic sites are not split')
						sys.exit(1)
					AD = AD.split(',')[1].strip()

					pool_short_id = "_".join(pool.split("_")[1:3])
					unique_variant_id = f"{pool_short_id}:{CHROM}:{POS}:{REF}:{ALT}"

					feature_values = []
					for annotation in annotations:
							feature_values.append(var_info[annotation_idx[annotation]])
					
					if unique_variant_id in theoretical_pins:
						is_theo_pin = 1
					else:
						is_theo_pin = 0
					if unique_variant_id in pool_pins:
						is_pool_pin = 1
					else:
						is_pool_pin = 0
					if unique_variant_id in true_variants:
						label = 1
					else:
						label = 0
						
					print(
						gene_target,
						variant_id,
						unique_variant_id,
						label,
						is_pool_pin,
						is_theo_pin,
						is_lof,
						is_p,
						is_lofp,
						is_acmg,
						is_snv,
						CHROM,
						POS,
						REF,
						ALT,
						AD,
						DP_site,
						round(int(AD)/int(DP_site),6),
						DP_site - DP,
						GQ,
						"\t".join(feature_values),
						sep="\t",
						file=fout)

if __name__ == '__main__':
	experiment = 'E2'
	pool_vartables = Path("/ngc/projects2/dp_00005/data/dwf/projects/variant_filter/data/E2_pool/gatk_jointgenotyping/DoBSeqWF/results_non_lenient/variant_tables")
	pooltable = Path("/ngc/projects2/dp_00005/data/dwf/projects/variant_filter/data/E2_pool/gatk_jointgenotyping/DoBSeqWF/cramtable.tsv")
	decodetable = Path("/ngc/projects2/dp_00005/data/dwf/data/pilot_E2/bam/decodetable_E2.tsv")
	wgs_vartable_folder = Path("/ngc/projects2/dp_00005/data/dwf/projects/variant_filter/data//E2_wgs/gatk_hc/DoBSeqWF/results/variant_tables/")
	output_table = Path(f"/ngc/projects2/dp_00005/data/dwf/projects/variant_filter/repo/output/{experiment}/dataset.tsv")
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
