import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
import pandas as pd
from collections import namedtuple, Counter
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

S_idx = namedtuple("S_idx", ["AD", "DP", "GT"])

all_features = [
    "AD_sample", "DP_site", "VAF", "GQ_sample",
    "ASSEMBLED_HAPS", "AS_BaseQRankSum", "AS_FS", "AS_MQ", "AS_MQRankSum",
    "AS_QD", "AS_ReadPosRankSum", "AS_SOR", "ClippingRankSum", "HAPCOMP", 
    "HAPDOM", "HEC", "LikelihoodRankSum", "MLEAC", "MLEAF", "X_GCC", "X_HIL", "dDP"
]

broad = ["AS_FS", "AS_SOR", "AS_ReadPosRankSum", "AS_MQRankSum", "AS_QD", "AS_MQ"]

normal_loose = [
    "ASSEMBLED_HAPS", "AS_BaseQRankSum", "AS_ReadPosRankSum", "AS_SOR",
    "ClippingRankSum", "DP_site", "HAPCOMP", "HAPDOM", "HEC",
    "LikelihoodRankSum", "X_GCC"
]

normal_strict = [
    "AS_BaseQRankSum", "AS_ReadPosRankSum", "AS_SOR", "DP_site",
    "HAPDOM", "HEC", "LikelihoodRankSum", "X_GCC"
]

mrmr10snv = [
    "AS_SOR", "AS_MQRankSum", "HAPCOMP", "LikelihoodRankSum",
    "AS_ReadPosRankSum", "AS_FS", "AS_MQ", "VAF", "DP_site", "MLEAF"
]

mrmr5snv = [
    "AS_SOR", "AS_MQRankSum", "HAPCOMP", "LikelihoodRankSum", "AS_ReadPosRankSum"
]

mrmr10indel = [
    "MLEAF", "X_HIL", "VAF", "MLEAC", "AS_QD", "AD_sample",
    "AS_FS", "dDP", "GQ_sample", "AS_MQRankSum"
]

mrmr5indel = ["MLEAF", "X_HIL", "VAF", "MLEAC", "AS_QD"]

mrmr10 = [
    "HAPCOMP", "dDP", "DP_site", "X_HIL", "AS_SOR", "MLEAF",
    "LikelihoodRankSum", "AS_MQRankSum", "VAF", "ASSEMBLED_HAPS"
]

mrmr5 = ["HAPCOMP", "dDP", "DP_site", "X_HIL", "AS_SOR"]

feature_subsets_snv = {
    "all": all_features,
    "mrmr10": mrmr10snv,
    "mrmr5": mrmr5snv,
    "broad": broad,
    "normal_loose": normal_loose,
    "normal_strict": normal_strict
}

feature_subsets_indel = {
    "all": all_features,
    "mrmr10": mrmr10indel,
    "mrmr5": mrmr5indel,
    "broad": broad,
    "normal_loose": normal_loose,
    "normal_strict": normal_strict
}

feature_subsets_all = {
    "all": all_features,
    "mrmr10": mrmr10,
    "mrmr5": mrmr5,
    "broad": broad,
    "normal_loose": normal_loose,
    "normal_strict": normal_strict
}

def extract_features(dataset_path: Path, feature_subsets, skip_info: bool=True):
    """
    Extract features and prepare feature subsets from dataset.
    Returns features as a DataFrame with named columns (not numpy array).
    """
    dataset = pd.read_csv(dataset_path, sep='\t')

    # Extract features and labels
    meta_cols = [
        "variant_id", "unique_variant_id", "label",
        "is_snv", "CHROM", "POS", "REF", "ALT"
                ]

    feature_cols = [col for col in dataset.columns if col not in meta_cols]
    
    X = dataset[feature_cols]

    y = dataset['label'].values

    info = dataset[['variant_id', 'unique_variant_id', 'CHROM', 'POS',
                'REF', 'ALT', 'is_snv', 'AD_sample', 'DP_site', 'VAF']]

    return X, y, info, feature_subsets

def get_variants_from_pool(variant_table: Path, tool: str = "GATK") -> set:
    """
    Takes a path to a variant table and returns a set of variants.
    Each variant is represented as a string formatted as "CHROM:POS:ALT".

    Parameters:
    variant_table (Path): Path to a variant table.

    Returns:
    set: A set containing variants.
    """
    with open(variant_table, "r") as fin:
        header = fin.readline().strip().split("\t")
        # Check header format
        if header[0:5] != ["CHROM", "POS", "ID", "REF", "ALT"]:
            raise ValueError("Variant table is in unexpected format.")
        # Construct a set of variants with the format "CHROM:POS:ALT"
        variants = set()
        for line in fin:
            var_info = line.strip().split("\t")
            CHROM, POS, _, _, ALT = var_info[0:5]
            if tool == "GATK" or tool == "GATKGVCF":
                CHROM, POS, _, REF, ALT, _, _, AC = var_info[0:8]
            elif tool == "DV":
                CHROM, POS, _, REF, ALT, _, FILTER, _, GT = var_info[0:9]
                if FILTER != "PASS":
                    continue
            else:
                raise ValueError("Only GATK and DV is supported for now.")
            variants.add(f"{CHROM}:{POS}:{REF}:{ALT}")
    return variants


def pin_truth_variants(decodetable:Path, pooltable:Path, variant_tables:Path, matrix_size:int, tool:str='GATK', joint:bool=False, translate={}) -> dict:
	"""Returns a dictionary of sample specific variants."""

	samples = {}
	sample_variants = {}
	theoretical_pinpointables = set()
	variant_pools = {}
	sample_to_pools = {}

	with open(decodetable, "r") as fin:
		for line in fin:
			info = line.strip().split("\t")
			id, h, v = line.strip().split("\t")
			samples[id] = []
			samples[id].append(h.split("_")[3])
			samples[id].append(v.split("_")[3])
			
			sample_to_pools[id] = (int(h.split("_")[3][-2:])-1, int(v.split("_")[3][-2:])-1)
	
	with open(pooltable, "r") as fin:
		for line in fin:
			pool = line.strip().split("\t")[0]
			pool_short_id = pool.split("_")[3]
			variant_pools[pool_short_id] = set()
	
	for sample in samples:
		sample_variants[sample] = {}
		sample_variants[sample]['calls'] = set()
		sample_variants[sample]['unique'] = set()
		sample_variants[sample]['pinnable'] = set()
		if not joint:
			with open(variant_tables / (sample + f".{tool}.tsv"), 'r' ) as fin:
				header = fin.readline().strip().split('\t')

				if header[0:5] != ['CHROM', 'POS', 'ID', 'REF', 'ALT']:
					raise ValueError("Variant table is in unexpected format.")
				for line in fin:
					var_info = line.strip().split('\t')
					if tool == 'GATK':
						CHROM,POS,_,REF,ALT,_,_,AC = var_info[0:8]
					elif tool == 'DV':
						CHROM,POS,_,REF,ALT,_,FILTER,_,GT = var_info[0:9]
						if FILTER != 'PASS':
							continue
					else:
						raise ValueError("Only GATK is supported for now.")
					
					if ALT == '*' or REF == '*':
						continue
					varid = f"{CHROM}:{POS}:{REF}:{ALT}"
					sample_variants[sample]['calls'].add(varid)

					# ADD QC HERE:
					# DP = var_info[11]
					for pool in samples[sample]:
						variant_pools[pool].add(varid)
	if joint:
		with open(variant_tables / ('joint.GATK.tsv'), 'r' ) as fin:
			header = fin.readline().strip().split('\t')
			sample_idx = {}
			for sample in samples:
				if translate:
					sample_idx[sample] = S_idx(
						header.index(f"{translate[sample]}.AD"),
						header.index(f"{translate[sample]}.DP"),
						header.index(f"{translate[sample]}.GT")
					)
				else:
					sample_idx[sample] = S_idx(
						header.index(f"{sample}.AD"),
						header.index(f"{sample}.DP"),
						header.index(f"{sample}.GT")
					)
			
			# Confirm that the header is as expected
			if header[0:5] != ['CHROM', 'POS', 'ID', 'REF', 'ALT']:
				raise ValueError("Variant table is in unexpected format.")

			for line in fin:
				var_info = line.strip().split('\t')
				CHROM,POS,_,REF,ALT = var_info[0:5]
				if ALT == '*' or REF == '*':
					continue
				for sample in samples:
					AC = Counter(var_info[sample_idx[sample].GT].split('/'))[ALT]
					if AC == 0:
						continue
					else:
						varid = f"{CHROM}:{POS}:{REF}:{ALT}"
						sample_variants[sample]['calls'].add(varid)
						for pool in samples[sample]:
							if pool == '.':
								continue
							variant_pools[pool].add(varid)
	
	# Create lists of sets for storing variants in each pool
	horizontal_pools = [set() for _ in range(matrix_size)]  # Variant sets for each horizontal pool
	vertical_pools = [set() for _ in range(matrix_size)]  # Variant sets for each vertical pool
	
	for pool, variants in variant_pools.items():
		dim = pool[-3]
		idx = pool[-2:]
		if dim == 'H':
			horizontal_pools[int(idx)-1].update(variants)
		elif dim == 'V':
			vertical_pools[int(idx)-1].update(variants)
		else:
			sys.exit('ERROR')
	
	# Process each sample to identify unique variants
	
	for sample, (h_idx, v_idx) in sample_to_pools.items():

		# All pools except the current horizontal and vertical pool for the sample
		other_h_pools = set().union(*(horizontal_pools[i] for i in range(matrix_size) if i != h_idx))
		other_v_pools = set().union(*(vertical_pools[i] for i in range(matrix_size) if i != v_idx))
		
		# Unique in both dimensions
		unique_h_variants = horizontal_pools[h_idx].difference(other_h_pools)
		unique_v_variants = vertical_pools[v_idx].difference(other_v_pools)
		unique_pinnable = unique_h_variants.intersection(unique_v_variants)

		# Unique in one dimension but can appear in multiple in the other
		unique_one_dimension_h = unique_h_variants.intersection(vertical_pools[v_idx])
		unique_one_dimension_v = unique_v_variants.intersection(horizontal_pools[h_idx])
		all_pinnable = unique_one_dimension_h.union(unique_one_dimension_v)

		# Collecting results for each sample
		sample_variants[sample]['unique'] = unique_pinnable
		sample_variants[sample]['pinnable'] = all_pinnable

	# Return a dictionary of sample specific pinpointable variants
	# and a dictionary of pool specific variants
	for sample in sample_variants:
		for var in sample_variants[sample]['unique']:
			for pool in samples[sample]:
				theoretical_pinpointables.add(f"{pool}:{var}")

	return sample_variants, theoretical_pinpointables


def pin_pool_variants(decodetable:Path, pooltable:Path, variant_tables:Path, matrix_size:int, tool:str='GATK', vardict:dict="", naming='new'):
    """Returns a dictionary of sample specific variants."""	

    sample_to_pools = {}
    samples = {}
    pool_pinpointables = set()
    
    # Make list of length matrix_size with empty strings
    horizontal_pool_table = ["" for _ in range(matrix_size)]
    vertical_pool_table = ["" for _ in range(matrix_size)]

    # Get pool ids:
    # Decodetable gives short_id and creates a tuple with the horizontal and vertical pool ids
    with open(decodetable, "r") as fin:
        for line in fin:
            id, h, v = line.strip().split("\t")
            samples[id] = [h,v]
            sample_to_pools[id] = (int(h.split('_')[1])-1, int(v.split('_')[1])-1)

    if vardict == '':
        with open(pooltable, "r") as fin:
            for line in fin:
                if naming == 'new':
                    id = line.strip().split("\t")[0]
                    coor = id[-3]
                    idx = id[-2:]
                    if coor.startswith('H'):
                        horizontal_pool_table[int(idx)-1] = variant_tables / Path(f"{id}.{tool}.tsv")
                    elif coor.startswith('V'):
                        vertical_pool_table[int(idx)-1] = variant_tables / Path(f"{id}.{tool}.tsv")
                else:
                    id = line.strip().split("\t")[0]
                    if len(id.split('.')) > 1:
                        id = id.split('.')[0]
                    coor, idx = id.split('_')[1:3]
                    if coor.startswith('1'):
                        horizontal_pool_table[int(idx)-1] = variant_tables / Path(f"{id}.{tool}.tsv")
                    elif coor.startswith('2'):
                        vertical_pool_table[int(idx)-1] = variant_tables / Path(f"{id}.{tool}.tsv")

        # Create lists of sets for storing variants in each pool
        horizontal_pools = [set() for _ in range(matrix_size)]  # Variant sets for each horizontal pool
        vertical_pools = [set() for _ in range(matrix_size)]  # Variant sets for each vertical pool

        # Fill the variant sets from the variant tables
        for i in range(matrix_size):
            horizontal_pools[i].update(get_variants_from_pool(horizontal_pool_table[i]))
            vertical_pools[i].update(get_variants_from_pool(vertical_pool_table[i]))
    else:
        horizontal_pools = vardict['row']
        vertical_pools = vardict['column']

    # Dictionary to store the identified variants per sample
    sample_variants = {}

    # Process each sample to identify unique variants
    for sample, (h_idx, v_idx) in sample_to_pools.items():

        # All pools except the current horizontal and vertical pool for the sample
        other_h_pools = set().union(*(horizontal_pools[i] for i in range(matrix_size) if i != h_idx))
        other_v_pools = set().union(*(vertical_pools[i] for i in range(matrix_size) if i != v_idx))
        
        # Unique in both dimensions
        unique_h_variants = horizontal_pools[h_idx].difference(other_h_pools)
        unique_v_variants = vertical_pools[v_idx].difference(other_v_pools)
        unique_pinnable = unique_h_variants.intersection(unique_v_variants)

        # Unique in one dimension but can appear in multiple in the other
        unique_one_dimension_h = unique_h_variants.intersection(vertical_pools[v_idx])
        unique_one_dimension_v = unique_v_variants.intersection(horizontal_pools[h_idx])
        all_pinnable = unique_one_dimension_h.union(unique_one_dimension_v)

        # Collecting results for each sample
        sample_variants[sample] = {
            "unique": unique_pinnable,
            "pinnable": all_pinnable
        }

        # Return the dictionary of sample specific pinpointable variants
    # Return the dictionary of pool specific variants
    for sample in sample_variants:
        for var in sample_variants[sample]['unique']:
            for pool in samples[sample]:
                pool_pinpointables.add(f"{pool}:{var}")

    return sample_variants, pool_pinpointables

def get_pool_variants(
    pooltable: Path,
    variant_tables: Path,
    tool: str = "GATK",
    joint: bool = False,
    return_variant_format: str = "allele",
    variant_type: str = "all",
) -> set:
    """Get variants from variant tables and return a dictionary of variant pools."""

    pools = []
    variants = set()

    with open(pooltable, "r") as fin:
        for line in fin:
            pools.append(line.strip().split("\t")[0])

    if not joint:
        for pool in pools:
            with open(variant_tables / (pool + f".{tool}.tsv"), "r") as fin:
                header = fin.readline().strip().split("\t")

                # Confirm that the header is as expected
                if header[0:5] != ["CHROM", "POS", "ID", "REF", "ALT"]:
                    raise ValueError("Variant table is in unexpected format.")
                for line in fin:
                    var_info = line.strip().split("\t")

                    if tool == "GATK" or tool == "GATKGVCF":
                        CHROM, POS, _, REF, ALT, _, _, AC = var_info[0:8]
                    elif tool == "lofreq":
                        CHROM, POS, _, REF, ALT, _, _, _, _, _, DP4 = var_info[0:11]
                        DPsplit = DP4.split(",")
                        if (
                            int(DPsplit[2]) + int(DPsplit[3]) < 20
                        ):  # ALTERNATIVE SUPPORT
                            continue
                    elif tool == "octopus":
                        CHROM, POS, _, REF, ALT, _, _, AC = var_info[0:8]
                    else:
                        raise ValueError("Only GATK is supported for now.")

                    pool_short_id = "_".join(pool.split("_")[1:3])

                    if variant_type == "SNV":
                        if len(REF) > 1 or len(ALT) > 1:
                            continue
                    if variant_type == "INDEL":
                        if len(REF) == 1 and len(ALT) == 1:
                            continue

                    if return_variant_format == "genotype":
                        variant_id = f"{pool_short_id}:{CHROM}:{POS}:{REF}:{ALT}:{AC}"
                    elif return_variant_format == "allele":
                        variant_id = f"{pool_short_id}:{CHROM}:{POS}:{REF}:{ALT}"
                    elif return_variant_format == "site":
                        variant_id = f"{pool_short_id}:{CHROM}:{POS}:{REF}"
                    else:
                        raise ValueError(
                            "return_variant_format must be one of 'genotype', 'allele' or 'site'."
                        )
                    variants.add(variant_id)
    elif joint and tool == "CRISP":
        with open(variant_tables / ("crisp.tsv"), "r") as fin:
            for line in fin:
                var_info = line.strip().split("\t")
                CHROM, POS, REF, ALT = var_info[0:4]
                ALTs = ALT.split(",")
                samples = var_info[4:]

                for sample in samples:
                    pool = sample.split("=")[0]
                    ACs = sample.split("=")[1].split(",")
                    for n, ALT in enumerate(ALTs):
                        AC = int(ACs[n])
                        if AC == 0:
                            continue

                        if variant_type == "SNV":
                            if len(REF) > 1 or len(ALT) > 1:
                                continue
                        if variant_type == "INDEL":
                            if len(REF) == 1 and len(ALT) == 1:
                                continue

                        else:
                            pool_short_id = "_".join(pool.split("_")[1:3])
                            if return_variant_format == "genotype":
                                variant_id = (
                                    f"{pool_short_id}:{CHROM}:{POS}:{REF}:{ALT}:{AC}"
                                )
                            elif return_variant_format == "allele":
                                variant_id = (
                                    f"{pool_short_id}:{CHROM}:{POS}:{REF}:{ALT}"
                                )
                            elif return_variant_format == "site":
                                variant_id = f"{pool_short_id}:{CHROM}:{POS}:{REF}"
                            else:
                                raise ValueError(
                                    "return_variant_format must be one of 'genotype', 'allele' or 'site'."
                                )
                            variants.add(variant_id)
    elif joint and tool == "GATK":
        with open(variant_tables / ("joint.GATK.tsv"), "r") as fin:
            header = fin.readline().strip().split("\t")
            DP_site_idx = header.index("DP")

            pool_idx = {}
            for sample in pools:
                pool_idx[sample] = S_idx(
                    header.index(f"{sample}.AD"),
                    header.index(f"{sample}.DP"),
                    header.index(f"{sample}.GT"),
                )

            # Confirm that the header is as expected
            if header[0:5] != ["CHROM", "POS", "ID", "REF", "ALT"]:
                raise ValueError("Variant table is in unexpected format.")

            for line in fin:
                var_info = line.strip().split("\t")

                CHROM, POS, _, REF, ALT = var_info[0:5]
                if ALT == "*":
                    continue

                if variant_type == "SNV":
                    if len(REF) > 1 or len(ALT) > 1:
                        continue
                if variant_type == "INDEL":
                    if len(REF) == 1 and len(ALT) == 1:
                        continue

                DP_site = var_info[DP_site_idx].strip()
                for pool in pools:
                    AC = Counter(var_info[pool_idx[pool].GT].split("/"))[ALT]
                    if AC == 0:
                        continue
                    else:
                        DP = var_info[pool_idx[pool].DP].strip()
                        AD = var_info[pool_idx[pool].AD]
                        if AD.count(",") > 1:
                            print(var_info)
                            print(AD)
                            print("Multi-allelic sites are not split")
                            sys.exit(1)

                        AD = AD.split(",")[1].strip()
                        assert AD.isdigit()
                        assert DP.isdigit()
                        if int(DP) == 0:
                            VAF = 0
                        else:
                            VAF = int(AD) / int(DP)

                        pool_short_id = "_".join(pool.split("_")[1:3])
                        variant_id = f"{CHROM}:{POS}:{REF}:{ALT}"
                        unique_variant_id = f"{pool_short_id}:{CHROM}:{POS}:{REF}:{ALT}"
                        variants.add(unique_variant_id)
    else:
        raise ValueError("joint must be a boolean.")

    return variants

def get_truth_variants(
    decodetable: Path,
    variant_tables: Path,
    tool: str = "GATK",
    joint: bool = False,
) -> set:
    """Get variants from variant tables and return a dictionary of variant pools."""

    samples = {}
    variants = set()
    variants_info = {}

    # Decode table links pool and sample ids
    with open(decodetable, "r") as fin:
        for line in fin:
            info = line.strip().split("\t")
            samples[info[0]] = []
            samples[info[0]].append(info[1])
            samples[info[0]].append(info[2])

    if not joint:
        for sample in samples:
            with open(variant_tables / (sample + f".{tool}.tsv"), "r") as fin:
                header = fin.readline().strip().split("\t")

                for idx, element in enumerate(header):
                    if ".AD" in element:
                        AD_idx = idx
                    if ".DP" in element:
                        DP_sample_idx = idx
                
                if header[0:5] != ["CHROM", "POS", "ID", "REF", "ALT"]:
                    raise ValueError("Variant table is in unexpected format.")

                for line in fin:
                    var_info = line.strip().split("\t")

                    CHROM, POS, _, REF, ALT = var_info[0:5]
                    if ALT == "*" or REF == "*":
                        continue
                    is_snv = 1
                    if len(ALT) > 1 or len(REF) > 1:
                        is_snv = 0

                    DP_sample = var_info[DP_sample_idx].strip()
                    AD = var_info[AD_idx]
                    if AD.count(",") > 1:
                        print(var_info)
                        print("Multi-allelic sites are not split")
                        sys.exit(1)

                    AD = AD.split(",")[1].strip()
                    assert CHROM.startswith("chr")
                    assert POS.isdigit()
                    assert REF.isalpha()
                    assert ALT.isalpha()
                    assert AD.isdigit()
                    assert DP_sample.isdigit()

                    if tool == "GATK":
                        CHROM, POS, _, REF, ALT, _, _, AC = var_info[0:8]
                    elif tool == "DV":
                        CHROM, POS, _, REF, ALT, _, FILTER, _, GT = var_info[0:9]

                        AC = Counter(GT.split("/"))[ALT]
                        if FILTER != "PASS":
                            continue
                    else:
                        raise ValueError("Only GATK and DV is supported for now.")

                    for pool in samples[sample]:
                        if pool == ".":
                            continue
                        variant_id = f"{CHROM}:{POS}:{REF}:{ALT}"
                        unique_variant_id = f"{pool}:{CHROM}:{POS}:{REF}:{ALT}"
                        variants.add(unique_variant_id)
    elif joint:
        with open(variant_tables / ("joint.GATK.tsv"), "r") as fin:
            header = fin.readline().strip().split("\t")

            DP_site_idx = header.index("DP")

            sample_idx = {}
            for sample in samples:
                sample_idx[sample] = S_idx(
                    header.index(f"{sample}.AD"),
                    header.index(f"{sample}.DP"),
                    header.index(f"{sample}.GT"),
                )

            # Confirm that the header is as expected
            if header[0:5] != ["CHROM", "POS", "ID", "REF", "ALT"]:
                raise ValueError("Variant table is in unexpected format.")

            for line in fin:
                var_info = line.strip().split("\t")

                CHROM, POS, _, REF, ALT = var_info[0:5]
                if ALT == "*" or REF == "*":
                    continue
                is_snv = 1
                if len(ALT) > 1 or len(REF) > 1:
                    is_snv = 0

                DP_site = var_info[DP_site_idx].strip()

                assert CHROM.startswith("chr")
                assert POS.isdigit()
                assert REF.isalpha()
                assert ALT.isalpha()

                for sample in samples:
                    AC = Counter(var_info[sample_idx[sample].GT].split("/"))[ALT]
                    if AC == 0:
                        continue
                    else:
                        DP = var_info[sample_idx[sample].DP].strip()
                        AD = var_info[sample_idx[sample].AD]
                        if AD.count(",") > 1:
                            print(var_info)
                            print("Multi-allelic sites are not split")
                            sys.exit(1)

                        AD = AD.split(",")[1].strip()
                        assert AD.isdigit()
                        assert DP.isdigit()
                        if int(DP) == 0:
                            VAF = 0
                        else:
                            VAF = int(AD) / int(DP)

                        for pool in samples[sample]:
                            if pool == ".":
                                continue
                            variant_id = f"{CHROM}:{POS}:{REF}:{ALT}"
                            unique_variant_id = f"{pool}:{CHROM}:{POS}:{REF}:{ALT}"
                            variants.add(unique_variant_id)
    else:
        raise ValueError("joint must be a boolean.")
    return variants

def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculate performance metrics from TP, FP, FN counts."""
    metrics = {}
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['F1_score'] = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    metrics['FDR'] = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    metrics['total_variant_calls'] = tp + fp
    metrics['total_variants_in_truthset'] = tp + fn
    return metrics

def tabix_index(vcf_path: Path) -> None:
    """Index a bgzipped VCF with tabix."""
    vcf_path = Path(vcf_path)
    cmd = ["tabix", "-p", "vcf", str(vcf_path)]
    print(f"Indexing {vcf_path} ...")
    subprocess.run(cmd, check=True)

def normalize_vcf(
    input_vcf: Path,
    output_vcf: Path,
    reference: Path,
    bed: Path
    ) -> Path:
    """
    Normalize a VCF: split multi-allelic sites and remove duplicates.

    Args:
        input_vcf: input bgzipped VCF
        output_vcf: normalized bgzipped VCF

    Returns:
        Path to normalized, bgzipped, and tabix-indexed VCF
    """
    input_vcf = Path(input_vcf)
    output_vcf = Path(output_vcf)
    reference = Path(reference)
    
    if not Path(f"{input_vcf}.tbi").exists():
        tabix_index(input_vcf)
    

    cmd = [
        "bcftools", "norm",
        "-m", "-both",
        "-d", "exact",
        "-f", str(reference),
        "-R", str(bed),
        "--force",
        "-Oz",
        "-o", str(output_vcf),
        str(input_vcf)]

    subprocess.run(cmd, check=True)
    tabix_index(output_vcf)
    return output_vcf


def merge_vcfs(
    sample_ids: List[str],
    input_dir: Path,
    caller: str,
    output_vcf: Path,
    reference: Path,
    bed: Path
) -> Path:
    """
    Merge VCF files and keep unique variation.

    Args:
        sample_ids: list of sample IDs (expects {sample}.vcf.gz in input_dir)
        input_dir: directory containing VCFs
        output_vcf: final merged VCF.gz path
    """
    input_dir = Path(input_dir)
    output_vcf = Path(output_vcf)
    tmp_merged = output_vcf.with_suffix(".merged.tmp.vcf.gz")
    vcfs = [str(input_dir / f"{sid}.{caller}.vcf.gz") for sid in sample_ids]

    for vcf in vcfs:
        if not Path(f"{vcf}.tbi").exists():
            tabix_index(vcf)

    ## Merge
    merge_cmd = [
        "bcftools", "merge",
        "-m", "none",
        "--force-samples",
        "-f", "PASS",
        "-Oz",
        "-o", str(tmp_merged),
    ] + vcfs
    subprocess.run(merge_cmd, check=True)

    ## Remove duplicates and split multi-allelic
    normalize_vcf(tmp_merged, output_vcf, reference, bed)
    tmp_merged.unlink(missing_ok=True)
    tmp_merged.with_suffix('.tbi').unlink(missing_ok=True)
    return output_vcf

def subset_multisample_vcf(
    sample_ids: List[str],
    input_vcf: Path,
    output_vcf: Path,
    reference: Path,
    bed: Path,
    format_filter: str | None = None
) -> Path:
    """
    Extract a subset of samples from a multi-sample VCF

    Args:
        sample_ids: samples to keep
        input_vcf: input multi-sample VCF(.gz)
        output_vcf: output filtered VCF(.gz)
        reference: reference genome
        bed: BED file for regions
        format_filter: Optional FORMAT field to filter on (e.g., 'AC', 'AD', 'AF', 'VAF').
                      If None, uses -c 1 (default for GT-based VCFs).
                      If specified, filters on FORMAT/{field}>0

    Returns:
        Path to bgzipped, tabix-indexed VCF.
    """
    input_vcf = Path(input_vcf)
    output_vcf = Path(output_vcf)
    tmp_subset = output_vcf.with_suffix(".subset.tmp.vcf.gz")
    tmp_filtered = output_vcf.with_suffix(".filtered.tmp.vcf.gz")
    samples_arg = ",".join(sample_ids)

    # Subset samples
    subset_cmd = [
        "bcftools", "view",
        "-s", samples_arg,
        "-Oz",
        "-o", str(tmp_subset),
        str(input_vcf),
    ]
    subprocess.run(subset_cmd, check=True)
    tabix_index(tmp_subset)
    
    # Add filter: either -c 1 or FORMAT field filter
    filter_arg = "-c" if format_filter is None else "-i"
    filter_value = "1" if format_filter is None else f"FORMAT/{format_filter}>0"
    
    # Filter for only entries with actual calls
    filter_cmd = [
        "bcftools", "view",
        filter_arg, filter_value,
        "-Oz",
        "-o", str(tmp_filtered),
        str(tmp_subset),
    ]
    
    subprocess.run(filter_cmd, check=True)
    
    # Step 3: Normalize
    normalize_vcf(tmp_filtered, output_vcf, reference, bed)
    
    # Cleanup
    tmp_subset.unlink(missing_ok=True)
    tmp_subset.with_suffix('.tbi').unlink(missing_ok=True)
    tmp_filtered.unlink(missing_ok=True)
    
    return output_vcf


def intersect_vcfs_vcfeval(
    a_vcf: Path,
    b_vcf: Path,
    sdf_dir: Path,
    out_prefix: Path,
    singularity_img: Path | None = None,
    threads: int = 4,
    mem_gb: int = 8
) -> Path:
    """
    Intersect two VCFs using RTG vcfeval and return a bgzipped, indexed VCF
    containing the intersection (true positives).

    Args:
        vcf_a: first VCF(.gz)
        vcf_b: second VCF(.gz)
        sdf_dir: RTG SDF directory for the reference
        out_prefix: path to write intersection (.vcf.gz)
        singularity_img: optional path to rtgtools .sif to run via Singularity
        threads: vcfeval --threads
        mem_gb: Java heap for rtg (-Xmx)

    Returns:
        Path to bgzipped, tabix-indexed intersection VCF (tp.vcf.gz copy).
    """
    background = Path(b_vcf).resolve()
    calls = Path(a_vcf).resolve()
    sdf_dir = Path(sdf_dir)
    out_prefix = Path(out_prefix)

    with tempfile.TemporaryDirectory(prefix="vcfeval_tmp_") as tmp_root:
        tmp_root = Path(tmp_root)
        outdir = tmp_root / "vcfeval_out"
        # Build the vcfeval command (host or Singularity)
        base_cmd = []
        if singularity_img:
            base_cmd = ["singularity", "exec", "--cleanenv"]
            # Bind input paths
            for p in {tmp_root, Path.cwd(), background.parent, calls.parent, sdf_dir}:
                base_cmd += ["-B", f"{p}:{p}"]
            base_cmd += [str(singularity_img)]

        cmd = base_cmd + [
            "rtg", f"RTG_MEM={mem_gb}G", "vcfeval",
            "-t", str(sdf_dir),
            "-b", str(background),
            "-c", str(calls),
            "--squash-ploidy", "--sample", "ALT,ALT",
            "--threads", str(threads),
            "--output-mode", "split",
            "-o", str(outdir),
        ]

        subprocess.run(cmd, check=True)
        tp_o = out_prefix.with_suffix(".tp.vcf.gz")
        fp_o = out_prefix.with_suffix(".fp.vcf.gz")
        fn_o = out_prefix.with_suffix(".fn.vcf.gz")
        shutil.copyfile(outdir / "tp.vcf.gz", tp_o); tabix_index(tp_o)
        shutil.copyfile(outdir / "fp.vcf.gz", fp_o); tabix_index(fp_o)
        shutil.copyfile(outdir / "fn.vcf.gz", fn_o); tabix_index(fn_o)
    return {'both': tp_o, 'a_only': fp_o, 'b_only': fn_o}

def _wc_l(cmd: list[str]) -> int:
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=False)
    p2 = subprocess.Popen(["wc", "-l"], stdin=p1.stdout, stdout=subprocess.PIPE, text=True)
    p1.stdout.close()
    out, _ = p2.communicate()
    return int(out.strip()) if out and out.strip() else 0

def count_variant_types_bcftools(vcf: Path) -> Dict[str, int]:
    """
    Count variants by type using bcftools (no custom parsing).
    Returns: {'snv': N, 'indel': N, 'both': N}
    """
    vcf = Path(vcf)
    snv   = _wc_l(["bcftools", "view", "-H", "-v", "snps",   str(vcf)])
    indel = _wc_l(["bcftools", "view", "-H", "-v", "indels", str(vcf)])
    both  = _wc_l(["bcftools", "view", "-H",            str(vcf)])
    return {"snv": snv, "indel": indel, "both": both}

def call_state_by_type(tp: Path, fp: Path, fn: Path) -> Dict[str, Dict[str, int]]:
    """
    Count TP/FP/FN by type via bcftools (no custom parsing).
    Returns: {'snv': {'TP':..,'FP':..,'FN':..}, 'indel': {...}, 'both': {...}}
    """
    tp_c = count_variant_types_bcftools(tp)
    fp_c = count_variant_types_bcftools(fp)
    fn_c = count_variant_types_bcftools(fn)
    out = {}
    for k in ("snv","indel","both"):
        out[k] = {"TP": tp_c[k], "FP": fp_c[k], "FN": fn_c[k]}
    return out

def normalize_all_vcfs(
    vcf_dict: Dict[str, Path],
    output_dir: Path,
    reference: Path,
    bed: Optional[Path] = None,
    label: str = "vcf",
) -> Dict[str, Path]:
    """
    Normalize all VCFs in a dictionary.
    
    Args:
        vcf_dict: Dictionary mapping ID -> VCF path
        output_dir: Directory for normalized VCFs
        reference: Reference genome
        bed: Optional BED file
        label: Label for output files
    
    Returns:
        Dictionary mapping ID -> normalized VCF path
    """
    logger.info(f"Normalizing {len(vcf_dict)} {label} VCFs...")
    
    norm_dir = output_dir / f"normalized_{label}"
    norm_dir.mkdir(parents=True, exist_ok=True)
    
    normalized = {}
    for vcf_id, vcf_path in vcf_dict.items():
        logger.debug(f"  Normalizing {vcf_id}")
        norm_vcf = normalize_vcf(
            vcf_path,
            norm_dir / f"{vcf_id}_norm.vcf.gz",
            reference,
            target_bed=bed
        )
        normalized[vcf_id] = norm_vcf
    
    logger.info(f"Normalized {len(normalized)} {label} VCFs")
    return normalized

def find_private_variants(
    individual_id: str,
    individual_vcf: Path,
    other_vcfs: List[Path],
    output_dir: Path,
) -> Path:
    """
    Find variants private to an individual using bcftools isec.
    
    Assumes all VCFs are already normalized.
    
    Args:
        individual_id: ID of the individual
        individual_vcf: Normalized VCF for the individual
        other_vcfs: Normalized VCFs for all other individuals in their pools
        output_dir: Working directory
    
    Returns:
        Path to VCF with private variants
    """
    logger.debug(f"Finding private variants for {individual_id}")
    
    # Use bcftools isec to find variants unique to individual
    isec_dir = output_dir / individual_id
    isec_dir.mkdir(exist_ok=True)
    
    cmd = [
        "bcftools", "isec",
        "-n=1",  # Present in exactly 1 file
        "-w", "1",  # Write the first file's unique variants
        "-c", "none",
        "-p", str(isec_dir),
        "-Oz",
        "-W=tbi",
        str(individual_vcf)
    ] + [str(v) for v in other_vcfs]
    
    logger.debug(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"bcftools isec failed for {individual_id}")
        logger.error(f"stderr: {e.stderr}")
        raise
    
    # Output is in 0000.vcf.gz (first file's unique variants)
    private_vcf = isec_dir / "0000.vcf.gz"
    
    if not private_vcf.exists():
        logger.error(f"Private variants VCF not created: {private_vcf}")
        raise FileNotFoundError(f"bcftools isec did not create {private_vcf}")

    return private_vcf


if __name__ == "__main__":
    pass
