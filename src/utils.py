import sys
from pathlib import Path
import pandas as pd
from collections import namedtuple, Counter

Variant = namedtuple(
    "Variant",
    [
        "pool",
        "varid",
        "unique_varid",
        "CHROM",
        "POS",
        "REF",
        "ALT",
        "is_snv",
        "gene_target",
        "coding_uncertain",
        "gene_name",
        "annotation",
        "HGVSc",
        "HGVSp",
        "is_LOF",
        "LOF_gene",
        "CLNSIG",
        "is_P",
        "P_gene",
        "is_LOFP",
        "is_ACMG",
        "ACMG_category",
        "AC",
        "AD",
        "DP",
        "VAF",
    ],
)
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


def pin_truth_variants(
    decodetable: Path,
    pooltable: Path,
    variant_tables: Path,
    matrix_size: int,
    tool: str = "GATK",
) -> dict:
    """Returns a dictionary of sample specific variants."""

    samples = {}
    sample_variants = {}
    theoretical_pinpointables = set()
    variant_pools = {}
    sample_to_pools = {}

    with open(decodetable, "r") as fin:
        for line in fin:
            info = line.strip().split("\t")
            samples[info[0]] = []
            samples[info[0]].append(info[1])
            samples[info[0]].append(info[2])
            id, h, v = line.strip().split("\t")
            sample_to_pools[id] = (int(h.split("_")[1]) - 1, int(v.split("_")[1]) - 1)

    with open(pooltable, "r") as fin:
        for line in fin:
            pool = line.strip().split("\t")[0]
            pool_short_id = "_".join(pool.split("_")[1:3])
            variant_pools[pool_short_id] = set()

    for sample in samples:
        sample_variants[sample] = {}
        sample_variants[sample]["calls"] = set()
        sample_variants[sample]["unique"] = set()
        sample_variants[sample]["pinnable"] = set()
        with open(variant_tables / (sample + f".{tool}.tsv"), "r") as fin:
            header = fin.readline().strip().split("\t")

            if header[0:5] != ["CHROM", "POS", "ID", "REF", "ALT"]:
                raise ValueError("Variant table is in unexpected format.")
            for line in fin:
                var_info = line.strip().split("\t")
                CHROM, POS, _, _, ALT = var_info[0:5]
                if tool == "GATK":
                    CHROM, POS, _, REF, ALT, _, _, AC = var_info[0:8]
                elif tool == "DV":
                    CHROM, POS, _, REF, ALT, _, FILTER, _, GT = var_info[0:9]
                    if FILTER != "PASS":
                        continue
                else:
                    raise ValueError("Only GATK is supported for now.")

                varid = f"{CHROM}:{POS}:{REF}:{ALT}"
                sample_variants[sample]["calls"].add(varid)

                # ADD QC HERE:
                # DP = var_info[11]
                for pool in samples[sample]:
                    variant_pools[pool].add(varid)

    # Create lists of sets for storing variants in each pool
    horizontal_pools = [
        set() for _ in range(matrix_size)
    ]  # Variant sets for each horizontal pool
    vertical_pools = [
        set() for _ in range(matrix_size)
    ]  # Variant sets for each vertical pool

    for pool, variants in variant_pools.items():
        coor, idx = pool.split("_")[0:2]
        if coor.startswith("1"):
            horizontal_pools[int(idx) - 1].update(variants)
        elif coor.startswith("2"):
            vertical_pools[int(idx) - 1].update(variants)

    # Process each sample to identify unique variants

    for sample, (h_idx, v_idx) in sample_to_pools.items():
        # All pools except the current horizontal and vertical pool for the sample
        other_h_pools = set().union(
            *(horizontal_pools[i] for i in range(matrix_size) if i != h_idx)
        )
        other_v_pools = set().union(
            *(vertical_pools[i] for i in range(matrix_size) if i != v_idx)
        )

        # Unique in both dimensions
        unique_h_variants = horizontal_pools[h_idx].difference(other_h_pools)
        unique_v_variants = vertical_pools[v_idx].difference(other_v_pools)
        unique_pinnable = unique_h_variants.intersection(unique_v_variants)

        # Unique in one dimension but can appear in multiple in the other
        unique_one_dimension_h = unique_h_variants.intersection(vertical_pools[v_idx])
        unique_one_dimension_v = unique_v_variants.intersection(horizontal_pools[h_idx])
        all_pinnable = unique_one_dimension_h.union(unique_one_dimension_v)

        # Collecting results for each sample
        sample_variants[sample]["unique"] = unique_pinnable
        sample_variants[sample]["pinnable"] = all_pinnable

    # Return the dictionary of sample specific pinpointable variants
    # Return the dictionary of pool specific variants
    for sample in sample_variants:
        for var in sample_variants[sample]["unique"]:
            for pool in samples[sample]:
                theoretical_pinpointables.add(f"{pool}:{var}")

    return sample_variants, theoretical_pinpointables


def pin_pool_variants(
    decodetable: Path,
    pooltable: Path,
    variant_tables: Path,
    matrix_size: int,
    tool: str = "GATK",
):
    """Returns a dictionary of sample specific variants."""

    sample_to_pools = {}
    samples = {}
    pool_pinpointables = set()
    # make list of length matrix_size with empty strings
    horizontal_pool_table = ["" for _ in range(matrix_size)]
    vertical_pool_table = ["" for _ in range(matrix_size)]

    # Get pool ids:
    # Decodetable gives short_id and creates a tuple with the horizontal and vertical pool ids
    with open(decodetable, "r") as fin:
        for line in fin:
            id, h, v = line.strip().split("\t")
            samples[id] = [h, v]
            sample_to_pools[id] = (int(h.split("_")[1]) - 1, int(v.split("_")[1]) - 1)

    with open(pooltable, "r") as fin:
        for line in fin:
            # long_id contains four elements: batch, dimension, idx, and sample number.
            id = line.strip().split("\t")[0]
            if len(id.split(".")) > 1:
                id = id.split(".")[0]
            coor, idx = id.split("_")[1:3]
            if coor.startswith("1"):
                horizontal_pool_table[int(idx) - 1] = variant_tables / Path(
                    f"{id}.{tool}.tsv"
                )
            elif coor.startswith("2"):
                vertical_pool_table[int(idx) - 1] = variant_tables / Path(
                    f"{id}.{tool}.tsv"
                )

    # Create lists of sets for storing variants in each pool
    horizontal_pools = [
        set() for _ in range(matrix_size)
    ]  # Variant sets for each horizontal pool
    vertical_pools = [
        set() for _ in range(matrix_size)
    ]  # Variant sets for each vertical pool

    # Fill the variant sets from the variant tables
    for i in range(matrix_size):
        horizontal_pools[i].update(get_variants_from_pool(horizontal_pool_table[i]))
        vertical_pools[i].update(get_variants_from_pool(vertical_pool_table[i]))

    # Dictionary to store the identified variants per sample
    sample_variants = {}

    # Process each sample to identify unique variants
    for sample, (h_idx, v_idx) in sample_to_pools.items():
        # All pools except the current horizontal and vertical pool for the sample
        other_h_pools = set().union(
            *(horizontal_pools[i] for i in range(matrix_size) if i != h_idx)
        )
        other_v_pools = set().union(
            *(vertical_pools[i] for i in range(matrix_size) if i != v_idx)
        )

        # Unique in both dimensions
        unique_h_variants = horizontal_pools[h_idx].difference(other_h_pools)
        unique_v_variants = vertical_pools[v_idx].difference(other_v_pools)
        unique_pinnable = unique_h_variants.intersection(unique_v_variants)

        # Unique in one dimension but can appear in multiple in the other
        unique_one_dimension_h = unique_h_variants.intersection(vertical_pools[v_idx])
        unique_one_dimension_v = unique_v_variants.intersection(horizontal_pools[h_idx])
        all_pinnable = unique_one_dimension_h.union(unique_one_dimension_v)

        # Collecting results for each sample
        sample_variants[sample] = {"unique": unique_pinnable, "pinnable": all_pinnable}

        # Return the dictionary of sample specific pinpointable variants
    # Return the dictionary of pool specific variants
    for sample in sample_variants:
        for var in sample_variants[sample]["unique"]:
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


if __name__ == "__main__":
    pass
