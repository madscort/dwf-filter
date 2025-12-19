## Filtering false positive variants from two-dimensional overlapped pooled sequencing data

This repository contains code and models used in the paper: *A framework for sensitive detection of individual genetic variation in pooled sequencing data*.

### Installation
**1. In a clean folder - clone this repository:**
```Bash
git clone https://github.com/madscort/dwf-filter/ .
```

**2. Install dependencies using the [conda](https://docs.conda.io/en/latest/) package manager:**

```Bash
conda env create --name dwf --file environment.yaml
conda activate dwf
```
### Run filter model on minimal test-data

```Bash
src/05_dwf_filter.py \
    --input data/test_vcf/*.vcf \
    --snv-model data/model_instances/logistic_regression_snv_model.joblib \
    --indel-model data/model_instances/random_forest_indel_model.joblib \
    --threshold-type sensitivity
```
This will output four filtered VCF files in the current directory.
Each of them will contain two new annotations in their info column:
```ML_PROB``` indicating the variants predicted probability of being a true variant. ```ML_PRED``` indicating the
predicted status of the variant as either 1: true positive or 0: false positive based on the defined threshold.

### Simulation study

Our tool [`dwf-sim`](https://github.com/madscort/dwf-sim) was used for simulating DoBSeq matrices and the subsequent benchmarking.

It is included in the environment. As a test run:

```Bash
dwf-sim test
```

The study was performed in the following steps:

1. Phased high-coverage 1000 Genomes Project VCF files, population codes and pedigree info was downloaded from [**EBI**](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV).

2. The files were subset to illumina trushight hereditary target panel using the script: `src/07_subset_1KGP.sh`

3. Individuals for the matrices were sampled using the following `dwf-sim` command:

```Bash
dwf-sim sample-1kg \
  --vcf data/all.subset.noSV.sorted.vcf.gz \
  --sample-table data/sample_population_codes.tsv \
  --pedigree data/1kGP.3202_samples.pedigree_info.txt \
  --matrix-size {2,5,10,24} \
  --non-eur-fraction {0.0,0.45} \ # 0.45 for size=24
  --reference data/gatk-resource-bundle/hg38/v0/Homo_sapiens_assembly38.fasta \
  --output-dir 1kg 
```
4. All matrix simulations and coverages was done using dwf-sim wrapped in the following script: `src/08_simulate.sh` 

5. After running DoBSeqWF on simulation data, benchmarking was done using dwf-sim through the following script: `src/09_simulation_benchmark.sh` 

### Repository contents
```Bash
dwf-filter
├── models/                                      # Saved model instances
├── configs/                                     # Model training configurations
├── data/                                        # Test and plot data
├── src/                                         #
│   ├── 01_variant_calling_benchmark.py          # Script for re-producing table 1.
│   ├── 02_make_dataset.py                       # Creates labelled dataset for modelling
│   ├── 03_train_model.py                        # Train and validates a model using repeated nested CV.
│   ├── 04_predict_model.py                      # Re-trains on entire training set and predicts on held-out dataset.
│   ├── 05_dwf_filter.py                         # Script for running saved model directly on VCF files.
│   ├── 06a_variant_filter_benchmark.py          # Script for re-producing table 2 and data for figure 4.
│   ├── 06b_variant_filter_benchmark.py          # Script for re-producing data for figure 5.
│   ├── 07_subset_1KGP.sh                        # Script for subsetting 1000 Genomes Project VCFs.
│   ├── 08_simulate.sh                           # Script for simulating all matrix data in study.
│   ├── 09_simulation_benchmark.sh               # Script for benchmarking simulation data.
│   ├── utils.py                                 # Utility functions
│   ├── models/                                  # Model class
│   └── plot/                                    #
│       ├── plot_04_predict_model.R              # Script for re-producing figure 3.
│       ├── plot_06a_variant_filter_benchmark.R  # Script for re-producing figure 4.
│       ├── plot_06b_variant_filter_benchmark.R  # Script for re-producing figure 5.
│       └── plot_09_simulation_benchmark.R       # Script for re-producing figure 2.
├── LICENSE
├── environment.yaml
└── README.md
```

### Re-create figures
```Bash
Rscript src/plot/plot_04_predict_model.R
Rscript src/plot/plot_06a_variant_filter_benchmark.R
Rscript src/plot/plot_06b_variant_filter_benchmark.R
Rscript src/plot/plot_09_simulation_benchmark.R
```

