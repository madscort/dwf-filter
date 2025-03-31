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
│   ├── 06_variant_filter_benchmark.py           # Script for re-producing table 2 and data for figure 3+4.
│   ├── utils.py                                 # Utility functions
│   ├── models/                                  # Model class
│   └── plot/                                    #
│       ├── plot_04_predict_model.R              # Script for re-producing figure 2.
│       ├── plot_06a_variant_filter_benchmark.R  # Script for re-producing figure 3.
│       └── plot_06b_variant_filter_benchmark.R  # Script for re-producing figure 4.
├── LICENSE
├── environment.yaml
└── README.md
```

### Re-create figures
```Bash
Rscript src/plot/plot_04_predict_model.R
Rscript src/plot/plot_06a_variant_filter_benchmark.R
Rscript src/plot/plot_06b_variant_filter_benchmark.R
```
