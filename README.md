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
### Run filtering model on test-data

```Bash
./dwf_filter.py
  
```
This will output X


### Repository contents
```Bash
dwf-filter
├── models/                                      # Saved model instances
├── configs/                                     # Model training configurations
├── data/                                        # Test data
├── src/                                         #
│   ├── 01_variant_calling_benchmark.py          # Script for re-producing table 1.
│   ├── 02_make_dataset.py                       # Creates labelled dataset for modelling
│   ├── 03_train_model.py                        # Train and validates a model using repeated nested CV.
│   ├── 04_predict_model.py                      # Re-trains on entire training set and predicts on held-out dataset.
│   ├── 05_dwf_filter.py                         # Script for running saved model directly on VCF files.
│   ├── 06_variant_filter_benchmark.py           # Script for re-producing table 2.
│   ├── utils.py                                 # Utility functions
│   ├── models/                                  # Muti-model class
│   └── plot/                                    #
│       ├── plot_04_predict_model.R              # Script for re-producing figure 2.
│       └── plot_06_variant_filter_benchmark.R   # Script for re-producing figure 3+4.
├── LICENSE
├── environment.yaml
└── README.md
```
