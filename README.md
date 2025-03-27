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
dwf-filer
├── models/                              # Saved model instances
├── configs/                             # Model training configurations
├── data/                                # Test data
├── src/                                 #
│   ├── 01_variant_calling_benchmark.py  # 
│   ├── 02_make_dataset.py               #
│   ├── 03_train_model.py                #
│   ├── 04_predict_model.py              #
│   ├── 05_dwf_filter.py                 #
│   ├── 06_variant_filter_benchmark.py   #
│   └── X.py                             #
├── LICENSE
├── environment.yaml
└── README.md
```
