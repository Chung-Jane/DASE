# DASE
Data strorage format Analysis for ML System Efficiency

## Setup Instructions

### 1. Create Python virtual environment
Make sure Python version is **>= 3.7**.

Using conda:
```bash
conda create -n dase python=3.7
```
Activate the created virtual environment:
```bash
conda activate dase
```

### 2. Install required libraries
```bash
conda install pandas scikit-learn xgboost os matplotlib seaborn
```

### 3. Dataset
https://www.kaggle.com/datasets/supergus/multistage-continuousflow-manufacturing-process/data

## Analysis

### 1. SaveFiles.py
Save the raw data in each of the five storage formats: csv, json, parquet, feather, h5 with basic dataset and large-scale dataset

### 2. Experiment.py
Experiment for each dataset type

### 3. Plot.py
- `Data loading time (sec)`
- `Training time (sec)`
- `Inference time (sec)`
- `Write time (sec)`
- `Total execution time (sec)`
- `Maximum memory usage (MB)`
- `File size (KB)`
- `RMSE`

## Citation
TBD
