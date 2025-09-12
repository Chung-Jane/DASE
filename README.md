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

## 3. Dataset
Open dataset
https://www.kaggle.com/datasets/supergus/multistage-continuousflow-manufacturing-process/data

## Analysis

### 1. SaveFiles.py
Before running the simulation, you can modify the hyperparameters in the parameter file:
```bash
param_file_name = "./BSSSim/sim_param/ParamSettingTemplate.csv"
```

### 2. Run simulation example
Open and run the example script:
```bash
python RunExample3_Simulation.py
```

### 3. Output
Simulation statistics will be generated as CSV files including:
- `Avg_Num_Exchange_Per_Hour.csv` : Number of battery exchanges
- `Avg_Station_Operation_Cost_Per_Hour.csv` : Station operating costs
- `Avg_Alarm_To_Exchange_Time_Per_Excange.csv` : Total battery exchange time
- `Avg_Battery_Move_Distance_Per_Exchange.csv` : Battery exchange travel distance

## Citation
TBD
