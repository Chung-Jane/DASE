import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Repetition (results_test1.csv, results_test2.csv, ...)
all_files = glob.glob("results_basic/results_test*.csv")
# all_files = glob.glob("results_large/results_test*.csv")
df_list = [pd.read_csv(f) for f in all_files]
results_df = pd.concat(df_list, ignore_index=True)


# Data loading time
plt.figure(figsize=(6, 5))
ax = sns.barplot(data=results_df, x='format', y='load_time', errorbar="se", width=0.5, color='gray')
ax.set_axisbelow(True)
plt.xlabel('Data format', fontsize=15)
plt.ylabel('Data loading time (sec)', fontsize=15)
ax.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.grid()
plt.show()

# Maximum memory usage
plt.figure(figsize=(6, 5))
ax = sns.barplot(data=results_df, x='format', y='max_memory', errorbar="se", width=0.5, color='gray')
ax.set_axisbelow(True)
plt.xlabel('Data format', fontsize=15)
plt.ylabel('Maximum memory usage (MB)', fontsize=15)
ax.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.grid()
plt.show()

# CPU utilization
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=results_df, x='format', y='cpu_train_avg', errorbar="se", hue='model')
ax.set_axisbelow(True)
plt.xlabel('Data format', fontsize=15)
plt.ylabel('Avg. CPU utilization (%)', fontsize=15)
ax.tick_params(axis='both', labelsize=14)
plt.legend(title='Model', loc='upper right', title_fontsize=13, fontsize=12)
plt.tight_layout()
plt.grid()
plt.show()

# Training time
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=results_df, x='format', y='train_time', errorbar="se", hue='model')
ax.set_axisbelow(True)
plt.xlabel('Data format', fontsize=15)
plt.ylabel('Training time (sec)', fontsize=15)
ax.tick_params(axis='both', labelsize=14)
plt.legend(title='Model', loc='upper right', title_fontsize=13, fontsize=12)
plt.tight_layout()
plt.grid()
plt.show()

# Inference time
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=results_df, x='format', y='predict_time', errorbar="se", hue='model')
ax.set_axisbelow(True)
plt.xlabel('Data format', fontsize=15)
plt.ylabel('Inference time (sec)', fontsize=15)
ax.tick_params(axis='both', labelsize=14)
plt.legend(title='Model', loc='upper right', title_fontsize=13, fontsize=12)
plt.tight_layout()
plt.grid()
plt.show()

# Write time
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=results_df, x='format', y='write_time', errorbar="se", hue='model')
ax.set_axisbelow(True)
plt.xlabel('Data format', fontsize=15)
plt.ylabel('Write time (sec)', fontsize=15)
ax.tick_params(axis='both', labelsize=14)
plt.legend(title='Model', loc='upper right', title_fontsize=13, fontsize=12)
plt.tight_layout()
plt.grid()
plt.show()

# Total execution time
df_list = []

for f in all_files:
    df = pd.read_csv(f)
    
    # format & model
    format_order = ['csv', 'json', 'parquet', 'feather', 'h5']
    model_order = ['LR', 'RF', 'XGB', 'ANN']
    df['format'] = pd.Categorical(df['format'], categories=format_order, ordered=True)
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    df = df.sort_values(by=['format', 'model'])
    
    # accumulated total_time -> exec_time
    df['exec_time'] = df.groupby('format')['total_time'].diff().fillna(df['total_time'])
    
    df_list.append(df)

results_df = pd.concat(df_list, ignore_index=True)

plt.figure(figsize=(8, 5))
ax = sns.barplot(data=results_df, x='format', y='exec_time', errorbar="se", hue='model')
ax.set_axisbelow(True)
plt.xlabel('Data format', fontsize=15)
plt.ylabel('Total execution time (sec)', fontsize=15)
ax.tick_params(axis='both', labelsize=14)
plt.legend(title='Model', loc='upper right', title_fontsize=13, fontsize=12)
plt.tight_layout()
plt.grid()
plt.show()

# File size 
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=results_df, x='format', y='file_size', errorbar="se", hue='model')
ax.set_axisbelow(True)
plt.xlabel('Data format', fontsize=15)
plt.ylabel('File size (KB)', fontsize=15)
ax.tick_params(axis='both', labelsize=14)
plt.legend(title='Model', loc='upper right', title_fontsize=13, fontsize=12)
plt.tight_layout()
plt.grid()
plt.show()

# RMSE
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=results_df, x='format', y='RMSE', errorbar="se", hue='model')
ax.set_axisbelow(True)
plt.xlabel('Data format', fontsize=15)
plt.ylabel('RMSE', fontsize=15)
ax.tick_params(axis='both', labelsize=14)
plt.legend(title='Model', loc='upper right', title_fontsize=13, fontsize=12)
plt.tight_layout()
plt.grid()
plt.show()

