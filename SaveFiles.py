import pandas as pd
import numpy as np
import os

def amplify_with_noise(df, n=5, noise_std=0.01):
    augmented_list = [df]
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for _ in range(n - 1):
        noisy = df.copy()
        for col in numeric_cols:
            if "Property" not in col:
                noise = np.random.normal(loc=0.0, scale=noise_std, size=df[col].shape)
                noisy[col] = df[col] + noise
        augmented_list.append(noisy)

    amplified_df = pd.concat(augmented_list, ignore_index=True)
    return amplified_df


# Data
# =============================================================================
# path = "continuous_factory_process.csv"
# raw_data = pd.read_csv(path, index_col="time_stamp")
# 
# measured_data = [col for col in raw_data.columns if "Setpoint" not in col]
# factory_process_data = raw_data[measured_data]
# factory_process_data.to_csv("factory_process_data.csv")
# =============================================================================    


df = pd.read_csv("factory_process_data.csv")
df = df.iloc[:, 1:57]
keywords = ['Measurement5', 'Measurement6', 'Measurement7', 'Measurement11', 'Measurement14']
df = df.drop(columns=[col for col in df.columns if any(keyword in col for keyword in keywords)])
df = df.drop(columns=[col for col in df.columns if 'Stage1.Output.Measurement1.U.Actual' in col])

BASE_PATH = "./test_io/"
os.makedirs(BASE_PATH, exist_ok=True)

# Save files
df.to_csv(f"{BASE_PATH}raw_data.csv", index=False)
df.to_json(f"{BASE_PATH}raw_data.json", orient='records', lines=True)
df.to_parquet(f"{BASE_PATH}raw_data.parquet", index=False)
df.reset_index().to_feather(f"{BASE_PATH}raw_data.feather")
df.to_hdf(f"{BASE_PATH}raw_data.h5", key='data', mode='w', format='table')

# # Large-scale dataset: 10-times amplified with noise
# df_large = amplify_with_noise(df, n=10, noise_std=0.01)
# df_large.to_csv(f"{BASE_PATH}raw_data_large.csv", index=False)
# df_large.to_json(f"{BASE_PATH}raw_data_large.json", orient='records', lines=True)
# df_large.to_parquet(f"{BASE_PATH}raw_data_large.parquet", index=False)
# df_large.reset_index().to_feather(f"{BASE_PATH}raw_data_large.feather")
# df_large.to_hdf(f"{BASE_PATH}raw_data_large.h5", key='data', mode='w', format='table')
