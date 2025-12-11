import threading

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from memory_profiler import memory_usage
    import warnings
    import time
    import os
    import psutil
    import threading
    warnings.filterwarnings('ignore')

    def cpu_monitoring():
        stop_event = threading.Event()
        cpu_usage = []

        def run():
            while not stop_event.is_set():
                cpu_usage.append(psutil.cpu_percent(interval=0.1))

        thread = threading.Thread(target=run)
        thread.start()
        return stop_event, cpu_usage, thread
    
    
    BASE_PATH = "./test_io/"
    os.makedirs(BASE_PATH, exist_ok=True)
    FORMATS = ['csv', 'json', 'parquet', 'feather', 'h5']

    RESULTS_PATH = "./results_basic/"
    # RESULTS_PATH = "./results_large/"
    os.makedirs(RESULTS_PATH, exist_ok=True)

    REPEAT = 10

    for repeat_num in range(1, REPEAT+1):

        # Save the results
        results = []

        for fmt in FORMATS:
            start_total_time = time.time()
            filepath = f"{BASE_PATH}raw_data.{fmt}"
            # filepath = f"{BASE_PATH}raw_data_large.{fmt}"

            def load_func():
                if fmt == 'csv':
                    return pd.read_csv(filepath)
                elif fmt == 'json':
                    return pd.read_json(filepath, orient='records', lines=True)
                elif fmt == 'parquet':
                    return pd.read_parquet(filepath)
                elif fmt == 'feather':
                    return pd.read_feather(filepath)
                elif fmt == 'h5':
                    return pd.read_hdf(filepath, key='data')

            start_load = time.time()
            mem_usage, df = memory_usage((load_func,), retval=True, max_usage=True)
            end_load = time.time()
            load_time = end_load - start_load


            # Delete columns where the proportion of 0 >= 30%
            cols_to_drop = []

            for i in range(len(df.columns[:])):
                col = df.values[:,i]
                zero_ratio = (col == 0).mean()  #ratio of 0
                if zero_ratio > 0.3:
                    cols_to_drop.append(i)

            processed_df = df.drop(df.columns[cols_to_drop], axis=1)

            y_columns = [col for col in processed_df.columns if 'Output' in col]
            y = processed_df[y_columns]

            X = processed_df.iloc[:, 0:41]
            #y = processed_df.iloc[:, 41:50]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            models = {
            'LR': LinearRegression(),
            'RF': RandomForestRegressor(),
            'XGB': MultiOutputRegressor(XGBRegressor(verbosity=0, learning_rate=0.1)),
            'ANN': MLPRegressor(max_iter=500)
        }

            for model_name, model in models.items():

                X_scaler = StandardScaler()
                X_train_scaled = X_scaler.fit_transform(X_train)
                X_test_scaled = X_scaler.transform(X_test)
                y_scaler = StandardScaler()
                y_train_scaled = y_scaler.fit_transform(y_train)
                y_test_scaled = y_scaler.transform(y_test)

                stop_event, cpu_usage_train, thread = cpu_monitoring()
                start_train = time.time()
                model.fit(X_train_scaled, y_train_scaled)
                end_train = time.time()
                stop_event.set()
                thread.join()
                cpu_train_avg = np.mean(cpu_usage_train)
                train_time = end_train - start_train

                stop_event, cpu_usage_pred, thread = cpu_monitoring()
                start_pred = time.time()
                y_pred_scaled = model.predict(X_test_scaled)
                end_pred = time.time()
                stop_event.set()
                thread.join()
                cpu_pred_avg = np.mean(cpu_usage_pred)
                pred_time = end_pred - start_pred

                y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
                y_test_original = y_scaler.inverse_transform(y_test_scaled)

                column_names = [f"pred_{i}" for i in range(y_pred_original.shape[1])]
                pred_df = pd.DataFrame(y_pred_original, columns=column_names)
                pred_filepath = f"{BASE_PATH}pred_result_{model_name}.{fmt}"
                # pred_filepath = f"{BASE_PATH}pred_result_{model_name}_large.{fmt}"

                start_write = time.time()
                if fmt == 'csv':
                    pred_df.to_csv(pred_filepath, index=False)
                elif fmt == 'json':
                    pred_df.to_json(pred_filepath, orient='records', lines=True)
                elif fmt == 'parquet':
                    pred_df.to_parquet(pred_filepath, index=False)
                elif fmt == 'feather':
                    pred_df.reset_index().to_feather(pred_filepath)
                elif fmt == 'h5':
                    pred_df.to_hdf(pred_filepath, key='data', mode='w', format='table')
                end_write = time.time()
                write_time = end_write - start_write

                file_size_KB = os.path.getsize(pred_filepath) / 1024  # KB

                mse = mean_squared_error(y_test_original, y_pred_original)
                rmse = np.sqrt(mse)

                end_total_time = time.time()
                total_time = end_total_time - start_total_time

                # Results
                results.append({
                    "model": model_name,
                    "format": fmt,
                    "load_time": load_time,
                    "max_memory": mem_usage,
                    "cpu_train_avg": cpu_train_avg,
                    "train_time": train_time,
                    "predict_time": pred_time,
                    "write_time": write_time,
                    "total_time": total_time,
                    "file_size": file_size_KB,
                    "RMSE": rmse,
                })

        results_df = pd.DataFrame(results)
        print(f"Experiment {repeat_num} done!")

        results_df.to_csv(f"{RESULTS_PATH}results_test{repeat_num}.csv", index=False)

