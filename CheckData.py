import pandas as pd
import pyarrow.parquet as pq
import pyarrow.feather as feather
import h5py

# =============================================================================
# path = "test_io\\raw_data.csv"
# with open(path, 'r') as f:
#     lines = f.readlines()
#     for line in lines[:3]:
#         print(line.strip())
# =============================================================================
        
# =============================================================================
# path = "test_io\\raw_data.json"
# with open(path, 'r') as f:
#     lines = f.readlines()
#     for line in lines[:2]:
#         print(line.strip())
# =============================================================================
       
# =============================================================================
# path = "test_io\\raw_data.parquet" 
# schema = pq.read_schema(path)
# print(schema)
# =============================================================================

# =============================================================================
# path = "test_io\\raw_data.feather"
# table = feather.read_table(path)
# print(table.schema)
# =============================================================================

path = "test_io\\raw_data.h5"
with h5py.File(path, 'r') as f:
    f.visititems(lambda name, obj: print(name, type(obj)))