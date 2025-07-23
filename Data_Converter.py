import pandas as pd

df = pd.read_parquet('data/dataset.parquet')
print(df.head(10))
# df.to_csv('data/data_fraunhofer.csv', index=False)