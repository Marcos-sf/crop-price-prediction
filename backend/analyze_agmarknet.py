import pandas as pd
file_path = "app/data/processed/agmarknet_clean.csv"
df = pd.read_csv(file_path, nrows=20)
print("Columns:", list(df.columns))
print(df.head(10))