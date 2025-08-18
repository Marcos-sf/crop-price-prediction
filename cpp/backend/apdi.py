import pandas as pd

crop = "arecanut"   # Change as needed
mandi = "sirsi"     # Change as needed

file_path = f"app/data/processed/{crop}_{mandi}_for_training.csv"
df = pd.read_csv(file_path)

# Find the correct modal price column
modal_price_col = None
for col in df.columns:
    if col.strip().lower() in ['modal_price', 'modal price (rs./quintal)']:
        modal_price_col = col
        break

if not modal_price_col:
    print("No modal price column found!")
else:
    print(f"Stats for {modal_price_col} in {file_path}:")
    print("Min:", df[modal_price_col].min())
    print("Max:", df[modal_price_col].max())
    print("Mean:", df[modal_price_col].mean())
    print("Count < 20000:", (df[modal_price_col] < 20000).sum())
    print("Count 20000-40000:", ((df[modal_price_col] >= 20000) & (df[modal_price_col] < 40000)).sum())
    print("Count >= 40000:", (df[modal_price_col] >= 40000).sum())
    print("Total rows:", len(df))