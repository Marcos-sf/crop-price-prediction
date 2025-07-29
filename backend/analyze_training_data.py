import pandas as pd

crop = "arecanut"   # Change as needed
mandi = "sirsi"     # Change as needed

file_path = f"app/data/processed/{crop}_{mandi}_for_training.csv"
df = pd.read_csv(file_path)

# Try to find the correct modal price column
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
    print("Last 10 modal prices:", df[modal_price_col].tail(10).values)