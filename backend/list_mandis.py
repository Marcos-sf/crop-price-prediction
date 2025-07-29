import pandas as pd

crops = ["arecanut", "coconut"]

for crop in crops:
    input_path = f"app/data/processed/{crop}_filtered.csv"
    df = pd.read_csv(input_path)
    mandis = df['Market'].dropna().unique()
    print(f"{crop.title()} - Available Mandis ({len(mandis)}):")
    print(", ".join(sorted(mandis)))
    print() 