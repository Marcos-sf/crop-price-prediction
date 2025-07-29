import pandas as pd

# List of crops to filter
crops = ["arecanut", "coconut"]

# Prompt user for mandi/market name
mandi_name = input("Enter the mandi/market name to filter by (case-insensitive): ").strip()

for crop in crops:
    input_path = f"app/data/processed/{crop}_filtered.csv"
    df = pd.read_csv(input_path)
    filtered = df[df['Market'].str.lower() == mandi_name.lower()]
    output_path = f"app/data/processed/{crop}_{mandi_name.lower()}_filtered.csv"
    filtered.to_csv(output_path, index=False)
    print(f"Saved {len(filtered)} rows for {crop.title()} in {mandi_name} to {output_path}") 