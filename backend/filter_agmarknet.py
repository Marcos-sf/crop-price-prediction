import pandas as pd

# Load the cleaned Agmarknet data
input_path = "app/data/processed/agmarknet_clean.csv"
df = pd.read_csv(input_path)

# List of crops to filter
crops = ["Arecanut", "Coconut"]

for crop in crops:
    filtered = df[df['Commodity'].str.lower() == crop.lower()]
    output_path = f"app/data/processed/{crop.lower()}_filtered.csv"
    filtered.to_csv(output_path, index=False)
    print(f"Saved {len(filtered)} rows for {crop} to {output_path}") 