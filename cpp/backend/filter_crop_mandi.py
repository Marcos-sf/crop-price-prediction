import pandas as pd

# List available crops
crops = ["arecanut", "coconut"]

print("Available crops:")
for i, crop in enumerate(crops, 1):
    print(f"{i}. {crop.title()}")
crop_choice = input("Enter the crop name (arecanut/coconut): ").strip().lower()
if crop_choice not in crops:
    print("Invalid crop name. Exiting.")
    exit(1)

mandi_name = input("Enter the mandi/market name to filter by (case-insensitive): ").strip()

input_path = f"app/data/processed/{crop_choice}_filtered.csv"
df = pd.read_csv(input_path)
filtered = df[df['Market'].str.lower() == mandi_name.lower()]
output_path = f"app/data/processed/{crop_choice}_{mandi_name.lower()}_for_training.csv"
filtered.to_csv(output_path, index=False)
print(f"Saved {len(filtered)} rows for {crop_choice.title()} in {mandi_name} to {output_path}") 