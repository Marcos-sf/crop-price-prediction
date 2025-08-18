import pandas as pd

crops = ["arecanut", "coconut"]

for crop in crops:
    input_path = f"app/data/processed/{crop}_filtered.csv"
    df = pd.read_csv(input_path)
    mandi_counts = df['Market'].value_counts().head(6)
    print(f"Top 6 mandis for {crop.title()} (by record count):")
    for mandi, count in mandi_counts.items():
        print(f"  {mandi}: {count} records")
    print() 