import requests
import pandas as pd
import os
from time import sleep

# === CONFIGURATION ===
RAPIDAPI_KEY = "7dd55778dbmsh3010b41435f8157p12ebd1jsna63949f015e1"
mandi_coords = {
    "sirsi": (14.6197, 74.8354),
    "yellapur": (14.9621, 74.7097),
    "siddapur": (14.3432, 74.8945),
    "shimoga": (13.9299, 75.5681),
    "sagar": (14.1667, 75.0333),
    "kumta": (14.4251, 74.4189),
    "bangalore": (12.9716, 77.5946),
    "arasikere": (13.3132, 76.2577),
    "channarayapatna": (12.9066, 76.3886),
    "ramanagara": (12.7217, 77.2812),
    "sira": (13.7411, 76.9042),
    "tumkur": (13.3409, 77.1017)
}
crops = ["arecanut", "coconut"]
input_dir = "app/data/processed"

def fetch_meteostat_weather(lat, lon, start, end):
    url = f"https://meteostat.p.rapidapi.com/point/daily?lat={lat}&lon={lon}&start={start}&end={end}"
    headers = {
        "x-rapidapi-host": "meteostat.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data:
        raise Exception(f"No weather data returned: {data}")
    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["date"])
    return df

def enrich_all_with_weather():
    for crop in crops:
        for mandi, (lat, lon) in mandi_coords.items():
            input_csv = os.path.join(input_dir, f"{crop}_{mandi}_for_training.csv")
            output_csv = os.path.join(input_dir, f"{crop}_{mandi}_with_weather.csv")
            if not os.path.exists(input_csv):
                print(f"File not found: {input_csv} (skipping)")
                continue
            print(f"Processing {input_csv}...")
            price_df = pd.read_csv(input_csv)
            date_col = None
            for col in price_df.columns:
                if col.strip().lower() in ['date', 'arrival_date', 'price_date']:
                    date_col = col
                    break
            if not date_col:
                print(f"No date column found in {input_csv}, skipping.")
                continue
            price_df[date_col] = pd.to_datetime(price_df[date_col])
            start_date = price_df[date_col].min().strftime("%Y-%m-%d")
            end_date = price_df[date_col].max().strftime("%Y-%m-%d")
            try:
                weather_df = fetch_meteostat_weather(lat, lon, start_date, end_date)
            except Exception as e:
                print(f"Error fetching weather for {mandi}: {e}")
                continue
            merged_df = pd.merge(price_df, weather_df, left_on=date_col, right_on="date", how="left")
            merged_df.to_csv(output_csv, index=False)
            print(f"Saved with weather features: {output_csv}")
            sleep(1)  # To avoid hitting API rate limits

if __name__ == "__main__":
    enrich_all_with_weather()