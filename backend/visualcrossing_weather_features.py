import requests
import pandas as pd
import os

VC_API_KEY = "GF9YU725Z3YXWYX9F49TKZEA7" #GF9YU725Z3YXWYX9F49TKZEA7
mandi_locations = {
    "sirsi": "sirsi",
    "yellapur": "yellapur",
    "siddapur": "siddapur",
    "shimoga": "shimoga",
    "sagar": "sagar",
    "kumta": "kumta",
    "bangalore": "bangalore",
    "arasikere": "arasikere",
    "channarayapatna": "channarayapatna",
    "ramanagara": "ramanagara",
    "sira": "sira",
    "tumkur": "tumkur"
}
crops = ["arecanut", "coconut"]

def fetch_visualcrossing_weather(location, start_date, end_date, unit_group="metric"):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_date}/{end_date}"
    params = {
        "unitGroup": unit_group,
        "key": VC_API_KEY,
        "contentType": "json"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    days = data["days"]
    weather_df = pd.DataFrame([{
        "date": d["datetime"],
        "temp": d.get("temp"),
        "humidity": d.get("humidity"),
        "precip": d.get("precip"),
        "windspeed": d.get("windspeed"),
        "conditions": d.get("conditions")
    } for d in days])
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    return weather_df

def enrich_all_with_weather():
    input_dir = "app/data/processed"
    for crop in crops:
        for mandi, location in mandi_locations.items():
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
                weather_df = fetch_visualcrossing_weather(location, start_date, end_date)
            except Exception as e:
                print(f"Error fetching weather for {mandi}: {e}")
                continue
            merged_df = pd.merge(price_df, weather_df, left_on=date_col, right_on="date", how="left")
            merged_df.to_csv(output_csv, index=False)
            print(f"Saved with weather features: {output_csv}")

if __name__ == "__main__":
    enrich_all_with_weather() 