import requests
import pandas as pd
from time import sleep
import os

OWM_API_KEY = "5a955c4200def2ed3234e4144393c036"

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

def fetch_weather(lat, lon, date):
    dt = int(pd.Timestamp(date).timestamp())
    url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
    params = {
        "lat": lat,
        "lon": lon,
        "dt": dt,
        "appid": OWM_API_KEY,
        "units": "metric"
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"Error fetching weather for {lat},{lon} on {date}: {resp.text}")
        return None, None, None
    data = resp.json()
    temp = data["current"]["temp"]
    humidity = data["current"]["humidity"]
    rain = data["current"].get("rain", {}).get("1h", 0)
    return temp, humidity, rain

def enrich_with_weather(input_csv, output_csv, mandi):
    df = pd.read_csv(input_csv)
    date_col = None
    for col in df.columns:
        if col.strip().lower() in ['date', 'arrival_date', 'price_date']:
            date_col = col
            break
    if not date_col:
        raise Exception("No date column found in data.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    lat, lon = mandi_coords[mandi.lower()]
    temps, humidities, rains = [], [], []
    for idx, row in df.iterrows():
        date_str = row[date_col].strftime("%Y-%m-%d")
        temp, humidity, rain = fetch_weather(lat, lon, date_str)
        temps.append(temp)
        humidities.append(humidity)
        rains.append(rain)
        print(f"{date_str} | {mandi} | Temp: {temp}, Humidity: {humidity}, Rain: {rain}")
        sleep(1)  # Avoid API rate limits
    df["weather_temp"] = temps
    df["weather_humidity"] = humidities
    df["weather_rain"] = rains
    df.to_csv(output_csv, index=False)
    print(f"Saved with weather features: {output_csv}")

if __name__ == "__main__":
    input_dir = "app/data/processed"
    for crop in crops:
        for mandi in mandi_coords.keys():
            input_csv = os.path.join(input_dir, f"{crop}_{mandi}_for_training.csv")
            output_csv = os.path.join(input_dir, f"{crop}_{mandi}_with_weather.csv")
            if os.path.exists(input_csv):
                print(f"Processing {input_csv}...")
                enrich_with_weather(input_csv, output_csv, mandi)
            else:
                print(f"File not found: {input_csv} (skipping)") 