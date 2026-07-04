"""
fetch_data.py — Download Evanston weather data from Open-Meteo.

Fetches:
  1. Ground-truth daily actuals (ERA5 reanalysis): max temp, solar radiation,
     sunshine duration  -> data/actuals.csv
  2. Day-ahead forecast archive (what the model predicted 24h out)
     -> data/forecasts.csv

Run this locally (Open-Meteo blocks some automated environments):
    pip install requests
    python scripts/fetch_data.py

Notes:
  - Coordinates target the Dempster/Forest area of Evanston. Open-Meteo
    interpolates from a ~9 km model grid, so this represents the Evanston
    lakefront generally.
  - The previous-runs (forecast archive) API may not have data for the
    earliest years; missing days are saved as blanks and the analysis
    script reports coverage per year.
"""

import csv
import time

import requests

# --- Configuration ---
LATITUDE = 42.0409       # Dempster St & Forest Ave, Evanston, IL
LONGITUDE = -87.6796
START_DATE = "2021-06-01"  # five summers: 2021-2025
END_DATE = "2025-08-31"
TIMEZONE = "America/Chicago"


def get_json(url, params, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  Request failed ({e}); retrying in {wait}s...")
            time.sleep(wait)


def fetch_actuals():
    print("Fetching historical ground truth (ERA5)...")
    data = get_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "daily": "temperature_2m_max,shortwave_radiation_sum,sunshine_duration",
            "temperature_unit": "fahrenheit",
            "timezone": TIMEZONE,
        },
    )
    daily = data["daily"]
    with open("data/actuals.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "actual_max", "solar_radiation", "sunshine_hours"])
        for i, date in enumerate(daily["time"]):
            sun_sec = daily["sunshine_duration"][i]
            w.writerow([
                date,
                daily["temperature_2m_max"][i],
                daily["shortwave_radiation_sum"][i],
                round(sun_sec / 3600.0, 3) if sun_sec is not None else "",
            ])
    print(f"  Saved {len(daily['time'])} days -> data/actuals.csv")


def fetch_forecasts():
    # Note: the Previous Runs API does not offer a daily
    # temperature_2m_max_previous_day1 variable, so we fetch the hourly
    # day-ahead temperature and take the max per local calendar day.
    print("Fetching day-ahead forecast archive (previous runs, hourly)...")
    data = get_json(
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": "temperature_2m_previous_day1",
            "temperature_unit": "fahrenheit",
            "timezone": TIMEZONE,
        },
    )
    hourly = data["hourly"]
    times = hourly["time"]
    values = hourly["temperature_2m_previous_day1"]

    # Aggregate hourly -> daily max. Only keep days with full (or nearly
    # full) coverage so a lone stray hour can't masquerade as a daily max.
    per_day = {}
    for ts, v in zip(times, values):
        day = ts[:10]
        per_day.setdefault(day, []).append(v)

    n_missing = 0
    with open("data/forecasts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "forecast_max"])
        for day in sorted(per_day):
            vals = [v for v in per_day[day] if v is not None]
            if len(vals) >= 20:  # require >= 20 of 24 hours present
                w.writerow([day, round(max(vals), 1)])
            else:
                w.writerow([day, ""])
                n_missing += 1
    print(f"  Saved {len(per_day)} days -> data/forecasts.csv "
          f"({n_missing} missing/incomplete forecast days)")


if __name__ == "__main__":
    fetch_actuals()
    fetch_forecasts()
    print("Done. Now run: python scripts/analyze.py")
