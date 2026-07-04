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
# The Previous Runs (day-ahead forecast) archive has NO data before 2021 --
# temperature_2m_previous_day1 is empty for 2016-2020 and the API rejects
# dates before 2016-01-01. So 2021 is the earliest usable start for forecast
# error analysis, even though ERA5 actuals go back decades.
START_DATE = "2021-06-01"  # five summers: 2021-2025 (forecast-archive limited)
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


def _circ_mean(degs):
    """Vector-mean of wind directions in degrees (handles the 0/360 wrap)."""
    import math
    degs = [d for d in degs if d is not None]
    if not degs:
        return ""
    s = sum(math.sin(math.radians(d)) for d in degs)
    c = sum(math.cos(math.radians(d)) for d in degs)
    return round(math.degrees(math.atan2(s, c)) % 360, 1)


def fetch_features():
    """Candidate explanatory features for the error, used by traits.py.

    Daily air-mass descriptors plus afternoon (12-18h local) means of
    humidity, dewpoint, wind direction and cloud cover. These are all
    independent of actual_max (so they don't trivially correlate with the
    error, which is actual_max - forecast_max)."""
    print("Fetching explanatory features (archive)...")
    daily = get_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "start_date": START_DATE, "end_date": END_DATE,
            "daily": ("temperature_2m_min,wind_speed_10m_max,"
                      "wind_direction_10m_dominant,precipitation_sum,"
                      "et0_fao_evapotranspiration"),
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "timezone": TIMEZONE,
        },
    )["daily"]
    hourly = get_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "start_date": START_DATE, "end_date": END_DATE,
            "hourly": ("relative_humidity_2m,dew_point_2m,"
                       "wind_direction_10m,cloud_cover"),
            "temperature_unit": "fahrenheit", "timezone": TIMEZONE,
        },
    )["hourly"]

    # Aggregate the 12:00-18:00 local window per day.
    aft = {}
    for i, ts in enumerate(hourly["time"]):
        if 12 <= int(ts[11:13]) <= 18:
            d = aft.setdefault(ts[:10], {"rh": [], "dp": [], "wd": [], "cc": []})
            d["rh"].append(hourly["relative_humidity_2m"][i])
            d["dp"].append(hourly["dew_point_2m"][i])
            d["wd"].append(hourly["wind_direction_10m"][i])
            d["cc"].append(hourly["cloud_cover"][i])

    def mean(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 1) if v else ""

    with open("data/features.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "temp_min", "wind_max", "wind_dir_dom",
                    "precip", "et0", "aft_rh", "aft_dewpt",
                    "aft_wind_dir", "aft_cloud"])
        for i, day in enumerate(daily["time"]):
            a = aft.get(day, {"rh": [], "dp": [], "wd": [], "cc": []})
            w.writerow([
                day, daily["temperature_2m_min"][i],
                daily["wind_speed_10m_max"][i],
                daily["wind_direction_10m_dominant"][i],
                daily["precipitation_sum"][i],
                daily["et0_fao_evapotranspiration"][i],
                mean(a["rh"]), mean(a["dp"]), _circ_mean(a["wd"]), mean(a["cc"]),
            ])
    print(f"  Saved {len(daily['time'])} days -> data/features.csv")


if __name__ == "__main__":
    fetch_actuals()
    fetch_forecasts()
    fetch_features()
    print("Done. Now run: python scripts/analyze.py  then  python scripts/traits.py")
