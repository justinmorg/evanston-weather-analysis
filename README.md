# Evanston Weather Analysis

Does the day-ahead forecast systematically **underestimate the daily high on
sunny summer days** in Evanston, IL?

Location: Dempster St & Forest Ave area (42.0409, -87.6796). Open-Meteo
interpolates from a ~9 km model grid, so results represent the Evanston
lakefront generally. Period: summers (June-August) 2021-2025.

## Data

- **Actuals:** ERA5 reanalysis via Open-Meteo Historical API — daily max temp
  (F), shortwave radiation sum (MJ/m2), sunshine duration (hours).
- **Forecasts:** Open-Meteo Previous Runs API — the max temp that was
  predicted 24 hours ahead (`temperature_2m_max_previous_day1`).
- Error convention: `error = actual - forecast`; positive = underestimated.

## Usage

```bash
pip install requests pandas matplotlib
python scripts/fetch_data.py   # downloads data/actuals.csv, data/forecasts.csv
python scripts/analyze.py      # writes results/summary.txt and plots
```

Note: `fetch_data.py` must run from a normal network (Open-Meteo blocks some
automated environments). Cached CSVs are committed in `data/` so the analysis
is reproducible without re-fetching.

## Definitions

- **Sunny day:** >= 10 hours of bright sunshine
- **Cloudy day:** < 5 hours
- **Big miss:** forecast underestimated the high by >= 5 F
