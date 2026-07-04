# Evanston Weather Analysis

Does the day-ahead forecast systematically **underestimate the daily high on
sunny summer days** in Evanston, IL?

Location: Evanston lakefront reference point (42.0409, -87.6796). Open-Meteo
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
pip install requests pandas matplotlib scipy
python scripts/fetch_data.py   # data/actuals.csv, forecasts.csv, features.csv
python scripts/analyze.py      # results/summary.txt + 4 plots
python scripts/traits.py       # results/traits.txt + error_vs_humidity.png
```

Note: `fetch_data.py` must run from a normal network (Open-Meteo blocks some
automated environments). Cached CSVs are committed in `data/` so the analysis
is reproducible without re-fetching.

**Why the window starts in 2021:** the Previous Runs (day-ahead forecast)
archive has no data before 2021 — `temperature_2m_previous_day1` is empty for
2016-2020 and the API rejects dates before 2016. ERA5 actuals go back decades,
but without archived forecasts there is no error to compute, so 2021-2025 is
the full extent of this analysis.

## Forecast-band and trait findings

Pooling all days hides the pattern; conditioning on the forecast level shows a
lean toward underestimation for forecasts in **[65, 80) F** (~55% of days
under, and big misses run ~6:1 in the underestimate direction). See
`analyze.py`'s band section.

`traits.py` asks what separates the missed-low from missed-high days, using
only features independent of `actual_max`. The strongest clean separator is
**afternoon humidity**: dry afternoons underestimate (~+2 F, ~72% of days),
humid afternoons do not (~-0.2 F). Offshore (westerly) winds — which remove
Lake Michigan's cooling — also lean strongly toward underestimation. Together
these traits explain only ~15% of the band's error variance, so the effect is
real but partial; most of the day-to-day error is unmodeled noise.

## Live lightning map

`lightning/index.html` is a self-contained webpage (open it in any browser, or
serve via GitHub Pages) that plots real-time lightning strikes from the
[Blitzortung.org](https://www.blitzortung.org) volunteer detection network on a
dark map centered on an Evanston lakefront reference point. Each strike spawns
a ring expanding at the speed of sound (343 m/s) that fades out near ~20 km,
the practical limit of audible thunder, and the panel counts down the seconds
until the nearest thunder front reaches that point. A **Simulate storm** toggle
generates a synthetic cell drifting NE over Lake Michigan for quiet-sky days.
No API key or build step required.

## Definitions

- **Sunny day:** >= 10 hours of bright sunshine
- **Cloudy day:** < 5 hours
- **Big miss:** forecast underestimated the high by >= 5 F
