"""
traits.py — What distinguishes underestimated from overestimated forecasts?

Focuses on the [65, 80) F forecast band (where the underestimation lean is
concentrated) and asks which day characteristics separate the days the
day-ahead forecast missed LOW from the days it missed HIGH.

Important: the error is  actual_max - forecast_max.  Any feature built from
actual_max (e.g. the diurnal range actual_max - temp_min) is mechanically
correlated with the error and is NOT a real explanatory trait. We therefore
lean on features that are independent of actual_max: afternoon humidity,
dewpoint, cloud, wind, night-time low, solar radiation, evapotranspiration.

Reads data/{actuals,forecasts,features}.csv; writes results/traits.txt and
results/error_vs_humidity.png.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAND_LO, BAND_HI = 65.0, 80.0


def load():
    a = pd.read_csv("data/actuals.csv", parse_dates=["date"])
    f = pd.read_csv("data/forecasts.csv", parse_dates=["date"])
    ft = pd.read_csv("data/features.csv", parse_dates=["date"])
    df = a.merge(f, on="date").merge(ft, on="date")
    df = df[df["date"].dt.month.isin([6, 7, 8])].copy()
    df = df.dropna(subset=["actual_max", "forecast_max", "sunshine_hours",
                           "aft_rh", "temp_min"])
    df["error"] = df["actual_max"] - df["forecast_max"]
    return df[(df["forecast_max"] >= BAND_LO) & (df["forecast_max"] < BAND_HI)].copy()


def wind_sector(deg):
    # Evanston faces Lake Michigan to the east. Easterly = onshore (lake air,
    # cooling); westerly = offshore (land air, removes the lake's influence).
    if pd.isna(deg):
        return "na"
    if 45 <= deg < 135:
        return "onshore (E)"
    if 225 <= deg < 315:
        return "offshore (W)"
    return "other"


def out(f, text=""):
    print(text)
    f.write(text + "\n")


def main():
    os.makedirs("results", exist_ok=True)
    band = load()

    # Clean features only (none contain actual_max).
    clean = [
        ("aft_rh", "afternoon RH (%)"),
        ("aft_dewpt", "afternoon dewpoint (F)"),
        ("aft_cloud", "afternoon cloud (%)"),
        ("solar_radiation", "solar radiation (MJ)"),
        ("sunshine_hours", "sunshine (hrs)"),
        ("wind_max", "max wind (mph)"),
        ("temp_min", "night low (F)"),
        ("et0", "evapotranspiration"),
    ]

    with open("results/traits.txt", "w") as f:
        out(f, "=== TRAITS SEPARATING UNDER- vs OVER-ESTIMATED FORECASTS ===")
        out(f, f"Forecast band [{BAND_LO:.0f}, {BAND_HI:.0f}) F, summers 2021-2025, "
               f"n = {len(band)}")
        out(f, "Error = actual - forecast; features below are independent of actual.")
        out(f, "-" * 60)

        u = band[band["error"] > 0]
        o = band[band["error"] < 0]
        out(f, f"{'feature':<22}{'UNDER':>9}{'OVER':>9}{'diff':>8}{'r w/err':>9}")
        for c, lbl in clean:
            out(f, f"{lbl:<22}{u[c].mean():>9.1f}{o[c].mean():>9.1f}"
                   f"{u[c].mean()-o[c].mean():>8.1f}{band['error'].corr(band[c]):>9.3f}")
        out(f, "-" * 60)

        out(f, "Error by afternoon-RH tercile:")
        band["rh_tier"] = pd.qcut(band["aft_rh"], 3, labels=["dry", "mid", "humid"])
        g = band.groupby("rh_tier", observed=True).agg(
            n=("error", "size"), mean_err=("error", "mean"),
            pct_under=("error", lambda s: 100 * (s > 0).mean()))
        out(f, g.round(2).to_string())
        out(f, "-" * 60)

        out(f, "Error by afternoon wind sector (lake-breeze test):")
        band["sector"] = band["aft_wind_dir"].apply(wind_sector)
        g = band.groupby("sector", observed=True).agg(
            n=("error", "size"), mean_err=("error", "mean"),
            pct_under=("error", lambda s: 100 * (s > 0).mean()))
        out(f, g.round(2).to_string())
        out(f, "-" * 60)

        # Standardized OLS on clean features to see what survives together.
        feats = ["aft_rh", "solar_radiation", "wind_max", "temp_min"]
        X = band[feats].copy()
        X = (X - X.mean()) / X.std()
        X["const"] = 1.0
        y = band["error"].values
        beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
        pred = X.values @ beta
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out(f, "Standardized OLS (error ~ clean features); coef = F per 1 SD:")
        for name, b in zip(X.columns, beta):
            if name != "const":
                out(f, f"  {name:<18}{b:+.2f}")
        out(f, f"  R^2 = {r2:.3f}  (traits explain ~{100*r2:.0f}% of band variance)")

    # Plot: error vs afternoon humidity, the strongest clean separator.
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(band["aft_rh"], band["error"], c=band["error"],
                     cmap="coolwarm", vmin=-8, vmax=8, s=22, alpha=0.7,
                     edgecolor="gray", linewidth=0.3)
    m, b = np.polyfit(band["aft_rh"], band["error"], 1)
    xs = np.linspace(band["aft_rh"].min(), band["aft_rh"].max(), 50)
    plt.plot(xs, m * xs + b, "k--", lw=1.5,
             label=f"trend {m:+.3f} F per %RH")
    plt.axhline(0, color="black", lw=1)
    plt.title("Forecast Error vs Afternoon Humidity "
              f"(forecasts {BAND_LO:.0f}-{BAND_HI:.0f}F, Evanston summers)")
    plt.xlabel("Afternoon (12-18h) relative humidity (%)")
    plt.ylabel("Error (F): actual - forecast")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.colorbar(sc, label="Error (F)")
    plt.tight_layout()
    plt.savefig("results/error_vs_humidity.png", dpi=150)
    plt.close()
    print("\nWrote results/traits.txt and results/error_vs_humidity.png")


if __name__ == "__main__":
    main()
