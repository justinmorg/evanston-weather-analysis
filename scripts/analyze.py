"""
analyze.py — Day-ahead forecast bias on sunny summer days, Evanston IL.

Question: does the day-ahead forecast systematically underestimate the
daily high on sunny summer days (June-August)?

Reads data/actuals.csv and data/forecasts.csv (produced by fetch_data.py),
writes results/summary.txt and three plots to results/.

Error convention: error = actual_max - forecast_max
  positive  -> forecast UNDERestimated the high
  negative  -> forecast OVERestimated the high
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUNNY_HOURS = 10.0   # >= this many sunshine hours = "sunny"
CLOUDY_HOURS = 5.0   # <  this many sunshine hours = "cloudy"
BIG_MISS_F = 5.0     # threshold for a "big underestimation"
BAND_LO = 65.0       # forecast-band lower bound (inclusive)
BAND_HI = 80.0       # forecast-band upper bound (exclusive)


def load_data():
    actual = pd.read_csv("data/actuals.csv", parse_dates=["date"])
    forecast = pd.read_csv("data/forecasts.csv", parse_dates=["date"])
    df = pd.merge(actual, forecast, on="date", how="inner")
    df = df[df["date"].dt.month.isin([6, 7, 8])].copy()
    df["year"] = df["date"].dt.year

    total = len(df)
    df = df.dropna(subset=["actual_max", "forecast_max", "sunshine_hours"])
    dropped = total - len(df)

    df["error"] = df["actual_max"] - df["forecast_max"]
    df["sky"] = np.select(
        [df["sunshine_hours"] >= SUNNY_HOURS, df["sunshine_hours"] < CLOUDY_HOURS],
        ["sunny", "cloudy"], default="mixed",
    )
    return df, dropped


def welch_t(a, b):
    """Welch's t-test (two-sided) without scipy dependency; returns t, approx p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), dof)
    except ImportError:
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))  # normal approx
    return t, p


def line(f, text=""):
    print(text)
    f.write(text + "\n")


def main():
    os.makedirs("results", exist_ok=True)
    df, dropped = load_data()
    sunny = df[df["sky"] == "sunny"]
    cloudy = df[df["sky"] == "cloudy"]
    mixed = df[df["sky"] == "mixed"]

    with open("results/summary.txt", "w") as f:
        line(f, "=== EVANSTON SUMMER FORECAST BIAS (day-ahead max temp) ===")
        line(f, f"Period: {df['date'].min().date()} to {df['date'].max().date()} "
                f"(June-August only)")
        line(f, f"Days analyzed: {len(df)}  (dropped {dropped} with missing data)")
        line(f, f"Sunny days   (>= {SUNNY_HOURS:.0f} hrs sunshine): {len(sunny)}")
        line(f, f"Mixed days   : {len(mixed)}")
        line(f, f"Cloudy days  (<  {CLOUDY_HOURS:.0f} hrs sunshine): {len(cloudy)}")
        line(f, "-" * 55)
        line(f, "Mean error (actual - forecast; + = underestimated):")
        line(f, f"  All days   : {df['error'].mean():+.2f} F  "
                f"(MAE {df['error'].abs().mean():.2f} F)")
        line(f, f"  Sunny days : {sunny['error'].mean():+.2f} F  "
                f"(MAE {sunny['error'].abs().mean():.2f} F)")
        line(f, f"  Mixed days : {mixed['error'].mean():+.2f} F")
        line(f, f"  Cloudy days: {cloudy['error'].mean():+.2f} F  "
                f"(MAE {cloudy['error'].abs().mean():.2f} F)")

        t, p = welch_t(sunny["error"], cloudy["error"])
        line(f, f"Sunny vs cloudy mean error difference: "
                f"{sunny['error'].mean() - cloudy['error'].mean():+.2f} F "
                f"(Welch t = {t:.2f}, p = {p:.4f})")
        line(f, "-" * 55)

        big = sunny[sunny["error"] >= BIG_MISS_F]
        pct = 100 * len(big) / len(sunny) if len(sunny) else float("nan")
        line(f, f"Sunny days underestimated by >= {BIG_MISS_F:.0f} F: "
                f"{len(big)} of {len(sunny)} ({pct:.1f}%)")
        big_cloudy = cloudy[cloudy["error"] >= BIG_MISS_F]
        pct_c = 100 * len(big_cloudy) / len(cloudy) if len(cloudy) else float("nan")
        line(f, f"Cloudy days underestimated by >= {BIG_MISS_F:.0f} F: "
                f"{len(big_cloudy)} of {len(cloudy)} ({pct_c:.1f}%)  [baseline]")
        line(f, "-" * 55)

        line(f, "Per-year breakdown (summer only):")
        yearly = df.groupby("year").agg(
            days=("error", "size"),
            mean_error=("error", "mean"),
            sunny_days=("sky", lambda s: (s == "sunny").sum()),
        )
        yearly["sunny_mean_error"] = df[df["sky"] == "sunny"].groupby("year")["error"].mean()
        line(f, yearly.round(2).to_string())
        line(f, "-" * 55)

        corr = df["error"].corr(df["sunshine_hours"])
        line(f, f"Correlation of error vs sunshine hours (all days): r = {corr:+.3f}")

        if not big.empty:
            line(f, "")
            line(f, "Top 5 biggest sunny-day underestimations:")
            top = big.sort_values("error", ascending=False).head(5)
            line(f, top[["date", "forecast_max", "actual_max", "error",
                         "sunshine_hours"]].to_string(index=False))

        # --- Forecast-band analysis (65-80 F) -----------------------------
        # Pooling errors across all forecast levels can hide a bias that only
        # appears at a particular forecast range. Condition on the band.
        line(f, "")
        line(f, "=" * 55)
        line(f, f"FORECAST-BAND ANALYSIS: forecast in [{BAND_LO:.0f}, {BAND_HI:.0f}) F")
        line(f, "=" * 55)
        band = df[(df["forecast_max"] >= BAND_LO) &
                  (df["forecast_max"] < BAND_HI)].copy()
        nb = len(band)
        under = band[band["error"] > 0]
        over = band[band["error"] < 0]
        line(f, f"Days in band: {nb}")
        line(f, f"  Underestimated (actual > forecast): {len(under)} "
                f"({100*len(under)/nb:.1f}%)")
        line(f, f"  Overestimated  (actual < forecast): {len(over)} "
                f"({100*len(over)/nb:.1f}%)")
        line(f, f"  Exact                             : "
                f"{nb - len(under) - len(over)}")
        line(f, f"Mean error {band['error'].mean():+.2f} F, "
                f"median {band['error'].median():+.2f} F")
        line(f, f"Mean miss when UNDER: {under['error'].mean():+.2f} F   "
                f"when OVER: {over['error'].mean():+.2f} F")
        line(f, "Tail asymmetry (the key signal):")
        line(f, f"  underestimated by >= {BIG_MISS_F:.0f} F: "
                f"{len(band[band['error'] >= BIG_MISS_F])} days "
                f"({100*len(band[band['error'] >= BIG_MISS_F])/nb:.1f}%)")
        line(f, f"  overestimated  by >= {BIG_MISS_F:.0f} F: "
                f"{len(band[band['error'] <= -BIG_MISS_F])} days "
                f"({100*len(band[band['error'] <= -BIG_MISS_F])/nb:.1f}%)")
        line(f, f"Signed error sum: under {under['error'].sum():+.0f} F over "
                f"{len(under)} days vs over {over['error'].sum():+.0f} F over "
                f"{len(over)} days (net {band['error'].sum():+.0f} F)")

    # --- Plots ---
    plt.figure(figsize=(10, 6))
    bins = np.arange(-12, 13, 1)
    plt.hist(sunny["error"], bins=bins, alpha=0.7, color="orange",
             edgecolor="black", label=f"Sunny (n={len(sunny)})")
    plt.hist(cloudy["error"], bins=bins, alpha=0.5, color="steelblue",
             edgecolor="black", label=f"Cloudy (n={len(cloudy)})")
    plt.axvline(0, color="red", ls="--", lw=1.5, label="Perfect forecast")
    plt.axvline(sunny["error"].mean(), color="darkorange", ls=":", lw=2,
                label=f"Sunny mean {sunny['error'].mean():+.1f}F")
    plt.title("Evanston Summer Day-Ahead Forecast Error (Actual - Predicted), 2021-2025")
    plt.xlabel("degrees F underestimated (+) / overestimated (-)")
    plt.ylabel("Number of days")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/error_histogram.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(df["sunshine_hours"], df["error"], s=18, alpha=0.5,
                c=df["error"], cmap="coolwarm", vmin=-8, vmax=8)
    m, b = np.polyfit(df["sunshine_hours"], df["error"], 1)
    xs = np.linspace(0, df["sunshine_hours"].max(), 50)
    plt.plot(xs, m * xs + b, "k--", lw=1.5,
             label=f"Trend: {m:+.2f} F per sunshine hour")
    plt.axhline(0, color="red", lw=1)
    plt.title("Forecast Error vs Sunshine Hours (Evanston summers 2021-2025)")
    plt.xlabel("Sunshine hours")
    plt.ylabel("Error (F)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/error_vs_sunshine.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    yr = df.groupby("year")["error"].mean()
    yr_sunny = df[df["sky"] == "sunny"].groupby("year")["error"].mean()
    x = np.arange(len(yr))
    ax.bar(x - 0.2, yr.values, 0.4, label="All summer days", color="gray")
    ax.bar(x + 0.2, yr_sunny.reindex(yr.index).values, 0.4,
           label="Sunny days", color="orange")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, yr.index.astype(str))
    ax.set_ylabel("Mean error (F)")
    ax.set_title("Mean Day-Ahead Forecast Error by Summer")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("results/error_by_year.png", dpi=150)
    plt.close()

    # Signed-error histogram for the forecast band, colored by direction.
    band = df[(df["forecast_max"] >= BAND_LO) &
              (df["forecast_max"] < BAND_HI)]
    plt.figure(figsize=(10, 6))
    bins = np.arange(-12, 13, 1)
    plt.hist(band[band["error"] > 0]["error"], bins=bins, color="firebrick",
             alpha=0.75, edgecolor="black", label="Underestimated")
    plt.hist(band[band["error"] <= 0]["error"], bins=bins, color="steelblue",
             alpha=0.75, edgecolor="black", label="Over / exact")
    plt.axvline(0, color="black", lw=1.5)
    plt.axvline(band["error"].mean(), color="darkorange", ls=":", lw=2,
                label=f"Mean {band['error'].mean():+.1f}F")
    plt.title(f"Forecast Error for Forecasts in [{BAND_LO:.0f},{BAND_HI:.0f})F "
              f"(Evanston summers 2021-2025, n={len(band)})")
    plt.xlabel("degrees F underestimated (+) / overestimated (-)")
    plt.ylabel("Number of days")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/error_band_histogram.png", dpi=150)
    plt.close()

    print("\nPlots saved to results/: error_histogram.png, "
          "error_vs_sunshine.png, error_by_year.png, error_band_histogram.png")


if __name__ == "__main__":
    main()
