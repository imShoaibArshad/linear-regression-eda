"""Exploratory data analysis: summary stats and saved charts."""

from pathlib import Path

import matplotlib

# Headless backend so the script works on servers / CI without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

from .load import REPORTS_DIR, TARGET, load  # noqa: E402

sns.set_theme(style="whitegrid", palette="viridis")

NUMERIC_ONLY = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "rooms_per_household", "bedrooms_per_room", "population_per_household",
    TARGET,
]


def _save(fig: plt.Figure, name: str) -> Path:
    """Write a figure into reports/ and close it to free memory."""
    path = REPORTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")
    return path


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics, written to reports/summary.csv."""
    stats_df = df[NUMERIC_ONLY].describe().T
    stats_df["skew"] = df[NUMERIC_ONLY].skew()
    stats_df["missing"] = df[NUMERIC_ONLY].isna().sum()
    REPORTS_DIR.mkdir(exist_ok=True)
    stats_df.to_csv(REPORTS_DIR / "summary.csv")
    return stats_df


def plot_distributions(df: pd.DataFrame) -> None:
    """Histogram of price plus a grid of the numeric predictors."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df[TARGET] / 1e6, bins=50, kde=True, ax=ax)
    ax.set_title("Median house value distribution")
    ax.set_xlabel("Value (millions USD)")
    _save(fig, "01_target_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["median_income"], bins=50, kde=True, ax=ax, color="#b4781f")
    ax.set_title("District median income distribution")
    ax.set_xlabel("Income (scaled)")
    _save(fig, "02_income_distribution.png")


def plot_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Heatmap of Pearson correlations; returns the matrix for reporting."""
    corr = df[NUMERIC_ONLY].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, cmap="coolwarm", center=0, annot=False,
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation matrix")
    _save(fig, "03_correlation_heatmap.png")
    return corr


def plot_income_vs_price(df: pd.DataFrame) -> None:
    """The single strongest relationship in the dataset."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.regplot(
        data=df, x="median_income", y=df[TARGET] / 1e6,
        scatter_kws={"alpha": 0.25, "s": 12}, line_kws={"color": "#d9a441"},
        ax=ax,
    )
    ax.set_title("House value vs district median income")
    ax.set_xlabel("Median income (scaled)")
    ax.set_ylabel("Value (millions USD)")
    _save(fig, "04_income_vs_price.png")


def plot_geo(df: pd.DataFrame) -> None:
    """Geographic scatter: location drives price almost as much as income."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        df["longitude"], df["latitude"], c=df[TARGET] / 1e6,
        cmap="viridis", alpha=0.5, s=12,
    )
    fig.colorbar(sc, ax=ax, label="Value (millions USD)")
    ax.set_title("House value by location")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    _save(fig, "05_geographic_value.png")


def plot_by_ocean(df: pd.DataFrame) -> None:
    """Price spread per ocean-proximity category."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="ocean_proximity", y=df[TARGET] / 1e6, ax=ax)
    ax.set_title("House value by ocean proximity")
    ax.set_xlabel("")
    ax.set_ylabel("Value (millions USD)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, "06_price_by_ocean.png")


def run() -> None:
    """Execute the full EDA pipeline."""
    print("Running EDA...")
    df = load()
    print(f"  rows={len(df):,}  cols={df.shape[1]}")

    REPORTS_DIR.mkdir(exist_ok=True)

    summary(df)
    plot_distributions(df)
    corr = plot_correlation(df)
    plot_income_vs_price(df)
    plot_geo(df)
    plot_by_ocean(df)

    # Call out the strongest linear relationships with the target.
    top = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(5)
    print("\nStrongest correlations with house value:")
    for name, value in top.items():
        r = corr.loc[name, TARGET]
        print(f"  {name:<28} r = {r:+.3f}")

    # Quick significance check on the headline relationship.
    r, p = stats.pearsonr(df["median_income"], df[TARGET])
    print(f"\nIncome vs value: r = {r:.3f}, p = {p:.2e} "
          f"({'significant' if p < 0.001 else 'not significant'})")


if __name__ == "__main__":
    run()
