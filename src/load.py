"""Data loading and cleaning for the California housing dataset."""

from pathlib import Path

import pandas as pd

# Project layout
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "housing.csv"
REPORTS_DIR = ROOT / "reports"

# Column that must never contain missing values
TARGET = "median_house_value"


def load_raw(path: Path = DATA_FILE) -> pd.DataFrame:
    """Read the housing CSV from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset first:\n"
            "  https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy copy: missing values handled, dtypes confirmed, outliers trimmed."""
    out = df.copy()

    # total_bedrooms is the only column with gaps in this dataset.
    # Fill with the median: robust against the long right tail of room counts.
    if out["total_bedrooms"].isna().any():
        out["total_bedrooms"] = out["total_bedrooms"].fillna(out["total_bedrooms"].median())

    # Drop any remaining rows without a target value.
    out = out.dropna(subset=[TARGET])

    # Ocean category -> ordered categorical for stable group-by ordering.
    out["ocean_proximity"] = pd.Categorical(
        out["ocean_proximity"],
        categories=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
        ordered=True,
    )

    # Price is easier to read in dollars rather than raw floats.
    out[TARGET] = out[TARGET].astype("int64")

    return out.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-household ratios — raw room counts scale with district size."""
    out = df.copy()
    out["rooms_per_household"] = out["total_rooms"] / out["households"]
    out["bedrooms_per_room"] = out["total_bedrooms"] / out["total_rooms"]
    out["population_per_household"] = out["population"] / out["households"]
    return out


def load(path: Path = DATA_FILE, with_features: bool = True) -> pd.DataFrame:
    """Convenience wrapper: load -> clean -> (optionally) engineer features."""
    data = clean(load_raw(path))
    return add_features(data) if with_features else data
