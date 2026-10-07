"""Train and evaluate a linear regression baseline on the housing dataset."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402

from .load import REPORTS_DIR, TARGET, load  # noqa: E402

sns.set_theme(style="whitegrid", palette="viridis")

# log1p keeps the skewed price target usable for a linear model.
FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "rooms_per_household", "bedrooms_per_room", "population_per_household",
    "ocean_proximity",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_pipeline(numeric_cols: list, categorical_cols: list) -> Pipeline:
    """Scale numeric features, one-hot encode categories, then fit OLS."""
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return Pipeline([("pre", pre), ("model", LinearRegression())])


def evaluate(y_true, y_pred) -> dict:
    """Standard regression metrics."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def _save(fig: plt.Figure, name: str) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    fig.savefig(REPORTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def plot_predictions(y_test, y_pred) -> None:
    """Predicted vs actual — points on the diagonal mean a perfect model."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.3, s=14)
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#d9a441")
    ax.set_xlabel("Actual value (USD)")
    ax.set_ylabel("Predicted value (USD)")
    ax.set_title("Predicted vs actual house value")
    _save(fig, "07_predicted_vs_actual.png")


def plot_residuals(y_test, y_pred) -> None:
    """Residual histogram + residual-vs-prediction for spread/heteroscedasticity."""
    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, ax=ax)
    ax.axvline(0, color="#d9a441", linestyle="--")
    ax.set_title("Residual distribution")
    ax.set_xlabel("Actual - predicted (USD)")
    _save(fig, "08_residuals_hist.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.3, s=14)
    ax.axhline(0, color="#d9a441", linestyle="--")
    ax.set_xlabel("Predicted value (USD)")
    ax.set_ylabel("Residual (USD)")
    ax.set_title("Residuals vs predicted")
    _save(fig, "09_residuals_vs_pred.png")


def plot_coefficients(pipe: Pipeline) -> pd.DataFrame:
    """Turn fitted coefficients back into readable feature names."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_

    coef_df = (
        pd.DataFrame({"feature": names, "coefficient": coefs})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )
    coef_df.to_csv(REPORTS_DIR / "coefficients.csv", index=False)

    top = coef_df.head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=top, x="coefficient", y="feature", hue="feature",
                palette="viridis", legend=False, ax=ax)
    ax.set_title("Top 15 coefficients (standardised scale)")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("")
    _save(fig, "10_coefficients.png")

    return coef_df


def run() -> dict:
    """Full train/evaluate cycle. Returns the metrics dict."""
    print("Training linear regression baseline...")
    df = load()

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  train={len(X_train):,}  test={len(X_test):,}")

    numeric_cols = [c for c in FEATURES if c != "ocean_proximity"]
    pipe = build_pipeline(numeric_cols, ["ocean_proximity"])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\nTest-set performance")
    print(f"  R2   = {metrics['r2']:.3f}")
    print(f"  MAE  = ${metrics['mae']:,.0f}")
    print(f"  RMSE = ${metrics['rmse']:,.0f}")
    print(f"  MAPE = {metrics['mape']:.1f}%")

    plot_predictions(y_test.to_numpy(), y_pred)
    plot_residuals(y_test.to_numpy(), y_pred)
    coefs = plot_coefficients(pipe)

    print("\nTop 5 features by |coefficient|:")
    for _, row in coefs.head(5).iterrows():
        print(f"  {row['feature']:<40} {row['coefficient']:+,.0f}")

    return metrics


if __name__ == "__main__":
    run()
