# California Housing — EDA & Linear Regression Baseline

An exploratory data analysis and a reproducible linear-regression baseline for the
[California housing dataset](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)
(20,640 districts, 10 raw features).

Built to answer one question: **how much of house price variation can a transparent,
linear model explain — and which features drive it?**

## Results

| Metric | Value |
| --- | --- |
| R² | **0.597** |
| MAE | **$50,889** |
| RMSE | **$72,669** |
| MAPE | **29.9%** |

Train/test split: 16,512 / 4,128 (`random_state=42`, `test_size=0.2`)

### Strongest correlations with house value

| Feature | Pearson r |
| --- | --- |
| median_income | **+0.688** |
| bedrooms_per_room | −0.233 |
| rooms_per_household | +0.152 |
| latitude | −0.144 |
| total_rooms | +0.134 |

Income vs. price: `r = 0.688`, `p ≈ 0` — the relationship is not a sampling artefact.

### Top standardised coefficients

| Feature | Coefficient |
| --- | --- |
| ocean_proximity = ISLAND | +112,060 |
| median_income | +78,904 |
| latitude | −56,638 |
| longitude | −56,276 |
| ocean_proximity = INLAND | −53,618 |

Location and income dominate — coastal/inland districts carry large, measurable premiums.

## What's in here

```
linear-regression-eda/
├── run.py               # CLI entry point: --eda | --model | --all
├── data/housing.csv     # dataset (downloaded once, committed)
├── src/
│   ├── load.py          # loading, missing-value handling, feature engineering
│   ├── explore.py       # EDA: distributions, correlations, geography, groups
│   └── model.py         # preprocessing pipeline, training, evaluation, plots
└── reports/             # generated charts + metrics (regenerated on each run)
```

## Approach

1. **Load & clean** — `total_bedrooms` had missing values (filled with the median),
   target column dropped where null, ocean proximity cast to an ordered category.
2. **Feature engineering** — raw counts scale with district size, so three ratios are
   derived: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`.
3. **EDA** — distribution plots, correlation heatmap, income/price regression plot,
   geographic value map, and price spread per ocean-proximity group.
4. **Modelling** — a `ColumnTransformer` pipeline scales numeric features and one-hot
   encodes the categorical one, then fits ordinary least squares. Fitting happens on the
   training split only; the test split is never seen during training.
5. **Evaluation** — R², MAE, RMSE and MAPE, plus predicted-vs-actual and residual plots
   to check spread and heteroscedasticity.

## Notable findings

- **Income is the single strongest predictor** (r = +0.688), well ahead of every
  structural feature like room or bedroom counts.
- **Residuals fan out as price rises**, so error is larger on expensive districts —
  a sign the linear model underfits the top of the market (the target is also
  capped at `$500,001`, which truncates the upper tail).
- **ISLAND districts carry the largest positive coefficient**, but they represent very
  few rows — a caveat worth stating before quoting that number.

## Limitations (being honest about them)

- OLS assumptions are only partly met: the target is right-skewed and heteroscedastic.
- No hyperparameter search or regularisation — this is a **baseline**, deliberately.
- District-level aggregates mean every row describes a neighbourhood, not a house.

## Ideas for next steps

- Ridge / Lasso / ElasticNet to regularise the coefficients
- `log1p` transform on the target to tame skew
- Gradient boosting (XGBoost / HistGradientBoosting) for a non-linear comparison
- Cross-validation instead of a single split

## Run it

```bash
pip install -r requirements.txt
python run.py            # EDA + model
python run.py --eda      # just the exploratory analysis
python run.py --model    # just the model
```

Charts and metrics are written to `reports/`.

## Tech

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · SciPy
