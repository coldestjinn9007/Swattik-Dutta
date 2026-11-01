from __future__ import annotations
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_NAME = "Processed_Count_and_timining_data.csv"
TARGET = "Count"
FEATURES = [
    "est_LinkCount",
    "gPhaseDuration",
    "gRatio",
    "rPhaseDuration",
    "Penetration_rate",
]


def locate_dataset(source: Path, work_dir: Path) -> Path:
    """Return the path to the required CSV, extracting a ZIP if needed."""
    if source.is_file() and zipfile.is_zipfile(source):
        extract_dir = work_dir / "extracted_dataset"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source, "r") as zf:
            # Extract archive contents to the requested output folder.
            zf.extractall(extract_dir)
        matches = list(extract_dir.rglob(CSV_NAME))
    elif source.is_dir():
        matches = list(source.rglob(CSV_NAME))
    elif source.is_file() and source.name == CSV_NAME:
        matches = [source]
    else:
        raise FileNotFoundError(
            f"Input must be a ZIP archive, dataset folder, or {CSV_NAME} file.\n"
            f"Input received: {source}"
        )

    if not matches:
        raise FileNotFoundError(
            f"Could not find {CSV_NAME}. Check that the ZIP/folder contains the dataset."
        )
    if len(matches) > 1:
        print(f"Found {len(matches)} matching files; using: {matches[0]}")
    return matches[0]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Dataset ZIP archive, extracted folder, or CSV file")
    parser.add_argument(
        "--output-dir", default="webel_more_results",
        help="Directory for result CSV files and plots (default: webel_more_results)",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = locate_dataset(source, output_dir)
    print(f"\nDataset CSV: {csv_path}")
    print(f"Output folder: {output_dir}\n")

    # 1. Load and inspect data.
    df = pd.read_csv(csv_path)
    print("Original data shape:", df.shape)
    print("Column names:", list(df.columns))

    required = ["Start", TARGET] + FEATURES
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing from dataset: {missing_columns}")

    # The source data stores dates like 08/01/2022 11.48.
    df["dt"] = pd.to_datetime(
        df["Start"], format="%d/%m/%Y %H.%M", errors="coerce"
    )
    invalid_dates = int(df["dt"].isna().sum())
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)

    # 2. Convert analysis columns to numeric and clean missing/infinite values.
    for column in [TARGET] + FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    analysis_data = df[["dt", TARGET] + FEATURES].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().reset_index(drop=True)
    analysis_data = analysis_data.sort_values("dt").reset_index(drop=True)

    print("Invalid timestamps removed:", invalid_dates)
    print("Rows available after cleaning:", len(analysis_data))
    print("Date range:", analysis_data["dt"].min(), "to", analysis_data["dt"].max())
    print("\nTarget (Count) descriptive statistics:")
    print(analysis_data[TARGET].describe().round(3).to_string())

    if len(analysis_data) < 5:
        raise ValueError("Too few clean observations to perform a train/test split.")
    if (analysis_data[TARGET] < 0).any():
        raise ValueError("Poisson regression requires non-negative target counts.")

    # 3. Chronological split: earliest 80% for training; latest 20% for testing.
    split_index = int(len(analysis_data) * 0.8)
    train = analysis_data.iloc[:split_index].copy()
    test = analysis_data.iloc[split_index:].copy()
    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train[TARGET], test[TARGET]

    print(f"\nChronological split: {len(train)} train / {len(test)} test observations")
    print(f"Training mean Count: {y_train.mean():.3f}")
    print(f"Test mean Count: {y_test.mean():.3f}")
    print(f"Train period: {train['dt'].min()} to {train['dt'].max()}")
    print(f"Test period:  {test['dt'].min()} to {test['dt'].max()}")

    # 4. Fit models / generate predictions.
    # Baseline predicts the training-set mean for every test observation.
    predictions: dict[str, np.ndarray] = {
        "Training-mean baseline": np.repeat(float(y_train.mean()), len(test)),
        # This is an existing count estimate from the dataset, used directly.
        "Existing estimated link count": test["est_LinkCount"].to_numpy(dtype=float),
    }

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    predictions["Multiple linear regression"] = np.clip(
        linear_model.predict(X_test), 0, None
    )

    poisson_model = make_pipeline(
        StandardScaler(),
        PoissonRegressor(alpha=1.0, max_iter=2000),
    )
    poisson_model.fit(X_train, y_train)
    predictions["Poisson regression (L2 alpha=1.0)"] = poisson_model.predict(X_test)

    # 5. Metrics table.
    metric_rows = []
    for model_name, pred in predictions.items():
        metric_rows.append({
            "Model": model_name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": rmse(y_test, pred),
            "R2": r2_score(y_test, pred),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "webel_more_model_metrics.csv", index=False)

    print("\nMODEL PERFORMANCE ON THE CHRONOLOGICAL TEST SET")
    print(metrics.to_string(index=False, formatters={
        "MAE": "{:.3f}".format,
        "RMSE": "{:.3f}".format,
        "R2": "{:.3f}".format,
    }))

    # Percentage improvements for the Poisson model relative to baseline and link estimate.
    poisson_row = metrics.loc[metrics["Model"] == "Poisson regression (L2 alpha=1.0)"].iloc[0]
    print("\nPoisson-model error reduction (positive means lower error):")
    for comparison in ["Training-mean baseline", "Existing estimated link count"]:
        other = metrics.loc[metrics["Model"] == comparison].iloc[0]
        mae_reduction = 100 * (other["MAE"] - poisson_row["MAE"]) / other["MAE"]
        rmse_reduction = 100 * (other["RMSE"] - poisson_row["RMSE"]) / other["RMSE"]
        print(f"  vs {comparison}: MAE {mae_reduction:.1f}%; RMSE {rmse_reduction:.1f}%")

    # 6. Export per-observation predictions.
    prediction_table = pd.DataFrame({
        "datetime": test["dt"].to_numpy(),
        "observed_Count": y_test.to_numpy(),
        **{name: pred for name, pred in predictions.items()},
    })
    prediction_table.to_csv(output_dir / "webel_more_test_predictions.csv", index=False)

    # 7. Figures. Matplotlib's default style/colors are intentionally retained.
    # Figure 1: observed counts vs predictions in time order.
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(test["dt"], y_test, marker="o", linewidth=2, label="Observed count")
    ax.plot(test["dt"], predictions["Poisson regression (L2 alpha=1.0)"],
            marker="s", linestyle="--", label="Poisson regression")
    ax.set_title("Webel More: observed and estimated traffic counts (test set)")
    ax.set_xlabel("Observation time")
    ax.set_ylabel("Traffic count")
    ax.legend()
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_observed_predicted.png", dpi=300)
    fig.savefig(output_dir / "webel_more_observed_vs_predicted.png", dpi=300)
    plt.close(fig)

    # Figure 2: MAE comparison.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(metrics["Model"], metrics["MAE"])
    ax.set_title("Model comparison by mean absolute error (MAE)")
    ax.set_ylabel("MAE (lower is better)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_mae.png", dpi=300)
    plt.close(fig)

    # Figure 3: RMSE comparison.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(metrics["Model"], metrics["RMSE"])
    ax.set_title("Model comparison by root mean squared error (RMSE)")
    ax.set_ylabel("RMSE (lower is better)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_rmse.png", dpi=300)
    plt.close(fig)

    # Figure 4: observed vs predicted for the Poisson model.
    poisson_predictions = predictions["Poisson regression (L2 alpha=1.0)"]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, poisson_predictions)
    lower = min(float(y_test.min()), float(np.min(poisson_predictions)))
    upper = max(float(y_test.max()), float(np.max(poisson_predictions)))
    ax.plot([lower, upper], [lower, upper], linestyle="--", label="Perfect agreement")
    ax.set_title("Observed vs predicted counts: Poisson regression")
    ax.set_xlabel("Observed count")
    ax.set_ylabel("Predicted count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_scatter.png", dpi=300)
    plt.close(fig)

    # Additional simple diagnostic: correlation matrix for numeric analysis variables.
    corr = analysis_data[[TARGET] + FEATURES].corr(numeric_only=True)
    corr.to_csv(output_dir / "webel_more_correlation_matrix.csv")

    print("\nSaved files:")
    for name in [
        "webel_more_model_metrics.csv",
        "webel_more_test_predictions.csv",
        "webel_more_correlation_matrix.csv",
        "fig1_observed_predicted.png",
        "webel_more_observed_vs_predicted.png",
        "fig2_mae.png",
        "fig3_rmse.png",
        "fig4_scatter.png",
    ]:
        print(" -", output_dir / name)

    print("\nInterpretation note: this chronological test evaluates count estimation on the")
    print("last 20% of the available observations. It does not establish city-wide traffic")
    print("forecasting performance or prove that alpha=1.0 is the optimal regularization value.")


if __name__ == "__main__":
    main()
