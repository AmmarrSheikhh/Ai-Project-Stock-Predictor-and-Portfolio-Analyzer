"""Inspect saved out-of-sample stock predictions without retraining models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "proposal_pipeline_outputs"
MODEL_PATH = OUTPUT_DIR / "proposal_stock_model.joblib"
TEST_PATH = OUTPUT_DIR / "proposal_stock_test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "proposal_metrics.json"


def main() -> int:
    for path in (MODEL_PATH, TEST_PATH, METRICS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required diagnostic input is missing: {path}")

    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(TEST_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    required = {
        "Target_Up",
        "Ticker",
        "Random_Forest_Predicted_Up",
        "Random_Forest_Probability",
    }
    missing = required.difference(test.columns)
    if missing:
        raise ValueError(
            "Saved test predictions are missing columns: "
            + ", ".join(sorted(missing))
        )

    actual = pd.to_numeric(test["Target_Up"], errors="raise").astype(int)
    predicted = pd.to_numeric(
        test["Random_Forest_Predicted_Up"], errors="raise"
    ).astype(int)
    probability_up = pd.to_numeric(
        test["Random_Forest_Probability"], errors="raise"
    )
    classes = np.asarray(model.classes_)
    up_columns = np.flatnonzero(classes == 1)

    print("Saved Random Forest diagnostic (uses persisted held-out predictions)")
    print(
        "The saved model is the final all-data model; test metrics below use "
        "the stored chronological holdout outputs to avoid evaluating it on "
        "rows it has already seen."
    )
    print(f"Model classes_: {classes.tolist()}")
    if len(up_columns) == 1:
        print(
            "The saved model's P(UP) is read from probability column "
            f"{int(up_columns[0])}, whose class label is "
            f"{classes[up_columns[0]]!r}."
        )

    print(f"\nRows - train split: {metrics.get('train_rows', 'not recorded')}")
    print(f"Rows - test split: {len(test)}")
    train_counts = metrics.get("train_class_balance")
    if train_counts:
        print(f"Training class counts: {train_counts}")
    else:
        print("Training class counts: not recorded with these saved artifacts.")
    test_counts = actual.value_counts().sort_index()
    print("Test class balance:")
    for label, count in test_counts.items():
        print(f"  label {label}: {count} ({count / len(test):.2%})")

    majority_baseline = (
        metrics.get("model_comparison", {})
        .get("Majority Class Baseline", {})
        .get("accuracy")
    )
    if majority_baseline is None:
        majority_baseline = float(test_counts.max() / len(test))

    accuracy = accuracy_score(actual, predicted)
    balanced_accuracy = balanced_accuracy_score(actual, predicted)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    correct = int((actual == predicted).sum())

    print(f"\nMajority-class baseline accuracy: {majority_baseline:.4f}")
    print(f"Random Forest accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Predicted UP share: {(predicted == 1).mean():.2%}")
    print("Confusion matrix (rows=true [DOWN, UP], columns=predicted [DOWN, UP]):")
    print(matrix)
    binomial_p_value = binomtest(
        correct,
        len(test),
        p=majority_baseline,
        alternative="greater",
    ).pvalue
    print(
        "Binomial test of accuracy > majority baseline: "
        f"p={binomial_p_value:.6g}"
    )

    print("\nRandom Forest P(UP) distribution:")
    print(probability_up.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

    print("\nPer-ticker held-out results:")
    for ticker, group in test.groupby("Ticker", sort=True):
        ticker_actual = pd.to_numeric(group["Target_Up"], errors="raise").astype(int)
        ticker_predicted = pd.to_numeric(
            group["Random_Forest_Predicted_Up"], errors="raise"
        ).astype(int)
        print(
            f"  {ticker}: rows={len(group)}, "
            f"UP labels={(ticker_actual == 1).sum()}, "
            f"predicted UP={(ticker_predicted == 1).mean():.2%}, "
            f"accuracy={accuracy_score(ticker_actual, ticker_predicted):.2%}, "
            f"balanced accuracy="
            f"{balanced_accuracy_score(ticker_actual, ticker_predicted):.2%}, "
            f"F1={f1_score(ticker_actual, ticker_predicted, zero_division=0):.2%}"
        )

    if "News_Sentiment_Count" in test.columns:
        news_count = pd.to_numeric(
            test["News_Sentiment_Count"], errors="coerce"
        ).fillna(0)
        print(
            "\nTest rows with News_Sentiment_Count > 0: "
            f"{(news_count > 0).sum()} / {len(test)} ({(news_count > 0).mean():.2%})"
        )
    else:
        print("\nNews_Sentiment_Count is absent from the saved test predictions.")

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        calibrated_estimators = [
            getattr(calibrated, "estimator", None)
            for calibrated in getattr(model, "calibrated_classifiers_", [])
        ]
        calibrated_importances = [
            getattr(estimator, "feature_importances_", None)
            for estimator in calibrated_estimators
            if estimator is not None
        ]
        calibrated_importances = [
            values for values in calibrated_importances if values is not None
        ]
        if calibrated_importances:
            importances = np.mean(calibrated_importances, axis=0)
    if importances is not None:
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None:
            feature_names = metrics.get("stock_features", [])
        if len(feature_names) != len(importances):
            raise ValueError(
                "Saved model feature names do not match its feature importances."
            )
        print("\nSaved final model feature importances (descending):")
        for name, importance in sorted(
            zip(feature_names, importances),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"  {name}: {importance:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
