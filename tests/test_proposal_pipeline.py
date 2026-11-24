import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import proposal_pipeline as pipeline


def sample_prices(days=180):
    dates = pd.bdate_range("2022-01-03", periods=days)
    random = np.random.default_rng(42)
    frames = []
    for ticker, drift, volatility in [
        ("AAPL", 0.0004, 0.012),
        ("MSFT", 0.0002, 0.009),
    ]:
        returns = random.normal(drift, volatility, len(dates))
        close = 100 * np.cumprod(1 + returns)
        frames.append(pd.DataFrame({
            "Date": dates,
            "Ticker": ticker,
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        }))
    return pd.concat(frames, ignore_index=True)


class ProposalPipelineTests(unittest.TestCase):
    class PositiveModel:
        def predict(self, values):
            return np.ones(len(values), dtype=int)

        def predict_proba(self, values):
            return np.tile([0.3, 0.7], (len(values), 1))

    def test_signal_weight_changes_combined_direction(self):
        combined = pd.DataFrame({
            "Prediction_Probability": [0.6],
            "Sentiment_Signal": [0.1],
        })

        self.assertEqual(
            pipeline.apply_signal_weights(combined, 0.6).iloc[0][
                "Combined_Direction"
            ],
            "DOWN",
        )
        self.assertEqual(
            pipeline.apply_signal_weights(combined, 0.8).iloc[0][
                "Combined_Direction"
            ],
            "UP",
        )

    def test_manual_news_labels_are_persisted_and_override_weak_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "labels.csv"
            with patch.object(pipeline, "NEWS_LABELS_PATH", labels_path):
                pipeline.save_manual_news_labels(pd.DataFrame({
                    "Ticker": ["aapl"],
                    "Headline": ["Strong   quarterly gains"],
                    "Label": [-1],
                }))
                reviewed = pipeline.apply_manual_news_labels(pd.DataFrame({
                    "Ticker": ["AAPL"],
                    "Headline": ["strong quarterly gains"],
                    "Weak_Label": [1],
                }))

        self.assertEqual(reviewed.loc[0, "Manual_Label"], -1)
        self.assertEqual(reviewed.loc[0, "Weak_Label"], -1)

    def test_training_history_appends_successful_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.csv"
            with patch.object(pipeline, "TRAINING_HISTORY_PATH", history_path):
                pipeline.append_training_record({
                    "Trained_At_UTC": "2026-11-24T10:00:00+00:00",
                    "RF_F1": 0.51,
                })
                pipeline.append_training_record({
                    "Trained_At_UTC": "2026-11-24T11:00:00+00:00",
                    "RF_F1": 0.53,
                })
                history = pd.read_csv(history_path)

        self.assertEqual(len(history), 2)
        self.assertEqual(history.iloc[-1]["RF_F1"], 0.53)

    def test_manual_news_label_overrides_live_model_sentiment(self):
        prices = sample_prices(days=50)
        headline = "AAPL announces strong quarterly gains"
        news = pd.DataFrame({
            "Ticker": ["AAPL"],
            "Headline": [headline],
            "PublishedAt": [prices["Date"].max()],
        })
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "labels.csv"
            with patch.object(pipeline, "NEWS_LABELS_PATH", labels_path):
                pipeline.save_manual_news_labels(pd.DataFrame({
                    "Ticker": ["AAPL"],
                    "Headline": [headline],
                    "Label": [-1],
                }))
                _, combined = pipeline.predict_with_models(
                    self.PositiveModel(),
                    self.PositiveModel(),
                    prices,
                    news,
                    technical_weight=0.5,
                )

        apple = combined.loc[combined["Ticker"] == "AAPL"].iloc[0]
        self.assertEqual(apple["Predicted_Sentiment_Mean"], -1)
        self.assertEqual(apple["Combined_Direction"], "DOWN")

    def test_basket_optimizer_and_historical_projection(self):
        prices = sample_prices()
        weights, _, _, metrics = pipeline.build_portfolio(
            prices,
            selected_tickers=["AAPL", "MSFT"],
        )
        self.assertEqual(set(weights["Ticker"]), {"AAPL", "MSFT"})
        self.assertAlmostEqual(weights["Markowitz_Weight"].sum(), 1.0)
        self.assertLessEqual(
            metrics["markowitz_annualized_volatility"],
            metrics["equal_weight_annualized_volatility"] + 1e-8,
        )

        projection = pipeline.project_investment(
            prices,
            weights.set_index("Ticker")["Markowitz_Weight"],
            initial_amount=10_000,
            years=5,
        )
        self.assertGreater(projection["low"], 0)
        self.assertLessEqual(projection["low"], projection["base"])
        self.assertLessEqual(projection["base"], projection["high"])

    def test_training_compares_models_and_saves_cost_adjusted_curves(self):
        news = pipeline.prepare_news(pd.DataFrame({
            "Ticker": ["AAPL", "AAPL", "MSFT", "MSFT", "AAPL", "MSFT"],
            "Headline": [
                "company reports strong profit growth",
                "company faces lawsuit and losses",
                "company shares record gains",
                "company sees weak demand and decline",
                "company announces product update",
                "company announces neutral product update",
            ],
        }))
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "labels.csv"
            with patch.object(pipeline, "NEWS_LABELS_PATH", labels_path):
                result = pipeline.train_and_evaluate(
                    pipeline.prepare_prices(sample_prices()),
                    news,
                )

        metrics = result[3]
        test_output = result[4]
        backtest = result[5]
        self.assertEqual(
            set(metrics["model_comparison"]),
            {
                "Majority Class Baseline",
                "Logistic Regression",
                "Random Forest",
            },
        )
        self.assertTrue({
            "Ticker",
            "BuyHold_Cumulative_Value",
            "Random_Forest_Cumulative_Value",
            "Logistic_Regression_Cumulative_Value",
        }.issubset(backtest.columns))
        self.assertEqual(metrics["backtest_transaction_cost_bps"], 10)
        position = test_output["Random_Forest_Probability"].gt(0.5).astype(float)
        expected_returns = (
            position * test_output["Next_Day_Return"]
            - position.groupby(test_output["Ticker"]).diff().abs().fillna(0) * 0.001
        )
        np.testing.assert_allclose(
            test_output["Random_Forest_Strategy_Return"],
            expected_returns,
        )


if __name__ == "__main__":
    unittest.main()
