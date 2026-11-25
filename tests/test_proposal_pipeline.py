import json
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
        classes_ = np.array([0, 1])

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
            "NEUTRAL",
        )
        directions = pipeline.apply_signal_weights(
            pd.DataFrame({
                "Prediction_Probability": [0.45, 0.5, 0.55],
                "Sentiment_Signal": [np.nan, np.nan, np.nan],
                "Headline_Count": [0, 0, 0],
            })
        )["Combined_Direction"].tolist()
        self.assertEqual(directions, ["DOWN", "NEUTRAL", "UP"])

    def test_vader_finance_lexicon_distinguishes_polarity(self):
        self.assertEqual(
            pipeline.weak_sentiment_label(
                "Revenue beat forecasts with strong profit growth"
            ),
            1,
        )
        self.assertEqual(
            pipeline.weak_sentiment_label(
                "Company faces lawsuit, losses, and a downgrade"
            ),
            -1,
        )
        self.assertEqual(
            pipeline.weak_sentiment_label(
                "The company held its quarterly meeting"
            ),
            0,
        )

    def test_news_model_reports_only_independent_manual_holdout_metrics(self):
        manual = pd.DataFrame({
            "Ticker": ["AAPL"] * 12,
            "Headline": [
                "revenue growth beats estimates",
                "shares rise after strong earnings",
                "profit jumps on higher demand",
                "company posts record sales",
                "analysts upgrade strong outlook",
                "dividend increase announced",
                "quarterly losses widen",
                "shares fall after weak guidance",
                "company reports declining revenue",
                "lawsuit triggers downgrade",
                "profit falls amid lower demand",
                "forecast cut after poor results",
            ],
            "Label": [1] * 6 + [-1] * 6,
        })
        weak = pd.DataFrame({
            "Ticker": ["MSFT", "MSFT"],
            "Headline": [
                "another positive headline",
                "another negative headline",
            ],
            "Weak_Label": [1, -1],
        })

        _, metrics = pipeline.train_news_model(weak, manual)

        self.assertIn("manual_holdout_accuracy", metrics)
        self.assertGreater(metrics["manual_holdout_rows"], 0)
        self.assertNotIn("training_accuracy", metrics)

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

    def test_model_bundle_save_backs_up_prior_outputs_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "proposal_pipeline_outputs"
            output.mkdir()
            (output / "existing.txt").write_text("previous", encoding="utf-8")
            with patch.object(pipeline, "OUT", output):
                backup = pipeline.save_models(
                    {"model": "stock"},
                    {"model": "logistic"},
                    {"model": "news"},
                    {"accuracy": 0.5, "stock_features": []},
                    pd.DataFrame({"Date": []}),
                    pd.DataFrame({"Ticker": []}),
                )

            self.assertIsNotNone(backup)
            self.assertEqual(
                (backup / "existing.txt").read_text(encoding="utf-8"),
                "previous",
            )
            self.assertTrue((output / "proposal_stock_model.joblib").exists())
            self.assertTrue((output / "proposal_model_metadata.json").exists())

    def test_model_bundle_save_failure_preserves_active_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "proposal_pipeline_outputs"
            output.mkdir()
            marker = output / "active.txt"
            marker.write_text("active", encoding="utf-8")
            with patch.object(pipeline, "OUT", output):
                with patch.object(
                    pipeline.joblib,
                    "dump",
                    side_effect=OSError("disk full"),
                ):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        pipeline.save_models(
                            {"model": "stock"},
                            {"model": "logistic"},
                            {"model": "news"},
                            {},
                            pd.DataFrame(),
                            pd.DataFrame(),
                        )
            self.assertEqual(marker.read_text(encoding="utf-8"), "active")
            self.assertFalse(
                (Path(directory) / "proposal_pipeline_outputs_backup").exists()
            )

    def test_cli_retraining_requires_explicit_confirmation(self):
        with self.assertRaisesRegex(pipeline.InputError, "Retraining is gated"):
            pipeline.train_command(type("Args", (), {"confirm_retrain": False})())

    def test_manual_news_label_overrides_live_model_sentiment(self):
        prices = sample_prices(days=100)
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
        self.assertAlmostEqual(weights["Signal_Tilted_Weight"].sum(), 1.0)
        self.assertIn("signal_tilted_annualized_volatility", metrics)
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
                "AAPL company reports strong profit growth",
                "AAPL company faces lawsuit and losses",
                "MSFT company shares record gains",
                "MSFT company sees weak demand and decline",
                "AAPL company announces product update",
                "MSFT company announces neutral product update",
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
        json.dumps(metrics, allow_nan=False)
        self.assertEqual(
            set(metrics["model_comparison"]),
            {
                "Majority Class Baseline",
                "Previous-Day Direction",
                "Buy-and-Hold (Always UP)",
                "Logistic Regression",
                "Random Forest",
            },
        )
        self.assertIn(metrics["target_horizon_days"], {1, 5})
        self.assertEqual(set(metrics["target_horizon_selection"]["evaluations"]), {"1", "5"})
        self.assertTrue(metrics["feature_importance"])
        self.assertTrue({
            "Ticker",
            "BuyHold_Cumulative_Value",
            "Random_Forest_Cumulative_Value",
            "Logistic_Regression_Cumulative_Value",
            "Strategy_Equity",
            "Buy_Hold_Equity",
            "Equal_Weight_Equity",
        }.issubset(backtest.columns))
        self.assertTrue(np.isfinite(
            backtest[
                ["Strategy_Equity", "Buy_Hold_Equity", "Equal_Weight_Equity"]
            ].to_numpy()
        ).all())
        self.assertEqual(
            metrics["backtest_transaction_cost_bps"],
            pipeline.TRANSACTION_COST_BPS,
        )
        self.assertNotIn(
            "training_accuracy", metrics["news_model"]
        )
        inferred, _ = pipeline.predict_with_models(
            result[0],
            result[2],
            pipeline.prepare_prices(sample_prices()),
            news,
        )
        self.assertTrue(inferred["Status"].eq("ok").all())
        self.assertTrue(
            np.isfinite(inferred["Prediction_Probability"]).all()
        )
        trades = test_output.loc[test_output["Backtest_Trade"]]
        position = trades["Random_Forest_Probability"].ge(0.5).astype(float)
        turnover = position.groupby(trades["Ticker"]).diff().abs()
        turnover = turnover.fillna(position)
        expected_returns = (
            position * trades["Forward_Return"]
            - turnover * (pipeline.TRANSACTION_COST_BPS / 10_000)
        )
        np.testing.assert_allclose(
            trades["Random_Forest_Strategy_Return"],
            expected_returns,
        )

    def test_incomplete_current_session_bar_is_dropped_before_close(self):
        prices = pd.DataFrame({
            "Date": [
                pd.Timestamp("2026-11-24 00:00", tz="America/New_York"),
                pd.Timestamp("2026-11-25 00:00", tz="America/New_York"),
            ],
            "Ticker": ["AAPL", "AAPL"],
        })

        before_close = pipeline.drop_incomplete_daily_bars(
            prices,
            pd.Timestamp("2026-11-25 15:59", tz="America/New_York"),
        )
        after_close = pipeline.drop_incomplete_daily_bars(
            prices,
            pd.Timestamp("2026-11-25 16:00", tz="America/New_York"),
        )

        self.assertEqual(len(before_close), 1)
        self.assertEqual(len(after_close), 2)

    def test_scale_free_features_and_five_day_target_are_causal(self):
        source = sample_prices(days=90)
        prepared = pipeline.prepare_prices(source)
        cutoff = pd.bdate_range("2022-01-03", periods=65)[-1]
        changed_source = source.copy()
        future_rows = (
            changed_source["Date"].ge(cutoff)
            & changed_source["Ticker"].eq("AAPL")
        )
        changed_source.loc[future_rows, ["Open", "High", "Low", "Close"]] *= 1.2
        changed = pipeline.prepare_prices(changed_source)
        prior = prepared.loc[prepared["Date"] < cutoff].set_index(
            ["Ticker", "Date"]
        )
        changed_prior = changed.loc[changed["Date"] < cutoff].set_index(
            ["Ticker", "Date"]
        )
        np.testing.assert_allclose(
            prior[pipeline.SCALE_FREE_FEATURES],
            changed_prior[pipeline.SCALE_FREE_FEATURES],
            equal_nan=True,
        )

        news = pipeline.prepare_news(
            pd.DataFrame(columns=["Ticker", "Headline"])
        )
        merged = pipeline.merge_daily_sentiment(prepared, news)
        labeled = pipeline.add_targets(
            merged, horizon_days=5, dead_zone=0
        )
        aapl = labeled.loc[labeled["Ticker"].eq("AAPL")].sort_values("Date")
        first = aapl.iloc[0]
        expected_close = (
            prepared.loc[
                prepared["Ticker"].eq("AAPL")
                & prepared["Date"].eq(first["Date"])
            ]["Close"].iloc[0]
        )
        self.assertAlmostEqual(
            first["Next_Close"],
            prepared.loc[
                prepared["Ticker"].eq("AAPL")
                & prepared["Date"].eq(first["Target_Date"])
            ]["Close"].iloc[0],
        )
        self.assertAlmostEqual(
            first["Forward_Return"],
            first["Next_Close"] / expected_close - 1,
        )
        self.assertNotEqual(first["Target_Date"], first["Date"])
        self.assertEqual(
            len(aapl),
            len(
                merged.loc[
                    merged["Ticker"].eq("AAPL")
                ].dropna(
                    subset=pipeline.MODEL_FEATURES_WITHOUT_SENTIMENT
                )
            ) - 5,
        )

    def test_walk_forward_splits_purge_the_target_horizon(self):
        prepared = pipeline.prepare_prices(sample_prices(days=100))
        news = pipeline.prepare_news(
            pd.DataFrame(columns=["Ticker", "Headline"])
        )
        labeled = pipeline.add_targets(
            pipeline.merge_daily_sentiment(prepared, news),
            horizon_days=5,
            dead_zone=0,
        )
        for train_rows, validation_rows in pipeline.time_series_splits_by_date(
            labeled, n_splits=3, gap_days=5
        ):
            train = labeled.iloc[train_rows]
            validation = labeled.iloc[validation_rows]
            self.assertLess(
                train["Target_Date"].max(),
                validation["Date"].min(),
            )

    def test_prediction_skips_ticker_below_minimum_usable_history(self):
        prices = sample_prices(days=30)
        empty_news = pd.DataFrame(columns=["Ticker", "Headline"])

        stock_output, combined = pipeline.predict_with_models(
            self.PositiveModel(),
            self.PositiveModel(),
            prices,
            empty_news,
        )

        self.assertEqual(set(stock_output["Status"]), {"skipped"})
        self.assertTrue(
            stock_output["Message"].str.contains(
                str(pipeline.MIN_HISTORY_ROWS)
            ).all()
        )
        self.assertTrue(combined.empty)

    def test_no_news_is_distinct_from_neutral_news(self):
        prices = pd.DataFrame({
            "Ticker": ["AAPL", "AAPL"],
            "Date": pd.to_datetime(["2022-01-03", "2022-01-04"]),
        })
        news = pd.DataFrame({
            "Ticker": ["AAPL"],
            "News_Date": pd.to_datetime(["2022-01-03"]),
            "Weak_Label": [0],
        })

        merged = pipeline.merge_daily_sentiment(prices, news)

        self.assertEqual(merged.loc[0, "News_Sentiment"], 0)
        self.assertTrue(merged.loc[0, "Has_News"])
        self.assertEqual(merged.loc[0, "News_Count"], 1)
        self.assertTrue(pd.isna(merged.loc[1, "News_Sentiment"]))
        self.assertFalse(merged.loc[1, "Has_News"])
        self.assertEqual(merged.loc[1, "News_Count"], 0)

    def test_probability_column_is_selected_by_class_label(self):
        class ReverseClassOrder:
            classes_ = np.array([1, 0])

            def predict_proba(self, values):
                return np.tile([0.8, 0.2], (len(values), 1))

        probability = pipeline.probability_for_class(
            ReverseClassOrder(),
            pd.DataFrame({"feature": [0]}),
            target_class=1,
        )

        np.testing.assert_allclose(probability, [0.8])
        np.testing.assert_array_equal(
            pipeline.direction_from_probability(
                np.array([0.499, 0.5, 0.501])
            ),
            [0, 1, 1],
        )


if __name__ == "__main__":
    unittest.main()
