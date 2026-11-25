from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "proposal_pipeline_outputs"
NEWS_LABELS_PATH = OUT / "proposal_news_labels.csv"
TRAINING_HISTORY_PATH = OUT / "proposal_training_history.csv"
TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA"]

BASE_FEATURES = [
    "Daily_Return",
    "MA_10",
    "MA_20",
    "Price_Range",
    "Volume_Change",
    "Close_MA10_Ratio",
    "Close_MA20_Ratio",
    "RSI_14",
    "MACD",
    "MACD_Signal",
]
STOCK_FEATURES = BASE_FEATURES + ["News_Sentiment", "News_Sentiment_Count"]

POSITIVE_WORDS = {
    "beat", "growth", "gain", "gains", "buy", "bull", "bullish", "strong",
    "success", "outlook", "upgrade", "record", "profit", "positive", "rise",
}
NEGATIVE_WORDS = {
    "fall", "falls", "loss", "losses", "risk", "warning", "downgrade",
    "weak", "decline", "lawsuit", "cut", "cuts", "negative", "crisis",
}


class InputError(ValueError):
    pass


def load_manual_news_labels() -> pd.DataFrame:
    if not NEWS_LABELS_PATH.exists():
        return pd.DataFrame(columns=["Ticker", "Headline", "Label"])
    labels = pd.read_csv(NEWS_LABELS_PATH)
    required = {"Ticker", "Headline", "Label"}
    if not required.issubset(labels.columns):
        raise InputError(
            f"Manual news labels must contain: {', '.join(sorted(required))}."
        )
    labels["Label"] = pd.to_numeric(labels["Label"], errors="coerce")
    if labels["Label"].isna().any() or not labels["Label"].isin([-1, 0, 1]).all():
        raise InputError("Saved manual news labels must be -1, 0, or 1.")
    return labels[["Ticker", "Headline", "Label"]]


def save_manual_news_labels(labels: pd.DataFrame) -> None:
    required = {"Ticker", "Headline", "Label"}
    if not required.issubset(labels.columns):
        raise InputError(
            f"Manual news labels must contain: {', '.join(sorted(required))}."
        )
    cleaned = labels.loc[:, ["Ticker", "Headline", "Label"]].copy()
    cleaned["Ticker"] = cleaned["Ticker"].fillna("").astype(str).str.strip().str.upper()
    cleaned["Headline"] = cleaned["Headline"].fillna("").astype(str).str.strip()
    cleaned["Label"] = pd.to_numeric(cleaned["Label"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Label"])
    if not cleaned["Label"].isin([-1, 0, 1]).all():
        raise InputError("Manual news labels must be -1, 0, or 1.")
    cleaned = cleaned[
        cleaned["Ticker"].ne("") & cleaned["Headline"].ne("")
    ].drop_duplicates(["Ticker", "Headline"], keep="last")

    OUT.mkdir(parents=True, exist_ok=True)
    temporary = NEWS_LABELS_PATH.with_suffix(".csv.tmp")
    cleaned.to_csv(temporary, index=False)
    temporary.replace(NEWS_LABELS_PATH)


def _news_label_key(frame: pd.DataFrame) -> pd.Series:
    ticker = frame["Ticker"].fillna("").astype(str).str.strip().str.upper()
    headline = (
        frame["Headline"].fillna("").astype(str).str.lower()
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return ticker + "\0" + headline


def update_manual_news_labels(reviewed_news: pd.DataFrame) -> None:
    required = {"Ticker", "Headline", "Label"}
    if not required.issubset(reviewed_news.columns):
        raise InputError(
            f"Reviewed news must contain: {', '.join(sorted(required))}."
        )
    current = reviewed_news.loc[:, ["Ticker", "Headline", "Label"]].copy()
    current["Label"] = pd.to_numeric(current["Label"], errors="coerce")
    current_keys = set(_news_label_key(current))
    existing = load_manual_news_labels()
    if not existing.empty:
        existing = existing.loc[~_news_label_key(existing).isin(current_keys)]
    save_manual_news_labels(pd.concat([existing, current], ignore_index=True))


def apply_manual_news_labels(
    news: pd.DataFrame, labels: pd.DataFrame | None = None
) -> pd.DataFrame:
    prepared = news.copy()
    if "Weak_Label" not in prepared.columns:
        raise InputError("News data must be prepared before applying manual labels.")
    if labels is None:
        labels = load_manual_news_labels()

    prepared["Manual_Label"] = np.nan
    if labels.empty or prepared.empty:
        return prepared

    keyed_labels = labels.copy()
    keyed_labels["Ticker"] = keyed_labels["Ticker"].fillna("").astype(str)
    keyed_labels["Headline"] = keyed_labels["Headline"].fillna("").astype(str)
    label_values = dict(zip(_news_label_key(keyed_labels), keyed_labels["Label"]))
    prepared["Manual_Label"] = _news_label_key(prepared).map(label_values)
    has_manual_label = prepared["Manual_Label"].notna()
    prepared.loc[has_manual_label, "Weak_Label"] = prepared.loc[
        has_manual_label, "Manual_Label"
    ]
    return prepared


def read_table(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
    if not file_path.exists():
        raise InputError(f"File does not exist: {file_path}")

    with file_path.open("rb") as stream:
        header = stream.read(8)
    if header.startswith(b"PK") or header.startswith(b"\xD0\xCF"):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def weak_sentiment_label(text: str) -> int:
    tokens = set(re.findall(r"[a-z]+", str(text).lower()))
    score = len(tokens & POSITIVE_WORDS) - len(tokens & NEGATIVE_WORDS)
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


def prepare_prices(source: pd.DataFrame) -> pd.DataFrame:
    prices = source.copy()
    required = {"Date", "Ticker", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(prices.columns)
    if missing:
        raise InputError(
            "Price data is missing required columns: " + ", ".join(sorted(missing))
        )

    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce", utc=True)
    prices["Date"] = prices["Date"].dt.tz_localize(None)
    prices["Ticker"] = prices["Ticker"].astype(str).str.strip().str.upper()
    numeric = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = prices.dropna(subset=["Date", "Ticker"] + numeric)
    prices = prices.sort_values(["Ticker", "Date"])

    prices["Daily_Return"] = prices.groupby("Ticker")["Close"].pct_change()
    prices["MA_10"] = prices.groupby("Ticker")["Close"].transform(
        lambda values: values.rolling(10, min_periods=10).mean()
    )
    prices["MA_20"] = prices.groupby("Ticker")["Close"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    prices["Price_Range"] = (prices["High"] - prices["Low"]) / prices["Close"]
    prices["Volume_Change"] = prices.groupby("Ticker")["Volume"].pct_change()
    prices["Close_MA10_Ratio"] = prices["Close"] / prices["MA_10"] - 1
    prices["Close_MA20_Ratio"] = prices["Close"] / prices["MA_20"] - 1

    change = prices.groupby("Ticker")["Close"].diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    average_gain = gains.groupby(prices["Ticker"]).transform(
        lambda values: values.ewm(
            alpha=1 / 14, adjust=False, min_periods=14
        ).mean()
    )
    average_loss = losses.groupby(prices["Ticker"]).transform(
        lambda values: values.ewm(
            alpha=1 / 14, adjust=False, min_periods=14
        ).mean()
    )
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    prices["RSI_14"] = 100 - (100 / (1 + relative_strength))

    ema_12 = prices.groupby("Ticker")["Close"].transform(
        lambda values: values.ewm(span=12, adjust=False).mean()
    )
    ema_26 = prices.groupby("Ticker")["Close"].transform(
        lambda values: values.ewm(span=26, adjust=False).mean()
    )
    prices["MACD"] = ema_12 - ema_26
    prices["MACD_Signal"] = prices.groupby("Ticker")["MACD"].transform(
        lambda values: values.ewm(span=9, adjust=False).mean()
    )

    prices[BASE_FEATURES] = prices[BASE_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    prices[BASE_FEATURES] = prices.groupby("Ticker")[BASE_FEATURES].transform(
        lambda frame: frame.ffill()
    )
    return prices


def download_live_prices(period: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as error:
        raise InputError(
            "Live mode requires yfinance. Install it with: "
            "python -m pip install yfinance"
        ) from error

    frames = []
    for ticker in TICKERS:
        history = yf.Ticker(ticker).history(
            period=period, auto_adjust=False, actions=False
        )
        if history.empty:
            raise InputError(f"No live price data was returned for {ticker}.")
        history = history.reset_index()
        history["Date"] = pd.to_datetime(history["Date"], utc=True).dt.tz_localize(None)
        history["Ticker"] = ticker
        frames.append(history[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]])
    return pd.concat(frames, ignore_index=True)


def download_live_news() -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as error:
        raise InputError(
            "Live news requires yfinance. Install it with: "
            "python -m pip install yfinance"
        ) from error

    rows = []
    for ticker in TICKERS:
        for item in yf.Search(ticker, max_results=25).news:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            rows.append({
                "Ticker": ticker,
                "Headline": title,
                "PublishedAt": pd.to_datetime(
                    item.get("providerPublishTime"),
                    unit="s",
                    errors="coerce",
                    utc=True,
                ),
            })
    if not rows:
        raise InputError("Live news download returned no headlines.")
    return pd.DataFrame(rows)


def prepare_news(source: pd.DataFrame | None) -> pd.DataFrame:
    if source is None:
        source = download_live_news()
    news = source.copy()
    if "Headline" not in news.columns and "Clean_Headline" in news.columns:
        news = news.rename(columns={"Clean_Headline": "Headline"})
    if "Headline" not in news.columns:
        raise InputError("News data must contain a Headline column.")
    news["Headline"] = news["Headline"].fillna("").astype(str)
    news = news[news["Headline"].str.strip().ne("")]

    if "Ticker" not in news.columns:
        news["Ticker"] = "UNKNOWN"
    news["Ticker"] = news["Ticker"].fillna("UNKNOWN").astype(str).str.strip().str.upper()

    if "PublishedAt" in news.columns:
        news["PublishedAt"] = pd.to_datetime(
            news["PublishedAt"], errors="coerce", utc=True
        )
        news["News_Date"] = news["PublishedAt"].dt.tz_localize(None).dt.normalize()

    if "Sentiment" in news.columns:
        labels = pd.to_numeric(news["Sentiment"], errors="coerce")
        news["Weak_Label"] = labels.where(labels.isin([-1, 0, 1]))
    elif "Label" in news.columns:
        label_map = {"negative": -1, "neutral": 0, "positive": 1}
        news["Weak_Label"] = (
            news["Label"].astype(str).str.lower().str.strip().map(label_map)
        )
    else:
        news["Weak_Label"] = news["Headline"].map(weak_sentiment_label)

    return news.dropna(subset=["Weak_Label"])


def merge_daily_sentiment(prices: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    if "News_Date" not in news.columns:
        news["News_Date"] = pd.NaT

    dated = news.dropna(subset=["News_Date", "Ticker"]).copy()
    dated["News_Date"] = pd.to_datetime(dated["News_Date"]).dt.normalize()

    daily = (
        dated.groupby(["Ticker", "News_Date"], as_index=False)
        .agg(
            News_Sentiment=("Weak_Label", "mean"),
            News_Sentiment_Count=("Weak_Label", "size"),
        )
        .rename(columns={"News_Date": "Date"})
    )
    daily["Date"] = pd.to_datetime(daily["Date"]).dt.tz_localize(None)

    merged = prices.merge(daily, on=["Ticker", "Date"], how="left")
    merged["News_Sentiment"] = merged["News_Sentiment"].fillna(0)
    merged["News_Sentiment_Count"] = (
        merged["News_Sentiment_Count"].fillna(0).astype(int)
    )
    return merged


def train_news_model(news: pd.DataFrame) -> tuple[Pipeline, dict]:
    if news["Weak_Label"].nunique() < 2:
        raise InputError(
            "News data needs at least two sentiment labels. Supply Sentiment "
            "(-1/0/1) or Label (negative/neutral/positive)."
        )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    model.fit(news["Headline"], news["Weak_Label"].astype(int))
    predicted = model.predict(news["Headline"])
    return model, {
        "training_rows": int(len(news)),
        "training_accuracy": float(
            accuracy_score(news["Weak_Label"].astype(int), predicted)
        ),
        "label_distribution": {
            str(label): int(count)
            for label, count in news["Weak_Label"].value_counts().sort_index().items()
        },
    }


def add_targets(prices: pd.DataFrame) -> pd.DataFrame:
    labeled = prices.copy()
    labeled["Next_Close"] = labeled.groupby("Ticker")["Close"].shift(-1)
    labeled["Target_Up"] = (labeled["Next_Close"] > labeled["Close"]).astype(float)
    labeled.loc[labeled["Next_Close"].isna(), "Target_Up"] = np.nan
    return labeled.dropna(subset=STOCK_FEATURES + ["Target_Up"])


def train_and_evaluate(
    prices: pd.DataFrame, news: pd.DataFrame
) -> tuple[
    RandomForestClassifier, Pipeline, Pipeline, dict, pd.DataFrame, pd.DataFrame
]:
    news = apply_manual_news_labels(news)
    manual_training = load_manual_news_labels().rename(
        columns={"Label": "Weak_Label"}
    )
    news_training = pd.concat(
        [
            news[["Ticker", "Headline", "Weak_Label"]],
            manual_training[["Ticker", "Headline", "Weak_Label"]],
        ],
        ignore_index=True,
    )
    news_training["_label_key"] = _news_label_key(news_training)
    news_training = news_training.drop_duplicates(
        "_label_key", keep="last"
    ).drop(columns="_label_key")
    news_model, news_metrics = train_news_model(news_training)
    news_metrics["manual_label_rows"] = int(len(manual_training))
    merged = merge_daily_sentiment(prices, news)
    labeled = add_targets(merged)
    if len(labeled) < 100:
        raise InputError("At least 100 labeled stock observations are required.")

    dates = np.sort(labeled["Date"].unique())
    cutoff = dates[int(len(dates) * 0.8)]
    train = labeled[labeled["Date"] < cutoff]
    test = labeled[labeled["Date"] >= cutoff]
    if train.empty or test.empty:
        raise InputError("The data cannot produce a chronological 80/20 split.")
    if train["Target_Up"].nunique() < 2:
        raise InputError("Stock training data must contain both up and down targets.")

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[STOCK_FEATURES], train["Target_Up"].astype(int))
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    logistic_model.fit(train[STOCK_FEATURES], train["Target_Up"].astype(int))

    rf_probabilities = model.predict_proba(test[STOCK_FEATURES])[:, 1]
    rf_predictions = model.predict(test[STOCK_FEATURES])
    lr_probabilities = logistic_model.predict_proba(test[STOCK_FEATURES])[:, 1]
    lr_predictions = logistic_model.predict(test[STOCK_FEATURES])
    test_output = test.copy()
    test_output["Predicted_Up"] = rf_predictions
    test_output["Prediction_Probability"] = rf_probabilities
    test_output["Random_Forest_Predicted_Up"] = rf_predictions
    test_output["Random_Forest_Probability"] = rf_probabilities
    test_output["Logistic_Regression_Predicted_Up"] = lr_predictions
    test_output["Logistic_Regression_Probability"] = lr_probabilities

    ticker_stability = {}
    for ticker, group in test_output.groupby("Ticker"):
        ticker_stability[ticker] = {
            "rows": int(len(group)),
            "accuracy": float(
                accuracy_score(group["Target_Up"], group["Predicted_Up"])
            ),
            "f1": float(
                f1_score(
                    group["Target_Up"],
                    group["Predicted_Up"],
                    zero_division=0,
                )
            ),
        }

    test_output = test_output.sort_values(["Date", "Ticker"])
    test_output["Next_Day_Return"] = (
        test_output["Next_Close"] / test_output["Close"]
    ) - 1
    test_output["BuyHold_Cumulative_Value"] = (
        (1 + test_output["Next_Day_Return"].fillna(0))
        .groupby(test_output["Ticker"]).cumprod()
    )
    transaction_cost = 0.001
    strategy_returns = {}
    for prefix, probability_column in [
        ("Random_Forest", "Random_Forest_Probability"),
        ("Logistic_Regression", "Logistic_Regression_Probability"),
    ]:
        position = test_output[probability_column].gt(0.5).astype(float)
        turnover = position.groupby(test_output["Ticker"]).diff().abs().fillna(0)
        returns = (
            position * test_output["Next_Day_Return"].fillna(0)
            - turnover * transaction_cost
        )
        test_output[f"{prefix}_Strategy_Return"] = returns
        test_output[f"{prefix}_Cumulative_Value"] = (
            (1 + returns).groupby(test_output["Ticker"]).cumprod()
        )
        strategy_returns[prefix] = returns

    test_output["Strategy_Return"] = test_output["Random_Forest_Strategy_Return"]
    test_output["Strategy_Cumulative_Return"] = (
        test_output["Random_Forest_Cumulative_Value"]
    )
    test_output["BuyHold_Cumulative_Return"] = (
        test_output["BuyHold_Cumulative_Value"]
    )
    strategy_daily = (
        test_output.assign(Strategy_Return=strategy_returns["Random_Forest"])
        .groupby("Date")["Strategy_Return"].mean()
    )
    logistic_daily = (
        test_output.assign(
            Strategy_Return=strategy_returns["Logistic_Regression"]
        )
        .groupby("Date")["Strategy_Return"].mean()
    )
    buy_hold_daily = test_output.groupby("Date")["Next_Day_Return"].mean()

    strategy_curve = (1 + strategy_daily.fillna(0)).cumprod()
    logistic_curve = (1 + logistic_daily.fillna(0)).cumprod()
    buy_hold_curve = (1 + buy_hold_daily.fillna(0)).cumprod()

    backtest = pd.DataFrame({
        "Date": test_output["Date"].to_numpy(),
        "Ticker": test_output["Ticker"].to_numpy(),
        "BuyHold_Cumulative_Value": test_output[
            "BuyHold_Cumulative_Value"
        ].to_numpy(),
        "Random_Forest_Cumulative_Value": test_output[
            "Random_Forest_Cumulative_Value"
        ].to_numpy(),
        "Logistic_Regression_Cumulative_Value": test_output[
            "Logistic_Regression_Cumulative_Value"
        ].to_numpy(),
        "Strategy_Cumulative_Return": test_output[
            "Random_Forest_Cumulative_Value"
        ].to_numpy(),
        "BuyHold_Cumulative_Return": test_output[
            "BuyHold_Cumulative_Value"
        ].to_numpy(),
    })

    final_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(labeled[STOCK_FEATURES], labeled["Target_Up"].astype(int))
    final_logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    final_logistic_model.fit(
        labeled[STOCK_FEATURES], labeled["Target_Up"].astype(int)
    )

    model_comparison = {
        "Majority Class Baseline": {
            "accuracy": float(accuracy_score(
                test["Target_Up"], np.full(len(test), train["Target_Up"].mode()[0])
            )),
            "f1": float(f1_score(
                test["Target_Up"],
                np.full(len(test), train["Target_Up"].mode()[0]),
                zero_division=0,
            )),
        },
        "Logistic Regression": {
            "accuracy": float(accuracy_score(
                test_output["Target_Up"],
                test_output["Logistic_Regression_Predicted_Up"],
            )),
            "f1": float(f1_score(
                test_output["Target_Up"],
                test_output["Logistic_Regression_Predicted_Up"],
                zero_division=0,
            )),
        },
        "Random Forest": {
            "accuracy": float(accuracy_score(
                test_output["Target_Up"],
                test_output["Random_Forest_Predicted_Up"],
            )),
            "f1": float(f1_score(
                test_output["Target_Up"],
                test_output["Random_Forest_Predicted_Up"],
                zero_division=0,
            )),
        },
    }
    metrics = {
        "stock_features": STOCK_FEATURES,
        "split": "chronological 80/20",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "final_training_rows": int(len(labeled)),
        "accuracy": float(
            accuracy_score(
                test_output["Target_Up"],
                test_output["Random_Forest_Predicted_Up"],
            )
        ),
        "f1": float(
            f1_score(
                test_output["Target_Up"],
                test_output["Random_Forest_Predicted_Up"],
                zero_division=0,
            )
        ),
        "model_comparison": model_comparison,
        "ticker_stability": ticker_stability,
        "strategy_final_return": float(
            strategy_curve.iloc[-1] - 1
        ) if len(strategy_curve) else 0.0,
        "logistic_strategy_final_return": float(
            logistic_curve.iloc[-1] - 1
        ) if len(logistic_curve) else 0.0,
        "buy_hold_final_return": float(
            buy_hold_curve.iloc[-1] - 1
        ) if len(buy_hold_curve) else 0.0,
        "backtest_transaction_cost_bps": 10,
        "news_model": news_metrics,
    }
    return (
        final_model,
        final_logistic_model,
        news_model,
        metrics,
        test_output,
        backtest,
    )


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_training_record(record: dict[str, str | int | float]) -> None:
    if "Trained_At_UTC" not in record:
        raise InputError("Training history records need a UTC timestamp.")
    OUT.mkdir(parents=True, exist_ok=True)
    if TRAINING_HISTORY_PATH.exists():
        history = pd.read_csv(TRAINING_HISTORY_PATH)
    else:
        history = pd.DataFrame()
    history = pd.concat([history, pd.DataFrame([record])], ignore_index=True)
    temporary = TRAINING_HISTORY_PATH.with_suffix(".csv.tmp")
    history.to_csv(temporary, index=False)
    temporary.replace(TRAINING_HISTORY_PATH)


def save_models(
    stock_model: RandomForestClassifier,
    logistic_model: Pipeline,
    news_model: Pipeline,
    metrics: dict,
    backtest: pd.DataFrame,
    test_output: pd.DataFrame,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stock_path = OUT / "proposal_stock_model.joblib"
    logistic_path = OUT / "proposal_logistic_model.joblib"
    news_path = OUT / "proposal_news_model.joblib"
    joblib.dump(stock_model, stock_path)
    joblib.dump(logistic_model, logistic_path)
    joblib.dump(news_model, news_path)
    write_json(OUT / "proposal_metrics.json", metrics)
    backtest.to_csv(OUT / "proposal_backtest.csv", index=False)
    test_output.to_csv(OUT / "proposal_stock_test_predictions.csv", index=False)


def load_models() -> tuple[RandomForestClassifier, Pipeline | None, Pipeline]:
    stock_path = OUT / "proposal_stock_model.joblib"
    logistic_path = OUT / "proposal_logistic_model.joblib"
    news_path = OUT / "proposal_news_model.joblib"
    if not stock_path.exists() or not news_path.exists():
        raise InputError(
            "Proposal models are not trained yet. Run: "
            "python proposal_pipeline.py train"
        )
    logistic_model = joblib.load(logistic_path) if logistic_path.exists() else None
    return joblib.load(stock_path), logistic_model, joblib.load(news_path)


def apply_signal_weights(
    combined: pd.DataFrame, technical_weight: float = 0.7
) -> pd.DataFrame:
    if not 0 <= technical_weight <= 1:
        raise InputError("Technical signal weight must be between 0 and 1.")
    weighted = combined.copy()
    weighted["Combined_Signal"] = (
        technical_weight * weighted["Prediction_Probability"]
        + (1 - technical_weight) * weighted["Sentiment_Signal"]
    )
    weighted["Combined_Direction"] = np.where(
        weighted["Combined_Signal"] >= 0.5,
        "UP",
        "DOWN",
    )
    return weighted


def predict_with_models(
    stock_model: RandomForestClassifier | Pipeline,
    news_model: Pipeline,
    prices: pd.DataFrame,
    news: pd.DataFrame,
    technical_weight: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_prices = prepare_prices(prices)
    prepared_news = apply_manual_news_labels(prepare_news(news))

    news_predictions = prepared_news.copy()
    news_predictions["Predicted_Sentiment"] = news_model.predict(
        news_predictions["Headline"]
    )
    manually_labeled = news_predictions["Manual_Label"].notna()
    news_predictions.loc[manually_labeled, "Predicted_Sentiment"] = (
        news_predictions.loc[manually_labeled, "Manual_Label"]
    )
    news_predictions["Sentiment_Confidence"] = news_model.predict_proba(
        news_predictions["Headline"]
    ).max(axis=1)
    news_predictions.loc[manually_labeled, "Sentiment_Confidence"] = 1.0

    merged = merge_daily_sentiment(prepared_prices, prepared_news)
    usable = merged.dropna(subset=STOCK_FEATURES)
    latest = (
        usable.sort_values(["Ticker", "Date"])
        .groupby("Ticker", as_index=False)
        .tail(1)
    )
    stock_probabilities = stock_model.predict_proba(
        latest[STOCK_FEATURES]
    )[:, 1]
    stock_predictions = stock_model.predict(latest[STOCK_FEATURES])
    stock_output = pd.DataFrame({
        "Ticker": latest["Ticker"].to_numpy(),
        "Price_Date": latest["Date"].to_numpy(),
        "Close": latest["Close"].to_numpy(),
        "Predicted_Up": stock_predictions.astype(int),
        "Prediction_Probability": stock_probabilities,
        "RSI_14": latest["RSI_14"].to_numpy(),
        "MACD": latest["MACD"].to_numpy(),
        "News_Sentiment": latest["News_Sentiment"].to_numpy(),
        "News_Sentiment_Count": latest["News_Sentiment_Count"].to_numpy(),
    })

    news_summary = (
        news_predictions.groupby("Ticker", as_index=False)
        .agg(
            Predicted_Sentiment_Mean=("Predicted_Sentiment", "mean"),
            Headline_Count=("Predicted_Sentiment", "size"),
            Mean_Sentiment_Confidence=("Sentiment_Confidence", "mean"),
        )
    )
    combined = stock_output.merge(news_summary, on="Ticker", how="left")
    combined["Predicted_Sentiment_Mean"] = (
        combined["Predicted_Sentiment_Mean"].fillna(0)
    )
    combined["Headline_Count"] = (
        combined["Headline_Count"].fillna(0).astype(int)
    )
    combined["Mean_Sentiment_Confidence"] = (
        combined["Mean_Sentiment_Confidence"].fillna(0)
    )
    combined["Sentiment_Signal"] = (
        combined["Predicted_Sentiment_Mean"] + 1
    ) / 2
    return stock_output, apply_signal_weights(combined, technical_weight)


def build_portfolio(
    prices: pd.DataFrame,
    combined: pd.DataFrame | None = None,
    selected_tickers: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    prepared = prepare_prices(prices)
    returns = prepared.pivot_table(
        index="Date",
        columns="Ticker",
        values="Daily_Return",
    )
    requested_tickers = TICKERS if selected_tickers is None else selected_tickers
    unknown_tickers = set(requested_tickers).difference(TICKERS)
    if unknown_tickers:
        raise InputError(
            "Unsupported portfolio tickers: " + ", ".join(sorted(unknown_tickers))
        )
    returns = returns.reindex(
        columns=[ticker for ticker in requested_tickers if ticker in returns.columns]
    ).dropna(how="any")
    if len(returns.columns) < 2:
        raise InputError(
            "Portfolio analysis requires at least two tickers with aligned history."
        )

    covariance = returns.cov() * 252
    correlation = returns.corr()
    equal_weights = pd.Series(
        1 / len(covariance),
        index=covariance.index,
    )
    annual_volatility = np.sqrt(np.diag(covariance))
    inverse_volatility = 1 / np.maximum(annual_volatility, 1e-12)
    inverse_weights = pd.Series(
        inverse_volatility / inverse_volatility.sum(),
        index=covariance.index,
    )
    optimization = minimize(
        lambda candidate: float(
            candidate @ covariance.to_numpy() @ candidate
        ),
        x0=equal_weights.to_numpy(),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(equal_weights),
        constraints={"type": "eq", "fun": lambda candidate: candidate.sum() - 1},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not optimization.success:
        raise InputError(
            "Minimum-variance optimization failed: " + optimization.message
        )
    markowitz_weights = pd.Series(
        optimization.x,
        index=covariance.index,
    )

    equal_risk = float(
        np.sqrt(
            equal_weights.to_numpy()
            @ covariance.to_numpy()
            @ equal_weights.to_numpy()
        )
    )
    inverse_risk = float(
        np.sqrt(
            inverse_weights.to_numpy()
            @ covariance.to_numpy()
            @ inverse_weights.to_numpy()
        )
    )
    markowitz_risk = float(
        np.sqrt(
            markowitz_weights.to_numpy()
            @ covariance.to_numpy()
            @ markowitz_weights.to_numpy()
        )
    )
    risk_reduction = (
        (equal_risk - inverse_risk) / equal_risk * 100
        if equal_risk > 0
        else 0
    )

    weights = pd.DataFrame({
        "Ticker": covariance.index,
        "Equal_Weight": equal_weights.to_numpy(),
        "Inverse_Volatility_Weight": inverse_weights.to_numpy(),
        "Markowitz_Weight": markowitz_weights.to_numpy(),
        "Annualized_Volatility": annual_volatility,
    })

    if combined is not None:
        signal = combined.set_index("Ticker")["Combined_Signal"]
        weights["Combined_Signal"] = [
            float(signal[ticker]) if ticker in signal.index else 0.5
            for ticker in weights["Ticker"]
        ]
        risk_adjusted = (
            weights["Combined_Signal"]
            / np.maximum(weights["Annualized_Volatility"], 1e-12)
        )
        risk_adjusted = risk_adjusted.clip(lower=0)
        weights["Optimized_Weight"] = (
            risk_adjusted / risk_adjusted.sum()
            if risk_adjusted.sum() > 0
            else weights["Inverse_Volatility_Weight"]
        )
    else:
        weights["Combined_Signal"] = np.nan
        weights["Optimized_Weight"] = weights["Inverse_Volatility_Weight"]

    weights["Diversification_Action"] = np.where(
        weights["Optimized_Weight"] > weights["Equal_Weight"],
        "Increase",
        "Decrease_or_review",
    )

    metrics = {
        "trading_days": int(len(returns)),
        "tickers": list(covariance.index),
        "equal_weight_annualized_volatility": equal_risk,
        "inverse_volatility_annualized_volatility": inverse_risk,
        "markowitz_annualized_volatility": markowitz_risk,
        "risk_reduction_percent": float(risk_reduction),
        "markowitz_risk_reduction_percent": float(
            (equal_risk - markowitz_risk) / equal_risk * 100
            if equal_risk > 0 else 0
        ),
        "recommended_action": (
            "Markowitz weights minimize estimated variance for the selected "
            "long-only basket using historical covariance."
        ),
        "annualized_covariance": covariance.round(8).to_dict(),
        "correlation": correlation.round(8).to_dict(),
    }
    covariance_output = covariance.reset_index().rename(columns={"index": "Ticker"})
    correlation_output = correlation.reset_index().rename(columns={"index": "Ticker"})
    return weights, covariance_output, correlation_output, metrics


def project_investment(
    prices: pd.DataFrame,
    portfolio_weights: pd.Series,
    initial_amount: float,
    years: int,
) -> dict[str, float | int]:
    if initial_amount <= 0:
        raise InputError("Investment amount must be greater than zero.")
    if years < 1:
        raise InputError("Projection horizon must be at least one year.")
    weights = portfolio_weights.astype(float)
    if weights.empty or not np.isclose(weights.sum(), 1.0):
        raise InputError("Projection weights must add up to 100%.")

    prepared = prepare_prices(prices)
    close_prices = prepared.pivot_table(
        index="Date",
        columns="Ticker",
        values="Close",
    ).reindex(columns=weights.index).dropna(how="any")
    returns = close_prices.pct_change().dropna(how="any")
    if len(returns) < 2:
        raise InputError("At least three aligned price observations are required.")

    portfolio_returns = returns.mul(weights, axis="columns").sum(axis=1)
    if (portfolio_returns <= -1).any():
        raise InputError("Historical portfolio data contains invalid returns.")
    daily_log_returns = np.log1p(portfolio_returns)
    annual_log_return = float(daily_log_returns.mean() * 252)
    annual_volatility = float(daily_log_returns.std(ddof=1) * np.sqrt(252))
    horizon_volatility = annual_volatility * np.sqrt(years)
    median_log_value = np.log(initial_amount) + annual_log_return * years

    return {
        "years": years,
        "initial_amount": float(initial_amount),
        "annualized_return": float(np.expm1(annual_log_return)),
        "annualized_volatility": annual_volatility,
        "low": float(np.exp(median_log_value - 1.644854 * horizon_volatility)),
        "base": float(np.exp(median_log_value)),
        "high": float(np.exp(median_log_value + 1.644854 * horizon_volatility)),
    }


def train_command(args: argparse.Namespace) -> int:
    if args.stock_file:
        prices = read_table(args.stock_file)
    else:
        prices = download_live_prices(args.period)

    if args.news_file:
        news = prepare_news(read_table(args.news_file))
    else:
        news = prepare_news(download_live_news())

    prepared = prepare_prices(prices)
    (
        stock_model,
        logistic_model,
        news_model,
        metrics,
        test_output,
        backtest,
    ) = train_and_evaluate(prepared, news)
    save_models(
        stock_model,
        logistic_model,
        news_model,
        metrics,
        backtest,
        test_output,
    )
    print("Proposal-aligned models retrained and saved.")
    print(f"Stock accuracy: {metrics['accuracy']:.3f}")
    print(f"Stock F1: {metrics['f1']:.3f}")
    print(
        "Strategy cumulative return: "
        f"{metrics['strategy_final_return']:.2%}"
    )
    print(
        "Buy-and-Hold cumulative return: "
        f"{metrics['buy_hold_final_return']:.2%}"
    )
    print(f"Outputs: {OUT}")
    return 0


def predict_command(args: argparse.Namespace) -> int:
    stock_model, _, news_model = load_models()
    if args.stock_file:
        prices = read_table(args.stock_file)
    else:
        prices = download_live_prices(args.period)

    if args.news_file:
        news = read_table(args.news_file)
    else:
        news = download_live_news()

    stock_output, combined = predict_with_models(
        stock_model,
        news_model,
        prices,
        news,
    )

    weights, covariance, correlation, portfolio_metrics = build_portfolio(
        prices,
        combined,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    stock_output.to_csv(OUT / "proposal_stock_predictions.csv", index=False)
    combined.to_csv(OUT / "proposal_combined_predictions.csv", index=False)
    weights.to_csv(OUT / "proposal_portfolio_weights.csv", index=False)
    covariance.to_csv(OUT / "proposal_portfolio_covariance.csv", index=False)
    correlation.to_csv(OUT / "proposal_portfolio_correlation.csv", index=False)
    write_json(OUT / "proposal_portfolio_metrics.json", portfolio_metrics)

    print(f"Proposal-aligned predictions written to: {OUT}")
    print(combined.to_string(index=False))
    print("\nPortfolio weights:")
    print(weights.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Proposal-aligned stock, news, and portfolio pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--stock-file")
    train_parser.add_argument("--news-file")
    train_parser.add_argument("--period", default="10y")
    train_parser.set_defaults(handler=train_command)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--stock-file")
    predict_parser.add_argument("--news-file")
    predict_parser.add_argument("--period", default="1mo")
    predict_parser.set_defaults(handler=predict_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.handler(args)
    except (InputError, pd.errors.ParserError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
