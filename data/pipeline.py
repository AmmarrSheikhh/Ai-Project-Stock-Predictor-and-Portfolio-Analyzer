from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, time, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize
from scipy.stats import beta, binomtest
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.universe import load_default_tickers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "proposal_pipeline_outputs"
NEWS_LABELS_PATH = OUT / "proposal_news_labels.csv"
TRAINING_HISTORY_PATH = OUT / "proposal_training_history.csv"
TICKERS = load_default_tickers()
EXCHANGE_TIMEZONE = "America/New_York"
MARKET_CLOSE = time(16, 0)
MIN_HISTORY_ROWS = 60
MIN_LIVE_HISTORY_PERIOD = "1y"
NEWS_LOOKBACK_DAYS = 3
TRANSACTION_COST_BPS = 5
COMBINED_NEUTRAL_BAND = 0.05

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
LEGACY_STOCK_FEATURES = STOCK_FEATURES.copy()
SCALE_FREE_FEATURES = [
    "Close_MA10_Ratio",
    "Close_MA20_Ratio",
    "Price_Range",
    "Return_Lag_1",
    "Return_Lag_2",
    "Return_Lag_5",
    "Return_Lag_10",
    "Rolling_Volatility_10",
    "Rolling_Volatility_20",
    "RSI_14",
    "MACD_Close",
    "MACD_Signal_Close",
    "ATR_14_Close",
    "Volume_ZScore_20",
    "Bollinger_Position_20",
]
STOCK_FEATURES = SCALE_FREE_FEATURES + [
    "News_Sentiment",
    "News_Count",
    "Has_News",
]
MODEL_FEATURES_WITHOUT_SENTIMENT = [
    feature for feature in STOCK_FEATURES if feature != "News_Sentiment"
]
TARGET_HORIZONS_DAYS = (1, 5)
TARGET_DEAD_ZONE = 0.001

POSITIVE_WORDS = {
    "beat", "growth", "gain", "gains", "buy", "bull", "bullish", "strong",
    "success", "outlook", "upgrade", "record", "profit", "positive", "rise",
}
NEGATIVE_WORDS = {
    "fall", "falls", "loss", "losses", "risk", "warning", "downgrade",
    "weak", "decline", "lawsuit", "cut", "cuts", "negative", "crisis",
}
FINANCE_SENTIMENT_LEXICON = {
    **{word: 2.0 for word in POSITIVE_WORDS},
    **{word: -2.0 for word in NEGATIVE_WORDS},
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


@lru_cache(maxsize=1)
def _sentiment_analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as error:
        raise InputError(
            "News sentiment requires vaderSentiment. Install project "
            "dependencies with: python -m pip install -r requirements.txt"
        ) from error
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(FINANCE_SENTIMENT_LEXICON)
    return analyzer


def weak_sentiment_label(text: str) -> int:
    score = _sentiment_analyzer().polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return 1
    if score <= -0.05:
        return -1
    return 0


def _exchange_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(EXCHANGE_TIMEZONE).tz_localize(None)
    return timestamp.normalize()


def drop_incomplete_daily_bars(
    source: pd.DataFrame, now: pd.Timestamp | None = None
) -> pd.DataFrame:
    if "Date" not in source.columns:
        raise InputError("Price data must contain a Date column.")
    current_time = pd.Timestamp.now(tz=EXCHANGE_TIMEZONE) if now is None else pd.Timestamp(now)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize(EXCHANGE_TIMEZONE)
    else:
        current_time = current_time.tz_convert(EXCHANGE_TIMEZONE)
    if current_time.time() >= MARKET_CLOSE:
        return source.copy()

    today = current_time.normalize().tz_localize(None)
    price_dates = source["Date"].map(_exchange_date)
    return source.loc[price_dates.ne(today)].copy()


def stock_model_features(
    frame: pd.DataFrame, feature_columns: list[str] | None = None
) -> pd.DataFrame:
    columns = STOCK_FEATURES if feature_columns is None else feature_columns
    missing = set(columns).difference(frame.columns)
    if missing:
        raise InputError(
            "Stock features are missing columns: " + ", ".join(sorted(missing))
        )
    features = frame.loc[:, columns].copy()
    if "News_Sentiment" in features:
        # A zero imputation preserves numeric compatibility; Has_News and
        # News_Count distinguish missing headlines from observed neutral ones.
        features["News_Sentiment"] = features["News_Sentiment"].fillna(0)
    if "News_Count" in features:
        features["News_Count"] = features["News_Count"].fillna(0)
    if "News_Sentiment_Count" in features:
        features["News_Sentiment_Count"] = (
            features["News_Sentiment_Count"].fillna(0)
        )
    if "Has_News" in features:
        features["Has_News"] = features["Has_News"].fillna(False).astype(int)
    return features


def model_feature_columns(model) -> list[str]:
    columns = getattr(model, "feature_names_in_", None)
    if columns is not None:
        return [str(column) for column in columns]
    feature_count = getattr(model, "n_features_in_", None)
    if feature_count == len(LEGACY_STOCK_FEATURES):
        return LEGACY_STOCK_FEATURES
    return STOCK_FEATURES


def probability_for_class(
    model: RandomForestClassifier | Pipeline | CalibratedClassifierCV,
    features: pd.DataFrame,
    target_class: int,
) -> np.ndarray:
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == target_class)
    if len(matches) != 1:
        raise InputError(
            f"Model classes {classes.tolist()} do not contain exactly one "
            f"class {target_class}."
        )
    return model.predict_proba(features)[:, int(matches[0])]


def direction_from_probability(probabilities: np.ndarray) -> np.ndarray:
    return (np.asarray(probabilities) >= 0.5).astype(int)


def filter_ticker_relevant_news(news: pd.DataFrame) -> pd.DataFrame:
    if not {"Ticker", "Headline"}.issubset(news.columns):
        return news.iloc[0:0].copy()

    def is_relevant(row: pd.Series) -> bool:
        headline = str(row["Headline"]).casefold()
        ticker = str(row["Ticker"]).strip()
        if ticker and re.search(rf"\b{re.escape(ticker)}\b", headline, re.IGNORECASE):
            return True
        company = str(row.get("Company", "")).strip()
        company_name = re.sub(r"[^a-z0-9 ]+", " ", company.casefold()).strip()
        primary_name = company_name.split(" ")[0] if company_name else ""
        return bool(
            primary_name
            and len(primary_name) > 2
            and re.search(rf"\b{re.escape(primary_name)}\b", headline)
        )

    relevant = news.apply(is_relevant, axis=1)
    return news.loc[relevant].copy()


def recent_news_for_prices(
    news: pd.DataFrame,
    latest_prices: pd.DataFrame,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
) -> pd.DataFrame:
    if lookback_days < 1:
        raise InputError("The recent news window must be at least one day.")
    if news.empty or "News_Date" not in news.columns:
        return news.iloc[0:0].copy()

    recent = []
    latest_dates = latest_prices.set_index("Ticker")["Date"]
    for ticker, as_of in latest_dates.items():
        if pd.isna(as_of):
            continue
        start = as_of - pd.Timedelta(days=lookback_days - 1)
        ticker_news = news.loc[news["Ticker"].eq(ticker)]
        ticker_news = ticker_news.loc[
            ticker_news["News_Date"].ge(start)
            & ticker_news["News_Date"].le(as_of)
        ]
        if not ticker_news.empty:
            recent.append(ticker_news)
    if not recent:
        return news.iloc[0:0].copy()
    return pd.concat(recent, ignore_index=True)


def prepare_prices(source: pd.DataFrame) -> pd.DataFrame:
    prices = drop_incomplete_daily_bars(source)
    required = {"Date", "Ticker", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(prices.columns)
    if missing:
        raise InputError(
            "Price data is missing required columns: " + ", ".join(sorted(missing))
        )

    prices["Date"] = prices["Date"].map(_exchange_date)
    prices["Ticker"] = prices["Ticker"].astype(str).str.strip().str.upper()
    numeric = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = prices.dropna(subset=["Date", "Ticker"] + numeric)
    if prices.empty:
        raise InputError("No valid daily price rows remain after parsing.")
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

    close_by_ticker = prices.groupby("Ticker")["Close"]
    for lag in (1, 2, 5, 10):
        prices[f"Return_Lag_{lag}"] = close_by_ticker.pct_change(
            periods=lag, fill_method=None
        )
    returns_by_ticker = prices.groupby("Ticker")["Daily_Return"]
    for window in (10, 20):
        prices[f"Rolling_Volatility_{window}"] = (
            returns_by_ticker.transform(
                lambda values: values.rolling(
                    window, min_periods=window
                ).std()
            )
        )

    previous_close = close_by_ticker.shift(1)
    true_range = pd.concat(
        [
            prices["High"] - prices["Low"],
            (prices["High"] - previous_close).abs(),
            (prices["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    average_true_range = true_range.groupby(prices["Ticker"]).transform(
        lambda values: values.rolling(14, min_periods=14).mean()
    )
    prices["ATR_14_Close"] = average_true_range / prices["Close"]
    prices["MACD_Close"] = prices["MACD"] / prices["Close"]
    prices["MACD_Signal_Close"] = prices["MACD_Signal"] / prices["Close"]

    volume_mean = prices.groupby("Ticker")["Volume"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    volume_std = prices.groupby("Ticker")["Volume"].transform(
        lambda values: values.rolling(20, min_periods=20).std(ddof=0)
    )
    prices["Volume_ZScore_20"] = (
        (prices["Volume"] - volume_mean) / volume_std.replace(0, np.nan)
    ).mask(volume_std.eq(0), 0.0)
    bollinger_mean = close_by_ticker.transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    bollinger_std = close_by_ticker.transform(
        lambda values: values.rolling(20, min_periods=20).std(ddof=0)
    )
    prices["Bollinger_Position_20"] = (
        prices["Close"] - (bollinger_mean - 2 * bollinger_std)
    ) / (4 * bollinger_std.replace(0, np.nan))

    prices[SCALE_FREE_FEATURES] = prices[SCALE_FREE_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    prices[BASE_FEATURES] = prices[BASE_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    prices[BASE_FEATURES] = prices.groupby("Ticker")[BASE_FEATURES].transform(
        lambda frame: frame.ffill()
    )
    return prices


def _yahoo_request_error(error: Exception, data_type: str, ticker: str) -> InputError:
    message = str(error)
    if "rate limit" in message.casefold() or "too many requests" in message.casefold():
        return InputError(
            "Yahoo Finance rate-limited the request for "
            f"{ticker}. Wait a few minutes and retry."
        )
    return InputError(
        f"Yahoo Finance {data_type} request failed for {ticker}: {message}"
    )


def normalize_tickers(tickers: list[str] | tuple[str, ...] | None) -> list[str]:
    if tickers is None:
        return TICKERS.copy()
    normalized = []
    for value in tickers:
        normalized.extend(
            ticker.strip().upper()
            for ticker in str(value).split(",")
            if ticker.strip()
        )
    normalized = list(dict.fromkeys(normalized))
    if not normalized:
        raise InputError("Provide at least one ticker symbol.")
    invalid = [
        ticker for ticker in normalized
        if not re.fullmatch(r"[A-Z0-9.^=_-]{1,20}", ticker)
    ]
    if invalid:
        raise InputError(
            "Invalid ticker symbol(s): " + ", ".join(invalid)
        )
    return normalized


def download_live_prices(
    period: str,
    *,
    ensure_indicator_history: bool = False,
    tickers: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as error:
        raise InputError(
            "Live mode requires yfinance. Install it with: "
            "python -m pip install yfinance"
        ) from error

    fetch_period = period
    if ensure_indicator_history and period in {"1mo", "3mo", "6mo"}:
        fetch_period = MIN_LIVE_HISTORY_PERIOD

    selected_tickers = normalize_tickers(tickers)
    frames = []
    for ticker in selected_tickers:
        try:
            history = yf.Ticker(ticker).history(
                period=fetch_period, auto_adjust=False, actions=False
            )
        except Exception as error:
            raise _yahoo_request_error(error, "price", ticker) from error
        if history.empty:
            raise InputError(
                f"No live price data was returned for {ticker}. "
                "Yahoo may be temporarily unavailable or rate-limiting requests."
            )
        history = history.reset_index()
        history["Date"] = history["Date"].map(_exchange_date)
        history["Ticker"] = ticker
        frames.append(history[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]])
    return pd.concat(frames, ignore_index=True)


def download_live_news(
    tickers: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as error:
        raise InputError(
            "Live news requires yfinance. Install it with: "
            "python -m pip install yfinance"
        ) from error

    rows = []
    for ticker in normalize_tickers(tickers):
        try:
            headlines = yf.Search(ticker, max_results=25).news
        except Exception as error:
            raise _yahoo_request_error(error, "news", ticker) from error
        for item in headlines:
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
        return pd.DataFrame(columns=["Ticker", "Headline", "PublishedAt"])
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
        published_values = news["PublishedAt"].copy()
        news["News_Date"] = published_values.map(_exchange_date)
        news["PublishedAt"] = pd.to_datetime(
            news["PublishedAt"], errors="coerce", utc=True
        )

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

    if dated.empty:
        daily = pd.DataFrame(
            columns=[
                "Ticker",
                "Date",
                "News_Sentiment",
                "News_Sentiment_Count",
            ]
        )
        daily["Date"] = pd.to_datetime(daily["Date"])
    else:
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
    merged["News_Sentiment_Count"] = (
        merged["News_Sentiment_Count"].fillna(0).astype(int)
    )
    merged["News_Count"] = merged["News_Sentiment_Count"]
    merged["Has_News"] = merged["News_Count"].gt(0)
    return merged


def train_news_model(
    news: pd.DataFrame, manual_labels: pd.DataFrame | None = None
) -> tuple[Pipeline, dict]:
    weak_news = news.loc[:, ["Ticker", "Headline", "Weak_Label"]].copy()
    manual = (
        pd.DataFrame(columns=["Ticker", "Headline", "Label"])
        if manual_labels is None
        else manual_labels.loc[:, ["Ticker", "Headline", "Label"]].copy()
    )
    manual["Label"] = pd.to_numeric(manual["Label"], errors="coerce")
    manual = manual.dropna(subset=["Label"])
    validation = manual.iloc[0:0].copy()
    manual_fit = manual
    counts = manual["Label"].value_counts()
    validation_size = max(counts.size, int(np.ceil(len(manual) * 0.2)))
    if (
        len(manual) >= 10
        and counts.size >= 2
        and counts.min() >= 2
        and validation_size < len(manual)
    ):
        manual_fit, validation = train_test_split(
            manual,
            test_size=validation_size,
            random_state=42,
            stratify=manual["Label"],
        )
        heldout_headlines = set(
            validation["Headline"]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.casefold()
        )
        manual_fit = manual_fit.loc[
            ~manual_fit["Headline"]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.casefold()
            .isin(heldout_headlines)
        ]
        weak_news = weak_news.loc[
            ~weak_news["Headline"]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.casefold()
            .isin(heldout_headlines)
        ]

    manual_fit = manual_fit.rename(columns={"Label": "Weak_Label"})
    training = pd.concat([weak_news, manual_fit], ignore_index=True)
    training["_label_key"] = _news_label_key(training)
    training = training.drop_duplicates("_label_key", keep="last").drop(
        columns="_label_key"
    )
    if training["Weak_Label"].nunique() < 2:
        raise InputError(
            "News model training needs at least two sentiment labels. "
            "Supply labeled headlines or add human-reviewed labels."
        )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    model.fit(training["Headline"], training["Weak_Label"].astype(int))
    metrics = {
        "training_rows": int(len(training)),
        "weak_label_rows": int(len(weak_news)),
        "manual_label_rows": int(len(manual)),
        "label_distribution": {
            str(label): int(count)
            for label, count in training["Weak_Label"].value_counts().sort_index().items()
        },
        "evaluation_status": (
            "Held-out human-labeled evaluation"
            if not validation.empty
            else "Unavailable: no independent human-labeled holdout"
        ),
    }
    if not validation.empty:
        validation_predictions = model.predict(validation["Headline"])
        metrics["manual_holdout_rows"] = int(len(validation))
        metrics["manual_holdout_accuracy"] = float(
            accuracy_score(validation["Label"], validation_predictions)
        )
        metrics["manual_holdout_balanced_accuracy"] = float(
            balanced_accuracy_score(validation["Label"], validation_predictions)
        )
        metrics["manual_holdout_confusion_matrix"] = confusion_matrix(
            validation["Label"],
            validation_predictions,
            labels=[-1, 0, 1],
        ).tolist()
    return model, metrics


def add_targets(
    prices: pd.DataFrame,
    horizon_days: int = 1,
    dead_zone: float = TARGET_DEAD_ZONE,
) -> pd.DataFrame:
    if horizon_days < 1:
        raise InputError("The target horizon must be at least one trading day.")
    if dead_zone < 0:
        raise InputError("The target dead zone cannot be negative.")
    labeled = prices.copy()
    labeled["Next_Close"] = labeled.groupby("Ticker")["Close"].shift(
        -horizon_days
    )
    labeled["Target_Date"] = labeled.groupby("Ticker")["Date"].shift(
        -horizon_days
    )
    labeled["Forward_Return"] = labeled["Next_Close"] / labeled["Close"] - 1
    outside_dead_zone = labeled["Forward_Return"].abs().ge(dead_zone)
    labeled["Target_Up"] = np.where(
        outside_dead_zone,
        labeled["Forward_Return"].gt(0).astype(float),
        np.nan,
    )
    labeled.loc[labeled["Next_Close"].isna(), "Target_Up"] = np.nan
    return labeled.dropna(
        subset=MODEL_FEATURES_WITHOUT_SENTIMENT + ["Target_Up"]
    )


def time_series_splits_by_date(
    frame: pd.DataFrame,
    n_splits: int = 3,
    gap_days: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dates = np.sort(frame["Date"].dropna().unique())
    if len(dates) < n_splits + gap_days + 1:
        raise InputError(
            "Not enough distinct dates for leakage-safe walk-forward splits."
        )
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)
    date_values = frame["Date"].reset_index(drop=True)
    splits = []
    for train_dates, test_dates in splitter.split(dates):
        train_rows = np.flatnonzero(
            date_values.isin(dates[train_dates]).to_numpy()
        )
        test_rows = np.flatnonzero(
            date_values.isin(dates[test_dates]).to_numpy()
        )
        splits.append((train_rows, test_rows))
    return splits


def _stock_estimators() -> tuple[RandomForestClassifier, Pipeline]:
    forest = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    return forest, logistic


def _fit_time_calibrated_model(
    estimator,
    frame: pd.DataFrame,
    target: pd.Series,
    horizon_days: int,
) -> CalibratedClassifierCV:
    splits = time_series_splits_by_date(
        frame,
        n_splits=3,
        gap_days=horizon_days,
    )
    valid_splits = [
        (train_rows, validation_rows)
        for train_rows, validation_rows in splits
        if target.iloc[train_rows].nunique() == 2
        and target.iloc[validation_rows].nunique() == 2
    ]
    if len(valid_splits) < 2:
        raise InputError(
            "Not enough two-class chronological folds to calibrate "
            "the stock-model probabilities."
        )
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=valid_splits,
        ensemble=True,
    )
    calibrated.fit(
        stock_model_features(frame),
        target.astype(int),
    )
    return calibrated


def feature_importance_for_model(model) -> dict[str, float]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        candidates = [
            getattr(calibrated, "estimator", None)
            for calibrated in getattr(model, "calibrated_classifiers_", [])
        ]
        values = [
            getattr(estimator, "feature_importances_", None)
            for estimator in candidates
            if estimator is not None
        ]
        values = [value for value in values if value is not None]
        if values:
            importances = np.mean(values, axis=0)
    if importances is None:
        return {}
    feature_names = model_feature_columns(model)
    if len(feature_names) != len(importances):
        raise InputError(
            "Model feature names do not match its feature importances."
        )
    return {
        feature: float(importance)
        for feature, importance in zip(feature_names, importances)
    }


def walk_forward_evaluate(
    prices: pd.DataFrame, horizon_days: int
) -> dict:
    labeled = add_targets(
        prices,
        horizon_days=horizon_days,
        dead_zone=TARGET_DEAD_ZONE,
    )
    splits = time_series_splits_by_date(
        labeled,
        n_splits=3,
        gap_days=horizon_days,
    )
    fold_results = []
    for fold_number, (train_rows, validation_rows) in enumerate(splits, start=1):
        train = labeled.iloc[train_rows]
        validation = labeled.iloc[validation_rows]
        if train["Target_Up"].nunique() < 2:
            continue
        estimator, _ = _stock_estimators()
        train_features = stock_model_features(train)
        validation_features = stock_model_features(validation)
        estimator.fit(train_features, train["Target_Up"].astype(int))
        probabilities = probability_for_class(
            estimator, validation_features, 1
        )
        predicted = estimator.predict(validation_features)
        y_true = validation["Target_Up"].astype(int)
        majority = int(train["Target_Up"].mode().iloc[0])
        majority_predicted = np.full(len(validation), majority)
        previous_day = validation["Daily_Return"].gt(0).astype(int)
        fold_results.append({
            "fold": fold_number,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "accuracy": float(accuracy_score(y_true, predicted)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, predicted)
            ),
            "roc_auc": (
                float(roc_auc_score(y_true, probabilities))
                if y_true.nunique() == 2 else None
            ),
            "brier_score": float(
                brier_score_loss(y_true, probabilities)
            ),
            "majority_baseline_accuracy": float(
                accuracy_score(y_true, majority_predicted)
            ),
            "previous_day_direction_accuracy": float(
                accuracy_score(y_true, previous_day)
            ),
        })
    if not fold_results:
        raise InputError(
            f"No valid walk-forward folds for the {horizon_days}-day target."
        )
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "brier_score",
        "majority_baseline_accuracy",
        "previous_day_direction_accuracy",
    ]
    summary = {}
    for metric in metric_names:
        values = [
            fold[metric] for fold in fold_results if fold[metric] is not None
        ]
        summary[f"mean_{metric}"] = (
            float(np.mean(values)) if values else None
        )
        summary[f"std_{metric}"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
    return {
        "horizon_days": horizon_days,
        "dead_zone": TARGET_DEAD_ZONE,
        "folds": fold_results,
        **summary,
    }


def exact_accuracy_interval(
    correct: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    if total < 1:
        raise InputError("An accuracy interval requires at least one test row.")
    alpha = 1 - confidence
    lower = (
        float(beta.ppf(alpha / 2, correct, total - correct + 1))
        if correct else 0.0
    )
    upper = (
        float(beta.ppf(1 - alpha / 2, correct + 1, total - correct))
        if correct < total else 1.0
    )
    return lower, upper


def backtest_performance(
    daily_returns: pd.Series,
    daily_turnover: pd.Series | None = None,
    periods_per_year: float = 252.0,
) -> dict[str, float | None]:
    if periods_per_year <= 0:
        raise InputError("The annualization period count must be positive.")
    returns = pd.to_numeric(daily_returns, errors="coerce").dropna()
    turnover = (
        pd.Series(0.0, index=returns.index)
        if daily_turnover is None
        else pd.to_numeric(daily_turnover, errors="coerce")
        .reindex(returns.index)
        .fillna(0)
    )
    standard_deviation = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    annualization_factor = np.sqrt(periods_per_year)
    annualized_volatility = standard_deviation * annualization_factor
    sharpe = (
        float(returns.mean() / standard_deviation * annualization_factor)
        if standard_deviation > 0 else None
    )
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "turnover": float(turnover.sum()),
        "final_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0,
    }


def train_and_evaluate(
    prices: pd.DataFrame, news: pd.DataFrame
) -> tuple[
    CalibratedClassifierCV,
    CalibratedClassifierCV,
    Pipeline,
    dict,
    pd.DataFrame,
    pd.DataFrame,
]:
    manual_training = load_manual_news_labels()
    news_model, news_metrics = train_news_model(news, manual_training)
    feature_news = filter_ticker_relevant_news(
        apply_manual_news_labels(news, manual_training)
    )
    merged = merge_daily_sentiment(prices, feature_news)
    dates = np.sort(merged["Date"].dropna().unique())
    if len(dates) < MIN_HISTORY_ROWS:
        raise InputError(
            f"At least {MIN_HISTORY_ROWS} distinct daily dates are required."
        )
    cutoff = dates[int(len(dates) * 0.8)]
    development = merged.loc[merged["Date"] < cutoff]
    horizon_evaluations = {
        str(horizon): walk_forward_evaluate(development, horizon)
        for horizon in TARGET_HORIZONS_DAYS
    }
    selected_horizon = max(
        TARGET_HORIZONS_DAYS,
        key=lambda horizon: (
            horizon_evaluations[str(horizon)]["mean_balanced_accuracy"],
            -horizon,
        ),
    )
    labeled = add_targets(
        merged,
        horizon_days=selected_horizon,
        dead_zone=TARGET_DEAD_ZONE,
    )
    train = labeled.loc[
        (labeled["Date"] < cutoff)
        & (labeled["Target_Date"] < cutoff)
    ]
    test = labeled.loc[labeled["Date"] >= cutoff]
    if len(train) < 100 or test.empty:
        raise InputError(
            "The chronological split needs at least 100 training rows and "
            "a non-empty held-out test period."
        )
    if train.empty or test.empty:
        raise InputError("The data cannot produce a chronological 80/20 split.")
    if train["Target_Up"].nunique() < 2:
        raise InputError("Stock training data must contain both up and down targets.")

    forest_estimator, logistic_estimator = _stock_estimators()
    test_features = stock_model_features(test)
    y_train = train["Target_Up"].astype(int)
    y_test = test["Target_Up"].astype(int)
    model = _fit_time_calibrated_model(
        forest_estimator, train, y_train, selected_horizon
    )
    logistic_model = _fit_time_calibrated_model(
        logistic_estimator, train, y_train, selected_horizon
    )

    rf_probabilities = probability_for_class(model, test_features, 1)
    rf_predictions = direction_from_probability(rf_probabilities)
    lr_probabilities = probability_for_class(
        logistic_model, test_features, 1
    )
    lr_predictions = direction_from_probability(lr_probabilities)
    test_output = test.copy()
    test_output["Predicted_Up"] = rf_predictions
    test_output["Prediction_Probability"] = rf_probabilities
    test_output["Random_Forest_Predicted_Up"] = rf_predictions
    test_output["Random_Forest_Probability"] = rf_probabilities
    test_output["Logistic_Regression_Predicted_Up"] = lr_predictions
    test_output["Logistic_Regression_Probability"] = lr_probabilities

    ticker_stability = {}
    for ticker, group in test_output.groupby("Ticker"):
        ticker_actual = group["Target_Up"].astype(int)
        ticker_stability[ticker] = {
            "rows": int(len(group)),
            "accuracy": float(
                accuracy_score(ticker_actual, group["Predicted_Up"])
            ),
            "f1": float(
                f1_score(
                    ticker_actual,
                    group["Predicted_Up"],
                    zero_division=0,
                )
            ),
        }

    test_output = test_output.sort_values(["Date", "Ticker"])
    test_output["Backtest_Trade"] = False
    test_dates = np.sort(test_output["Date"].unique())
    entry_dates = set(test_dates[::selected_horizon])
    trades = test_output.loc[test_output["Date"].isin(entry_dates)].copy()
    trades["Forward_Return"] = (
        trades["Next_Close"] / trades["Close"]
    ) - 1
    trades["BuyHold_Cumulative_Value"] = (
        (1 + trades["Forward_Return"])
        .groupby(trades["Ticker"]).cumprod()
    )
    transaction_cost = TRANSACTION_COST_BPS / 10_000
    strategy_turnovers = {}
    for prefix, probability_column in [
        ("Random_Forest", "Random_Forest_Probability"),
        ("Logistic_Regression", "Logistic_Regression_Probability"),
    ]:
        position = trades[probability_column].ge(0.5).astype(float)
        turnover = position.groupby(trades["Ticker"]).diff().abs().fillna(0)
        first_observation = trades.groupby("Ticker").cumcount().eq(0)
        turnover.loc[first_observation] = position.loc[first_observation]
        returns = (
            position * trades["Forward_Return"]
            - turnover * transaction_cost
        )
        trades[f"{prefix}_Strategy_Return"] = returns
        trades[f"{prefix}_Cumulative_Value"] = (
            (1 + returns).groupby(trades["Ticker"]).cumprod()
        )
        strategy_turnovers[prefix] = turnover.groupby(trades["Date"]).mean()

    test_output.loc[trades.index, "Backtest_Trade"] = True
    for column in [
        "Forward_Return",
        "BuyHold_Cumulative_Value",
        "Random_Forest_Strategy_Return",
        "Random_Forest_Cumulative_Value",
        "Logistic_Regression_Strategy_Return",
        "Logistic_Regression_Cumulative_Value",
    ]:
        test_output[column] = np.nan
        test_output.loc[trades.index, column] = trades[column]
    test_output["Strategy_Return"] = test_output[
        "Random_Forest_Strategy_Return"
    ]
    test_output["Strategy_Cumulative_Return"] = test_output[
        "Random_Forest_Cumulative_Value"
    ]
    test_output["BuyHold_Cumulative_Return"] = test_output[
        "BuyHold_Cumulative_Value"
    ]
    strategy_daily = (
        trades.groupby("Date")["Random_Forest_Strategy_Return"].mean()
    )
    logistic_daily = (
        trades.groupby("Date")["Logistic_Regression_Strategy_Return"].mean()
    )
    equal_weight_period_returns = trades.groupby("Date")["Forward_Return"].mean()

    strategy_curve = (1 + strategy_daily.fillna(0)).cumprod()
    logistic_curve = (1 + logistic_daily.fillna(0)).cumprod()
    equal_weight_curve = (1 + equal_weight_period_returns.fillna(0)).cumprod()
    buy_hold_curve = trades.groupby("Date")[
        "BuyHold_Cumulative_Value"
    ].mean()
    curve_frame = pd.DataFrame({
        "Date": strategy_curve.index,
        "Strategy_Equity": strategy_curve.to_numpy(),
        "Buy_Hold_Equity": buy_hold_curve.reindex(strategy_curve.index).to_numpy(),
        "Equal_Weight_Equity": equal_weight_curve.reindex(
            strategy_curve.index
        ).to_numpy(),
    })

    backtest = pd.DataFrame({
        "Date": trades["Date"].to_numpy(),
        "Ticker": trades["Ticker"].to_numpy(),
        "BuyHold_Cumulative_Value": trades[
            "BuyHold_Cumulative_Value"
        ].to_numpy(),
        "Random_Forest_Cumulative_Value": trades[
            "Random_Forest_Cumulative_Value"
        ].to_numpy(),
        "Logistic_Regression_Cumulative_Value": trades[
            "Logistic_Regression_Cumulative_Value"
        ].to_numpy(),
        "Strategy_Cumulative_Return": trades[
            "Random_Forest_Cumulative_Value"
        ].to_numpy(),
        "BuyHold_Cumulative_Return": trades[
            "BuyHold_Cumulative_Value"
        ].to_numpy(),
    }).merge(curve_frame, on="Date", how="left")

    buy_hold_returns = buy_hold_curve.pct_change().fillna(0)
    periods_per_year = 252 / selected_horizon
    performance = {
        "strategy": backtest_performance(
            strategy_daily,
            strategy_turnovers["Random_Forest"],
            periods_per_year,
        ),
        "buy_and_hold": backtest_performance(
            buy_hold_returns, periods_per_year=periods_per_year
        ),
        "equal_weight": backtest_performance(
            equal_weight_period_returns,
            periods_per_year=periods_per_year,
        ),
        "logistic_regression": backtest_performance(
            logistic_daily,
            strategy_turnovers["Logistic_Regression"],
            periods_per_year,
        ),
    }

    final_labeled = labeled
    final_forest_estimator, final_logistic_estimator = _stock_estimators()
    final_model = _fit_time_calibrated_model(
        final_forest_estimator,
        final_labeled,
        final_labeled["Target_Up"],
        selected_horizon,
    )
    final_logistic_model = _fit_time_calibrated_model(
        final_logistic_estimator,
        final_labeled,
        final_labeled["Target_Up"],
        selected_horizon,
    )

    majority_label = int(train["Target_Up"].mode().iloc[0])
    majority_predictions = np.full(len(test), majority_label)
    always_up_predictions = np.ones(len(test), dtype=int)
    previous_day_predictions = test["Daily_Return"].gt(0).astype(int)
    model_comparison = {
        "Majority Class Baseline": {
            "accuracy": float(accuracy_score(
                y_test, majority_predictions
            )),
            "f1": float(f1_score(
                y_test,
                majority_predictions,
                zero_division=0,
            )),
        },
        "Previous-Day Direction": {
            "accuracy": float(accuracy_score(y_test, previous_day_predictions)),
            "f1": float(f1_score(
                y_test, previous_day_predictions, zero_division=0
            )),
        },
        "Buy-and-Hold (Always UP)": {
            "accuracy": float(accuracy_score(y_test, always_up_predictions)),
            "f1": float(f1_score(
                y_test, always_up_predictions, zero_division=0
            )),
        },
        "Logistic Regression": {
            "accuracy": float(accuracy_score(
                y_test,
                test_output["Logistic_Regression_Predicted_Up"],
            )),
            "f1": float(f1_score(
                y_test,
                test_output["Logistic_Regression_Predicted_Up"],
                zero_division=0,
            )),
        },
        "Random Forest": {
            "accuracy": float(accuracy_score(
                y_test,
                test_output["Random_Forest_Predicted_Up"],
            )),
            "f1": float(f1_score(
                y_test,
                test_output["Random_Forest_Predicted_Up"],
                zero_division=0,
            )),
        },
    }
    correct_count = int(
        np.sum(test_output["Random_Forest_Predicted_Up"].to_numpy() == y_test)
    )
    accuracy_interval = exact_accuracy_interval(correct_count, len(y_test))
    calibration_fraction, calibration_mean = calibration_curve(
        y_test,
        rf_probabilities,
        n_bins=10,
        strategy="quantile",
    )
    try:
        calibration_auc = float(roc_auc_score(y_test, rf_probabilities))
    except ValueError:
        calibration_auc = None
    majority_baseline_accuracy = model_comparison[
        "Majority Class Baseline"
    ]["accuracy"]
    metrics = {
        "stock_features": STOCK_FEATURES,
        "split": "chronological 80/20",
        "target_horizon_days": int(selected_horizon),
        "target_dead_zone": TARGET_DEAD_ZONE,
        "target_horizon_selection": {
            "selection_data": "first 80% of dates only",
            "criterion": "highest mean walk-forward balanced accuracy",
            "evaluations": horizon_evaluations,
        },
        "direction_threshold": 0.5,
        "direction_tie_policy": "UP",
        "probability_calibration": {
            "method": "sigmoid",
            "cross_validation": "expanding chronological folds with horizon gap",
        },
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "final_training_rows": int(len(labeled)),
        "accuracy": float(accuracy_score(y_test, rf_predictions)),
        "f1": float(
            f1_score(
                y_test,
                rf_predictions,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, rf_predictions)),
        "roc_auc": calibration_auc,
        "brier_score": float(brier_score_loss(y_test, rf_probabilities)),
        "accuracy_confidence_interval_95": list(accuracy_interval),
        "accuracy_binomial_p_value_vs_majority": float(
            binomtest(
                correct_count,
                len(y_test),
                p=majority_baseline_accuracy,
                alternative="greater",
            ).pvalue
        ),
        "calibration_curve": {
            "mean_predicted_probability": calibration_mean.tolist(),
            "fraction_positive": calibration_fraction.tolist(),
        },
        "probability_distribution": {
            key: float(value)
            for key, value in pd.Series(rf_probabilities)
            .describe()
            .items()
        },
        "feature_importance": feature_importance_for_model(final_model),
        "confusion_matrix": confusion_matrix(
            y_test,
            rf_predictions,
            labels=[0, 1],
        ).tolist(),
        "train_class_balance": {
            str(label): int(count)
            for label, count in y_train.value_counts().sort_index().items()
        },
        "test_class_balance": {
            str(label): int(count)
            for label, count in y_test.value_counts().sort_index().items()
        },
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "model_comparison": model_comparison,
        "ticker_stability": ticker_stability,
        "training_date_range": [
            str(labeled["Date"].min().date()),
            str(labeled["Date"].max().date()),
        ],
        "backtest_entry_frequency_days": int(selected_horizon),
        "strategy_final_return": float(
            strategy_curve.iloc[-1] - 1
        ) if len(strategy_curve) else 0.0,
        "logistic_strategy_final_return": float(
            logistic_curve.iloc[-1] - 1
        ) if len(logistic_curve) else 0.0,
        "buy_hold_final_return": float(
            buy_hold_curve.iloc[-1] - 1
        ) if len(buy_hold_curve) else 0.0,
        "equal_weight_final_return": float(
            equal_weight_curve.iloc[-1] - 1
        ) if len(equal_weight_curve) else 0.0,
        "backtest_transaction_cost_bps": TRANSACTION_COST_BPS,
        "backtest_performance": performance,
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


def _source_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def save_models(
    stock_model: RandomForestClassifier | CalibratedClassifierCV,
    logistic_model: Pipeline | CalibratedClassifierCV,
    news_model: Pipeline,
    metrics: dict,
    backtest: pd.DataFrame,
    test_output: pd.DataFrame,
) -> Path | None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=".proposal_pipeline_outputs_staging_",
        dir=OUT.parent,
    ))
    backup_root = OUT.parent / "proposal_pipeline_outputs_backup"
    backup_path = None
    try:
        if OUT.exists():
            shutil.copytree(OUT, staging, dirs_exist_ok=True)

        joblib.dump(stock_model, staging / "proposal_stock_model.joblib")
        joblib.dump(logistic_model, staging / "proposal_logistic_model.joblib")
        joblib.dump(news_model, staging / "proposal_news_model.joblib")
        write_json(staging / "proposal_metrics.json", metrics)
        backtest.to_csv(staging / "proposal_backtest.csv", index=False)
        test_output.to_csv(
            staging / "proposal_stock_test_predictions.csv",
            index=False,
        )
        metadata = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "source_revision": _source_revision(),
            "stock_features": metrics.get("stock_features", []),
            "target_horizon_days": metrics.get("target_horizon_days"),
            "training_date_range": metrics.get("training_date_range"),
            "held_out_metrics": {
                key: metrics.get(key)
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "roc_auc",
                    "brier_score",
                    "majority_baseline_accuracy",
                )
            },
        }
        write_json(staging / "proposal_model_metadata.json", metadata)

        if OUT.exists():
            backup_root.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ"
            )
            backup_path = backup_root / f"proposal_pipeline_outputs_{timestamp}"
            OUT.rename(backup_path)
        try:
            staging.rename(OUT)
        except OSError:
            if backup_path is not None and backup_path.exists() and not OUT.exists():
                backup_path.rename(OUT)
            raise
        return backup_path
    except Exception:
        if backup_path is not None and backup_path.exists() and not OUT.exists():
            backup_path.rename(OUT)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_models() -> tuple[
    RandomForestClassifier | CalibratedClassifierCV,
    Pipeline | CalibratedClassifierCV | None,
    Pipeline,
]:
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
    if "Headline_Count" in weighted.columns:
        has_news = pd.to_numeric(
            weighted["Headline_Count"], errors="coerce"
        ).fillna(0).gt(0)
    else:
        has_news = weighted["Sentiment_Signal"].notna()
    combined_with_news = (
        technical_weight * weighted["Prediction_Probability"]
        + (1 - technical_weight) * weighted["Sentiment_Signal"]
    )
    weighted["Combined_Signal"] = np.where(
        has_news,
        combined_with_news,
        weighted["Prediction_Probability"],
    )
    weighted["News_Adjustment"] = (
        weighted["Combined_Signal"] - weighted["Prediction_Probability"]
    )
    weighted["Combined_Direction"] = np.select(
        [
            weighted["Combined_Signal"] >= 0.5 + COMBINED_NEUTRAL_BAND,
            weighted["Combined_Signal"] <= 0.5 - COMBINED_NEUTRAL_BAND,
        ],
        ["UP", "DOWN"],
        default="NEUTRAL",
    )
    return weighted


def predict_with_models(
    stock_model: RandomForestClassifier | Pipeline | CalibratedClassifierCV,
    news_model: Pipeline,
    prices: pd.DataFrame,
    news: pd.DataFrame,
    technical_weight: float = 0.7,
    news_lookback_days: int = NEWS_LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_prices = prepare_prices(prices)
    prepared_news = filter_ticker_relevant_news(
        apply_manual_news_labels(prepare_news(news))
    )

    news_predictions = prepared_news.copy()
    if not news_predictions.empty:
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
    else:
        news_predictions["Predicted_Sentiment"] = pd.Series(dtype=float)
        news_predictions["Sentiment_Confidence"] = pd.Series(dtype=float)

    merged = merge_daily_sentiment(prepared_prices, prepared_news)
    usable = merged.dropna(subset=MODEL_FEATURES_WITHOUT_SENTIMENT)
    usable_counts = usable.groupby("Ticker").size()
    source_tickers = (
        prices["Ticker"].dropna().astype(str).str.strip().str.upper().unique()
    )
    skipped_rows = []
    for ticker in source_tickers:
        count = int(usable_counts.get(ticker, 0))
        if count < MIN_HISTORY_ROWS:
            skipped_rows.append({
                "Ticker": ticker,
                "Status": "skipped",
                "Message": (
                    f"Insufficient usable history: {count} rows; "
                    f"{MIN_HISTORY_ROWS} required."
                ),
            })

    eligible_tickers = [
        ticker for ticker in source_tickers
        if int(usable_counts.get(ticker, 0)) >= MIN_HISTORY_ROWS
    ]
    latest = (
        usable.loc[usable["Ticker"].isin(eligible_tickers)]
        .sort_values(["Ticker", "Date"])
        .groupby("Ticker", as_index=False)
        .tail(1)
    )
    if latest.empty:
        stock_output = pd.DataFrame(skipped_rows)
        combined = pd.DataFrame(columns=[
            "Ticker",
            "Prediction_Probability",
            "Headline_Count",
            "Sentiment_Signal",
        ])
    else:
        model_features = stock_model_features(
            latest, model_feature_columns(stock_model)
        )
        stock_probabilities = probability_for_class(
            stock_model, model_features, 1
        )
        stock_predictions = direction_from_probability(stock_probabilities)
        successful = pd.DataFrame({
            "Ticker": latest["Ticker"].to_numpy(),
            "Price_Date": latest["Date"].dt.date.to_numpy(),
            "Data_As_Of": latest["Date"].dt.date.to_numpy(),
            "Close": latest["Close"].to_numpy(),
            "Predicted_Up": stock_predictions.astype(int),
            "Prediction_Probability": stock_probabilities,
            "RSI_14": latest["RSI_14"].to_numpy(),
            "MACD": latest["MACD"].to_numpy(),
            "News_Sentiment": latest["News_Sentiment"].to_numpy(),
            "News_Sentiment_Count": latest["News_Sentiment_Count"].to_numpy(),
            "News_Count": latest["News_Count"].to_numpy(),
            "Has_News": latest["Has_News"].to_numpy(),
            "Status": "ok",
            "Message": "",
        })
        stock_output = pd.concat(
            [successful, pd.DataFrame(skipped_rows)],
            ignore_index=True,
        )

        recent_news = recent_news_for_prices(
            news_predictions, latest, news_lookback_days
        )
        if recent_news.empty:
            news_summary = pd.DataFrame(columns=[
                "Ticker",
                "Predicted_Sentiment_Mean",
                "Headline_Count",
                "Mean_Sentiment_Confidence",
            ])
        else:
            news_summary = (
                recent_news.groupby("Ticker", as_index=False)
                .agg(
                    Predicted_Sentiment_Mean=("Predicted_Sentiment", "mean"),
                    Headline_Count=("Predicted_Sentiment", "size"),
                    Mean_Sentiment_Confidence=("Sentiment_Confidence", "mean"),
                )
            )
        recent_counts = (
            news_summary.set_index("Ticker")["Headline_Count"]
            if not news_summary.empty
            else pd.Series(dtype=int)
        )
        stock_output["News_Count"] = (
            stock_output["Ticker"].map(recent_counts).fillna(0).astype(int)
        )
        stock_output["Has_News"] = stock_output["News_Count"].gt(0)
        combined = successful.merge(news_summary, on="Ticker", how="left")
        combined["Headline_Count"] = (
            combined["Headline_Count"].fillna(0).astype(int)
        )
        combined["Has_News"] = combined["Headline_Count"].gt(0)
        combined["News_Count"] = combined["Headline_Count"]
        combined["Sentiment_Signal"] = (
            combined["Predicted_Sentiment_Mean"] + 1
        ) / 2
    return stock_output, apply_signal_weights(combined, technical_weight)


def build_portfolio(
    prices: pd.DataFrame,
    combined: pd.DataFrame | None = None,
    selected_tickers: list[str] | None = None,
    max_weight: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    prepared = prepare_prices(prices)
    returns = prepared.pivot_table(
        index="Date",
        columns="Ticker",
        values="Daily_Return",
    )
    requested_tickers = (
        sorted(returns.columns)
        if selected_tickers is None
        else normalize_tickers(selected_tickers)
    )
    returns = returns.reindex(
        columns=[ticker for ticker in requested_tickers if ticker in returns.columns]
    ).dropna(how="any")
    if len(returns.columns) < 2:
        raise InputError(
            "Portfolio analysis requires at least two tickers with aligned history."
        )

    covariance = pd.DataFrame(
        LedoitWolf().fit(returns).covariance_ * 252,
        index=returns.columns,
        columns=returns.columns,
    )
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
    if not 1 / len(equal_weights) <= max_weight <= 1:
        raise InputError(
            "The maximum portfolio weight must be between 1 / ticker count and 1."
        )
    optimization = minimize(
        lambda candidate: float(
            candidate @ covariance.to_numpy() @ candidate
        ),
        x0=equal_weights.to_numpy(),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * len(equal_weights),
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
    if not np.isclose(markowitz_weights.sum(), 1.0):
        raise InputError("Minimum-variance portfolio weights do not sum to 1.")

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
    signal = pd.Series(0.5, index=covariance.index)
    if combined is not None and not combined.empty:
        combined_signals = combined.set_index("Ticker")["Combined_Signal"]
        signal = pd.Series(
            [
                float(combined_signals[ticker])
                if ticker in combined_signals.index
                else 0.5
                for ticker in covariance.index
            ],
            index=covariance.index,
        )
    signal_tilted_raw = signal / np.maximum(annual_volatility, 1e-12)
    signal_tilted_raw = signal_tilted_raw.clip(lower=0)
    signal_tilted_weights = (
        signal_tilted_raw / signal_tilted_raw.sum()
        if signal_tilted_raw.sum() > 0
        else inverse_weights
    )
    for allocation in (
        equal_weights,
        inverse_weights,
        markowitz_weights,
        signal_tilted_weights,
    ):
        if not np.isclose(allocation.sum(), 1.0):
            raise InputError("Portfolio allocation weights must sum to 1.")
    signal_tilted_risk = float(np.sqrt(
        signal_tilted_weights.to_numpy()
        @ covariance.to_numpy()
        @ signal_tilted_weights.to_numpy()
    ))

    def risk_contribution(portfolio_weights: pd.Series) -> np.ndarray:
        variance = float(
            portfolio_weights.to_numpy()
            @ covariance.to_numpy()
            @ portfolio_weights.to_numpy()
        )
        if variance <= 0:
            return np.zeros(len(portfolio_weights))
        return (
            portfolio_weights.to_numpy()
            * (covariance.to_numpy() @ portfolio_weights.to_numpy())
            / variance
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
        "Signal_Tilted_Weight": signal_tilted_weights.to_numpy(),
        "Annualized_Volatility": annual_volatility,
        "Equal_Weight_Risk_Contribution": risk_contribution(equal_weights),
        "Inverse_Volatility_Risk_Contribution": risk_contribution(inverse_weights),
        "Markowitz_Risk_Contribution": risk_contribution(markowitz_weights),
        "Signal_Tilted_Risk_Contribution": risk_contribution(
            signal_tilted_weights
        ),
    })
    weights["Combined_Signal"] = signal.to_numpy()
    weights["Diversification_Action"] = np.where(
        weights["Signal_Tilted_Weight"] > weights["Equal_Weight"],
        "Increase",
        "Decrease_or_review",
    )
    signal_tilted_risk_reduction = (
        (equal_risk - signal_tilted_risk) / equal_risk * 100
        if equal_risk > 0
        else 0
    )

    risk_comparison = {
        "equal_weight": equal_risk,
        "inverse_volatility": inverse_risk,
        "markowitz_minimum_variance": markowitz_risk,
        "signal_tilted": signal_tilted_risk,
    }
    if markowitz_risk < equal_risk:
        recommended_action = (
            "The minimum-variance estimate is "
            f"{(equal_risk - markowitz_risk) / equal_risk:.1%} below "
            "the equal-weight estimate for this historical sample."
        )
    else:
        recommended_action = (
            "The minimum-variance estimate is not below the equal-weight "
            "estimate for this historical sample."
        )
    metrics = {
        "trading_days": int(len(returns)),
        "tickers": list(covariance.index),
        "equal_weight_annualized_volatility": equal_risk,
        "inverse_volatility_annualized_volatility": inverse_risk,
        "markowitz_annualized_volatility": markowitz_risk,
        "signal_tilted_annualized_volatility": signal_tilted_risk,
        "risk_reduction_percent": float(risk_reduction),
        "markowitz_risk_reduction_percent": float(
            (equal_risk - markowitz_risk) / equal_risk * 100
            if equal_risk > 0 else 0
        ),
        "signal_tilted_risk_reduction_percent": float(
            signal_tilted_risk_reduction
        ),
        "portfolio_annualized_volatility": risk_comparison,
        "minimum_observations_warning": (
            "Fewer than 120 aligned daily returns; covariance and allocations "
            "may be unstable."
            if len(returns) < 120 else None
        ),
        "covariance_estimator": "Ledoit-Wolf shrinkage, annualized by 252",
        "maximum_weight": max_weight,
        "recommended_action": recommended_action,
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
    if not args.confirm_retrain:
        raise InputError(
            "Retraining is gated. Re-run with --confirm-retrain and type "
            "RETRAIN at the confirmation prompt."
        )
    if not sys.stdin.isatty():
        raise InputError(
            "CLI retraining requires an interactive terminal so the "
            "confirmation can be entered."
        )
    if input(
        "This will replace the active model bundle after backing it up. "
        "Type RETRAIN to continue: "
    ).strip() != "RETRAIN":
        print("Retraining cancelled.")
        return 2

    if args.stock_file:
        prices = read_table(args.stock_file)
        if args.tickers:
            selected = normalize_tickers(args.tickers)
            prices = prices.loc[
                prices["Ticker"].astype(str).str.upper().isin(selected)
            ]
    else:
        prices = download_live_prices(args.period, tickers=args.tickers)

    if args.news_file:
        news_source = read_table(args.news_file)
    else:
        news_source = download_live_news(args.tickers)
    if args.stock_file and args.tickers and "Ticker" in news_source:
        selected = normalize_tickers(args.tickers)
        news_source = news_source.loc[
            news_source["Ticker"].astype(str).str.upper().isin(selected)
        ]
    news = prepare_news(news_source)

    prepared = prepare_prices(prices)
    (
        stock_model,
        logistic_model,
        news_model,
        metrics,
        test_output,
        backtest,
    ) = train_and_evaluate(prepared, news)
    backup_path = save_models(
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
        f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}; "
        f"majority baseline: {metrics['majority_baseline_accuracy']:.3f}"
    )
    print(f"Selected forecast horizon: {metrics['target_horizon_days']} sessions")
    if backup_path is not None:
        print(f"Previous model bundle backed up to: {backup_path}")
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
        if args.tickers:
            selected = normalize_tickers(args.tickers)
            prices = prices.loc[
                prices["Ticker"].astype(str).str.upper().isin(selected)
            ]
    else:
        prices = download_live_prices(
            args.period,
            ensure_indicator_history=True,
            tickers=args.tickers,
        )

    if args.news_file:
        news = read_table(args.news_file)
    else:
        news = download_live_news(args.tickers)

    stock_output, combined = predict_with_models(
        stock_model,
        news_model,
        prices,
        news,
        news_lookback_days=args.news_lookback_days,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    stock_output.to_csv(OUT / "proposal_stock_predictions.csv", index=False)
    combined.to_csv(OUT / "proposal_combined_predictions.csv", index=False)

    print(f"Proposal-aligned predictions written to: {OUT}")
    print(combined.to_string(index=False))
    if combined["Ticker"].nunique() < 2:
        print(
            "\nPortfolio analysis skipped: at least two tickers need "
            "usable aligned price history."
        )
        return 0

    weights, covariance, correlation, portfolio_metrics = build_portfolio(
        prices,
        combined,
        combined["Ticker"].astype(str).tolist(),
    )
    weights.to_csv(OUT / "proposal_portfolio_weights.csv", index=False)
    covariance.to_csv(OUT / "proposal_portfolio_covariance.csv", index=False)
    correlation.to_csv(OUT / "proposal_portfolio_correlation.csv", index=False)
    write_json(OUT / "proposal_portfolio_metrics.json", portfolio_metrics)
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
    train_parser.add_argument(
        "--confirm-retrain",
        action="store_true",
        help="Enable the interactive confirmation prompt before replacing models.",
    )
    train_parser.add_argument(
        "--tickers",
        nargs="+",
        help="Ticker symbols, for example --tickers AAPL MSFT",
    )
    train_parser.set_defaults(handler=train_command)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--stock-file")
    predict_parser.add_argument("--news-file")
    predict_parser.add_argument("--period", default="6mo")
    predict_parser.add_argument(
        "--tickers",
        nargs="+",
        help="Ticker symbols, for example --tickers AAPL MSFT",
    )
    predict_parser.add_argument(
        "--news-lookback-days",
        type=int,
        default=NEWS_LOOKBACK_DAYS,
    )
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
