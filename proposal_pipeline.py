from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "proposal_pipeline_outputs"
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
    if "Headline" not in news.columns:
        raise InputError("News data must contain a Headline column.")
    news["Headline"] = news["Headline"].fillna("").astype(str)
    news = news[news["Headline"].str.strip().ne("")]

    if "Ticker" not in news.columns:
        news["Ticker"] = "UNKNOWN"

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
            accuracy_score(news["Weak_Label"], predicted)
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
) -> tuple[RandomForestClassifier, Pipeline, dict, pd.DataFrame, pd.DataFrame]:
    news_model, news_metrics = train_news_model(news)
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

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[STOCK_FEATURES], train["Target_Up"].astype(int))

    probabilities = model.predict_proba(test[STOCK_FEATURES])[:, 1]
    predictions = model.predict(test[STOCK_FEATURES])
    test_output = test.copy()
    test_output["Predicted_Up"] = predictions
    test_output["Prediction_Probability"] = probabilities

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
    strategy_returns = (
        test_output["Prediction_Probability"].gt(0.5).astype(float)
        * test_output["Next_Day_Return"].fillna(0)
    )
    strategy_daily = (
        test_output.assign(Strategy_Return=strategy_returns)
        .groupby("Date")["Strategy_Return"]
        .mean()
    )
    buy_hold_daily = test_output.groupby("Date")["Next_Day_Return"].mean()

    strategy_curve = (1 + strategy_daily.fillna(0)).cumprod()
    buy_hold_curve = (1 + buy_hold_daily.fillna(0)).cumprod()

    backtest = pd.DataFrame({
        "Date": strategy_curve.index,
        "Strategy_Cumulative_Return": strategy_curve.to_numpy(),
        "BuyHold_Cumulative_Return": buy_hold_curve.reindex(strategy_curve.index).fillna(0).add(1).cumprod().to_numpy(),
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

    metrics = {
        "stock_features": STOCK_FEATURES,
        "split": "chronological 80/20",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "final_training_rows": int(len(labeled)),
        "accuracy": float(
            accuracy_score(test_output["Target_Up"], test_output["Predicted_Up"])
        ),
        "f1": float(
            f1_score(
                test_output["Target_Up"],
                test_output["Predicted_Up"],
                zero_division=0,
            )
        ),
        "ticker_stability": ticker_stability,
        "strategy_final_return": float(
            strategy_curve.iloc[-1] - 1
        ) if len(strategy_curve) else 0.0,
        "buy_hold_final_return": float(
            buy_hold_curve.iloc[-1] - 1
        ) if len(strategy_curve) else 0.0,
        "news_model": news_metrics,
    }
    return final_model, news_model, metrics, test_output, backtest


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_models(
    stock_model: RandomForestClassifier,
    news_model: Pipeline,
    metrics: dict,
    backtest: pd.DataFrame,
    test_output: pd.DataFrame,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stock_path = OUT / "proposal_stock_model.joblib"
    news_path = OUT / "proposal_news_model.joblib"
    joblib.dump(stock_model, stock_path)
    joblib.dump(news_model, news_path)
    write_json(OUT / "proposal_metrics.json", metrics)
    backtest.to_csv(OUT / "proposal_backtest.csv", index=False)
    test_output.to_csv(OUT / "proposal_stock_test_predictions.csv", index=False)


def load_models() -> tuple[RandomForestClassifier, Pipeline]:
    stock_path = OUT / "proposal_stock_model.joblib"
    news_path = OUT / "proposal_news_model.joblib"
    if not stock_path.exists() or not news_path.exists():
        raise InputError(
            "Proposal models are not trained yet. Run: "
            "python proposal_pipeline.py train"
        )
    return joblib.load(stock_path), joblib.load(news_path)


def predict_with_models(
    stock_model: RandomForestClassifier,
    news_model: Pipeline,
    prices: pd.DataFrame,
    news: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_prices = prepare_prices(prices)
    prepared_news = prepare_news(news)

    news_predictions = prepared_news.copy()
    news_predictions["Predicted_Sentiment"] = news_model.predict(
        news_predictions["Headline"]
    )
    news_predictions["Sentiment_Confidence"] = news_model.predict_proba(
        news_predictions["Headline"]
    ).max(axis=1)

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
    combined["Combined_Signal"] = (
        0.7 * combined["Prediction_Probability"]
        + 0.3 * combined["Sentiment_Signal"]
    )
    combined["Combined_Direction"] = np.where(
        combined["Combined_Signal"] >= 0.5,
        "UP",
        "DOWN",
    )
    return stock_output, combined


def build_portfolio(
    prices: pd.DataFrame,
    combined: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    prepared = prepare_prices(prices)
    returns = prepared.pivot_table(
        index="Date",
        columns="Ticker",
        values="Daily_Return",
    )
    returns = returns.reindex(
        columns=[ticker for ticker in TICKERS if ticker in returns.columns]
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
    risk_reduction = (
        (equal_risk - inverse_risk) / equal_risk * 100
        if equal_risk > 0
        else 0
    )

    weights = pd.DataFrame({
        "Ticker": covariance.index,
        "Equal_Weight": equal_weights.to_numpy(),
        "Inverse_Volatility_Weight": inverse_weights.to_numpy(),
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
        "risk_reduction_percent": float(risk_reduction),
        "recommended_action": (
            "Inverse-volatility or signal-adjusted weights reduce the "
            "historical portfolio volatility baseline."
        ),
        "annualized_covariance": covariance.round(8).to_dict(),
        "correlation": correlation.round(8).to_dict(),
    }
    covariance_output = covariance.reset_index().rename(columns={"index": "Ticker"})
    correlation_output = correlation.reset_index().rename(columns={"index": "Ticker"})
    return weights, covariance_output, correlation_output, metrics


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
    stock_model, news_model, metrics, test_output, backtest = (
        train_and_evaluate(prepared, news)
    )
    save_models(
        stock_model,
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
    stock_model, news_model = load_models()
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


raise SystemExit(main())
