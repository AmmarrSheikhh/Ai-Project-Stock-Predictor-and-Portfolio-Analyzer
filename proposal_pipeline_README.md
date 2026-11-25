# Proposal-Aligned Pipeline

This new pipeline implements the proposal items that are missing from the old
Deliverable scripts. It does not modify the existing project files or provide
web deployment.

## Run

Train/retrain from historical files:

```powershell
python proposal_pipeline.py train --stock-file stock_prices_cleaned.xls --news-file stock_news_cleaned.xls
```

Retrain using live Yahoo data:

```powershell
python proposal_pipeline.py train
```

Predict using the saved proposal models:

```powershell
python proposal_pipeline.py predict
```

## Implemented proposal scope

- RSI-14 and MACD stock features.
- Sentiment as an exogenous stock feature using date/ticker daily aggregation.
- Random Forest and Logistic Regression model comparison against a majority
  baseline.
- Chronological per-ticker backtests against Buy-and-Hold with 10 bps
  transaction costs.
- Ticker-level accuracy and F1 stability measures.
- Covariance and correlation matrices.
- Equal-weight, inverse-volatility, and Markowitz minimum-variance comparison.
- Signal-adjusted recommended weights.
- Before-and-after annualized volatility measurement.
- Human-readable increase/decrease portfolio actions.
- Persisted manual news labels used immediately for predictions and included
  in future retraining.
- Historical low/base/high investment scenarios for a selected basket.

## News alignment limitation

The historical news/stock merge is only as good as the supplied news file.
The bundled news file is small and uses weak lexicon labels. It is merged only
by exact ticker and normalized publication date. This is not a substitute for
a true historical, professionally labelled news/price dataset.
