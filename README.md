# W Trade

W Trade is a university project that combines stock-direction prediction,
news sentiment analysis, backtesting, and portfolio-risk analysis in an
interactive Streamlit dashboard.

The project is an educational prototype. It does not execute trades and should
not be used as investment advice.

## Features

- Random Forest or Logistic Regression stock direction prediction
- RSI-14 and MACD technical indicators
- TF-IDF and Logistic Regression news sentiment
- News sentiment merged into stock model features
- Adjustable technical/news signal weighting
- Chronological 80/20 stock evaluation
- Majority-baseline and classifier model comparison
- Per-ticker Buy-and-Hold versus cost-adjusted strategy backtesting
- Ticker-level accuracy and F1 stability
- User-selected stock baskets
- Covariance-based Markowitz minimum-variance optimization
- Equal-weight, inverse-volatility, Markowitz, and signal-adjusted allocation comparison
- Signal-adjusted portfolio allocations
- Historical low/base/high investment scenarios
- Persistent human news labels that override sentiment immediately and are
  included in future model retraining
- Live Yahoo Finance or uploaded-file inputs
- Saved model retraining and prediction
- Cyberpunk neon Streamlit interface

## Run locally

### 1. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Start the application

```powershell
streamlit run proposal_app.py
```

Open the local URL shown by Streamlit, usually
`http://localhost:8501`.

## Application pages

### Predict

Choose either live Yahoo Finance data or uploaded files, then run prediction.

The predictions use the saved models. The page supports:

- Viewing all available stocks
- Searching for a specific ticker
- Selecting one stock for a focused view
- Choosing Random Forest or Logistic Regression after retraining
- Adjusting the technical-versus-news signal blend
- Stock probability and combined-signal charts
- Downloadable prediction CSV files

### Portfolio

Choose a basket of supported tickers and calculate:

- Annualized covariance
- Correlation
- Equal-weight portfolio volatility
- Inverse-volatility portfolio volatility
- Markowitz minimum-variance allocations
- Signal-adjusted allocation suggestions
- Dollar allocations for a user-entered investment amount
- Historical low/base/high value scenarios over a selectable horizon

The scenarios use historical basket returns and are illustrative, not forecasts
or guarantees. They assume daily rebalancing at the selected allocation.

### News Review

Review current Yahoo headlines or uploaded news and label each headline as
positive, neutral, negative, or use the model's sentiment. Saved labels are
stored in `proposal_pipeline_outputs/proposal_news_labels.csv`, override
sentiment in subsequent predictions, and are included when the models are
retrained.

### Model Status

Displays stored model metrics, ticker stability, model comparisons, and
selectable per-ticker backtest curves. Strategy curves include 10 basis points
of transaction cost per position change.

### Retrain

Retrains and replaces the proposal-aligned stock and news models.

Retraining is intentional and should not run automatically on every prediction.
Each successful run is recorded in
`proposal_pipeline_outputs/proposal_training_history.csv` with its timestamp,
data sources and date coverage, row counts, model scores, backtest returns, and
training duration. The page shows the latest run and recent history and lets
you download the full log.

### Model Status

Displays stored model metrics, ticker stability, model comparisons, per-ticker
backtest curves, and project limitations.

## Upload file formats

### Stock file

Use CSV, XLS, or XLSX containing:

- `Date`
- `Ticker`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

CSV-disguised `.xls` files are supported.

### News file

Use CSV, XLS, or XLSX containing:

- `Ticker`
- `Headline`

Optional columns:

- `PublishedAt`
- `Sentiment` with values `-1`, `0`, or `1`
- `Label` with values `negative`, `neutral`, or `positive`

## Requirements

- Python 3.10 or newer
- Internet access for Yahoo Finance live data
- Streamlit
- Plotly
- pandas
- NumPy
- scikit-learn
- SciPy
- yfinance
- joblib
- openpyxl
- xlrd

## Deployment

The recommended free deployment target is Streamlit Community Cloud.

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select this repository and the main branch.
5. Set the entry point to `proposal_app.py`.
6. Deploy.

## Important limitations

- Stock performance is close to chance and is not reliable enough for real
  investment decisions.
- The bundled news corpus is small.
- News weak labels can be incorrect.
- Yahoo Finance availability and rate limits can affect live downloads.
- Retraining may need a network connection.
- Backtests include 10 bps per position change, but do not model taxes,
  slippage, liquidity, or live trading.

## Repository file guide

### Required app files

```text
proposal_app.py
proposal_pipeline.py
requirements.txt
.streamlit/config.toml
proposal_pipeline_outputs/
stock_prices_cleaned.xls
stock_news_cleaned.xls
```

Do not upload virtual-environment folders or local caches.
