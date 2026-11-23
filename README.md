# W Trade

W Trade is a university project that combines stock-direction prediction,
news sentiment analysis, backtesting, and portfolio-risk analysis in an
interactive Streamlit dashboard.

The project is an educational prototype. It does not execute trades and should
not be used as investment advice.

## Features

- Random Forest stock direction prediction
- RSI-14 and MACD technical indicators
- TF-IDF and Logistic Regression news sentiment
- News sentiment merged into stock model features
- Chronological 80/20 stock evaluation
- Buy-and-Hold versus strategy backtesting
- Ticker-level accuracy and F1 stability
- Covariance and correlation analysis
- Equal-weight and inverse-volatility portfolio comparison
- Signal-adjusted portfolio allocations
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
- Stock probability and combined-signal charts
- Downloadable prediction CSV files

### Portfolio

Calculates:

- Annualized covariance
- Correlation
- Equal-weight portfolio volatility
- Inverse-volatility portfolio volatility
- Signal-adjusted allocation suggestions

### Retrain

Retrains and replaces the proposal-aligned stock and news models.

Retraining is intentional and should not run automatically on every prediction.

### Model Status

Displays stored model metrics, ticker stability, backtest values, and limitations.

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
- The project does not include transaction costs or real-time trading.

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

### Optional documentation

```text
README.md
proposal_app_README.md
proposal_pipeline_README.md
```

### Optional university submission evidence

```text
Proposal F26-9 (1).pdf
F26-9_Del4.pdf
F26-9_Del3.py
F26-9_Del4.py
F26-9 Stock_Module.ipynb
F26-9 News_Module.ipynb
Del 5.png
```

Do not upload virtual-environment folders or local caches.
