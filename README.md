# W Trade

W Trade is an educational Streamlit prototype for stock-direction modeling,
financial-news sentiment, strategy backtesting, and portfolio-risk analysis.
It does not execute trades and is not investment advice.

## Features

- Live Yahoo Finance data or user-uploaded stock and news files
- Completed-session daily prices, normalized to exchange dates
- RSI, MACD, scale-free technical features, and configurable ticker universe
- Financial news sentiment with separate no-news handling and manual labels
- Random Forest and Logistic Regression model evaluation
- Chronological, purged validation and probability calibration for newly
  trained stock models
- Majority-class, balanced-accuracy, confusion-matrix, and backtest reporting
- Portfolio allocation comparisons and risk summaries
- Explicit, password-protected Streamlit retraining
- Read-only diagnostic script for the currently saved model artifacts

The current saved models have **not** been retrained by these changes. Their
stored metrics still describe the legacy model, and those artifacts are
preserved until a retrain is explicitly confirmed.

## Requirements and installation

The pinned dependencies were verified with Python 3.12.8 on Windows.

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

For app-only use, install `requirements.txt` instead of
`requirements-dev.txt`.

## Run the application

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

The original entry point remains available:

```powershell
.\venv\Scripts\python.exe -m streamlit run proposal_app.py
```

Both entry points use the same app. The established page structure is Predict,
Portfolio, Retrain, Model Status, News Review, and About. The ticker defaults
are in `ticker_universe.json`; the UI also allows selecting a custom basket.

### Predict and data handling

- Live downloads request a longer history than the selected chart period to
  allow indicators to warm up. Prediction requires at least 60 usable rows
  after feature processing; tickers below this threshold are skipped with a
  per-ticker explanation.
- A current US daily bar is omitted until the New York market has closed at
  16:00. Displayed price dates are calendar dates, and results show their
  `Data as of` date.
- The default prediction period is six months. Short periods that cannot meet
  the minimum-history requirement are not offered as prediction periods.
- News is aggregated over the configured recent window. Neutral headlines are
  a numeric zero sentiment; no matching recent headlines are represented
  separately and cause a stock-only prediction.
- Results show the stock-model probability, news adjustment, combined
  probability, and headline count as separate values.
- Yahoo errors, including rate limiting, are surfaced in the app. Live Yahoo
  requests require internet access; uploaded-file prediction works without
  Yahoo data.

### Upload formats

Stock CSV, XLS, or XLSX files require `Date`, `Ticker`, `Open`, `High`, `Low`,
`Close`, and `Volume` columns.

News CSV, XLS, or XLSX files require `Ticker` and `Headline`. `PublishedAt`,
`Sentiment` (`-1`, `0`, or `1`), and `Label` (`negative`, `neutral`, or
`positive`) are optional.

## Diagnostics and tests

The diagnostic reads the existing held-out predictions and saved models; it
does not retrain or overwrite model files.

```powershell
.\venv\Scripts\python.exe .\diagnose_model.py
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m ruff check .
```

## Retraining safeguards

Prediction never retrains models automatically. Retraining replaces the saved
model bundle, so back up the project or its model outputs before proceeding.
The CLI requires `--confirm-retrain` and an interactive `RETRAIN` confirmation.
The Streamlit Retrain page requires an explicit checkbox and a password read
from `.streamlit/secrets.toml`; create that local file using
`.streamlit/secrets.toml.example` as a template. Never commit the real secrets
file or share its password.

The CLI training command is:

```powershell
.\venv\Scripts\python.exe .\proposal_pipeline.py train --stock-file .\stock_prices_cleaned.xls --news-file .\stock_news_cleaned.xls --confirm-retrain
```

The command prompts for `RETRAIN` before training. Successful retraining backs
up the previous output bundle under `proposal_pipeline_outputs_backup\`.
Tests train only in memory or temporary directories.

## Model and evaluation notes

- The direction target uses a future close/return relative to the current
  completed session; features are computed from current and prior data. New
  model validation is chronological and purges overlapping target horizons.
- Class-weight balancing and sigmoid calibration apply only when the user
  explicitly retrains. The diagnostic and Model Status metrics continue to
  describe the currently saved artifact until then.
- Strategies include a five-basis-point transaction cost per position change.
  Backtest and portfolio estimates are historical illustrations, not forecasts.
- The bundled news data is small and may not align with current market dates.
  The live prediction window can therefore contain no news for a ticker.
- Yahoo availability and rate limits can affect live data retrieval. The
  prototype does not model taxes, liquidity, slippage, or execution.

## Project layout

- `app.py`: Streamlit page orchestration
- `proposal_app.py`: backward-compatible Streamlit entry point
- `data/pipeline.py`: data preparation, features, modeling, evaluation,
  prediction, portfolio analysis, and guarded persistence
- `proposal_pipeline.py`: backward-compatible pipeline module and CLI entry
- `data/universe.py`, `ticker_universe.json`: ticker configuration
- `ui/page_components/`, `ui/`: page helpers and accessibility/theme behavior
- `diagnose_model.py`: read-only saved-model diagnostics
- `tests/`: focused pipeline and app tests
- `proposal_pipeline_outputs/`: persisted models and metrics
