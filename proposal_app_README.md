# Proposal-Aligned Streamlit UI

This is a separate Streamlit interface for the proposal-aligned pipeline.
It does not modify or replace the existing scripts or notebooks.

## Install

Install the UI dependency:

```powershell
python -m pip install streamlit
```

The project should already have the machine-learning and data dependencies
used by `proposal_pipeline.py`.

## Run locally

```powershell
streamlit run proposal_app.py
```

Streamlit will open in the default browser.

## Pages

- **Predict:** live or file-based prediction, model selection, and adjustable
  technical/news weighting.
- **Portfolio:** select a ticker basket, compare allocation strategies, and
  view illustrative investment scenarios.
- **News Review:** label headlines; saved labels override sentiment immediately
  and are included in future retraining.
- **Retrain:** explicitly replace the proposal-aligned models.
- **Model Status:** model comparisons, ticker stability, and per-ticker
  transaction-cost backtests.
- **About:** project explanation and limitations.

The portfolio estimate uses historical returns to show low/base/high scenarios.
It is not a forecast or guarantee and assumes daily rebalancing at the selected
allocation.

The Retrain page preserves a downloadable history of successful runs in
`proposal_pipeline_outputs/proposal_training_history.csv`, including data
coverage, model metrics, backtest returns, and training duration.

## Live prediction

The prediction page downloads:

- One month of Yahoo stock history.
- The latest Yahoo headlines.

It loads saved proposal models. It does not retrain unless Retrain is used.

## File prediction

Use the sidebar upload controls:

- Stock file: `Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `Volume`.
- News file: `Ticker`, `Headline`, and optionally `PublishedAt`,
  `Sentiment`, or `Label`.

## Free deployment

Streamlit Community Cloud is the recommended first deployment option.
Push this file and its project dependencies to GitHub, then create a
Streamlit Community Cloud app with this entry point:

```text
proposal_app.py
```
