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

- **Predict:** live or file-based stock/news prediction.
- **Portfolio:** covariance, correlation, volatility, and allocations.
- **Retrain:** explicitly replace the proposal-aligned models.
- **Model Status:** metrics and training limitations.
- **About:** project explanation.

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
