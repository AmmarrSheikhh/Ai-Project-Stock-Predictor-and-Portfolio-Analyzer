def render(streamlit) -> None:
    streamlit.subheader("About this project")
    streamlit.markdown(
        """
        W Trade is a university prototype for stock-direction research,
        news sentiment, and portfolio-risk analysis.

        1. **Stock direction** uses a calibrated Random Forest or Logistic
           Regression with scale-free technical features, including RSI-14,
           normalized MACD, rolling volatility, ATR, and volume z-scores.
        2. **News sentiment** uses VADER with a financial lexicon extension;
           the TF-IDF classifier reports independent evaluation only when a
           human-labeled holdout is available.
        3. **Portfolio risk** compares equal-weight, inverse-volatility,
           Markowitz minimum-variance, and signal-tilted allocations using
           shrinkage covariance estimates.

        The application does not place trades, guarantee returns, or provide
        regulated financial advice. Investment scenarios are illustrative
        historical estimates, not predictions. Direction models can remain
        close to chance even after calibration.
        """
    )
