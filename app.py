from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from streamlit.errors import StreamlitSecretNotFoundError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.pipeline import (  # noqa: E402
    COMBINED_NEUTRAL_BAND,
    MIN_HISTORY_ROWS,
    NEWS_LOOKBACK_DAYS,
    TICKERS,
    TRAINING_HISTORY_PATH,
    TRANSACTION_COST_BPS,
    InputError,
    append_training_record,
    apply_manual_news_labels,
    apply_signal_weights,
    build_portfolio,
    download_live_news,
    download_live_prices,
    load_models,
    normalize_tickers,
    predict_with_models,
    prepare_news,
    prepare_prices,
    project_investment,
    save_models,
    train_and_evaluate,
    update_manual_news_labels,
)
from ui.page_components.about import render as render_about  # noqa: E402
from ui.page_components.news_review import render as render_news_review  # noqa: E402
from ui.page_components.portfolio import render_risk_summary  # noqa: E402
from ui.theme import inject_accessibility_styles  # noqa: E402

OUTPUT_DIR = ROOT / "proposal_pipeline_outputs"
METRICS_PATH = OUTPUT_DIR / "proposal_metrics.json"
PORTFOLIO_METRICS_PATH = OUTPUT_DIR / "proposal_portfolio_metrics.json"


def configured_retrain_password() -> str | None:
    try:
        password = st.secrets.get("RETRAIN_PASSWORD")
    except StreamlitSecretNotFoundError:
        return None
    return str(password) if password else None
BACKTEST_PATH = OUTPUT_DIR / "proposal_backtest.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "proposal_stock_test_predictions.csv"
LOGISTIC_MODEL_PATH = OUTPUT_DIR / "proposal_logistic_model.joblib"

st.set_page_config(
    page_title="W Trade",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #08111f;
        --panel: #101d2e;
        --panel-2: #14243a;
        --border: #26364a;
        --text: #edf4ff;
        --muted: #91a4bd;
        --accent: #49a3ff;
        --green: #43d39e;
        --red: #ff6b7a;
        --gold: #e7c96f;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 74% 12%, rgba(0, 247, 255, 0.12), transparent 24%),
            radial-gradient(ellipse at 18% 82%, rgba(168, 85, 247, 0.13), transparent 28%),
            radial-gradient(ellipse at 62% 68%, rgba(255, 45, 149, 0.055), transparent 25%),
            linear-gradient(135deg, #050711 0%, #09101d 55%, #060912 100%);
        color: var(--text);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(46, 214, 255, 0.045) 1px, transparent 1px),
            linear-gradient(90deg, rgba(46, 214, 255, 0.045) 1px, transparent 1px);
        background-size: 62px 62px;
        transform: perspective(520px) rotateX(54deg) scale(1.5) translateY(26%);
        transform-origin: center 85%;
        mask-image: linear-gradient(to bottom, transparent 10%, black 72%);
        animation: cyber-grid 10s linear infinite;
    }

    .stApp::after {
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background:
            linear-gradient(105deg, transparent 42%, rgba(37, 208, 255, 0.055) 49%, transparent 56%),
            repeating-linear-gradient(0deg, rgba(255,255,255,.018) 0, rgba(255,255,255,.018) 1px, transparent 1px, transparent 4px);
        animation: neon-scan 8s linear infinite;
    }

    @keyframes cyber-grid {
        from { background-position: 0 0, 0 0; }
        to { background-position: 0 62px, 62px 62px; }
    }

    @keyframes neon-scan {
        0% { transform: translateY(-18%); opacity: .18; }
        50% { opacity: .48; }
        100% { transform: translateY(18%); opacity: .18; }
    }

    .starfield,
    .starfield::before,
    .starfield::after {
        position: fixed;
        inset: 0;
        content: "";
        pointer-events: none;
        z-index: 0;
    }

    .starfield {
        background-image:
            radial-gradient(circle, rgba(88,239,255,.92) 0 1px, transparent 1.8px),
            radial-gradient(circle, rgba(255,86,196,.72) 0 1px, transparent 1.8px),
            radial-gradient(circle, rgba(255,255,255,.52) 0 1px, transparent 1.8px);
        background-size: 170px 170px, 230px 230px, 310px 310px;
        background-position: 8px 22px, 110px 65px, 35px 145px;
        opacity: 0.46;
        animation: star-drift 38s linear infinite;
    }

    .starfield::before {
        background:
            radial-gradient(circle at 12% 18%, rgba(255,255,255,.55) 0 1px, transparent 2px),
            radial-gradient(circle at 78% 30%, rgba(73,163,255,.58) 0 1px, transparent 2px),
            radial-gradient(circle at 42% 76%, rgba(255,255,255,.38) 0 1px, transparent 2px);
        background-size: 260px 260px, 320px 320px, 390px 390px;
        opacity: 0.48;
        animation: star-drift-slow 70s linear infinite reverse;
    }

    .starfield::after {
        background: linear-gradient(
            115deg,
            transparent 15%,
            rgba(73,163,255,.03) 35%,
            transparent 55%
        );
        transform: translateX(-40%);
        animation: aurora 16s ease-in-out infinite alternate;
    }

    @keyframes star-drift {
        from { background-position: 8px 22px, 110px 65px, 35px 145px; }
        to { background-position: 178px 192px, 340px 295px, 345px 455px; }
    }

    @keyframes star-drift-slow {
        from { transform: translateY(0) scale(1); }
        to { transform: translateY(-90px) scale(1.04); }
    }

    @keyframes aurora {
        from { transform: translateX(-55%) scaleX(.9); opacity: .20; }
        to { transform: translateX(45%) scaleX(1.15); opacity: .55; }
    }

    [data-testid="stAppViewContainer"] > .main {
        position: relative;
        z-index: 1;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(8, 13, 28, 0.98), rgba(6, 9, 20, 0.98)),
            repeating-linear-gradient(90deg, transparent 0 11px, rgba(45, 214, 255, .035) 12px 13px);
        border-right: 1px solid rgba(45, 214, 255, 0.18);
        box-shadow: 12px 0 36px rgba(0, 0, 0, 0.30);
    }

    [data-testid="stSidebar"] * {
        color: #dce8f7;
    }

    [data-testid="stSidebar"] .stButton {
        width: 100%;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 44px;
        margin: 2px 0;
        border-radius: 7px;
        justify-content: flex-start;
        text-align: left;
        font-weight: 680;
        letter-spacing: .25px;
        border: 1px solid transparent;
        background: transparent;
        color: #91a8c3;
        box-shadow: none;
        transition: all .16s ease;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(91, 224, 255, .22);
        color: #effeff;
        background: linear-gradient(90deg, rgba(45, 214, 255, .13), rgba(168, 85, 247, .07));
        box-shadow: inset 2px 0 0 rgba(65, 220, 255, .72);
        transform: translateX(2px);
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        border: 1px solid rgba(91, 224, 255, .26);
        background:
            linear-gradient(90deg, rgba(45, 214, 255, .18), rgba(168, 85, 247, .09));
        color: #effeff;
        box-shadow:
            inset 3px 0 0 #41dcff,
            0 0 16px rgba(45, 214, 255, .08);
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(16,29,46,.94), rgba(20,36,58,.94));
        border: 1px solid rgba(133, 170, 211, 0.17);
        border-radius: 15px;
        padding: 16px;
        backdrop-filter: blur(14px);
        box-shadow: 0 16px 32px rgba(0, 0, 0, 0.22);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
    }

    .stTabs [aria-selected="true"] {
        background: var(--panel-2);
        color: var(--accent) !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 24px 26px;
        margin-bottom: 18px;
        border: 1px solid rgba(65, 220, 255, 0.22);
        border-radius: 16px;
        background:
            linear-gradient(110deg, rgba(65, 220, 255, 0.13), transparent 48%),
            linear-gradient(145deg, rgba(13,25,47,.96), rgba(8,12,28,.96));
        box-shadow:
            0 18px 42px rgba(0, 0, 0, 0.24),
            inset 0 0 28px rgba(65, 220, 255, 0.025);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        right: -70px;
        top: -145px;
        border: 1px solid rgba(73,163,255,.24);
        border-radius: 50%;
        box-shadow:
            0 0 0 28px rgba(73,163,255,.025),
            0 0 0 58px rgba(73,163,255,.018);
    }

    .hero h1 {
        margin: 0;
        font-size: 32px;
        color: #f4fdff;
        letter-spacing: -0.4px;
        text-shadow: 0 0 24px rgba(65, 220, 255, 0.12);
    }

    .hero p {
        margin: 7px 0 0;
        color: #91a8c3;
    }

    .stButton > button[kind="primary"] {
        border: 1px solid rgba(65, 220, 255, 0.84);
        border-radius: 8px;
        background: linear-gradient(90deg, #0b8da8 0%, #176bb2 60%, #7d3ba8 100%);
        color: #ffffff;
        box-shadow: 0 0 18px rgba(65, 220, 255, 0.14);
    }

    .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid rgba(255, 86, 196, 0.48);
        color: #ffdff4;
        background: rgba(74, 25, 78, 0.28);
    }

    .status-up {
        color: var(--green);
        font-weight: 700;
    }

    .status-down {
        color: var(--red);
        font-weight: 700;
    }

    .dataframe {
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    .stAlert {
        border-radius: 10px;
    }

    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]) {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 8px;
        background: rgba(16, 29, 46, 0.68);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>AI-Driven Stock &amp; Portfolio Intelligence</h1>
        <p>Live stock direction &bull; News sentiment &bull; Portfolio risk &bull; Proposal-aligned analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="starfield"></div>', unsafe_allow_html=True)
inject_accessibility_styles(st)


@st.cache_data(ttl=900)
def cached_live_prices(
    period: str,
    ensure_indicator_history: bool = False,
    tickers: tuple[str, ...] = (),
) -> pd.DataFrame:
    return download_live_prices(
        period,
        ensure_indicator_history=ensure_indicator_history,
        tickers=tickers or None,
    )


@st.cache_data(ttl=900)
def cached_live_news(tickers: tuple[str, ...] = ()) -> pd.DataFrame:
    return download_live_news(tickers or None)


def data_editor_price_upload(label: str):
    uploaded = st.sidebar.file_uploader(
        label,
        type=["csv", "xls", "xlsx"],
        key=label,
    )
    if uploaded is None:
        return None
    return read_uploaded_table(uploaded)


def read_uploaded_table(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    uploaded.seek(0)
    header = uploaded.read(8)
    uploaded.seek(0)

    csv_disguised_as_excel = (
        name.endswith(".xls")
        and not header.startswith(b"PK")
        and not header.startswith(b"\xD0\xCF")
    )
    if csv_disguised_as_excel:
        return pd.read_csv(uploaded)
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def read_json_if_present(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def download_csv(data: pd.DataFrame, filename: str) -> None:
    st.download_button(
        f"Download {filename}",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 4px 10px;">
            <div style="font-size:24px;font-weight:780;margin-top:3px;color:#F4FDFF;
                        letter-spacing:.8px;text-shadow:0 0 18px rgba(65,220,255,.16);">
                W Trade
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:11px;color:#71869F;text-transform:uppercase;'
        'letter-spacing:1.2px;margin:5px 4px 7px;">Navigation</div>',
        unsafe_allow_html=True,
    )
    if "page" not in st.session_state:
        st.session_state["page"] = "Predict"

    nav_items = [
        "Predict",
        "Portfolio",
        "News Review",
        "Retrain",
        "Model Status",
        "About",
    ]
    for nav_name in nav_items:
        selected = st.session_state["page"] == nav_name
        if st.button(
            nav_name,
            key=f"nav_{nav_name}",
            type="primary" if selected else "secondary",
            use_container_width=True,
        ):
            st.session_state["page"] = nav_name
            st.rerun()

    page = st.session_state["page"]
    st.divider()
    with st.expander("Quick start", expanded=False):
        st.markdown(
            """
            **1. Predict**  
            Choose live Yahoo data or upload files, then run prediction.

            **2. Portfolio**  
            Calculate risk and suggested allocations.

            **3. Retrain**  
            Update both models only when you want them retrained.
            """
        )
    st.header("Data Source")
    data_source = st.radio("Choose input", ["Live Yahoo", "Upload files"])

    stock_period = st.selectbox(
        "Stock history period",
        ["6mo", "1y", "5y", "10y"],
        index=0,
    )
    st.caption(
        "Prediction feature warm-up downloads at least 1 year of daily prices; "
        "the result is dated to the latest completed session."
    )
    ticker_universe_input = st.text_input(
        "Ticker universe",
        value=", ".join(TICKERS),
        help="Enter one or more symbols separated by commas.",
    )
    try:
        available_tickers = normalize_tickers(
            ticker_universe_input.split(",")
        )
    except InputError as error:
        st.error(str(error))
        available_tickers = []
    selected_tickers = st.multiselect(
        "Portfolio basket",
        options=available_tickers,
        default=available_tickers,
        help="This basket filters prediction, portfolio, and backtest results.",
    )
    model_options = ["Random Forest"]
    if LOGISTIC_MODEL_PATH.exists():
        model_options.append("Logistic Regression")
    selected_model = st.selectbox("Stock prediction model", model_options)
    if not LOGISTIC_MODEL_PATH.exists():
        st.caption("Retrain models to enable Logistic Regression.")
    technical_weight = st.slider(
        "Technical signal weight",
        min_value=50,
        max_value=90,
        value=70,
        step=5,
        format="%d%%",
        help="The remaining weight is assigned to news sentiment.",
    ) / 100
    news_lookback_days = st.slider(
        "Recent news window (calendar days)",
        min_value=1,
        max_value=14,
        value=NEWS_LOOKBACK_DAYS,
    )

if data_source == "Upload files":
    uploaded_prices = data_editor_price_upload("Stock price file")
    uploaded_news = data_editor_price_upload("News headline file")
else:
    uploaded_prices = None
    uploaded_news = None


if page == "Predict":
    st.subheader("Stock and news prediction")

    model_metrics = read_json_if_present(METRICS_PATH)
    forecast_horizon = model_metrics.get("target_horizon_days", 1)
    st.info(
        "Predictions use the currently saved proposal models. "
        f"Use Retrain to update the models. Forecast horizon: "
        f"{forecast_horizon} trading session(s), selected by walk-forward validation."
    )

    if not selected_tickers:
        st.warning("Select at least one ticker in the sidebar basket.")
    if selected_tickers and st.button("Run prediction", type="primary"):
        try:
            rf_model, logistic_model, news_model = load_models()
            stock_model = (
                logistic_model
                if selected_model == "Logistic Regression"
                else rf_model
            )
            if stock_model is None:
                raise InputError(
                    "Logistic Regression is not trained yet. Run Retrain first."
                )
            if uploaded_prices is not None:
                prices = uploaded_prices
            else:
                prices = cached_live_prices(
                    stock_period,
                    ensure_indicator_history=True,
                    tickers=tuple(selected_tickers),
                )

            if uploaded_news is not None:
                news = uploaded_news
            else:
                news = cached_live_news(tuple(selected_tickers))

            stock_output, combined = predict_with_models(
                stock_model,
                news_model,
                prices,
                news,
                technical_weight,
                news_lookback_days,
            )
            st.session_state["stock_output"] = stock_output
            st.session_state["combined_output"] = combined
            st.session_state["prediction_model"] = selected_model
            st.session_state["prediction_tickers"] = tuple(selected_tickers)

            if "News_Sentiment" in stock_output.columns:
                st.success("Prediction complete.")
        except Exception as error:
            st.error(str(error))

    if selected_tickers and "stock_output" in st.session_state:
        stored_tickers = st.session_state.get("prediction_tickers", ())
        if (
            st.session_state.get("prediction_model") != selected_model
            or stored_tickers != tuple(selected_tickers)
        ):
            st.info(
                "The displayed results are from the previous model or basket. "
                "Run prediction again to refresh them."
            )
        stock_output = st.session_state["stock_output"].copy()
        combined = apply_signal_weights(
            st.session_state["combined_output"],
            technical_weight,
        )
        st.session_state["combined_output"] = combined
        stock_output = stock_output[
            stock_output["Ticker"].astype(str).isin(selected_tickers)
        ].copy()
        skipped = (
            stock_output.loc[stock_output["Status"].eq("skipped")].copy()
            if "Status" in stock_output.columns
            else pd.DataFrame()
        )
        for _, row in skipped.iterrows():
            st.warning(f"{row['Ticker']}: {row['Message']}")
        if "Status" in stock_output.columns:
            stock_output = stock_output.loc[
                stock_output["Status"].eq("ok")
            ].copy()
        if "Data_As_Of" not in stock_output.columns and "Price_Date" in stock_output.columns:
            stock_output["Data_As_Of"] = pd.to_datetime(
                stock_output["Price_Date"], errors="coerce"
            ).dt.date
        combined = combined[
            combined["Ticker"].astype(str).isin(selected_tickers)
        ].copy()
        if stock_output.empty:
            if skipped.empty:
                st.warning("No predictions are available for the selected basket.")
            else:
                st.info(
                    f"Each selected ticker needs at least {MIN_HISTORY_ROWS} "
                    "usable history rows before a prediction is available."
                )
            st.stop()

        overview = st.columns(4)
        overview[0].metric(
            "Stocks analyzed",
            len(stock_output),
        )
        overview[1].metric(
            "UP forecasts",
            int(stock_output["Predicted_Up"].sum()),
        )
        overview[2].metric(
            "Mean P(UP)",
            f"{stock_output['Prediction_Probability'].mean():.2%}",
        )
        overview[3].metric(
            "Headlines",
            int(combined["Headline_Count"].fillna(0).sum()),
        )

        st.markdown("### View stocks")
        filter_mode = st.radio(
            "Stock view",
            ["All stocks", "Specific stock"],
            horizontal=True,
            label_visibility="collapsed",
        )

        available_tickers = sorted(stock_output["Ticker"].dropna().astype(str).unique())
        if filter_mode == "All stocks":
            ticker_search = st.text_input(
                "Search stocks",
                placeholder="Search by ticker, for example NVDA",
                key="predict_stock_search",
            )
            query = ticker_search.strip().upper()
            if query:
                visible_tickers = [
                    ticker for ticker in available_tickers
                    if query in ticker.upper()
                ]
            else:
                visible_tickers = available_tickers
        else:
            selected_ticker = st.selectbox(
                "Choose an available stock",
                available_tickers,
                key="specific_stock",
            )
            visible_tickers = [selected_ticker]

        stock_output = stock_output[
            stock_output["Ticker"].astype(str).isin(visible_tickers)
        ].copy()
        combined = combined[
            combined["Ticker"].astype(str).isin(visible_tickers)
        ].copy()

        if stock_output.empty:
            st.warning("No stocks match the current search.")
            st.stop()

        stock_output_display = stock_output.copy()
        stock_output_display["News_Status"] = np.where(
            stock_output_display.get(
                "Has_News",
                stock_output_display["News_Sentiment_Count"].gt(0),
            ),
            "News available",
            "No recent news",
        )
        stock_output_display["Direction"] = stock_output_display[
            "Predicted_Up"
        ].map({1: "UP", 0: "DOWN"})
        stock_output_display["Probability"] = (
            stock_output_display["Prediction_Probability"] * 100
        ).round(2).astype(str) + "%"

        st.markdown("### Latest stock direction")
        st.dataframe(
            stock_output_display[
                [
                    "Ticker",
                    "Data_As_Of",
                    "Price_Date",
                    "Close",
                    "Direction",
                    "Probability",
                    "RSI_14",
                    "MACD",
                    "News_Status",
                    "News_Count",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Stock probabilities")
        stock_chart = stock_output.sort_values("Prediction_Probability").copy()
        stock_chart["Probability_Margin"] = (
            stock_chart["Prediction_Probability"] - 0.5
        )
        stock_chart["Margin_Points"] = (
            stock_chart["Probability_Margin"] * 100
        ).round(2)
        stock_colors = [
            "#48E0C2" if direction == 1 else "#FF5E95"
            for direction in stock_chart["Predicted_Up"]
        ]

        probability_figure = go.Figure()
        probability_figure.add_vline(
            x=0,
            line_width=1,
            line_dash="dash",
            line_color="#91A4BD",
        )
        probability_figure.add_trace(
            go.Bar(
                x=stock_chart["Margin_Points"],
                y=stock_chart["Ticker"],
                orientation="h",
                marker={
                    "color": stock_colors,
                    "line": {"color": "rgba(240,248,255,.42)", "width": 1},
                    "cornerradius": 5,
                },
                text=[
                    f"{probability:.2%} probability"
                    for probability in stock_chart["Prediction_Probability"]
                ],
                textposition="outside",
                customdata=stock_chart["Prediction_Probability"],
                hovertemplate="<b>%{y}</b><br>Probability: %{customdata:.2%}<br>Margin: %{x:+.2f} points<extra></extra>",
            )
        )
        probability_figure.update_layout(
            template="plotly_dark",
            height=max(240, 80 + 44 * len(stock_chart)),
            margin={"l": 20, "r": 100, "t": 15, "b": 25},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,17,31,.45)",
            bargap=0.42,
            showlegend=False,
            xaxis={
                "title": "Margin above/below the 50% neutral line",
                "ticksuffix": " pts",
                "zeroline": False,
                "gridcolor": "rgba(132,173,213,.15)",
                "range": [
                    max(-16, (stock_chart["Margin_Points"].min() - 2)),
                    min(16, (stock_chart["Margin_Points"].max() + 2)),
                ],
            },
            yaxis={"gridcolor": "rgba(132,173,213,.08)", "tickfont": {"size": 13}},
            font={"color": "#EDF4FF"},
        )
        st.plotly_chart(probability_figure, use_container_width=True)

        download_csv(
            stock_output,
            "proposal_stock_predictions.csv",
        )

        st.markdown("### Combined stock and news signal")
        if (
            data_source == "Live Yahoo"
            and combined["Headline_Count"].fillna(0).sum() == 0
        ):
            st.info(
                "Yahoo returned no headlines for the recent-news window. "
                "This may mean limited coverage or rate limiting; combined "
                "signals are stock-only for this run."
            )
        st.caption(
            f"Combined P(UP) between "
            f"{(0.5 - COMBINED_NEUTRAL_BAND):.0%} and "
            f"{(0.5 + COMBINED_NEUTRAL_BAND):.0%} is labeled NEUTRAL "
            "(no directional action)."
        )
        combined_display = combined.copy()
        combined_display["Stock_Model_P_UP_%"] = (
            combined_display["Prediction_Probability"] * 100
        ).round(2)
        combined_display["News_Adjustment_%"] = (
            combined_display["News_Adjustment"] * 100
        ).round(2)
        combined_display["Combined_P_UP_%"] = (
            combined_display["Combined_Signal"] * 100
        ).round(2).astype(str) + "%"
        combined_display["News_Status"] = np.where(
            combined_display["Has_News"],
            "News available",
            "No recent news",
        )
        st.dataframe(
            combined_display[
                [
                    "Ticker",
                    "Data_As_Of",
                    "Stock_Model_P_UP_%",
                    "News_Adjustment_%",
                    "Combined_P_UP_%",
                    "News_Count",
                    "News_Status",
                    "Combined_Direction",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Combined signal margin")
        combined_chart = combined.sort_values("Combined_Signal").copy()
        combined_chart["Combined_Margin_Points"] = (
            (combined_chart["Combined_Signal"] - 0.5) * 100
        ).round(2)
        combined_colors = [
            "#48E0C2" if direction == "UP" else "#FF5E95"
            for direction in combined_chart["Combined_Direction"]
        ]
        combined_figure = go.Figure()
        combined_figure.add_vline(
            x=0,
            line_width=1,
            line_dash="dash",
            line_color="#91A4BD",
        )
        combined_figure.add_trace(
            go.Bar(
                x=combined_chart["Combined_Margin_Points"],
                y=combined_chart["Ticker"],
                orientation="h",
                marker={
                    "color": combined_colors,
                    "line": {"color": "rgba(240,248,255,.42)", "width": 1},
                    "cornerradius": 5,
                },
                text=[
                    f"{signal:.2%}"
                    for signal in combined_chart["Combined_Signal"]
                ],
                textposition="outside",
                customdata=combined_chart["Combined_Signal"],
                hovertemplate="<b>%{y}</b><br>Combined signal: %{customdata:.2%}<br>Margin: %{x:+.2f} points<extra></extra>",
            )
        )
        combined_figure.update_layout(
            template="plotly_dark",
            height=max(240, 80 + 44 * len(combined_chart)),
            margin={"l": 20, "r": 100, "t": 15, "b": 25},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,17,31,.45)",
            bargap=0.42,
            showlegend=False,
            xaxis={
                "title": "Margin above/below the 50% neutral line",
                "ticksuffix": " pts",
                "zeroline": False,
                "gridcolor": "rgba(132,173,213,.15)",
                "range": [
                    max(-16, combined_chart["Combined_Margin_Points"].min() - 2),
                    min(16, combined_chart["Combined_Margin_Points"].max() + 2),
                ],
            },
            yaxis={"gridcolor": "rgba(132,173,213,.08)", "tickfont": {"size": 13}},
            font={"color": "#EDF4FF"},
        )
        st.plotly_chart(combined_figure, use_container_width=True)

        download_csv(
            combined,
            "proposal_combined_predictions.csv",
        )


elif page == "News Review":
    render_news_review(
        st,
        pd,
        selected_tickers,
        uploaded_news,
        cached_live_news,
        prepare_news,
        apply_manual_news_labels,
        load_models,
        InputError,
        update_manual_news_labels,
    )


elif page == "Portfolio":
    st.subheader("Portfolio risk and diversification")
    portfolio_period = st.selectbox(
        "Portfolio return history",
        ["3mo", "6mo", "1y", "5y", "10y"],
        index=3,
        key="portfolio_period",
    )
    investment_amount = st.number_input(
        "Investment amount (USD)",
        min_value=100.0,
        max_value=100_000_000.0,
        value=10_000.0,
        step=1_000.0,
    )
    projection_years = st.slider(
        "Projection horizon (years)",
        min_value=1,
        max_value=30,
        value=5,
    )
    allocation_strategy = st.selectbox(
        "Allocation used for investment estimate",
        [
            "Markowitz minimum variance",
            "Inverse volatility",
            "Signal-adjusted",
            "Equal weight",
        ],
    )
    allocation_columns = {
        "Markowitz minimum variance": "Markowitz_Weight",
        "Inverse volatility": "Inverse_Volatility_Weight",
        "Signal-adjusted": "Signal_Tilted_Weight",
        "Equal weight": "Equal_Weight",
    }

    uploaded_price = st.session_state.get("Stock price file")
    uploaded_price_name = getattr(uploaded_price, "name", "")
    portfolio_signature = (
        tuple(selected_tickers),
        portfolio_period,
        data_source,
        uploaded_price_name,
    )

    if len(selected_tickers) < 2:
        st.warning("Select at least two tickers in the sidebar basket.")
    if len(selected_tickers) >= 2 and st.button(
        "Run portfolio analysis", type="primary"
    ):
        try:
            if uploaded_prices is not None:
                prices = uploaded_prices
            else:
                prices = cached_live_prices(
                    portfolio_period, tickers=tuple(selected_tickers)
                )

            combined = st.session_state.get("combined_output")
            if combined is not None:
                combined = apply_signal_weights(combined, technical_weight)
                combined = combined[
                    combined["Ticker"].isin(selected_tickers)
                ]
            weights, covariance, correlation, metrics = build_portfolio(
                prices,
                combined,
                selected_tickers,
            )
            st.session_state["weights"] = weights
            st.session_state["covariance"] = covariance
            st.session_state["correlation"] = correlation
            st.session_state["portfolio_metrics"] = metrics
            st.session_state["portfolio_prices"] = prices
            st.session_state["portfolio_signature"] = portfolio_signature
            st.success("Portfolio analysis complete.")
        except Exception as error:
            st.error(str(error))

    if "weights" in st.session_state:
        if st.session_state.get("portfolio_signature") != portfolio_signature:
            st.info("The basket or data source changed. Run portfolio analysis again.")
        else:
            weights = st.session_state["weights"].copy()
            covariance = st.session_state["covariance"]
            correlation = st.session_state["correlation"]
            metrics = st.session_state["portfolio_metrics"]

            render_risk_summary(st, metrics)

            st.markdown("### Basket allocation strategies")
            weight_figure = go.Figure()
            for column, label in [
                ("Equal_Weight", "Equal weight"),
                ("Inverse_Volatility_Weight", "Inverse volatility"),
                ("Markowitz_Weight", "Markowitz minimum variance"),
                ("Signal_Tilted_Weight", "Signal-tilted"),
            ]:
                weight_figure.add_bar(
                    x=weights["Ticker"],
                    y=weights[column],
                    name=label,
                )
            weight_figure.update_layout(
                template="plotly_dark",
                barmode="group",
                yaxis={"title": "Portfolio weight", "tickformat": ".0%"},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8,17,31,.45)",
                font={"color": "#EDF4FF"},
            )
            st.plotly_chart(weight_figure, use_container_width=True)

            selected_weight_column = allocation_columns[allocation_strategy]
            weights["Investment_Amount"] = (
                weights[selected_weight_column] * investment_amount
            )
            st.markdown("### Selected basket and dollar allocations")
            st.dataframe(
                weights[
                    [
                        "Ticker",
                        "Annualized_Volatility",
                        "Equal_Weight",
                        "Inverse_Volatility_Weight",
                        "Markowitz_Weight",
                        "Signal_Tilted_Weight",
                        "Equal_Weight_Risk_Contribution",
                        "Inverse_Volatility_Risk_Contribution",
                        "Markowitz_Risk_Contribution",
                        "Signal_Tilted_Risk_Contribution",
                        "Investment_Amount",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Annualized_Volatility": st.column_config.NumberColumn(
                        "Annualized volatility", format="%.2%%"
                    ),
                    "Equal_Weight": st.column_config.NumberColumn(
                        "Equal weight", format="%.2%%"
                    ),
                    "Inverse_Volatility_Weight": st.column_config.NumberColumn(
                        "Inverse-volatility weight", format="%.2%%"
                    ),
                    "Markowitz_Weight": st.column_config.NumberColumn(
                        "Markowitz weight", format="%.2%%"
                    ),
                    "Signal_Tilted_Weight": st.column_config.NumberColumn(
                        "Signal-tilted weight", format="%.2%%"
                    ),
                    "Equal_Weight_Risk_Contribution": st.column_config.NumberColumn(
                        "Equal-weight risk contribution", format="%.2%%"
                    ),
                    "Inverse_Volatility_Risk_Contribution": st.column_config.NumberColumn(
                        "Inverse-volatility risk contribution", format="%.2%%"
                    ),
                    "Markowitz_Risk_Contribution": st.column_config.NumberColumn(
                        "Markowitz risk contribution", format="%.2%%"
                    ),
                    "Signal_Tilted_Risk_Contribution": st.column_config.NumberColumn(
                        "Signal-tilted risk contribution", format="%.2%%"
                    ),
                    "Investment_Amount": st.column_config.NumberColumn(
                        "Suggested investment", format="$%.2f"
                    ),
                },
            )
            download_csv(weights, "proposal_portfolio_weights.csv")

            st.markdown("### Historical investment scenarios")
            try:
                projection = project_investment(
                    st.session_state["portfolio_prices"],
                    weights.set_index("Ticker")[selected_weight_column],
                    investment_amount,
                    projection_years,
                )
                st.caption(
                    f"Illustrative {projection_years}-year estimate using "
                    f"{portfolio_period} of the basket's historical daily returns. "
                    "Scenarios are the 5th, "
                    "50th, and 95th percentiles under a log-return assumption; "
                    "weights are assumed to be rebalanced daily. Not a forecast "
                    "or guarantee."
                )
                projection_columns = st.columns(3)
                projection_columns[0].metric(
                    "Low scenario (5th percentile)",
                    f"${projection['low']:,.2f}",
                )
                projection_columns[1].metric(
                    "Base scenario (median)",
                    f"${projection['base']:,.2f}",
                )
                projection_columns[2].metric(
                    "High scenario (95th percentile)",
                    f"${projection['high']:,.2f}",
                )
                projection_figure = go.Figure()
                projection_figure.add_bar(
                    x=["Initial investment", "Low", "Base", "High"],
                    y=[
                        investment_amount,
                        projection["low"],
                        projection["base"],
                        projection["high"],
                    ],
                    marker_color=["#91A4BD", "#FF5E95", "#48E0C2", "#49A3FF"],
                )
                projection_figure.update_layout(
                    template="plotly_dark",
                    yaxis={"title": "Illustrative value (USD)", "tickprefix": "$"},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(8,17,31,.45)",
                    font={"color": "#EDF4FF"},
                )
                st.plotly_chart(projection_figure, use_container_width=True)
                st.write(
                    f"Historical annualized return: "
                    f"{projection['annualized_return']:.2%} · "
                    f"annualized volatility: "
                    f"{projection['annualized_volatility']:.2%}"
                )
            except InputError as error:
                st.warning(str(error))

            covariance_values = covariance.set_index("Ticker")
            correlation_values = correlation.set_index("Ticker")
            st.markdown("### Covariance")
            st.dataframe(covariance_values, use_container_width=True)
            st.markdown("### Correlation")
            st.dataframe(
                correlation_values.style.background_gradient(
                    cmap="RdYlGn", axis=None
                ),
                use_container_width=True,
            )


elif page == "Retrain":
    st.subheader("Retrain and replace proposal models")
    st.warning(
        "Retraining replaces the stored proposal stock and news models. "
        "It should be run explicitly rather than on every prediction."
    )

    training_history = (
        pd.read_csv(TRAINING_HISTORY_PATH)
        if TRAINING_HISTORY_PATH.exists()
        else pd.DataFrame()
    )
    if not training_history.empty:
        latest = training_history.sort_values("Trained_At_UTC").iloc[-1]
        st.markdown("### Last successful training")
        st.caption(
            f"Completed {latest['Trained_At_UTC']} · "
            f"{latest['Price_Source']} · {latest['Price_History_Period']} · "
            f"price dates {latest['Price_Date_From']} to "
            f"{latest['Price_Date_To']}"
        )
        latest_columns = st.columns(7)
        latest_columns[0].metric(
            "Training rows",
            f"{int(latest['Training_Rows']):,}",
        )
        latest_columns[1].metric(
            "RF balanced accuracy",
            f"{latest.get('RF_Balanced_Accuracy', latest.get('RF_Accuracy', 0)):.2%}",
        )
        latest_columns[2].metric(
            "Random Forest F1",
            f"{latest['RF_F1']:.2%}",
        )
        latest_columns[3].metric(
            "Logistic Regression F1",
            f"{latest['LR_F1']:.2%}",
        )
        latest_columns[4].metric(
            "Majority baseline",
            f"{latest.get('Majority_Baseline_Accuracy', 0):.2%}",
        )
        latest_columns[5].metric(
            "Forecast horizon",
            f"{int(latest.get('Target_Horizon_Days', 1))} session(s)",
        )
        latest_columns[6].metric(
            "Human-labeled headlines",
            f"{int(latest['Human_Labeled_News_Rows']):,}",
        )
        st.caption(f"News source: {latest['News_Source']}")
    else:
        st.info(
            "No retraining history is available yet. The first successful run "
            "after this feature is added will appear here; older runs cannot "
            "be reconstructed because their data source and period were not saved."
        )

    retrain_period = st.selectbox(
        "Retraining stock history",
        ["1y", "5y", "10y"],
        index=2,
        key="retrain_period",
    )

    expected_password = configured_retrain_password()
    if expected_password is None:
        st.info(
            "Retraining is locked. Configure RETRAIN_PASSWORD in "
            ".streamlit/secrets.toml to enable it."
        )
        confirm_retrain = False
    else:
        confirm_checkbox = st.checkbox(
            "I reviewed the evaluation and understand this will replace the "
            "active models. The existing bundle will be backed up first.",
            key="confirm_retrain",
        )
        entered_password = st.text_input(
            "Retraining password",
            type="password",
            key="retrain_password",
        )
        if entered_password and entered_password != expected_password:
            st.error("The retraining password is incorrect.")
        confirm_retrain = (
            confirm_checkbox and entered_password == expected_password
        )

    if st.button(
        "Retrain models",
        type="primary",
        disabled=not confirm_retrain,
    ):
        try:
            training_started = time.perf_counter()
            price_upload = st.session_state.get("Stock price file")
            news_upload = st.session_state.get("News headline file")
            price_source = (
                f"Uploaded file: {getattr(price_upload, 'name', 'stock data')}"
                if uploaded_prices is not None
                else "Yahoo Finance"
            )
            news_source = (
                f"Uploaded file: {getattr(news_upload, 'name', 'news data')}"
                if uploaded_news is not None
                else "Yahoo Finance"
            )
            if uploaded_prices is not None:
                prices = uploaded_prices.loc[
                    uploaded_prices["Ticker"]
                    .astype(str)
                    .str.upper()
                    .isin(selected_tickers)
                ]
            else:
                prices = cached_live_prices(
                    retrain_period, tickers=tuple(selected_tickers)
                )

            if uploaded_news is not None:
                news = prepare_news(
                    uploaded_news.loc[
                        uploaded_news["Ticker"]
                        .astype(str)
                        .str.upper()
                        .isin(selected_tickers)
                    ]
                    if "Ticker" in uploaded_news
                    else uploaded_news
                )
            else:
                news = prepare_news(
                    cached_live_news(tuple(selected_tickers))
                )

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
            model_comparison = metrics["model_comparison"]
            training_record = {
                "Trained_At_UTC": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "Price_Source": price_source,
                "News_Source": news_source,
                "Price_History_Period": (
                    "Uploaded file" if uploaded_prices is not None else retrain_period
                ),
                "Price_Rows": int(len(prepared)),
                "Training_Rows": int(metrics["train_rows"]),
                "Test_Rows": int(metrics["test_rows"]),
                "Split_Method": metrics["split"],
                "Stock_Features": ", ".join(metrics["stock_features"]),
                "Price_Date_From": prepared["Date"].min().date().isoformat(),
                "Price_Date_To": prepared["Date"].max().date().isoformat(),
                "Tickers": ", ".join(sorted(prepared["Ticker"].unique())),
                "News_Training_Rows": int(
                    metrics["news_model"]["training_rows"]
                ),
                "Human_Labeled_News_Rows": int(
                    metrics["news_model"].get("manual_label_rows", 0)
                ),
                "RF_Accuracy": model_comparison["Random Forest"]["accuracy"],
                "RF_F1": model_comparison["Random Forest"]["f1"],
                "RF_Balanced_Accuracy": metrics["balanced_accuracy"],
                "Majority_Baseline_Accuracy": (
                    metrics["majority_baseline_accuracy"]
                ),
                "RF_Brier_Score": metrics["brier_score"],
                "Target_Horizon_Days": metrics["target_horizon_days"],
                "LR_Accuracy": model_comparison["Logistic Regression"]["accuracy"],
                "LR_F1": model_comparison["Logistic Regression"]["f1"],
                "RF_Strategy_Return": metrics["strategy_final_return"],
                "LR_Strategy_Return": metrics["logistic_strategy_final_return"],
                "Buy_Hold_Return": metrics["buy_hold_final_return"],
                "Transaction_Cost_Bps": metrics["backtest_transaction_cost_bps"],
                "Training_Seconds": round(
                    time.perf_counter() - training_started, 2
                ),
            }
            try:
                append_training_record(training_record)
            except Exception as history_error:
                st.error(
                    "Models were retrained and saved, but the training-history "
                    f"record could not be written: {history_error}"
                )
            else:
                st.success("Models retrained, saved, and recorded.")
            if backup_path is not None:
                st.info(f"Previous model bundle backed up to: {backup_path}")
            st.metric("Stock accuracy", f"{metrics['accuracy']:.2%}")
            st.metric("Stock F1", f"{metrics['f1']:.2%}")
            st.metric(
                "Strategy cumulative return",
                f"{metrics['strategy_final_return']:.2%}",
            )
            st.metric(
                "Buy-and-Hold cumulative return",
                f"{metrics['buy_hold_final_return']:.2%}",
            )
            st.markdown("### Out-of-sample model comparison")
            st.dataframe(
                pd.DataFrame.from_dict(
                    metrics["model_comparison"],
                    orient="index",
                ).rename(columns={
                    "accuracy": "Accuracy",
                    "f1": "F1",
                }),
                use_container_width=True,
            )
            st.json(metrics)
        except Exception as error:
            st.error(str(error))

    if TRAINING_HISTORY_PATH.exists():
        training_history = pd.read_csv(TRAINING_HISTORY_PATH)
    if not training_history.empty:
        st.markdown("### Training history")
        history_columns = [
            "Trained_At_UTC",
            "Price_Source",
            "News_Source",
            "Price_History_Period",
            "Price_Date_From",
            "Price_Date_To",
            "Tickers",
            "Price_Rows",
            "Training_Rows",
            "Test_Rows",
            "Split_Method",
            "Stock_Features",
            "News_Training_Rows",
            "Human_Labeled_News_Rows",
            "RF_Accuracy",
            "RF_F1",
            "RF_Balanced_Accuracy",
            "Majority_Baseline_Accuracy",
            "RF_Brier_Score",
            "Target_Horizon_Days",
            "LR_Accuracy",
            "LR_F1",
            "RF_Strategy_Return",
            "LR_Strategy_Return",
            "Buy_Hold_Return",
            "Transaction_Cost_Bps",
            "Training_Seconds",
        ]
        recent_history = (
            training_history.sort_values("Trained_At_UTC", ascending=False)
            .head(20)
        )
        st.dataframe(
            recent_history[
                [
                    column for column in history_columns
                    if column in recent_history.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "RF_Accuracy": st.column_config.NumberColumn(
                    "RF accuracy", format="%.2%%"
                ),
                "RF_F1": st.column_config.NumberColumn("RF F1", format="%.2%%"),
                "RF_Balanced_Accuracy": st.column_config.NumberColumn(
                    "RF balanced accuracy", format="%.2%%"
                ),
                "Majority_Baseline_Accuracy": st.column_config.NumberColumn(
                    "Majority baseline", format="%.2%%"
                ),
                "RF_Brier_Score": st.column_config.NumberColumn(
                    "RF Brier score", format="%.3f"
                ),
                "LR_Accuracy": st.column_config.NumberColumn(
                    "LR accuracy", format="%.2%%"
                ),
                "LR_F1": st.column_config.NumberColumn("LR F1", format="%.2%%"),
                "RF_Strategy_Return": st.column_config.NumberColumn(
                    "RF backtest return", format="%.2%%"
                ),
                "LR_Strategy_Return": st.column_config.NumberColumn(
                    "LR backtest return", format="%.2%%"
                ),
                "Buy_Hold_Return": st.column_config.NumberColumn(
                    "Buy-and-hold return", format="%.2%%"
                ),
                "Training_Seconds": st.column_config.NumberColumn(
                    "Training time (seconds)", format="%.2f"
                ),
            },
        )
        st.download_button(
            "Download full training history",
            data=training_history.to_csv(index=False).encode("utf-8"),
            file_name="proposal_training_history.csv",
            mime="text/csv",
        )


elif page == "Model Status":
    st.subheader("Current model status")

    metrics = read_json_if_present(METRICS_PATH)
    portfolio_metrics = read_json_if_present(PORTFOLIO_METRICS_PATH)

    if not metrics:
        st.warning(
            "No proposal models or metrics are stored yet. "
            "Run Retrain first."
        )
    else:
        st.markdown("### Stock model")
        st.write(f"**Accuracy:** {metrics.get('accuracy', 0):.2%}")
        st.write(f"**F1:** {metrics.get('f1', 0):.2%}")
        st.write(
            f"**Training rows:** {metrics.get('final_training_rows', 'Unavailable')}"
        )
        st.write(
            f"**Features:** {', '.join(metrics.get('stock_features', []))}"
        )
        feature_importance = metrics.get("feature_importance", {})
        if feature_importance:
            importance_frame = (
                pd.Series(feature_importance, name="Importance")
                .sort_values(ascending=False)
                .head(20)
                .sort_values()
                .rename_axis("Feature")
                .reset_index()
            )
            importance_figure = go.Figure()
            importance_figure.add_bar(
                x=importance_frame["Importance"],
                y=importance_frame["Feature"],
                orientation="h",
                marker_color="#48E0C2",
            )
            importance_figure.update_layout(
                template="plotly_dark",
                title="Random Forest feature importance (top 20)",
                xaxis_title="Impurity-based importance",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8,17,31,.45)",
                font={"color": "#EDF4FF"},
            )
            st.plotly_chart(importance_figure, use_container_width=True)
        if metrics.get("target_horizon_days"):
            st.caption(
                f"Forecast target: {metrics['target_horizon_days']} trading "
                f"session(s); returns within ±{metrics.get('target_dead_zone', 0):.2%} "
                "are excluded from direction labels. "
                f"P(UP) threshold: {metrics.get('direction_threshold', 0.5):.0%}; "
                f"ties: {metrics.get('direction_tie_policy', 'UP')}."
            )
        st.caption("Random Forest uses class_weight='balanced'.")
        if metrics.get("train_class_balance"):
            st.write(
                f"**Training class counts:** "
                f"{metrics['train_class_balance']}"
            )
        if metrics.get("test_class_balance"):
            st.write(
                f"**Test class counts:** {metrics['test_class_balance']}"
            )

        model_comparison = metrics.get("model_comparison", {})
        if model_comparison:
            st.markdown("### Out-of-sample model comparison")
            st.dataframe(
                pd.DataFrame.from_dict(
                    model_comparison,
                    orient="index",
                ).rename(columns={
                    "accuracy": "Accuracy",
                    "f1": "F1",
                }),
                use_container_width=True,
            )

        if TEST_PREDICTIONS_PATH.exists():
            saved_test = pd.read_csv(TEST_PREDICTIONS_PATH)
            prediction_column = "Random_Forest_Predicted_Up"
            if {"Target_Up", prediction_column}.issubset(saved_test.columns):
                actual = pd.to_numeric(saved_test["Target_Up"], errors="raise").astype(int)
                predicted = pd.to_numeric(
                    saved_test[prediction_column], errors="raise"
                ).astype(int)
                balanced_accuracy = metrics.get("balanced_accuracy")
                if balanced_accuracy is None:
                    balanced_accuracy = balanced_accuracy_score(actual, predicted)
                baseline = metrics.get("majority_baseline_accuracy")
                if baseline is None:
                    baseline = model_comparison.get(
                        "Majority Class Baseline", {}
                    ).get("accuracy")
                st.markdown("### Held-out baseline and class performance")
                performance_columns = st.columns(3)
                performance_columns[0].metric(
                    "Random Forest accuracy",
                    f"{accuracy_score(actual, predicted):.2%}",
                )
                performance_columns[1].metric(
                    "Majority-class baseline",
                    f"{baseline:.2%}" if baseline is not None else "Unavailable",
                )
                performance_columns[2].metric(
                    "Balanced accuracy",
                    f"{balanced_accuracy:.2%}",
                )
                matrix = metrics.get("confusion_matrix")
                if matrix is None:
                    matrix = confusion_matrix(
                        actual, predicted, labels=[0, 1]
                    ).tolist()
                st.dataframe(
                    pd.DataFrame(
                        matrix,
                        index=["True DOWN", "True UP"],
                        columns=["Predicted DOWN", "Predicted UP"],
                    ),
                    use_container_width=True,
                )

        accuracy_interval = metrics.get("accuracy_confidence_interval_95")
        if accuracy_interval and len(accuracy_interval) == 2:
            low, high = accuracy_interval
            st.caption(
                f"95% exact binomial confidence interval for held-out accuracy: "
                f"{low:.2%}–{high:.2%}. "
                f"One-sided test vs. training majority baseline: "
                f"p={metrics.get('accuracy_binomial_p_value_vs_majority', float('nan')):.3f} "
                "(naive iid assumption; serial dependence is not adjusted)."
            )
            baseline = metrics.get("majority_baseline_accuracy")
            if baseline is not None and low <= baseline <= high:
                st.warning(
                    "The accuracy interval includes the majority-class baseline; "
                    "the result does not establish a reliable improvement."
                )

        horizon_selection = metrics.get("target_horizon_selection", {})
        horizon_evaluations = horizon_selection.get("evaluations", {})
        if horizon_evaluations:
            st.markdown("### Walk-forward horizon comparison")
            horizon_rows = []
            for horizon, result in horizon_evaluations.items():
                horizon_rows.append({
                    "Horizon (sessions)": int(horizon),
                    "Mean balanced accuracy": result.get(
                        "mean_balanced_accuracy"
                    ),
                    "Balanced accuracy SD": result.get(
                        "std_balanced_accuracy"
                    ),
                    "Mean ROC AUC": result.get("mean_roc_auc"),
                    "Mean Brier score": result.get("mean_brier_score"),
                    "Majority baseline": result.get(
                        "mean_majority_baseline_accuracy"
                    ),
                    "Previous-day baseline": result.get(
                        "mean_previous_day_direction_accuracy"
                    ),
                })
            st.dataframe(
                pd.DataFrame(horizon_rows),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"Selected {metrics.get('target_horizon_days', 'unknown')}-session "
                "target using walk-forward balanced accuracy on the development "
                "period only; the final chronological test period was not used "
                "for horizon selection."
            )

        calibration = metrics.get("calibration_curve", {})
        mean_probabilities = calibration.get("mean_predicted_probability", [])
        observed_fractions = calibration.get("fraction_positive", [])
        if mean_probabilities and observed_fractions:
            calibration_figure = go.Figure()
            calibration_figure.add_scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect calibration",
                line={"dash": "dash"},
            )
            calibration_figure.add_scatter(
                x=mean_probabilities,
                y=observed_fractions,
                mode="lines+markers",
                name="Random Forest",
            )
            calibration_figure.update_layout(
                template="plotly_dark",
                title="Held-out probability calibration",
                xaxis={"title": "Mean predicted P(UP)", "range": [0, 1]},
                yaxis={"title": "Observed UP frequency", "range": [0, 1]},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8,17,31,.45)",
                font={"color": "#EDF4FF"},
            )
            st.plotly_chart(calibration_figure, use_container_width=True)

        st.markdown("### Ticker stability")
        stability = metrics.get("ticker_stability", {})
        if stability:
            st.dataframe(
                pd.DataFrame(stability).T,
                use_container_width=True,
            )

        st.markdown("### News model")
        news_metrics = metrics.get("news_model", {})
        st.write(
            f"**Training rows:** "
            f"{news_metrics.get('training_rows', 'Unavailable')}"
        )
        st.caption(news_metrics.get(
            "evaluation_status",
            "Independent human-labeled evaluation is unavailable.",
        ))
        if "manual_holdout_accuracy" in news_metrics:
            st.write(
                f"**Human-labeled holdout accuracy:** "
                f"{news_metrics['manual_holdout_accuracy']:.2%}"
            )
            st.write(
                f"**Human-labeled holdout balanced accuracy:** "
                f"{news_metrics['manual_holdout_balanced_accuracy']:.2%}"
            )

        st.markdown("### Backtest")
        st.caption(
            f"Predictions held for {metrics.get('backtest_entry_frequency_days', 1)} "
            "trading session(s) per non-overlapping entry."
        )
        st.write(
            f"Strategy: {metrics.get('strategy_final_return', 0):.2%}"
        )
        st.write(
            f"Buy-and-Hold: "
            f"{metrics.get('buy_hold_final_return', 0):.2%}"
        )
        backtest_performance = metrics.get("backtest_performance", {})
        if backtest_performance:
            performance_table = pd.DataFrame(backtest_performance).T.rename(
                columns={
                    "annualized_volatility": "Annualized volatility",
                    "sharpe_ratio": "Sharpe ratio",
                    "max_drawdown": "Maximum drawdown",
                    "win_rate": "Win rate",
                    "turnover": "Turnover",
                    "final_return": "Final return",
                }
            )
            st.dataframe(
                performance_table,
                use_container_width=True,
            )

    if BACKTEST_PATH.exists():
        backtest = pd.read_csv(BACKTEST_PATH)
        required_curve_columns = {
            "Ticker",
            "Date",
            "BuyHold_Cumulative_Value",
            "Random_Forest_Cumulative_Value",
            "Logistic_Regression_Cumulative_Value",
        }
        if required_curve_columns.issubset(backtest.columns):
            backtest["Date"] = pd.to_datetime(backtest["Date"])
            available_backtest_tickers = [
                ticker for ticker in selected_tickers
                if ticker in set(backtest["Ticker"].astype(str))
            ]
            if available_backtest_tickers:
                st.markdown(
                    "### Per-stock backtest "
                    f"({metrics.get('backtest_transaction_cost_bps', TRANSACTION_COST_BPS)} "
                    "bps transaction cost)"
                )
                backtest_ticker = st.selectbox(
                    "Backtest ticker",
                    available_backtest_tickers,
                )
                backtest_model = st.selectbox(
                    "Backtest model",
                    ["Random Forest", "Logistic Regression"],
                )
                model_curve = (
                    "Logistic_Regression_Cumulative_Value"
                    if backtest_model == "Logistic Regression"
                    else "Random_Forest_Cumulative_Value"
                )
                curve_data = backtest[
                    backtest["Ticker"].astype(str) == backtest_ticker
                ].sort_values("Date")
                curve_figure = go.Figure()
                curve_figure.add_scatter(
                    x=curve_data["Date"],
                    y=curve_data["BuyHold_Cumulative_Value"],
                    mode="lines",
                    name="Buy and hold",
                    line={"color": "#91A4BD"},
                )
                curve_figure.add_scatter(
                    x=curve_data["Date"],
                    y=curve_data[model_curve],
                    mode="lines",
                    name=f"{backtest_model} after costs",
                    line={"color": "#48E0C2"},
                )
                curve_figure.update_layout(
                    template="plotly_dark",
                    yaxis={"title": "Growth of $1"},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(8,17,31,.45)",
                    font={"color": "#EDF4FF"},
                )
                st.plotly_chart(curve_figure, use_container_width=True)
        else:
            st.info("Retrain to generate the per-ticker, cost-adjusted backtest.")

    if portfolio_metrics:
        st.markdown("### Latest portfolio metrics")
        st.json(portfolio_metrics)


else:
    render_about(st)

