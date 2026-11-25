from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from proposal_pipeline import (  # noqa: E402
    InputError,
    TICKERS,
    TRAINING_HISTORY_PATH,
    append_training_record,
    apply_manual_news_labels,
    apply_signal_weights,
    build_portfolio,
    download_live_news,
    download_live_prices,
    load_models,
    predict_with_models,
    prepare_news,
    prepare_prices,
    project_investment,
    train_and_evaluate,
    save_models,
    update_manual_news_labels,
)


OUTPUT_DIR = ROOT / "proposal_pipeline_outputs"
METRICS_PATH = OUTPUT_DIR / "proposal_metrics.json"
PORTFOLIO_METRICS_PATH = OUTPUT_DIR / "proposal_portfolio_metrics.json"
BACKTEST_PATH = OUTPUT_DIR / "proposal_backtest.csv"
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


@st.cache_data(ttl=900)
def cached_live_prices(period: str) -> pd.DataFrame:
    return download_live_prices(period)


@st.cache_data(ttl=900)
def cached_live_news() -> pd.DataFrame:
    return download_live_news()


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
        ["1mo", "3mo", "6mo", "1y", "5y", "10y"],
        index=0,
    )
    selected_tickers = st.multiselect(
        "Portfolio basket",
        options=TICKERS,
        default=TICKERS,
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

if data_source == "Upload files":
    uploaded_prices = data_editor_price_upload("Stock price file")
    uploaded_news = data_editor_price_upload("News headline file")
else:
    uploaded_prices = None
    uploaded_news = None


if page == "Predict":
    st.subheader("Stock and news prediction")

    st.info(
        "Predictions use the currently saved proposal models. "
        "Use Retrain to update the models."
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
                prices = cached_live_prices(stock_period)

            if uploaded_news is not None:
                news = uploaded_news
            else:
                news = cached_live_news()

            stock_output, combined = predict_with_models(
                stock_model,
                news_model,
                prices,
                news,
                technical_weight,
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
        combined = combined[
            combined["Ticker"].astype(str).isin(selected_tickers)
        ].copy()
        if stock_output.empty:
            st.warning("No predictions are available for the selected basket.")
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
            "Average confidence",
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
                    "Price_Date",
                    "Close",
                    "Direction",
                    "Probability",
                    "RSI_14",
                    "MACD",
                    "News_Sentiment",
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
        combined_display = combined.copy()
        combined_display["Combined_Signal_%"] = (
            combined_display["Combined_Signal"] * 100
        ).round(2).astype(str) + "%"
        st.dataframe(
            combined_display[
                [
                    "Ticker",
                    "Prediction_Probability",
                    "Predicted_Sentiment_Mean",
                    "Headline_Count",
                    "Combined_Direction",
                    "Combined_Signal_%",
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
    st.subheader("Review and label financial news")
    st.caption(
        "Your labels are saved locally, override model sentiment immediately, "
        "and are used as training labels the next time you retrain."
    )

    try:
        raw_news = uploaded_news if uploaded_news is not None else cached_live_news()
        reviewed = apply_manual_news_labels(prepare_news(raw_news))
        if selected_tickers:
            reviewed = reviewed[
                reviewed["Ticker"].isin(selected_tickers)
            ]
        else:
            st.warning("Select tickers in the sidebar basket to review their news.")
            st.stop()
        reviewed = reviewed.drop_duplicates(
            ["Ticker", "Headline"], keep="last"
        ).reset_index(drop=True)
        if reviewed.empty:
            st.info("No headlines were found for the selected basket.")
            st.stop()
        try:
            _, _, news_model = load_models()
            predicted_labels = news_model.predict(reviewed["Headline"])
        except InputError:
            predicted_labels = reviewed["Weak_Label"].to_numpy()
            st.info(
                "No trained news model is available. Current labels use the "
                "weak lexicon until you retrain."
            )

        sentiment_names = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        editor = reviewed[["Ticker", "Headline"]].copy()
        editor["Model sentiment"] = [
            sentiment_names[int(label)] for label in predicted_labels
        ]
        editor["Your label"] = reviewed["Manual_Label"].map(
            lambda value: sentiment_names[int(value)]
            if pd.notna(value) else "Use model"
        )
        st.write(f"Headlines to review: {len(editor)}")
        edited = st.data_editor(
            editor,
            use_container_width=True,
            hide_index=True,
            disabled=["Ticker", "Headline", "Model sentiment"],
            column_config={
                "Your label": st.column_config.SelectboxColumn(
                    "Your label",
                    options=["Use model", "Positive", "Neutral", "Negative"],
                    required=True,
                )
            },
            key="news_label_editor",
        )
        if st.button("Save news labels", type="primary"):
            reviewed_labels = edited[["Ticker", "Headline", "Your label"]].copy()
            reviewed_labels["Label"] = reviewed_labels["Your label"].map({
                "Negative": -1,
                "Neutral": 0,
                "Positive": 1,
            })
            update_manual_news_labels(
                reviewed_labels[["Ticker", "Headline", "Label"]]
            )
            st.success("News labels saved and will be used immediately.")
            st.rerun()
    except Exception as error:
        st.error(str(error))


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
        "Signal-adjusted": "Optimized_Weight",
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
                prices = cached_live_prices(portfolio_period)

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

            columns = st.columns(4)
            columns[0].metric(
                "Equal-weight volatility",
                f"{metrics['equal_weight_annualized_volatility']:.2%}",
            )
            columns[1].metric(
                "Inverse-volatility volatility",
                f"{metrics['inverse_volatility_annualized_volatility']:.2%}",
            )
            columns[2].metric(
                "Markowitz volatility",
                f"{metrics['markowitz_annualized_volatility']:.2%}",
            )
            columns[3].metric(
                "Markowitz risk reduction",
                f"{metrics['markowitz_risk_reduction_percent']:.2f}%",
            )

            st.markdown("### Basket allocation strategies")
            weight_figure = go.Figure()
            for column, label in [
                ("Equal_Weight", "Equal weight"),
                ("Inverse_Volatility_Weight", "Inverse volatility"),
                ("Markowitz_Weight", "Markowitz minimum variance"),
                ("Optimized_Weight", "Signal-adjusted"),
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
                        "Optimized_Weight",
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
                    "Optimized_Weight": st.column_config.NumberColumn(
                        "Signal-adjusted weight", format="%.2%%"
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
        latest_columns = st.columns(4)
        latest_columns[0].metric(
            "Training rows",
            f"{int(latest['Training_Rows']):,}",
        )
        latest_columns[1].metric(
            "Random Forest F1",
            f"{latest['RF_F1']:.2%}",
        )
        latest_columns[2].metric(
            "Logistic Regression F1",
            f"{latest['LR_F1']:.2%}",
        )
        latest_columns[3].metric(
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

    if st.button("Retrain models", type="primary"):
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
                prices = uploaded_prices
            else:
                prices = cached_live_prices(retrain_period)

            if uploaded_news is not None:
                news = prepare_news(uploaded_news)
            else:
                news = prepare_news(cached_live_news())

            prepared = prepare_prices(prices)
            (
                stock_model,
                logistic_model,
                news_model,
                metrics,
                test_output,
                backtest,
            ) = train_and_evaluate(prepared, news)
            save_models(
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
            recent_history[history_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "RF_Accuracy": st.column_config.NumberColumn(
                    "RF accuracy", format="%.2%%"
                ),
                "RF_F1": st.column_config.NumberColumn("RF F1", format="%.2%%"),
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
        st.write(
            f"**Training accuracy:** "
            f"{news_metrics.get('training_accuracy', 0):.2%}"
        )

        st.markdown("### Backtest")
        st.write(
            f"Strategy: {metrics.get('strategy_final_return', 0):.2%}"
        )
        st.write(
            f"Buy-and-Hold: "
            f"{metrics.get('buy_hold_final_return', 0):.2%}"
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
                st.markdown("### Per-stock backtest (10 bps transaction cost)")
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
    st.subheader("About this project")
    st.markdown(
        """
        This project combines three proposed modules:

        1. **Stock Trend Predictor** - Random Forest or Logistic Regression
           with technical features, including RSI and MACD.
        2. **News Sentiment Signal** - TF-IDF and Logistic Regression.
           Saved human labels override model sentiment and are included on retraining.
        3. **Portfolio Risk Analyzer** - selectable stock baskets, covariance,
           correlation, Markowitz minimum-variance, inverse-volatility, and
           signal-adjusted allocations.

        The UI does not implement a trading system. It does not place trades,
        guarantee returns, or provide regulated financial advice. Investment
        scenarios are illustrative historical estimates, not predictions.

        News labels may be weakly generated where verified labels are absent.
        Stock accuracy near 50% is close to random direction prediction.
        """
    )

