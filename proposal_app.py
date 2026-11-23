from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from proposal_pipeline import (  # noqa: E402
    TICKERS,
    build_portfolio,
    download_live_news,
    download_live_prices,
    load_models,
    predict_with_models,
    prepare_news,
    prepare_prices,
    read_table,
    train_and_evaluate,
    save_models,
)


OUTPUT_DIR = ROOT / "proposal_pipeline_outputs"
METRICS_PATH = OUTPUT_DIR / "proposal_metrics.json"
PORTFOLIO_METRICS_PATH = OUTPUT_DIR / "proposal_portfolio_metrics.json"

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

    if st.button("Run prediction", type="primary"):
        try:
            stock_model, news_model = load_models()
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
            )
            st.session_state["stock_output"] = stock_output
            st.session_state["combined_output"] = combined

            if "News_Sentiment" in stock_output.columns:
                st.success("Prediction complete.")
        except Exception as error:
            st.error(str(error))

    if "stock_output" in st.session_state:
        stock_output = st.session_state["stock_output"]
        combined = st.session_state["combined_output"]

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


elif page == "Portfolio":
    st.subheader("Portfolio risk and diversification")

    portfolio_period = st.selectbox(
        "Portfolio return history",
        ["3mo", "6mo", "1y", "5y", "10y"],
        index=2,
        key="portfolio_period",
    )

    if st.button("Run portfolio analysis", type="primary"):
        try:
            if uploaded_prices is not None:
                prices = uploaded_prices
            else:
                prices = cached_live_prices(portfolio_period)

            combined = st.session_state.get("combined_output")
            weights, covariance, correlation, metrics = build_portfolio(
                prices,
                combined,
            )
            st.session_state["weights"] = weights
            st.session_state["covariance"] = covariance
            st.session_state["correlation"] = correlation
            st.session_state["portfolio_metrics"] = metrics
            st.success("Portfolio analysis complete.")
        except Exception as error:
            st.error(str(error))

    if "weights" in st.session_state:
        weights = st.session_state["weights"]
        covariance = st.session_state["covariance"]
        correlation = st.session_state["correlation"]
        metrics = st.session_state["portfolio_metrics"]

        columns = st.columns(3)
        columns[0].metric(
            "Equal-weight volatility",
            f"{metrics['equal_weight_annualized_volatility']:.2%}",
        )
        columns[1].metric(
            "Inverse-volatility volatility",
            f"{metrics['inverse_volatility_annualized_volatility']:.2%}",
        )
        columns[2].metric(
            "Estimated risk reduction",
            f"{metrics['risk_reduction_percent']:.2f}%",
        )

        weight_display = weights.copy()
        for column in [
            "Equal_Weight",
            "Inverse_Volatility_Weight",
            "Optimized_Weight",
        ]:
            weight_display[column] = (
                weight_display[column] * 100
            ).round(2).astype(str) + "%"

        st.markdown("### Suggested allocations")
        st.dataframe(
            weight_display,
            use_container_width=True,
            hide_index=True,
        )
        download_csv(weights, "proposal_portfolio_weights.csv")

        covariance_values = covariance.set_index("Ticker")
        correlation_values = correlation.set_index("Ticker")

        st.markdown("### Covariance")
        st.dataframe(covariance_values, use_container_width=True)

        st.markdown("### Correlation")
        st.dataframe(
            correlation_values.style.background_gradient(cmap="RdYlGn", axis=None),
            use_container_width=True,
        )

        st.markdown("### Optimized allocation")
        allocation_chart = weights.set_index("Ticker")[
            "Optimized_Weight"
        ]
        allocation_frame = allocation_chart.reset_index()
        allocation_frame.columns = ["Ticker", "Optimized_Weight"]
        allocation_frame = allocation_frame.sort_values("Optimized_Weight")
        allocation_frame["Weight_%"] = (
            allocation_frame["Optimized_Weight"] * 100
        ).round(2)
        allocation_colors = [
            "#48E0C2" if action == "Increase" else "#FF5E95"
            for action in weights.set_index("Ticker").loc[
                allocation_frame["Ticker"], "Diversification_Action"
            ]
        ]
        allocation_figure = go.Figure(
            go.Bar(
                x=allocation_frame["Weight_%"],
                y=allocation_frame["Ticker"],
                orientation="h",
                marker={
                    "color": allocation_colors,
                    "line": {"color": "rgba(240,248,255,.42)", "width": 1},
                    "cornerradius": 5,
                },
                text=[f"{weight:.2f}% " for weight in allocation_frame["Weight_%"]],
                textposition="outside",
                customdata=allocation_frame["Optimized_Weight"],
                hovertemplate="<b>%{y}</b><br>Weight: %{customdata:.2%}<extra></extra>",
            )
        )
        allocation_figure.update_layout(
            template="plotly_dark",
            height=max(260, 70 + 44 * len(allocation_frame)),
            margin={"l": 20, "r": 90, "t": 10, "b": 25},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,17,31,.45)",
            bargap=0.42,
            showlegend=False,
            xaxis={
                "title": "Optimized portfolio weight",
                "ticksuffix": "%",
                "range": [0, max(5, allocation_frame["Weight_%"].max() * 1.18)],
                "gridcolor": "rgba(132,173,213,.15)",
            },
            yaxis={"gridcolor": "rgba(132,173,213,.08)", "tickfont": {"size": 13}},
            font={"color": "#EDF4FF"},
        )
        st.plotly_chart(allocation_figure, use_container_width=True)


elif page == "Retrain":
    st.subheader("Retrain and replace proposal models")
    st.warning(
        "Retraining replaces the stored proposal stock and news models. "
        "It should be run explicitly rather than on every prediction."
    )

    retrain_period = st.selectbox(
        "Retraining stock history",
        ["1y", "5y", "10y"],
        index=2,
        key="retrain_period",
    )

    if st.button("Retrain models", type="primary"):
        try:
            if uploaded_prices is not None:
                prices = uploaded_prices
            else:
                prices = cached_live_prices(retrain_period)

            if uploaded_news is not None:
                news = prepare_news(uploaded_news)
            else:
                news = prepare_news(cached_live_news())

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
            st.success("Models retrained and saved.")
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
            st.json(metrics)
        except Exception as error:
            st.error(str(error))


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

    if portfolio_metrics:
        st.markdown("### Latest portfolio metrics")
        st.json(portfolio_metrics)


else:
    st.subheader("About this project")
    st.markdown(
        """
        This project combines three proposed modules:

        1. **Stock Trend Predictor** - Random Forest with technical features,
           including RSI and MACD.
        2. **News Sentiment Signal** - TF-IDF and Logistic Regression.
           News sentiment is merged into the stock feature data.
        3. **Portfolio Risk Analyzer** - covariance, correlation,
           inverse-volatility weights, and signal-adjusted allocations.

        The UI does not implement a trading system. It does not place trades,
        guarantee returns, or provide regulated financial advice.

        News labels may be weakly generated where verified labels are absent.
        Stock accuracy near 50% is close to random direction prediction.
        """
    )

