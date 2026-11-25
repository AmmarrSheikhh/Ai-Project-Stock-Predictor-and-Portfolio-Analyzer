def render_risk_summary(streamlit, metrics: dict) -> None:
    columns = streamlit.columns(5)
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
    columns[4].metric(
        "Signal-tilted volatility",
        f"{metrics['signal_tilted_annualized_volatility']:.2%}",
    )
    streamlit.caption(metrics["recommended_action"])
    if metrics["minimum_observations_warning"]:
        streamlit.warning(metrics["minimum_observations_warning"])
