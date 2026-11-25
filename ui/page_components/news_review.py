"""News-review content rendered within the main W Trade app."""

def render(
    streamlit,
    pandas,
    selected_tickers,
    uploaded_news,
    cached_live_news,
    prepare_news,
    apply_manual_news_labels,
    load_models,
    input_error,
    update_manual_news_labels,
) -> None:
    streamlit.subheader("Review and label financial news")
    streamlit.caption(
        "Your labels are saved locally, override model sentiment immediately, "
        "and are used as training labels the next time you retrain."
    )

    try:
        raw_news = (
            uploaded_news
            if uploaded_news is not None
            else cached_live_news(tuple(selected_tickers))
        )
        reviewed = apply_manual_news_labels(prepare_news(raw_news))
        if selected_tickers:
            reviewed = reviewed[
                reviewed["Ticker"].isin(selected_tickers)
            ]
        else:
            streamlit.warning(
                "Select tickers in the sidebar basket to review their news."
            )
            streamlit.stop()
        reviewed = reviewed.drop_duplicates(
            ["Ticker", "Headline"], keep="last"
        ).reset_index(drop=True)
        if reviewed.empty:
            streamlit.info("No headlines were found for the selected basket.")
            streamlit.stop()

        try:
            _, _, news_model = load_models()
            predicted_labels = news_model.predict(reviewed["Headline"])
        except input_error:
            predicted_labels = reviewed["Weak_Label"].to_numpy()
            streamlit.info(
                "No trained news model is available. Current labels use "
                "VADER sentiment until you retrain."
            )

        sentiment_names = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        editor = reviewed[["Ticker", "Headline"]].copy()
        editor["Model sentiment"] = [
            sentiment_names[int(label)] for label in predicted_labels
        ]
        editor["Your label"] = reviewed["Manual_Label"].map(
            lambda value: (
                sentiment_names[int(value)]
                if pandas.notna(value)
                else "Use model"
            )
        )
        streamlit.write(f"Headlines to review: {len(editor)}")
        edited = streamlit.data_editor(
            editor,
            use_container_width=True,
            hide_index=True,
            disabled=["Ticker", "Headline", "Model sentiment"],
            column_config={
                "Your label": streamlit.column_config.SelectboxColumn(
                    "Your label",
                    options=["Use model", "Positive", "Neutral", "Negative"],
                    required=True,
                )
            },
            key="news_label_editor",
        )
        if streamlit.button("Save news labels", type="primary"):
            reviewed_labels = edited[["Ticker", "Headline", "Your label"]].copy()
            reviewed_labels["Label"] = reviewed_labels["Your label"].map({
                "Negative": -1,
                "Neutral": 0,
                "Positive": 1,
            })
            update_manual_news_labels(
                reviewed_labels[["Ticker", "Headline", "Label"]]
            )
            streamlit.success("News labels saved and will be used immediately.")
            streamlit.rerun()
    except input_error as error:
        streamlit.error(str(error))
