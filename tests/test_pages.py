from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from data.pipeline import InputError
from pages.news_review import render


def test_news_review_renders_uploaded_news_without_fetching_live_data():
    news = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "Headline": ["Company reports steady outlook"],
            "Manual_Label": [float("nan")],
            "Weak_Label": [0],
        }
    )
    model = SimpleNamespace(predict=lambda headlines: [0])
    streamlit = Mock()
    streamlit.column_config = SimpleNamespace(SelectboxColumn=Mock())
    streamlit.data_editor.side_effect = lambda editor, **kwargs: editor
    streamlit.button.return_value = False

    render(
        streamlit,
        pd,
        ["AAA"],
        news,
        Mock(side_effect=AssertionError("unexpected live-news request")),
        lambda value: value,
        lambda value: value,
        lambda: (None, None, model),
        InputError,
        Mock(),
    )

    editor = streamlit.data_editor.call_args.args[0]
    assert editor.loc[0, "Model sentiment"] == "Neutral"
    assert editor.loc[0, "Your label"] == "Use model"
