from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_retraining_is_locked_without_streamlit_secret():
    app = AppTest.from_file(
        str(ROOT / "app.py"),
        default_timeout=30,
    ).run()
    app.button(key="nav_Retrain").click().run()

    assert not app.exception
    retrain_button = next(
        button for button in app.button if button.label == "Retrain models"
    )
    assert retrain_button.disabled


def test_offline_navigation_pages_render_without_errors():
    app = AppTest.from_file(
        str(ROOT / "app.py"),
        default_timeout=30,
    ).run()

    for page in ("Portfolio", "Model Status", "About"):
        app.button(key=f"nav_{page}").click().run()
        assert not app.exception
