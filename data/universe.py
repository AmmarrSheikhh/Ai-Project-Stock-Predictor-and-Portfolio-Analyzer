import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TICKER_CONFIG_PATH = ROOT / "ticker_universe.json"


def load_default_tickers() -> list[str]:
    configuration = json.loads(
        TICKER_CONFIG_PATH.read_text(encoding="utf-8")
    )
    tickers = configuration.get("default_tickers")
    if (
        not isinstance(tickers, list)
        or not tickers
        or any(not isinstance(ticker, str) or not ticker.strip() for ticker in tickers)
    ):
        raise ValueError(
            f"{TICKER_CONFIG_PATH} must contain a non-empty default_tickers list."
        )
    return list(dict.fromkeys(ticker.strip().upper() for ticker in tickers))
