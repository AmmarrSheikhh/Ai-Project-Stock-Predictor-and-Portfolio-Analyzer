"""Backward-compatible import and CLI entry point for :mod:`data.pipeline`."""

from __future__ import annotations

import sys

from data import pipeline as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
