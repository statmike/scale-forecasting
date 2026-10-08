"""NeuralProphet's console output is the forecast, not the library's chatter.

Importing ``neuralprophet`` turns on ``logging.captureWarnings`` process-wide with its own stderr
handler, and its training loop trips hundreds of pandas ``FutureWarning``s and Lightning
``PossibleUserWarning``s per fit. Each would reach the terminal — or a committed notebook output —
prefixed with the absolute path of the site-packages file that raised it, which on a workstation is
a home directory and a username (see ``test_notebook_hygiene.py``). The wrapper therefore runs the
library under a warning filter; this pins that it keeps doing so on every public path.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("neuralprophet")

from scale_forecasting.models.base_model import ModelContext
from scale_forecasting.models.neuralprophet_model import NeuralProphetModel

_SILENCED = (UserWarning, FutureWarning, DeprecationWarning)


def _series(n: int = 90) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(50.0 + 5.0 * np.sin(np.arange(n) * 2 * np.pi / 7), index=idx, name="y")


def _escaped(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        f"{w.category.__name__}: {str(w.message)[:80]}"
        for w in caught
        if issubclass(w.category, _SILENCED)
    ]


def test_local_fit_and_predict_let_no_library_warning_escape() -> None:
    model = NeuralProphetModel({"epochs": 2}, ModelContext(freq="D", horizon=7, seed=7))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(_series())
        frame = model.predict(7)
    assert len(frame) == 7
    assert not _escaped(caught), _escaped(caught)


def test_panel_fit_and_predict_let_no_library_warning_escape() -> None:
    model = NeuralProphetModel(
        {"epochs": 2, "training_mode": "global"}, ModelContext(freq="D", horizon=7, seed=7)
    )
    series_map = {"a": (_series(), None), "b": (_series() * 1.5, None)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_panel(series_map)
        out = model.predict_panel(7)
    assert set(out) == {"a", "b"} and all(len(f) == 7 for f in out.values())
    assert not _escaped(caught), _escaped(caught)
