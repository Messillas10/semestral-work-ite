"""Tests for src/plots.py.

A picture cannot be compared with an assert: the tests only check that a
Figure is returned with the expected number of bars. The look was
checked by eye.
"""

import os

from matplotlib.figure import Figure

from src.data import load_data, clean_data, eligible_players
from src.model import predict_season, evaluate
from src import plots

TESTS_DIR = os.path.dirname(__file__)
TOY_CSV = os.path.join(TESTS_DIR, "toy_data.csv")


def load(path):
    """Load, clean and filter the CSV."""

    return eligible_players(clean_data(load_data(path)))


def test_season_chart():
    """Two bars (predicted and real) per player shown."""

    fig = plots.season_chart(predict_season(load(TOY_CSV), 2001), top=2)

    assert isinstance(fig, Figure)

    assert len(fig.axes[0].patches) == 4  # 2 PLAYERS x 2 BARS


def test_accuracy_chart():
    """One bar per season."""

    summary, _ = evaluate(load(TOY_CSV))

    fig = plots.accuracy_chart(summary)

    assert len(fig.axes[0].patches) == 3  # 3 SEASONS
