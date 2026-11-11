"""Tests for src/data.py.

They use tests/toy_data.csv: 12 hand-written rows (3 seasons x 4 players),
small enough to check every expected value by hand.
"""

import os

from src.data import load_data, clean_data, eligible_players

TOY_CSV = os.path.join(os.path.dirname(__file__), "toy_data.csv")


def test_load_data():
    """The toy CSV has 12 rows and 3 seasons."""

    df = load_data(TOY_CSV)

    assert len(df) == 12  # NO ROW LOST
    assert sorted(df["Year"].unique()) == [2001, 2002, 2003]  # ALL SEASONS


def test_load_data_missing_file():
    """A path that does not exist raises FileNotFoundError."""

    try:

        load_data("no_such_file.csv")  # MUST FAIL

    except FileNotFoundError:

        return  # EXPECTED ERROR THEN PASS

    raise AssertionError("FileNotFoundError was not raised")  # NO ERROR THEN FAIL


def test_clean_data():
    """Empty percentages become 0."""

    clean = clean_data(load_data(TOY_CSV))

    assert clean[["FG%", "3P%", "FT%"]].isna().sum().sum() == 0  # NO EMPTY LEFT


def test_eligible_players():
    """Players under half of their team's games are dropped; exactly half is enough."""

    eligible = eligible_players(load_data(TOY_CSV))

    assert len(eligible) == 9  # 12 - 3 (20/82, 40/82, 10/50)
    assert "Epsilon Forward" in eligible["Player"].values  # 41 OF 82 -> KEPT
