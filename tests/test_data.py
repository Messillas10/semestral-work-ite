"""Tests for src/data.py.

They use tests/toy_data.csv: 12 hand-written rows (3 seasons x 4 players),
small enough to check every expected value by hand.
"""

import os
import tempfile

from src.data import load_data, clean_data, eligible_players

TOY_CSV = os.path.join(os.path.dirname(__file__), "toy_data.csv")


def test_load_data_reads_all_rows():
    """The toy CSV has 12 rows and 3 seasons."""
    df = load_data(TOY_CSV)
    assert len(df) == 12
    assert sorted(df["Year"].unique()) == [2001, 2002, 2003]


def test_load_data_missing_file():
    """A path that does not exist raises FileNotFoundError."""
    try:
        load_data("no_such_file.csv")
    except FileNotFoundError:
        return
    raise AssertionError("FileNotFoundError was not raised")


def test_load_data_missing_column():
    """A CSV without the Share column raises ValueError naming it."""
    df = load_data(TOY_CSV).drop(columns="Share")
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "bad.csv")
        df.to_csv(path, index=False)
        try:
            load_data(path)
        except ValueError as error:
            assert "Share" in str(error)
            return
    raise AssertionError("ValueError was not raised")


def test_clean_data_fills_empty_percentages():
    """The 4 empty 3P% values become 0 and no percentage is left empty."""
    df = load_data(TOY_CSV)
    assert df["3P%"].isna().sum() == 4
    clean = clean_data(df)
    assert clean[["FG%", "3P%", "FT%"]].isna().sum().sum() == 0
    alpha_2001 = clean[(clean["Player"] == "Alpha Center") & (clean["Year"] == 2001)]
    assert alpha_2001["3P%"].iloc[0] == 0


def test_clean_data_does_not_change_input():
    """clean_data works on a copy: the original DataFrame keeps its empty values."""
    df = load_data(TOY_CSV)
    clean_data(df)
    assert df["3P%"].isna().sum() == 4


def test_eligible_players_removes_short_seasons():
    """Players under half of their team's games are dropped (3 of the 12 rows)."""
    eligible = eligible_players(load_data(TOY_CSV))
    assert len(eligible) == 9
    removed = [("Delta Bench", 2001), ("Zeta Rookie", 2002), ("Delta Bench", 2003)]
    for player, year in removed:
        rows = eligible[(eligible["Player"] == player) & (eligible["Year"] == year)]
        assert len(rows) == 0


def test_eligible_players_limit_is_inclusive():
    """Exactly half of the games is enough (41 of 82)."""
    eligible = eligible_players(load_data(TOY_CSV))
    assert "Epsilon Forward" in eligible["Player"].values


def test_eligible_players_short_season():
    """In a 50-game season, 30 games is enough and 10 is not."""
    eligible = eligible_players(load_data(TOY_CSV))
    season_2003 = eligible[eligible["Year"] == 2003]
    assert "Gamma Wing" in season_2003["Player"].values
    assert "Delta Bench" not in season_2003["Player"].values
