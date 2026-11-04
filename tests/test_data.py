"""Tests for src/data.py.

They use tests/toy_data.csv: 12 hand-written rows (3 seasons x 4 players),
small enough to check every expected value by hand.
"""

import os
import tempfile  # FOLDER THAT DELETES ITSELF

from src.data import load_data, clean_data, eligible_players 

TOY_CSV = os.path.join(os.path.dirname(__file__), "toy_data.csv")


def test_load_data_reads_all_rows():
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


def test_load_data_missing_column():
    """A CSV without the Share column raises ValueError naming it."""

    df = load_data(TOY_CSV).drop(columns="Share")  # BAD DATA ON PURPOSE

    with tempfile.TemporaryDirectory() as folder:  # DELETED AT THE END

        path = os.path.join(folder, "bad.csv")
        df.to_csv(path, index=False)  # WRITE THE BAD CSV

        try:

            load_data(path)  # MUST FAIL

        except ValueError as error:

            assert "Share" in str(error)  # MESSAGE NAMES THE COLUMN
            return  # EXPECTED ERROR THEN PASS
        
    raise AssertionError("ValueError was not raised")  # NO ERROR THEN FAIL


def test_clean_data_fills_empty_percentages():
    """The 4 empty 3P% values become 0 and no percentage is left empty."""

    df = load_data(TOY_CSV)

    assert df["3P%"].isna().sum() == 4  # PRE-CHECK: THE TOY CSV HAS THEM

    clean = clean_data(df)
    assert clean[["FG%", "3P%", "FT%"]].isna().sum().sum() == 0  # NO EMPTY LEFT

    alpha_2001 = clean[(clean["Player"] == "Alpha Center") & (clean["Year"] == 2001)]  # ONE ROW
    assert alpha_2001["3P%"].iloc[0] == 0  # EMPTY -> 0


def test_clean_data_does_not_change_input():
    """clean_data works on a copy: the original DataFrame keeps its empty values."""

    df = load_data(TOY_CSV)

    clean_data(df)  # RESULT IGNORED ON PURPOSE

    assert df["3P%"].isna().sum() == 4  # ORIGINAL UNTOUCHED


def test_eligible_players_removes_short_seasons():
    """Players under half of their team's games are dropped (3 of the 12 rows)."""

    eligible = eligible_players(load_data(TOY_CSV))
    assert len(eligible) == 9  # 12 - 3

    removed = [("Delta Bench", 2001), ("Zeta Rookie", 2002), ("Delta Bench", 2003)]  # 20/82, 40/82, 10/50

    for player, year in removed:

        rows = eligible[(eligible["Player"] == player) & (eligible["Year"] == year)]
        assert len(rows) == 0  # REALLY GONE


def test_eligible_players_limit_is_inclusive():
    """Exactly half of the games is enough (41 of 82)."""

    eligible = eligible_players(load_data(TOY_CSV))
    assert "Epsilon Forward" in eligible["Player"].values  # 41 >= 41 -> KEPT


def test_eligible_players_short_season():
    """In a 50-game season, 30 games is enough and 10 is not."""

    eligible = eligible_players(load_data(TOY_CSV))
    season_2003 = eligible[eligible["Year"] == 2003]  # 50-GAME SEASON
    
    assert "Gamma Wing" in season_2003["Player"].values  # 30 >= 25 -> KEPT
    assert "Delta Bench" not in season_2003["Player"].values  # 10 < 25 -> DROPPED
