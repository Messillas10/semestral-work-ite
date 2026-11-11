"""Tests for src/model.py.

The toy data (tests/toy_data.csv) has too few candidates to train a
meaningful model, so the toy tests check properties that must hold for
any data. The quality of the model is checked once on the real dataset.
"""

import os

from src.data import load_data, clean_data, eligible_players
from src.model import predict_season, evaluate, top_k_hits

TESTS_DIR = os.path.dirname(__file__)  # FOLDER OF THIS FILE
TOY_CSV = os.path.join(TESTS_DIR, "toy_data.csv")  # TOY DATASET
REAL_CSV = os.path.join(TESTS_DIR, "..", "data", "player_mvp_stats.csv")  # REAL DATASET


def load(path):
    """Load, clean and filter the CSV."""

    return eligible_players(clean_data(load_data(path)))


def test_predict_season():
    """Only the asked season, ranks 1..n, no negative share."""

    result = predict_season(load(TOY_CSV), 2001)

    assert set(result["Year"]) == {2001}  # NO OTHER SEASON
    assert list(result["Rank"]) == [1, 2, 3]  # 3 ELIGIBLE PLAYERS, IN ORDER
    assert (result["Predicted"] >= 0).all()  # CLIPPED AT 0


def test_predict_season_unknown_season():
    """A season that is not in the data raises ValueError."""

    try:

        predict_season(load(TOY_CSV), 1950)  # NOT IN THE TOY DATA

    except ValueError:

        return  # EXPECTED ERROR -> PASS

    raise AssertionError("ValueError was not raised")  # NO ERROR -> FAIL


def test_no_leakage():
    """Changing the real Share of the predicted season changes nothing."""

    df = load(TOY_CSV)

    before = predict_season(df, 2001)  # NORMAL PREDICTION

    changed = df.copy()  # DO NOT TOUCH df
    changed.loc[changed["Year"] == 2001, "Share"] = 0.5  # FAKE ANSWERS FOR 2001

    after = predict_season(changed, 2001)  # PREDICT AGAIN

    assert list(before["Predicted"]) == list(after["Predicted"])  # SAME -> 2001 NOT USED


def test_real_data_result():
    """Regression test: the result documented in docs/tests.md stays the same."""

    summary, _ = evaluate(load(REAL_CSV))

    assert top_k_hits(summary, 1) == 21  # TOP-1 21/31
    assert top_k_hits(summary, 3) == 29  # TOP-3 29/31
