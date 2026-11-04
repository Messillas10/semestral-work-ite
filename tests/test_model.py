"""Tests for src/model.py.

The toy data (tests/toy_data.csv) has too few candidates to train a
meaningful model, so the toy tests check properties that must hold for
any data: output format, no leakage, training on candidates only.
The quality of the model is checked once on the real dataset.
"""

import os

import pandas as pd

from src.data import load_data, clean_data, eligible_players
from src.model import FEATURES, predict_season, evaluate, top_k_hits, coefficients

TESTS_DIR = os.path.dirname(__file__)
TOY_CSV = os.path.join(TESTS_DIR, "toy_data.csv")
REAL_CSV = os.path.join(TESTS_DIR, "..", "data", "player_mvp_stats.csv")


def load(path):
    """Load, clean and filter a CSV, like the program does."""
    return eligible_players(clean_data(load_data(path)))


def test_predict_season_format():
    """Only the asked season, ranks 1..n, sorted, no negative share."""
    result = predict_season(load(TOY_CSV), 2001)
    assert len(result) == 3  # 3 ELIGIBLE PLAYERS IN 2001
    assert set(result["Year"]) == {2001}
    assert list(result["Rank"]) == [1, 2, 3]
    assert result["Predicted"].is_monotonic_decreasing
    assert (result["Predicted"] >= 0).all()


def test_predict_season_unknown_season():
    """A season that is not in the data raises ValueError."""
    try:
        predict_season(load(TOY_CSV), 1950)
    except ValueError as error:
        assert "1950" in str(error)
        return
    raise AssertionError("ValueError was not raised")


def test_predict_season_no_leakage():
    """Changing the real Share of the predicted season changes nothing."""
    df = load(TOY_CSV)
    before = predict_season(df, 2001)
    changed = df.copy()
    changed.loc[changed["Year"] == 2001, "Share"] = 0.5
    after = predict_season(changed, 2001)
    assert list(before["Player"]) == list(after["Player"])
    assert list(before["Predicted"]) == list(after["Predicted"])


def test_model_trains_on_candidates_only():
    """A player without votes in a training season does not affect the model."""
    df = load(TOY_CSV)
    before = predict_season(df, 2001)
    changed = df.copy()
    gamma_2003 = (changed["Player"] == "Gamma Wing") & (changed["Year"] == 2003)
    changed.loc[gamma_2003, "PTS"] = 99.0  # SHARE 0 -> NOT A CANDIDATE
    after = predict_season(changed, 2001)
    assert list(before["Predicted"]) == list(after["Predicted"])


def test_evaluate_format():
    """One summary row per season and every eligible player predicted once."""
    df = load(TOY_CSV)
    summary, predictions = evaluate(df)
    assert list(summary["Year"]) == [2001, 2002, 2003]
    assert len(predictions) == len(df)
    assert (summary["Correct"] == (summary["MVP rank"] == 1)).all()
    assert list(summary["Real MVP"]) == ["Alpha Center", "Beta Guard", "Alpha Center"]


def test_top_k_hits():
    """Counts the seasons with the real MVP at rank <= k."""
    summary = pd.DataFrame({"MVP rank": [1, 2, 5]})
    assert top_k_hits(summary, 1) == 1
    assert top_k_hits(summary, 3) == 2
    assert top_k_hits(summary, 10) == 3


def test_coefficients_format():
    """One weight per feature, largest absolute value first."""
    weights = coefficients(load(TOY_CSV))
    assert sorted(weights.index) == sorted(FEATURES)
    assert weights.abs().is_monotonic_decreasing


def test_real_data_documented_result():
    """Regression test: the result documented in the README stays the same."""
    df = load(REAL_CSV)
    summary, _ = evaluate(df)
    assert top_k_hits(summary, 1) == 21
    assert top_k_hits(summary, 3) == 29
    assert predict_season(df, 2016)["Player"].iloc[0] == "Stephen Curry"
    weights = coefficients(df)
    assert list(weights.index[:2]) == ["PTS", "W/L%"]  # SCORING AND TEAM WINS
