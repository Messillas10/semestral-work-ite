"""Tests for src/profiles.py.

Toy data after the eligibility filter: 9 players in 3 seasons.
MVPs: Alpha Center 2001, Beta Guard 2002, Alpha Center 2003.
"""

import os

from src.data import load_data, clean_data, eligible_players
from src.profiles import mark_mvps, profile, find_clusters, describe_clusters

TESTS_DIR = os.path.dirname(__file__)
TOY_CSV = os.path.join(TESTS_DIR, "toy_data.csv")
REAL_CSV = os.path.join(TESTS_DIR, "..", "data", "player_mvp_stats.csv")


def load(path):
    """Load, clean and filter the CSV."""

    return eligible_players(clean_data(load_data(path)))


def test_mark_mvps():
    """One MVP per season: the player with the highest Share."""

    df = mark_mvps(load(TOY_CSV))

    mvps = df[df["MVP"]].sort_values("Year")

    assert list(mvps["Player"]) == ["Alpha Center", "Beta Guard", "Alpha Center"]


def test_profile():
    """Group sizes and mean points calculated by hand."""

    groups, _ = profile(load(TOY_CSV))

    assert list(groups.loc["Players"]) == [3, 3, 9]  # MVPS, OTHER CANDIDATES, LEAGUE

    assert groups.loc["PTS", "MVPs"] == 29.0  # (30 + 28 + 29) / 3


def test_find_clusters():
    """k clusters, and the same result if run twice (fixed random_state)."""

    df = load(TOY_CSV)

    first, _ = find_clusters(df, k=2)
    second, _ = find_clusters(df, k=2)

    assert set(first["Cluster"]) == {0, 1}

    assert list(first["Cluster"]) == list(second["Cluster"])


def test_real_data_clusters():
    """Regression test: with k = 4 the 31 MVPs fall in two clusters (11 + 20)."""

    clustered, _ = find_clusters(load(REAL_CSV), k=4)

    assert sorted(describe_clusters(clustered)["MVPs"]) == [0, 0, 11, 20]
