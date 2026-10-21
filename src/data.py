"""Load and prepare the NBA MVP dataset.

One row of the CSV = one player in one season.
"""

import pandas as pd

REQUIRED_COLUMNS = [
    "Player", "Pos", "Year", "Tm",
    "G", "MP", "PTS", "AST", "TRB", "STL", "BLK", "TOV", "FG%", "3P%", "FT%",
    "W", "L", "W/L%",
    "Share",
]
PERCENT_COLUMNS = ["FG%", "3P%", "FT%"]


def load_data(path):
    """Read the CSV file and check that it has every required column.

    Args:
        path: location of the CSV file.

    Returns:
        DataFrame with one row per player and season.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if a required column is missing.
    """
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return df


def clean_data(df):
    """Replace empty shooting percentages with 0.

    A player with no attempts (e.g. never shot a three) has an empty
    percentage in the CSV.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        New DataFrame without empty values in PERCENT_COLUMNS.
    """
    df = df.copy()
    df[PERCENT_COLUMNS] = df[PERCENT_COLUMNS].fillna(0)
    return df


def eligible_players(df):
    """Keep only players who played at least half of their team's games.

    The limit is relative to the season length, so it also works for the
    shortened seasons (1999: 50 games, 2012: 66 games).

    Args:
        df: DataFrame with the columns G, W and L.

    Returns:
        DataFrame with the eligible rows only.
    """
    min_games = (df["W"] + df["L"]) / 2  # TEAM GAMES / 2
    return df[df["G"] >= min_games]
