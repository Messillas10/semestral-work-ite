"""Describe what kind of player wins the MVP.

Two analyses:
- profile: mean stats of MVP winners vs other vote receivers vs all
  eligible players, plus the model coefficients.
- clusters: k-means groups players into archetypes by playing style,
  then counts how many MVPs fall into each archetype.

All functions expect the DataFrame returned by data.eligible_players.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from sklearn.preprocessing import StandardScaler

from src.model import FEATURES, coefficients

STYLE_FEATURES = [  # HOW A PLAYER PLAYS (NO G, MP, TOV, W/L%)
    "PTS", "AST", "TRB", "STL", "BLK",
    "FG%", "3P%", "FT%",
]
RANDOM_STATE = 0  # SAME CLUSTERS IN EVERY RUN


def mark_mvps(df):
    """Add a boolean column MVP: True for the highest Share of each season.

    Args:
        df: eligible players.

    Returns:
        New DataFrame with the extra column MVP.
    """

    df = df.copy()  # DO NOT TOUCH THE INPUT

    best = df.groupby("Year")["Share"].transform("max")  # HIGHEST SHARE OF THE SEASON, ON EVERY ROW

    df["MVP"] = (df["Share"] == best) & (df["Share"] > 0)  # WINNER OF THAT SEASON

    return df


def profile(df):
    """Compare MVP winners with the other candidates and the league.

    Args:
        df: eligible players.

    Returns:
        Tuple (groups, weights):
        - groups: mean of each model feature (rows) for the groups
          MVPs, Other candidates and League (columns), plus a row Players.
        - weights: model coefficients (see model.coefficients).
    """

    df = mark_mvps(df)

    groups = {
        "MVPs": df[df["MVP"]],  # ONE PER SEASON
        "Other candidates": df[(df["Share"] > 0) & ~df["MVP"]],  # VOTES BUT DID NOT WIN
        "League": df,  # ALL ELIGIBLE PLAYERS
    }

    table = pd.DataFrame({name: rows[FEATURES].mean() for name, rows in groups.items()})  # FEATURE x GROUP
    table.loc["Players"] = [len(rows) for rows in groups.values()]  # GROUP SIZE

    return table, coefficients(df)  # WHAT MVPS HAVE + WHAT THE MODEL WEIGHS


def find_clusters(df, k=4):
    """Group players into k archetypes with k-means on the style features.

    The features are standardised first, otherwise PTS (around 10) would
    dominate the percentages (around 0.4).

    Args:
        df: eligible players.
        k: number of clusters (at least 2).

    Returns:
        Tuple (clustered, silhouette):
        - clustered: copy of df with the columns Cluster (0..k-1) and MVP.
        - silhouette: silhouette score (-1..1, higher = better separated).

    Raises:
        ValueError: if k < 2 or k >= number of players.
    """

    if k < 2 or k >= len(df):
        raise ValueError(f"k must be between 2 and {len(df) - 1}")  # SILHOUETTE NEEDS 2..N-1 CLUSTERS

    X = StandardScaler().fit_transform(df[STYLE_FEATURES])  # SAME SCALE FOR EVERY STAT

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)  # 10 STARTS, KEEP THE BEST
    labels = kmeans.fit_predict(X)  # CLUSTER OF EACH PLAYER

    clustered = mark_mvps(df)
    clustered["Cluster"] = labels

    return clustered, silhouette_score(X, labels)


def describe_clusters(clustered):
    """Mean style of each cluster, its size and its number of MVPs.

    Args:
        clustered: first value returned by find_clusters.

    Returns:
        DataFrame with one row per cluster: the mean of each style
        feature and W/L%, Players and MVPs.
    """

    by_cluster = clustered.groupby("Cluster")

    table = by_cluster[STYLE_FEATURES + ["W/L%"]].mean()  # TYPICAL PLAYER OF EACH CLUSTER
    table["Players"] = by_cluster.size()  # HOW MANY PLAYERS
    table["MVPs"] = by_cluster["MVP"].sum()  # TRUE COUNTS AS 1

    return table

