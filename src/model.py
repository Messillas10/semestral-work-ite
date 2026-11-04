"""Predict the MVP vote share with a linear regression.

Validation is leave-one-season-out: to predict a season, the model is
trained on all the other seasons, so the predicted season is never seen
during training.

The model is trained only on players who received MVP votes (Share > 0):
it learns what separates the winner from the other candidates. Predictions
are made for every eligible player of the season.

All functions expect the DataFrame returned by data.eligible_players
(cleaned, eligible players only).
"""

import pandas as pd  # DATAFRAMES
from sklearn.linear_model import LinearRegression  # THE MODEL
from sklearn.pipeline import make_pipeline  # CHAIN SCALER + MODEL
from sklearn.preprocessing import StandardScaler  # MEAN 0, STD 1

FEATURES = [  # INPUT COLUMNS OF THE MODEL
    "G", "MP", "PTS", "AST", "TRB", "STL", "BLK", "TOV",  # PER-GAME STATS
    "FG%", "3P%", "FT%",  # SHOOTING EFFICIENCY
    "W/L%",  # TEAM SUCCESS
]


def train_model(df):
    """Train the model on the vote receivers of df.

    The features are standardised (mean 0, std 1) so that the
    coefficients can be compared with each other.

    Args:
        df: training seasons.

    Returns:
        Fitted scikit-learn Pipeline (StandardScaler + LinearRegression).
    """

    candidates = df[df["Share"] > 0]  # ONLY PLAYERS WITH VOTES

    model = make_pipeline(StandardScaler(), LinearRegression())  # FIRST SCALE THEN LINEAR REGRESSION
    model.fit(candidates[FEATURES], candidates["Share"])  # X = STATS, Y = SHARE

    return model  # FITTED PIPELINE


def predict_season(df, season):
    """Rank the players of one season, training on all the other seasons.

    Args:
        df: all seasons.
        season: season to predict (end year, e.g. 2016 = 2015-16).

    Returns:
        DataFrame sorted by predicted share, with the columns Rank, Player,
        Pos, Tm, Year, Predicted and Share (real). Predicted is clipped
        at 0 because a negative share has no meaning.

    Raises:
        ValueError: if the season is not in df.
    """

    if season not in df["Year"].values:  # UNKNOWN SEASON

        first, last = df["Year"].min(), df["Year"].max()  # RANGE FOR THE MESSAGE

        raise ValueError(f"Season {season} not in data ({first}-{last})")  # CLEAR ERROR

    model = train_model(df[df["Year"] != season])  # TRAIN ON THE OTHER SEASONS

    test = df[df["Year"] == season]  # ALL ELIGIBLE PLAYERS OF THE SEASON

    result = test[["Player", "Pos", "Tm", "Year", "Share"]].copy()  # COLUMNS TO SHOW

    result["Predicted"] = model.predict(test[FEATURES])  # PREDICTED SHARE

    result = result.sort_values("Predicted", ascending=False)  # BEST FIRST

    result.insert(0, "Rank", range(1, len(result) + 1))  # RANK 1..N AS FIRST COLUMN

    result["Predicted"] = result["Predicted"].clip(lower=0)  # AFTER RANKING: NEGATIVE -> 0

    return result.reset_index(drop=True)  # INDEX 0..N-1


def evaluate(df):
    """Predict every season and compare with the real MVP.

    Args:
        df: all seasons.

    Returns:
        Tuple (summary, predictions):
        - summary: one row per season with Year, Predicted MVP, Real MVP,
        MVP rank (rank the model gave to the real MVP) and Correct.
        - predictions: predict_season of every season, joined.
    """

    seasons = sorted(df["Year"].unique())  # 1991..2021
    predictions = pd.concat(  # ONE TABLE WITH EVERY SEASON
        [predict_season(df, season) for season in seasons], ignore_index=True  # LEAVE-ONE-SEASON-OUT
    )

    rows = []  # ONE DICT PER SEASON
    for season, group in predictions.groupby("Year"):  # GROUP = PLAYERS OF ONE SEASON

        real_mvp = group.loc[group["Share"].idxmax()]  # HIGHEST REAL SHARE = MVP

        rows.append({
            "Year": season,
            "Predicted MVP": group.iloc[0]["Player"],  # RANK 1
            "Real MVP": real_mvp["Player"],  # WHO WON
            "MVP rank": real_mvp["Rank"],  # WHERE THE MODEL PUT HIM
        })

    summary = pd.DataFrame(rows)  # LIST OF DICTS -> TABLE
    summary["Correct"] = summary["MVP rank"] == 1  # TRUE IF THE MODEL GOT IT
    
    return summary, predictions  # PREDICTIONS ARE REUSED BY THE APP


def top_k_hits(summary, k):
    """Count the seasons where the real MVP is among the top k predictions.

    Args:
        summary: first value returned by evaluate.
        k: 1 = predicted MVP is the real one, 3 = real MVP in the top 3.

    Returns:
        Number of seasons (int).
    """

    if k <= 0:
        raise ValueError("k must be a positive integer") # K MUST BE POSITIVE
    return int((summary["MVP rank"] <= k).sum())  # EACH TRUE COUNTS AS 1


def coefficients(df):
    """Weight of each statistic in a model trained on all seasons.

    The features are standardised, so a bigger absolute value means a
    bigger effect on the predicted share.

    Args:
        df: all seasons.

    Returns:
        Series indexed by feature, sorted by absolute value (largest first).
    """

    model = train_model(df)  # ALL SEASONS: NOTHING TO PREDICT HERE

    weights = pd.Series(model[-1].coef_, index=FEATURES)  # [-1] = LINEAR REGRESSION STEP
    
    return weights.sort_values(key=abs, ascending=False)  # BIGGEST EFFECT FIRST, SIGN IGNORED
