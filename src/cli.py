"""Command-line interface: read the arguments, run the analysis, print it.

This is the only module that prints or saves files; the other modules
return data, so the Streamlit app can reuse them.

Commands (see docs/design.md):
    predict SEASON [--top N]   MVP race of one past season
    evaluate                   predict every season, accuracy
    profile                    MVPs vs other candidates vs league
    clusters [--k K]           player archetypes with k-means
Common options: --data PATH, --output DIR.
"""

import argparse
import os
import sys

from src.data import load_data, clean_data, eligible_players
from src.model import predict_season, evaluate, top_k_hits
from src.profiles import profile, find_clusters, describe_clusters
from src import plots

DEFAULT_DATA = os.path.join("data", "player_mvp_stats.csv")
DEFAULT_OUTPUT = "output"


def build_parser():
    """Define the commands and options.

    Returns:
        argparse.ArgumentParser.
    """

    common = argparse.ArgumentParser(add_help=False)  # OPTIONS SHARED BY EVERY COMMAND
    common.add_argument("--data", default=DEFAULT_DATA, help="CSV file (default: %(default)s)")
    common.add_argument("--output", default=DEFAULT_OUTPUT, help="folder for the charts (default: %(default)s)")

    parser = argparse.ArgumentParser(prog="main.py", description="NBA MVP predictor (seasons 1991-2021).")
    commands = parser.add_subparsers(dest="command", required=True)  # ONE COMMAND IS MANDATORY

    predict = commands.add_parser("predict", parents=[common], help="MVP race of one past season")
    predict.add_argument("season", type=int, help="season end year, e.g. 2016 = 2015-16")
    predict.add_argument("--top", type=int, default=5, help="players shown (default: %(default)s)")

    commands.add_parser("evaluate", parents=[common], help="predict every season and measure accuracy")
    commands.add_parser("profile", parents=[common], help="MVPs vs other candidates vs league")

    clusters = commands.add_parser("clusters", parents=[common], help="player archetypes (k-means)")
    clusters.add_argument("--k", type=int, default=4, help="number of clusters (default: %(default)s)")

    return parser


def save(fig, folder, name):
    """Save a Figure as PNG in folder and return the path."""

    os.makedirs(folder, exist_ok=True)  # CREATE output/ IF NEEDED
    path = os.path.join(folder, name)
    fig.savefig(path, dpi=120)

    return path


def run_predict(df, args):
    """Print the top players of one season and save the chart."""

    if args.top < 1:
        raise ValueError("--top must be at least 1")

    result = predict_season(df, args.season)
    top = result.head(args.top)

    print(f"Season {args.season}: model trained on the other {df['Year'].nunique() - 1} seasons\n")
    print(f"{'Rank':<6}{'Player':<26}{'Predicted share':>16}{'Real share':>12}")

    for _, row in top.iterrows():
        print(f"{row['Rank']:<6}{row['Player']:<26}{row['Predicted']:>16.3f}{row['Share']:>12.3f}")

    predicted = result.iloc[0]["Player"]
    real = result.loc[result["Share"].idxmax(), "Player"]  # HIGHEST REAL SHARE
    verdict = "correct" if predicted == real else "wrong"

    print(f"\nPredicted MVP: {predicted}  | Real MVP: {real}  -> {verdict}")
    print("Note: predicted shares are compressed (max around 0.6); the ranking is what matters.")

    print(f"Chart: {save(plots.season_chart(result, args.top), args.output, f'season_{args.season}.png')}")


def run_evaluate(df, args):
    """Print the result of every season and the accuracy, save the chart."""

    summary, _ = evaluate(df)
    seasons = len(summary)

    print(f"{'Year':<6}{'Predicted MVP':<26}{'Real MVP':<26}{'Real MVP rank':>13}")

    for _, row in summary.iterrows():
        mark = "" if row["Correct"] else "  x"
        print(f"{row['Year']:<6}{row['Predicted MVP']:<26}{row['Real MVP']:<26}{row['MVP rank']:>13}{mark}")

    print(f"\nTop-1 accuracy: {top_k_hits(summary, 1)}/{seasons}")
    print(f"Top-3 accuracy: {top_k_hits(summary, 3)}/{seasons}")

    print(f"Chart: {save(plots.accuracy_chart(summary), args.output, 'accuracy.png')}")


def run_profile(df, args):
    """Print the group means and the model coefficients, save the charts."""

    groups, weights = profile(df)

    print("Mean stats per group\n")
    print(groups.round(3).to_string())

    print("\nModel coefficients (standardised, biggest effect first)\n")
    print(weights.round(3).to_string())

    print(f"\nChart: {save(plots.profile_chart(groups), args.output, 'profile.png')}")
    print(f"Chart: {save(plots.coefficients_chart(weights), args.output, 'coefficients.png')}")


def run_clusters(df, args):
    """Print the archetypes and their number of MVPs, save the chart."""

    clustered, silhouette = find_clusters(df, args.k)

    print(f"k-means with k = {args.k}, silhouette = {silhouette:.3f} (low = groups overlap)\n")
    print(describe_clusters(clustered).round(2).to_string())

    print(f"\nChart: {save(plots.clusters_chart(clustered), args.output, f'clusters_k{args.k}.png')}")


COMMANDS = {  # COMMAND NAME -> FUNCTION
    "predict": run_predict,
    "evaluate": run_evaluate,
    "profile": run_profile,
    "clusters": run_clusters,
}


def main(argv=None):
    """Parse the arguments and run the command.

    Args:
        argv: list of arguments (None = the real command line).

    Returns:
        Exit code: 0 = ok, 1 = error (message printed to stderr).
    """

    args = build_parser().parse_args(argv)

    try:
        df = eligible_players(clean_data(load_data(args.data)))
        COMMANDS[args.command](df, args)

    except (FileNotFoundError, ValueError) as error:  # EXPECTED ERRORS: SHORT MESSAGE, NO TRACEBACK
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0
