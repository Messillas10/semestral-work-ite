"""Charts of the project.

Every function returns a matplotlib Figure and never saves or shows it:
the command line saves it as PNG (fig.savefig) and the Streamlit app
draws it (st.pyplot). Both interfaces get the same charts.

Figure is used directly instead of pyplot, so no global state is kept
between charts and no window is opened.
"""

from matplotlib.figure import Figure  # ONE INDEPENDENT FIGURE PER CHART
from matplotlib.patches import Patch  # COLOUR SQUARES FOR A LEGEND

# COMMON STYLE: SAME COLOURS IN THE PNGS AND IN THE APP
BLUE = "#2a78d6"  # MAIN SERIES (PREDICTED, CORRECT, POSITIVE)
ORANGE = "#eb6834"  # SECOND SERIES (REAL, MISSED, NEGATIVE)
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]  # FIXED ORDER, ONE PER CLUSTER
INK = "#52514e"  # AXES AND TEXT
GRID = "#e4e3df"  # BACKGROUND GRID


def _new_chart(title, width=8, height=5):
    """Create a figure with one axes in the common style."""

    fig = Figure(figsize=(width, height), layout="constrained")
    ax = fig.add_subplot()
    _style(ax, title)

    return fig, ax


def _style(ax, title):
    """Light grid, no top/right border, grey text."""

    ax.set_title(title, loc="left", color="black")
    ax.spines[["top", "right"]].set_visible(False)  # ONLY THE AXES THAT MATTER
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.tick_params(colors=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)  # GRID BEHIND THE BARS


def season_chart(prediction, top=5):
    """Predicted vs real share of the top players of one season.

    Args:
        prediction: DataFrame returned by model.predict_season.
        top: number of players shown.

    Returns:
        Figure with horizontal bars, best player at the top.
    """

    rows = prediction.head(top).iloc[::-1]  # REVERSED: BARH DRAWS FROM THE BOTTOM
    season = int(prediction["Year"].iloc[0])

    fig, ax = _new_chart(f"Season {season}: predicted vs real MVP share", height=1 + 0.6 * len(rows))

    y = range(len(rows))  # ONE POSITION PER PLAYER
    ax.barh([i + 0.2 for i in y], rows["Predicted"], height=0.38, color=BLUE, label="Predicted")
    ax.barh([i - 0.2 for i in y], rows["Share"], height=0.38, color=ORANGE, label="Real")

    ax.set_yticks(list(y), rows["Player"])
    ax.set_xlabel("MVP vote share")
    ax.set_xlim(0, 1)  # SHARE IS ALWAYS 0..1
    ax.grid(axis="y", visible=False)  # ONLY VERTICAL GRID LINES
    ax.legend(frameon=False, loc="lower right")

    return fig


def accuracy_chart(summary):
    """Rank the model gave to the real MVP in every season.

    Args:
        summary: first value returned by model.evaluate.

    Returns:
        Figure with one bar per season; rank 1 = correct prediction.
    """

    fig, ax = _new_chart("Rank of the real MVP in the model's prediction", width=10, height=4)

    colors = [BLUE if correct else ORANGE for correct in summary["Correct"]]  # HIT / MISS
    ax.bar(summary["Year"], summary["MVP rank"], color=colors, width=0.7)

    legend = [Patch(color=BLUE, label="Correct (rank 1)"), Patch(color=ORANGE, label="Missed")]
    ax.legend(handles=legend, frameon=False, loc="upper left")  # ONE BAR CALL -> LEGEND BY HAND

    ax.set_xlabel("Season (end year)")
    ax.set_ylabel("Rank of the real MVP")
    ax.yaxis.get_major_locator().set_params(integer=True)  # RANKS ARE WHOLE NUMBERS
    ax.grid(axis="x", visible=False)

    return fig


def coefficients_chart(weights):
    """Weight of each statistic in the model.

    Args:
        weights: Series returned by model.coefficients.

    Returns:
        Figure with horizontal bars, biggest effect at the top.
    """

    fig, ax = _new_chart("What the model weighs (standardised coefficients)", height=5)

    weights = weights.iloc[::-1]  # BIGGEST AT THE TOP
    colors = [BLUE if w > 0 else ORANGE for w in weights]  # RAISES / LOWERS THE SHARE
    ax.barh(weights.index, weights.values, color=colors)

    ax.axvline(0, color=INK, linewidth=0.8)  # ZERO LINE
    ax.set_xlabel("Effect of +1 standard deviation on the predicted share")
    ax.grid(axis="y", visible=False)

    return fig


def profile_chart(groups, stats=("PTS", "AST", "TRB", "W/L%")):
    """MVPs vs other candidates vs league, one small chart per statistic.

    Each statistic has its own scale (points vs percentages), so they are
    drawn side by side instead of on one shared axis.

    Args:
        groups: first value returned by profiles.profile.
        stats: statistics to show.

    Returns:
        Figure with one bar chart per statistic.
    """

    fig = Figure(figsize=(3 * len(stats), 3.5), layout="constrained")
    axes = fig.subplots(1, len(stats))  # ONE AXES PER STAT
    fig.suptitle("Average MVP vs other candidates vs league", x=0.01, ha="left")

    names = list(groups.columns)  # MVPs, Other candidates, League
    colors = [BLUE, ORANGE, INK]

    for ax, stat in zip(axes, stats):
        _style(ax, stat)
        values = groups.loc[stat]
        ax.bar(range(len(names)), values, color=colors)
        ax.set_xticks(range(len(names)), [n.replace(" ", "\n") for n in names])  # TWO-LINE LABELS
        ax.grid(axis="x", visible=False)

    return fig


def clusters_chart(clustered, x="PTS", y="TRB"):
    """Players coloured by cluster, MVPs marked with a star.

    Args:
        clustered: first value returned by profiles.find_clusters.
        x, y: statistics for the two axes.

    Returns:
        Scatter Figure.
    """

    fig, ax = _new_chart(f"Player archetypes ({x} vs {y}), MVPs as stars")

    for cluster, rows in clustered.groupby("Cluster"):
        color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
        ax.scatter(rows[x], rows[y], s=8, alpha=0.35, color=color, label=f"Cluster {cluster}")

    mvps = clustered[clustered["MVP"]]
    ax.scatter(mvps[x], mvps[y], s=120, marker="*", color="black", edgecolor="white", label="MVP")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    legend = ax.legend(frameon=False, markerscale=1.5)
    for handle in legend.legend_handles:
        handle.set_alpha(1)  # SOLID COLOURS IN THE LEGEND, NOT TRANSPARENT

    return fig
