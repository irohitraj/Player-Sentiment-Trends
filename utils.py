import config
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def roll(g: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Aggregate sentiment statistics over a grouping key.

    Computes:
    - Number of posts
    - Share of positive, neutral, and negative sentiments

    Args:
        df (pd.DataFrame): Input dataframe with sentiment columns.
        key (str): Column name used for grouping.

    Returns:
        pd.DataFrame: Aggregated dataframe with sentiment metrics
    """

    agg = (
        g.groupby(key, dropna=False)
            .agg(
            n_posts=("sentiment_score", "count"),
            #             mean_sentiment=("sentiment_score", "mean"),
            share_positive=("pred_label", lambda s: (s == "positive").mean()),
            share_neutral=("pred_label", lambda s: (s == "neutral").mean()),
            share_negative=("pred_label", lambda s: (s == "negative").mean()),
        )
            .reset_index()
    )

    return agg


def get_trend(df, freq):
    """
    Generate time-based sentiment trends.

    Supported frequencies:
    - "day"
    - "week"
    - "month"

    Args:
        df (pd.DataFrame): Input dataframe with 'date' column.
        freq (str): Frequency for aggregation.

    Returns:
        pd.DataFrame: Aggregated trend dataframe.
    """
    freq_map = {
        "day": df["date"].dt.date.astype(str),
        "week": df["date"].dt.to_period("W").apply(lambda r: r.start_time),
        "month": df["date"].dt.to_period("M").astype(str),
    }
    return roll(df, freq_map[freq])


def plot_trend_df(df: pd.DataFrame, title: str = "Trend"):
    time_col = df.columns[0]

    Path(config.output_plot_path).mkdir(parents=True, exist_ok=True)  # create if not exists

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    plt.figure(figsize=(12, 6))

    plt.plot(df[time_col], df["share_positive"], label="Positive", color='green', marker='o')
    plt.plot(df[time_col], df["share_neutral"], label="Neutral", linestyle=':', marker='o')
    plt.plot(df[time_col], df["share_negative"], label="Negative", color='red', marker='o')

    plt.title(title)
    plt.xlabel("Time Frame")
    plt.ylabel("Share of each sentiment")
    plt.legend()

    sentiment_file = Path(config.output_plot_path) / "sentiment_trends.png"
    plt.savefig(sentiment_file, dpi=300)

    plt.show()

    fig, ax = plt.subplots(figsize=(12, 4))

    bars = ax.bar(df[time_col], df["n_posts"], color="blue", alpha=0.6, width=8)

    # add labels
    ax.bar_label(bars, padding=3)
    ax.set_title(f"{title} - Tweet Volume")
    ax.set_xlabel("Time")
    ax.set_ylabel("#Tweets")
    volume_file = Path(config.output_plot_path) / "tweet_volume.png"
    plt.savefig(volume_file, dpi=300)
    plt.show()
