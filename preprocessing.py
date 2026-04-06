import re
import pandas as pd


def clean_tweet(tweet: str, max_chars: int) -> str:
    """
    Clean raw tweet text by removing noise and truncating length.

    Steps:
    - Remove URLs
    - Remove mentions (@user)
    - Remove hashtag symbols (#)
    - Normalize whitespace
    - Truncate to max_chars

    Args:
        tweet (str): Raw tweet text.
        max_chars (int): Maximum allowed length of cleaned text.

    Returns:
        str: Cleaned and normalized tweet text.
             Returns empty string if input is None or invalid.

    """

    if tweet is None:
        return ""

    s = str(tweet).strip()

    # s? can accomodate http and https, \S is for whitespace, re.I ignore case sensitivity and can match case and it doesnot overwrite original string
    URL_RE = re.compile(r"https?://\S+|pic\.twitter\.com/\S+", re.I)

    if not s:
        return ""

    s = URL_RE.sub(" ", s)  # url replaced with space

    s = re.sub(r"\s+", " ", s).strip()

    s = re.sub(r"@\w+", "", s)

    s = re.sub(r"#", "", s)

    s = re.sub(r"[^\w\s.,!?;:'\"()-]", "", s)

    if len(s) > max_chars:
        s = s[:max_chars]

    return s


def parse_datetimes(df: pd.DataFrame):
    """
     Parse datetime information from multiple columns with fallback logic.

    Priority:
    1. 'indexed' column with full timestamp format
    2. 'indexed' column with date-only format
    3. 'published' column as fallback

    Args:
        df (pd.DataFrame): Input dataframe containing 'indexed' and 'published'.

    Returns:
        pd.Series: Parsed datetime values

    """

    dt = pd.to_datetime(df['indexed'], format="%m/%d/%y %H:%M:%S", errors="coerce")
    blanks_in_indexed = dt.isna()

    if blanks_in_indexed.any():
        dt2 = pd.to_datetime(df['indexed'], format="%m/%d/%y", errors="coerce")
        dt.loc[blanks_in_indexed] = dt2.loc[blanks_in_indexed]

    still_empty = dt.isna()
    if still_empty.any():
        dt.loc[still_empty] = pd.to_datetime(df.loc[still_empty, "published"], errors="coerce", dayfirst=False)
    return dt