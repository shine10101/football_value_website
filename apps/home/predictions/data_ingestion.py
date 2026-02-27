import logging
from datetime import date
from io import StringIO
from urllib.request import urlopen

import pandas as pd

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 30


def _current_season():
    """Auto-detect the current football season (Aug-Jul cycle).
    e.g. Feb 2026 → '2526', Oct 2025 → '2526'."""
    today = date.today()
    year = today.year
    if today.month >= 8:
        return f"{str(year)[-2:]}{str(year + 1)[-2:]}"
    return f"{str(year - 1)[-2:]}{str(year)[-2:]}"


LEAGUE_CODES = [
    'E0', 'E1', 'E2', 'E3', 'EC',
    'SC0', 'SC1', 'SC2', 'SC3',
    'D1', 'D2', 'I1', 'I2', 'F1', 'F2',
    'SP1', 'SP2', 'N1', 'B1', 'T1', 'P1',
]


def _fetch_csv(url):
    """Fetch a CSV from a URL and return a DataFrame."""
    resp = urlopen(url, timeout=_REQUEST_TIMEOUT)
    return pd.read_csv(
        StringIO(resp.read().decode('utf-8-sig')),
        parse_dates=['Date'],
        date_format='%d/%m/%Y',
    )


def get_links(season=None):
    """Fetch historical data for all leagues for the given season."""
    if season is None:
        season = _current_season()

    data = []
    data_dct = {}
    failed = []
    for league in LEAGUE_CODES:
        url = f'https://football-data.co.uk/mmz4281/{season}/{league}.csv'
        try:
            df = _fetch_csv(url)
            entry = (df,)
            data.append(entry)
            data_dct[league] = entry
        except Exception as e:
            failed.append(league)
            logger.warning("Failed to fetch league %s from %s: %s", league, url, e)
            continue

    if failed:
        logger.warning(
            "Failed to fetch %d/%d leagues: %s",
            len(failed), len(LEAGUE_CODES), ', '.join(failed),
        )
    logger.info("Successfully fetched %d/%d leagues", len(data_dct), len(LEAGUE_CODES))

    return data, data_dct


def get_fixtures():
    """Fetch upcoming fixtures with betting odds.

    Returns only fixtures dated today or later. Raises on failure so the
    caller can decide how to handle it.
    """
    url = 'https://www.football-data.co.uk/fixtures.csv'
    logger.info("Fetching fixtures from %s", url)
    try:
        data = _fetch_csv(url)
    except Exception as e:
        logger.error("Failed to fetch fixtures from %s: %s", url, e)
        raise

    total = len(data)

    # Filter to upcoming fixtures only
    today = pd.Timestamp(date.today())
    if 'Date' in data.columns:
        data = data[data['Date'] >= today].reset_index(drop=True)

    logger.info(
        "Fetched %d fixtures (%d upcoming from today onwards)",
        total, len(data),
    )
    return data
