import pandas as pd
from datetime import date


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


def get_links(season=None):
    """Fetch historical data for all leagues for the given season."""
    if season is None:
        season = _current_season()

    data = []
    data_dct = {}
    for league in LEAGUE_CODES:
        url = f'https://football-data.co.uk/mmz4281/{season}/{league}.csv'
        try:
            df = pd.read_csv(url, parse_dates=['Date'], date_format='%d/%m/%Y')
            entry = (df,)
            data.append(entry)
            data_dct[league] = entry
        except Exception:
            continue

    return data, data_dct


def get_fixtures():
    """Fetch upcoming fixtures with betting odds."""
    url = 'https://www.football-data.co.uk/fixtures.csv'
    data = pd.read_csv(url, parse_dates=['Date'], date_format='%d/%m/%Y')
    return data
