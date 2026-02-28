"""Custom MCP tools for pipeline validation and monitoring."""

import os
from claude_agent_sdk import tool
from typing import Any


@tool("validate_predictions_csv", "Validate predictions.csv for data quality issues", {})
async def validate_predictions_csv(args: dict[str, Any]) -> dict[str, Any]:
    """Check probabilities, odds, duplicates, dates in the predictions CSV."""
    import pandas as pd
    from django.conf import settings
    from datetime import date

    path = settings.PREDICTIONS_CSV
    issues = []

    if not os.path.exists(path):
        return {"content": [{"type": "text", "text": f"FAIL: File not found: {path}"}]}

    df = pd.read_csv(path)

    # Check required columns
    required = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'HWin', 'Draw', 'AWin',
                'Max_Value', 'Max_Value_Result']
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    # Check probability sums
    if all(c in df.columns for c in ['HWin', 'Draw', 'AWin']):
        prob_sum = df['HWin'] + df['Draw'] + df['AWin']
        bad_rows = df[abs(prob_sum - 1.0) > 0.05]
        if len(bad_rows) > 0:
            issues.append(
                f"{len(bad_rows)} rows have probability sum outside [0.95, 1.05]: "
                f"range [{prob_sum.min():.3f}, {prob_sum.max():.3f}]"
            )

    # Check odds validity
    for col in ['Odds_H', 'Odds_D', 'Odds_A']:
        if col in df.columns:
            bad = df[df[col].notna() & (df[col] <= 1.0)]
            if len(bad) > 0:
                issues.append(f"{len(bad)} rows have {col} <= 1.0")

    # Check duplicates
    key_cols = ['Div', 'Date', 'HomeTeam', 'AwayTeam']
    if all(c in df.columns for c in key_cols):
        dupes = df.duplicated(subset=key_cols, keep=False)
        if dupes.any():
            issues.append(f"{dupes.sum()} duplicate match entries found")

    # Check dates
    if 'Date' in df.columns:
        try:
            dates = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.date
            past = dates[dates < date.today()]
            if len(past) > 0:
                issues.append(f"{len(past)} predictions have dates in the past")
            null_dates = dates.isna().sum()
            if null_dates > 0:
                issues.append(f"{null_dates} rows have unparseable dates")
        except Exception as e:
            issues.append(f"Date parsing error: {e}")

    # NaN check on critical columns
    for col in ['HWin', 'Draw', 'AWin', 'Max_Value']:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                issues.append(f"{nan_count} NaN values in {col}")

    status = "PASS" if not issues else "FAIL"
    lines = [
        f"CSV Validation: {status}",
        f"  File: {path}",
        f"  Rows: {len(df)}",
        f"  Columns: {list(df.columns)}",
    ]
    if issues:
        lines.append("  Issues:")
        for issue in issues:
            lines.append(f"    - {issue}")
    else:
        lines.append("  No issues found.")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


@tool("check_data_source_availability", "Check if football-data.co.uk is reachable", {})
async def check_data_source_availability(args: dict[str, Any]) -> dict[str, Any]:
    """HEAD request to football-data.co.uk endpoints to check availability."""
    from urllib.request import urlopen, Request
    from urllib.error import URLError

    urls = {
        'fixtures': 'https://www.football-data.co.uk/fixtures.csv',
        'historical_E0': 'https://www.football-data.co.uk/mmz4281/2526/E0.csv',
    }

    lines = ["Data Source Availability:"]
    for name, url in urls.items():
        try:
            req = Request(url, method='HEAD')
            resp = urlopen(req, timeout=10)
            lines.append(f"  {name}: OK (status {resp.status})")
        except URLError as e:
            lines.append(f"  {name}: UNAVAILABLE ({e.reason})")
        except Exception as e:
            lines.append(f"  {name}: ERROR ({e})")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}
