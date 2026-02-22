import logging

import pandas as pd
from django.conf import settings

from . import data_ingestion, poisson_analysis, calculate_value

logger = logging.getLogger(__name__)


def _fixture_predictions(fixtures, data_dct):
    """Generate predictions for upcoming fixtures using historical league data."""
    leagues = fixtures['Div'].unique()
    predictions = []
    for league in leagues:
        try:
            league_fixtures = fixtures[fixtures['Div'] == league]
            league_data = data_dct[league][0]
            predictions.append(poisson_analysis.analysis(league_data, league_fixtures))
        except Exception:
            logger.warning("Skipping league %s (no historical data)", league)
            continue

    if not predictions:
        return pd.DataFrame()

    return pd.concat(predictions)


def _safe_float(val, default=0.0):
    """Convert a value to float, returning default for NaN/None/missing."""
    try:
        result = float(val)
        if pd.isna(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def archive_predictions(pred_df):
    """Save predictions to the database for historical tracking."""
    from apps.home.models import Prediction

    archived = 0
    for _, row in pred_df.iterrows():
        try:
            date_val = pd.to_datetime(row['Date']).date()
        except Exception:
            continue

        pred_ftr = str(row.get('Pred_FTR', ''))
        if pd.isna(row.get('Pred_FTR')):
            pred_ftr = ''
        max_value_result = str(row.get('Max_Value_Result', ''))
        if pd.isna(row.get('Max_Value_Result')):
            max_value_result = ''

        defaults = {
            'h_win': _safe_float(row.get('HWin')),
            'draw': _safe_float(row.get('Draw')),
            'a_win': _safe_float(row.get('AWin')),
            'pred_ftr': pred_ftr,
            'pred_fthg': _safe_float(row.get('Pred_FTHG')),
            'pred_ftag': _safe_float(row.get('Pred_FTAG')),
            'btts_yes': _safe_float(row.get('BTTS_Yes')),
            'btts_no': _safe_float(row.get('BTTS_No')),
            'over25': _safe_float(row.get('Over25')),
            'under25': _safe_float(row.get('Under25')),
            'over25_value': _safe_float(row.get('Over25_Value')) or None,
            'under25_value': _safe_float(row.get('Under25_Value')) or None,
            'max_value': _safe_float(row.get('Max_Value')),
            'max_value_result': max_value_result,
            'odds_h': _safe_float(row.get('AvgH', row.get('B365H'))) or None,
            'odds_d': _safe_float(row.get('AvgD', row.get('B365D'))) or None,
            'odds_a': _safe_float(row.get('AvgA', row.get('B365A'))) or None,
            'best_odds': _safe_float(row.get('Best_Odds')) or None,
            'odds_over25': _safe_float(row.get('Odds_Over25')) or None,
            'odds_under25': _safe_float(row.get('Odds_Under25')) or None,
        }

        _, created = Prediction.objects.update_or_create(
            div=str(row.get('Div', '')),
            date=date_val,
            home_team=str(row.get('HomeTeam', '')),
            away_team=str(row.get('AwayTeam', '')),
            defaults=defaults,
        )
        if created:
            archived += 1

    logger.info("Archived %d new predictions (%d total in batch)", archived, len(pred_df))


def resolve_results(data_dct):
    """Match unresolved predictions against actual results from historical data."""
    from apps.home.models import Prediction

    unresolved = Prediction.objects.filter(resolved=False)
    if not unresolved.exists():
        logger.info("No unresolved predictions to resolve")
        return 0

    # Build a lookup from historical data: (div, date, home, away) -> (ftr, fthg, ftag)
    results_lookup = {}
    for league_code, (league_df,) in data_dct.items():
        required_cols = {'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG'}
        if not required_cols.issubset(league_df.columns):
            continue
        for _, row in league_df.iterrows():
            try:
                match_date = pd.to_datetime(row['Date']).date()
            except Exception:
                continue
            if pd.isna(row['FTR']):
                continue
            key = (league_code, match_date, str(row['HomeTeam']), str(row['AwayTeam']))
            results_lookup[key] = (str(row['FTR']), int(row['FTHG']), int(row['FTAG']))

    resolved_count = 0
    for pred in unresolved:
        key = (pred.div, pred.date, pred.home_team, pred.away_team)
        if key in results_lookup:
            ftr, fthg, ftag = results_lookup[key]
            pred.actual_ftr = ftr
            pred.actual_fthg = fthg
            pred.actual_ftag = ftag
            pred.resolved = True
            pred.save(update_fields=['actual_ftr', 'actual_fthg', 'actual_ftag', 'resolved'])
            resolved_count += 1

    logger.info("Resolved %d predictions with actual results", resolved_count)
    return resolved_count


def run_predictions():
    """Full pipeline: fetch data, generate predictions, save to CSV.
    Returns the predictions DataFrame."""
    logger.info("Starting prediction pipeline...")

    logger.info("Fetching historical data...")
    _data, data_dct = data_ingestion.get_links()

    # Resolve past predictions with actual results
    logger.info("Resolving past predictions...")
    resolve_results(data_dct)

    logger.info("Fetching upcoming fixtures...")
    fixtures = data_ingestion.get_fixtures()

    logger.info("Generating predictions...")
    pred = _fixture_predictions(fixtures, data_dct)

    if pred.empty:
        logger.warning("No predictions generated")
        return pred

    logger.info("Calculating value...")
    pred = calculate_value.value(pred)

    output_path = settings.PREDICTIONS_CSV
    pred.to_csv(output_path, index=False)
    logger.info("Saved %d predictions to %s", len(pred), output_path)

    # Archive predictions to database
    logger.info("Archiving predictions...")
    archive_predictions(pred)

    return pred
